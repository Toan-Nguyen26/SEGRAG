import os
import json
import numpy as np
import faiss
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import uuid
from argparse import ArgumentParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.docstore.document import Document
import spacy

model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")


def split_text_into_segmented_chunks_at_sentences_level(num_sentences, text, max_chunk_size=512, tokenizer=None, segmented_sentences=None):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    sentence_index = 0  # Index to track the current sentence
    l = len(segmented_sentences)  # Total number of sentences

    if segmented_sentences is None or num_sentences != len(list(doc.sents)):
        raise ValueError("segmented_sentences must have the same length as the number of sentences in the text.")

    for sentence in doc.sents:
        sentence_text = sentence.text
        sentence_tokens = sentence_text.split()

        for word in sentence_tokens:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for the space

        # Check segment end condition
        if current_length >= max_chunk_size or (sentence_index < num_sentences and segmented_sentences[sentence_index] == 1):
            chunk_text = ' '.join(current_chunk).strip()
            tokens = tokenizer(chunk_text, return_tensors='pt', truncation=True, max_length=max_chunk_size, padding='max_length')['input_ids'][0]

            if len(tokens) <= max_chunk_size:
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
        sentence_index += 1

        # Safeguard: Stop processing if sentence_index reaches the last sentence
        if sentence_index >= num_sentences:
            break

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())

    return chunks

def split_text_into_segmented_chunks_at_word_level(num_sentences, text, max_chunk_size=512, tokenizer=None, segmented_sentences=None):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    sentences = list(doc.sents)  
    chunks = []
    current_chunk = []
    sentence_index = 0
    
    for sentence in sentences:
        words = sentence.text.split() 
        
        for word in words:
            potential_chunk = ' '.join(current_chunk + [word]).strip()
            potential_tokens = tokenizer(potential_chunk, return_tensors='pt', truncation=True, max_length=max_chunk_size, padding=False)['input_ids'][0]
            
            if len(potential_tokens) >= max_chunk_size:
                # Finalize the current chunk and start a new one
                chunks.append(' '.join(current_chunk).strip())
                current_chunk = [word]  # Start the new chunk with the current word
            else:
                current_chunk.append(word)

            # If the end of a segment is reached, finalize the chunk
            if segmented_sentences[sentence_index] == 1 and word == words[-1]:
                chunks.append(' '.join(current_chunk).strip())
                current_chunk = []
        
        sentence_index += 1

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return chunks

def split_text_into_chunks(text, max_chunk_size, tokenizer=None):
    """
    Splits the input text into smaller chunks such that each chunk does not exceed the max_chunk_size when tokenized.
    """
    # Split the text into sentences (or just rough character-based chunks)
    words = text.split()  # Split by words to maintain word boundaries
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for the space
        
        if current_length >= max_chunk_size:
            chunk_text = ' '.join(current_chunk).strip()
            # Ensure the chunk is within the max token length when tokenized
            tokens = tokenizer(chunk_text, return_tensors='pt', truncation=True, max_length=max_chunk_size, padding='max_length')['input_ids'][0]
            
            if len(tokens) <= max_chunk_size:
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
    
    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return chunks

def create_faiss_index_from_json(json_file_path='data/concatenated_documents_new.json', output_faiss_path='data/faiss_index_256.index', output_ids_path='data/document_ids_256.json'):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documents = json.load(json_file)

    # Prepare lists to store embeddings and document info
    embeddings = []
    document_chunks = []

    # Load the model and tokenizer
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    # model = AutoModelForMaskedLM.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")

    for doc in documents:
        content = doc['content']
        doc_id = doc['id']

        # Split the text into smaller chunks that can be tokenized within model limits
        text_chunks = split_text_into_chunks(content, max_chunk_size=256, tokenizer=tokenizer)

        for chunk_text in text_chunks:
        # Encode the chunk
            embedding = model.encode(chunk_text)
            chunk_uuid = str(uuid.uuid4())  # Generate a unique UUID for each chunk

            # Store the embedding and related information
            embeddings.append(embedding)
            document_chunks.append({
                'id': chunk_uuid,
                'doc_id': doc_id,
                'chunk': chunk_text,
                'embedding': embedding.tolist()  # Convert to list for JSON serialization
            })

    # Convert embeddings to a numpy array
    embeddings = np.array(embeddings)

    # Create a FAISS index
    embedding_dim = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    index.add(embeddings)  # Add the embeddings to the index

    # Save the FAISS index
    os.makedirs(os.path.dirname(output_faiss_path), exist_ok=True)
    faiss.write_index(index, output_faiss_path)

    # Save the document chunks with IDs and embeddings
    with open(output_ids_path, 'w', encoding='utf-8') as id_file:
        json.dump(document_chunks, id_file, ensure_ascii=False, indent=4)

    print(f"FAISS index and document chunk information have been saved to {output_faiss_path} and {output_ids_path}")

def create_segmendtaion_faiss_index_from_directory(json_directory_path, 
                                                   output_faiss_path, 
                                                   output_ids_path,
                                                   chunk_size=256):
    # Prepare lists to store embeddings and document info
    embeddings = []
    document_chunks = []

    # Load the model and tokenizer
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")

    # Iterate over each JSON file in the directory
    for json_filename in os.listdir(json_directory_path):
        if json_filename.endswith(".json"):
            json_file_path = os.path.join(json_directory_path, json_filename)

            # Load the JSON data from the file
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                doc = json.load(json_file)

            content = doc['content']
            doc_id = doc['id']
            segmented_sentences = doc['segmented_sentences']
            num_sentences = doc['num_sentences']
            # Split the text into smaller chunks that can be tokenized within model limits
            text_chunks = split_text_into_segmented_chunks_at_word_level(num_sentences, content, max_chunk_size=chunk_size, tokenizer=tokenizer, segmented_sentences=segmented_sentences)

            for chunk_text in text_chunks:
                # Encode the chunk
                embedding = model.encode(chunk_text)
                chunk_uuid = str(uuid.uuid4())  # Generate a unique UUID for each chunk

                # Store the embedding and related information
                embeddings.append(embedding)
                document_chunks.append({
                    'id': chunk_uuid,
                    'doc_id': doc_id,
                    'chunk': chunk_text,
                    'embedding': embedding.tolist()  # Convert to list for JSON serialization
                })

    # Convert embeddings to a numpy array
    embeddings = np.array(embeddings)

    # Create a FAISS index
    embedding_dim = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    index.add(embeddings)  # Add the embeddings to the index

    # Save the FAISS index
    os.makedirs(os.path.dirname(output_faiss_path), exist_ok=True)
    faiss.write_index(index, output_faiss_path)

    # Save the document chunks with IDs and embeddings
    os.makedirs(os.path.dirname(output_ids_path), exist_ok=True)
    with open(output_ids_path, 'w', encoding='utf-8') as id_file:
        json.dump(document_chunks, id_file, ensure_ascii=False, indent=4)

    print(f"FAISS index and document chunk information have been saved to {output_faiss_path} and {output_ids_path}")

def embedding_testing():
    # Load the model and tokenizer
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    output_faiss_path = '/faiss_index_256.index'
    output_ids_path = "document_chunks.json"
    documents = [
        {
            "id": "doc1",
             "content": [
                # Group 1: Artificial Intelligence and Machine Learning
                "Artificial intelligence is revolutionizing various industries.",
                "Machine learning is a subset of artificial intelligence focused on training algorithms.",
                "Deep learning is a technique within machine learning that uses neural networks with many layers.",
                
                # Group 2: Natural Language Processing and AI Applications
                "Natural language processing allows machines to understand and generate human language.",
                "Computer vision enables machines to interpret and analyze visual information.",
                "AI ethics is an important consideration for ensuring responsible technology development.",
                
                # Group 3: Robotics and Autonomous Systems
                "Robotics is an interdisciplinary field integrating AI with hardware to create intelligent machines.",
                "Autonomous vehicles use machine learning to navigate without human intervention.",
                "Reinforcement learning trains agents to make decisions through trial and error.",
                
                # # Group 4: Generative Models and AI Techniques
                # "Generative models like GANs can create realistic images from noise.",
                # "Neural networks are at the core of many AI advancements.",
                # "Data science involves using various techniques to extract insights from data."
            ]
        }
    ]


    # Prepare lists to store embeddings and document info
    embeddings = []
    document_chunks = []

    for doc in documents:
        content = doc['content']
        doc_id = doc['id']
        # print(content)
        # sentences = content.split('. ')
        for sentence in content:
            # Encode the sentence
            embedding = model.encode(sentence)
            chunk_uuid = str(uuid.uuid4())  # Generate a unique UUID for each sentence

            # Store the embedding and related information
            embeddings.append(embedding)
            document_chunks.append({
                'id': chunk_uuid,
                'doc_id': doc_id,
                'chunk': sentence,
                'embedding': embedding.tolist()  # Convert to list for JSON serialization
            })

    # Convert embeddings to a numpy array
    embeddings = np.array(embeddings)

    # Create a FAISS index
    embedding_dim = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    index.add(embeddings)  # Add the embeddings to the index

   # Save the FAISS index
    os.makedirs(os.path.dirname(output_faiss_path), exist_ok=True)
    faiss.write_index(index, output_faiss_path)

    # Save the document chunks with IDs and embeddings
    with open(output_ids_path, 'w', encoding='utf-8') as id_file:
        json.dump(document_chunks, id_file, ensure_ascii=False, indent=4)

    print(f"FAISS index and document chunk information have been saved to {output_faiss_path} and {output_ids_path}")

def main(args):
    if args.dataset:
        # json_directory_path = f'data/{args.dataset}/individual_documents'
        json_directory_path = f'data/{args.dataset}/test_documents'
        embedding_testing()
        # create_segmendtaion_faiss_index_from_directory(json_directory_path=json_directory_path, 
        #                              output_faiss_path=f'data/{args.dataset}/faiss_index/{args.chunk_type}_{args.chunk_size}.index', 
        #                              output_ids_path=f'data/{args.dataset}/embeddings/{args.chunk_type}_{args.chunk_size}.json',
        #                              chunk_size=args.chunk_size)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please choose 'squad' or 'narrative_qa'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whenever it is squad or narrative_qa',  required=True, type=str, default="squad")
    parser.add_argument('--chunk_size', help='Size of the chunks to split the text into', type=int, default=256)
    parser.add_argument('--chunk_type', help='What is the chunking strategy: 256, 512, semantic, segmentation', type=str, default='256')
    # By default the command to run is:
    # python prepare_vec_db_toan.py --dataset squad --chunk_size 256 --chunk_type segmentation

    main(parser.parse_args())