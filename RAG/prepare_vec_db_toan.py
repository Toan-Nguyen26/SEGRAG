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

def split_text_into_segmented_chunks(text, max_chunk_size=512, tokenizer=None, segmented_sentences=None):
    """
    Splits the input text into smaller chunks such that each chunk does not exceed the max_chunk_size when tokenized.
    The chunking process respects the segment boundaries defined by the segmented_sentences list.
    """
    words = text.split()  # Split by words to maintain word boundaries
    chunks = []
    current_chunk = []
    current_length = 0
    
    sentence_index = 0  # Index to track the current sentence
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for the space
        
        if current_length >= max_chunk_size or segmented_sentences[sentence_index] == 1:
            chunk_text = ' '.join(current_chunk).strip()
            # Ensure the chunk is within the max token length when tokenized
            tokens = tokenizer(chunk_text, return_tensors='pt', truncation=True, max_length=max_chunk_size, padding='max_length')['input_ids'][0]
            
            if len(tokens) <= max_chunk_size:
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
        
        # Move to the next sentence after a space or punctuation
        if word.endswith(('.', '!', '?')):
            sentence_index += 1

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return chunks


def split_text_into_chunks(text, max_chunk_size=256, tokenizer=None):
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
        print("Hello")
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

def split_json_content(file_path, output_path, chunk_size=256):
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

    # Split the content
    split_documents = []
    for document in data:
        if 'content' in document:
            chunks = text_splitter.split_text(document['content'])
            chunk_uuid = str(uuid.uuid4()) 
            for i, chunk in enumerate(chunks):
                split_documents.append({
                    "id": document["id"],
                    "doc_id": chunk_uuid,
                    "content": chunk
                })

    # Save the split content to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(split_documents, output_file, ensure_ascii=False, indent=4)

# Call the function
# split_json_content('data/concatenated_documents.json', 'data/split_documents.json')

# For indexing the split documents of squad
create_faiss_index_from_json(son_file_path='data/concatenated_documentsjson', output_faiss_path='data/faiss_index_256.index', output_ids_path='data/document_ids_256.json')

def main(args):
    if args.dataset == 'squad':
        create_concatenated_documents_squad_json()
    elif args.dataset == 'narrative_qa':
        create_concatenated_documents_narrativeqa_json()
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please choose 'squad' or 'narrative_qa'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whenever it is squad or narrative_qa',  required=True, type=str, default="squad")
    # By default the command to run is:
    # python document_concat.py --dataset squad

    main(parser.parse_args())