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
import logging

model = SentenceTransformer("BAAI/bge-m3", cache_folder='/path/to/local/cache')
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir='/path/to/local/cache')


# def chunk_text_by_tokens(text, chunk_size, tokenizer):
#     """
#     This function chunks the text into chunks of `chunk_size` tokens.
#     """
#     tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
#     chunks = []
#     for i in range(0, len(tokens), chunk_size):
#         chunk_tokens = tokens[i:i+chunk_size]
#         chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
#         chunks.append(chunk_text)
#     print(len(chunks))
#     return chunks

def chunk_text_by_tokens(text, chunk_size, tokenizer, max_words_per_chunk=4000):
    """
    This function chunks the text into smaller chunks of tokens after splitting the text into smaller
    word-based chunks to avoid tokenizing the entire text at once.
    """
    # First, split the text into smaller word chunks to avoid tokenizing large texts at once
    words = text.split()  # Split the text into words
    chunks = []
    
    # Iterate through the words and create smaller chunks of words
    for i in range(0, len(words), max_words_per_chunk):
        chunk_words = words[i:i + max_words_per_chunk]
        chunk_text = " ".join(chunk_words)  # Join the words back into a chunk of text
        
        # Now tokenize the chunk of text
        tokens = tokenizer(chunk_text, return_tensors='pt', truncation=False)['input_ids'][0]
        
        # Check if the token length exceeds the limit (8192 tokens)
        if len(tokens) > 8192:
            logging.info(f"Token length exceeded: {len(tokens)} tokens (Limit: 8192) for chunk starting with: '{chunk_text[:100]}'")

        # Further split tokens into model's max token length (chunk_size)
        for j in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[j:j + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
    
    print(f"Number of token chunks: {len(chunks)}")
    return chunks


def chunk_text_by_segment(text, seg_array, tokenizer, title=None, doc_id=None):
    # Load spaCy model for sentence segmentation
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]  # Extract sentences from spaCy
    max_chunk_size = 8192
    chunks = []
    current_chunk = []
    total_chunks = 0  # Track the total number of chunks

    logging.info(f"Processing {title} with id {doc_id}.")
    
    for i, is_segment_end in enumerate(seg_array):
        if i >= len(sentences):
            break

        # Tokenize the current sentence
        sentence_tokens = tokenizer(sentences[i], return_tensors='pt', truncation=False)['input_ids'][0].tolist()
        
        # Check if adding the new sentence tokens would exceed max_chunk_size
        if len(current_chunk) + len(sentence_tokens) > max_chunk_size:
            # Store the current chunk
            chunk_size = len(current_chunk)
            chunks.append((current_chunk, chunk_size))
            
            logging.info(f"Chunk {total_chunks + 1}: Size = {chunk_size} tokens.")
            if chunk_size > max_chunk_size:
                logging.info(f"Chunk size exceeds the limit: {chunk_size} tokens (Limit: {max_chunk_size}).")
            total_chunks += 1
            current_chunk = []  # Reset the chunk

        # Add sentence tokens to the current chunk
        current_chunk.extend(sentence_tokens)

        # If it's the end of a segment, store the current chunk
        if is_segment_end == 1:
            chunk_size = len(current_chunk)
            chunks.append((current_chunk, chunk_size))
            logging.info(f"Chunk {total_chunks + 1}: Size = {chunk_size} tokens.")
            total_chunks += 1
            current_chunk = []  # Reset the chunk

    # Add any remaining tokens after the loop
    if current_chunk:
        chunk_size = len(current_chunk)
        chunks.append((current_chunk, chunk_size))
        logging.info(f"Chunk {total_chunks + 1}: Size = {chunk_size} tokens.")
        total_chunks += 1

    # Convert tokens back to text and return both the text and chunk size
    chunk_texts_and_sizes = [(tokenizer.decode(torch.tensor(chunk), skip_special_tokens=True), size) for chunk, size in chunks]

    logging.info(f"Total number of chunks: {total_chunks}")
    logging.info("--------------------------------------------------")

    return chunk_texts_and_sizes
def determine_chunk_size():
    if args.chunk_type == '256':
        model.max_seq_length = 256
        return 256
    elif args.chunk_type == '512':
        model.max_seq_length = 512
        return 512
    else:
        model.max_seq_length = 8192
        return 8192
    

def create_segmendtaion_faiss_index_from_directory(json_directory_path, 
                                                   output_faiss_path, 
                                                   output_ids_path):
    # Prepare lists to store embeddings and document info
    embeddings = []
    document_chunks = []
    chunk_size = determine_chunk_size()

    # Iterate over each JSON file in the directory
    for json_filename in os.listdir(json_directory_path):
        if json_filename.endswith(".json"):
            json_file_path = os.path.join(json_directory_path, json_filename)

            # Load the JSON data from the file
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                doc = json.load(json_file)

            content = doc['content']
            doc_id = doc['id']
            title = doc['title']
            segmented_sentences = doc['segmented_sentences']
            num_sentences = doc['num_sentences']
            # Split the text into smaller chunks that can be tokenized within model limits
            if args.chunk_type == '256' or args.chunk_type == '512':
                text_chunks = chunk_text_by_tokens(content, chunk_size, tokenizer)
            else:
                text_chunks = chunk_text_by_segment(content, segmented_sentences, tokenizer, title, doc_id)
            # text_chunks = chunk_text_by_tokens(content, chunk_size, tokenizer)
            # text_chunks = split_text_into_segmented_chunks_at_word_level(num_sentences, content, max_chunk_size=chunk_size, tokenizer=tokenizer, segmented_sentences=segmented_sentences)
            print(f"Processing document {doc_id} with {len(text_chunks)} chunks")
            # Iterate through each chunk and its size
            chunk_id = 1
            for chunk_text, c_size in text_chunks:
                # Encode the chunk
                embedding = model.encode(chunk_text)

                # Store the embedding and related information
                document_chunks.append({
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'title': title,
                    'chunk': chunk_text,
                    'chunk_size': c_size,  # Include the size of the chunk
                    'embedding': embedding.tolist()  # Convert to list for JSON serialization
                })
                chunk_id += 1
                embeddings.append(embedding)

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

def main(args):
    if args.dataset:
        json_directory_path = f'data/{args.dataset}/individual_documents'
        # embedding_testing()
        logging.basicConfig(filename=f'{args.dataset}_embedding.txt', level=logging.INFO)
        create_segmendtaion_faiss_index_from_directory(json_directory_path=json_directory_path, 
                                     output_faiss_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.index', 
                                     output_ids_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json')
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please choose 'squad' or 'narrativeqa' or 'quality'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whenever it is squad or narrative_qa',  required=True, type=str, default="squad")
    parser.add_argument('--chunk_type', help='What is the chunking strategy: 256, 512, seg, segclus', type=str, default='256')
    args = parser.parse_args() 
    main(args)

# Example running command
# python embedding.py --dataset squad --chunk_type 256