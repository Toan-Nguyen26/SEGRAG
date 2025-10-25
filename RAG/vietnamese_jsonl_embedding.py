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
import logging
import random

# Generate a random number for the log file name
random_number = random.randint(1000, 9999)

# Set up logging with the random number in the file name
logging.basicConfig(filename=f'file.txt', level=logging.INFO)
model = SentenceTransformer("BAAI/bge-m3", cache_folder='/path/to/local/cache')
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir='/path/to/local/cache')

def configure_logging(dataset_name, chunk_type):
    # Create a directory for logs if it doesn't exist
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)

    # Create a dynamic log file name based on the dataset name and chunk type
    log_filename = os.path.join(log_directory, f'{dataset_name}_{chunk_type}_embedding.log')

    # Configure logging settings
    logging.basicConfig(
        filename=log_filename,
        filemode='a',  # Append to existing log file
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Log the dataset name and chunk type to distinguish the log session
    logging.info(f"Starting logging for dataset: {dataset_name}, chunk type: {chunk_type}")

def chunk_text_by_tokens(text, chunk_size, tokenizer, max_words_per_chunk=2000):
    # First, split the text into smaller word chunks to avoid tokenizing large texts at once
    words = text.split()  # Split the text into words
    chunks = []
    total_chunks = 0  # Track the total number of chunks
    
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
            decoded_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append((decoded_text, len(chunk_tokens)))  # Return both text and token size as a tuple
            total_chunks += 1
            print(f"Chunk {total_chunks}: {len(chunk_tokens)} tokens.")
            logging.info(f"Chunk {total_chunks}: {len(chunk_tokens)} tokens.")
    
    print(f"Number of token chunks: {len(chunks)}")
    return chunks

def determine_chunk_size():
    if args.chunk_type == '1024':
        model.max_seq_length = 1024
        return 1024
    elif args.chunk_type == '512':
        model.max_seq_length = 512
        return 512
    elif args.chunk_type == '256':
        model.max_seq_length = 256
        return 256
    elif args.chunk_type == '2048':
        model.max_seq_length = 2048
        return 2048
    else:
        model.max_seq_length = 4092
        return 4092
    
def create_segmentation_faiss_index_from_jsonl(jsonl_file_path, output_faiss_path, output_ids_path):
    # Prepare lists to store embeddings and document info
    embeddings = []
    document_chunks = []
    chunk_size = determine_chunk_size()

    total_chunk_size = 0  # Variable to track total size of all chunks
    total_chunks_count = 0  # Variable to track total number of chunks

    # Read the JSONL file
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            doc = json.loads(line.strip())
            sentences = doc.get('sentences', [])
            doc_id = doc.get('file', str(uuid.uuid4()))  # Use file as ID or generate UUID
            title = doc.get('title', 'Untitled')
            
            # Skip Q&A entries (they have empty sentences)
            if not sentences:
                print(f"Skipping document {doc_id} (Q&A entry with no sentences)")
                continue
            
            content = " ".join(sentences)  # Reconstruct content from sentences

            # Split the text into smaller chunks based on chunking strategy
            # Only token-based chunking is supported
            if args.chunk_type in ['256', '512', '1024', '2048']:
                text_chunks = chunk_text_by_tokens(content, chunk_size, tokenizer)
            else:
                print(f"⚠️ Unsupported chunk_type: {args.chunk_type}. Defaulting to 512 tokens.")
                logging.warning(f"Unsupported chunk_type: {args.chunk_type}. Defaulting to 512 tokens.")
                text_chunks = chunk_text_by_tokens(content, 512, tokenizer)

            print(f"Processing document {doc_id} with {len(text_chunks)} chunks")
            logging.info(f"Processing document {doc_id} ({title}) with {len(text_chunks)} chunks")
            
            # Iterate through each chunk and its size
            chunk_id = 1
            for chunk_text, c_size in text_chunks:
                # Encode the chunk
                total_chunk_size += c_size
                total_chunks_count += 1
                embedding = model.encode(chunk_text)

                # Store the embedding and related information
                document_chunks.append({
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'title': title,
                    'chunk': chunk_text,
                    'chunk_size': c_size,
                    'embedding': embedding.tolist()  # Convert to list for JSON serialization
                })
                chunk_id += 1
                embeddings.append(embedding)

    # Save the document chunks with IDs and embeddings
    os.makedirs(os.path.dirname(output_ids_path), exist_ok=True)
    with open(output_ids_path, 'w', encoding='utf-8') as id_file:
        json.dump(document_chunks, id_file, ensure_ascii=False, indent=4)

    # Convert embeddings to a numpy array
    embeddings = np.array(embeddings)

    # Create a FAISS index
    embedding_dim = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    index.add(embeddings)  # Add the embeddings to the index

    # Save the FAISS index
    os.makedirs(os.path.dirname(output_faiss_path), exist_ok=True)
    faiss.write_index(index, output_faiss_path)

    logging.info(f"Total number of chunks for chunk type {args.chunk_type} the dataset {args.dataset} is: {len(embeddings)}")
    logging.info(f"Average chunk size is: {total_chunk_size/total_chunks_count if total_chunks_count > 0 else 0}")
    print(f"Total number of chunks for chunk type {args.chunk_type} the dataset {args.dataset} is: {len(embeddings)}")
    print(f"Average chunk size is: {total_chunk_size/total_chunks_count if total_chunks_count > 0 else 0}")
    print(f"FAISS index and document chunk information have been saved to {output_faiss_path} and {output_ids_path}")

def main(args):
    if args.dataset:
        # Construct input file path
        if args.original_data.endswith('.jsonl'):
            jsonl_file_path = args.original_data
        else:
            jsonl_file_path = f'{args.original_data}.jsonl'
        
        # Check if file exists
        if not os.path.exists(jsonl_file_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")
        
        configure_logging(args.dataset, args.chunk_type)
        
        # Create output paths
        output_dir = os.path.join('data', args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        
        create_segmentation_faiss_index_from_jsonl(
            jsonl_file_path=jsonl_file_path,
            output_faiss_path=os.path.join(output_dir, f'{args.dataset}_{args.chunk_type}.index'),
            output_ids_path=os.path.join(output_dir, f'{args.dataset}_{args.chunk_type}.json')
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please provide a dataset name.")
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name (e.g., squad, narrativeqa, vietnamese_law)', required=True, type=str)
    parser.add_argument('--chunk_type', help='Chunking strategy: 256, 512, 1024, 2048', type=str, default='512')
    parser.add_argument('--original_data', help='Path to input JSONL file', type=str, required=True)
    args = parser.parse_args()
    main(args)