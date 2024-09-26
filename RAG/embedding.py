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
from cluster.cluster_helper_functions import combine_sentences
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.docstore.document import Document
import spacy
from segment_clustering import cluster_segment
import logging
import random
import matplotlib.pyplot as plt

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

def semantic_chunking(text, percentiles=85, max_token_length=4000):
    # Step 1: Split the text into sentences
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]  # Extract sentences from spaCy
    
    # Step 2: Group sentences (currently grouping 3 at a time, but this can be adjusted)
    combined_sentences = combine_sentences(sentences, buffer_size=1)
    
    # Step 3: Tokenize and embed each group of combined sentences
    sentence_embeddings = []
    for group in combined_sentences:
        tokenized_group = tokenizer(group, return_tensors='pt', truncation=False)['input_ids'][0]
        if len(tokenized_group) <= 8000:
          embedded_group = model.encode(group, convert_to_tensor=True)
          sentence_embeddings.append(embedded_group)
    print("done")
    # Step 4: Compare cosine similarity between consecutive sentence groups
    cosine_similarities = []
    for i in range(len(sentence_embeddings) - 1):
        cos_sim = util.pytorch_cos_sim(sentence_embeddings[i], sentence_embeddings[i+1])
        cosine_similarities.append(cos_sim.item())  # Convert tensor to float
    similarity_threshold = np.percentile(cosine_similarities, percentiles)
    print(f"Calculated {percentiles}th percentile similarity threshold: {similarity_threshold}")

    # Step 5: Chunk based on the cosine similarity threshold and token length limit
    chunks = []
    current_chunk = combined_sentences[0]  # Start with the first group of sentences
    current_chunk_tokens = tokenizer(current_chunk, return_tensors='pt', truncation=False)['input_ids'][0]
    
    total_chunks = 0
    for i in range(len(cosine_similarities)):
        next_group = combined_sentences[i+1]
        next_group_tokens = tokenizer(next_group, return_tensors='pt', truncation=False)['input_ids'][0]

        if cosine_similarities[i] >= similarity_threshold:
            # Check if adding the next group exceeds the token limit
            print(f"Current chunk tokens: {len(current_chunk_tokens)}, Next group tokens: {len(next_group_tokens)}")
            if len(current_chunk_tokens) + len(next_group_tokens) <= max_token_length:
                # If not, merge the next group into the current chunk
                current_chunk += " " + next_group
                current_chunk_tokens = torch.cat((current_chunk_tokens, next_group_tokens))  # Merge tokens
            else:
                # If it exceeds the limit, store the current chunk and start a new one
                chunks.append((current_chunk, len(current_chunk_tokens)))
                logging.info(f"Chunk {total_chunks + 1}: Size = {len(current_chunk_tokens)} tokens.")
                total_chunks += 1

                # Start a new chunk with the next group
                current_chunk = next_group
                current_chunk_tokens = next_group_tokens
        else:
            # If the similarity is low, store the current chunk
            chunks.append((current_chunk, len(current_chunk_tokens)))
            logging.info(f"Chunk {total_chunks + 1}: Size = {len(current_chunk_tokens)} tokens.")
            total_chunks += 1

            # Start a new chunk
            current_chunk = next_group
            current_chunk_tokens = next_group_tokens

    # Add the last chunk if not already added
    if current_chunk:
        chunks.append((current_chunk, len(current_chunk_tokens)))
        logging.info(f"Chunk {total_chunks + 1}: Size = {len(current_chunk_tokens)} tokens.")
        total_chunks += 1

    # Step 6: Return the chunks and their token sizes
    print(f"Semantic Chunking complete. Number of chunks: {len(chunks)}")
    for i, (chunk, token_size) in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:20]}... (Token size: {token_size})")  # Print the first 20 characters of each chunk and token size
    
    return chunks

def chunk_text_by_segment(text, seg_array, tokenizer, title=None, doc_id=None):
    # Load spaCy model for sentence segmentation
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]  # Extract sentences from spaCy
    max_chunk_size = 4096
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
    

def create_segmendtaion_faiss_index_from_directory(json_directory_path, 
                                                   output_faiss_path, 
                                                   output_ids_path):
    # Prepare lists to store embeddings and document info
    embeddings = []
    document_chunks = []
    chunk_size = determine_chunk_size()

    total_chunk_size = 0  # Variable to track total size of all chunks
    total_chunks_count = 0  # Variable to track total number of chunks
    chunk_id = 1
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
            # Split the text into smaller chunks that can be tokenized within model limits
            if args.chunk_type == '256' or args.chunk_type == '512' or args.chunk_type == '1024' or args.chunk_type == '2048':
                text_chunks = chunk_text_by_tokens(content, chunk_size, tokenizer)
            elif args.chunk_type == 'semantic':
                text_chunks = semantic_chunking(content, tokenizer)
            else:
                text_chunks = chunk_text_by_segment(content, segmented_sentences, tokenizer, title, doc_id)
            print(f"Processing document {doc_id} with {len(text_chunks)} chunks")
            # Iterate through each chunk and its size
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
                    'chunk_size': c_size,  # Include the size of the chunk
                    'embedding': embedding.tolist()  # Convert to list for JSON serialization
                })
                chunk_id += 1
                embeddings.append(embedding)

     # Save the document chunks with IDs and embeddings
    
    # Save the document chunks with IDs and embeddings
    os.makedirs(os.path.dirname(output_ids_path), exist_ok=True)
    with open(output_ids_path, 'w', encoding='utf-8') as id_file:
        json.dump(document_chunks, id_file, ensure_ascii=False, indent=4)
    # Should have a step here to determine that we're segment_cluster or not here

    if args.is_cluster == True:    
        with open(output_ids_path, 'r', encoding='utf-8') as id_file:
            loaded_data = json.load(id_file)
        loaded_data, embeddings, total_chunk_size, total_chunks_count = cluster_segment(loaded_data, embeddings, args.max_file_size, args.k)
        with open(output_ids_path, 'w', encoding='utf-8') as id_file:
            json.dump(loaded_data, id_file, ensure_ascii=False, indent=4)

    if args.is_mul_vector == True:
        flat_embeddings = []
        for chunk in loaded_data:
            chunk_embeddings = [np.array(chunk['cluster_embedding'], dtype=np.float32)] + [np.array(emb, dtype=np.float32) for emb in chunk['segment_embeddings']]
            if len(chunk_embeddings) != 2:
                flat_embeddings.extend(chunk_embeddings)
                chunk['vector_count'] = len(chunk_embeddings)
            else:
                flat_embeddings.append(np.array(chunk['cluster_embedding'], dtype=np.float32))
                chunk['vector_count'] = 1
            print(f"Chunk {chunk['chunk_id']} has {chunk['vector_count']} vectors")

        embeddings = np.array(flat_embeddings, dtype=np.float32)
    else:
        embeddings = np.array(embeddings, dtype=np.float32)

    # Ensure embeddings are contiguous in memory
    embeddings = np.ascontiguousarray(embeddings)

    # Create a FAISS index
    embedding_dim = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    index.add(embeddings)  # Add the embeddings to the index

    # Save the FAISS index
    os.makedirs(os.path.dirname(output_faiss_path), exist_ok=True)
    faiss.write_index(index, output_faiss_path)

    with open(output_ids_path, 'w', encoding='utf-8') as id_file:
        json.dump(loaded_data, id_file, ensure_ascii=False, indent=4)


    logging.info(f"Total number of chunks for chunk type {args.max_file_size} the dataset {args.dataset} is: {len(embeddings)}")
    logging.info(f"Avarage chunk size is : {total_chunk_size/total_chunks_count}")
    print(f"Total number of chunks for chunk type {args.max_file_size} the dataset {args.dataset} is: {len(embeddings)}")
    print(f"Avarage chunk size is : {total_chunk_size/total_chunks_count}")
    print(f"FAISS index and document chunk information have been saved to {output_faiss_path} and {output_ids_path}")

def main(args):
    if args.dataset:
        if args.max_file_size != 0:
            json_directory_path = f'{args.original_data}/{args.dataset}/individual_documents_2048'
        else:
            json_directory_path = f'{args.original_data}/{args.dataset}/individual_documents'
        # embedding_testing()
        logging.basicConfig(filename=f'{args.dataset}_{args.chunk_type}_embedding.txt', level=logging.INFO)
        create_segmendtaion_faiss_index_from_directory(json_directory_path=json_directory_path, 
                                     output_faiss_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.index', 
                                     output_ids_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json')
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please choose 'squad' or 'narrativeqa' or 'quality'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whenever it is squad or narrative_qa',  required=True, type=str, default="squad")
    parser.add_argument('--chunk_type', help='What is the chunking strategy: 256, 512, seg, segclus, semantic', type=str, default='256')
    parser.add_argument('--max_file_size', help='Output path for the embeddings', type=int, default=0)
    parser.add_argument('--k', help='k variable', type=float, default=0.5)
    parser.add_argument('--is_cluster', help='Enable clustering of segments', action='store_true')
    parser.add_argument('--is_mul_vector', help='Enable multiple vectors', action='store_true')
    parser.add_argument('--original_data', help='Enable data path', type=str, default='data_512_1024') 
    args = parser.parse_args() 
    main(args)

# Example running command
# python embedding.py --dataset squad --chunk_type 256