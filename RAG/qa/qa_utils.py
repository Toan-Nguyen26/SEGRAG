import faiss
import json
import os
import openai
from sentence_transformers import SentenceTransformer
import evaluate

f1_metric = evaluate.load("f1") # type: ignore
model = SentenceTransformer("BAAI/bge-m3", cache_folder='/path/to/local/cache')
# Smoothing function for BLEU-4 score , this might not be good but we have to do it though

def compute_best_f1(chatbot_answer, golden_answers):
    best_f1 = 0
    for golden_answer in golden_answers:
        # Tokenize both strings (you can use more sophisticated tokenizers if needed)
        golden_answer = str(golden_answer)
        chatbot_tokens = chatbot_answer.split()
        golden_tokens = golden_answer.split()

        # Create sets for both
        chatbot_set = set(chatbot_tokens)
        golden_set = set(golden_tokens)

        # Compute true positives, false positives, false negatives
        true_positives = len(chatbot_set & golden_set)
        false_positives = len(chatbot_set - golden_set)
        false_negatives = len(golden_set - chatbot_set)

        # Compute precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Compute F1
        if precision + recall > 0:
            current_f1 = 2 * (precision * recall) / (precision + recall)
        else:
            current_f1 = 0

        # Keep track of the highest F1 score
        if current_f1 > best_f1:
            best_f1 = current_f1

    print("Best F1 score:", best_f1)
    return best_f1

def bleu_smoothing(bleu_4, bleu_result):
    if bleu_4 == 0:
        for precision in reversed(bleu_result['precisions']):
            if precision > 0:
                return precision
        return 0.1
    else:
        return bleu_4

def load_faiss_index_and_document_store(json_file_path, faiss_index_path):
    # Load your FAISS index
    index = faiss.read_index(faiss_index_path)

    # Load the document metadata (e.g., original texts or chunk info)
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        document_store = json.load(json_file)
    
    return index, document_store

def encode_query(query):
    query_embedding = model.encode([query])
    return query_embedding

def search_faiss_index(query_embedding, index, top_k=5):
    # Perform the search on FAISS index
    D, I = index.search(query_embedding, top_k)
    return I  # Return the indices of the top chunks

def get_top_chunks(indices, document_store):
    top_chunks = []
    for idx in indices[0]:
        # Assuming document_store contains the relevant chunk text and metadata
        chunk_info = {
            'title': document_store[idx]['title'],
            'doc_id': document_store[idx]['doc_id'],
            'chunk': document_store[idx]['chunk'],
            'cluster_embedding': document_store[idx]['cluster_embedding'],
            'segment_embeddings': document_store[idx]['segment_embeddings']
        }
        top_chunks.append(chunk_info)
    return top_chunks

def handle_multiple_vectors(indices, document_store, top_k):
    # Search all vectors
    top_chunks = []
    seen_chunks = set()
    for idx in indices[0]:
        # Assuming document_store contains the relevant chunk text and metadata
        # Get chunk info from the document store
        chunk_text = document_store[idx]['chunk']
        
        # Check if the chunk already exists in top_chunks
        if chunk_text in seen_chunks:
            print(f"Skipping duplicate chunk: {chunk_text}")
            continue  # Skip if the chunk already exists
        chunk_info = {
            'title': document_store[idx]['title'],
            'doc_id': document_store[idx]['doc_id'],
            'chunk': document_store[idx]['chunk'],
            'cluster_embedding': document_store[idx]['cluster_embedding'],
            'segment_embeddings': document_store[idx]['segment_embeddings']
        }
        seen_chunks.add(chunk_text)  # Mark this chunk as seen
        top_chunks.append(chunk_info)
        # Break if we've reached top_k
        if len(top_chunks) >= top_k:
            break

    return top_chunks

    
    chunk_scores = {}
    current_index = 0
    for chunk in document_store:
        vector_count = chunk['vector_count']
        chunk_scores[chunk['chunk_id']] = max(D[0][current_index:current_index + vector_count])
        current_index += vector_count

    # Sort chunks by their highest scores and get top_k
    top_chunk_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:top_k]
    return [chunk for chunk in document_store if chunk['chunk_id'] in top_chunk_ids]

def ask_question_and_retrieve_chunks(question, index, document_store, top_k, is_mul_vector):
    query_embedding = encode_query(question)
    if is_mul_vector:
        indices = search_faiss_index(query_embedding, index, 50)
        top_chunks = handle_multiple_vectors(indices, document_store, top_k)
    else:
        indices = search_faiss_index(query_embedding, index, top_k)
        top_chunks = get_top_chunks(indices, document_store)
    return top_chunks

def generate_short_answer_from_chunks(question, chunks):
    # Create a prompt by concatenating the chunks
    chunk_text = " ".join([chunk['chunk'] for chunk in chunks])
    prompt = f"Based on the following information, answer the question in less than 100 tokens:\n\n{chunk_text}\n\nQuestion: {question}"
    # Send the prompt to the API
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # You can use "gpt-4" if you have access to that model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100  # Limit the response to 100 tokens
    )

    # Extract the response
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def load_data(json_file_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documents = json.load(json_file)
    return documents

def load_json_folder(folder_path):
    # Load all JSON files from the specified folder
    json_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)
                json_files.append(json_data)
    return json_files
