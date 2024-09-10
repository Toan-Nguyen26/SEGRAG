import faiss
import json
import os
import openai
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3", cache_folder='/path/to/local/cache')
# Smoothing function for BLEU-4 score , this might not be good but we have to do it though

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
            'embedding': document_store[idx]['embedding']
        }
        top_chunks.append(chunk_info)
    return top_chunks

def ask_question_and_retrieve_chunks(question, index, document_store, top_k):
    query_embedding = encode_query(question)
    indices = search_faiss_index(query_embedding, index, top_k)
    top_chunks = get_top_chunks(indices, document_store)
    return top_chunks

def generate_short_answer_from_chunks(question, chunks):
    # Create a prompt by concatenating the chunks
    chunk_text = " ".join([chunk['chunk'] for chunk in chunks])
    prompt = f"Based on the following information, answer the question in less than 100 tokens:\n\n{chunk_text}\n\nQuestion: {question}"
    print(prompt)
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
