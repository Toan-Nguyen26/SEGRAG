import faiss
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
# Load environment variables from the .env file
load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
# Load your FAISS index
index = faiss.read_index(os.getenv("FAISS_INDEX_512_PATH"))

# Load the document metadata (e.g., original texts or chunk info)
with open(os.getenv("DOCUMENT_IDS_512_PATH"), 'r', encoding='utf-8') as f:
    document_store = json.load(f)

# Load the model for encoding queries
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")


def encode_query(query):
    query_embedding = model.encode([query])
    return query_embedding

def search_faiss_index(query_embedding, top_k=5):
    # Perform the search on FAISS index
    D, I = index.search(query_embedding, top_k)
    return I  # Return the indices of the top chunks

def get_top_chunks(indices):
    top_chunks = []
    for idx in indices[0]:
        # Assuming document_store contains the relevant chunk text and metadata
        chunk_info = {
            'id': document_store[idx]['id'],
            'doc_id': document_store[idx]['doc_id'],
            'chunk': document_store[idx]['chunk'],
            'embedding': document_store[idx]['embedding']
        }
        top_chunks.append(chunk_info)
    return top_chunks

def ask_question_and_retrieve_chunks(question):
    query_embedding = encode_query(question)
    indices = search_faiss_index(query_embedding, top_k=10)
    top_chunks = get_top_chunks(indices)
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

def load_data(json_file_path='data/concatenated_documents_new.json'):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documents = json.load(json_file)
    return documents

def test_openai_api():
    try:
        
        # Define a simple prompt
        prompt = "Hello, how are you today?"

        # Send the prompt to the OpenAI API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Who's Alexander Hamilton",
                }
            ],
            model="gpt-4o-mini",
            max_tokens=100
        )

        # Extract the response and token u  sage
        output = chat_completion.choices[0].message
        total_tokens = chat_completion.usage.total_tokens
        prompt_tokens = chat_completion.usage.prompt_tokens
        completion_tokens = chat_completion.usage.completion_tokens

        # output = chat_completion.['choices'][0]['message']
        # total_tokens = chat_completion['usage']['total_tokens']
        # prompt_tokens = chat_completion['usage']['prompt_tokens']
        # completion_tokens = chat_completion['usage']['completion_tokens']

        # Calculate cost (estimate)
        # As per the latest pricing for gpt-4o-mini:
        # $0.150 per 1,000,000 prompt tokens (input)
        # $0.600 per 1,000,000 completion tokens (output)
        cost_per_1M_prompt_tokens = 0.150  # $ per 1M input tokens
        cost_per_1M_completion_tokens = 0.600  # $ per 1M output tokens

        prompt_cost = (prompt_tokens / 1_000_000) * cost_per_1M_prompt_tokens
        completion_cost = (completion_tokens / 1_000_000) * cost_per_1M_completion_tokens
        estimated_cost = prompt_cost + completion_cost

        # Print the results
        print("API Test Response:")
        print(output)
        print(f"\nTotal Tokens Used: {total_tokens}")
        print(f"Prompt Tokens Used: {prompt_tokens}")
        print(f"Completion Tokens Used: {completion_tokens}")
        print(f"Estimated Cost: ${estimated_cost:.6f}")

    except Exception as e:
        print(f"An error occurred: {e}")

def test_doc_length(json_file_path='data/concatenated_documents_new.json'):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documents = json.load(json_file)

    # Initialize a counter for the total number of questions
    total_questions = 0

    # Loop through each document to count the questions
    for i, doc in documents:
        total_questions += len(doc[i]['qas'])

    print(f"Total number of questions in the JSON: {total_questions}")

# Test the OpenAI API
# test_doc_length()

if __name__ == "__main__":
    documents = load_data()
    all_results = []
    correct_retrievals = 0
    # for doc in documents:
    for qa in documents[0]['qas']:
        question = qa['question']
        golden_answers = qa['answers']
        top_chunks = ask_question_and_retrieve_chunks(question)
        # short_answer = generate_short_answer_from_chunks(question, top_chunks)

        # Check if the golden answer is in any of the retrieved chunks
        is_answer_found = any(any(answer in chunk['chunk'] for answer in golden_answers) for chunk in top_chunks)

        # Increment correct_retrievals if answer is found
        if is_answer_found:
            correct_retrievals += 1

            # Prepare result entry
        result = {
            'question': question,
            # 'short_answer': short_answer,
            "golden_answer": golden_answers,
            "golden_context": qa['context'],
            'chunks': [
                {
                    'id': chunk['id'],
                    'document_id': chunk['doc_id'],
                    'text': chunk['chunk']
                }
                for chunk in top_chunks
            ]
        }
        all_results.append(result)

        print(f"Question: {question}")
        # print(f"Short Answer: {short_answer}")
        print(f"Answer Found in Chunks: {is_answer_found}")
        print("--------------------------------------------------")

    # Save all results to a JSON file
    # output_file = 'retrieved_short_answers.json'
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=4)

    # print(f"Results have been saved to {output_file}")

# Calculate overall Answer Recall (AR)
answer_recall = correct_retrievals / len(documents[0]['qas'])
print(f"Overall Answer Recall (AR): {answer_recall:.2f}")
# total_questions = len(documents[0]['qas'])
# correct_retrievals = 0

