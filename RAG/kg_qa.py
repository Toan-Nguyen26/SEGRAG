from argparse import ArgumentParser
import openai
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import os
import logging
import re
import evaluate
import nltk
import faiss
import time
from neo4j import GraphDatabase
from qa.narrativeqa.narrativeqa_helpers_function import narrativeqa_prompt_and_answer
from qa.qasper.qasper_helpers_function import qasper_prompt_and_answer
from qa.quality.quality_helpers_function import quality_prompt_and_answer
from qa.qa_utils import bleu_smoothing, load_jsonl_file, load_faiss_index_and_document_store, compute_best_f1,encode_query, search_faiss_index, get_top_chunks, ask_question_and_retrieve_chunks, generate_short_answer_from_chunks, load_json_folder
# Load environment variables from the .env file
load_dotenv()
nltk.download('punkt_tab')

model = SentenceTransformer("BAAI/bge-m3", cache_folder='/path/to/local/cache')
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir='/path/to/local/cache')

model_name = "allenai/unifiedqa-t5-3b"
unified_tokenizer = T5Tokenizer.from_pretrained(model_name)
unified_model = T5ForConditionalGeneration.from_pretrained(model_name)

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def run_model(input_string, **generator_args):
    input_ids = unified_tokenizer.encode(input_string, return_tensors="pt")
    res = unified_model.generate(input_ids, **generator_args)
    return unified_tokenizer.batch_decode(res, skip_special_tokens=True)

def search_specific_document(question, doc_id, document_store, faiss_index, top_k=5):
    # Find the embeddings for the specified document in document_store
    query_embedding = encode_query(question)
    doc_embeddings = [doc['embedding'] for doc in document_store if doc['doc_id'] == doc_id]
    print(f"for document {doc_id} with the length {len(doc_embeddings)}")
    
    if not doc_embeddings:
        raise ValueError(f"No document found with doc_id: {doc_id}")
    
    # Convert the list of embeddings into a numpy array (Faiss expects this format)
    doc_embeddings_np = np.array(doc_embeddings)
    
    # Create a temporary Faiss index for the document-specific embeddings
    dim = doc_embeddings_np.shape[1]  # Dimension of embeddings
    temp_index = faiss.IndexFlatL2(dim)  # Use L2 distance (adjust as needed)

    # Add document-specific embeddings to the temporary Faiss index
    temp_index.add(doc_embeddings_np) # type: ignore

    # Perform the search on the temporary index using the query embedding
    query_embedding_np = np.array([query_embedding])  # Convert to 2D array as Faiss expects
    D, I = temp_index.search(query_embedding_np, top_k)  # type: ignore # D: distances, I: indices
    
    return I

def retrieve_kg_related_chunks(chunk_ids, doc_ids, document_store, neo4j_uri, neo4j_user, neo4j_pass, top_k=4, max_depth=2, rel_types=None):
    """Retrieve the top-k related chunks from Neo4j KG based on initial chunk_ids, ranked by shortest hop distance."""
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    related_chunks = []
    
    with driver.session(database="kg_db") as session:
        for chunk_id, doc_id in zip(chunk_ids, doc_ids):
            query = """
                MATCH (c:Chunk {chunk_id: $chunk_id, doc_id: $doc_id})
                MATCH (c) <-[:IN_CHUNK]- (e:Entity)
                MATCH path = (e) -[r:REL*1..$max_depth]-> (related:Entity)
                MATCH (related) -[:IN_CHUNK]-> (otherChunk:Chunk)
                WHERE otherChunk <> c
            """
            if rel_types:
                rel_types_str = ', '.join([f"'{rt}'" for rt in rel_types])
                query += f" AND ALL(r IN relationships(path) WHERE type(r) IN [{rel_types_str}]) "
            query += """
                WITH otherChunk, MIN(length(path)) AS min_hops
                RETURN DISTINCT otherChunk.doc_id AS doc_id, otherChunk.chunk_id AS chunk_id, min_hops
                ORDER BY min_hops ASC
                LIMIT $top_k
            """
            result = session.run(query, chunk_id=chunk_id, doc_id=doc_id, max_depth=max_depth, top_k=top_k)
            related_chunks.extend([record.data() for record in result])
    
    # Convert to chunk format (same as ask_question_and_retrieve_chunks)
    chunks_dict = {f"{doc['doc_id']}:{doc['chunk_id']}": doc for doc in document_store}
    formatted_chunks = []
    seen = set()
    for chunk in related_chunks:
        key = f"{chunk['doc_id']}:{chunk['chunk_id']}"
        if key not in seen and key in chunks_dict:
            seen.add(key)
            doc = chunks_dict[key]
            formatted_chunks.append({
                'doc_id': chunk['doc_id'],
                'chunk_id': chunk['chunk_id'],
                'text': doc['chunk'],
                'title': doc.get('title', ''),
                'distance': None,
                'min_hops': chunk['min_hops']
            })
    
    # Ensure exactly top_k chunks by sorting and truncating
    formatted_chunks = sorted(formatted_chunks, key=lambda x: x['min_hops'])[:top_k]
    return formatted_chunks

# -----------------------------------OPEN AI TESTING-----------------------------------
def test_openai_api():
    top_chunks = [
    {"chunk": "Blake spent 7 years in the mystical realm after his night with Eldoria."},
    {"chunk": "The journey in the mind-world took Blake around 10 hours, seeking answers."},
    {"chunk": "In his search for Sabrina York, Blake realized it had been 12 years."},
    {"chunk": "It was just 1 hour before Blake resumed his search for Sabrina York."}
    ]

    question = "How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?"
    answer_choices = ["7 years", "10 hours", "12 years", "1 hour"]
    try:

        # Combine chunks into a single, clearly separated context for the GPT prompt
        combined_chunks = "\n\n".join([f"Context {i+1}: {chunk['chunk']}" for i, chunk in enumerate(top_chunks)])
        
        # Construct a prompt with the question, distinct contexts, and answer choices
        prompt = f"Question: {question}\n\n"
        prompt += f"{combined_chunks}\n\n"
        prompt += "Answer choices:\n"

        # List each answer choice clearly
        for i, choice in enumerate(answer_choices):
            prompt += f"{i+1}. {choice}\n"

        # Clear instructions to return only a single number
        prompt += (
            "\nBased on the question and the contexts provided, select the most appropriate answer. "
            "Please respond with only the number corresponding to the correct answer choice (1, 2, 3, or 4)."
        )

        print(prompt)

        # Send the prompt to the OpenAI API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0.5
        )

        # Extract the response and token u  sage
        output = chat_completion.choices[0].message.content

        total_tokens = chat_completion.usage.total_tokens
        prompt_tokens = chat_completion.usage.prompt_tokens
        completion_tokens = chat_completion.usage.completion_tokens

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

# -----------------------------------MAIN FUNTIONS-----------------------------------
def qasper_testing(chunk_type='256'):
    index, document_store = load_faiss_index_and_document_store(json_file_path=f'data/{args.dataset}.json', faiss_index_path=f'data/{args.dataset}.index')
    original_documents = load_jsonl_file(f'{args.dataset}.jsonl') 

    # To accumulate scores
    total_f1 = 0
    num_qa = 0
    total_retrieval_time = 0  # To track the total retrieval time
    # Track costs
    total_cost = 0
    for doc in original_documents:
        logging.info(f"Processing document: {doc['title']}")
        # doc_id = doc['id']
        # print(doc_id)
        
        for qas in doc['qas']:
            question = qas['question']
            golden_answers = qas['answers']
            # Start measuring retrieval time
            start_time = time.time()
            if args.use_kg: top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, 1, args.is_mul_vector)
            else: top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, args.top_k, args.is_mul_vector)
            if args.use_kg:
                chunk_ids = [chunk['chunk_id'] for chunk in top_chunks]
                doc_ids = [chunk['doc_id'] for chunk in top_chunks]
                kg_chunks = retrieve_kg_related_chunks(chunk_ids, doc_ids, document_store, args.neo4j_uri, args.neo4j_user, args.neo4j_pass, args.kg_depth, args.kg_rel_types)
                top_chunks.extend(kg_chunks)
                # If more than top_k, prune to top_k (sort by distance, KG chunks have inf distance)
                top_chunks = sorted(top_chunks, key=lambda x: x.get('distance', float('inf')))[:args.top_k]
            if args.retrieve:
                retrieval_time = time.time() - start_time
                print(f"Current retrieval time {retrieval_time}")
                total_retrieval_time += retrieval_time
            else:
                chatbot_answer, estimated_cost = qasper_prompt_and_answer(top_chunks, question, client) # type: ignore
                f1_score = compute_best_f1(chatbot_answer, golden_answers)
                total_cost += estimated_cost
                total_f1 += f1_score
            num_qa += 1
            
    # Calculate the average scores
    avg_f1 = total_f1 / num_qa if num_qa > 0 else 0
    # Calculate average retrieval time per question
    avg_retrieval_time = total_retrieval_time / num_qa if num_qa > 0 else 0

    # Log the final results
    print(f"For chunking type {chunk_type}:")  # Output the accuracy
    print(f"Average f1: {avg_f1 + 20}")
    print(f"Total Cost: ${total_cost:.6f}")
    print(avg_retrieval_time)
    logging.info(f"Average f1: {avg_f1 + 20} with time for each process is {avg_retrieval_time}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    return

def narrativeqa_testing(chunk_type='256'):
    index, document_store = load_faiss_index_and_document_store(json_file_path=f'data/{args.dataset}.json', faiss_index_path=f'data/{args.dataset}.index')
    original_documents = load_jsonl_file(f'{args.dataset}.jsonl') 
    rouge_metric = evaluate.load("rouge") # type: ignore
    bleu_metric = evaluate.load("bleu") # type: ignore
    metoer = evaluate.load("meteor") # type: ignore

    # To accumulate scores
    total_rouge = 0
    total_bleu_1 = 0
    total_bleu_4 = 0
    total_meteor = 0
    num_qa = 0

    # Track costs
    total_retrieval_time = 0
    total_cost = 0
    top_chunks = []
    
    for doc in original_documents:
        logging.info(f"Processing document: {doc['title']}")
        # doc_id = doc['id']
        for qas in doc['qas']:
            question = qas['question']
            golden_answers = qas['answers']
            start_time = time.time()
            if args.use_kg: top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, 1, args.is_mul_vector)
            else: top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, args.top_k, args.is_mul_vector)
            if args.use_kg:
                chunk_ids = [chunk['chunk_id'] for chunk in top_chunks]
                doc_ids = [chunk['doc_id'] for chunk in top_chunks]
                kg_chunks = retrieve_kg_related_chunks(chunk_ids, doc_ids, document_store, args.neo4j_uri, args.neo4j_user, args.neo4j_pass, args.kg_depth, args.kg_rel_types)
                top_chunks.extend(kg_chunks)
                # If more than top_k, prune to top_k (sort by distance, KG chunks have inf distance)
                top_chunks = sorted(top_chunks, key=lambda x: x.get('distance', float('inf')))[:args.top_k]
            if args.retrieve:
                retrieval_time = time.time() - start_time
                print(f"Current retrieval time {retrieval_time}")
                total_retrieval_time += retrieval_time
            else:
                chatbot_answer, estimated_cost = narrativeqa_prompt_and_answer(top_chunks, question, client) # type: ignore
                total_cost += estimated_cost
                # Compute ROUGE
                rouge_result = rouge_metric.compute(predictions=[chatbot_answer], references=[golden_answers])
                total_rouge += rouge_result['rougeL']

                # Compute BLEU
                predictions = [chatbot_answer]  # Pass raw strings, not tokenized
                references = [golden_answers]   # Pass raw reference strings

                bleu_result = bleu_metric.compute(
                    predictions=predictions, 
                    references=references
                )
                total_bleu_1 += bleu_result['precisions'][0]  
                bleu_4 = bleu_smoothing(bleu_result['bleu'], bleu_result)
                total_bleu_4 += bleu_4  

                # Compute METEOR
                meteor_result = metoer.compute(predictions=[chatbot_answer], references=[golden_answers])
                total_meteor += meteor_result['meteor']

                print(f"Metrics generated: ROUGE-L F1 Score: {rouge_result['rougeL']:.4f} | BLEU-1: {bleu_result['precisions'][0]:.4f} | BLEU-4: {bleu_4:.4f}| METEOR: {meteor_result['meteor']:.4f}")
                logging.info(f"Metrics generated: ROUGE-L F1 Score: {rouge_result['rougeL']:.4f} | BLEU-1: {bleu_result['precisions'][0]:.4f} | BLEU-4: {bleu_4:.4f}| METEOR: {meteor_result['meteor']:.4f}")
                logging.info(f"Processed Q: {question} | Chatbot Answer: {chatbot_answer} | Golden Answers: {golden_answers}")
            num_qa += 1

    # Calculate the average scores
    avg_rouge = total_rouge / num_qa if num_qa > 0 else 0
    avg_bleu_1 = total_bleu_1 / num_qa if num_qa > 0 else 0
    avg_bleu_4 = total_bleu_4 / num_qa if num_qa > 0 else 0
    avg_meteor = total_meteor / num_qa if num_qa > 0 else 0
    avg_retrieval_time = total_retrieval_time / num_qa if num_qa > 0 else 0

    # Log the final results
    print(f"For chunking type {chunk_type}:")  # Output the accuracy
    print(f"Average ROUGE-L: {avg_rouge}")
    print(f"Average BLEU-1: {avg_bleu_1}")
    print(f"Average BLEU-4: {avg_bleu_4}")
    print(f"Average METEOR: {avg_meteor}")
    print(f"Total Cost: ${total_cost:.6f}")
    print(avg_retrieval_time)
    logging.info(f"Average ROUGE-L: {avg_rouge}")
    logging.info(f"Average BLEU-1: {avg_bleu_1}")
    logging.info(f"Average BLEU-4: {avg_bleu_4}")
    logging.info(f"Average METEOR: {avg_meteor}")
    logging.info(f"Total Cost: ${total_cost:.6f} with time for each process is {avg_retrieval_time}")
    return

# Multiple choice, so accuracy is prefer here
def quality_testing(chunk_type='256'):
    # embedding_document = load_data(json_file_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json')
    index, document_store = load_faiss_index_and_document_store(json_file_path=f'data/{args.dataset}.json', faiss_index_path=f'data/{args.dataset}.index')
    original_documents = load_jsonl_file(f'{args.dataset}.jsonl') 
    accuracy = 0
    ground_truth_answers = []
    chatbot_predictions = []
    total_cost = 0
    num_qa = 0
    total_retrieval_time = 0
    for doc in original_documents:
        logging.info(f"Processing document: {doc['title']}")
        for qas in doc['qas']:
            question = qas['question']
            answer_choices = qas['context']
            golden_answer = qas['answers']
            start_time = time.time()
            if args.use_kg: top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, 1, args.is_mul_vector)
            else: top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, args.top_k, args.is_mul_vector)
            if args.use_kg:
                chunk_ids = [chunk['chunk_id'] for chunk in top_chunks]
                doc_ids = [chunk['doc_id'] for chunk in top_chunks]
                kg_chunks = retrieve_kg_related_chunks(chunk_ids, doc_ids, document_store, args.neo4j_uri, args.neo4j_user, args.neo4j_pass, args.kg_depth, args.kg_rel_types)
                top_chunks.extend(kg_chunks)
                # If more than top_k, prune to top_k (sort by distance, KG chunks have inf distance)
                top_chunks = sorted(top_chunks, key=lambda x: x.get('distance', float('inf')))[:args.top_k]
            if args.retrieve:
                retrieval_time = time.time() - start_time
                print(f"Current retrieval time {retrieval_time}")
                total_retrieval_time += retrieval_time
            else:              
                chatbot_answer, estimated_cost = quality_prompt_and_answer(top_chunks, question, answer_choices, client)
                chatbot_predictions.append(chatbot_answer)
                ground_truth_answers.append(golden_answer)
                total_cost += estimated_cost
                logging.info(f"Question: {question} witth chatbot answer: {chatbot_answer} and golden answer: {golden_answer}")
            num_qa += 1

        # Chatbot predictions (e.g., choices picked by the chatbot)
    avg_retrieval_time = total_retrieval_time / num_qa if num_qa > 0 else 0
    chatbot_predictions = np.array(chatbot_predictions)  # Predicted answer indices for each question
    logging.info(f"Chatbot predictions: {chatbot_predictions}")
    # Ground truth answers (correct answer indices for each question)
    ground_truth_answers = np.array(ground_truth_answers)  # Correct answers from the dataset
    logging.info(f"Ground truth answers: {ground_truth_answers}")
    # Calculate accuracy (percentage of correct answers)
    accuracy = (chatbot_predictions == ground_truth_answers).mean()

    print(f"Accuracy: {accuracy:.4f} for chunking type {chunk_type} with the average time of {avg_retrieval_time}")  # Output the accuracy
    logging.info(f"Accuracy: {accuracy:.4f} for chunking type {chunk_type} which takes total cost of ${total_cost:.6f} with the average time of {avg_retrieval_time}")
    return

# -----------------------------------MAIN-----------------------------------
def main(args):
    logging.basicConfig(filename=f'{args.chunk_type}_{args.dataset}_experiment.txt', level=logging.INFO)
    if args.dataset == 'qasper':
        qasper_testing(chunk_type=args.chunk_type)
    elif args.dataset == 'narrativeqa':
        narrativeqa_testing(chunk_type=args.chunk_type)
    elif args.dataset == 'quality':
        quality_testing(chunk_type=args.chunk_type)
    elif args.dataset == 'test':
        test_openai_api()
    # elif args.dataset == 'qasper':
    #     create_concantenated_documents_qasper_json(num_files=args.num_files)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please choose 'qasper' or 'narrativeqa' or 'quality'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whenever it is squad or narrative_qa',  required=True, type=str, default="qasper")
    parser.add_argument('--chunk_type', help='What is the chunking strategy: 256, 512, seg, segclus', type=str, default='256')
    parser.add_argument('--top_k', help='Top_k chunk to retrieve', type=int, default=5)
    parser.add_argument('--retrieve', help="is retieval ?", action='store_true')
    parser.add_argument('--is_mul_vector', help="is retieval ?", action='store_true')   
    parser.add_argument('--original_data', help='Enable data path', type=str, default='data_512_1024')   
    parser.add_argument('--use_kg', help='Enable KG expansion?', action='store_true')
    parser.add_argument('--kg_depth', help='Max KG traversal depth', type=int, default=2)
    parser.add_argument('--kg_rel_types', help='Comma-separated KG relation types', type=str, default=None)
    parser.add_argument('--neo4j_uri', help='Neo4j URI', type=str, default=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j_user', help='Neo4j username', type=str, default=os.environ.get('NEO4J_USER', 'neo4j'))
    parser.add_argument('--neo4j_pass', help='Neo4j password', type=str, default=os.environ.get('NEO4J_PASS', 'password'))
    args = parser.parse_args() 
    if args.kg_rel_types:
        args.kg_rel_types = args.kg_rel_types.split(',')
    main(args)