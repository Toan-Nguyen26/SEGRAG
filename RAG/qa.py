from argparse import ArgumentParser
import faiss
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer
import numpy as np
import os
import logging
import re
import evaluate
from qa.qa_utils import bleu_smoothing, load_faiss_index_and_document_store, encode_query, search_faiss_index, get_top_chunks, ask_question_and_retrieve_chunks, generate_short_answer_from_chunks, load_json_folder
# Load environment variables from the .env file
load_dotenv()

model = SentenceTransformer("BAAI/bge-m3", cache_folder='/path/to/local/cache')
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir='/path/to/local/cache')

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

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

def qasper_prompt_and_answer(top_chunks, question):
    try:
        # First Turn: Generate a longer, relevant answer based on the context
        combined_chunks = "\n\n".join([f"Context {i+1}: {chunk['chunk']}" for i, chunk in enumerate(top_chunks)])

        first_turn_prompt = f"Question: {question}\n\n"
        first_turn_prompt += f"{combined_chunks}\n\n"
        first_turn_prompt += (
            "Based on the provided contexts, generate a detailed response explaining the relevant information to "
            "answer the question."
        )

        # First turn: Retrieve longer, detailed answer
        first_turn_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": first_turn_prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=500, 
            temperature=0.0
        )

        # Extract the longer generated answer from the first turn
        long_answer = first_turn_completion.choices[0].message.content

def narrativeqa_prompt_and_answer(top_chunks, question):
    try:
        # First Turn: Generate a longer, relevant answer based on the context
        combined_chunks = "\n\n".join([f"Context {i+1}: {chunk['chunk']}" for i, chunk in enumerate(top_chunks)])

        first_turn_prompt = f"Question: {question}\n\n"
        first_turn_prompt += f"{combined_chunks}\n\n"
        first_turn_prompt += (
            "Based on the provided contexts, generate a detailed response explaining the relevant information to "
            "answer the question."
        )

        # First turn: Retrieve longer, detailed answer
        first_turn_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": first_turn_prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=500, 
            temperature=0.0
        )

        # Extract the longer generated answer from the first turn
        long_answer = first_turn_completion.choices[0].message.content

        # Second Turn: Refine the long answer using few-shot examples and include the question again
        few_shot_examples = f"""
        Example 1:
        Question: Who is the protagonist?
        Long Answer: In the novel, Captain Ahab is the one leading the voyage to hunt the great white whale. He is obsessed with the whale, which he names Moby Dick. Ishmael narrates the journey, but it is Ahab who drives the plot with his obsession.
        Concise Answer: Captain Ahab

        Example 2:
        Question: What happens to Frodo at the end of 'The Lord of the Rings'?
        Long Answer: Frodo returns to the Shire after destroying the One Ring but feels out of place in his old life. Eventually, Frodo leaves Middle-earth with Gandalf and the Elves to find peace across the sea.
        Concise Answer: Frodo leaves Middle-earth with Gandalf and the Elves

        Example 3:
        Question: What is Neo's role in 'The Matrix'?
        Long Answer: Neo, played by Keanu Reeves, discovers he is "The One" who can manipulate the Matrix. He leads the fight against the machines that control humanity. The Oracle informs Neo of his potential to bring about the end of the war between humans and machines.
        Concise Answer: Neo is "The One" who leads the fight against the machines

        Example 4:
        Question: How many siblings does Katniss Everdeen have in 'The Hunger Games'?
        Long Answer: Katniss takes care of her younger sister, Primrose, after their father's death. She is extremely protective of her sister, Prim.
        Concise Answer: 1

        Example 5:
        Question: Who is Harry Potter's best friend?
        Long Answer: Throughout the series, Harry's best friend is Ron Weasley. They meet during their first year at Hogwarts and share many adventures together. Ron is always by Harry's side, and their bond strengthens over time.
        Concise Answer: Ron Weasley

        Example 6:
        Question: What is the name of the ship in 'Star Trek'?
        Long Answer: The main ship in the Star Trek series is the USS Enterprise. It is a starship that explores space, led by Captain Kirk and his crew. The Enterprise is well-known for its mission to explore new worlds.
        Concise Answer: USS Enterprise
        """

        second_turn_prompt = f"Question: {question}\n\n"
        second_turn_prompt += f"Long Answer: {long_answer}\n\n"
        second_turn_prompt += "Below are a few examples that show how to generate concise answers based on detailed long answers. Use these examples to help guide your final response."
        second_turn_prompt += f"{few_shot_examples}\n\n"
        second_turn_prompt += (
            "Please review the long answer and generate a concise and accurate final answer that is relevant to the question. "
            "Answer in a single sentence or two."
        )

        # Second turn: Refine the answer into the final response
        second_turn_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": second_turn_prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=50,
            temperature=0.1,
        )

        # Extract the final refined answer from the second turn
        final_output = second_turn_completion.choices[0].message.content

        # Calculate cost estimation
        total_tokens_first = first_turn_completion.usage.total_tokens
        total_tokens_second = second_turn_completion.usage.total_tokens
        
        prompt_tokens_first = first_turn_completion.usage.prompt_tokens
        prompt_tokens_second = second_turn_completion.usage.prompt_tokens
        
        completion_tokens_first = first_turn_completion.usage.completion_tokens
        completion_tokens_second = second_turn_completion.usage.completion_tokens

        cost_per_1M_prompt_tokens = 0.150  # $ per 1M input tokens
        cost_per_1M_completion_tokens = 0.600  # $ per 1M output tokens

        # Calculating costs for each turn
        prompt_cost_first = (prompt_tokens_first / 1_000_000) * cost_per_1M_prompt_tokens
        completion_cost_first = (completion_tokens_first / 1_000_000) * cost_per_1M_completion_tokens
        
        prompt_cost_second = (prompt_tokens_second / 1_000_000) * cost_per_1M_prompt_tokens
        completion_cost_second = (completion_tokens_second / 1_000_000) * cost_per_1M_completion_tokens
        
        # Total estimated cost
        estimated_cost = prompt_cost_first + completion_cost_first + prompt_cost_second + completion_cost_second

        logging.info(f"First Turn Prompt: {first_turn_prompt}")
        logging.info(f"Second Turn Prompt: {second_turn_prompt}")
        logging.info(f"Estimated cost: ${estimated_cost:.6f} total")

        return final_output, estimated_cost

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return -1

def quality_prompt_and_answer(top_chunks, question, answer_choices):
    try:
        # First Turn: Generate longer, relevant information from context
        combined_chunks = "\n\n".join([f"Context {i+1}: {chunk['chunk']}" for i, chunk in enumerate(top_chunks)])
        
        first_turn_prompt = f"Question: {question}\n\n"
        first_turn_prompt += f"{combined_chunks}\n\n"
        first_turn_prompt += "Based on the contexts provided, please generate a detailed response explaining the relevant information to answer the question."

        # First turn: Generate long relevant answer
        first_turn_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": first_turn_prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=500,  # Increase this limit based on how much detail you need
            temperature=0.0
        )

        # Extract the longer generated answer from the first turn
        long_answer = first_turn_completion.choices[0].message.content

        # Second Turn: Refine the long answer to the final answer choice
        second_turn_prompt = f"Based on the following long answer:\n\n{long_answer}\n\n"
        second_turn_prompt += "Please review the answer and select the most appropriate choice from the following:\n"
        for i, choice in enumerate(answer_choices):
            second_turn_prompt += f"{i+1}. {choice}\n"
        
        second_turn_prompt += (
            "\nPlease respond with **only** the number corresponding to the correct answer choice (1, 2, 3, or 4), "
            "with no additional text."
        )
        
        # Second turn: Refine the answer to match the choice
        second_turn_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": second_turn_prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0.0
        )

        # Extract the response from the second turn
        final_output = second_turn_completion.choices[0].message.content
        match = re.search(r'\b[1-4]\b', final_output)  # Match numbers 1 to 4

        if match:
            result = int(match.group(0))
        else:
            logging.warning(f"Invalid output received: {final_output}. Unable to extract a valid number.")
            result = 1  # Handle the case with a fallback, as needed
        
        prompt_tokens_first = first_turn_completion.usage.prompt_tokens
        prompt_tokens_second = second_turn_completion.usage.prompt_tokens
        
        completion_tokens_first = first_turn_completion.usage.completion_tokens
        completion_tokens_second = second_turn_completion.usage.completion_tokens

        cost_per_1M_prompt_tokens = 0.150  # $ per 1M input tokens
        cost_per_1M_completion_tokens = 0.600  # $ per 1M output tokens

        # Calculating costs for each turn
        prompt_cost_first = (prompt_tokens_first / 1_000_000) * cost_per_1M_prompt_tokens
        completion_cost_first = (completion_tokens_first / 1_000_000) * cost_per_1M_completion_tokens
        
        prompt_cost_second = (prompt_tokens_second / 1_000_000) * cost_per_1M_prompt_tokens
        completion_cost_second = (completion_tokens_second / 1_000_000) * cost_per_1M_completion_tokens
        
        # Total estimated cost
        estimated_cost = prompt_cost_first + completion_cost_first + prompt_cost_second + completion_cost_second
        
        logging.info(f"First Turn Prompt: {first_turn_prompt}")
        logging.info(f"Second Turn Prompt: {second_turn_prompt}")
        logging.info(f"Estimated cost: ${estimated_cost:.6f} total")

        return result, estimated_cost

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return -1

# -----------------------------------MAIN FUNTIONS-----------------------------------
def qasper_testing(chunk_type='256'):
    return

def narrativeqa_testing(chunk_type='256'):
    index, document_store = load_faiss_index_and_document_store(json_file_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json', faiss_index_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.index')
    original_documents = load_json_folder(folder_path=f'data/{args.dataset}/individual_documents')
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    # To accumulate scores
    total_rouge = 0
    total_bleu_1 = 0
    total_bleu_4 = 0
    num_qa = 0

    # Track costs
    total_cost = 0
    for doc in original_documents:
        logging.info(f"Processing document: {doc['title']}")
        for qas in doc['qas']:
            question = qas['question']
            golden_answers = qas['answers']
            top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, args.top_k)
            chatbot_answer, estimated_cost = narrativeqa_prompt_and_answer(top_chunks, question)
            total_cost += estimated_cost
            print(golden_answers)
            print(chatbot_answer)
            rouge_result = rouge_metric.compute(predictions=[chatbot_answer], references=[golden_answers])
            
            predictions = [chatbot_answer]  # Pass raw strings, not tokenized
            references = [golden_answers]   # Pass raw reference strings

            # Compute BLEU
            bleu_result = bleu_metric.compute(
                predictions=predictions, 
                references=references
            )
            total_rouge += rouge_result['rougeL']

            total_bleu_1 += bleu_result['precisions'][0]  
            bleu_4 = bleu_smoothing(bleu_result['bleu'], bleu_result)
            total_bleu_4 += bleu_4  

            num_qa += 1
            print(f"Metrics generated: ROUGE-L F1 Score: {rouge_result['rougeL']:.4f} | BLEU-1: {bleu_result['precisions'][0]:.4f} | BLEU-4: {bleu_4:.4f}")
            logging.info(f"Metrics generated: ROUGE-L F1 Score: {rouge_result['rougeL']:.4f} | BLEU-1: {bleu_result['precisions'][0]:.4f} | BLEU-4: {bleu_4:.4f}")
            logging.info(f"Processed Q: {question} | Chatbot Answer: {chatbot_answer} | Golden Answers: {golden_answers}")

    # Calculate the average scores
    avg_rouge = total_rouge / num_qa if num_qa > 0 else 0
    avg_bleu_1 = total_bleu_1 / num_qa if num_qa > 0 else 0
    avg_bleu_4 = total_bleu_4 / num_qa if num_qa > 0 else 0

    # Log the final results
    print(f"For chunking type {chunk_type}:")  # Output the accuracy
    print(f"Average ROUGE-L: {avg_rouge}")
    print(f"Average BLEU-1: {avg_bleu_1}")
    print(f"Average BLEU-4: {avg_bleu_4}")
    print(f"Total Cost: ${total_cost:.6f}")
    logging.info(f"Average ROUGE-L: {avg_rouge}")
    logging.info(f"Average BLEU-1: {avg_bleu_1}")
    logging.info(f"Average BLEU-4: {avg_bleu_4}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    return

# Multiple choice, so accuracy is prefer here
def quality_testing(chunk_type='256'):
    # embedding_document = load_data(json_file_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json')
    index, document_store = load_faiss_index_and_document_store(json_file_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json', faiss_index_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.index')
    original_documents = load_json_folder(folder_path=f'data/{args.dataset}/individual_documents')
    accuracy = 0
    ground_truth_answers = []
    chatbot_predictions = []
    total_cost = 0
    for doc in original_documents:
        logging.info(f"Processing document: {doc['title']}")
        for qas in doc['qas']:
            question = qas['question']
            answer_choices = qas['context']
            golden_answer = qas['answers']
            top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, args.top_k)
            chatbot_answer, estimated_cost = quality_prompt_and_answer(top_chunks, question, answer_choices)
            chatbot_predictions.append(chatbot_answer)
            ground_truth_answers.append(golden_answer)
            total_cost += estimated_cost
            logging.info(f"Question: {question} witth chatbot answer: {chatbot_answer} and golden answer: {golden_answer}")

        # Chatbot predictions (e.g., choices picked by the chatbot)
    chatbot_predictions = np.array(chatbot_predictions)  # Predicted answer indices for each question
    logging.info(f"Chatbot predictions: {chatbot_predictions}")
    # Ground truth answers (correct answer indices for each question)
    ground_truth_answers = np.array(ground_truth_answers)  # Correct answers from the dataset
    logging.info(f"Ground truth answers: {ground_truth_answers}")
    # Calculate accuracy (percentage of correct answers)
    accuracy = (chatbot_predictions == ground_truth_answers).mean()

    print(f"Accuracy: {accuracy:.4f} for chunking type {chunk_type}")  # Output the accuracy
    logging.info(f"Accuracy: {accuracy:.4f} for chunking type {chunk_type} which takes total cost of ${total_cost:.6f}")
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
    args = parser.parse_args() 
    main(args)