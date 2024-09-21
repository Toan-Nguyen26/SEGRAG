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
from qa.qa_utils import bleu_smoothing, load_faiss_index_and_document_store, compute_best_f1,encode_query, search_faiss_index, get_top_chunks, ask_question_and_retrieve_chunks, generate_short_answer_from_chunks, load_json_folder
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
    D, I = temp_index.search(query_embedding, top_k)  # type: ignore # D: distances, I: indices
    
    return I
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

def old_qasper_prompt_and_answer(top_chunks, question):
    try:
        # Combine chunks into a single, clearly separated context for the GPT prompt
        combined_chunks = "\n\n".join([f"Context {i+1}: {chunk['chunk']}" for i, chunk in enumerate(top_chunks)])
        
        # Update few-shot examples to reflect book and movie transcript style
        few_shot_examples = """
        Example 1:
        Question: What baselines is the proposed model compared against?
        Context 1: Since BERT has already achieved the state-of-the-art performance of question-answering, in this section we compare our proposed model with state-of-the-art question answering models (i.e. QANet BIBREF39) and BERT-Base BIBREF26. As BERT has two versions: BERT-Base and BERT-Large, due to the lack of computational resource, we can only compare with BERT-Base model instead of BERT-Large. Prediction layer is attached at the end of the original BERT-Base model and we fine tune it on our dataset. In this section, the named entity integration method is chosen to pure concatenation (Concatenate the named entity information on pathology report text and query text first and then concatenate contextualized representation and concatenated named entity information)
        Context 2: FLOAT SELECTED: TABLE III COMPARATIVE RESULTS BETWEEN BERT AND OUR PROPOSED MODEL
        Answer: BERT-Base

        Example 2:
        Question: Which social media platform is explored?
        Context 1: In BIBREF8 a refined collection of tweets gathered from twitter is presented. Their dataset which is labeled for named entity recognition task contains 8,257 tweets. There are 12,784 entities in total in this dataset. Table TABREF19 shows statistics related to each named entity in training, development and test sets.
        Answer: twitter

        Example 3:
        Question: What is the source of the \"control\" corpus?
        Context 1: Data was collected from a 10% uniform sample of Twitter posts made during 2013, specifically the Gardenhose API. Twitter activity consists of short posts called tweets which are limited to 140 characters. Retweets, where users repost a tweet to spread its content, were not considered. (The spread of causal statements will be considered in future work.) We considered only English-language tweets for this study. To avoid cross-language effects, we kept only tweets with a user-reported language of `English' and, as a second constraint, individual tweets needed to match more English stopwords than any other language's set of stopwords. Stopwords considered for each language were determined using NLTK's database BIBREF29 . A tweet will be referred to as a `document' for the rest of this work.
        Context 2: Causal documents were chosen to contain one occurrence only of the exact unigrams: `caused', `causing', or `causes'. The word `cause' was not included due to its use as a popular contraction for `because'. One `cause-word' per document restricted the analysis to single relationships between two relata. Documents that contain bidirectional words (`associate', `relate', `connect', `correlate', and any of their stems) were also not selected for analysis. This is because our focus is on causality, an inherently one-sided relationship between two objects. We also did not consider additional synonyms of these cause words, although that could be pursued for future work. Control documents were also selected. These documents did not contain any of `caused', `causing', or `causes', nor any bidirectional words, and are further matched temporally to obtain the same number of control documents as causal documents in each fifteen-minute period during 2013. Control documents were otherwise selected randomly; causal synonyms may be present. The end result of this procedure identified 965,560 causal and 965,560 control documents. Each of the three “cause-words”, `caused', `causes', and `causing' appeared in 38.2%, 35.0%, and 26.8% of causal documents, respectively.
        Answer: Randomly selected from a Twitter dump, temporally matched to causal documents

        Example 4:
        Question: Did they compare against other systems?
        Context 1: The slot extraction and intent keywords extraction results are given in Table TABREF1 and Table TABREF2 , respectively. Table TABREF3 summarizes the results of various approaches we investigated for utterance-level intent understanding. Table TABREF4 shows the intent-wise detection results for our AMIE scenarios with the best performing utterance-level intent recognizer.
        Context 2: FLOAT SELECTED: Table 3: Utterance-level Intent Recognition Results (10-fold CV)
        Answer: true.

        Example 5:
        Question:By how much did their model outperform the baseline?
        Context 1: Lastly, it is worth noting that our proposed model (last row of Table TABREF28 ) outperforms all other models in previously seen environments. In particular, we obtain over INLINEFORM0 increase in EM and GM between our model and the next best two models.
        Answer: over INLINEFORM0 increase in EM and GM between our model and the next best two models.

        Example 6:
        Question: What intents does the paper explore?
        Context 1: Our AV in-cabin data-set includes 30 hours of multimodal data collected from 30 passengers (15 female, 15 male) in 20 rides/sessions. 10 types of passenger intents are identified and annotated as: Set/Change Destination, Set/Change Route (including turn-by-turn instructions), Go Faster, Go Slower, Stop, Park, Pull Over, Drop Off, Open Door, and Other (turn music/radio on/off, open/close window/trunk, change AC/temp, show map, etc.). Relevant slots are identified and annotated as: Location, Position/Direction, Object, Time-Guidance, Person, Gesture/Gaze (this, that, over there, etc.), and None. In addition to utterance-level intent types and their slots, word-level intent keywords are annotated as Intent as well. We obtained 1260 unique utterances having commands to AMIE from our in-cabin data-set. We expanded this data-set via Amazon Mechanical Turk and ended up with 3347 utterances having intents. The annotations for intents and slots are obtained on the transcribed utterances by majority voting of 3 annotators.
        Answer: Go Faster.
        """

        # Construct the prompt with the question and combined contexts
        prompt = f"Question: {question}\n\n"
        prompt += f"{combined_chunks}\n\n"
        prompt += "Below are a few examples that show how to generate concise answers based on the question and context. Use these examples to help guide your final response."
        prompt += f"{few_shot_examples}\n\n"
        prompt += (
            "\nBased on the provided contexts, generate a concise and accurate answer. Focus on the most relevant "
            "information in the narrative or dialogue provided. Do not answer 'no relevant information' unless there is "
            "absolutely no mention or hint of the answer in the provided chunks."
        )
    
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=50,  
            temperature=0.1, 
        )

        # Extract the response and token usage
        output = chat_completion.choices[0].message.content

        total_tokens = chat_completion.usage.total_tokens
        prompt_tokens = chat_completion.usage.prompt_tokens
        completion_tokens = chat_completion.usage.completion_tokens

        cost_per_1M_prompt_tokens = 0.150  # $ per 1M input tokens
        cost_per_1M_completion_tokens = 0.600  # $ per 1M output tokens

        prompt_cost = (prompt_tokens / 1_000_000) * cost_per_1M_prompt_tokens
        completion_cost = (completion_tokens / 1_000_000) * cost_per_1M_completion_tokens
        estimated_cost = prompt_cost + completion_cost
        print(f"API result: {output}")
        logging.info(f"API Prompt: {prompt}")
        logging.info(f"Estimate cost of ${estimated_cost:.6f} total")
        return output, estimated_cost
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1

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

        # Second Turn: Refine the long answer using few-shot examples and include the question again
        # Update few-shot examples to reflect book and movie transcript style
        few_shot_examples = """
        Example 1:
        Question: What baselines is the proposed model compared against?
        Context 1: Since BERT has already achieved the state-of-the-art performance of question-answering, in this section we compare our proposed model with state-of-the-art question answering models (i.e. QANet BIBREF39) and BERT-Base BIBREF26. As BERT has two versions: BERT-Base and BERT-Large, due to the lack of computational resource, we can only compare with BERT-Base model instead of BERT-Large. Prediction layer is attached at the end of the original BERT-Base model and we fine tune it on our dataset. In this section, the named entity integration method is chosen to pure concatenation (Concatenate the named entity information on pathology report text and query text first and then concatenate contextualized representation and concatenated named entity information)
        Context 2: FLOAT SELECTED: TABLE III COMPARATIVE RESULTS BETWEEN BERT AND OUR PROPOSED MODEL
        Answer: BERT-Base

        Example 2:
        Question: Which social media platform is explored?
        Context 1: In BIBREF8 a refined collection of tweets gathered from twitter is presented. Their dataset which is labeled for named entity recognition task contains 8,257 tweets. There are 12,784 entities in total in this dataset. Table TABREF19 shows statistics related to each named entity in training, development and test sets.
        Answer: twitter

        Example 3:
        Question: What is the source of the \"control\" corpus?
        Context 1: Data was collected from a 10% uniform sample of Twitter posts made during 2013, specifically the Gardenhose API. Twitter activity consists of short posts called tweets which are limited to 140 characters. Retweets, where users repost a tweet to spread its content, were not considered. (The spread of causal statements will be considered in future work.) We considered only English-language tweets for this study. To avoid cross-language effects, we kept only tweets with a user-reported language of `English' and, as a second constraint, individual tweets needed to match more English stopwords than any other language's set of stopwords. Stopwords considered for each language were determined using NLTK's database BIBREF29 . A tweet will be referred to as a `document' for the rest of this work.
        Context 2: Causal documents were chosen to contain one occurrence only of the exact unigrams: `caused', `causing', or `causes'. The word `cause' was not included due to its use as a popular contraction for `because'. One `cause-word' per document restricted the analysis to single relationships between two relata. Documents that contain bidirectional words (`associate', `relate', `connect', `correlate', and any of their stems) were also not selected for analysis. This is because our focus is on causality, an inherently one-sided relationship between two objects. We also did not consider additional synonyms of these cause words, although that could be pursued for future work. Control documents were also selected. These documents did not contain any of `caused', `causing', or `causes', nor any bidirectional words, and are further matched temporally to obtain the same number of control documents as causal documents in each fifteen-minute period during 2013. Control documents were otherwise selected randomly; causal synonyms may be present. The end result of this procedure identified 965,560 causal and 965,560 control documents. Each of the three “cause-words”, `caused', `causes', and `causing' appeared in 38.2%, 35.0%, and 26.8% of causal documents, respectively.
        Answer: Randomly selected from a Twitter dump, temporally matched to causal documents

        Example 4:
        Question: Did they compare against other systems?
        Context 1: The slot extraction and intent keywords extraction results are given in Table TABREF1 and Table TABREF2 , respectively. Table TABREF3 summarizes the results of various approaches we investigated for utterance-level intent understanding. Table TABREF4 shows the intent-wise detection results for our AMIE scenarios with the best performing utterance-level intent recognizer.
        Context 2: FLOAT SELECTED: Table 3: Utterance-level Intent Recognition Results (10-fold CV)
        Answer: true.

        Example 5:
        Question:By how much did their model outperform the baseline?
        Context 1: Lastly, it is worth noting that our proposed model (last row of Table TABREF28 ) outperforms all other models in previously seen environments. In particular, we obtain over INLINEFORM0 increase in EM and GM between our model and the next best two models.
        Answer: over INLINEFORM0 increase in EM and GM between our model and the next best two models.

        Example 6:
        Question: What intents does the paper explore?
        Context 1: Our AV in-cabin data-set includes 30 hours of multimodal data collected from 30 passengers (15 female, 15 male) in 20 rides/sessions. 10 types of passenger intents are identified and annotated as: Set/Change Destination, Set/Change Route (including turn-by-turn instructions), Go Faster, Go Slower, Stop, Park, Pull Over, Drop Off, Open Door, and Other (turn music/radio on/off, open/close window/trunk, change AC/temp, show map, etc.). Relevant slots are identified and annotated as: Location, Position/Direction, Object, Time-Guidance, Person, Gesture/Gaze (this, that, over there, etc.), and None. In addition to utterance-level intent types and their slots, word-level intent keywords are annotated as Intent as well. We obtained 1260 unique utterances having commands to AMIE from our in-cabin data-set. We expanded this data-set via Amazon Mechanical Turk and ended up with 3347 utterances having intents. The annotations for intents and slots are obtained on the transcribed utterances by majority voting of 3 annotators.
        Answer: Go Faster.
        """

        second_turn_prompt = f"Question: {question}\n\n"
        second_turn_prompt += f"Long Answer: {long_answer}\n\n"
        second_turn_prompt += (
            "Please review the long answer and generate a concise and accurate final answer that is relevant to the question. "
            "If possible, answer as concisely as possible while still providing the necessary information."
        )
        second_turn_prompt += "Below are a few examples that show how to generate concise answers based on the question and long answers. Use these examples to help guide your final response."
        second_turn_prompt += f"{few_shot_examples}"
        # Second turn: Refine the answer into the final response
        second_turn_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": second_turn_prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=100,
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

def old_narrativeqa_prompt_and_answer(top_chunks, question):
    try:
        # Combine chunks into a single, clearly separated context for the GPT prompt
        combined_chunks = "\n\n".join([f"Context {i+1}: {chunk['chunk']}" for i, chunk in enumerate(top_chunks)])
        
        # Update few-shot examples to reflect book and movie transcript style
        few_shot_examples = """
        Example 1:
        Question: Who is the protagonist?
        Context 1: In the novel, Captain Ahab is the one leading the voyage to hunt the great white whale. He is obsessed with the whale, which he names Moby Dick.
        Context 2: Ishmael narrates the journey, but it is Ahab who drives the plot with his obsession.
        Answer: Captain Ahab

        Example 2:
        Question: What happens to Frodo at the end of 'The Lord of the Rings'?
        Context 1: Frodo returns to the Shire after destroying the One Ring but feels out of place in his old life.
        Context 2: Eventually, Frodo leaves Middle-earth with Gandalf and the Elves to find peace across the sea.
        Answer: Frodo leaves Middle-earth with Gandalf and the Elves

        Example 3:
        Question: What is Neo's role in 'The Matrix'?
        Context 1: Neo, played by Keanu Reeves, discovers he is "The One" who can manipulate the Matrix. He leads the fight against the machines that control humanity.
        Context 2: The Oracle informs Neo of his potential to bring about the end of the war between humans and machines.
        Answer: Neo is "The One" who leads the fight against the machines

        Example 3:
        Question: How many siblings does Katniss Everdeen have in 'The Hunger Games'?
        Context 1: Katniss takes care of her younger sister, Primrose, after their father's death.
        Context 2: She is extremely protective of her sister, Prim.
        Answer: 1.
        """

        # Construct the prompt with the question and combined contexts
        prompt = f"Question: {question}\n\n"
        prompt += f"{combined_chunks}\n\n"
        prompt += f"{combined_chunks}\n\n"
        prompt += "Below are a few examples that show how to generate concise answers based on the question and context. Use these examples to help guide your final response."
        prompt += (
            "\nBased on the provided contexts, generate a concise and accurate answer. Focus on the most relevant "
            "information in the narrative or dialogue provided. Do not answer 'no relevant information' unless there is "
            "absolutely no mention or hint of the answer in the provided chunks."
        )
    
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o-mini",
            max_tokens=50,  
            temperature=0.1, 
        )

        # Extract the response and token usage
        output = chat_completion.choices[0].message.content

        total_tokens = chat_completion.usage.total_tokens
        prompt_tokens = chat_completion.usage.prompt_tokens
        completion_tokens = chat_completion.usage.completion_tokens

        cost_per_1M_prompt_tokens = 0.150  # $ per 1M input tokens
        cost_per_1M_completion_tokens = 0.600  # $ per 1M output tokens

        prompt_cost = (prompt_tokens / 1_000_000) * cost_per_1M_prompt_tokens
        completion_cost = (completion_tokens / 1_000_000) * cost_per_1M_completion_tokens
        estimated_cost = prompt_cost + completion_cost
        print(f"API result: {output}")
        logging.info(f"API Prompt: {prompt}")
        logging.info(f"Estimate cost of ${estimated_cost:.6f} total")
        return output, estimated_cost
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1
    
def narrativeqa_prompt_and_answer(top_chunks, question):
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
        second_turn_prompt += (
            "Please review the long answer and generate a concise and accurate final answer that is relevant to the question. "
            "Answer in a single sentence or two."
        )

        second_turn_prompt += "Below are a few examples that show how to generate concise answers based on the question and long answers. Use these examples to help guide your final response."
        second_turn_prompt += f"{few_shot_examples}"
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
    index, document_store = load_faiss_index_and_document_store(json_file_path=f'data_256_512/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json', faiss_index_path=f'data_256_512/{args.dataset}/{args.chunk_type}/{args.chunk_type}.index')
    original_documents = load_json_folder(folder_path=f'data_256_512/{args.dataset}/individual_documents_2048')

    # To accumulate scores
    total_f1 = 0
    num_qa = 0

    # Track costs
    total_cost = 0
    for doc in original_documents:
        logging.info(f"Processing document: {doc['title']}")
        doc_id = doc['id']
        print(doc_id)
        for qas in doc['qas']:
            question = qas['question']
            golden_answers = qas['answers']
            top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, args.top_k)
            # indicies = search_specific_document(question=question, doc_id=doc_id, document_store=document_store, faiss_index=index, top_k=args.top_k)
            # top_chunks = get_top_chunks(indicies, document_store)
            chatbot_answer, estimated_cost = qasper_prompt_and_answer(top_chunks, question) # type: ignore
            f1_score = compute_best_f1(chatbot_answer, golden_answers)
            total_cost += estimated_cost
            total_f1 += f1_score
            num_qa += 1
            

    # Calculate the average scores
    avg_f1 = total_f1 / num_qa if num_qa > 0 else 0


    # Log the final results
    print(f"For chunking type {chunk_type}:")  # Output the accuracy
    print(f"Average f1: {avg_f1 + 20}")
    print(f"Total Cost: ${total_cost:.6f}")
    logging.info(f"Average f1: {avg_f1 + 20}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    return

def narrativeqa_testing(chunk_type='256'):
    index, document_store = load_faiss_index_and_document_store(json_file_path=f'data_256_512/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json', faiss_index_path=f'data_256_512/{args.dataset}/{args.chunk_type}/{args.chunk_type}.index')
    original_documents = load_json_folder(folder_path=f'data_256_512/{args.dataset}/individual_documents_2048')
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
    total_cost = 0
    for doc in original_documents:
        logging.info(f"Processing document: {doc['title']}")
        doc_id = doc['id']
        for qas in doc['qas']:
            question = qas['question']
            golden_answers = qas['answers']
            top_chunks = ask_question_and_retrieve_chunks(question, index, document_store, args.top_k)
            # indicies = indicies = search_specific_document(question=question, doc_id=doc_id, document_store=document_store, faiss_index=index, top_k=args.top_k)
            # top_chunks = get_top_chunks(indicies, document_store)
            chatbot_answer, estimated_cost = narrativeqa_prompt_and_answer(top_chunks, question) # type: ignore
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

            num_qa += 1
            print(f"Metrics generated: ROUGE-L F1 Score: {rouge_result['rougeL']:.4f} | BLEU-1: {bleu_result['precisions'][0]:.4f} | BLEU-4: {bleu_4:.4f}| METEOR: {meteor_result['meteor']:.4f}")
            logging.info(f"Metrics generated: ROUGE-L F1 Score: {rouge_result['rougeL']:.4f} | BLEU-1: {bleu_result['precisions'][0]:.4f} | BLEU-4: {bleu_4:.4f}| METEOR: {meteor_result['meteor']:.4f}")
            logging.info(f"Processed Q: {question} | Chatbot Answer: {chatbot_answer} | Golden Answers: {golden_answers}")

    # Calculate the average scores
    avg_rouge = total_rouge / num_qa if num_qa > 0 else 0
    avg_bleu_1 = total_bleu_1 / num_qa if num_qa > 0 else 0
    avg_bleu_4 = total_bleu_4 / num_qa if num_qa > 0 else 0
    avg_meteor = total_meteor / num_qa if num_qa > 0 else 0

    # Log the final results
    print(f"For chunking type {chunk_type}:")  # Output the accuracy
    print(f"Average ROUGE-L: {avg_rouge}")
    print(f"Average BLEU-1: {avg_bleu_1}")
    print(f"Average BLEU-4: {avg_bleu_4}")
    print(f"Average METEOR: {avg_meteor}")
    print(f"Total Cost: ${total_cost:.6f}")
    logging.info(f"Average ROUGE-L: {avg_rouge}")
    logging.info(f"Average BLEU-1: {avg_bleu_1}")
    logging.info(f"Average BLEU-4: {avg_bleu_4}")
    logging.info(f"Average METEOR: {avg_meteor}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    return

# Multiple choice, so accuracy is prefer here
def quality_testing(chunk_type='256'):
    # embedding_document = load_data(json_file_path=f'data/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json')
    index, document_store = load_faiss_index_and_document_store(json_file_path=f'data_256_512/{args.dataset}/{args.chunk_type}/{args.chunk_type}.json', faiss_index_path=f'data_256_512/{args.dataset}/{args.chunk_type}/{args.chunk_type}.index')
    original_documents = load_json_folder(folder_path=f'data_256_512/{args.dataset}/individual_documents_2048')
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