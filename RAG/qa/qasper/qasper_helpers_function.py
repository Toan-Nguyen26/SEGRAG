import logging

def qasper_prompt_and_answer(top_chunks, question, client):
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
