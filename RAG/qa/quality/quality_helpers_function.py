import logging
import re


def quality_prompt_and_answer(top_chunks, question, answer_choices, client):
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
