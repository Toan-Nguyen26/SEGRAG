import logging

def narrativeqa_prompt_and_answer(top_chunks, question, client):
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
