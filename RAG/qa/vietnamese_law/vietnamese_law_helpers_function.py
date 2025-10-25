import logging
import re


def vietnamese_law_prompt_and_answer(top_chunks, question, client):
    """
    Handle Vietnamese multiple choice legal questions
    Uses 2-turn approach like quality_prompt_and_answer
    
    The question format already contains choices:
    "Vi phạm hành chính là?\nA. hành vi có lỗi...\nB. ...\nC. ...\nD. ..."
    """
    try:
        # First Turn: Generate longer, relevant information from context
        combined_chunks = "\n\n".join([f"Ngữ cảnh {i+1}: {chunk['chunk']}" for i, chunk in enumerate(top_chunks)])
        
        first_turn_prompt = f"Câu hỏi: {question}\n\n"
        first_turn_prompt += f"{combined_chunks}\n\n"
        first_turn_prompt += "Dựa trên các ngữ cảnh được cung cấp, hãy tạo ra một phản hồi chi tiết giải thích thông tin liên quan để trả lời câu hỏi."

        # First turn: Generate long relevant answer
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

        # Second Turn: Refine the long answer to select the correct choice
        # Extract the choices from the question
        # Question format: "Question text?\nA. choice1\nB. choice2\nC. choice3\nD. choice4"
        second_turn_prompt = f"Dựa trên câu trả lời chi tiết sau:\n\n{long_answer}\n\n"
        second_turn_prompt += f"Và câu hỏi gốc:\n{question}\n\n"
        second_turn_prompt += (
            "Vui lòng xem xét câu trả lời và chọn đáp án đúng nhất. "
            "Chỉ trả lời bằng **một chữ cái** (A, B, C, hoặc D), không có văn bản bổ sung."
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
        final_output = second_turn_completion.choices[0].message.content.strip()
        
        # Extract the letter (A, B, C, or D)
        match = re.search(r'\b([A-D])\b', final_output.upper())
        
        if match:
            result = match.group(1)
        else:
            logging.warning(f"Invalid output received: {final_output}. Unable to extract a valid letter.")
            result = "A"  # Fallback to A if parsing fails
        
        # Calculate costs
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
        logging.info(f"Final Answer: {result}")
        logging.info(f"Estimated cost: ${estimated_cost:.6f} total")

        return result, estimated_cost

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return "A", 0  # Return default answer and 0 cost on error