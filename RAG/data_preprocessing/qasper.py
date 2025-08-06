import json
import os
import pprint
from datasets import load_dataset
from .preprocess_utils import clean_text, count_sentences, preprocess_text, sanitize_filename
from .document_model import DocumentEntry, QAEntry

def get_answer_from_entry(answer_entry):
    if answer_entry['extractive_spans']:  # Check if extractive_spans is not empty
        return answer_entry['extractive_spans']
    elif answer_entry['yes_no'] is not None:  # Check if yes_no is not None
        return answer_entry['yes_no']
    elif answer_entry['free_form_answer']:  # Check if free_form_answer is not empty
        return answer_entry['free_form_answer']
    return None  # Return None if all are empty

def flatten_answer(answer):
    if isinstance(answer, list):
        # If the answer is a list, flatten it by concatenating all elements into one list
        return [item for sublist in answer for item in (sublist if isinstance(sublist, list) else [sublist])]
    return [answer]

def create_concantenated_documents_qasper_json(output_dir='data/qasper/individual_documents', num_files=10):
    # Load the NarrativeQA dataset (train split)
    dataset = load_dataset('allenai/qasper', split='train')

    # for item in dataset:
    #     sample = item
    #     break
    # # Write the sample to a JSON file
    # with open('qasper_sample.json', 'w') as f:
    #     json.dump(sample, f, indent=4)

    # Initialize variables
    unique_titles = set()
    document_id = 1  # Start document IDs from 1
    total_qas_count = 0
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for item in dataset:
        document_title = item['title']
        
        # Check if we've reached the maximum number of unique titles
        if len(unique_titles) >= num_files:
            break


        if document_title not in unique_titles:
            # first, we need to concatenate all the paragraphs in the document
            doc_content = item["abstract"]
            for caption in item["figures_and_tables"]["caption"]:
                doc_content += caption
            for paragraph in item["full_text"]["paragraphs"]:
                for content in paragraph:
                    if content.strip():
                       doc_content += content  
                doc_content += "\n"  # Add a newline character
            content = clean_text(preprocess_text(doc_content))

            # Create a new document entry
            document_entry = DocumentEntry(id=document_id, title=document_title, content=content, num_sentences=count_sentences(content))

            # Add the title to the set of unique titles
            unique_titles.add(document_title)
            document_id += 1
            # Dictionary to accumulate combined entries by question
            qa_entries = {}

           # Iterate over questions and corresponding answers
            for question, answer_data in zip(item['qas']['question'], item['qas']['answers']):
                add_entry = True
                # Initialize or retrieve the combined QA entry for this question
                for answer in answer_data['answer']:
                    if answer["unanswerable"] == True:
                        # If the question is unanswerable, skip adding QAEntry for this question
                        add_entry = False
                        break  # No need to check further answers if unanswerable is found

                if not add_entry:
                    continue

                if question not in qa_entries:
                    qa_entries[question] = QAEntry(
                        question=question,
                        context=[],  # Initialize context as an empty list to accumulate evidence
                        answers=[]   # Initialize answers as an empty list to accumulate answers
                    )
                
                # Iterate over the list of answers for the current question
                for answer in answer_data['answer']:
                    evidence = answer['evidence']  # Get the evidence for this answer
                    extracted_answer = get_answer_from_entry(answer)  # Get the correct answer
                    
                    # Flatten the extracted answer and append to the answers list
                    flat_answer = flatten_answer(extracted_answer)  # Ensure the answer is not a list of lists
                    qa_entries[question].answers.extend(flat_answer)  # Add flattened answers to the existing list
                    
                    # Append evidence to the context
                    qa_entries[question].context.extend(evidence)  # Add the evidence (context)

                # Increment the total_qas counter
                total_qas_count += 1

            # Convert the qa_entries dictionary to a list of QAEntry objects
            document_entry.qas = list(qa_entries.values())

            # Create a valid filename using the ID and title
            os.makedirs(output_dir, exist_ok=True)
            # Create a valid filename using the ID and title
            filename = sanitize_filename(f"{document_entry.id}.json")
            filepath = os.path.join(output_dir, filename)
            
            # Save the document entry to a JSON file
            with open(filepath, 'w', encoding='utf-8') as json_file:
                json.dump(document_entry.to_dict(), json_file, ensure_ascii=False, indent=4)

            print(f"Saved document '{filename}' with {len(document_entry.qas)} Q&A pairs.")

    print(f"Total number of unique documents saved: {len(unique_titles)}")
    print(f"Total number of question-answer pairs (qas): {total_qas_count}")