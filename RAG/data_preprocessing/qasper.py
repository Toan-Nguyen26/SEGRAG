import json
import os
from datasets import load_dataset
from .preprocess_utils import clean_text, count_sentences, preprocess_text, sanitize_filename
from .document_model import DocumentEntry, QAEntry
import nltk
from tqdm import tqdm

# Ensure NLTK sentence tokenizer is available
nltk.download('punkt', quiet=True)

def get_answer_from_entry(answer_entry):
    if answer_entry['extractive_spans']:  # Check if extractive_spans is not empty
        return answer_entry['extractive_spans']
    elif answer_entry['yes_no'] is not None:  # Check if yes_no is not None
        return answer_entry['yes_no']
    elif answer_entry['free_form_answer']:  # Check if free_form_answer is not empty
        return answer_entry['free_form_answer']
    return None  # Return None if all are empty

def flatten_answer(answer):
    if answer is None:
        return []
    if isinstance(answer, list):
        # Convert each item to string to ensure consistency
        return [str(item) for item in answer]
    # Convert single answer (e.g., boolean or string) to a list with string representation
    return [str(answer)]

def create_concantenated_documents_qasper_json(output_dir='data/qasper', num_files=10, output_file='qasper.jsonl'):
    """
    Process Qasper dataset and save as a JSONL file for text segmentation tasks.
    
    Args:
        output_dir (str): Directory to save the output JSONL file
        num_files (int): Maximum number of unique documents to process
        output_file (str): Name of the output JSONL file
    """
    # Load the Qasper dataset (train split)
    dataset = load_dataset('allenai/qasper', split='train')
    
    # Initialize variables
    unique_titles = set()
    examples = []
    total_qas_count = 0
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for the output JSONL file
    out_file = os.path.join(output_dir, output_file)
    
    # Iterate through the dataset with progress bar
    for item in tqdm(dataset, desc="Processing Qasper documents"):
        document_title = item['title']
        
        # Check if we've reached the maximum number of unique titles
        if len(unique_titles) >= num_files:
            break

        # Process the document only if its title is not already in the set
        if document_title not in unique_titles:
            # Concatenate abstract, figure captions, and full-text paragraphs
            doc_content = item["abstract"]
            for caption in item["figures_and_tables"]["caption"]:
                doc_content += caption
            for paragraph in item["full_text"]["paragraphs"]:
                for content in paragraph:
                    if content.strip():
                        doc_content += content
                doc_content += "\n"  # Add a newline character
            content = clean_text(preprocess_text(doc_content))
            
            # Skip the document if the content length exceeds 100,000 characters
            if len(content) > 100_000:
                print(f"Skipping document '{document_title}' due to content length > 100k characters.")
                continue

            # Tokenize the content into sentences
            try:
                sentences = nltk.sent_tokenize(content)
            except Exception as e:
                print(f"Error tokenizing document '{document_title}': {e}")
                continue

            # Skip documents with no sentences
            if not sentences:
                print(f"Skipping document '{document_title}' due to no sentences.")
                continue

            # Create an empty labels list
            labels = []

            # Create a new document entry
            document_entry = DocumentEntry(
                id=len(unique_titles) + 1,
                title=document_title,
                num_sentences=len(sentences)
            )

            # Add the title to the set of unique titles
            unique_titles.add(document_title)

            # Dictionary to accumulate combined entries by question
            qa_entries = {}
            
            # Iterate over questions and corresponding answers
            for question, answer_data in zip(item['qas']['question'], item['qas']['answers']):
                add_entry = True
                # Check if the question is unanswerable
                for answer in answer_data['answer']:
                    if answer["unanswerable"] == True:
                        add_entry = False
                        break

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
                    flat_answer = flatten_answer(extracted_answer)
                    qa_entries[question].answers.extend(flat_answer)
                    
                    # Append evidence to the context
                    qa_entries[question].context.extend(evidence)

                total_qas_count += 1

            # Convert the qa_entries dictionary to a list of QAEntry objects
            document_entry.qas = list(qa_entries.values())

            # Create example dictionary (inspired by process_wiki_folder)
            example = {
                "file": sanitize_filename(f"{document_entry.id}_{document_title}.json"),
                "sentences": sentences,
                "labels": labels,
                "title": document_entry.title,
                "qas": [qa.to_dict() for qa in document_entry.qas]
            }

            # Add example to the list
            examples.append(json.dumps(example, ensure_ascii=False) + "\n")
            
            print(f"Processed document '{document_title}' with {len(sentences)} sentences and {len(document_entry.qas)} Q&A pairs.")

    # Write examples to the JSONL file
    print(f"Saving {len(examples)} examples to {out_file}")
    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            f.writelines(examples)
    except Exception as e:
        print(f"Error writing to {out_file}: {e}")
        return None

    print(f"Total number of unique documents saved: {len(unique_titles)}")
    print(f"Total number of question-answer pairs (qas): {total_qas_count}")
    
    return out_file