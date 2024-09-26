import json
import os
from datasets import load_dataset
from .preprocess_utils import clean_text, count_sentences, preprocess_text, sanitize_filename
from .document_model import DocumentEntry, QAEntry

def create_individual_documents_narrativeqa_json(output_dir='data/narrativeqa/individual_documents', num_files=50):
    # Load the NarrativeQA dataset (train split)
    dataset = load_dataset('deepmind/narrativeqa', split='train')
    
    # Initialize variables
    unique_titles = set()
    document_id = 1  # Start document IDs from 1
    total_qas_count = 0
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through the dataset
    for item in dataset:
        document_title = item['document']['summary']['title']
        
        # Check if we've reached the maximum number of unique titles
        if len(unique_titles) >= num_files:
            break

        # Process the document only if its title is not already in the set
        if document_title not in unique_titles:
            content = clean_text(preprocess_text(item['document']['text']))
            
            # Skip the document if the content length exceeds 150,000 characters
            if len(content) > 100_000:
                print(f"Skipping document '{document_title}' due to content length > 50k characters.")
                continue

            # Create a new document entry
            document_entry = DocumentEntry(id=document_id, title=document_title, content=content, num_sentences=count_sentences(content))

            # Add the title to the set of unique titles
            unique_titles.add(document_title)
            document_id += 1

            # Create a question-answer pair
            qa_pair = QAEntry(
                question=item['question']['text'],
                context="",
                answers=[answer['text'] for answer in item['answers']]
            )

            document_entry.qas.append(qa_pair)
            # Increment the total_qas counter
            total_qas_count += 1

            # Create a valid filename using the ID and title
            filename = sanitize_filename(f"{document_entry.id}.json")
            filepath = os.path.join(output_dir, filename)
            
            # Save the document entry to a JSON file
            with open(filepath, 'w', encoding='utf-8') as json_file:
                json.dump(document_entry.to_dict(), json_file, ensure_ascii=False, indent=4)

            print(f"Saved document '{filename}' with {len(document_entry.qas)} Q&A pairs.")

    print(f"Total number of unique documents saved: {len(unique_titles)}")
    print(f"Total number of question-answer pairs (qas): {total_qas_count}")

