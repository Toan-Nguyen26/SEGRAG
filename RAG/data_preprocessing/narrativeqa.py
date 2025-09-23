import json
import os
from datasets import load_dataset
from .preprocess_utils import clean_text, count_sentences, preprocess_text, sanitize_filename
from .document_model import DocumentEntry, QAEntry
import nltk
from tqdm import tqdm

# def create_individual_documents_narrativeqa_json(output_dir='data/narrativeqa/individual_documents', num_files=50):
#     # Load the NarrativeQA dataset (train split)
#     dataset = load_dataset('deepmind/narrativeqa', split='train')
    
#     # Initialize variables
#     unique_titles = set()
#     document_id = 1  # Start document IDs from 1
#     total_qas_count = 0
    
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Iterate through the dataset
#     for item in dataset:
#         document_title = item['document']['summary']['title']
        
#         # Check if we've reached the maximum number of unique titles
#         if len(unique_titles) >= num_files:
#             break

#         # Process the document only if its title is not already in the set
#         if document_title not in unique_titles:
#             content = clean_text(preprocess_text(item['document']['text']))
#             print(len(content))
#             # Skip the document if the content length exceeds 150,000 characters
#             if len(content) > 100_000:
#                 # print(f"Skipping document '{document_title}' due to content length > 50k characters.")
#                 continue

#             # Create a new document entry
#             document_entry = DocumentEntry(id=document_id, title=document_title, content=content, num_sentences=count_sentences(content))

#             # Add the title to the set of unique titles
#             unique_titles.add(document_title)
#             document_id += 1

#             # Create a question-answer pair
#             qa_pair = QAEntry(
#                 question=item['question']['text'],
#                 context="",
#                 answers=[answer['text'] for answer in item['answers']]
#             )

#             document_entry.qas.append(qa_pair)
#             # Increment the total_qas counter
#             total_qas_count += 1

#             # Create a valid filename using the ID and title
#             filename = sanitize_filename(f"{document_entry.id}.json")
#             filepath = os.path.join(output_dir, filename)
            
#             # Save the document entry to a JSON file
#             with open(filepath, 'w', encoding='utf-8') as json_file:
#                 json.dump(document_entry.to_dict(), json_file, ensure_ascii=False, indent=4)

#             print(f"Saved document '{filename}' with {len(document_entry.qas)} Q&A pairs.")

#     print(f"Total number of unique documents saved: {len(unique_titles)}")
#     print(f"Total number of question-answer pairs (qas): {total_qas_count}")

# Ensure NLTK sentence tokenizer is available
nltk.download('punkt', quiet=True)

def create_individual_documents_narrativeqa_json(output_dir='data/narrativeqa', num_files=50, output_file='narrativeqa.jsonl'):
    """
    Process NarrativeQA dataset and save as a JSONL file for text segmentation tasks.
    
    Args:
        output_dir (str): Directory to save the output JSONL file
        num_files (int): Maximum number of unique documents to process
        output_file (str): Name of the output JSONL file
    """
    # Load the NarrativeQA dataset (train split)
    dataset = load_dataset('deepmind/narrativeqa', split='train')
    
    # Initialize variables
    unique_titles = set()
    examples = []
    total_qas_count = 0
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for the output JSONL file
    out_file = os.path.join(output_dir, output_file)
    
    # Iterate through the dataset with progress bar
    for item in tqdm(dataset, desc="Processing NarrativeQA documents"):
        document_title = item['document']['summary']['title']
        
        # Check if we've reached the maximum number of unique titles
        if len(unique_titles) >= num_files:
            break

        # Process the document only if its title is not already in the set
        if document_title not in unique_titles:
            content = clean_text(preprocess_text(item['document']['text']))
            
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
                # content=content,  # Keep original content for reference
                num_sentences=len(sentences)
            )

            # Add the title to the set of unique titles
            unique_titles.add(document_title)

            # Create a question-answer pair
            qa_pair = QAEntry(
                question=item['question']['text'],
                context="",
                answers=[answer['text'] for answer in item['answers']]
            )

            document_entry.qas.append(qa_pair)
            total_qas_count += 1

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