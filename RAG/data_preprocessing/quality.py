import json
import os
from .preprocess_utils import clean_text, count_sentences, preprocess_text, sanitize_filename
from .document_model import DocumentEntry, QAEntry
import nltk
from tqdm import tqdm

# Ensure NLTK sentence tokenizer is available
nltk.download('punkt', quiet=True)

def create_concatenated_documents_quality_json(output_dir='data/quality', num_files=10, output_file='quality.jsonl', input_file='QuALITY.v1.0.1.htmlstripped.dev'):
    """
    Process QuALITY dataset and save as a JSONL file for text segmentation tasks.
    
    Args:
        output_dir (str): Directory to save the output JSONL file
        num_files (int): Maximum number of unique documents to process
        output_file (str): Name of the output JSONL file
        input_file (str): Path to the input QuALITY .dev file
    """
    # Initialize variables
    unique_titles = set()
    examples = []
    total_qas_count = 0
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for the output JSONL file
    out_file = os.path.join(output_dir, output_file)
    
    # Read the QuALITY .dev file
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading input file '{input_file}': {e}")
        return None
    
    # Split the content by newlines to handle multiple JSON objects
    json_objects = content.split('\n')
    
    # Iterate through JSON objects with progress bar
    for json_object in tqdm(json_objects, desc="Processing QuALITY documents"):
        if json_object.strip():  # Skip empty lines
            try:
                json_object = clean_text(json_object)
                data = json.loads(json_object)
                
                # Extract the title and article
                title = data.get('title', 'No Title')
                
                # Check if we've reached the maximum number of unique titles
                if len(unique_titles) >= num_files:
                    break

                # Process the document only if its title is not already in the set
                if title not in unique_titles:
                    content = clean_text(preprocess_text(data.get('article', 'No Article')))
                    
                    # Skip the document if the content length exceeds 100,000 characters
                    if len(content) > 100_000:
                        print(f"Skipping document '{title}' due to content length > 100k characters.")
                        continue

                    # Tokenize the content into sentences
                    try:
                        sentences = nltk.sent_tokenize(content)
                    except Exception as e:
                        print(f"Error tokenizing document '{title}': {e}")
                        continue

                    # Skip documents with no sentences
                    if not sentences:
                        print(f"Skipping document '{title}' due to no sentences.")
                        continue

                    # Create an empty labels list
                    labels = []

                    # Create a new document entry
                    document_entry = DocumentEntry(
                        id=len(unique_titles) + 1,
                        title=title,
                        num_sentences=len(sentences)
                    )

                    # Add the title to the set of unique titles
                    unique_titles.add(title)

                    # Process questions
                    questions = data.get('questions', [])
                    for item in questions:
                        qa_pair = QAEntry(
                            question=item['question'],
                            context=item['options'],
                            answers=item['gold_label']
                        )
                        document_entry.qas.append(qa_pair)
                        total_qas_count += 1

                    # Create example dictionary (inspired by process_wiki_folder)
                    example = {
                        "file": sanitize_filename(f"{document_entry.id}_{title}.json"),
                        "sentences": sentences,
                        "labels": labels,
                        "title": document_entry.title,
                        "qas": [qa.to_dict() for qa in document_entry.qas]
                    }

                    # Add example to the list
                    examples.append(json.dumps(example, ensure_ascii=False) + "\n")
                    
                    print(f"Processed document '{title}' with {len(sentences)} sentences and {len(document_entry.qas)} Q&A pairs.")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

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