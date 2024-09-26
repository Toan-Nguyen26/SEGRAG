import json
import os
from .preprocess_utils import clean_text, count_sentences, preprocess_text, sanitize_filename
from .document_model import DocumentEntry, QAEntry


def create_concatenated_documents_quality_json(output_dir='data/quality/individual_documents', num_files=10):
    # Path to the .dev file
    file_path = 'QuALITY.v1.0.1.htmlstripped.dev'

    # Initialize variables
    unique_titles = set()
    document_id = 1  # Start document IDs from 1
    total_qas_count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content by newlines to handle multiple JSON objects
    json_objects = content.split('\n')
    
    data_list = []
    for json_object in json_objects:
        if json_object.strip():  # Skip empty lines
            try:
                json_object = clean_text(json_object)
                data = json.loads(json_object)
                # Extract the title, article, and questions
                title = data.get('title', 'No Title')
                # Check if we've reached the maximum number of unique titles
                if len(unique_titles) >= num_files:
                    break

                # Process the document only if its title is not already in the set
                if title not in unique_titles:
                    content = clean_text(preprocess_text(data.get('article', 'No Article')))
                    questions = data.get('questions', [])
                    document_entry = DocumentEntry(id=document_id, title=title, content=content, num_sentences=count_sentences(content))
                    # Add the title to the set of unique titles
                    unique_titles.add(title)
                    document_id += 1
                    for item in questions:
                        qa_pair = QAEntry(
                            question=item['question'],
                            context=item['options'],
                            answers=item['gold_label']
                        )
                        document_entry.qas.append(qa_pair)
                        total_qas_count += 1

                    # Create a valid filename using the ID and title
                    os.makedirs(output_dir, exist_ok=True)
                    filename = sanitize_filename(f"{document_entry.id}.json")
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save the document entry to a JSON file
                    with open(filepath, 'w', encoding='utf-8') as json_file:
                        json.dump(document_entry.to_dict(), json_file, ensure_ascii=False, indent=4)

                    print(f"Saved document '{filename}' with {len(document_entry.qas)} Q&A pairs.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    
    # Process the data_list as needed
    print(data_list)
