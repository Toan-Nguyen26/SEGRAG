import json
import os
from urllib.parse import unquote
from datasets import load_dataset
from .preprocess_utils import clean_text, count_sentences, preprocess_text, sanitize_filename
from .document_model import DocumentEntry, QAEntry

def create_concatenated_documents_squad_json(output_dir='data/squad/individual_documents', num_files=10):
    # Load the SQuAD dataset (SQuAD v1.1 in this case)
    dataset = load_dataset('squad', split='train')

    # Initialize variables
    documents = []
    unique_titles = set()
    document_id = 1  # Start document IDs from 1
    title_qas_count = {}
    total_qas_count = 0
    
    # Iterate over the dataset to find the first `num_files` unique titles and concatenate their unique contexts
    for example in dataset:
        # Ensure that the example is a dictionary and contains the 'title' and 'context' keys
        if isinstance(example, dict) and 'title' in example and 'context' in example:
            title = unquote(example['title'])
            content = example['context']
            if title not in unique_titles:
                unique_titles.add(title)
                document_entry = DocumentEntry(id=document_id, title=title)
                documents.append(document_entry)
                document_id += 1  # Increment document ID
                title_qas_count[title] = 0  # Initialize Q&A count for this title

            # Find the document entry with the matching title
            for doc in documents:
                if doc.title == title:
                    # Add context if it's not already in the content list
                    if content not in doc.content:
                        doc.content.append(content)
                    qas_entry = QAEntry(
                        question=example['question'],
                        context=content,
                        answers=example['answers']['text']
                    )
                    doc.qas.append(qas_entry)
                    title_qas_count[title] += 1  # Increment Q&A count for this title
                    total_qas_count += 1  # Increment total Q&A count
            
            # Stop when we have `num_files` unique titles
            if len(unique_titles) >= num_files:
                break

    # Concatenate the contexts for each document, count sentences, and save each as a separate JSON file
    os.makedirs(output_dir, exist_ok=True)
    
    for doc in documents:
        doc.content = "\n".join(doc.content)
        doc.content = preprocess_text(doc.content)
        doc.num_sentences = count_sentences(doc.content)
        
        # Save each document as a separate JSON file with the ID as the filename
        output_file_path = os.path.join(output_dir, f"{doc.id}.json")
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(doc.to_dict(), json_file, ensure_ascii=False, indent=4)
        
        print(f"Document ID {doc.id} saved to {output_file_path}")

    # Print the number of Q&As per title and the total Q&A count
    for title, count in title_qas_count.items():
        print(f"Title: {title}, Q&A Count: {count}")
    print(f"Total Q&A Count: {total_qas_count}")

