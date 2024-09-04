import os
import json
import numpy as np
import faiss
import re
import spacy
from datasets import load_dataset
from bs4 import BeautifulSoup
from argparse import ArgumentParser
from urllib.parse import unquote 
from document_model import DocumentEntry, QAEntry
# Load the spaCy model

def count_sentences(text):
    # Load the Spacy model
    nlp = spacy.load("en_core_web_sm")  
    doc = nlp(text)
    
    return len(list(doc.sents))

def sanitize_filename(filename):
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
     # Handle escaped quotes and other escape sequences properly
    text = text.encode('utf-8').decode('unicode_escape')
    # Remove newline characters and unnecessary spaces
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with a space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

def clean_text(text):
    # Define the unwanted characters
    unwanted_prefix = "\u00c3\u00af\u00c2\u00bb\u00c2\u00bf"
    if text.startswith(unwanted_prefix):
        return text[len(unwanted_prefix):]
    return text

def create_concatenated_documents_squad_json(output_dir='data/squad/individual_documents', max_titles=10):
    # Load the SQuAD dataset (SQuAD v1.1 in this case)
    dataset = load_dataset('squad', split='train')

    # Initialize variables
    documents = []
    unique_titles = set()
    document_id = 1  # Start document IDs from 1
    title_qas_count = {}
    total_qas_count = 0
    
    # Iterate over the dataset to find the first `max_titles` unique titles and concatenate their unique contexts
    for example in dataset:
        # Ensure that the example is a dictionary and contains the 'title' and 'context' keys
        if isinstance(example, dict) and 'title' in example and 'context' in example:
            title = unquote(example['title'])
            content = example['context']
            if title not in unique_titles:
                unique_titles.add(title)
                document_entry = DocumentEntry(id=document_id, title=title)
                # document_entry = {
                #     'id': document_id,
                #     'title': title,
                #     'qas': [],
                #     'content': [],
                #     "num_sentences": 0,
                #     "segmented_sentences": []
                # }
                documents.append(document_entry)
                document_id += 1  # Increment document ID
                title_qas_count[title] = 0  # Initialize Q&A count for this title

            # Find the document entry with the matching title
            for doc in documents:
                if doc['title'] == title:
                    # Add context if it's not already in the content list
                    if content not in doc['content']:
                        doc.content.append(content)
                    qas_entry = QAEntry(
                        question=example['question'],
                        context=content,
                        answers=example['answers']['text']
                    )
                    # qas_entry = {
                    #     'question': (example['question']),
                    #     'context': content,
                    #     'answers': example['answers']['text']
                    # }
                    # doc['qas'].append(qas_entry)
                    doc.qas.append(qas_entry)
                    title_qas_count[title] += 1  # Increment Q&A count for this title
                    total_qas_count += 1  # Increment total Q&A count
            
            # Stop when we have `max_titles` unique titles
            if len(unique_titles) >= max_titles:
                break

    # Concatenate the contexts for each document, count sentences, and save each as a separate JSON file
    os.makedirs(output_dir, exist_ok=True)
    
    for doc in documents:
        doc['content'] = "\n".join(doc['content'])
        doc['content'] = preprocess_text(doc['content'])
        doc['num_sentences'] = count_sentences(doc['content'])
        
        # Save each document as a separate JSON file with the ID as the filename
        output_file_path = os.path.join(output_dir, f"{doc['id']}.json")
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(doc, json_file, ensure_ascii=False, indent=4)
        
        print(f"Document ID {doc['id']} saved to {output_file_path}")

    # Print the number of Q&As per title and the total Q&A count
    for title, count in title_qas_count.items():
        print(f"Title: {title}, Q&A Count: {count}")
    print(f"Total Q&A Count: {total_qas_count}")


def create_individual_documents_narrativeqa_json(output_dir='data/narrativeqa/individual_documents', max_titles=10):
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
        if len(unique_titles) >= max_titles:
            break

        # Process the document only if its title is not already in the set
        if document_title not in unique_titles:
            content = clean_text(preprocess_text(item['document']['text']))
            
            # Skip the document if the content length exceeds 150,000 characters
            if len(content) > 150_000:
                print(f"Skipping document '{document_title}' due to content length > 150k characters.")
                continue

            # Create a new document entry
            document_entry = {
                "id": document_id,
                "title": document_title,
                "qas": [],
                "content": content,
                "num_sentences": count_sentences(content),
                "segmented_sentences": []
            }

            # Add the title to the set of unique titles
            unique_titles.add(document_title)
            document_id += 1

            # Create a question-answer pair
            qa_pair = {
                "question": item['question']['text'],
                "context": "",
                "answers": [answer['text'] for answer in item['answers']]
            }

            # Add the question-answer pair to the document entry
            document_entry["qas"].append(qa_pair)

            # Increment the total_qas counter
            total_qas_count += 1

            # Create a valid filename using the ID and title
            filename = sanitize_filename(f"{document_entry['id']}.json")
            print(filename)
            filepath = os.path.join(output_dir, filename)
            
            # Save the document entry to a JSON file
            with open(filepath, 'w', encoding='utf-8') as json_file:
                json.dump(document_entry, json_file, ensure_ascii=False, indent=4)

            print(f"Saved document '{filename}' with {len(document_entry['qas'])} Q&A pairs.")

    print(f"Total number of unique documents saved: {len(unique_titles)}")
    print(f"Total number of question-answer pairs (qas): {total_qas_count}")

def main(args):
    if args.dataset == 'squad':
        create_concatenated_documents_squad_json()
    elif args.dataset == 'narrative_qa':
        create_individual_documents_narrativeqa_json()
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please choose 'squad' or 'narrative_qa'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whenever it is squad or narrative_qa',  required=True, type=str, default="squad")
    # By default the command to run is:
    # python document_concat.py --dataset squad

    main(parser.parse_args())