#!/usr/bin/env python3
"""
Knowledge Graph Generator for RAG Datasets

This script creates a Knowledge Graph (KG) for a specified dataset (QASPER, Quality, NarrativeQA).
Each document forms an isolated sub-graph with dataset-specific entities and relations.
Output is saved as GraphML files for integration with FAISS-based RAG.

Usage:
    python create_knowledge_graph.py --dataset qasper --input RAG/data_512_1024 --output kgs

Dependencies:
    - networkx for graph creation
    - spacy, underthesea for entity extraction
    - transformers for relation extraction
    - pathlib, argparse, json for data handling
"""

import argparse
import json
from pathlib import Path
import spacy
from transformers import pipeline
import networkx as nx
from tqdm import tqdm

# Dataset-specific relation rules (customize as needed)
RELATION_RULES = {
    "qasper": {
        "patterns": [
            lambda doc: [(s.root.text, "justifies_answer", obj.text) for s in doc.sents for obj in s.ents if "question" in s.text.lower()],
            lambda doc: [(ent1.text, "cites", ent2.text) for s in doc.sents for ent1, ent2 in zip(s.ents, s.ents[1:]) if "cite" in s.text.lower()]
        ],
        "description": "QA-focused: question-answer links, scientific relations"
    },
    "quality": {
        "patterns": [
            lambda doc: [(s.root.text, "subsumes_topic", obj.text) for s in doc.sents for obj in s.noun_chunks if "topic" in s.text.lower()],
            lambda doc: [(ent1.text, "references", ent2.text) for s in doc.sents for ent1, ent2 in zip(s.ents, s.ents[1:])]
        ],
        "description": "General text: topic hierarchies, quality links"
    },
    "narrativeqa": {
        "patterns": [
            lambda doc: [(ent1.text, "interacts_with", ent2.text) for s in doc.sents for ent1, ent2 in zip(s.noun_chunks, s.noun_chunks[1:])],
            lambda doc: [(s.root.text, "precedes_event", obj.text) for s in doc.sents for obj in s.ents if "after" in s.text.lower()]
        ],
        "description": "Narrative: character interactions, event sequences"
    }
}

def load_documents(dataset_name, input_path):
    """
    Load chunked documents from dataset-specific path.

    Args:
        dataset_name (str): One of 'qasper', 'quality', 'narrativeqa'
        input_path (str): Base path to RAG data (e.g., 'RAG/data_512_1024')

    Returns:
        list: List of dicts with doc_id, full_text, sentences, chunks
    """
    # Construct the file path directly using the dataset name
    file_path = Path(input_path) / f"{dataset_name}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file {file_path} not found.")

    docs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Assuming the JSON file is a list of documents or a dictionary
            # For a list of documents, iterate through it
            if isinstance(data, list):
                for i, doc_data in enumerate(data):
                    full_text = " ".join(doc_data.get("sentences", []))
                    if not full_text:
                        print(f"Skipping empty doc: {file_path.stem} (index {i})")
                        continue
                    docs.append({
                        "doc_id": f"{file_path.stem}_{i}",
                        "dataset": dataset_name,
                        "full_text": full_text,
                        "sentences": doc_data.get("sentences", []),
                        "chunks": doc_data.get("segmented_sentences", []),
                        "metadata": doc_data.get("section_topic_labels", [])
                    })
            # If the JSON file is a single document dictionary, handle it
            elif isinstance(data, dict):
                full_text = " ".join(data.get("sentences", []))
                if full_text:
                    docs.append({
                        "doc_id": file_path.stem,
                        "dataset": dataset_name,
                        "full_text": full_text,
                        "sentences": data.get("sentences", []),
                        "chunks": data.get("segmented_sentences", []),
                        "metadata": data.get("section_topic_labels", [])
                    })
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(docs)} documents from {file_path.name}")
    return docs

def extract_entities_relations(doc, dataset_name):
    """
    Extract entities and dataset-specific relations for a document.
    
    Args:
        doc (dict): Document with full_text, sentences, chunks
        dataset_name (str): Dataset name for rule selection
    
    Returns:
        dict: {"entities": [(name, type)], "relations": [(head, rel, tail)]}
    """
    # Initialize NLP tools
    nlp = spacy.load("en_core_web_sm")
    re_pipeline = pipeline("relation-extraction", model="Babelscape/rebel-large", framework="pt")

    # Use spaCy for entity extraction on all documents
    spacy_doc = nlp(doc["full_text"])
    entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]

    # Extract relations: ML + Rules
    relations = re_pipeline(doc["full_text"], return_all_scores=False) or []
    if not isinstance(relations, list):
        relations = []
    
    # Apply dataset-specific rules
    if dataset_name in RELATION_RULES:
        for rule in RELATION_RULES[dataset_name]["patterns"]:
            relations.extend(rule(spacy_doc))
    
    # Attach chunk IDs to relations (link to FAISS)
    relations_with_chunks = []
    for head, rel, tail in relations[:10]:
        chunk_id = next((i for i, chunk in enumerate(doc["chunks"]) if head in doc["sentences"][i] or tail in doc["sentences"][i]), -1)
        relations_with_chunks.append((head, rel, tail, {"chunk_id": chunk_id}))
    
    return {"entities": entities, "relations": relations_with_chunks}

def build_subgraph(entities, relations, doc_id, dataset_name, output_path):
    """
    Build an isolated sub-graph for a document and save as GraphML.
    
    Args:
        entities (list): [(name, type)]
        relations (list): [(head, rel, tail, metadata)]
        doc_id (str): Document identifier
        dataset_name (str): Dataset name
        output_path (str): Base output directory
    
    Returns:
        nx.DiGraph: Sub-graph for the document
    """
    G = nx.DiGraph()
    for ent, ent_type in entities:
        G.add_node(ent, type=ent_type, doc_id=doc_id, dataset=dataset_name)
    
    for head, rel, tail, metadata in relations:
        G.add_edge(head, tail, relation=rel, **metadata)
    
    # Save as GraphML
    kg_path = Path(output_path) / dataset_name
    kg_path.mkdir(parents=True, exist_ok=True)
    graph_file = kg_path / f"{doc_id}.graphml"
    nx.write_graphml(G, graph_file)
    print(f"Saved sub-graph for {doc_id} to {graph_file}")
    return G

def create_knowledge_graph(dataset_name, input_path, output_path):
    """
    Main function to generate KG for a dataset.
    
    Args:
        dataset_name (str): One of 'qasper', 'quality', 'narrativeqa'
        input_path (str): Path to input data
        output_path (str): Path to save KG files
    """
    if dataset_name not in ["qasper", "quality", "narrativeqa"]:
        raise ValueError(f"Invalid dataset: {dataset_name}. Choose 'qasper', 'quality', or 'narrativeqa'.")
    
    print(f"Generating KG for {dataset_name}: {RELATION_RULES[dataset_name]['description']}")
    
    # Load documents
    docs = load_documents(dataset_name, input_path)
    if not docs:
        print(f"No documents found for {dataset_name}")
        return
    
    # Process each document
    for doc in tqdm(docs, desc=f"Processing {dataset_name} documents"):
        try:
            # Extract entities and relations
            kg_data = extract_entities_relations(doc, dataset_name)
            
            # Build and save sub-graph
            build_subgraph(kg_data["entities"], kg_data["relations"], doc["doc_id"], dataset_name, output_path)
        except Exception as e:
            print(f"Error processing doc {doc['doc_id']}: {e}")
            continue
    
    print(f"Completed KG generation for {dataset_name}. Output saved to {output_path}/{dataset_name}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Knowledge Graph for a RAG dataset")
    parser.add_argument("--dataset", required=True, choices=["qasper", "quality", "narrativeqa"], 
                       help="Dataset to process (qasper, quality, narrativeqa)")
    parser.add_argument("--input", default="RAG/data_512_1024", 
                       help="Path to input data directory")
    parser.add_argument("--output", default="kgs", 
                       help="Path to save KG GraphML files")
    
    args = parser.parse_args()
    
    create_knowledge_graph(args.dataset, args.input, args.output)
    print("KG generation complete!")