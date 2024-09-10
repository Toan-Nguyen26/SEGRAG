import json
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(filename='segment_clustering.log', level=logging.INFO, format='%(message)s')

def group_chunks_by_doc_and_chunk_id(documents):
    grouped_data = defaultdict(dict)  # No need for 'chunks', just directly add to document ID
    
    for idx, doc in enumerate(documents):
        doc_id = doc['doc_id'] 
        chunk_id = doc['chunk_id']  
        embedding = np.array(doc['embedding'])   
        title = doc['title']
        chunk = doc['chunk']                      
        chunk_size = doc['chunk_size']         

        # Group by document ID and create chunk directly under doc_id
        grouped_data[title][chunk_id] = {
            'doc_id': doc_id,
            'chunk_id': chunk_id,
            'title': title,
            'chunk': chunk,
            'chunk_size': chunk_size,
            'embedding': embedding
        }
        
    return grouped_data

def write_json_to_file(output_data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Output written to {output_file_path}")

def determine_average_threshold(similarity_matrix, factor=1.0):
    # Calculate the average similarity
    similarities = similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)]
    average_similarity = np.mean(similarities)
    return factor * average_similarity  

def determine_percentile_threshold(similarity_matrix, percentile=90):
    similarities = similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)]
    threshold = np.percentile(similarities, percentile)
    return threshold

def hard_code_graph(embeddings):
    # graph = {
    #     1: {2, 6, 8, 9},
    #     2: {1, 6, 4, 7},
    #     3: {4, 5},
    #     4: {2, 3, 5, 7},
    #     5: {3, 4},
    #     6: {1, 2},
    #     7: {2, 4},
    #     8: {1, 9},
    #     9: {1, 8}
    # }
    graph = {
        1: {2, 3},
        2: {1, 3},
        3: {1,2,4},
        4: {3},
        5: {}
    }
    
    print(f"The graph is {graph}")
    return graph, set(graph.keys())

def bron_kerbosch(R, P, X, cliques, graph):
    logging.info("\nCalling Bron-Kerbosch with:")
    logging.info(f"R (current clique): {R}")
    logging.info(f"P (potential nodes): {P}")
    logging.info(f"X (excluded nodes): {X}")
    
    if len(P) == 0 and len(X) == 0:
        cliques.append(R)
        logging.info(f"Maximal clique found: {R}")
    else:
        while P:
            v = P.pop() 
            logging.info(f"\nExploring vertex: {v}")
            logging.info(f"Neighbors of {v}: {graph[v]}")
            bron_kerbosch(R.union([v]), P.intersection(graph[v]), X.intersection(graph[v]), cliques, graph)
            X.add(v)  
            logging.info(f"Updated P: {P}")
            logging.info(f"Updated X: {X}")

def bron_kerbosch_with_pivot(R, P, X, cliques, graph):
    if len(P) == 0 and len(X) == 0:
        cliques.append(sorted(R))
    else:
        pivot = max(P.union(X), key=lambda u: len(graph.get(u, set())), default=None)
        for v in list(P - graph.get(pivot, set())):
            bron_kerbosch_with_pivot(R.union([v]), P.intersection(graph[v]), X.intersection(graph[v]), cliques, graph)
            P.remove(v)
            X.add(v) 
