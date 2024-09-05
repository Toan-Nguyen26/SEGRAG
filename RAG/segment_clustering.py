import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import logging

logging.basicConfig(filename='segment_clustering.log', level=logging.INFO, format='%(message)s')

# =============================HELPER FUNCTIONS=============================== #
# Function to determine the average similarity threshold
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

# =============================MAIN FUNCTIONS=============================== #

def create_relatedness_graph(embeddings, threshold):
    graph = {i: set() for i in range(len(embeddings))}
    similarity_matrix = cosine_similarity(embeddings)
    # print(determine_percentile_threshold(similarity_matrix))
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > threshold:
                graph[i].add(j)
                graph[j].add(i)
    print(f"The graph is {graph}")
    return graph, set(graph.keys())

def find_maximal_cliques_with_pivot(graph):
    cliques = []
    bron_kerbosch_with_pivot(set(), set(graph.keys()), set(), cliques, graph)
    sorted_cliques= sorted(cliques, key=lambda x: x[0])
    logging.info(f"All maximal cliques: {sorted_cliques}")
    print(f"All maximal cliques: {sorted_cliques}")
    return sorted_cliques

def create_initial_segments(cliques, all_segments):
    SG = []  # Initial set of segments
    segment_map = {}
    used_segments = set()
    
    for clique in cliques:
        clique = sorted(clique)
        for i in range(len(clique) - 1):
            if clique[i + 1] - clique[i] == 1:
                Si, Sj = clique[i], clique[i + 1]
                if Si not in segment_map and Sj not in segment_map:
                    new_segment = {Si, Sj}
                    SG.append(new_segment)
                    segment_map[Si] = new_segment
                    segment_map[Sj] = new_segment
                    used_segments.update({Si, Sj})
                elif Si in segment_map and Sj not in segment_map:
                    segment_map[Si].add(Sj)
                    segment_map[Sj] = segment_map[Si]
                    used_segments.add(Sj)
                elif Si not in segment_map and Sj in segment_map:
                    segment_map[Sj].add(Si)
                    segment_map[Si] = segment_map[Sj]
                    used_segments.add(Si)
    
    for segment in all_segments:
        if segment not in used_segments:
            SG.append({segment})

    # Sort each segment internally for display purposes
    sorted_segments = [sorted(segment) for segment in SG]
    # Sort the list of segments by the first element of each segment
    sorted_segments = sorted(sorted_segments, key=lambda x: x[0])
    print(f"Initial segments: {sorted_segments}")
    return sorted_segments

def merge_segments(SG, cliques):
    new_SG = []
    i = 0

    while i < len(SG):
        sgi = SG[i]
        merged = False
        if i < len(SG) - 1:
            sgi1 = SG[i + 1]
            for clique in cliques:
                # Check if there's an intersection between the two segments and the clique
                if any(item in clique for item in sgi) and any(item in clique for item in sgi1):
                    # Merge the two segments by combining and sorting them, avoiding duplicates
                    merged_segment = sorted(set(sgi + sgi1))
                    new_SG.append(merged_segment)  # Add the merged segment
                    merged = True
                    i += 2  # Skip the next segment as it has been merged
                    break
        
        if not merged:
            new_SG.append(sgi) 
            i += 1

    print(f"Final segments: {new_SG}")
    return new_SG

# Main function to process JSON input and run Bron-Kerbosch
def process_json_and_bron_kerbosch(json_file_path, threshold=0.5):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documents = json.load(json_file)
    embeddings = [np.array(doc['embedding']) for doc in documents]
    # relatedness_graph, all_segments = hard_code_graph(embeddings)
    relatedness_graph, all_segments = create_relatedness_graph(embeddings, threshold)
    maximal_cliques = find_maximal_cliques_with_pivot(relatedness_graph)
    initial_segments = create_initial_segments(maximal_cliques, all_segments)
    merged_segments = merge_segments(initial_segments, maximal_cliques)
    return merged_segments

# Example usage
if __name__ == "__main__":
    # Provide the path to your JSON file
    json_file_path = 'document_chunks.json'

    # Run the Bron-Kerbosch algorithm on the JSON file's embeddings
    merged_segments = process_json_and_bron_kerbosch(json_file_path)
