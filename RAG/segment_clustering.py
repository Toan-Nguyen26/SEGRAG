import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(filename='segment_clustering.log', level=logging.INFO, format='%(message)s')

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
    logging.info("\nCalling Bron-Kerbosch with:")
    logging.info(f"R (current clique): {R}")
    logging.info(f"P (potential nodes): {P}")
    logging.info(f"X (excluded nodes): {X}")
    
    if len(P) == 0 and len(X) == 0:
        # A maximal clique is found
        cliques.append(R)
        logging.info(f"Maximal clique found: {R}")
    else:
        pivot = max(P.union(X), key=lambda u: len(graph.get(u, set())), default=None)
        logging.info(f"Chosen pivot: {pivot}")
        for v in list(P - graph.get(pivot, set())):
            logging.info(f"\nExploring vertex: {v}")
            logging.info(f"Neighbors of {v}: {graph[v]}")
            bron_kerbosch_with_pivot(R.union([v]), P.intersection(graph[v]), X.intersection(graph[v]), cliques, graph)
            P.remove(v)
            X.add(v) 
            logging.info(f"Updated P: {P}")
            logging.info(f"Updated X: {X}")

def find_maximal_cliques_with_pivot(graph):
    cliques = []
    # Explicitly specify the nodes if needed
    bron_kerbosch_with_pivot(set(), set(graph.keys()), set(), cliques, graph)
    logging.info(f"All maximal cliques: {cliques}")
    return cliques


def create_initial_segments(cliques):
    """
    Step 1: Create initial segments by merging adjacent sentences found in at least one maximal clique.
    """
    SG = []  # Initial set of segments
    segment_map = {}

    for clique in cliques:
        clique = sorted(clique)  # Ensure the cliques are sorted
        for i in range(len(clique) - 1):
            Si, Sj = clique[i], clique[i + 1]
            if Si not in segment_map and Sj not in segment_map:
                # Both are new, create a new segment
                new_segment = {Si, Sj}
                SG.append(new_segment)
                segment_map[Si] = new_segment
                segment_map[Sj] = new_segment
            elif Si in segment_map and Sj not in segment_map:
                # Si is part of a segment, add Sj to it
                segment_map[Si].add(Sj)
                segment_map[Sj] = segment_map[Si]
            elif Si not in segment_map and Sj in segment_map:
                # Sj is part of a segment, add Si to it
                segment_map[Sj].add(Si)
                segment_map[Si] = segment_map[Sj]
            # Else: both Si and Sj are already part of segments, no action needed
    print(f"Initial segments: {SG}")
    return SG


def merge_segments(SG):
    """
    Step 2: Merges adjacent segments based on overlap or adjacency.
    """
    i = 0
    while i < len(SG) - 1:
        sgi = SG[i]
        sgi1 = SG[i + 1]

        # Check if there is any overlap between the current segment and the next
        if len(sgi.intersection(sgi1)) > 0:
            # Merge the two segments
            merged_segment = sgi.union(sgi1)
            SG[i] = merged_segment
            SG.pop(i + 1)  # Remove the next segment that was merged
        else:
            # Move to the next pair of segments
            i += 1
    print(f"Final segments: {SG}")
    return SG
# Shit ain't working
def create_relatedness_graph(embeddings, threshold=0.0):
    graph = {i: set() for i in range(len(embeddings))}
    similarity_matrix = cosine_similarity(embeddings)
    # Print the similarity matrix for debugging
    print("Similarity Matrix:")
    print(similarity_matrix)
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = float(similarity_matrix[i, j])
            print(f"Comparing sentence {i} and {j} with similarity {similarity}")
            print(f"Type of similarity: {type(similarity)}")
            print(f"Type of threshold: {type(threshold)}")
            print(f"Similarity > Threshold: {similarity > threshold}")
            
            # Debugging the threshold comparison
            if similarity > threshold:
                print(f"--> Adding edge between sentence {i} and {j} (similarity: {similarity})")
                graph[i].add(j)
                graph[j].add(i)
            else:
                print(f"--> No edge added between sentence {i} and {j} (similarity: {similarity})")

    print(f"The graph is {graph}")
    return graph


def hard_code_graph(embeddings):
    graph = {
        1: {2, 6, 8, 9},
        2: {1, 6, 4, 7},
        3: {4, 5},
        4: {2, 3, 5, 7},
        5: {3, 4},
        6: {1, 2},
        7: {2, 4},
        8: {1, 9},
        9: {1, 8}
    }
    # graph = {
    #     1: {2, 6, 8, 9},
    #     2: {4, 6, 7},
    #     3: {4, 5},
    #     4: {5, 7},
    #     8: {9}
    # }
    # graph = {
    #     1: {2, 3},
    #     2: {1, 3},
    #     3: {1,2,4},
    #     4: {3},
    #     5: {}
    # }
    
    
    # Ensure all nodes are included in the graph
    # for i in range(len(embeddings)):
    #     if i not in graph:
    #         graph[i] = set()
    
    print(f"The graph is {graph}")
    return graph


# Main function to process JSON input and run Bron-Kerbosch
def process_json_and_bron_kerbosch(json_file_path, threshold=0.75):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documents = json.load(json_file)
    embeddings = [np.array(doc['embedding']) for doc in documents]
    # son = hard_code_graph(embeddings)
    relatedness_graph = create_relatedness_graph(embeddings, threshold)
    # maximal_cliques = find_maximal_cliques_with_pivot(son)
    # initial_segments = create_initial_segments(maximal_cliques)
    # merged_segments = merge_segments(initial_segments)

    return
    # return merged_segments

# Example usage
if __name__ == "__main__":
    # Provide the path to your JSON file
    json_file_path = 'document_chunks.json'

    # Run the Bron-Kerbosch algorithm on the JSON file's embeddings
    merged_segments = process_json_and_bron_kerbosch(json_file_path)
