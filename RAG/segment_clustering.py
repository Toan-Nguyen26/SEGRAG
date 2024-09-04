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
        # R = sorted(R)
        cliques.append(R)
        logging.info(f"Maximal clique found: {R}")
    else:
        # Choose a pivot vertex from P union X to reduce the number of recursive calls
        pivot = max(P.union(X), key=lambda u: len(graph.get(u, set())), default=None)
        logging.info(f"Chosen pivot: {pivot}")
        # Explore only the vertices in P that are not neighbors of the pivot
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
    print(f"All maximal cliques: {cliques}")
    return cliques


def create_initial_segments(cliques, all_segments):
    """
    Step 1: Create initial segments by merging adjacent sentences found in at least one maximal clique.
    """
    SG = []  # Initial set of segments
    segment_map = {}
    used_segments = set()
    for clique in cliques:
        clique = sorted(clique)
        for i in range(len(clique) -1):
            if clique[i+1] - clique[i] == 1:
                Si, Sj = clique[i], clique[i + 1]
                if Si not in segment_map and Sj not in segment_map:
                    new_segment = {Si, Sj}
                    SG.append(new_segment)
                    segment_map[Si] = new_segment
                    segment_map[Sj] = new_segment
                    used_segments.update({clique[i], clique[i+1]})
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
    print(f"Initial segments: {SG}")
    return SG

def sort_segments(SG):
    # Sort each segment individually and convert back to sets
    sorted_segments = [set(sorted(segment)) for segment in SG]
    
    # Sort the list of segments by the first element of each segment (converted to a list for sorting)
    sorted_segments = sorted(sorted_segments, key=lambda x: sorted(x)[0])
    
    return sorted_segments

#This needs a lot of work
def merge_segments(SG, cliques):
    new_SG = []
    i = 0

    while i < len(SG):
        sgi = SG[i]
        merged = False
        if i < len(SG) - 1:
            sgi1 = SG[i + 1]
            for clique in cliques:
                if sgi.intersection(clique) and sgi1.intersection(clique):
                    # Merge the two segments
                    merged_segment = sgi.union(sgi1)
                    new_SG.append(merged_segment)  # Add the merged segment
                    merged = True
                    i += 2  # Skip the next segment as it has been merged
                    break
        
        if not merged:
            new_SG.append(sgi) 
            i += 1

    print(f"Final segments: {new_SG}")
    return new_SG

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
    
    
    # Ensure all nodes are included in the graph
    # for i in range(len(embeddings)):
    #     if i not in graph:
    #         graph[i] = set()
    
    print(f"The graph is {graph}")
    return graph, set(graph.keys())


# Main function to process JSON input and run Bron-Kerbosch
def process_json_and_bron_kerbosch(json_file_path, threshold=0.75):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documents = json.load(json_file)
    embeddings = [np.array(doc['embedding']) for doc in documents]
    son, all_segments = hard_code_graph(embeddings)
    # relatedness_graph = create_relatedness_graph(embeddings, threshold)
    maximal_cliques = find_maximal_cliques_with_pivot(son)
    initial_segments = create_initial_segments(maximal_cliques, all_segments)
    sorted_initial_segments = sort_segments(initial_segments)
    merged_segments = merge_segments(sorted_initial_segments, maximal_cliques)
    return merged_segments

# Example usage
if __name__ == "__main__":
    # Provide the path to your JSON file
    json_file_path = 'document_chunks.json'

    # Run the Bron-Kerbosch algorithm on the JSON file's embeddings
    merged_segments = process_json_and_bron_kerbosch(json_file_path)
