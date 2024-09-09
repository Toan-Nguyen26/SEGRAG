from argparse import ArgumentParser
import json
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
import logging

logging.basicConfig(filename='segment_clustering.log', level=logging.INFO, format='%(message)s')

# =============================HELPER FUNCTIONS=============================== #
# Function to determine the average similarity threshold


def group_chunks_by_doc_and_chunk_id(documents):
    grouped_data = defaultdict(dict)  # No need for 'chunks', just directly add to document ID
    
    for idx, doc in enumerate(documents):
        doc_id = doc['doc_id'] 
        chunk_id = doc['chunk_id']  
        embedding = np.array(doc['embedding'])   
        doc_title = doc['title']
        text = doc['chunk']                      
        chunk_size = doc['chunk_size']         

        # Group by document ID and create chunk directly under doc_id
        grouped_data[doc_title][chunk_id] = {
            'embedding': embedding,
            'text': text,
            'chunk_size': chunk_size
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

# =============================MAIN FUNCTIONS=============================== #

def create_relatedness_graph(embeddings, threshold):
    graph = {i: set() for i in range(len(embeddings))}
    similarity_matrix = cosine_similarity(embeddings)
    threshold = determine_percentile_threshold(similarity_matrix)
    # print(determine_percentile_threshold(similarity_matrix))
    print(threshold)
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
def process_json_and_bron_kerbosch_with_text(grouped_data, embeddings, threshold=0.5):
    # Run the existing Bron-Kerbosch process
    relatedness_graph, all_segments = create_relatedness_graph(embeddings, threshold)
    maximal_cliques = find_maximal_cliques_with_pivot(relatedness_graph)
    initial_segments = create_initial_segments(maximal_cliques, all_segments)
    merged_segments = merge_segments(initial_segments, maximal_cliques)
    
    # Create the new JSON structure with concatenated text and total length
    new_json = []
    
    for segment in merged_segments:
        # Concatenate the text and sum chunk sizes for the merged segment
        concatenated_text = " ".join([grouped_data['texts'][i] for i in segment])
        total_length = sum([grouped_data['chunk_sizes'][i] for i in segment])
        
        # Add the merged segment information to the new JSON structure
        new_json.append({
            'segment_indices': segment,  # Indices of the original chunks in the merged segment
            'concatenated_text': concatenated_text,
            'total_length': total_length
        })
    
    return new_json

def process_json_with_merged_segments(grouped_data, threshold=0.5):
    new_json = []
    
    for doc_id, doc_data in grouped_data.items():
        # Get embeddings from document data
        embeddings = [doc_data[chunk]['embedding'] for chunk in doc_data]
        title = f"Document {doc_id}"  # Use document ID as the title
        print("\n")
        print(f"Embedding document {doc_id} with {len(embeddings)} chunks")
        # If the document has only one chunk, no need to calculate similarities
        if len(embeddings) == 1:
            # Treat this single chunk as its own segment
            chunk_id = list(doc_data.keys())[0]  # Get the single chunk's ID
            concatenated_text = doc_data[chunk_id]['text']
            total_length = doc_data[chunk_id]['chunk_size']
            
            new_json.append({
                'title': title,
                'segment_indices': [chunk_id],
                'concatenated_text': concatenated_text,
                'total_length': total_length
            })
            continue  # Skip to the next document
        # Step 1: Create relatedness graph based on embeddings and threshold
        relatedness_graph, all_segments = create_relatedness_graph(embeddings, threshold)

        # Step 2: Apply Bron-Kerbosch to find maximal cliques
        maximal_cliques = find_maximal_cliques_with_pivot(relatedness_graph)

        # Step 3: Create initial segments based on maximal cliques
        initial_segments = create_initial_segments(maximal_cliques, all_segments)

        # Step 4: Merge segments into bigger segments
        merged_segments = merge_segments(initial_segments, maximal_cliques)

        # Step 5: For each merged segment, concatenate the text and sum the chunk sizes
        for segment in merged_segments:
            # Use .get() method to avoid KeyError in case the key doesn't exist
            concatenated_text = " ".join([doc_data.get(i, {}).get('text', '') for i in segment])
            total_length = sum([doc_data.get(i, {}).get('chunk_size', 0) for i in segment])
            
            # Only add segments that have valid text
            if concatenated_text.strip():  # Ensure there's actual text
                new_json.append({
                    'title': title,
                    'segment_indices': [i for i in segment],  # Use the actual chunk IDs
                    'concatenated_text': concatenated_text,
                    'total_length': total_length
                })
    
    return new_json

# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whether it is squad, narrative_qa, or quality', required=True, type=str, default="squad")
    args = parser.parse_args()

    # Provide the path to your JSON file
    json_file_path = f'data/{args.dataset}/seg/seg.json'

    # Open and load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group chunks by document ID and chunk ID
    grouped_data = group_chunks_by_doc_and_chunk_id(data)
    
    # Initialize a list to hold the merged results
    merged_results = []
    
    # Iterate over each document in the grouped data
    for doc_id, doc_data in grouped_data.items():
        # Extract embeddings and process each document with Bron-Kerbosch algorithm
        merged_json = process_json_with_merged_segments(grouped_data)
        
        # Add merged results to the list
        merged_results.extend(merged_json)

    # Create the directory if it doesn't exist
    output_dir = f'data/{args.dataset}/segclus'
    os.makedirs(output_dir, exist_ok=True)

    # Now proceed to write the file
    with open(f'{output_dir}/segclus.json', 'w', encoding='utf-8') as out_file:
        json.dump(merged_results, out_file, indent=4)

    print(f"Output written to {output_dir}/segclus.json")