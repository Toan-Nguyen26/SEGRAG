from argparse import ArgumentParser
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from cluster.cluster_helper_functions import group_chunks_by_doc_and_chunk_id, write_json_to_file, determine_average_threshold, determine_percentile_threshold, hard_code_graph, bron_kerbosch, bron_kerbosch_with_pivot
import logging
import faiss
import spacy

logging.basicConfig(filename='segment_clustering.log', level=logging.INFO, format='%(message)s')
# =============================MAIN FUNCTIONS=============================== #
model = SentenceTransformer("BAAI/bge-m3", cache_folder='/path/to/local/cache')
final_embeddings = []
# Initialize a list to hold the merged results
merged_results = []

def create_relatedness_graph(embeddings, threshold):
    graph = {i + 1: set() for i in range(len(embeddings))}
    similarity_matrix = cosine_similarity(embeddings)
    threshold = determine_percentile_threshold(similarity_matrix)
    # print(threshold)
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > threshold:
                graph[i + 1].add(j + 1)
                graph[j + 1].add(i + 1)
    print(f"The graph is {graph}")
    return graph, set(graph.keys())


def create_relatedness_graph_soft(embeddings, k=1.0):
    # Initialize graph
    graph = {i + 1: set() for i in range(len(embeddings))}
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Flatten similarity matrix (exclude diagonal) to calculate mean and std
    upper_triangle_values = similarity_matrix[np.triu_indices(len(embeddings), k=1)]
    
    # Calculate mean and standard deviation
    mean_similarity = np.mean(upper_triangle_values)
    std_similarity = np.std(upper_triangle_values)
    
    # Set the initial threshold based on mean + k * std
    initial_threshold = mean_similarity + k * std_similarity
    print(f"Initial threshold: {initial_threshold}")
    
    # Function to calculate the edge density
    def calculate_edge_density(graph, num_nodes):
        num_edges = sum(len(neighbors) for neighbors in graph.values()) / 2  # Since the graph is undirected
        max_edges = num_nodes * (num_nodes - 1) / 2
        return num_edges / max_edges
    
    # Step 1: Construct graph using the initial threshold
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > initial_threshold:
                graph[i + 1].add(j + 1)
                graph[j + 1].add(i + 1)
    
    # Step 2: Calculate initial edge density
    current_density = calculate_edge_density(graph, len(embeddings))
    print(f"Initial edge density: {current_density}")
    
    # Step 3: Apply edge density "softening"
    # Soften the threshold based on edge density (reduce threshold by a factor based on density)
    softened_threshold = initial_threshold * (1 - current_density)
    print(f"Softened threshold: {softened_threshold}")
    
    # Reset graph and use the softened threshold
    graph = {i + 1: set() for i in range(len(embeddings))}
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > softened_threshold:
                graph[i + 1].add(j + 1)
                graph[j + 1].add(i + 1)
    
    print(f"Final graph is {graph}")
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

# def merge_single_sentence_segments(new_SG, cliques):
#     final_SG = []
#     i = 0

#     while i < len(new_SG):
#         sgi = new_SG[i]
#         nlp = spacy.load("en_core_web_sm")
#         doc = nlp(text)
#         sentences = [sent.text for sent in doc.sents]  # Extract sentences from spaCy
#         # merged = False
#         # if i < len(new_SG) - 1:
#         #     sgi1 = new_SG[i + 1]
#         #     for clique in cliques:
#         #         # Check if there's an intersection between the two segments and the clique
#         #         if any(item in clique for item in sgi) and any(item in clique for item in sgi1):
#         #             # Merge the two segments by combining and sorting them, avoiding duplicates
#         #             merged_segment = sorted(set(sgi + sgi1))
#         #             final_SG.append(merged_segment)  # Add the merged segment
#         #             merged = True
#         #             i += 2  # Skip the next segment as it has been merged
#         #             break
        
#         # if not merged:
#         #     final_SG.append(sgi) 
#         #     i += 1

#     print(f"Final segments: {final_SG}")
#     return final_SG


# Main function to process JSON input and run Bron-Kerbosch
def process_json_and_bron_kerbosch_with_text(grouped_data, embeddings, threshold=0.5):
    # Run the existing Bron-Kerbosch process
    relatedness_graph, all_segments = create_relatedness_graph(embeddings, threshold)
    maximal_cliques = find_maximal_cliques_with_pivot(relatedness_graph)
    initial_segments = create_initial_segments(maximal_cliques, all_segments)
    merged_segments = merge_segments(initial_segments, maximal_cliques)
    
    # Create the new JSON structure with concatenated text and total length
    new_json = []
    return
    
def process_json_with_merged_segments(grouped_data, loaded_data, list_embeddings, max_chunk_size, threshold=0.5):
    for doc_id, doc_data in grouped_data.items():
        embeddings = [doc_data[chunk]['embedding'] for chunk in doc_data]
        first_chunk_key = next(iter(doc_data))
        title = doc_data[first_chunk_key]["title"]
        id = doc_data[first_chunk_key]["doc_id"]
        print("\n")
        print(f"Embedding document {doc_id} with {len(embeddings)} chunks")
        # If the document has only one chunk, no need to calculate similarities
        if len(embeddings) == 1:
            # Treat this single chunk as its own segment
            # chunk_id = list(doc_data.keys())[0]  # Get the single chunk's ID            
            # merged_results.append({
            #     'chunk_id': doc_data[chunk_id]['chunk_id'],
            #     'doc_id': id,
            #     'title': title,
            #     'chunk': doc_data[chunk_id]['chunk'],
            #     'chunk_size': doc_data[chunk_id]['chunk_size'],
            #     "embedding": doc_data[chunk_id]['embedding'].tolist()
            # })
            continue  # Skip to the next document

        # Step 1: Create relatedness graph based on embeddings and threshold
        relatedness_graph, all_segments = create_relatedness_graph(embeddings, threshold)

        # Step 2: Apply Bron-Kerbosch to find maximal cliques
        maximal_cliques = find_maximal_cliques_with_pivot(relatedness_graph)

        # Step 3: Create initial segments based on maximal cliques
        initial_segments = create_initial_segments(maximal_cliques, all_segments)

        # Step 4: Merge segments into bigger segments
        merged_segments = merge_segments(initial_segments, maximal_cliques)

        # new_chunk_id = len(embeddings)+1  
        new_chunk_id = 1

        total_embeddings = []
        total_data = []
        added_time = 0
        for segment in merged_segments:
            # print(segment)
            # if len(segment) == 1:
            #     continue 

            current_chunk = []  # To hold concatenated text chunks
            current_embedding_list = []  # To hold embeddings for averaging
            total_length = 0  # Initialize total length
            
            for chunk_id in segment:
                chunk_text = doc_data[chunk_id]['chunk']
                chunk_length = doc_data[chunk_id]['chunk_size']
                chunk_embedding = doc_data[chunk_id]['embedding']
                
                # If adding this chunk exceeds the 3000 token limit, save the current segment and start a new one
                # if total_length + chunk_length > max_chunk_size:
                #     if chunk_text:  # Make sure there's data to save
                #         # Save the current segment
                #         concatenated_chunk = " ".join(current_chunk)
                #         embedding = model.encode(concatenated_chunk)
                #         # embedding = np.mean(current_embedding_list, axis=0)

                #         loaded_data.append({
                #             'chunk_id': new_chunk_id,  # Use the first chunk ID of the segment
                #             'doc_id': id,
                #             'title': title,
                #             'chunk': concatenated_chunk,
                #             'chunk_size': total_length,
                #             "embedding": embedding.tolist()
                #         })
                #         new_chunk_id += 1
                #         list_embeddings.append(embedding)
                    
                #     # Reset for the next segment
                #     current_chunk = []
                #     current_embedding_list = []
                #     total_length = 0
                #     added_time = 0


                # Add the chunk to the current segment
                current_chunk.append(chunk_text)
                current_embedding_list.append(chunk_embedding)
                total_length += chunk_length
                # added_time += 1
            # Add the last segment if it hasn't been added yet
            if current_chunk:
                concatenated_chunk = " ".join(current_chunk)
                embedding = model.encode(concatenated_chunk)
                # embedding = np.mean(current_embedding_list, axis=0)
                total_data.append({
                    'chunk_id': new_chunk_id,  # Use the first chunk ID of the segment
                    'doc_id': id,
                    'title': title,
                    'chunk': concatenated_chunk,
                    'chunk_size': total_length,
                    "embedding": embedding.tolist()
                })
                new_chunk_id += 1
                added_time = 0
                total_embeddings.append(embedding)
    return total_data, total_embeddings # type: ignore

def cluster_segment(loaded_data, embeddings, max_chunk_size):
    grouped_data = group_chunks_by_doc_and_chunk_id(loaded_data)
    return process_json_with_merged_segments(grouped_data, loaded_data,  embeddings, max_chunk_size)

# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whether it is squad, narrative_qa, or quality', required=True, type=str, default="squad")
    parser.add_argument('--max_chunk_size', help='whether it is squad, narrative_qa, or quality', required=True, type=int, default=1024)
    args = parser.parse_args()
    logger = logging.basicConfig(filename=f'{args.dataset}_segment_clustering.log', level=logging.INFO, format='%(message)s')
    # Provide the path to your JSON file
    json_file_path = f'data/{args.dataset}/seg/seg.json'
    index_file_path = f'data/{args.dataset}/seg/seg.index'
    # TEST json path
    # json_file_path = f'data/{args.dataset}/512/512.json'

    output_faiss_path = f'data/{args.dataset}/segclus/segclus.index'
    os.makedirs(os.path.dirname(output_faiss_path), exist_ok=True)
    output_ids_path = f'data/{args.dataset}/segclus/segclus.json'

    os.makedirs(os.path.dirname(output_ids_path), exist_ok=True)
    # Open and load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group chunks by document ID and chunk ID
    grouped_data = group_chunks_by_doc_and_chunk_id(data)
    # Iterate over each document in the grouped data
    # for doc_id, doc_data in grouped_data.items():
    #     # Extract embeddings and process each document with Bron-Kerbosch algorithm
    #     # print(grouped_data.)
    process_json_with_merged_segments(merged_results, grouped_data)

    # Save the updated JSON back to a file
    with open(f'data/{args.dataset}/segclus/segclus.json', 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=4)
    # Convert embeddings to a numpy array
    embeddings = np.array(final_embeddings)

    # Create a FAISS index
    embedding_dim = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    index.add(embeddings)  # Add the embeddings to the index

    faiss.write_index(index, output_faiss_path)
    print(f"{len(final_embeddings)} chunks have been clustered and saved to {output_ids_path}")
    print(f"FAISS index and document chunk information have been saved to {output_faiss_path}")

