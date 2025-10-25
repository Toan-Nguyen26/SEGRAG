import json
import os
import uuid
import argparse
from openai import OpenAI
from tqdm import tqdm
from neo4j import GraphDatabase
import Levenshtein
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import uuid
from typing import List, Dict, Any, Tuple, Set

# Define the models that match your desired JSON structure

class Entity(BaseModel):
    # UUID is generated in your downstream Python logic, but the model expects a string
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="A unique UUID for the entity.")
    name: str = Field(description="The canonical name of the entity.")
    type: str = Field(description="One of the allowed Entity Types (e.g., 'Character/Person', 'Location/Place').")

class Relation(BaseModel):
    head: str = Field(description="The UUID of the head entity.")
    relation: str = Field(description="One of the allowed Relation Types (e.g., 'INTERACTS_WITH', 'CAUSES').")
    tail: str = Field(description="The UUID of the tail entity.")

class KnowledgeGraphExtraction(BaseModel):
    """The complete required output structure."""
    entities: List[Entity] = Field(default_factory=list, description="A list of extracted entities.")
    relations: List[Relation] = Field(default_factory=list, description="A list of extracted relations.")

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
if not client.api_key:
    # This check is good practice but might be silenced if the environment handles the key
    print("WARNING: OPENAI_API_KEY not set in environment variables.")

# Dataset-specific entity and relationship types
DATASET_CONFIG = {
    'narrativeqa': {
        'entity_types': ['Character/Person', 'Location/Place', 'Event', 'Object/Item', 'Group/Organization', 'Theme/Concept', 'Time/Period'],
        'rel_types': ['INTERACTS_WITH', 'IS_FRIEND_OF', 'IS_ENEMY_OF', 'IS_FAMILY_OF', 'CAUSES', 'RESULTS_IN', 'OCCURS_IN', 'SET_IN', 'OWNS', 'USES', 'FOLLOWS', 'PRECEDES', 'REPRESENTS', 'SYMBOLIZES', 'INVOLVES']
    },
    'quality': {
        'entity_types': ['Person/Character', 'Location/Place', 'Event/Action', 'Concept/Idea', 'Organization/Entity', 'Object/Artifact', 'Time/Period'],
        'rel_types': ['CAUSES', 'RESULTS_IN', 'DISCUSSES', 'EXPLAINS', 'INVOLVES', 'CONCERNS', 'IS_ABOUT', 'DEFINES', 'BELIEVES_IN', 'FEARS', 'OCCURS_DURING', 'SET_IN', 'USES_AS_EXAMPLE', 'CONTRASTS_WITH', 'SIMILAR_TO']
    },
    'qasper': {
        'entity_types': ['Author/Person', 'Paper/Document', 'Method/Model', 'Dataset', 'Task', 'Metric', 'Concept/Term', 'Organization/Institution'],
        'rel_types': ['AUTHORS', 'PROPOSED_BY', 'USES', 'BUILDS_ON', 'EVALUATES_ON', 'TESTS_ON', 'ACHIEVES', 'REPORTS', 'ADDRESSES', 'SOLVES', 'CITES', 'REFERENCES', 'COMPARES_TO', 'OUTPERFORMS', 'REQUIRES', 'DEPENDS_ON']
    }
}

def extract_entities_relations(chunk_text, dataset, model="gpt-4o-mini", max_retries=3):
    """
    Extracts entities and relations from a text chunk using a specific dataset
    configuration and enforces strict JSON output using the OpenAI API.
    """
    config = DATASET_CONFIG.get(dataset, {
        'entity_types': ['Person', 'Location', 'Organization', 'Event', 'Concept'],
        'rel_types': ['RELATED_TO', 'CAUSES', 'LOCATED_IN']
    })
    entity_types_str = ', '.join(config['entity_types'])
    rel_types_str = ', '.join(config['rel_types'])
    
    prompt = f"""
    You are an expert in extracting structured data from text. Extract key entities and relationships from the following text chunk. 
    
    STRICT RULES:
    1. Use ONLY these entity types: {entity_types_str}.
    2. Use ONLY these relation types: {rel_types_str}.
    3. Normalize entities (e.g., "Luke Skywalker" and "Luke" should be the same entity).
    4. Relations must be directional, referencing entity IDs.
    
    Text: {chunk_text}
    """
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a specialized text-to-Knowledge Graph data extractor. Your ONLY job is to output a single, valid JSON object conforming strictly to the provided Pydantic schema."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format=KnowledgeGraphExtraction,
            )

            parsed_data: KnowledgeGraphExtraction = completion.choices[0].message.parsed
            extraction = parsed_data.model_dump()
            return extraction
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Pydantic/API Error: {e} - Retrying...")
            continue
            
    print(f"Failed to get valid Pydantic object after {max_retries} attempts for chunk: {chunk_text[:50]}... - Skipping")
    return {"entities": [], "relations": []}

def merge_entities(all_entities):
    """Performs fuzzy matching on entity names to create canonical IDs."""
    merged = {}
    for ent in all_entities:
        matched_key = None
        for key, canonical_ent in merged.items():
            # Match if name ratio is high AND types are identical
            if Levenshtein.ratio(ent['name'], key) > 0.8 and ent['type'] == canonical_ent['type']:
                matched_key = key
                break
        
        if matched_key is None:
            # New canonical entity
            merged[ent['name']] = {'id': ent['id'], 'type': ent['type']}
        # If matched, we don't update `merged`, keeping the original canonical ID
            
    return merged

# OPTIMIZATION: Helper function for transactional execution
def _execute_write_transaction(driver, database, query, parameters):
    """Executes a Cypher query within a transaction for stability."""
    def execute_tx(tx, query, parameters):
        tx.run(query, **parameters)

    with driver.session(database=database) as session:
        try:
            # execute_write provides retry logic and handles session closing
            session.execute_write(execute_tx, query, parameters)
        except Exception as e:
            # Only print the transaction failure, allow main loop to continue
            print(f"Neo4j Transaction Failed: {e}")

def build_knowledge_graph(dataset, json_file, neo4j_uri, neo4j_user, neo4j_pass, neo4j_db, extraction_file=None, max_documents=None):
    """
    Builds a Chunk-Only Knowledge Graph using saved extractions if available,
    or runs the LLM extraction if not.
    
    OPTIMIZED: Neo4j writes are performed document-by-document for stability.
    """
    # 1. Validation and Setup
    valid_schemes = ['bolt', 'bolt+ssc', 'bolt+s', 'neo4j', 'neo4j+ssc', 'neo4j+s']
    if not neo4j_uri or not any(neo4j_uri.startswith(scheme + '://') for scheme in valid_schemes):
        raise ValueError(f"Invalid Neo4j URI: {neo4j_uri}. Must start with one of {valid_schemes}")

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Chunk input file not found: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    # --- DOCUMENT LIMIT LOGIC ---
    # Group all chunks by document ID to determine which documents to process
    chunks_by_initial_doc: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in all_chunks:
        doc_id = chunk['doc_id']
        chunks_by_initial_doc.setdefault(doc_id, []).append(chunk)

    # Apply max_documents limit if set
    doc_ids_to_process = list(chunks_by_initial_doc.keys())
    if max_documents is not None:
        doc_ids_to_process = doc_ids_to_process[:max_documents]
        print(f"Limiting processing to the first {max_documents} unique documents: {len(doc_ids_to_process)} documents selected.")

    # Create the final list of chunks to process
    chunks_to_process = [
        chunk 
        for doc_id in doc_ids_to_process
        for chunk in chunks_by_initial_doc[doc_id]
    ]
    # ---------------------------------
    
    # Determine the path for the extracted KG data
    if extraction_file:
        extraction_output_path = extraction_file
    else:
        output_dir = os.path.dirname(json_file)
        extraction_output_path = os.path.join(output_dir, 'kg_extractions', 'extractions.json')
        os.makedirs(os.path.dirname(extraction_output_path), exist_ok=True)
    
    extractions = []
    all_entities_list = []
    
    # 2. Resumable Extraction Logic
    if os.path.exists(extraction_output_path):
        print(f"✅ Found existing extractions file at {extraction_output_path}. Loading data to skip LLM calls.")
        with open(extraction_output_path, 'r', encoding='utf-8') as f:
            all_extractions = json.load(f)
        
        # Filter loaded extractions to only include documents we are processing
        extractions = [ext for ext in all_extractions if ext['doc_id'] in doc_ids_to_process]
        
        # Re-populate all_entities_list for global merging/re-alignment from the filtered set
        all_entities_list = [
            ent for ext in extractions for ent in ext['entities']
        ]
    else:
        print(f"⚠️ No existing extractions file found. Starting LLM extraction (This may take time)...")
        # Ensure 'chunk' data includes necessary metadata before extraction
        for i, chunk in enumerate(tqdm(chunks_to_process, desc=f"Extracting for {dataset}")):
            extraction = extract_entities_relations(chunk['chunk'], dataset) 
            extraction['doc_id'] = chunk['doc_id']
            extraction['chunk_id'] = chunk['chunk_id']
            extraction['title'] = chunk.get('title', '')
            extraction['chunk_text'] = chunk['chunk'] # Store original text for node properties
            
            all_entities_list.extend(extraction['entities'])
            extractions.append(extraction)

        with open(extraction_output_path, 'w', encoding='utf-8') as f:
            json.dump(extractions, f, ensure_ascii=False, indent=4)
        print(f"Extractions saved at {extraction_output_path}")

    # 3. Global Entity Merging & ID Re-alignment (Required for cross-document linking)
    print("Performing global entity normalization (Levenshtein fuzzy matching)...")
    global_entities = merge_entities(all_entities_list)
    
    for ext in tqdm(extractions, desc="Re-aligning entity IDs"):
        for ent in ext['entities']:
            matched_name = next((key for key in global_entities if Levenshtein.ratio(ent['name'], key) > 0.8 and ent['type'] == global_entities[key]['type']), None)
            if matched_name:
                ent['id'] = global_entities[matched_name]['id']

    # 4. In-Memory Linking for Chunk-Only Graph
    entity_to_chunks = {}
    all_unique_chunks_data: List[Dict[str, Any]] = []
    unique_chunk_keys: Set[Tuple[str, str]] = set()
    
    # Collect all chunk data and map entities to the chunks they appear in
    for ext in extractions:
        doc_id = ext['doc_id']
        chunk_id = ext['chunk_id']
        chunk_key = (doc_id, chunk_id)
        
        if chunk_key not in unique_chunk_keys:
            unique_chunk_keys.add(chunk_key)
            all_unique_chunks_data.append({
                'doc_id': doc_id, 
                'chunk_id': chunk_id,
                'title': ext.get('title', ''),
                'chunk_text': ext.get('chunk_text', '') 
            })

        for ent in ext['entities']:
            entity_id = ent['id']
            # Map the shared, canonical entity ID to the chunks they appear in
            entity_to_chunks.setdefault(entity_id, set()).add(chunk_key)

    direct_chunk_links = set()
    print("Identifying direct chunk-to-chunk links based on shared entity IDs (Intra-Document Only)...")
    
    for entity_id, chunk_keys in tqdm(entity_to_chunks.items(), desc="Processing shared entities"):
        chunk_keys_list = list(chunk_keys)
        if len(chunk_keys_list) > 1:
            for i in range(len(chunk_keys_list)):
                for j in range(i + 1, len(chunk_keys_list)):
                    c1 = chunk_keys_list[i]
                    c2 = chunk_keys_list[j]
                    
                    # *** NEW CHECK: Only create link if both chunks are from the SAME document ***
                    if c1[0] == c2[0]:
                        # Use a sorted tuple to ensure the link (A, B) is the same as (B, A)
                        link = tuple(sorted([c1, c2])) 
                        direct_chunk_links.add(link)

    # 5. Build Chunk-Only KG in Neo4j (Document-by-Document Batched UNWIND)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    try:
        driver.verify_connectivity()
    except Exception as e:
        raise ConnectionError(f"Neo4j connection failed: {e}")

    # OPTIMIZATION: Use transactional function for all writes
    # A. Clean up and Create Schema Constraints
    _execute_write_transaction(driver, neo4j_db, "MATCH (n) DETACH DELETE n", {})
    _execute_write_transaction(driver, neo4j_db, 
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.doc_id, c.chunk_id) IS UNIQUE", 
        {}
    )
    # CONSTRAINT for Document node
    _execute_write_transaction(driver, neo4j_db, 
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE", 
        {}
    )
    
    # Group all data by document ID for iterative processing
    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for chunk_data in all_unique_chunks_data:
        doc_id = chunk_data['doc_id']
        chunks_by_doc.setdefault(doc_id, []).append(chunk_data)

    # Group links by *only* document ID (since links are now intra-document)
    links_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for c1, c2 in direct_chunk_links:
        link_data = {'d1': c1[0], 'c1': c1[1], 'd2': c2[0], 'c2': c2[1]}
        # Since c1[0] must equal c2[0] now, we only need to group by one of them
        links_by_doc.setdefault(c1[0], []).append(link_data)
        

    print(f"Starting Neo4j insertion document by document ({len(chunks_by_doc)} unique documents)...")

    # Loop through each document and process its nodes and relationships
    for doc_id, chunk_list in tqdm(chunks_by_doc.items(), desc="Inserting Nodes/Rels by Document"):
        
        # B. Batched Document and Chunk Node Creation with Linking for current document
        doc_chunk_link_query = """
            UNWIND $chunks AS c
            // 1. Find or create the Document node
            MERGE (dnode:Document {doc_id: c.doc_id})
            ON CREATE SET dnode.title = c.title, dnode.source = $dataset
            
            // 2. Find or create the Chunk node
            MERGE (cnode:Chunk {doc_id: c.doc_id, chunk_id: c.chunk_id})
            SET cnode.title = c.title, cnode.chunk_text = c.chunk_text, cnode.source = $dataset
            
            // 3. Link Chunk to Document
            MERGE (cnode)-[:BELONGS_TO]->(dnode)
        """
        _execute_write_transaction(driver, neo4j_db, doc_chunk_link_query, {'chunks': chunk_list, 'dataset': dataset})

        # C. Batched Chunk-to-Chunk Relationship Creation for current document
        if doc_id in links_by_doc:
            links_for_unwind = links_by_doc[doc_id]
            
            # Sub-batch the links for extra stability
            BATCH_SIZE = 5000
            for i in range(0, len(links_for_unwind), BATCH_SIZE):
                batch = links_for_unwind[i:i + BATCH_SIZE]
                rel_query = """
                    UNWIND $links AS l
                    MATCH (c1:Chunk {doc_id: l.d1, chunk_id: l.c1})
                    MATCH (c2:Chunk {doc_id: l.d2, chunk_id: l.c2})
                    // Create bidirectional relationship based on shared entity
                    MERGE (c1)-[:RELATES_TO {via: "shared_entity"}]->(c2)
                    MERGE (c2)-[:RELATES_TO {via: "shared_entity"}]->(c1)
                """
                _execute_write_transaction(driver, neo4j_db, rel_query, {'links': batch})
        
    print(f"Chunk-Only Knowledge Graph built for {dataset} in Neo4j database {neo4j_db}. Optimized for traversal.")
    driver.close()

def retrieve_for_rag(top_chunk_ids, dataset, json_file, neo4j_uri, neo4j_user, neo4j_pass, neo4j_db, max_related=5):
    """
    Retrieve chunks and their related chunks from the KG for RAG.
    This version is optimized for the Chunk-Only KG structure.
    Args:
        top_chunk_ids: List of (doc_id, chunk_id) tuples from FAISS top-k.
        dataset: Dataset name (e.g., 'narrativeqa').
        json_file: Path to chunk JSON file (used for fallback, but ideally text is retrieved from Neo4j).
        neo4j_uri, neo4j_user, neo4j_pass: Neo4j connection details.
        neo4j_db: Neo4j database name.
        max_related: Max number of related chunks per top-k chunk.
    Returns:
        List of (doc_id, chunk_id, chunk_text) for top-k and related chunks.
    """
    # NOTE: Since we updated build_knowledge_graph to store chunk_text on the node, 
    # we can retrieve the text directly from Neo4j, making this function more self-contained.

    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    
    # Use a set to manage unique (doc_id, chunk_id) to prevent duplication
    retrieved_chunks_set = set()
    related_chunks_data = [] # Stores (doc_id, chunk_id, chunk_text)
    
    # Transform input list of tuples into list of dictionaries for UNWIND
    input_chunks_for_query = [{'doc_id': d, 'chunk_id': c} for d, c in top_chunk_ids]

    # Query 1: Get the text of the initial top-k chunks and mark them as seen
    with driver.session(database=neo4j_db) as session:
        top_k_query = """
        UNWIND $chunks AS c_in
        MATCH (c1:Chunk {doc_id: c_in.doc_id, chunk_id: c_in.chunk_id})
        RETURN c1.doc_id AS doc_id, c1.chunk_id AS chunk_id, c1.chunk_text AS chunk_text
        """
        initial_results = session.run(top_k_query, chunks=input_chunks_for_query)
        
        for record in initial_results:
            key = (record['doc_id'], record['chunk_id'])
            if key not in retrieved_chunks_set:
                retrieved_chunks_set.add(key)
                related_chunks_data.append((record['doc_id'], record['chunk_id'], record['chunk_text']))

    # Query 2: Traverse from the initial chunks to find related context
    with driver.session(database=neo4j_db) as session:
        for doc_id, chunk_id in top_chunk_ids:
            # The retrieval query remains the same, but because the graph is structured
            # not to have cross-document RELATES_TO links, this will only return
            # chunks from the same document.
            related_query = """
                MATCH (c1:Chunk {doc_id: $doc_id, chunk_id: $chunk_id})-[:RELATES_TO]->(c2:Chunk)
                WHERE c1 <> c2
                RETURN DISTINCT c2.doc_id AS doc_id, c2.chunk_id AS chunk_id, c2.chunk_text AS chunk_text
                LIMIT $max_related
            """
            result = session.run(related_query, doc_id=doc_id, chunk_id=chunk_id, max_related=max_related)
            
            for record in result:
                key = (record['doc_id'], record['chunk_id'])
                if key not in retrieved_chunks_set:
                    retrieved_chunks_set.add(key)
                    related_chunks_data.append((record['doc_id'], record['chunk_id'], record['chunk_text']))

    driver.close()
    return related_chunks_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from chunked dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g., 'narrativeqa', 'quality', 'qasper')")
    parser.add_argument('--json_file', type=str, required=True, help="Path to chunk input JSON file (e.g., 'data/narrativeqa/semantic/semantic.json')")
    parser.add_argument('--extraction_file', type=str, required=False, help="Path to the output JSON file containing extracted entities and relations (e.g., 'data/kg_extractions/extractions.json'). If provided, LLM output is saved here and loaded from here on subsequent runs.")
    parser.add_argument('--neo4j_uri', type=str, default=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'), help="Neo4j URI")
    parser.add_argument('--neo4j_user', type=str, default=os.environ.get('NEO4J_USER', 'neo4j'), help="Neo4j username")
    parser.add_argument('--neo4j_pass', type=str, default=os.environ.get('NEO4J_PASS', 'password'), help="Neo4j password")
    parser.add_argument('--neo4j_db', type=str, default=os.environ.get('NEO4J_DB', 'neo4J'), help="Neo4j database name (e.g., 'narrativeqa_db')")
    parser.add_argument('--max_documents', type=int, default=None, help="Maximum number of unique documents to process. Default is all documents.")
    
    args = parser.parse_args()
    build_knowledge_graph(
        args.dataset, 
        args.json_file, 
        args.neo4j_uri, 
        args.neo4j_user, 
        args.neo4j_pass, 
        args.neo4j_db,
        args.extraction_file,
        args.max_documents # Pass the new argument
    )
