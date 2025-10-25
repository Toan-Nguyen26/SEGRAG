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
from typing import List

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
    raise ValueError("OPENAI_API_KEY not set in environment variables")

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
    # Define the JSON structure template for clear instruction
    json_structure_template = (
    "{\n"
    '    "entities": [{"id": "UUID", "name": "Entity Name", "type": "One of the allowed types"}],\n'
    '    "relations": [{"head": "head_entity_id", "relation": "One of the allowed types", "tail": "tail_entity_id"}]\n'
    "}"
    )

    config = DATASET_CONFIG.get(dataset, {
        'entity_types': ['Person', 'Location', 'Organization', 'Event', 'Concept'],
        'rel_types': ['RELATED_TO', 'CAUSES', 'LOCATED_IN']
    })
    entity_types_str = ', '.join(config['entity_types'])
    rel_types_str = ', '.join(config['rel_types'])
    
    # NOTE: The prompt can be simpler, as the Pydantic model handles the structure
    prompt = f"""
    You are an expert in extracting structured data from text. Extract key entities and relationships from the following text chunk. 
    
    STRICT RULES:
    1. Use ONLY these entity types: {entity_types_str}.
    2. Use ONLY these relation types: {rel_types_str}.
    3. Normalize entities (e.g., "Luke Skywalker" and "Luke" should be the same entity).
    4. Relations must be directional, referencing entity IDs.
    
    Text: {chunk_text}
    """
    
    # 🌟 KEY CHANGE: Use parse() with the Pydantic model
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
                # Pass the Pydantic class here!
                response_format=KnowledgeGraphExtraction,
            )

            # The result is already a Pydantic object, which can be converted to a dictionary
            parsed_data: KnowledgeGraphExtraction = completion.choices[0].message.parsed
            
            # Convert Pydantic object to a standard Python dictionary for the rest of your pipeline
            extraction = parsed_data.model_dump()
            
            # The UUIDs are now automatically handled by the Pydantic default_factory!
            return extraction
            
        except Exception as e:
            # Pydantic exceptions or other parsing/API errors will land here
            print(f"Attempt {attempt + 1}/{max_retries}: Pydantic/API Error: {e} - Retrying...")
            continue
            
    print(f"Failed to get valid Pydantic object after {max_retries} attempts for chunk: {chunk_text[:50]}... - Skipping")
    return {"entities": [], "relations": []}

def merge_entities(all_entities):
    merged = {}
    for ent in all_entities:
        matched = False
        for key in list(merged):
            if Levenshtein.ratio(ent['name'], key) > 0.8 and ent['type'] == merged[key]['type']:
                matched = True
                break
        if not matched:
            merged[ent['name']] = {'id': ent['id'], 'type': ent['type']}
    return merged

def build_knowledge_graph_old(dataset, json_file, neo4j_uri, neo4j_user, neo4j_pass, neo4j_db):
    # Validate Neo4j URI
    valid_schemes = ['bolt', 'bolt+ssc', 'bolt+s', 'neo4j', 'neo4j+ssc', 'neo4j+s']
    if not neo4j_uri or not any(neo4j_uri.startswith(scheme + '://') for scheme in valid_schemes):
        raise ValueError(f"Invalid Neo4j URI: {neo4j_uri}. Must start with one of {valid_schemes}")

    # Load chunks from JSON file
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Extract entities/rels
    extractions = []
    all_entities_list = []
    for chunk in tqdm(chunks, desc=f"Extracting for {dataset}"):
        extraction = extract_entities_relations(chunk['chunk'], dataset)
        extraction['doc_id'] = chunk['doc_id']
        extraction['chunk_id'] = chunk['chunk_id']
        extraction['title'] = chunk.get('title', '')
        all_entities_list.extend(extraction['entities'])
        extractions.append(extraction)

    # Global entity merging
    global_entities = merge_entities(all_entities_list)
    for ext in extractions:
        for ent in ext['entities']:
            matched_name = next((key for key in global_entities if Levenshtein.ratio(ent['name'], key) > 0.8 and ent['type'] == global_entities[key]['type']), None)
            if matched_name:
                ent['id'] = global_entities[matched_name]['id']

    # Save extractions
    output_dir = os.path.dirname(json_file)
    os.makedirs(os.path.join(output_dir, 'kg_extractions'), exist_ok=True)
    output_path = os.path.join(output_dir, 'kg_extractions', 'extractions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extractions, f, ensure_ascii=False, indent=4)
    print(f"Extractions saved at {output_path}")

    # Build KG in Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    try:
        driver.verify_connectivity()
    except Exception as e:
        raise ConnectionError(f"Neo4j connection failed: {e}")

    with driver.session(database=neo4j_db) as session:
        session.run("MATCH (n) DETACH DELETE n")
        for name, data in tqdm(global_entities.items(), desc="Creating entities"):
            session.run("""
                MERGE (e:Entity {id: $id, name: $name, type: $type})
                """, id=data['id'], name=name, type=data['type'])

        for ext in tqdm(extractions, desc="Creating relations and chunk links"):
            for rel in ext['relations']:
                session.run("""
                    MATCH (h:Entity {id: $head})
                    MATCH (t:Entity {id: $tail})
                    MERGE (h)-[r:REL {type: $rel_type}]->(t)
                    SET r.doc_id = $doc_id, r.chunk_id = $chunk_id
                    """, head=rel['head'], tail=rel['tail'], rel_type=rel['relation'],
                         doc_id=ext['doc_id'], chunk_id=ext['chunk_id'])

            for ent in ext['entities']:
                session.run("""
                    MATCH (e:Entity {id: $ent_id})
                    MERGE (c:Chunk {doc_id: $doc_id, chunk_id: $chunk_id})
                    MERGE (e)-[:IN_CHUNK]->(c)
                    MERGE (c)-[:CONTAINS]->(e) 
                    """, ent_id=ent['id'], doc_id=ext['doc_id'], chunk_id=ext['chunk_id'])

        # Add chunk-to-chunk relationships
        session.run("""
            MATCH (c1:Chunk)<-[:IN_CHUNK]-(e:Entity)-[:IN_CHUNK]->(c2:Chunk)
            WHERE c1 <> c2
            MERGE (c1)-[:RELATES_TO {via: "shared_entity"}]->(c2)
        """)
        session.run("""
            MATCH (c1:Chunk)<-[:IN_CHUNK]-(e1:Entity)-[:REL*1..2]->(e2:Entity)-[:IN_CHUNK]->(c2:Chunk)
            WHERE c1 <> c2
            MERGE (c1)-[:RELATES_TO {via: "entity_path"}]->(c2)
        """)

    print(f"Knowledge Graph built for {dataset} in Neo4j database {neo4j_db}. Supports Chunk->Entity and Chunk->Chunk traversal.")
    driver.close()

def build_knowledge_graph(dataset, json_file, neo4j_uri, neo4j_user, neo4j_pass, neo4j_db):
    """
    Builds a Chunk-Only Knowledge Graph using saved extractions if available,
    or runs the LLM extraction if not. The graph contains only Chunk nodes
    linked by RELATES_TO based on shared entities.
    """
    # 1. Validation and Setup
    valid_schemes = ['bolt', 'bolt+ssc', 'bolt+s', 'neo4j', 'neo4j+ssc', 'neo4j+s']
    if not neo4j_uri or not any(neo4j_uri.startswith(scheme + '://') for scheme in valid_schemes):
        raise ValueError(f"Invalid Neo4j URI: {neo4j_uri}. Must start with one of {valid_schemes}")

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    output_dir = os.path.dirname(json_file)
    extraction_output_path = os.path.join(output_dir, 'kg_extractions', 'extractions.json')
    
    extractions = []
    all_entities_list = []

    # 2. Resumable Extraction Logic (New)
    if os.path.exists(extraction_output_path):
        print(f"✅ Found existing extractions file at {extraction_output_path}. Loading data to skip LLM calls.")
        with open(extraction_output_path, 'r', encoding='utf-8') as f:
            extractions = json.load(f)
        
        # Re-populate all_entities_list for global merging/re-alignment
        all_entities_list = [
            ent for ext in extractions for ent in ext['entities']
        ]
    else:
        # Run the slow LLM extraction loop (Original logic)
        print(f"⚠️ No existing extractions file found. Starting LLM extraction (This may take time)...")
        for chunk in tqdm(chunks, desc=f"Extracting for {dataset}"):
            # NOTE: Assuming extract_entities_relations and other helper functions are imported/defined.
            extraction = extract_entities_relations(chunk['chunk'], dataset) 
            extraction['doc_id'] = chunk['doc_id']
            extraction['chunk_id'] = chunk['chunk_id']
            extraction['title'] = chunk.get('title', '')
            
            all_entities_list.extend(extraction['entities'])
            extractions.append(extraction)

        # Save the extractions (Crucial for next run)
        os.makedirs(os.path.join(output_dir, 'kg_extractions'), exist_ok=True)
        with open(extraction_output_path, 'w', encoding='utf-8') as f:
            json.dump(extractions, f, ensure_ascii=False, indent=4)
        print(f"Extractions saved at {extraction_output_path}")

    # 3. Global Entity Merging & ID Re-alignment (Required for linking logic)
    global_entities = merge_entities(all_entities_list)
    
    # Re-align entity IDs in the extractions data (critical step)
    for ext in extractions:
        for ent in ext['entities']:
            # Find the globally merged ID
            matched_name = next((key for key in global_entities if Levenshtein.ratio(ent['name'], key) > 0.8 and ent['type'] == global_entities[key]['type']), None)
            if matched_name:
                ent['id'] = global_entities[matched_name]['id']

    # 4. In-Memory Linking for Chunk-Only Graph (New Optimized Logic)
    entity_to_chunks = {}
    all_unique_chunks_data = set()
    
    for ext in extractions:
        doc_id = ext['doc_id']
        chunk_id = ext['chunk_id']
        chunk_key = (doc_id, chunk_id)
        all_unique_chunks_data.add(chunk_key)

        for ent in ext['entities']:
            entity_id = ent['id']
            # Map the shared entity ID to the chunks it appears in
            entity_to_chunks.setdefault(entity_id, set()).add(chunk_key)

    direct_chunk_links = set()
    print("Identifying direct chunk-to-chunk links based on shared entity IDs...")
    
    for entity_id, chunk_keys in tqdm(entity_to_chunks.items(), desc="Processing shared entities"):
        chunk_keys_list = list(chunk_keys)
        if len(chunk_keys_list) > 1:
            # Create a link between every pair of chunks that share this entity
            for i in range(len(chunk_keys_list)):
                for j in range(i + 1, len(chunk_keys_list)):
                    c1 = chunk_keys_list[i]
                    c2 = chunk_keys_list[j]
                    link = tuple(sorted([c1, c2])) 
                    direct_chunk_links.add(link)

    # 5. Build Chunk-Only KG in Neo4j (Batched UNWIND)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    try:
        driver.verify_connectivity()
    except Exception as e:
        raise ConnectionError(f"Neo4j connection failed: {e}")

    with driver.session(database=neo4j_db) as session:
        session.run("MATCH (n) DETACH DELETE n")
        
        # A. Create Schema Constraints
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.doc_id, c.chunk_id) IS UNIQUE")

        # B. Batched Chunk Node Creation
        print(f"Creating {len(all_unique_chunks_data)} unique Chunk nodes...")
        chunks_for_unwind = [{'doc_id': d, 'chunk_id': c} for d, c in all_unique_chunks_data]
        session.run("""
            UNWIND $chunks AS c
            MERGE (cnode:Chunk {doc_id: c.doc_id, chunk_id: c.chunk_id})
            """, chunks=chunks_for_unwind)

        # C. Batched Chunk-to-Chunk Relationship Creation
        print(f"Creating {len(direct_chunk_links)} RELATES_TO relationships...")
        links_for_unwind = [
            {'d1': c1[0], 'c1': c1[1], 'd2': c2[0], 'c2': c2[1]}
            for c1, c2 in direct_chunk_links
        ]
        
        session.run("""
            UNWIND $links AS l
            MATCH (c1:Chunk {doc_id: l.d1, chunk_id: l.c1})
            MATCH (c2:Chunk {doc_id: l.d2, chunk_id: l.c2})
            // Create bidirectional relationship based on shared entity
            MERGE (c1)-[:RELATES_TO {via: "shared_entity"}]->(c2)
            MERGE (c2)-[:RELATES_TO {via: "shared_entity"}]->(c1)
            """, links=links_for_unwind)
        
    print(f"Chunk-Only Knowledge Graph built for {dataset} in Neo4j database {neo4j_db}. Optimized for traversal.")
    driver.close()
    
def retrieve_for_rag(top_chunk_ids, dataset, json_file, neo4j_uri, neo4j_user, neo4j_pass, neo4j_db, max_related=5):
    """
    Retrieve chunks and their related chunks from the KG for RAG.
    Args:
        top_chunk_ids: List of (doc_id, chunk_id) tuples from FAISS top-k.
        dataset: Dataset name (e.g., 'narrativeqa').
        json_file: Path to chunk JSON file.
        neo4j_uri, neo4j_user, neo4j_pass: Neo4j connection details.
        neo4j_db: Neo4j database name.
        max_related: Max number of related chunks per top-k chunk.
    Returns:
        List of (doc_id, chunk_id, chunk_text) for top-k and related chunks.
    """
    # Load chunks
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    
    # Expand to related chunks
    related_chunks = []
    with driver.session(database=neo4j_db) as session:
        for doc_id, chunk_id in top_chunk_ids:
            # Get the chunk text
            chunk = next((c for c in chunks if c['doc_id'] == doc_id and c['chunk_id'] == chunk_id), None)
            if not chunk:
                print(f"Chunk not found: {doc_id}, {chunk_id}")
                continue
            related_chunks.append((doc_id, chunk_id, chunk['chunk']))

            # Option 1: Use Chunk -> CONTAINS -> Entity -> REL -> Entity -> IN_CHUNK -> Chunk
            result = session.run("""
                MATCH (c1:Chunk {doc_id: $doc_id, chunk_id: $chunk_id})-[:CONTAINS]->(e:Entity)
                MATCH (e)-[:REL*1..2]->(e2:Entity)-[:IN_CHUNK]->(c2:Chunk)
                WHERE c1 <> c2
                RETURN DISTINCT c2.doc_id AS doc_id, c2.chunk_id AS chunk_id
                LIMIT $max_related
                """, doc_id=doc_id, chunk_id=chunk_id, max_related=max_related)
            related = [(r['doc_id'], r['chunk_id']) for r in result]

            # Option 2: Use direct Chunk -> RELATES_TO -> Chunk
            result = session.run("""
                MATCH (c1:Chunk {doc_id: $doc_id, chunk_id: $chunk_id})-[:RELATES_TO]->(c2:Chunk)
                WHERE c1 <> c2
                RETURN DISTINCT c2.doc_id AS doc_id, c2.chunk_id AS chunk_id
                LIMIT $max_related
                """, doc_id=doc_id, chunk_id=chunk_id, max_related=max_related)
            related.extend([(r['doc_id'], r['chunk_id']) for r in result])

            # Deduplicate and add chunk text
            seen = {(doc_id, chunk_id)}
            for r_doc_id, r_chunk_id in related:
                if (r_doc_id, r_chunk_id) not in seen:
                    chunk = next((c for c in chunks if c['doc_id'] == r_doc_id and c['chunk_id'] == r_chunk_id), None)
                    if chunk:
                        related_chunks.append((r_doc_id, r_chunk_id, chunk['chunk']))
                        seen.add((r_doc_id, r_chunk_id))

    driver.close()
    return related_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from chunked dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g., 'narrativeqa', 'quality', 'qasper')")
    parser.add_argument('--json_file', type=str, required=True, help="Path to chunk JSON file (e.g., 'data/narrativeqa/semantic/semantic.json')")
    parser.add_argument('--neo4j_uri', type=str, default=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'), help="Neo4j URI")
    parser.add_argument('--neo4j_user', type=str, default=os.environ.get('NEO4J_USER', 'neo4j'), help="Neo4j username")
    parser.add_argument('--neo4j_pass', type=str, default=os.environ.get('NEO4J_PASS', 'password'), help="Neo4j password")
    parser.add_argument('--neo4j_db', type=str, default=os.environ.get('NEO4J_DB', 'neo4j'), help="Neo4j database name (e.g., 'narrativeqa_db')")
    
    args = parser.parse_args()
    build_knowledge_graph(args.dataset, args.json_file, args.neo4j_uri, args.neo4j_user, args.neo4j_pass, args.neo4j_db)