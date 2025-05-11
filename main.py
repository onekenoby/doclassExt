# main.py
from preprocess.text_extractor import extract_text_from_file
from gemini.gemini_client import (
    generate_structured_schema_and_cypher,
    generate_semantic_narrative # You might adjust how narrative is generated with chunking
)
# Import the function for individual query execution
from graphdb.graph_builder import execute_cypher_queries_individually

# --- Configuration ---
CHUNK_SIZE = 4000  # Approximate characters per chunk. Adjust based on your document and testing.
# You might need a chunking strategy that respects paragraph/sentence boundaries
# for better results, but simple character-based splitting is shown here.

def simple_text_chunker(text: str, chunk_size: int):
    """Splits text into chunks of approximately chunk_size."""
    # A more sophisticated chunker would consider sentence/paragraph boundaries
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        # Add 1 for the space character
        if current_size + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def main():
    # 1. Extract raw document text
    # Make sure to use the correct path to sample.pdf
    # Adjusted path assuming sample.pdf is in the same dir as main.py. Adjust if needed.
    file_path = "samples/sample.pdf"
    print(f"Extracting text from {file_path}...")
    paragraphs = extract_text_from_file(file_path)
    doc_text = "\n".join(paragraphs)
    print(f"Text extracted. Total characters: {len(doc_text)}")


    # 2. Split text into chunks
    chunks = simple_text_chunker(doc_text, CHUNK_SIZE)
    print(f"Document split into {len(chunks)} chunks.")

    all_cypher_statements = []
    all_hierarchies = [] # You would need merging logic for these
    all_schemas = []     # You would need merging logic for these

    print("\n--- Processing Chunks ---")
    for i, chunk in enumerate(chunks):
        print(f"\nProcessing Chunk {i+1}/{len(chunks)} (Size: {len(chunk)} characters)...")

        # 3. Generate structured hierarchy, schema, and Cypher for the chunk
        # This call now uses the gemini_client with lower temperature
        try:
            chunk_result = generate_structured_schema_and_cypher(chunk)
            if not chunk_result or not chunk_result.get("cypher"): # Use .get() for safer access
                print(f"--- WARNING: No valid result or Cypher statements generated for Chunk {i+1}. ---")
                continue

            chunk_hierarchy = chunk_result.get("hierarchy")
            chunk_schema = chunk_result.get("schema")
            chunk_cypher_statements = chunk_result.get("cypher", [])

            all_hierarchies.append(chunk_hierarchy) # Collect results
            all_schemas.append(chunk_schema)       # Collect results
            all_cypher_statements.extend(chunk_cypher_statements) # Collect all Cypher statements

            # Optional: Print results for each chunk
            # print(f"Hierarchy for Chunk {i+1}: {json.dumps(chunk_hierarchy, indent=2)}")
            # print(f"Schema for Chunk {i+1}: {json.dumps(chunk_schema, indent=2)}")
            # print(f"Cypher statements for Chunk {i+1}:")
            # for stmt in chunk_cypher_statements:
            #     print(stmt)

        except Exception as e:
            print(f"--- ERROR processing Chunk {i+1}: {e} ---")


    print("\n--- Finished Processing Chunks ---")

    # --- Merging Results (Conceptual - Requires Implementation) ---
    # At this point, you have lists of hierarchies, schemas, and a combined
    # list of all Cypher statements from all chunks.
    # A full implementation would require logic here to:
    # 1. Merge `all_hierarchies` into a single document hierarchy.
    # 2. Merge `all_schemas` into a single overall schema (collecting all unique node labels and relationship types).
    # 3. Process `all_cypher_statements` to build the graph. This is the most complex part.
    #    Simply running all statements might work if `execute_cypher_queries_individually`
    #    handles variable scope correctly, but you might get duplicate nodes for entities
    #    mentioned in multiple chunks. A robust solution would involve using MERGE
    #    statements or post-processing the Cypher to identify and merge duplicate nodes.

    # For demonstration, we will execute the collected Cypher statements individually.
    # This resolves variable conflicts *within* each statement but might create
    # duplicate nodes if the same entity appears in different chunks.
    print("\n--- Executing All Collected Cypher Statements ---")
    if all_cypher_statements:
        execute_cypher_queries_individually(all_cypher_statements)
    else:
        print("No Cypher statements collected to execute.")

    # 4. Semantic interpretation narrative (using collected or merged schema/hierarchy)
    # For a full document narrative, you would ideally use a merged hierarchy and schema.
    # For simplicity here, we might generate a narrative based on the collected schema/hierarchy
    # or process the original text again with a different prompt focused on summarization.
    # Or, you could take the largest/most representative chunk's narrative.
    # This part needs adjustment based on how you implement merging.
    # As a placeholder, let's use the schema/hierarchy from the first chunk if available,
    # or a simple message if not.

    print("\n=== Generating Semantic Interpretation ===")
    if all_hierarchies and all_schemas:
        # Using the first chunk's schema/hierarchy for narrative as a simple example
        # Replace with logic using merged_hierarchy and merged_schema in a full implementation
        try:
             narrative = generate_semantic_narrative(all_hierarchies[0], all_schemas[0])
             print("\n" + narrative)
        except Exception as e:
             print(f"--- ERROR generating semantic narrative: {e} ---")
             print("Narrative generation skipped.")
    else:
        print("Cannot generate semantic narrative: No schema or hierarchy available from chunks.")


if __name__ == "__main__":
    main()