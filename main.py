# main.py
from preprocess.text_extractor import extract_text_from_file
from gemini.gemini_client import (
    generate_structured_schema_and_cypher,
    generate_semantic_narrative
)
from graphdb.graph_builder import execute_cypher_queries

def main():
    # 1. Extract raw document text
    text = extract_text_from_file("samples/sample.pdf")

    # 2. Generate structured hierarchy, schema, and Cypher
    result = generate_structured_schema_and_cypher(text)
    if not result or not result["cypher"]:
        print("\n--- ERROR: No valid result or Cypher statements generated. ---\n")
        return

    hierarchy = result["hierarchy"]
    schema = result["schema"]
    cypher_statements = result["cypher"]

    # 3. Populate Neo4j
    execute_cypher_queries("\n".join(cypher_statements))

    # 4. Semantic interpretation narrative
    narrative = generate_semantic_narrative(hierarchy, schema)
    print("\n=== Semantic Interpretation ===\n")
    print(narrative)

if __name__ == "__main__":
    main()