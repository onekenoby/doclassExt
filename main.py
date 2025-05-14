"""
End-to-end runner:  PDF → text → Gemini → Neo4j graph
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from gemini.gemini_client import (
    generate_structured_schema_and_cypher,
    generate_semantic_narrative,
)
from graphdb.graph_builder import execute_cypher_batched
from preprocess.text_extractor import extract_text_from_file
from preprocess.text_processor import preprocess_paragraphs
from preprocess.chunker import chunk_by_tokens

# ───────────────────────────  settings  ────────────────────────────
MAX_TOKENS_PER_CHUNK = 1500       # keep each prompt safely under model limit
DEFAULT_PDF          = Path("samples/sample.pdf")
#max_workers = 4

# -------------------------------------------------------------------
def main() -> None:
    # --------  pick PDF path  --------------------------------------
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF

    print(f"Extracting text from {file_path} …")
    raw_paragraphs: List[str] = extract_text_from_file(file_path)
    clean_text: str = preprocess_paragraphs(raw_paragraphs)

    chunks: List[str] = chunk_by_tokens(clean_text, max_tokens=MAX_TOKENS_PER_CHUNK)
    print(f"Document split into {len(chunks)} chunk(s).")

    # concurrency adapts to job size (helps stay inside rate-limits)
    max_workers = 2

    all_cypher: List[str] = []
    all_hierarchies: List[Dict] = []
    all_schemas: List[Dict] = []

    # --------  parallel LLM calls  ---------------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(generate_structured_schema_and_cypher, chunk): idx
            for idx, chunk in enumerate(chunks, 1)
        }

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"✖️  Chunk {idx} failed: {e}")
                continue

            if not isinstance(res, dict):
                print(f"✖️  Chunk {idx} produced no usable JSON.")
                continue

            all_hierarchies.append(res.get("hierarchy"))
            all_schemas.append(res.get("schema"))
            cypher_part = res.get("cypher", [])
            if isinstance(cypher_part, str):
                all_cypher.extend(s for s in cypher_part.split(";") if s.strip())
            elif isinstance(cypher_part, list):
                all_cypher.extend(cypher_part)
            print(f"✅ Chunk {idx} done.")

    # --------  write graph  ----------------------------------------
    if all_cypher:
        print("\nWriting graph to Neo4j …")
        execute_cypher_batched(all_cypher)
    else:
        print("\n⚠️  No Cypher statements were generated – graph step skipped.")

    # --------  optional narrative  ---------------------------------
    if all_hierarchies and all_schemas:
        try:
            narrative = generate_semantic_narrative(
                all_hierarchies[0],
                all_schemas[0],
            )
            print("\n=== Semantic Narrative ===\n")
            print(narrative)
        except Exception as e:
            print(f"✖️  Narrative skipped: {e}")


if __name__ == "__main__":
    main()
