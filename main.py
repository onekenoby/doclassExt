"""
End-to-end runner for PDF → text → Gemini → Neo4j graph
"""

from __future__ import annotations

import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gemini.gemini_client import (
    generate_structured_schema_and_cypher,
    generate_semantic_narrative,
)
from graphdb.graph_builder import execute_cypher_batched
from preprocess.text_extractor import extract_text_from_file

# ───────────────────────────  settings  ────────────────────────────
CHUNK_SIZE  = 6_000        # characters per LLM call
MAX_WORKERS = 4            # parallel Gemini calls (watch your quota)

# -------------------------------------------------------------------
def simple_text_chunker(text: str, chunk_size: int):
    """Split text into word-boundary chunks of ≤ chunk_size."""
    return textwrap.wrap(text, width=chunk_size,
                         break_long_words=False,
                         break_on_hyphens=False)


def main() -> None:
    # --------  pick PDF path  --------------------------------------
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = Path("samples/sample.pdf")

    print(f"Extracting text from {file_path} …")
    paragraphs = extract_text_from_file(file_path)
    full_text  = "\n".join(paragraphs)

    chunks = simple_text_chunker(full_text, CHUNK_SIZE)
    print(f"Document split into {len(chunks)} chunk(s).")

    all_cypher: list[str] = []
    all_hierarchies: list[dict] = []
    all_schemas: list[dict] = []

    # --------  parallel LLM calls  ---------------------------------
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
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
                all_cypher.extend([s for s in cypher_part.split(";") if s.strip()])
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
            narrative = generate_semantic_narrative(all_hierarchies[0], all_schemas[0])
            print("\n=== Semantic Narrative ===\n")
            print(narrative)
        except Exception as e:
            print(f"✖️  Narrative skipped: {e}")


if __name__ == "__main__":
    main()
