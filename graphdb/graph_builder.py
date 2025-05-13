# graphdb/graph_builder.py
"""
Neo4j helper   –   now with a **batched** executor for 5-10× speed-up.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from neo4j import GraphDatabase, exceptions

# ── Neo4j connection ─────────────────────────────────────────────────────────
URI      = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
POOLSIZE = int(os.getenv("NEO4J_POOL_SIZE", "50"))

driver = GraphDatabase.driver(
    URI,
    auth=(USER, PASSWORD),
    max_connection_pool_size=POOLSIZE,
)

# ── Cypher sanitiser  (arrow + numeric-label fixes) ──────────────────────────
INVALID_LABEL  = re.compile(r":([A-Za-z_]*[0-9][A-Za-z0-9_]*)")
UNDIRECTED_REL = re.compile(r"-\s*\[:([A-Z_]+)\]\s*-(\s*\()")

def sanitise_cypher(statements: list[str]) -> list[str]:
    fixed: list[str] = []
    for stmt in statements:
        s = stmt.strip().rstrip(";")            # drop trailing semicolons
        s = UNDIRECTED_REL.sub(r"-[:\1]->\2", s)
        s = INVALID_LABEL.sub(lambda m: f":`{m.group(1)}`", s)
        if s:
            fixed.append(s)
    return fixed

# ─────────────────────────────────────────────────────────────────────────────
#  Executors
# ─────────────────────────────────────────────────────────────────────────────
def execute_cypher_batched(statements: list[str], batch_size: int = 250) -> None:
    """
    Send each Cypher statement separately but inside the *same* transaction
    (per batch).  This avoids the “one query only” Bolt rule and gives
    ∼10× speed-up versus one-Tx-per-statement.
    """
    if not statements:
        return

    with driver.session() as session:
        for i in range(0, len(statements), batch_size):
            chunk = sanitise_cypher(statements[i : i + batch_size])
            if not chunk:
                continue

            try:
                with session.begin_transaction() as tx:
                    for stmt in chunk:
                        tx.run(stmt)
                print(f"✅ Executed {len(chunk)} stmt(s) [{i}-{i+len(chunk)-1}]")
            except exceptions.Neo4jError as err:
                print(f"⚠️  Neo4j error in batch {i}-{i+len(chunk)-1}: {err}")


# ─────────────────────────────────────────────────────────────────────────────
#  One-shot helper
# ─────────────────────────────────────────────────────────────────────────────
def build_graph_from_pdf(pdf_path: str | Path) -> None:
    from preprocess.text_extractor import extract_text_from_file
    from gemini.gemini_client import generate_structured_schema_and_cypher

    pdf_path = Path(pdf_path)
    raw_text = "\n".join(extract_text_from_file(pdf_path))

    result = generate_structured_schema_and_cypher(raw_text)
    cypher_raw = result.get("cypher", [])
    statements = (
        [s.strip() for s in cypher_raw.split(";") if s.strip()]
        if isinstance(cypher_raw, str)
        else list(cypher_raw)
    )

    execute_cypher_batched(sanitise_cypher(statements))
    print("✅ Graph build complete.")


__all__ = [
    "sanitise_cypher",
    "execute_cypher_batched",
    "execute_cypher_queries_individually",
    "build_graph_from_pdf",
]
