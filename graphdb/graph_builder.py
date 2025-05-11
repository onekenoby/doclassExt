# graphdb/graph_builder.py
"""
Neo4j helper with safety-net sanitiser for incoming Cypher.
"""

import os
import re
from pathlib import Path
from typing import List

from neo4j import GraphDatabase, exceptions

# ── Neo4j connection (override via env) ──────────────────────────────────────
URI      = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
driver   = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cypher sanitiser  (arrow + numeric-label fixes)
# ─────────────────────────────────────────────────────────────────────────────
INVALID_LABEL = re.compile(r":([0-9][A-Za-z0-9_]*)")
UNDIRECTED_REL = re.compile(r"-\s*\[:([A-Z_]+)\]\s*-(\s*\()")

def sanitise_cypher(statements: List[str]) -> List[str]:
    """
    • Insert a default '->' arrow on undirected relationships produced by LLMs.
    • Back-tick labels that start with a digit (e.g. :3A_Informatica → :`3A_Informatica`)
    """
    fixed: list[str] = []
    for stmt in statements:
        s = UNDIRECTED_REL.sub(r"-[:\1]->\2", stmt)                   # force arrow →
        s = INVALID_LABEL.sub(lambda m: f":`{m.group(1)}`", s)        # back-tick digits
        fixed.append(s)
    return fixed

# ─────────────────────────────────────────────────────────────────────────────
# 2. Low-level executor (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def execute_cypher_queries_individually(statements: List[str]) -> None:
    """
    Run each Cypher statement in its own transaction.
    """
    with driver.session() as session:
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                session.run(stmt)
                print(f"✅ Executed: {stmt[:80]}...")
            except exceptions.CypherSyntaxError as err:
                print(f"⚠️ Syntax error in:\n{stmt}\n→ {err.message}\n")
            except exceptions.Neo4jError as err:
                print(f"⚠️ Neo4j error:\n{stmt}\n→ {err.message}\n")
            except Exception as err:
                print(f"❌ Unexpected error:\n{stmt}\n→ {err}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. One-shot helper (unchanged, but call sanitiser)
# ─────────────────────────────────────────────────────────────────────────────
def build_graph_from_pdf(pdf_path: str | Path) -> None:
    from preprocess.text_extractor import extract_text_from_file
    from gemini.gemini_client import generate_structured_schema_and_cypher

    pdf_path = Path(pdf_path)
    text = "\n".join(extract_text_from_file(str(pdf_path)))

    result = generate_structured_schema_and_cypher(text)
    raw = result.get("cypher", [])
    statements = (
        [s.strip() for s in raw.split(";") if s.strip()]
        if isinstance(raw, str)
        else list(raw)
    )

    execute_cypher_queries_individually(sanitise_cypher(statements))
    print("✅ Graph build complete.")

__all__ = ["sanitise_cypher", "execute_cypher_queries_individually", "build_graph_from_pdf"]
