# graphdb/graph_builder.py
"""Utilities to push Gemini‑generated Cypher into Neo4j.

✓   Uses `preprocess.text_extractor.extract_text_from_file` (no more missing
    symbol).
✓   Keeps full semicolon‑terminated statements intact so variables declared in
    node patterns are still in scope when the relationship pattern appears on
    the same line.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions
from preprocess.text_extractor import extract_text_from_file  # ✅ fixed import

# ──────────────────────────────────────────────────────────────
# Environment & connection
# ──────────────────────────────────────────────────────────────

# Locate and load the project‑root .env
env_path = Path(__file__).resolve().parents[1] / ".env"
print(f"[DEBUG] Loading .env from {env_path}")
load_dotenv(env_path, override=True)

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
print(f"[DEBUG] Neo4j → URI={URI!r}, USER={USER!r}, PASS_LOADED={bool(PASSWORD)}")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# ──────────────────────────────────────────────────────────────
# Pipeline: PDF → text → Gemini → Cypher → Neo4j
# ──────────────────────────────────────────────────────────────

from gemini.gemini_client import generate_structured_schema_and_cypher  # late import to avoid heavy deps on load

def build_graph_from_pdf(pdf_path: str | Path) -> None:
    """End‑to‑end ingestion of *pdf_path* into Neo4j."""
    pdf_path = Path(pdf_path)

    # 1️⃣ Extract raw document text (paragraph list ➜ string)
    paragraphs: List[str] = extract_text_from_file(str(pdf_path))
    doc_text = "\n".join(paragraphs)

    # 2️⃣ Gemini: hierarchy, schema, cypher
    result = generate_structured_schema_and_cypher(doc_text)
    cypher_script = "\n".join(result.get("cypher", []))

    # 3️⃣ Push to Neo4j
    execute_cypher_queries(cypher_script)

    # 4️⃣ ...


def execute_cypher_queries(cypher_script: str) -> None:
    """Execute all Cypher statements in *cypher_script* against Neo4j."""

    statements = cypher_script.strip().split(";\n")
    with driver.session() as session:
        for stmt in statements:
            try:
                session.run(stmt)
            except exceptions.CypherSyntaxError as e:
                print(f"⚠️ Cypher syntax error:\n{stmt}\n→ {e.message}\n")
            except exceptions.Neo4jError as e:
                print(f"⚠️ Neo4j error:\n{stmt}\n→ {e.message}\n")
