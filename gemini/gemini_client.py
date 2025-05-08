# gemini/gemini_client.py

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from json import JSONDecodeError
from google.generativeai import types

# Load project‑root .env so that GEMINI_API_KEY loads correctly
load_dotenv(find_dotenv())

# ────────────────  Gemini configuration  ────────────────
# You can switch model versions here if needed (e.g. "models/gemini-1.5-pro-latest")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")


############################################
# Utility helper to grab the first JSON blob
############################################

def extract_json(s: str) -> str:
    """Return the first balanced‑brace JSON object found inside *s*."""
    start = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == '{':
            if start is None:
                start = i
            depth += 1
        elif ch == '}' and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return s[start: i + 1]
    return s


###############################################################
# 1️⃣  MAIN ENTRY – STRUCTURED KG & CYPHER GENERATION PROMPT  #
###############################################################

def generate_structured_schema_and_cypher(text: str) -> dict:
    """Given raw document text, produce a rich JSON spec and Cypher script.

    The prompt below is engineered to maximise **coverage** and **granularity**
    of the resulting knowledge‑graph while keeping the output machine‑parsable.
    """
    prompt = f"""
    You are an expert **knowledge‑graph architect** and **triple extractor**.
    Your goal is to convert the following document into a *dense* and *coherent*
    knowledge graph, surfacing as many meaningful **entities** (nodes) and
    **relationships** (edges) as the text reasonably supports.  
    Aim for high *recall* while keeping *precision* acceptable (≥ 0.6).

    **Mandatory:** Ensure **all relationships** are set in the Graph Database Schema, as visible with the `call db.schema.visualization()` command.

    Return **one** valid JSON object with **exactly** these keys:

    1.  \"hierarchy\"  – a recursive outline capturing topical structure.
    2.  \"schema\" – describes the graph model.
    3.  \"cypher\" – **array** of Cypher statements.

    --- 

    ##  Extraction guidelines
    • Create **separate nodes** for distinct real‑world entities: persons, orgs, locations, events, concepts, dates, numerical facts, URLs, etc.  
    • **Identify and extract ALL relationships** among the nodes, whether **explicit** or **implicit**.
    • **Mandatory balance** – The graph must contain at least ⌈nodes ÷ 2⌉ relationship statements.
    • Relationships **must** be emitted even if they're inferred or subtle (use `NO_RELATIONSHIP` for placeholders).

    Document Text ↓↓↓
    {text}
    """

    # === Call the model ===
    response = model.generate_content(prompt)
    payload = response.text.strip()

    # ── Strip possible ``` fences ──────────────────────────────
    if payload.startswith("```"):
        lines = payload.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        payload = "\n".join(lines).strip()

    # ── Parse the JSON or fall back to best‑effort extraction ──
    try:
        return json.loads(payload)
    except JSONDecodeError:
        snippet = extract_json(payload)
        try:
            return json.loads(snippet)
        except JSONDecodeError:
            raise ValueError(f"Invalid JSON received from model:\n{payload}")


################################################
# 2️⃣  NATURAL‑LANGUAGE NARRATIVE (unchanged)   #
################################################

def generate_semantic_narrative(hierarchy: dict, schema: dict) -> str:
    """Italian narrative summary of the graph structure for UI/tooltips."""
    h = json.dumps(hierarchy, ensure_ascii=False)
    s = json.dumps(schema, ensure_ascii=False)
    prompt = f"""
Sei un esperto di knowledge graph. Usa la gerarchia e lo schema seguenti per
produrre una narrazione fluida ed avvincente (in italiano) che spieghi cosa
rappresenta il grafo, evidenzi i concetti chiave e le relazioni più
significative.

Gerarchia:
{h}

Schema:
{s}

Risposta:
"""
    response = model.generate_content(prompt)
    narrative = response.text.strip()
    if narrative.startswith("```") and narrative.endswith("```"):
        narrative = narrative.strip("```").strip()
    return narrative