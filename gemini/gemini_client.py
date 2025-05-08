# gemini/gemini_client.py

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from json import JSONDecodeError

# Load project‑root .env so that GEMINI_API_KEY loads correctly
load_dotenv(find_dotenv())

# ──────────────── Gemini configuration ────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Initialize the GenerativeModel.
# The temperature is set in the generate_content call for flexibility.
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
# 1️⃣ MAIN ENTRY – STRUCTURED KG & CYPHER GENERATION PROMPT  #
###############################################################

def generate_structured_schema_and_cypher(text: str) -> dict:
    """Generate a structured schema and Cypher script from raw document text."""

    # Tip: For very large documents, consider splitting the 'text' into smaller
    # chunks and processing each chunk separately. You would then need to
    # merge the results (hierarchy, schema, and cypher) from each chunk.
    # This is a more advanced modification. Adjusting temperature is a good
    # first step for improving stability.

    prompt = f"""
    Given raw document text, produce a rich JSON spec and Cypher script.

    The prompt below is engineered to maximise **coverage** and **granularity**
    of the resulting knowledge‑graph while keeping the output machine-parsable.
    Aim for high *recall* while keeping *precision* acceptable (≥ 0.6).

    **Crucially, prioritize valid Cypher syntax.** Follow these rules:
    - Use CREATE statements to create nodes and relationships.
    - Node creation: CREATE (node_variable:NodeType {{property1: value1, property2: value2, ...}})
    - Relationship creation: CREATE (node1)-[:RELATIONSHIP_TYPE {{property1: value1, ...}}]->(node2)
    - Always provide node labels (e.g., :Person, :Organization, :Document).
    - Always provide relationship types (e.g., :RELATES_TO, :PART_OF, :HAS_PART).
    - Do NOT include CREATE CONSTRAINT statements. These are handled separately.
    - Ensure all strings are properly escaped if necessary.
    - Return Cypher statements as a JSON array of strings.

    Return **one** valid JSON object with **exactly** these keys:

    1.  \"hierarchy\"  – a recursive outline capturing topical structure.
    2.  \"schema\" – describes the graph model.
    3.  \"cypher\" – **array** of Cypher statements.

    ---

    ## Extraction guidelines
    • Create **separate nodes** for distinct real‑world entities: persons, orgs, locations, events, concepts, dates, numerical facts, URLs, etc.
    • **Identify and extract ALL relationships** among the nodes, whether **explicit** or **implicit**.
    • **Mandatory balance** – The graph must contain at least ⌈nodes ÷ 2⌉ relationship statements.
    • Relationships **must** be emitted even if they're inferred or subtle (use `NO_RELATIONSHIP` for placeholders).

    Document Text ↓↓↓
    {text}
    """

    try:
        # Set a lower temperature for more stable and deterministic results.
        # A value closer to 0 reduces randomness. Experiment with values
        # like 0.0, 0.1, 0.2 etc.
        response = model.generate_content(prompt, generation_config={"temperature": 0.1})
        payload = response.text.strip()

        # ── Strip possible ``` fences ──────────────────────────────
        # The model might wrap the JSON in markdown code blocks
        if payload.startswith("```"):
            lines = payload.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            payload = "\n".join(lines).strip()

        # ── Parse the JSON or fall back to best‑effort extraction ──
        # Attempt to parse the direct output, or extract JSON if embedded
        try:
            return json.loads(payload)
        except JSONDecodeError:
            snippet = extract_json(payload)
            try:
                return json.loads(snippet)
            except JSONDecodeError:
                # If JSON extraction also fails, raise an error
                raise ValueError(f"Invalid JSON received from model:\n{payload}")
        except Exception as e:
            # Catch any other exceptions during JSON processing
            print(f"Error during JSON processing: {e}")
            raise # Re-raise the exception to be handled by the caller
    except Exception as e:
        # Catch exceptions during the API call itself
        print(f"ERROR: Gemini API call failed: {e}")
        raise  # Re-raise the exception to be handled by the caller


################################################
# 2️⃣ NATURAL‑LANGUAGE NARRATIVE (unchanged)   #
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
    # Temperature can also be adjusted for the narrative generation if needed.
    # For creative text like a narrative, a slightly higher temperature might
    # be desirable, but for consistency, you might keep it low here too.
    # We will leave it at the default for now as it's less critical for graph stability.
    response = model.generate_content(prompt)
    narrative = response.text.strip()
    if narrative.startswith("```") and narrative.endswith("```"):
        narrative = narrative.strip("```").strip()
    return narrative