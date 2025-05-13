# gemini/gemini_client.py
"""
Re-prompt + hardened JSON handling for Gemini Flash 2.0
"""

import os
import json
from json import JSONDecodeError

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# ─────────────────────────────────────────────────────────────────────────────
#  Gemini basic setup
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())                               # allow .env at repo root
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
model = genai.GenerativeModel(model_name="models/gemini-2.5-flash-preview-04-17")
# ─────────────────────────────────────────────────────────────────────────────
#  Helper: grab first {...} blob (best-effort) from messy output
# ─────────────────────────────────────────────────────────────────────────────
def extract_json(s: str) -> str:
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
            if start is None:
                start = i
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                return s[start : i + 1]
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN – generate hierarchy / schema / Cypher
# ─────────────────────────────────────────────────────────────────────────────
def generate_structured_schema_and_cypher(text: str) -> dict:
    """
    Ask Gemini to turn raw *text* into:
       {
         "hierarchy": ...,
         "schema":    ...,
         "cypher":    [...]
       }
    The prompt contains STRICT syntax+escaping instructions; we then attempt up
    to 3 JSON-decode passes, progressively extracting the innermost braces.
    """

    prompt = f"""
        You are a JSON-only extraction assistant.

        ╭─  OUTPUT CONTRACT  —  ZERO-PROSE MODE  ───────────────────────────╮
        │ • Return **exactly one** JSON object and nothing else.            │
        │ • Your reply **must** parse with `json.loads()` as sent.          │
        │ • Validate before sending:                                        │
        │       parsed = json.loads(reply)                                  │
        │       echo  = json.dumps(parsed, ensure_ascii=False)              │
        │   Send **echo** (not the original draft).                         │
        ╰───────────────────────────────────────────────────────────────────╯

        ── REQUIRED KEYS (no others) ───────────────────────────────────────
        "hierarchy"     : nested outline → list/trees of headings / sections
        "nodes"         : list of {{ "label": safe_identifier, "name": original }}
        "relationships" : list of {{ "subject", "verb", "object", "type", "name" }}
        "leaders"       : top-5 nouns by frequency (fewer if < 5 nouns)
        "cypher"        : array of single-line Cypher MERGE statements

        ── Sanitise **every** label / rel-type ──────────────────────────────
        • Start with a letter.                                               
        • Transform raw token → **safe_identifier**:                         
            – Replace apostrophes (’ or ')  → “_”                            
            – Replace spaces                → “_”                            
            – Strip other punctuation (keep letters, digits, underscores)    
            – If first char is a digit, prepend “N_”.                        
        • Store the unsanitised string in `name`.                            
        Example raw “L’utente” → label `L_utente` with {{ "name": "L’utente" }}

        ── Extraction rules ────────────────────────────────────────────────
        • “hierarchy” = heading / sub-heading structure (max depth ≈ 4).     
        • Subject & object must be in the **same sentence** for a triple.    
        • One entry per unique (subject, verb, object).                      

        ── Cypher rules ────────────────────────────────────────────────────
        • Directed only: (a)-[:TYPE]->(b)  or  (b)<-[:TYPE]-(a)               
        • Identifiers already safe → never use back-ticks.                   
        • Put long text in a `name` property, not in labels/types.           
        • Emit **no** CONSTRAINTS.                                           
        • Target ≥ ⌈nodes ÷ 2⌉ relationships.                                

        ── JSON string escaping (critical) ─────────────────────────────────
        • Escape **only**:   "  →  \\\"      \\  →  \\\\                       
        • No other escapes.  Every Cypher string is one line.                

        ── SELF-CHECK BEFORE SENDING ───────────────────────────────────────
        1. Build the JSON dict.                                              
        2. parsed = json.loads(json.dumps(dict, ensure_ascii=False))         
        3. Send `json.dumps(parsed, ensure_ascii=False)`  **and nothing else**.

        Document text ↓↓↓
        {text}
        """



    # ── Call Gemini with low randomness for stability ───────────────────────
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            #"max_output_tokens": 2048,       # fix upper bound
            "top_p": 1.0, "top_k": 0,        # disable sampling
        },
    )
    payload = response.text.strip()

    # Strip ``` fences if present
    if payload.startswith("```"):
        lines = payload.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        payload = "\n".join(lines).strip()

    # ── Robust JSON decoding: try up to 3 times, each time tightening range ─
    for attempt in range(3):
        try:
            return json.loads(payload)
        except JSONDecodeError as err:
            if attempt == 2:
                print(f"Invalid JSON after 3 attempts: {err}")
                raise ValueError(f"Invalid JSON received from model:\n{payload}") from err
            # extract the *most* JSON-looking substring and retry
            payload = extract_json(payload)


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRA – natural-language narrative (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def generate_semantic_narrative(hierarchy: dict, schema: dict) -> str:
    h = json.dumps(hierarchy, ensure_ascii=False)
    s = json.dumps(schema,    ensure_ascii=False)
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
    return model.generate_content(prompt).text.strip()
