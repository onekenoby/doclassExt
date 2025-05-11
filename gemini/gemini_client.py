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
Given raw document text, produce a *single* JSON object with three keys:
 1. "hierarchy"   – nested outline of the content
 2. "schema"      – {{ "node_labels": [...], "relationship_types": [...] }}
 3. "cypher"      – **array** of Cypher CREATE / MERGE statements

### Cypher rules
- **Directed only** → in CREATE/MERGE you *must* use an arrow, e.g.
      (a)-[:REL_TYPE]->(b)   or   (b)<-[:REL_TYPE]-(a)
  Undirected `-[:REL]-` is forbidden.
- Node labels and relationship types must start with a letter.
  If the source name begins with a digit, either wrap it in back-ticks
  (``:`3A_Informatica`` → :`3A_Informatica`) *or* prepend a letter
  (e.g. 3A_Informatica → A3A_Informatica).
- Map verbose names/descriptions to *properties* (usually `name` or
  `description`) rather than in the label/type.
- Do **NOT** emit CONSTRAINTS.
- Aim for at least ⌈nodes ÷ 2⌉ relationships.

### JSON escaping (critical)
Inside the JSON you return **every back-slash must be doubled (\\\\)**.
That includes the back-slashes the Cypher engine itself needs:
- `\\`  → `\\\\`
- `\"`  → `\\\"`
- new-line → `\\\\n`, carriage return → `\\\\r`, etc.

Document text ↓↓↓
{text}
"""

    # ── Call Gemini with low randomness for stability ───────────────────────
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            ## "max_output_tokens": 2048,       # fix upper bound
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
