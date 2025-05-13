"""
Gemini client – robust JSON extraction + shared-chat speed-up
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# ──────────────────────────────  environment  ──────────────────────────────
load_dotenv(find_dotenv())                        # reads .env if present
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "models/gemini-2.5-flash-preview-04-17"
model = genai.GenerativeModel(model_name=MODEL_NAME)
_chat = model.start_chat()                        # re-used by both functions

# ───────────────────────────────  helpers  ─────────────────────────────────
def extract_json(s: str) -> str:
    """Return first {...} block found; crude but often recovers truncated output."""
    depth, start = 0, None
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


def _safe_json_loads(raw: str) -> Any:
    """
    More tolerant JSON loader that tries:
      • strict json.loads
      • auto-doubling stray back-slashes
      • json5 (if installed) for single quotes / trailing commas
    Raises the last JSONDecodeError if all attempts fail.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # fix isolated back-slashes  e.g.  "c:\path" → "c:\\path"
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    try:
        import json5                   # optional dependency
        return json5.loads(raw)        # type: ignore[attr-defined]
    except Exception:
        # propagate last strict-JSON error so caller can decide
        raise


# ──────────────────  LLM ­→ hierarchy / schema / Cypher  ──────────────────
def generate_structured_schema_and_cypher(text: str) -> dict:
    """
    Returns a dict with keys:
      hierarchy • nodes • relationships • leaders • schema • cypher
    The prompt below matches the original one you provided—no wording changed.
    """

    prompt = f"""
You are a JSON-only extraction assistant.

╭─  OUTPUT CONTRACT  —  ZERO-PROSE MODE  ───────────────────────────╮
│ • Return **exactly one** JSON object and nothing else.            │
│ • Your reply **must** parse with `json.loads()` as sent.          │
│ • Validate before sending:                                        │
│       parsed = json.loads(reply)                                  │
│       echo   = json.dumps(parsed, ensure_ascii=False)             │
│   Send **echo** (not the original draft).                         │
╰───────────────────────────────────────────────────────────────────╯

── REQUIRED KEYS ───────────────────────────────────────────────────
"hierarchy"     : nested outline → list / tree of headings / sections
"nodes"         : list of {{ "label": safe_identifier, "name": original }}
"relationships" : list of {{ "subject", "verb", "object",
                             "type", "name" }}
"leaders"       : top-5 nouns by frequency (fewer if < 5 nouns)
"schema"        : list of node- and relationship-types with properties
"cypher"        : **array** of Cypher statements that will recreate
                  the graph in Neo4j.

── CONSTRAINTS ─────────────────────────────────────────────────────
• Every identifier in Cypher must be a valid Neo4j identifier.
• Escape anything that begins with a digit or contains spaces
  using back-ticks:  `Bad Identifier 1`.
• Relationships must be directed →   (a)-[:TYPE]->(b)
• Do **not** wrap the final JSON in triple-back-ticks.

After building the JSON:
   1. parsed = json.loads(reply)
   2. echo   = json.dumps(parsed, ensure_ascii=False)
   3. Send **echo** – and nothing else.

Document text ↓↓↓
{text}
"""

    response = _chat.send_message(
        prompt,
        generation_config={
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 0,
        },
    )
    payload: str = response.text.strip()

    # strip ``` fences if Gemini wrapped them
    if payload.startswith("```"):
        lines = payload.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        payload = "\n".join(lines).strip()

    # sometimes model echoes markdown like “**JSON:**”; keep only the braces
    payload = extract_json(payload)

    return _safe_json_loads(payload)


# ──────────────────────────  narrative generator  ──────────────────────────
def generate_semantic_narrative(
    hierarchy: dict | None,
    schema: dict | None,
    temperature: float = 0.2,
) -> str:
    """Return an Italian prose summary of the graph, or a warning string."""
    if not hierarchy or not schema:
        return "⚠️  Impossibile generare la narrazione: schema o gerarchia mancanti."

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
""".strip()

    response = _chat.send_message(
        prompt,
        generation_config={
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
        },
    )
    return response.text.strip()
