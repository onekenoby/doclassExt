"""
Gemini client – robust JSON extraction, automatic retries,
rate-gate, shared chat, and your exact strict prompt.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from google.api_core.exceptions import ResourceExhausted

from preprocess.chunker import chunk_by_tokens
from utils.rate_gate import RateGate

# ─────────────────────────  environment / model  ──────────────────────────
load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#gemini-2.5-flash-preview-04-17   gemini-2.5-pro-preview-05-06
MODEL_NAME = "models/gemini-2.5-flash-preview-04-17" 
model = genai.GenerativeModel(model_name=MODEL_NAME)
_chat = model.start_chat()                       # shared session

_gate = RateGate(rate_per_sec=2.0)               # adjust to your quota

# ─────────────────────────────  helpers  ───────────────────────────────────
def extract_json(text: str) -> str:
    "Return the first {...} JSON block found in *text*."
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
            if start is None:
                start = i
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return text


def _safe_json_loads(raw: str) -> Any:
    """
    Tolerant JSON loader – tries strict JSON, then fixes stray back-slashes,
    then json5 (if installed) for trailing commas / single quotes.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    try:
        import json5                     # optional dependency
        return json5.loads(raw)          # type: ignore[attr-defined]
    except Exception:
        raise                            # propagate last error


def _retry_chat(prompt: str, *, temperature: float = 0.0):
    """
    Send *prompt* to Gemini with exponential back-off on 429 errors.
    """
    last_err: Exception | None = None
    while True:                          # retry indefinitely on 429
         _gate.wait()
         try:
             return _chat.send_message(
                 prompt,
                generation_config={
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": 0,
                },
            )
         except ResourceExhausted as err:
             delay = getattr(err, "retry_delay", None)
             delay_s = delay.seconds if delay else 8
             print(f"⏳ 429 – sleeping {delay_s}s …")
             time.sleep(delay_s)
             continue   # loop until success


# ─────────────────  strict-prompt JSON extraction (single attempt) ─────────
def _extract_json_once(text: str, *, max_attempts: int = 4) -> Dict:
    """
    One pass (with up to *max_attempts* re-prompts) to get valid JSON.
    """
    base_prompt = f"""
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
• Nodes must be unique. Any *node replication* absolutely has allowed.
After building the JSON:
   1. parsed = json.loads(reply)
   2. echo   = json.dumps(parsed, ensure_ascii=False)
   3. Send **echo** – and nothing else.

Document text ↓↓↓
{text}
"""

    for attempt in range(max_attempts):
        prompt = base_prompt if attempt == 0 else (
            "⚠️  The previous reply was invalid JSON. "
            "Please resend the **complete** JSON object only — no markdown.\n\n"
            + base_prompt
        )

        response = _retry_chat(prompt, temperature=0.0)
        payload = response.text.strip()

        if payload.startswith("```"):
            lines = payload.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            payload = "\n".join(lines).strip()

        payload_body = extract_json(payload)
        if not payload_body.strip():
            err = ValueError("Empty payload")
        else:
            try:
                return _safe_json_loads(payload_body)
            except Exception as e:      # json / json5 errors
                err = e  # <- Fix: assign to `err`
                pass

        print(f"↩️  JSON parse failed (attempt {attempt+1}/{max_attempts}): {err}")
        if attempt == max_attempts - 1:
            raise


# ─────────────────────  recursive fallback (split & merge)  ────────────────
def _with_fallback(text: str, depth: int = 0, max_depth: int = 5, *, max_attempts: int = 4) -> Dict:
    if depth > 1:
        raise RuntimeError("Max split depth reached")

    try:
        return _extract_json_once(text, max_attempts=max_attempts)
    except Exception as err:
        if len(text.split()) < 120:      # too small already
            raise err
        print("🔪  Splitting failed chunk in half and retrying …")
        halves = chunk_by_tokens(text, max_tokens=len(text.split()) // 2)
        merged = {k: [] for k in ("hierarchy", "nodes", "relationships", "leaders", "schema", "cypher")}
        for part in halves:
            try:
                sub = _with_fallback(part, depth + 1, max_depth, max_attempts=max_attempts)
                for k in merged:
                    merged[k] += sub.get(k, [])
            except Exception as e:
                print(f"✖️  Sub-chunk still failed: {e}")
        return merged



# ─────────────────────  PUBLIC API for main.py  ────────────────────────────
def generate_structured_schema_and_cypher(text: str) -> Dict:
    "Main entry – returns dict with hierarchy, schema, cypher, …"
    return _with_fallback(text)


def generate_semantic_narrative(
    hierarchy: Dict | None,
    schema: Dict | None,
    *,
    temperature: float = 0.0
) -> str:
    """Return an Italian prose narrative of the graph (with 429 handling)."""
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

    response = _retry_chat(prompt, temperature=temperature)
    return response.text.strip()
