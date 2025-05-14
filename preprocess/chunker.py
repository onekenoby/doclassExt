# preprocess/chunker.py
from __future__ import annotations
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")   # good proxy for Gemini 2.5

def chunk_by_tokens(text: str, max_tokens: int = 3_000) -> list[str]:
    words = text.split()
    out, buf, buf_tokens = [], [], 0

    for w in words:
        w_tokens = max(1, len(w) // 4)          # cheap count
        if buf_tokens + w_tokens > max_tokens:
            out.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(w)
        buf_tokens += w_tokens

    if buf:
        out.append(" ".join(buf))
    return out
