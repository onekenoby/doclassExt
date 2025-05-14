# preprocess/text_processor.py
from __future__ import annotations
import re
from collections import Counter
from typing import List

BULLET = re.compile(r"^[\u2022\u2023\u25E6\u2043•\-\*]\s+")

def _dedupe_headers_footers(lines: List[str], max_fraction: float = 0.2) -> List[str]:
    """
    Remove repeating page headers / footers: keep any line that appears
    on > max_fraction of the pages ONLY once overall.
    """
    freq = Counter(lines)
    cutoff = max_fraction * len(lines)
    return [ln for ln in lines if freq[ln] < cutoff]

def _fix_hyphenation(lines: List[str]) -> List[str]:
    """Join lines that were hyphen-broken across PDF line wraps."""
    out: list[str] = []
    buf = ""
    for ln in lines:
        if ln.endswith("-") and not ln.endswith("--"):
            buf += ln[:-1]        # drop hyphen
        else:
            out.append(buf + ln)
            buf = ""
    if buf:
        out.append(buf)
    return out

def preprocess_paragraphs(paragraphs: List[str]) -> str:
    # 1) flatten and trim
    lines = [ln.strip() for para in paragraphs for ln in para.splitlines()]
    lines = [ln for ln in lines if ln]

    # 2) kill headers/footers that repeat on many pages
    lines = _dedupe_headers_footers(lines)

    # 3) repair hyphenated words
    lines = _fix_hyphenation(lines)

    # 4) normalise bullets → sentences
    norm: list[str] = []
    for ln in lines:
        if BULLET.match(ln):
            ln = BULLET.sub("", ln)
        norm.append(ln)

    # 5) squeeze multiple spaces / weird unicode
    text = re.sub(r"\s+", " ", " ".join(norm))
    text = text.replace("\u00A0", " ")          # NBSP → normal space
    return text.strip()
