# preprocess/text_extractor.py
"""
Fast text-extraction helper.

Key points
──────────
• **Single-pass** parsing of each page – no redundant full-doc
  `page.get_text()` call.
• **Per-page OCR fallback** – we render an image *only* for pages whose
  text layer is empty.
• Works for both PDF and DOCX; DOCX branch unchanged.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import List

import fitz                          # PyMuPDF
import docx
from preprocess.ocr_extractor import ocr_from_image


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_file(filepath: str | Path) -> List[str]:
    mime, _ = mimetypes.guess_type(str(filepath))

    if mime == "application/pdf":
        return _extract_from_pdf(Path(filepath))
    elif mime in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                  "application/msword"}:
        return _extract_paragraphs_from_docx(Path(filepath))

    raise ValueError(f"Unsupported file type: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
#  PDF branch – single pass + on-demand OCR
# ─────────────────────────────────────────────────────────────────────────────
def _extract_from_pdf(pdf_path: Path) -> List[str]:
    doc = fitz.open(pdf_path)
    paragraphs: list[str] = []

    for page in doc:                                 # loop once
        blocks = page.get_text("blocks")
        if blocks:                                   # normal text layer
            for _x0, _y0, _x1, _y1, text, *_ in blocks:
                t = text.strip()
                if t:
                    paragraphs.append(t)
        else:                                        # fallback OCR (this page only)
            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                ocr_text = ocr_from_image(pix.tobytes())
                t = ocr_text.strip()
                if t:
                    paragraphs.append(t)

    return paragraphs


# ─────────────────────────────────────────────────────────────────────────────
#  DOCX branch – unchanged
# ─────────────────────────────────────────────────────────────────────────────
def _extract_paragraphs_from_docx(docx_path: Path) -> List[str]:
    doc = docx.Document(docx_path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
