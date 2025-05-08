# preprocess/text_extractor.py
import fitz  # PyMuPDF
import docx
import mimetypes
from pdf2image import convert_from_path
from preprocess.ocr_extractor import ocr_from_image


def extract_text_from_file(filepath):
    """
    Extract paragraphs from PDF/DOCX, fallback to OCR if needed.
    Returns list of paragraph strings.
    """
    mime = mimetypes.guess_type(filepath)[0]
    if mime == 'application/pdf':
        doc = fitz.open(filepath)
        text = " ".join([page.get_text() for page in doc])
        if not text.strip():
            images = convert_from_path(filepath)
            text = " ".join([ocr_from_image(img) for img in images])
    else:
        return extract_paragraphs_from_docx(filepath)

    paragraphs = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            t = b[4].strip()
            if t:
                paragraphs.append(t)
    return paragraphs


def extract_paragraphs_from_docx(filepath):
    doc = docx.Document(filepath)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
