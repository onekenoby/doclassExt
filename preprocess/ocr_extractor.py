import pytesseract
from PIL import Image

def ocr_from_image(image_path_or_obj):
    """
    Extract text from an image via OCR.
    """
    if isinstance(image_path_or_obj, str):
        image = Image.open(image_path_or_obj)
    else:
        image = image_path_or_obj
    return pytesseract.image_to_string(image)