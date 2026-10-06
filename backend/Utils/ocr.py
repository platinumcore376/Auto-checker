import io
import os
import re
import logging
from PIL import Image
try:
    import google.generativeai as genai  # Retained per plan; not configured at runtime
except ImportError:
    genai = None
from google.cloud import vision

logger = logging.getLogger("autochecker.ocr")

def _extract_sort_index(filename: str) -> int:
    """
    Safely extracts an integer index from a filename like line_1.png or segment_12.png.
    Falls back gracefully to infinity if no integer is found to prevent sorting crashes (BUG-07).
    """
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def ocr_from_image(segmented_folder: str) -> str:
    """
    Performs OCR on all PNG line images in the segmented folder and returns concatenated text.
    Handles empty folders and sorts files robustly using regex.
    """
    if not os.path.exists(segmented_folder):
        logger.warning(f"Segmented folder does not exist: {segmented_folder}")
        return ""

    client = vision.ImageAnnotatorClient()
    full_text = ""

    filenames = [f for f in os.listdir(segmented_folder) if f.lower().endswith(".png")]
    filenames.sort(key=_extract_sort_index)

    logger.debug(f"Processing {len(filenames)} segmented line images in {segmented_folder}")

    for filename in filenames:
        image_path = os.path.join(segmented_folder, filename)
        try:
            with Image.open(image_path) as image:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                content = img_byte_arr.getvalue()

            gcv_image = vision.Image(content=content)
            response = client.document_text_detection(image=gcv_image)

            if response.error.message:
                raise RuntimeError(f"Google Cloud Vision API error: {response.error.message}")

            if response.full_text_annotation and response.full_text_annotation.text:
                full_text += response.full_text_annotation.text.strip() + " "
        except Exception as e:
            logger.error(f"Error processing OCR on {filename}: {str(e)}")
            raise

    return full_text.strip()
