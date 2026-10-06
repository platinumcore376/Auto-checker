import asyncio
import base64
import io
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from PIL import Image

import config
from auth import auth_router, require_user
from Utils.image_similarity import image_similarity, load_clip_model
from Utils.ocr import ocr_from_image
from Utils.segmentation import (
    NoTextDetectedError,
    SegmentationError,
    segment_lines_and_find_diagrams,
)
from Utils.similarity import load_sentence_transformer, text_similarity

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger("autochecker.server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager (S3 / PERF-01).
    Validates auth secrets and warms up ML models once at startup.
    """
    logger.info("Initializing AutoChecker backend...")
    # Validate auth configuration (Phase 4 / SEC-01, SEC-02)
    config.validate_auth_config()

    models_status = {"sentence_transformer": False, "clip": False}
    try:
        load_sentence_transformer()
        models_status["sentence_transformer"] = True
    except Exception as e:
        logger.warning(f"Could not pre-load SentenceTransformer at startup: {e}")

    try:
        load_clip_model()
        models_status["clip"] = True
    except Exception as e:
        logger.warning(f"Could not pre-load OpenCLIP at startup: {e}")

    app.state.models_loaded = models_status
    logger.info(f"Model startup status: {models_status}")
    yield
    logger.info("AutoChecker backend shutting down.")

app = FastAPI(title="AutoChecker Backend", lifespan=lifespan)
app.include_router(auth_router)

# CORS configuration (SEC-04)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "running", "service": "AutoChecker Backend"}

@app.get("/health")
def health_check():
    """Health check reporting model loading status (S3)."""
    return {
        "status": "healthy",
        "models_loaded": getattr(app.state, "models_loaded", {"sentence_transformer": False, "clip": False}),
    }

def _validate_pdf_content(sheet_bytes: bytes, filename: str) -> int:
    """
    Validates file size, PDF magic bytes, and page count (SEC-05).
    Returns total page count.
    """
    if len(sheet_bytes) == 0:
        raise HTTPException(status_code=400, detail=f"Answer sheet '{filename}' is empty.")

    if len(sheet_bytes) > config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Answer sheet '{filename}' exceeds maximum allowed size of {config.MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # Check %PDF magic bytes
    if not sheet_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file '{filename}' is not a valid PDF document (missing %PDF header)."
        )

    # Validate page count using pdfinfo without rasterizing full document
    try:
        info = pdfinfo_from_bytes(sheet_bytes)
        page_count = info.get("Pages", 1)
        if page_count > config.MAX_PAGES_PER_PDF:
            raise HTTPException(
                status_code=422,
                detail=f"Answer sheet '{filename}' has {page_count} pages, exceeding the maximum allowed ({config.MAX_PAGES_PER_PDF} pages)."
            )
        return page_count
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Could not read PDF metadata for {filename}: {e}")
        return 1

@app.post("/similarity")
async def similarity(
    answer_key_text: str = Form(...),
    answer_key_diagram: UploadFile = File(...),
    answer_sheets: List[UploadFile] = File(...),
    current_user: dict = Depends(require_user),
):
    """
    Evaluates student answer sheets against textual and diagrammatic keys.
    Uses request-isolated directories (BUG-01) and safe per-sheet mapping (BUG-03).
    """
    if not answer_key_text.strip():
        raise HTTPException(status_code=400, detail="Answer key text cannot be empty.")

    if len(answer_sheets) > config.MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=422,
            detail=f"Too many files uploaded ({len(answer_sheets)}). Maximum allowed is {config.MAX_FILES_PER_REQUEST}."
        )

    # Process and validate answer key diagram
    diagram_bytes = await answer_key_diagram.read()
    if not diagram_bytes:
        raise HTTPException(status_code=400, detail="Answer key diagram file is empty.")

    try:
        diagram_image = Image.open(io.BytesIO(diagram_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image uploaded for answer key diagram: {str(e)}")

    # Request-isolated working directory (S1 / BUG-01)
    req_work_dir = tempfile.mkdtemp(prefix="autochecker_req_")
    logger.info(f"Created isolated request directory: {req_work_dir}")

    try:
        response_data: Dict[int, list] = {}

        for idx, sheet in enumerate(answer_sheets):
            sheet_folder = os.path.join(req_work_dir, f"sheet_{idx}")
            os.makedirs(sheet_folder, exist_ok=True)

            sheet_bytes = await sheet.read()
            _validate_pdf_content(sheet_bytes, sheet.filename or f"sheet_{idx}.pdf")

            # Convert PDF pages safely using asyncio.to_thread (PERF-02)
            try:
                pages = await asyncio.to_thread(
                    convert_from_bytes,
                    sheet_bytes,
                    first_page=1,
                    last_page=config.MAX_PAGES_PER_PDF,
                )
            except Exception as e:
                logger.error(f"PDF rasterization failed for {sheet.filename}: {e}")
                raise HTTPException(status_code=422, detail=f"Cannot process PDF '{sheet.filename}': {str(e)}")

            if not pages:
                raise HTTPException(status_code=422, detail=f"PDF '{sheet.filename}' contains no readable pages.")

            # Process first page (current architecture processes single-page answer sheet)
            first_page = pages[0].convert("RGB")
            page_np = np.array(first_page)

            # Segmentation & diagram extraction
            try:
                seg_result = await asyncio.to_thread(
                    segment_lines_and_find_diagrams,
                    page_np,
                    output_folder=sheet_folder,
                    sheet_idx=idx,
                )
                segmented_folder = seg_result["segmented_folder"]
                text_crop_path = seg_result["text_crop_path"]
                diagram_path = seg_result["diagram_path"]
            except NoTextDetectedError:
                logger.warning(f"No text lines found in {sheet.filename}")
                seg_result = {"segmented_folder": None, "text_crop_path": None, "diagram_path": None}
                segmented_folder, text_crop_path, diagram_path = None, None, None

            # OCR processing
            sheet_text = ""
            if segmented_folder and os.path.exists(segmented_folder):
                try:
                    sheet_text = await asyncio.to_thread(ocr_from_image, segmented_folder)
                except Exception as e:
                    logger.warning(f"OCR failed on sheet {idx}: {e}")
                    sheet_text = ""

            # Text similarity scoring
            if sheet_text:
                text_sim = await asyncio.to_thread(
                    text_similarity,
                    answer_key_text,
                    sheet_text,
                )
            else:
                text_sim = 0.0

            # Diagram similarity scoring (BUG-03, BUG-08)
            diagram_score = 0.0
            encoded_diagram = ""
            if diagram_path and os.path.exists(diagram_path):
                try:
                    with Image.open(diagram_path) as d_img:
                        d_img_rgb = d_img.convert("RGB")
                        sim_scores = await asyncio.to_thread(
                            image_similarity,
                            diagram_image,
                            [d_img_rgb],
                        )
                        if len(sim_scores) > 0:
                            diagram_score = float(sim_scores[0])

                        buf = io.BytesIO()
                        d_img_rgb.save(buf, format="PNG")
                        encoded_diagram = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception as e:
                    logger.error(f"Diagram similarity calculation failed for sheet {idx}: {e}")
                    diagram_score = 0.0
                    encoded_diagram = ""

            # Text crop base64 encoding
            encoded_text_crop = ""
            if text_crop_path and os.path.exists(text_crop_path):
                try:
                    with Image.open(text_crop_path) as t_img:
                        buf = io.BytesIO()
                        t_img.save(buf, format="PNG")
                        encoded_text_crop = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception as e:
                    logger.warning(f"Failed to encode text crop for sheet {idx}: {e}")
                    encoded_text_crop = ""

            response_data[idx] = [
                float(text_sim),
                float(diagram_score),
                sheet_text,
                encoded_diagram,
                encoded_text_crop,
            ]

        return JSONResponse(content=response_data)

    finally:
        # Clean up temporary request directory (BUG-01)
        shutil.rmtree(req_work_dir, ignore_errors=True)
        logger.info(f"Cleaned up request directory: {req_work_dir}")
