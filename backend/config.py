import os
from typing import List, Optional

# ==============================================================================
# Model Configurations
# ==============================================================================
SENTENCE_TRANSFORMER_MODEL: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "stsb-roberta-large")
# Optional HuggingFace commit hash / revision for reproducibility (ML-MODEL-01)
SENTENCE_TRANSFORMER_REVISION: Optional[str] = os.getenv("SENTENCE_TRANSFORMER_REVISION", None)

CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED: str = os.getenv("CLIP_PRETRAINED", "openai")

# ==============================================================================
# Scoring Configurations
# ==============================================================================
# Rounding mode for text similarity: "ceil" (default, rounds up to nearest 0.1) or "round"
SCORE_ROUNDING: str = os.getenv("SCORE_ROUNDING", "ceil")

# OpenCLIP score transform constants (image_similarity.py):
# Values >= 0.9 represent high visual alignment and remain unchanged.
# Values < 0.9 are aggressively scaled down using 0.5 * (x / 0.85) to penalize inaccurate sketches.
CLIP_SCORE_THRESHOLD: float = 0.9
CLIP_SCALE_FACTOR: float = 0.5
CLIP_DIVISOR: float = 0.85

# ==============================================================================
# Upload & Security Limits (SEC-05)
# ==============================================================================
# Maximum file size per uploaded PDF (default 20 MB)
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", 20 * 1024 * 1024))

# Maximum pages rasterized per PDF (default 30 pages)
MAX_PAGES_PER_PDF: int = int(os.getenv("MAX_PAGES_PER_PDF", 30))

# Maximum answer sheets accepted per similarity request (default 50)
MAX_FILES_PER_REQUEST: int = int(os.getenv("MAX_FILES_PER_REQUEST", 50))

# Allowed CORS origins (SEC-04)
_raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS: List[str] = [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]

# Logging Level (Roadmap #11)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ==============================================================================
# Segmentation & Computer Vision Parameters
# ==============================================================================
SEGMENTATION_MIN_HEIGHT: int = 30
SEGMENTATION_PADDING: int = 10
SEGMENTATION_MIN_CONTOUR_WIDTH: int = 5000
HISTOGRAM_LINE_THRESHOLD: int = 50000
DIAGRAM_MIN_WIDTH: int = 40
DIAGRAM_MIN_HEIGHT: int = 400
TEXT_CROP_PADDING: int = 40
