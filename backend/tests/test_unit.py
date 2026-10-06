import os
import tempfile
import torch
import numpy as np
import pytest
from fastapi import HTTPException

import config
from Utils.image_similarity import transform, image_similarity
from Utils.ocr import _extract_sort_index
from Utils.similarity import text_similarity
from Utils.segmentation import (
    correct_tilt,
    segment_lines_and_find_diagrams,
    NoTextDetectedError,
)
from server import _validate_pdf_content

# ==============================================================================
# 1. Score Transform Unit Tests
# ==============================================================================
def test_clip_score_transform_high_match():
    """Values >= 0.9 should remain untouched."""
    scores = torch.tensor([0.90, 0.95, 1.0])
    transformed = transform(scores)
    assert torch.allclose(transformed, scores)

def test_clip_score_transform_low_match():
    """Values < 0.9 should be scaled down significantly via 0.5 * (x / 0.85)."""
    score = torch.tensor([0.85])
    transformed = transform(score)
    expected = 0.5 * (0.85 / 0.85)  # 0.5
    assert pytest.approx(transformed.item(), 0.001) == expected

    zero_score = torch.tensor([0.0])
    assert transform(zero_score).item() == 0.0

# ==============================================================================
# 2. Empty Diagram List Guard (BUG-08)
# ==============================================================================
def test_image_similarity_empty_list():
    """Empty diagram list must return an empty list, not crash with RuntimeError."""
    truth_img = MagicMock = object()
    result = image_similarity(truth_img, [])
    assert result == []

# ==============================================================================
# 3. OCR Filename Regex Sorting (BUG-07)
# ==============================================================================
def test_ocr_sort_index_natural_ordering():
    """Filenames must be sorted numerically (line_1, line_2, ... line_10)."""
    filenames = ["line_10.png", "line_1.png", "line_2.png", "segment_99.png"]
    sorted_files = sorted(filenames, key=_extract_sort_index)
    assert sorted_files == ["line_1.png", "line_2.png", "line_10.png", "segment_99.png"]

def test_ocr_sort_index_unusual_names():
    """Filenames without digits should not crash and sort to the end."""
    filenames = ["unknown.png", "line_5.png", "other_test.png"]
    sorted_files = sorted(filenames, key=_extract_sort_index)
    assert sorted_files[0] == "line_5.png"

# ==============================================================================
# 4. Text Similarity Rounding Modes (ML-SCORE-02)
# ==============================================================================
def test_text_similarity_ceil_rounding(monkeypatch):
    """Ceil rounding rounds up to nearest 0.1."""
    monkeypatch.setattr(config, "SCORE_ROUNDING", "ceil")
    sim = text_similarity("answer key", "student answer")
    assert sim in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def test_text_similarity_round_rounding(monkeypatch):
    """Round mode preserves standard 2 decimal places."""
    monkeypatch.setattr(config, "SCORE_ROUNDING", "round")
    sim = text_similarity("answer key", "student answer")
    assert isinstance(sim, float)

def test_text_similarity_empty_inputs():
    """Empty strings return 0.0 without running models."""
    assert text_similarity("", "test") == 0.0
    assert text_similarity("test", "") == 0.0

# ==============================================================================
# 5. Upload Validation Unit Tests (SEC-05)
# ==============================================================================
def test_validate_pdf_empty_file():
    with pytest.raises(HTTPException) as exc:
        _validate_pdf_content(b"", "empty.pdf")
    assert exc.value.status_code == 400

def test_validate_pdf_invalid_magic_bytes():
    with pytest.raises(HTTPException) as exc:
        _validate_pdf_content(b"NOT_A_PDF_DOCUMENT", "bad.pdf")
    assert exc.value.status_code == 422
    assert "not a valid PDF document" in exc.value.detail

def test_validate_pdf_exceeds_max_size():
    oversize_bytes = b"%PDF" + (b"0" * (config.MAX_FILE_SIZE_BYTES + 10))
    with pytest.raises(HTTPException) as exc:
        _validate_pdf_content(oversize_bytes, "large.pdf")
    assert exc.value.status_code == 413

# ==============================================================================
# 6. Segmentation Request Isolation & No Global State (BUG-01, BUG-02)
# ==============================================================================
def test_segmentation_without_globals():
    """Ensures segmentation output paths stay completely inside the given directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a synthetic white image with a couple of black horizontal bars
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        img[100:140, 50:450] = 0
        img[200:240, 50:450] = 0

        # Run segmentation on sheet 1
        res1 = segment_lines_and_find_diagrams(img, output_folder=temp_dir, sheet_idx=1)
        assert os.path.exists(res1["segmented_folder"])
        assert os.path.dirname(res1["segmented_folder"]) == temp_dir

        # Run segmentation on sheet 2 in same root without global collision
        sheet2_dir = os.path.join(temp_dir, "sheet_2")
        res2 = segment_lines_and_find_diagrams(img, output_folder=sheet2_dir, sheet_idx=2)
        assert os.path.exists(res2["segmented_folder"])
        assert os.path.dirname(res2["segmented_folder"]) == sheet2_dir

def test_segmentation_no_text_raises_specific_error():
    """Blank image should raise NoTextDetectedError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        blank_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        with pytest.raises(NoTextDetectedError):
            segment_lines_and_find_diagrams(blank_img, output_folder=temp_dir)
