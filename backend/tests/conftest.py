import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

@pytest.fixture(autouse=True)
def mock_external_ml_and_cloud(monkeypatch):
    """
    Autouse fixture to strictly enforce Ground Rule 7:
    Mocks Google Vision and stubs SentenceTransformer and OpenCLIP loaders
    so tests never make network calls, download weights, or incur costs.
    """
    # 1. Stub SentenceTransformer
    mock_st_model = MagicMock()
    mock_st_model.encode.side_effect = lambda text: np.array([0.5, 0.5, 0.5])
    mock_st_model.eval.return_value = None

    import Utils.similarity as sim_module
    monkeypatch.setattr(sim_module, "_model_cache", {"stsb-roberta-large": (mock_st_model, "cpu")})
    monkeypatch.setattr(sim_module, "load_sentence_transformer", lambda *args, **kwargs: (mock_st_model, "cpu"))

    # 2. Stub OpenCLIP
    import torch
    mock_clip_model = MagicMock()
    mock_clip_model.encode_image.side_effect = lambda tensor: torch.randn(1, 512)
    mock_clip_model.eval.return_value = None
    mock_preprocess = lambda img: torch.randn(3, 224, 224)

    import Utils.image_similarity as img_sim_module
    monkeypatch.setattr(img_sim_module, "_model", mock_clip_model)
    monkeypatch.setattr(img_sim_module, "_preprocess", mock_preprocess)
    monkeypatch.setattr(img_sim_module, "_device", "cpu")
    monkeypatch.setattr(img_sim_module, "load_clip_model", lambda: (mock_clip_model, mock_preprocess, "cpu"))

    # Mock cosine similarity for image similarity
    import torch
    monkeypatch.setattr(
        torch.nn.functional,
        "cosine_similarity",
        lambda a, b: torch.tensor([0.92])
    )

    # 3. Stub Google Cloud Vision Client
    mock_vision_client = MagicMock()
    mock_response = MagicMock()
    mock_response.error.message = ""
    mock_response.full_text_annotation.text = "Handwritten answer text extracted via OCR"
    mock_vision_client.document_text_detection.return_value = mock_response

    monkeypatch.setattr("google.cloud.vision.ImageAnnotatorClient", lambda: mock_vision_client)

@pytest.fixture
def test_client():
    """Returns an authenticated FastAPI TestClient configured for AutoChecker."""
    from fastapi.testclient import TestClient
    from server import app
    from auth import create_access_token
    import config

    with TestClient(app) as client:
        token = create_access_token({"sub": "admin@rait.ac.in"})
        client.cookies.set(config.COOKIE_NAME, token)
        yield client
