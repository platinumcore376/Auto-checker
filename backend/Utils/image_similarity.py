import logging
import threading
from typing import List, Tuple
from PIL import Image
from config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_SCORE_THRESHOLD,
    CLIP_SCALE_FACTOR,
    CLIP_DIVISOR,
)

logger = logging.getLogger("autochecker.image_similarity")

_model = None
_preprocess = None
_tokenizer = None
_device = None
_model_lock = threading.Lock()

def load_clip_model():
    """
    Thread-safe loader and cache for OpenCLIP model and transforms.
    Supports warm-up during application lifespan (PERF-01).
    """
    global _model, _preprocess, _tokenizer, _device
    with _model_lock:
        if _model is None:
            import torch
            import open_clip

            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Loading OpenCLIP '{CLIP_MODEL_NAME}' ({CLIP_PRETRAINED}) on {_device}")
            _model, _preprocess, _tokenizer = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=CLIP_PRETRAINED
            )
            _model = _model.to(_device)
            _model.eval()  # ML-MODEL-02
        return _model, _preprocess, _device

def transform(x):
    """
    Score transform:
    Values >= CLIP_SCORE_THRESHOLD (0.9) remain unchanged.
    Values < 0.9 are scaled down significantly using CLIP_SCALE_FACTOR * (x / CLIP_DIVISOR).
    """
    import torch
    return torch.where(
        x >= CLIP_SCORE_THRESHOLD,
        x,
        CLIP_SCALE_FACTOR * (x / CLIP_DIVISOR)
    )

def image_similarity(truth_file, image_files: List):
    """
    Computes visual similarity between truth_file and each image in image_files.
    Returns empty list if image_files is empty (BUG-08).
    """
    if not image_files:
        return []

    import torch
    model, preprocess, device = load_clip_model()

    # Preprocessing the ground truth reference diagram
    truth_tensor = preprocess(truth_file).unsqueeze(0).to(device)
    
    # Preprocessing the candidate diagram images
    test_tensors = torch.stack([preprocess(img) for img in image_files]).to(device)
    
    # Generating image embeddings
    with torch.no_grad():
        truth_embed = model.encode_image(truth_tensor)
        test_embeds = model.encode_image(test_tensors)
        
    similarities = torch.nn.functional.cosine_similarity(truth_embed, test_embeds)
    transformed = transform(similarities).cpu().numpy()
    return transformed
