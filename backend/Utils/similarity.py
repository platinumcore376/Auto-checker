import logging
import threading
from math import ceil
from typing import Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from config import SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_REVISION, SCORE_ROUNDING

logger = logging.getLogger("autochecker.similarity")

_model_cache: Dict[str, Tuple[object, str]] = {}
_model_lock = threading.Lock()

def load_sentence_transformer(
    model_name: Optional[str] = None,
    revision: Optional[str] = None
):
    """
    Thread-safe loader and cache for SentenceTransformer models.
    Supports warm-up during application lifespan (PERF-01).
    """
    model_key = model_name or SENTENCE_TRANSFORMER_MODEL
    target_revision = revision if revision is not None else SENTENCE_TRANSFORMER_REVISION

    with _model_lock:
        if model_key in _model_cache:
            return _model_cache[model_key]

        from sentence_transformers import SentenceTransformer
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading SentenceTransformer '{model_key}' (revision={target_revision}) on {device}")
        
        load_kwargs = {}
        if target_revision:
            load_kwargs["revision"] = target_revision

        model = SentenceTransformer(model_key, **load_kwargs).to(device)
        model.eval()
        _model_cache[model_key] = (model, device)
        return model, device

def text_similarity(
    key_answer: str,
    test_answer: str,
    model_name: Optional[str] = None,
    revision: Optional[str] = None
) -> float:
    """
    Computes cosine similarity between key_answer and test_answer embeddings.
    Rounding mode is governed by SCORE_ROUNDING in config.py (default: 'ceil').
    """
    if not key_answer or not test_answer:
        return 0.0

    model, device = load_sentence_transformer(model_name, revision)
    
    key_embeddings = model.encode(key_answer)
    test_embeddings = model.encode(test_answer)

    similarity = cosine_similarity([key_embeddings], [test_embeddings])[0][0]
    sim_value = float(similarity.item() if hasattr(similarity, "item") else similarity)

    # Clamp raw similarity to valid [0.0, 1.0] range to prevent floating point overflow
    sim_value = max(0.0, min(1.0, round(sim_value, 6)))

    if SCORE_ROUNDING == "round":
        final_score = round(sim_value, 2)
    else:
        # Default behavior: ceil to nearest 0.1, clamped at 1.0
        final_score = min(1.0, ceil(sim_value * 10) / 10)

    logger.debug(f"Computed text similarity: raw={sim_value:.4f}, final={final_score}")
    return final_score
