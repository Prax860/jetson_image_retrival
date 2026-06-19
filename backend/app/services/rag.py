"""
RAG service — CLIP embedding engine.

Keeps the CLIP model as a module-level singleton so it is loaded once
per process, not once per request.

Public functions
----------------
embed_image_file(path)  → List[float]   – embed a saved image file
embed_text(text)        → List[float]   – embed a natural-language query
index_record(record, vector_store)      – embed + upsert into Chroma
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.app.core.config import get_settings
from backend.app.core.exceptions import EmbeddingError
from backend.app.core.logging import get_logger
from backend.app.models.alert import AlertRecord
from backend.app.repositories.vector_store import VectorStoreRepository

logger = get_logger(__name__)

# ── Module-level CLIP singleton ───────────────────────────────────────────────

_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None


def _load_clip() -> tuple[CLIPModel, CLIPProcessor]:
    global _model, _processor
    if _model is None:
        cfg = get_settings()
        logger.info("Loading CLIP: %s on %s", cfg.CLIP_MODEL_NAME, cfg.EMBEDDING_DEVICE)
        _processor = CLIPProcessor.from_pretrained(cfg.CLIP_MODEL_NAME)
        _model = CLIPModel.from_pretrained(cfg.CLIP_MODEL_NAME)
        _model.eval()
        _model.to(cfg.EMBEDDING_DEVICE)
        logger.info("CLIP ready.")
    return _model, _processor  # type: ignore[return-value]


# ── Public API ────────────────────────────────────────────────────────────────

def embed_image_file(image_path: str | Path) -> List[float]:
    """
    Open an image from disk and return its normalised CLIP embedding.
    """
    cfg = get_settings()
    model, processor = _load_clip()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise EmbeddingError(f"Cannot open image {image_path}: {exc}") from exc

    try:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(cfg.EMBEDDING_DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)

            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output

            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().tolist()
    except Exception as exc:
        raise EmbeddingError(f"Image embedding failed: {exc}") from exc


def embed_text(text: str) -> List[float]:
    """
    Encode a natural-language string and return its normalised CLIP embedding.
    Used to turn a user's chatbot query into a search vector.
    """
    cfg = get_settings()
    model, processor = _load_clip()

    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(cfg.EMBEDDING_DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)

            # Newer transformers return BaseModelOutputWithPooling
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output

            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats[0].cpu().tolist()
    except Exception as exc:
        raise EmbeddingError(f"Text embedding failed: {exc}") from exc


def index_record(
    record: AlertRecord,
    vector_store: VectorStoreRepository,
) -> None:
    """
    Embed the image in *record* and upsert it into the vector store.
    """
    embedding = embed_image_file(record.image_path)

    metadata = {
        "id": record.id,
        "camera_id": record.camera_id,
        "timestamp": record.timestamp.isoformat(),
        "alert_type": record.alert_type or "",
        "confidence": record.confidence if record.confidence is not None else -1.0,
        "location_label": record.location_label or "",
        "image_path": record.image_path,
        "image_filename": record.image_filename,
        "extra": record.extra,          # _sanitise() in repo will JSON-encode this
        "indexed_at": record.indexed_at.isoformat(),
    }

    vector_store.upsert(record.id, embedding, metadata)
    logger.info("Indexed alert %s (camera=%s)", record.id, record.camera_id)
