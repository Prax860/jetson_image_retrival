"""
RAG service — CLIP embedding engine.

Keeps the CLIP model as a module-level singleton so it is loaded once
per process, not once per request.

Public functions
----------------
embed_image_file(path)  → List[float]   – embed a saved image file
embed_text(text)        → List[float]   – embed a natural-language query
index_record(record, vector_store)      – embed + upsert into Chroma

Chroma stores only the fields needed for hybrid search:
    id, camera_id, label, timestamp, confidence, caption, image_path,
    alert_type, location_label, image_filename, indexed_at

Heavy / non-searchable fields (bbox, frame_num, object_id, class_id, extra)
live exclusively in data/metadata/<id>.json.
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
from backend.app.utils.metadata import load_metadata, update_metadata

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
    """Open an image from disk and return its normalised CLIP embedding."""
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
    """Encode a natural-language string and return its normalised CLIP embedding."""
    cfg = get_settings()
    model, processor = _load_clip()

    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(cfg.EMBEDDING_DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
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

    Only slim / searchable fields are stored in Chroma.
    The full metadata lives in data/metadata/<id>.json.
    """
    embedding = embed_image_file(record.image_path)

    # -- Load the canonical metadata JSON written by ingest.py ----------------
    meta = load_metadata(record.id)

    # -- Stamp indexed_at into the JSON only when the file actually exists -----
    # Calling update_metadata on a missing file would silently create a ghost
    # JSON containing only {"indexed_at": ...}, discarding all ingest metadata.
    indexed_at = record.indexed_at.isoformat()
    if meta:
        update_metadata(record.id, {"indexed_at": indexed_at})
    else:
        logger.warning(
            "metadata.json not found for %s during indexing — "
            "Chroma metadata will be built from AlertRecord fields only.",
            record.id,
        )

    # -- Build the slim Chroma payload ----------------------------------------
    # Only fields useful for filtering / re-ranking go here.
    # bbox, frame_num, object_id, class_id, extra stay in JSON only.
    chroma_metadata = {
        "id":             record.id,
        "camera_id":      record.camera_id,
        "timestamp":      record.timestamp.isoformat(),
        # Prefer JSON value (already persisted) → AlertRecord field → empty string
        "label":          meta.get("label") or record.label or "",
        "alert_type":     record.alert_type or "",
        "confidence":     record.confidence if record.confidence is not None else -1.0,
        "location_label": record.location_label or "",
        "image_path":     record.image_path,
        "image_filename": record.image_filename,
        "caption":        meta.get("caption", ""),
        "indexed_at":     indexed_at,
    }

    vector_store.upsert(record.id, embedding, chroma_metadata)
    logger.info("Indexed alert %s (camera=%s)", record.id, record.camera_id)