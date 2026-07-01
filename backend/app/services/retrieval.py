"""
Retrieval service.

Takes a natural-language query, embeds it with CLIP, queries Chroma,
and returns SearchResult objects that include the base64-encoded image
so the Streamlit UI can render them directly.

After Chroma returns matching ids, the canonical metadata is loaded from
data/metadata/<id>.json so the full record (bbox, frame_num, etc.) is
available to the frontend.  Chroma metadata is used only as a fallback
when the JSON file is missing.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.config import get_settings
from backend.app.core.exceptions import RetrievalError
from backend.app.core.logging import get_logger
from backend.app.models.alert import AlertRecord, BBox, SearchResult
from backend.app.repositories.vector_store import VectorStoreRepository
from backend.app.services.rag import embed_text
from backend.app.services.highlight import highlight_image_b64
from backend.app.utils.metadata import load_metadata

logger = get_logger(__name__)


def search_alerts(
    query: str,
    vector_store: VectorStoreRepository,
    top_k: Optional[int] = None,
    camera_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    min_score: float = 0.0,
    where: Optional[Dict[str, Any]] = None,
    original_query: Optional[str] = None,
) -> List[SearchResult]:
    """
    Semantic search over indexed alert images.

    Parameters
    ----------
    query:        Natural-language description, e.g. "person in red jacket".
    vector_store: Injected repository.
    top_k:        Max results (falls back to DEFAULT_TOP_K from config).
    camera_id:    Legacy filter — only return alerts from this camera.
                  Ignored when `where` is supplied.
    alert_type:   Legacy filter — only return alerts of this type.
                  Applied as Python post-filter when `where` is supplied.
    min_score:    Cosine similarity threshold in [0, 1].
    where:        Pre-built Chroma where-clause dict (from query_pipeline.py).
    original_query: Raw user query before intent parsing. Used as the
                  highlight prompt (color/garment parsing happens inside
                  highlight.py). Falls back to *query* if omitted.

    Returns
    -------
    List of SearchResult sorted by descending similarity, each with a
    base64-encoded image ready for rendering.
    """
    cfg = get_settings()
    k = top_k or cfg.DEFAULT_TOP_K

    if not query.strip():
        raise RetrievalError("Query must not be empty.")

    logger.info(
        "Search | query=%r | top_k=%d | where=%s | camera=%s",
        query, k, where, camera_id or "*",
    )

    # 1. Embed the query
    query_vector = embed_text(query)

    # 2. Resolve the Chroma where-clause
    chroma_where = where
    if chroma_where is None and camera_id:
        chroma_where = {"camera_id": camera_id}

    # 3. Fetch from vector store (over-fetch to allow post-filtering)
    fetch_k = min(k * 4, cfg.MAX_TOP_K * 4)
    raw = vector_store.query(
        query_embedding=query_vector,
        top_k=fetch_k,
        where=chroma_where,
    )

    logger.info("=" * 70)
    logger.info("Query: %s", query)
    for meta, score in raw:
        logger.info(
            "%.4f | camera=%s | label=%s",
            score, meta.get("camera_id"), meta.get("label"),
        )
    logger.info("=" * 70)

    # 4. Build results, loading full metadata from JSON
    results: List[SearchResult] = []
    rank = 1

    for chroma_meta, score in raw:
        if score < min_score:
            continue

        record_id = chroma_meta.get("id", "")

        # Load canonical metadata; fall back to Chroma metadata if missing
        full_meta = load_metadata(record_id) if record_id else {}
        if not full_meta:
            logger.warning(
                "metadata.json missing for id=%s — falling back to Chroma metadata",
                record_id,
            )
            full_meta = chroma_meta

        record = _meta_to_record(full_meta)

        # Secondary Python-side filter on alert_type
        if alert_type and record.alert_type != alert_type:
            continue

        image_b64 = _load_image_b64(record.image_path)

        # Highlight matching person(s) via YOLO person detection + CLIP
        # zero-shot color classification (see highlight.py for the full
        # pipeline). record_bbox is passed for call-signature compatibility
        # but is intentionally IGNORED inside highlight_image_b64 — stored
        # metadata bboxes are (a) computed at ingest time with no relation
        # to any color/attribute named in a later search query, and (b) can
        # be pixel-misaligned with this image if it was resized between the
        # Jetson and this backend. Every box drawn is computed fresh on the
        # actual image bytes below.
        highlight_query = original_query or query
        if image_b64:
            record_bbox = record.bbox.model_dump() if record.bbox else None
            image_b64 = highlight_image_b64(
                image_b64,
                highlight_query,
                record_bbox=record_bbox,
            )

        results.append(
            SearchResult(record=record, score=round(score, 4), rank=rank, image_b64=image_b64)
        )
        rank += 1
        if rank > k:
            break

    logger.info("Returned %d results for query %r", len(results), query)
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meta_to_record(meta: dict) -> AlertRecord:
    """Construct an AlertRecord from a metadata dict (JSON or Chroma fallback)."""

    # confidence: stored as -1.0 sentinel when absent
    confidence_raw = meta.get("confidence", -1.0)
    try:
        confidence = float(confidence_raw) if float(confidence_raw) >= 0 else None
    except (TypeError, ValueError):
        confidence = None

    # extra: may be a JSON string (Chroma) or already a dict (JSON file)
    extra_raw = meta.get("extra", {})
    if isinstance(extra_raw, str):
        try:
            extra_raw = json.loads(extra_raw)
        except Exception:
            extra_raw = {}

    # bbox: may be a dict or None
    bbox_raw = meta.get("bbox")
    bbox_model: Optional[BBox] = None
    if isinstance(bbox_raw, dict):
        try:
            bbox_model = BBox(**bbox_raw)
        except Exception:
            pass

    # indexed_at: may be absent in legacy records
    indexed_at_raw = meta.get("indexed_at")
    try:
        indexed_at = datetime.fromisoformat(indexed_at_raw) if indexed_at_raw else datetime.utcnow()
    except Exception:
        indexed_at = datetime.utcnow()

    return AlertRecord(
        id=meta.get("id", ""),
        image_path=meta.get("image_path", ""),
        image_filename=meta.get("image_filename", ""),
        camera_id=meta.get("camera_id", ""),
        timestamp=datetime.fromisoformat(meta["timestamp"]),
        alert_type=meta.get("alert_type") or None,
        confidence=confidence,
        location_label=meta.get("location_label") or None,
        extra=extra_raw,
        # new fields (present in JSON, absent in legacy Chroma fallback)
        label=meta.get("label") or None,
        frame_num=meta.get("frame_num"),
        object_id=meta.get("object_id"),
        class_id=meta.get("class_id"),
        bbox=bbox_model,
        caption=meta.get("caption", ""),
        ocr=meta.get("ocr", ""),
        metadata_path=meta.get("metadata_path"),
        indexed_at=indexed_at,
    )


def _load_image_b64(image_path: str) -> str:
    """Read image bytes from disk and return base64-encoded string."""
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image file not found: %s", image_path)
        return ""
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception as exc:
        logger.error("Cannot read image %s: %s", image_path, exc)
        return ""