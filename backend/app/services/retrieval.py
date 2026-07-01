"""
Retrieval service.

Takes a natural-language query, embeds it with CLIP, queries Chroma,
runs a generic attribute-verification re-ranking pass over the
candidates (see reranker.py / verifiers/), and returns SearchResult
objects that include the base64-encoded image so the Streamlit UI can
render them directly.

Candidate pool sizing:
    - If the query names a verifiable attribute (color/object/etc.),
      CLIP's role shifts from "final ranker" to "candidate gatherer" —
      we fetch every image matching `where` (up to _MAX_VERIFICATION_POOL)
      so a low CLIP score can't silently hide a real match from the
      verifier. Verification (with its NO_MATCH/UNKNOWN distinction) is
      what actually decides relevance in this mode.
    - Otherwise (plain queries like "camera 5", "show all persons"),
      behavior and cost are unchanged from the original implementation:
      a small over-fetch pool, no verification pass.

After Chroma returns matching ids, the canonical metadata is loaded from
data/metadata/<id>.json so the full record (bbox, frame_num, etc.) is
available to the frontend. Chroma metadata is used only as a fallback
when the JSON file is missing.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from backend.app.core.config import get_settings
from backend.app.core.exceptions import RetrievalError
from backend.app.core.logging import get_logger
from backend.app.models.alert import AlertRecord, BBox, SearchResult
from backend.app.repositories.vector_store import VectorStoreRepository
from backend.app.services.rag import embed_text
from backend.app.services.highlight import highlight_image_b64
from backend.app.services.reranker import maybe_rerank_by_attribute
from backend.app.services.verifiers.registry import parse_attribute_query
from backend.app.utils.metadata import load_metadata

logger = get_logger(__name__)

# Fallback candidate pool for plain (non-attribute) queries — matches the
# original implementation's over-fetch factor, keeps cost/behavior
# identical when nothing is verified.
_DEFAULT_CANDIDATE_POOL = 20

# Hard ceiling even in "verify everything" mode, so a runaway query can't
# force-open/verify an unbounded number of images in one request. Raise
# this (or move it into Settings) if your collection legitimately exceeds it.
_MAX_VERIFICATION_POOL = 2000


def _collection_size(vector_store: VectorStoreRepository, where: Optional[Dict[str, Any]]) -> Optional[int]:
    """
    Best-effort count of documents matching `where` (or the whole
    collection if where is None). Returns None if the repository doesn't
    expose the underlying Chroma collection, so callers can fall back
    gracefully instead of crashing.
    """
    col = getattr(vector_store, "_col", None)
    if col is None:
        return None
    try:
        if where:
            got = col.get(where=where, include=[])
            return len(got.get("ids") or [])
        return col.count()
    except Exception as exc:
        logger.warning("Could not determine collection size for exhaustive fetch: %s", exc)
        return None


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
    Semantic search over indexed alert images, with a second-stage
    attribute-verification re-rank on top of CLIP similarity.

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
                  highlight/verification prompt (attribute parsing happens
                  inside verifiers/registry.py). Falls back to *query* if
                  omitted.

    Returns
    -------
    List of SearchResult sorted by descending final_score, each with a
    base64-encoded image ready for rendering.
    """
    cfg = get_settings()
    k = top_k or cfg.DEFAULT_TOP_K

    if not query.strip():
        raise RetrievalError("Query must not be empty.")

    highlight_query = original_query or query

    # Parse up front — this alone decides fetch-pool sizing and whether
    # verification runs later. Purely text-based, no side effects.
    attribute_query = parse_attribute_query(highlight_query)

    logger.info(
        "Search | query=%r | top_k=%d | where=%s | camera=%s | attribute=%s",
        query, k, where, camera_id or "*",
        attribute_query.value if attribute_query else None,
    )

    # 1. Embed the query
    query_vector = embed_text(query)

    # 2. Resolve the Chroma where-clause
    chroma_where = where
    if chroma_where is None and camera_id:
        chroma_where = {"camera_id": camera_id}

    # 3. Decide fetch pool size
    if attribute_query is not None:
        total = _collection_size(vector_store, chroma_where)
        if total is None:
            # Repository doesn't support counting — fall back to the hard
            # ceiling rather than the old small pool, since the whole
            # point of this mode is "don't let CLIP prematurely truncate".
            fetch_k = _MAX_VERIFICATION_POOL
        else:
            fetch_k = min(max(total, 1), _MAX_VERIFICATION_POOL)
        logger.info("Exhaustive fetch mode (attribute query) | fetch_k=%d", fetch_k)
    else:
        fetch_k = min(max(k * 4, _DEFAULT_CANDIDATE_POOL), cfg.MAX_TOP_K * 4)

    raw = vector_store.query(
        query_embedding=query_vector,
        top_k=fetch_k,
        where=chroma_where,
    )

    logger.info("=" * 70)
    logger.info("Query: %s | fetched %d candidates", query, len(raw))
    for meta, score in raw:
        logger.info(
            "%.4f | camera=%s | label=%s",
            score, meta.get("camera_id"), meta.get("label"),
        )
    logger.info("=" * 70)

    # 4. Resolve full metadata + apply score/alert_type filters, build
    #    lightweight candidate structs (image not yet base64-encoded/boxed
    #    — that's deferred until after reranking so we don't waste work on
    #    candidates that end up discarded by verification).
    candidates: List[_Candidate] = []
    for chroma_meta, score in raw:
        if score < min_score:
            continue

        record_id = chroma_meta.get("id", "")
        full_meta = load_metadata(record_id) if record_id else {}
        if not full_meta:
            logger.warning(
                "metadata.json missing for id=%s — falling back to Chroma metadata",
                record_id,
            )
            full_meta = chroma_meta

        record = _meta_to_record(full_meta)

        if alert_type and record.alert_type != alert_type:
            continue

        candidates.append(_Candidate(record=record, clip_score=score))

    # 5. Attribute verification + generic re-rank (no-op if attribute_query
    #    is None — see reranker.py / verifiers/registry.py).
    reranked = maybe_rerank_by_attribute(
        query=highlight_query,
        candidates=candidates,
        get_image=_candidate_image,
        get_clip_score=lambda c: c.clip_score,
    )

    # 6. Build final SearchResults for the top_k re-ranked candidates only.
    results: List[SearchResult] = []
    rank = 1
    for reranked_candidate in reranked:
        if rank > k:
            break
        candidate = reranked_candidate.item
        record = candidate.record

        image_b64 = _load_image_b64(record.image_path)
        if image_b64:
            record_bbox = record.bbox.model_dump() if record.bbox else None
            image_b64 = highlight_image_b64(image_b64, highlight_query, record_bbox=record_bbox)

        results.append(
            SearchResult(
                record=record,
                score=round(reranked_candidate.final_score, 4),
                rank=rank,
                image_b64=image_b64,
            )
        )
        rank += 1

    logger.info(
        "Returned %d results for query %r (candidates evaluated=%d, verified=%s)",
        len(results), query, len(candidates), attribute_query is not None,
    )
    return results


# ── Candidate helpers ─────────────────────────────────────────────────────────

class _Candidate:
    """Lightweight holder used only during the retrieval/verification pass."""

    __slots__ = ("record", "clip_score", "_pil_image", "_image_loaded")

    def __init__(self, record: AlertRecord, clip_score: float):
        self.record = record
        self.clip_score = clip_score
        self._pil_image = None
        self._image_loaded = False


def _candidate_image(candidate: "_Candidate") -> Optional[Image.Image]:
    """Lazily load and cache the PIL image for a candidate (verification only)."""
    if candidate._image_loaded:
        return candidate._pil_image
    candidate._image_loaded = True
    path = Path(candidate.record.image_path)
    if not path.exists():
        logger.warning("Image file not found for verification: %s", path)
        return None
    try:
        candidate._pil_image = Image.open(path).convert("RGB")
    except Exception as exc:
        logger.error("Cannot open image %s for verification: %s", path, exc)
        candidate._pil_image = None
    return candidate._pil_image


# ── Helpers (unchanged from the original implementation) ─────────────────────

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