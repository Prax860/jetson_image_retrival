"""
Query pipeline — hybrid RAG orchestrator.

This is the single public entry point for the Streamlit UI (or any future
front-end).  It owns the end-to-end flow:

    1. Extract structured intent from the user's query  (intent.py)
    2. Translate IntentFilter → Chroma `where` clause    (local)
    3. Run hybrid retrieval                              (retrieval.py)
    4. (future) Post-process / summarise results

By design, neither intent.py nor retrieval.py is aware of each other.
All composition happens here.

Public API
----------
run_query(query, vector_store, top_k, min_score) -> HybridQueryResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
import re
from typing import Any, Dict, List, Optional

from backend.app.core.config import get_settings
from backend.app.core.exceptions import QueryPipelineError
from backend.app.core.logging import get_logger
from backend.app.models.alert import SearchResult
from backend.app.models.intent import IntentFilter
from backend.app.repositories.vector_store import VectorStoreRepository
from backend.app.services.intent import extract_intent
from backend.app.utils.camera_ids import parse_camera_id_from_query
from backend.app.services.retrieval import search_alerts

logger = get_logger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class HybridQueryResult:
    """
    Everything the UI needs to render a search response.

    Attributes
    ----------
    results     : Ranked list of matching alert images.
    intent      : The structured filters extracted by the LLM — useful for
                  displaying "Searching camera 2 for persons after 15:00" in
                  the UI without re-parsing.
    where_clause: The Chroma filter that was applied (for debugging/logging).
    """
    results: List[SearchResult]
    intent: IntentFilter
    where_clause: Optional[Dict[str, Any]]


# ── Public API ────────────────────────────────────────────────────────────────

def run_query(
    query: str,
    vector_store: VectorStoreRepository,
    top_k: Optional[int] = None,
    min_score: float = 0.0,
) -> HybridQueryResult:
    """
    Run the full hybrid RAG pipeline for one user query.

    Parameters
    ----------
    query        : Raw natural-language string from the user.
    vector_store : Injected Chroma repository.
    top_k        : Maximum results to return (falls back to config default).
    min_score    : Cosine similarity threshold passed to retrieval.

    Returns
    -------
    HybridQueryResult with ranked images, extracted intent, and Chroma filter.
    """
    cfg = get_settings()
    k = top_k or cfg.DEFAULT_TOP_K

    # ── Step 1: Intent extraction ─────────────────────────────────────────────
    intent = extract_intent(query)
    logger.info("Original query: %s", query)
    # If the LLM omitted the camera_id entirely (or failed), try a lightweight
    # regex-based extraction from the raw query so simple inputs like "cam2"
    # still produce a metadata filter.
    if not intent.camera_id:
        raw_cam = parse_camera_id_from_query(query)
        if raw_cam:
            logger.info("Parsed camera_id from query fallback: %s", raw_cam)
            intent = intent.model_copy(update={"camera_id": raw_cam})
    logger.info("Raw LLM output (post-fallback): %s", intent.model_dump(exclude_none=True))

    # ── Step 2: Build Chroma where-clause ────────────────────────────────────
    normalized_intent = _normalize_intent(intent, vector_store, query)
    logger.info("Normalized IntentFilter: %s", normalized_intent)

    if _is_trivial_query(query) and not normalized_intent.has_metadata_filters():
        logger.info("Generated where clause: None")
        logger.info("Number of results: 0")
        return HybridQueryResult(results=[], intent=normalized_intent, where_clause=None)

    where = _build_where_clause(normalized_intent, vector_store)
    logger.info("Generated where clause: %s", where)

    # ── Step 3: Hybrid retrieval ──────────────────────────────────────────────
    try:
        results = search_alerts(
            query=normalized_intent.semantic_query,
            vector_store=vector_store,
            top_k=k,
            min_score=min_score,
            where=where,            # pre-built filter — retrieval.py passes it
        )                           # straight through to Chroma
    except Exception as exc:
        raise QueryPipelineError(f"Retrieval failed: {exc}") from exc

    logger.info(
        "Pipeline complete | query=%r | filters=%s | results=%d",
        query,
        where,
        len(results),
    )
    logger.info("Number of results: %d", len(results))

    return HybridQueryResult(results=results, intent=normalized_intent, where_clause=where)


# ── Chroma filter builder ─────────────────────────────────────────────────────

def _build_where_clause(intent: IntentFilter, vector_store: Optional[VectorStoreRepository] = None) -> Optional[Dict[str, Any]]:
    """
    Translate an IntentFilter into a Chroma `where` dict.

    Chroma supports:
        {"field": {"$eq": value}}
        {"field": {"$gte": value}}
        {"$and": [clause, clause, ...]}

    Timestamp handling:
        Chroma stores timestamps as ISO-8601 strings.
        String comparison works correctly for ISO format, so we build
        ISO datetime strings from the date + time components and use
        $gte / $lte on the "timestamp" field.

    Returns None if no filters are active (= search entire collection).
    """
    if not intent.has_metadata_filters():
        return None

    conditions: List[Dict[str, Any]] = []

    # Exact-match filters
    if intent.camera_id:
        cam = intent.camera_id
        # Always include the normalized string form
        cam_conditions: List[Dict[str, Any]] = [{"camera_id": {"$eq": cam}}]

        # If the camera looks numeric, check whether stored Chroma metadata
        # uses integers for camera_id; if so, include an integer equality
        # branch to match both representations. We only attempt this when a
        # `vector_store` is available with a `._col.get()` method to avoid
        # touching external systems in unit tests.
        try:
            if str(cam).isdigit() and vector_store is not None and hasattr(vector_store, "_col"):
                # Inspect a small sample of stored metadatas to detect int types
                try:
                    sample = vector_store._col.get(include=["metadatas"]) or {}
                    metas = sample.get("metadatas") or []
                    found_int = False
                    for m in metas:
                        if m is None:
                            continue
                        c = m.get("camera_id")
                        if isinstance(c, int):
                            found_int = True
                            break
                    if found_int:
                        cam_int = int(str(cam))
                        cam_conditions.append({"camera_id": {"$eq": cam_int}})
                except Exception:
                    # If anything goes wrong inspecting the collection, skip int branch
                    pass
        except Exception:
            pass

        if len(cam_conditions) == 1:
            conditions.append(cam_conditions[0])
        else:
            conditions.append({"$or": cam_conditions})

    if intent.label:
        conditions.append({"label": {"$eq": intent.label}})

    if intent.alert_type:
        conditions.append({"alert_type": {"$eq": intent.alert_type}})

    if intent.min_confidence is not None:
        conditions.append({"confidence": {"$gte": intent.min_confidence}})

    # Timestamp range filters
    # We resolve date + time independently so either can be None.
    # If only a time is given (no date), we cannot safely build an ISO
    # timestamp without knowing the date, so we skip the filter and let
    # Python-side post-filtering handle it (see _post_filter_by_time).
    if intent.date:
        ts_after, ts_before = _build_timestamp_range(
            intent.date, intent.time_after, intent.time_before
        )
        if ts_after:
            conditions.append({"timestamp": {"$gte": ts_after}})
        if ts_before:
            conditions.append({"timestamp": {"$lte": ts_before}})

    if not conditions:
        return None

    return {"$and": conditions} if len(conditions) > 1 else conditions[0]


def _normalize_intent(
    intent: IntentFilter,
    vector_store: VectorStoreRepository,
    query: str,
) -> IntentFilter:
    """Normalize intent fields to the camera format already stored in Chroma."""
    updates: Dict[str, Any] = {}

    if intent.camera_id:
        normalized_camera_id = vector_store.normalize_camera_id(intent.camera_id)
        if normalized_camera_id and normalized_camera_id != intent.camera_id:
            updates["camera_id"] = normalized_camera_id

    if not intent.semantic_query:
        updates["semantic_query"] = intent.label or intent.camera_id or intent.alert_type or query

    if not updates:
        return intent

    return intent.model_copy(update=updates)


def _is_trivial_query(query: str) -> bool:
    """Return True for greeting-style queries that should not trigger search."""
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    if not tokens:
        return True

    trivial_tokens = {
        "hi",
        "hello",
        "hey",
        "yo",
        "thanks",
        "thank",
        "you",
        "please",
        "morning",
        "afternoon",
        "evening",
        "there",
    }
    return len(tokens) <= 3 and all(token in trivial_tokens for token in tokens)


def _build_timestamp_range(
    date_str: str,
    time_after: Optional[str],
    time_before: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Build ISO timestamp strings for a date + optional time bounds.

    Returns (ts_after, ts_before).  Either can be None.
    """
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        logger.warning("Could not parse date: %r — skipping timestamp filter", date_str)
        return None, None

    ts_after: Optional[str] = None
    ts_before: Optional[str] = None

    if time_after:
        try:
            h, m = map(int, time_after.split(":"))
            ts_after = datetime.combine(d, time(h, m)).isoformat()
        except Exception:
            logger.warning("Could not parse time_after: %r", time_after)

    if time_before:
        try:
            h, m = map(int, time_before.split(":"))
            ts_before = datetime.combine(d, time(h, m)).isoformat()
        except Exception:
            logger.warning("Could not parse time_before: %r", time_before)

    # If only a date is given (no time bounds), filter the whole day
    if ts_after is None and ts_before is None:
        ts_after = datetime.combine(d, time.min).isoformat()
        ts_before = datetime.combine(d, time.max).isoformat()

    return ts_after, ts_before