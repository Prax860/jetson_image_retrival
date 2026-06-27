"""
FastAPI v1 router.

Endpoints
---------
POST   /api/v1/ingest         Push one alert image + metadata from Jetson
POST   /api/v1/search         Hybrid RAG search (LLM intent + CLIP similarity)
GET    /api/v1/collections    Collection stats (total alerts, camera list)
DELETE /api/v1/alerts/{id}    Remove a specific alert
GET    /api/v1/health         Liveness probe
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.core.config import get_settings
from backend.app.core.exceptions import (
    EmbeddingError,
    IngestError,
    IntentExtractionError,
    QueryPipelineError,
    RetrievalError,
)
from backend.app.core.logging import get_logger
from backend.app.repositories.vector_store import VectorStoreRepository, get_vector_store
from backend.app.schemas.alert import (
    AlertResultItem,
    CollectionStatsResponse,
    HealthResponse,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from backend.app.services.ingest import ingest_alert
from backend.app.services.rag import index_record
from backend.app.services.query_pipeline import run_query

logger = get_logger(__name__)
router = APIRouter()

VSDep = Annotated[VectorStoreRepository, Depends(get_vector_store)]


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Push an alert image from a Jetson device",
    description=(
        "Accepts a multipart/form-data upload with the image file and metadata fields. "
        "The server saves the image, writes a metadata JSON, embeds with CLIP, "
        "and stores the vector in Chroma."
    ),
)
async def ingest_endpoint(
    vector_store: VSDep,
    image: UploadFile = File(..., description="Alert image file (JPEG, PNG, etc.)"),
    camera_id: str = Form(..., description="Unique camera / device identifier"),
    timestamp: datetime = Form(..., description="Alert datetime in ISO-8601 format"),
    alert_type: Optional[str] = Form(None, description="e.g. 'motion', 'person', 'vehicle'"),
    confidence: Optional[float] = Form(None, description="Detection confidence 0–1"),
    location_label: Optional[str] = Form(None, description="Human-readable camera location"),
    extra_json: Optional[str] = Form(
        None, description="Any additional metadata as a JSON object string"
    ),
    # ── NEW: Jetson detection fields ──────────────────────────────────────────
    label: Optional[str] = Form(None, description="Detection class label, e.g. 'person'"),
    frame_num: Optional[int] = Form(None, description="Frame number in the video stream"),
    object_id: Optional[int] = Form(None, description="Tracker object ID"),
    class_id: Optional[int] = Form(None, description="Numeric class index from detector"),
    bbox: Optional[str] = Form(
        None, description="Bounding box as JSON: {left, top, width, height}"
    ),
) -> IngestResponse:
    # Parse extra_json
    extra: dict = {}
    if extra_json:
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"extra_json is not valid JSON: {exc}",
            )

    # Parse bbox JSON string → dict
    bbox_dict: Optional[dict] = None
    if bbox:
        try:
            bbox_dict = json.loads(bbox)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"bbox is not valid JSON: {exc}",
            )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image upload.",
        )

    try:
        record = ingest_alert(
            image_bytes=image_bytes,
            original_filename=image.filename or "upload.jpg",
            camera_id=camera_id,
            timestamp=timestamp,
            alert_type=alert_type,
            confidence=confidence,
            location_label=location_label,
            extra=extra,
            # new fields
            label=label,
            frame_num=frame_num,
            object_id=object_id,
            class_id=class_id,
            bbox=bbox_dict,
        )
    except IngestError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        index_record(record, vector_store)
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {exc}",
        )

    return IngestResponse(
        id=record.id,
        camera_id=record.camera_id,
        timestamp=record.timestamp,
        alert_type=record.alert_type,
        confidence=record.confidence,
        location_label=record.location_label,
        image_filename=record.image_filename,
    )


# ── Search ────────────────────────────────────────────────────────────────────

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid RAG search by natural-language query",
    description=(
        "The query is parsed by a local LLM to extract structured filters "
        "(camera, label, time range). Those filters narrow the Chroma candidate "
        "set before CLIP similarity re-ranks the results. "
        "Example: 'Show persons from camera 2 after 3 PM'."
    ),
)
async def search_endpoint(
    body: SearchRequest,
    vector_store: VSDep,
) -> SearchResponse:
    try:
        pipeline_result = run_query(
            query=body.query,
            vector_store=vector_store,
            top_k=body.top_k,
            min_score=body.min_score,
        )

    except IntentExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Intent extraction unavailable: {exc}",
        )
    except (QueryPipelineError, RetrievalError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected search error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    results = pipeline_result.results
    intent  = pipeline_result.intent

    items = [
        AlertResultItem(
            rank=r.rank,
            score=r.score,
            id=r.record.id,
            camera_id=r.record.camera_id,
            timestamp=r.record.timestamp,
            alert_type=r.record.alert_type,
            confidence=r.record.confidence,
            location_label=r.record.location_label,
            image_filename=r.record.image_filename,
            image_b64=r.image_b64,
            extra=r.record.extra,
            # new fields surfaced to the frontend
            label=r.record.label,
            frame_num=r.record.frame_num,
            object_id=r.record.object_id,
            class_id=r.record.class_id,
            bbox=r.record.bbox,
            caption=r.record.caption,
            ocr=r.record.ocr,
        )
        for r in results
    ]

    return SearchResponse(
        query=body.query,
        total=len(items),
        results=items,
        applied_filters=intent.model_dump(exclude_none=True, exclude={"semantic_query"}),
        semantic_query=intent.semantic_query,
    )


# ── Collection stats ──────────────────────────────────────────────────────────

@router.get(
    "/collections",
    response_model=CollectionStatsResponse,
    summary="Stats about the indexed alert collection",
)
async def collection_stats(vector_store: VSDep) -> CollectionStatsResponse:
    return CollectionStatsResponse(
        total_alerts=vector_store.count(),
        cameras=vector_store.list_cameras(),
    )


@router.delete(
    "/alerts/{alert_id}",
    summary="Delete a specific alert by ID",
    status_code=status.HTTP_200_OK,
)
async def delete_alert(alert_id: str, vector_store: VSDep) -> dict:
    vector_store.delete(alert_id)
    return {"deleted": alert_id}


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(vector_store: VSDep) -> HealthResponse:
    cfg = get_settings()
    return HealthResponse(
        status="ok",
        version=cfg.APP_VERSION,
        total_indexed=vector_store.count(),
        clip_model=cfg.CLIP_MODEL_NAME,
    )