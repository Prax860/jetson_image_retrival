"""
models/alert.py

Domain models for alert records and search results.

AlertRecord holds all per-alert data.  The new fields (label, frame_num,
object_id, class_id, bbox, metadata_path) match what the Jetson api_worker
sends; they default to None / {} so existing call-sites stay compatible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0


class AlertRecord(BaseModel):
    # ── Core identity ─────────────────────────────────────────────────────────
    id: str
    camera_id: str
    timestamp: datetime

    # ── Image ─────────────────────────────────────────────────────────────────
    image_path: str
    image_filename: str

    # ── Detection metadata (original fields) ──────────────────────────────────
    alert_type: Optional[str] = None
    confidence: Optional[float] = None
    location_label: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    # ── NEW: Jetson detection fields ──────────────────────────────────────────
    label: Optional[str] = None          # e.g. "person", "vehicle"
    frame_num: Optional[int] = None
    object_id: Optional[int] = None
    class_id: Optional[int] = None
    bbox: Optional[BBox] = None

    # ── NEW: path to canonical metadata JSON ──────────────────────────────────
    metadata_path: Optional[str] = None

    # ── Enrichment placeholders ───────────────────────────────────────────────
    caption: str = ""
    ocr: str = ""

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    record: AlertRecord
    score: float
    rank: int
    image_b64: str = ""


# ── API / schema helpers (kept here so imports stay stable) ──────────────────

class AlertResultItem(BaseModel):
    rank: int
    score: float
    id: str
    camera_id: str
    timestamp: datetime
    alert_type: Optional[str] = None
    confidence: Optional[float] = None
    location_label: Optional[str] = None
    image_filename: str
    image_b64: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)
    # expose enriched fields to the frontend
    label: Optional[str] = None
    frame_num: Optional[int] = None
    object_id: Optional[int] = None
    class_id: Optional[int] = None
    bbox: Optional[BBox] = None
    caption: str = ""
    ocr: str = ""