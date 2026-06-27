"""
utils/metadata.py

Thin helpers for reading and writing per-alert metadata JSON files.
The file at  data/metadata/<record_id>.json  is the canonical source of
truth for all alert metadata; Chroma only holds the fields needed for
hybrid search (id, camera_id, label, timestamp, confidence, caption,
image_path).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def _meta_path(record_id: str) -> Path:
    meta_dir = Path("data") / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir / f"{record_id}.json"


def save_metadata(record_id: str, data: Dict[str, Any]) -> Path:
    """Serialise *data* to  data/metadata/<record_id>.json  and return the path."""
    path = _meta_path(record_id)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.debug("Saved metadata: %s", path)
    return path


def load_metadata(record_id: str) -> Dict[str, Any]:
    """Load and return the metadata dict for *record_id*.  Returns {} if missing."""
    path = _meta_path(record_id)
    if not path.exists():
        logger.warning("Metadata file not found: %s", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read metadata %s: %s", path, exc)
        return {}


def update_metadata(record_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *updates* into the existing metadata file and re-save it."""
    data = load_metadata(record_id)
    data.update(updates)
    save_metadata(record_id, data)
    
    return data