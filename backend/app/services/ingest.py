"""
Ingest service.

Handles a single alert push from the Jetson:
1. Validate the image (extension, size).
2. Save it to IMAGE_STORE_DIR / camera_id / <filename>.
3. Write a canonical metadata JSON to data/metadata/<record_id>.json.
4. Build and return an AlertRecord (no embedding here — that's rag.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from backend.app.core.config import get_settings
from backend.app.core.exceptions import IngestError
from backend.app.core.logging import get_logger
from backend.app.models.alert import AlertRecord, BBox
from backend.app.utils.metadata import save_metadata

logger = get_logger(__name__)


def ingest_alert(
    image_bytes: bytes,
    original_filename: str,
    camera_id: str,
    timestamp: datetime,
    alert_type: Optional[str] = None,
    confidence: Optional[float] = None,
    location_label: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    # ── NEW: Jetson detection fields ──────────────────────────────────────────
    label: Optional[str] = None,
    frame_num: Optional[int] = None,
    object_id: Optional[int] = None,
    class_id: Optional[int] = None,
    bbox: Optional[Dict[str, Any]] = None,   # raw dict from JSON form field
) -> AlertRecord:
    """
    Validate, save, and create an AlertRecord for one Jetson alert.

    Parameters
    ----------
    image_bytes:       Raw image bytes from the upload.
    original_filename: Filename sent by the client (used to determine extension).
    camera_id:         Identifier of the camera that triggered the alert.
    timestamp:         Datetime of the alert event (from the device).
    alert_type:        Optional classification label (motion, person, vehicle…).
    confidence:        Optional detection confidence in [0, 1].
    location_label:    Human-readable camera location description.
    extra:             Any additional key-value metadata from the device.
    label:             Detection class label from the Jetson (e.g. "person").
    frame_num:         Frame number within the video stream.
    object_id:         Tracker object ID.
    class_id:          Numeric class index from the detector.
    bbox:              Bounding box dict {left, top, width, height}.

    Returns
    -------
    AlertRecord with image_path and metadata_path populated.
    """
    cfg = get_settings()

    # 1. Extension check
    suffix = Path(original_filename).suffix.lower()
    if suffix not in cfg.ALLOWED_IMAGE_EXTENSIONS:
        raise IngestError(
            f"Unsupported image type '{suffix}'. "
            f"Allowed: {cfg.ALLOWED_IMAGE_EXTENSIONS}"
        )

    # 2. Size check
    size_mb = len(image_bytes) / 1_048_576
    if size_mb > cfg.MAX_IMAGE_SIZE_MB:
        raise IngestError(
            f"Image too large: {size_mb:.1f} MB > {cfg.MAX_IMAGE_SIZE_MB} MB limit"
        )

    # 3. Verify it's actually a valid image
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as exc:
        raise IngestError(f"Invalid image data: {exc}") from exc

    # 4. Build a stable filename: <record_id><suffix>
    record_id = str(uuid.uuid4())
    filename = f"{record_id}{suffix}"

    # 5. Save under IMAGE_STORE_DIR / camera_id /
    camera_dir = cfg.IMAGE_STORE_DIR / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    save_path = camera_dir / filename
    save_path.write_bytes(image_bytes)
    logger.info("Saved alert image: %s", save_path)

    # 6. Parse bbox into a BBox model if provided
    bbox_model: Optional[BBox] = None
    if bbox:
        try:
            bbox_model = BBox(**bbox)
        except Exception as exc:
            logger.warning("Could not parse bbox %s: %s", bbox, exc)

    # 7. Write canonical metadata JSON
    meta_dict: Dict[str, Any] = {
        "id": record_id,
        "camera_id": camera_id,
        "timestamp": timestamp.isoformat(),
        "label": label,
        "confidence": confidence,
        "alert_type": alert_type,
        "frame_num": frame_num,
        "object_id": object_id,
        "class_id": class_id,
        "bbox": bbox_model.model_dump() if bbox_model else None,
        "image_path": str(save_path),
        "image_filename": filename,
        "location_label": location_label,
        "caption": "",
        "ocr": "",
        "extra": extra or {},
    }
    # First save to learn the path, then stamp metadata_path into the JSON
    # so load_metadata() callers always get the full self-referential record.
    meta_path = save_metadata(record_id, meta_dict)
    meta_dict["metadata_path"] = str(meta_path)
    save_metadata(record_id, meta_dict)
    logger.info("Saved alert metadata: %s", meta_path)

    return AlertRecord(
        id=record_id,
        image_path=str(save_path),
        image_filename=filename,
        camera_id=camera_id,
        timestamp=timestamp,
        alert_type=alert_type,
        confidence=confidence,
        location_label=location_label,
        extra=extra or {},
        label=label,
        frame_num=frame_num,
        object_id=object_id,
        class_id=class_id,
        bbox=bbox_model,
        metadata_path=str(meta_path),
    )