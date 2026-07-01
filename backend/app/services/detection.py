"""
Shared object detection utilities (YOLO).

Both highlight.py (visualization boxes) and the attribute verifiers
(verifiers/*) need "find people / objects in this image" — this module
is the single place that owns the YOLO model and the person-detection
logic, so neither caller duplicates it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image

from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_YOLO_MODEL_NAME = "yolov8n.pt"   # tiny, fast, CPU-friendly
_YOLO_PERSON_CLASS_ID = 0          # COCO class 0 = "person"
_YOLO_CONF_THRESHOLD = 0.35
_YOLO_MIN_INPUT_SIZE = 640          # YOLOv8's native training resolution

_yolo_model = None


def load_yolo():
    """Lazily load and cache the YOLO detector (shared across the app)."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install ultralytics` "
                "to enable object/person detection."
            ) from exc
        logger.info("Loading YOLO detector: %s", _YOLO_MODEL_NAME)
        _yolo_model = YOLO(_YOLO_MODEL_NAME)
        logger.info("YOLO ready.")
    return _yolo_model


def _upscale_if_small(img: Image.Image) -> Tuple[Image.Image, float]:
    """
    Upscale small images before YOLO for better recall.

    YOLOv8 was trained at 640px. If the incoming image's longer side is
    well below that (common with resized/thumbnail alert images), person/
    object detection recall drops sharply. We upscale before detection and
    the caller maps boxes back to the original image's coordinate space.
    This does NOT recover detail lost by an earlier downscale — it just
    gives the detector a fairer shot at what's actually there.

    Returns (image_to_run_detection_on, scale_factor_applied).
    """
    orig_w, orig_h = img.size
    longer_side = max(orig_w, orig_h)
    if longer_side >= _YOLO_MIN_INPUT_SIZE:
        return img, 1.0

    scale = _YOLO_MIN_INPUT_SIZE / longer_side
    new_size = (int(round(orig_w * scale)), int(round(orig_h * scale)))
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS
    resized = img.resize(new_size, resample_method)
    logger.info(
        "Upscaling small image %dx%d -> %dx%d before YOLO (scale=%.2f)",
        orig_w, orig_h, new_size[0], new_size[1], scale,
    )
    return resized, scale


def _boxes_from_result(
    r, orig_w: int, orig_h: int, scale: float
) -> List[Tuple[int, int, int, int, str, float]]:
    """Yield (x1, y1, x2, y2, class_name, conf) mapped back to original coords."""
    out: List[Tuple[int, int, int, int, str, float]] = []
    if r.boxes is None:
        return out
    names = r.names
    for b in r.boxes:
        xyxy = b.xyxy[0].tolist()
        x1, y1, x2, y2 = (v / scale for v in xyxy)
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        out.append((x1, y1, x2, y2, names.get(cls_id, str(cls_id)), conf))
    return out


def detect_persons(img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detect every person in *img*, always computed fresh on the actual image
    bytes in hand (never from stored metadata — the stored bbox has no
    relation to a later query and can be pixel-misaligned after resizing
    in transit). Returns (x1, y1, x2, y2) boxes in original image
    coordinates.
    """
    orig_w, orig_h = img.size
    detect_img, scale = _upscale_if_small(img)

    model = load_yolo()
    results = model.predict(
        source=detect_img,
        classes=[_YOLO_PERSON_CLASS_ID],
        conf=_YOLO_CONF_THRESHOLD,
        verbose=False,
    )

    boxes: List[Tuple[int, int, int, int]] = []
    for r in results:
        for x1, y1, x2, y2, _name, _conf in _boxes_from_result(r, orig_w, orig_h, scale):
            boxes.append((x1, y1, x2, y2))

    logger.info("YOLO detected %d person(s) in %dx%d image", len(boxes), orig_w, orig_h)
    return boxes


def detect_persons_with_conf(img: Image.Image) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """
    Detect persons and return [((x1, y1, x2, y2), confidence), ...].

    This is useful for reranking pipelines that want to reward matches with
    stronger detector evidence rather than treating all detections equally.
    """
    orig_w, orig_h = img.size
    detect_img, scale = _upscale_if_small(img)

    model = load_yolo()
    results = model.predict(
        source=detect_img,
        classes=[_YOLO_PERSON_CLASS_ID],
        conf=_YOLO_CONF_THRESHOLD,
        verbose=False,
    )

    detections: List[Tuple[Tuple[int, int, int, int], float]] = []
    for r in results:
        for x1, y1, x2, y2, _name, conf in _boxes_from_result(r, orig_w, orig_h, scale):
            detections.append(((x1, y1, x2, y2), float(conf)))

    logger.info("YOLO detected %d person(s) in %dx%d image", len(detections), orig_w, orig_h)
    return detections


def detect_objects(
    img: Image.Image,
    class_names: List[str],
    conf_threshold: float = _YOLO_CONF_THRESHOLD,
) -> Dict[str, List[Tuple[Tuple[int, int, int, int], float]]]:
    """
    Detect specific COCO object classes in *img* (e.g. "bottle", "backpack",
    "cell phone", "laptop"). Returns {class_name: [(box, conf), ...]}.

    Only classes that exist in the YOLO model's label set are queried; a
    class name YOLO doesn't know about (e.g. "helmet", not in COCO) is
    silently skipped so callers can fall back to a different verifier
    (see verifiers/clip_zero_shot_verifier.py).
    """
    model = load_yolo()
    name_to_id = {v: k for k, v in model.names.items()}
    class_ids = [name_to_id[c] for c in class_names if c in name_to_id]

    result: Dict[str, List[Tuple[Tuple[int, int, int, int], float]]] = {c: [] for c in class_names}
    if not class_ids:
        return result

    orig_w, orig_h = img.size
    detect_img, scale = _upscale_if_small(img)

    results = model.predict(
        source=detect_img,
        classes=class_ids,
        conf=conf_threshold,
        verbose=False,
    )
    for r in results:
        for x1, y1, x2, y2, name, conf in _boxes_from_result(r, orig_w, orig_h, scale):
            if name in result:
                result[name].append(((x1, y1, x2, y2), conf))

    return result