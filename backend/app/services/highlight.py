"""
Highlight service — person detection + CLIP zero-shot attribute matching.

WHY THIS REPLACES GRAD-CAM (AND WHY IT NO LONGER USES METADATA BBOXES)
------------------------------------------------------------------------
CLIP's global image-text similarity score is pooled over the whole image via
a CLS token. Its gradient w.r.t. that single pooled score is a poor spatial
signal — it tends to activate diffusely (background, floor, furniture edges)
rather than the actual object described by the query, especially for small
attributes like "blue shirt" in a wide surveillance frame with multiple
people. No amount of layer/threshold tuning fixes this: CLIP was never
trained to be dense-spatially-grounded.

What CLIP IS very good at: classifying what's inside a *tight crop*
("is this crop a person wearing a blue shirt or a red shirt or ...").

Additionally, this service NO LONGER uses any bbox from stored alert
metadata. Two independent problems with that:
    1. The stored bbox represents "a person was detected" at ingest time —
       it has no relationship to a color/attribute mentioned in a later
       search query, so using it for a query like "blue shirt" was
       drawing the wrong region entirely.
    2. Images are resized between the Jetson (where the original detection
       and its bbox were computed) and this backend (Windows), so even the
       raw bbox pixel coordinates no longer line up with the image actually
       being rendered here.

So the pipeline now ALWAYS computes fresh boxes on the actual image bytes
in hand:
    1. Detect every person in the frame with a real object detector (YOLO).
       (Shared with the attribute-verification stage — see detection.py.)
    2. Crop each detected person.
    3. Zero-shot classify each crop's clothing color with CLIP
       (crop embedding vs. a small set of color-prompt text embeddings).
    4. Draw a box around every person whose classified color matches the
       query's requested color — so ALL matching people get boxed, not just
       the single best (and often wrong) region in the whole frame.

If the query doesn't mention a recognizable garment/color (e.g. a query
like "person near the exit"), this falls back to a single Grad-CAM box
region, computed fresh on the current image as well.
"""

from __future__ import annotations

import base64
import re
from io import BytesIO
from typing import List, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.services.rag import _load_clip
from backend.app.services.detection import detect_persons as _detect_persons

logger = get_logger(__name__)

logger.info(">>> HIGHLIGHT.PY (YOLO + CLIP color-matching version) LOADED <<<")

# ── Visualization constants ───────────────────────────────────────────────────
_BOX_COLOR = (255, 0, 0)  # Bright red
_BOX_WIDTH = 5
_MAX_BOX_AREA_RATIO = 0.20
_MIN_BOX_SIDE_RATIO = 0.03
_MAX_BOX_SIDE_RATIO = 0.55

# Note: YOLO model loading and person detection now live in
# backend/app/services/detection.py, shared with the attribute-
# verification stage (verifiers/color_verifier.py). `_detect_persons`
# above is that shared function, imported under its old local name so
# the rest of this file (_match_persons_by_color, etc.) is unchanged.


# ── Query parsing: extract requested color + garment ─────────────────────────

_COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "grey", "gray",
    "orange", "purple", "pink", "brown", "navy", "maroon", "beige",
]

_GARMENTS = [
    "shirt", "t-shirt", "tshirt", "jacket", "hoodie", "sweater",
    "coat", "top", "dress", "uniform", "vest", "kurta",
]


def _parse_color_and_garment(text_query: str) -> Tuple[Optional[str], str]:
    """
    Extract a requested clothing color and garment noun from a free-text
    query, e.g. "give me image of all blue shirt persons" -> ("blue", "shirt").
    Returns (color_or_None, garment_defaulting_to_"shirt").
    """
    q = text_query.lower()

    color = next((c for c in _COLORS if re.search(rf"\b{c}\b", q)), None)

    garment = next((g for g in _GARMENTS if re.search(rf"\b{g}\b", q)), "shirt")
    if garment in ("t-shirt", "tshirt"):
        garment = "t-shirt"

    return color, garment


# ── CLIP zero-shot color classification per person crop ──────────────────────

def _classify_crop_colors(
    model,
    processor,
    crops: List[Image.Image],
    garment: str,
    device: str,
) -> List[Tuple[str, float]]:
    """
    For each crop, compute a softmax distribution over _COLORS using CLIP
    zero-shot classification, and return the (argmax_color, probability)
    for each crop.
    """
    if not crops:
        return []

    prompts = [f"a photo of a person wearing a {c} {garment}" for c in _COLORS]

    inputs = processor(
        text=prompts,
        images=crops,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image: [n_crops, n_colors]
        logits = outputs.logits_per_image
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for row in probs:
        idx = int(np.argmax(row))
        results.append((_COLORS[idx], float(row[idx])))
    return results


# ── Fallback: Grad-CAM single-region box (kept for non-attribute queries) ────
# Used only when the query doesn't mention a recognizable color+garment, so
# there's nothing for the person-crop classifier to filter on. Computed
# fresh on the current image, same as everything else in this file.

_CAM_LAYER_OFFSET_FROM_END = 3


def _get_clip_vision_cam_layer(model, layer_offset: int = _CAM_LAYER_OFFSET_FROM_END):
    layers = model.vision_model.encoder.layers
    idx = max(0, len(layers) - 1 - layer_offset)
    return layers[idx]


def _unwrap_layer_output(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def _grad_cam(model, processor, img: Image.Image, text_query: str, device: str) -> np.ndarray:
    inputs = processor(
        images=img, text=[text_query], return_tensors="pt", padding=True, truncation=True,
    ).to(device)
    inputs["pixel_values"].requires_grad_(True)

    activations, gradients = [], []

    def save_activation(module, input, output):
        activations.append(_unwrap_layer_output(output))

    def save_gradient(module, grad_input, grad_output):
        gradients.append(_unwrap_layer_output(grad_output))

    target_layer = _get_clip_vision_cam_layer(model)
    h_act = target_layer.register_forward_hook(save_activation)
    h_grad = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.zero_grad()
        with torch.enable_grad():
            outputs = model(**inputs)
            score = outputs.logits_per_image[0, 0]
            score.backward()
    finally:
        h_act.remove()
        h_grad.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

    act = activations[0].detach().cpu().numpy()[:, 1:, :]
    grad = gradients[0].detach().cpu().numpy()[:, 1:, :]
    weights = np.mean(grad, axis=1, keepdims=True)
    cam = np.maximum(np.sum(weights * act, axis=2), 0)

    patch_size = model.config.vision_config.patch_size
    input_size = processor.image_processor.size["shortest_edge"]
    grid_size = input_size // patch_size
    cam = cam.reshape(1, grid_size, grid_size)

    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)

    try:
        resample_method = Image.Resampling.BILINEAR
    except AttributeError:
        resample_method = Image.BILINEAR

    return np.array(Image.fromarray(cam[0]).resize(img.size, resample_method))


def _get_bounding_box_from_heatmap(heatmap: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    h, w = heatmap.shape
    image_area = float(h * w)
    threshold_percentiles = [99.5, 99.0, 98.5, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 90.0]

    try:
        import cv2
    except ImportError:
        cv2 = None

    best_box, best_score = None, -1.0

    for pct in threshold_percentiles:
        threshold = float(np.percentile(heatmap, pct))
        binary = (heatmap >= threshold).astype(np.uint8)
        if not np.any(binary):
            continue

        if cv2 is None:
            ys, xs = np.where(binary > 0)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area_ratio = ((x2 - x1 + 1) * (y2 - y1 + 1)) / image_area
            if area_ratio > _MAX_BOX_AREA_RATIO:
                continue
            score = float(heatmap[ys, xs].mean()) * (1.0 - area_ratio)
            if score > best_score:
                best_score, best_box = score, (x1, y1, x2, y2)
            continue

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary * 255, connectivity=8)
        if n_labels <= 1:
            continue

        for label in range(1, n_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            if bw < int(_MIN_BOX_SIDE_RATIO * w) or bh < int(_MIN_BOX_SIDE_RATIO * h):
                continue
            if bw > int(_MAX_BOX_SIDE_RATIO * w) or bh > int(_MAX_BOX_SIDE_RATIO * h):
                continue
            area_ratio = (bw * bh) / image_area
            if area_ratio > _MAX_BOX_AREA_RATIO:
                continue
            mask = labels == label
            local_mean = float(heatmap[mask].mean())
            local_peak = float(heatmap[mask].max())
            score = (0.6 * local_peak + 0.4 * local_mean) * (1.0 - area_ratio)
            if score > best_score:
                best_score, best_box = score, (x, y, x + bw - 1, y + bh - 1)

    if best_box is None:
        peak_threshold = float(np.percentile(heatmap, 98.5))
        mask = heatmap >= peak_threshold
        ys, xs = np.where(mask)
        if len(xs) == 0:
            ys, xs = np.indices(heatmap.shape)
            weights = heatmap.flatten()
        else:
            weights = heatmap[ys, xs]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-9:
            x_center, y_center = w // 2, h // 2
        else:
            x_center = int(np.sum(xs * weights) / weight_sum)
            y_center = int(np.sum(ys * weights) / weight_sum)
        box_w, box_h = max(28, int(0.10 * w)), max(28, int(0.10 * h))
        x1 = max(0, x_center - box_w // 2)
        y1 = max(0, y_center - box_h // 2)
        x2 = min(w - 1, x1 + box_w)
        y2 = min(h - 1, y1 + box_h)
        best_box = (x1, y1, x2, y2)

    x1, y1, x2, y2 = best_box
    pad = 8
    return (max(0, x1 - pad), max(0, y1 - pad), min(w - 1, x2 + pad), min(h - 1, y2 + pad))


# ── Public API ────────────────────────────────────────────────────────────────

# Minimum classifier confidence to accept a color match. Tune this if you're
# getting too many false positives (raise it) or missing real matches
# (lower it). 0.30 is a reasonable starting point for a 15-way softmax.
_COLOR_MATCH_MIN_PROB = 0.30


def highlight_image_b64(
    image_b64: str,
    text_query: str,
    record_bbox: Mapping[str, int] | None = None,
) -> str:
    """
    Draws bounding boxes around every person matching the requested clothing
    color in *text_query* (e.g. "blue shirt person" -> boxes every detected
    person whose shirt is classified as blue).

    IMPORTANT: `record_bbox` is accepted for backward-compatible call
    signature but is intentionally IGNORED. Stored metadata boxes are
    computed at ingest time on the Jetson and (a) don't correspond to any
    attribute mentioned in a later search query, and (b) can be pixel-
    misaligned with this image if it was resized in transit. All boxes are
    computed fresh, on the actual image bytes in hand, every call.

    Falls back to a single Grad-CAM region if the query has no recognizable
    color+garment (so there's nothing to filter person crops on).

    Returns a new base64-encoded JPEG with box(es) drawn.
    """
    if not image_b64:
        return image_b64

    cfg = get_settings()
    device = cfg.EMBEDDING_DEVICE

    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        logger.error(f"Failed to decode base64 image: {exc}")
        return image_b64

    try:
        color, garment = _parse_color_and_garment(text_query)
        logger.info(
            "highlight_image_b64: parsed color=%r garment=%r from query=%r (image=%dx%d)",
            color, garment, text_query, img.size[0], img.size[1],
        )

        if color:
            boxes = _match_persons_by_color(img, color, garment, device)
            if boxes:
                return _draw_boxes_and_encode(img, boxes)
            logger.info(
                "No person matched color=%r garment=%r for query %r — "
                "drawing nothing (frame likely has no such person).",
                color, garment, text_query,
            )
            # Intentionally return the undecorated image rather than a
            # misleading fallback box: if the frame genuinely has no
            # matching person, boxing something else would be worse.
            return image_b64

        # No color/garment in the query (e.g. "give me all images", "show
        # persons near the exit") — there is no specific attribute to
        # localize, so we deliberately do NOT draw a box. Grad-CAM was
        # previously used here as a fallback, but it produces the same
        # unreliable, misleading boxes on random background regions that
        # motivated ripping it out of the color-matching path in the first
        # place — there's no reason to keep it around for the "no attribute"
        # case either. Better to show a clean, unmodified image than a
        # confidently wrong box.
        logger.info(
            "Query %r has no recognizable color/garment — returning image "
            "unmodified (no localization target).",
            text_query,
        )
        return image_b64

    except Exception as exc:
        logger.error(f"Failed to compute highlight: {exc}", exc_info=True)
        return image_b64


_CROP_MIN_SIDE_FOR_CLASSIFICATION = 32  # pixels; below this, upscale before CLIP


def _prep_crop_for_classification(crop: Image.Image) -> Image.Image:
    """
    Upscale very small crops before handing them to CLIP. This doesn't
    invent detail, but it avoids CLIPProcessor's own internal resize
    silently upsampling with a lower-quality default filter, and gives a
    slightly more stable signal for tiny crops.
    """
    w, h = crop.size
    if min(w, h) >= _CROP_MIN_SIDE_FOR_CLASSIFICATION:
        return crop
    scale = _CROP_MIN_SIDE_FOR_CLASSIFICATION / max(1, min(w, h))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS
    return crop.resize(new_size, resample_method)


def _match_persons_by_color(
    img: Image.Image,
    color: str,
    garment: str,
    device: str,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect all persons, classify each crop's garment color, keep matches.

    Fallback: if YOLO finds nobody, this does NOT mean "no matches" — the
    caller only ever passes in images that already passed a `label=person`
    filter upstream (i.e. the alert pipeline already confirmed a person is
    in this image). A missed YOLO detection on a low-resolution/thumbnail
    image is far more likely than a genuinely person-free image reaching
    this function. So when detection comes back empty, we classify the
    WHOLE image as if it were the person crop, rather than silently
    dropping a real result.

    Note: this whole-image fallback is specific to the *visualization*
    path (drawing a box). The verification path (verifiers/color_verifier.py)
    deliberately does NOT do this — it returns UNKNOWN instead, since a
    wrongly-confident box drawn here is a cosmetic issue, whereas a
    wrongly-confident MATCH/NO_MATCH in verification silently changes
    which images the user sees at all.
    """
    person_boxes = _detect_persons(img)

    if not person_boxes:
        logger.info(
            "No persons detected by YOLO — falling back to whole-image "
            "classification (this image was already confirmed to contain "
            "a person by the upstream alert filter)."
        )
        person_boxes = [(0, 0, img.size[0] - 1, img.size[1] - 1)]

    crops = [_prep_crop_for_classification(img.crop(b)) for b in person_boxes]
    model, processor = _load_clip()
    classifications = _classify_crop_colors(model, processor, crops, garment, device)

    matched: List[Tuple[int, int, int, int]] = []
    for box, (pred_color, prob) in zip(person_boxes, classifications):
        logger.info(
            "Person crop %s -> predicted color=%s (p=%.3f), target=%s",
            box, pred_color, prob, color,
        )
        if pred_color == color and prob >= _COLOR_MATCH_MIN_PROB:
            matched.append(box)

    logger.info("Matched %d/%d persons to color=%r", len(matched), len(person_boxes), color)
    return matched


def _draw_boxes_and_encode(img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> str:
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([(x1, y1), (x2, y2)], outline=_BOX_COLOR, width=_BOX_WIDTH)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")