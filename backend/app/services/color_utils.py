"""
Shared color-matching utilities for clothing-attribute verification and
highlighting.

Two responsibilities live here, used identically by both
verifiers/color_verifier.py and highlight.py so they can never diverge
again:

1. Torso-crop heuristic — cut a YOLO person box down to roughly
   "shoulders to mid-chest" before classification, instead of feeding
   CLIP the entire box (head, hair, arms, legs, chair, desk, floor...).
   No pose estimation or segmentation; fixed fractions of the existing
   box height/width.

2. Canonical color matching via prompt-variant aggregation — CLIP is
   asked about several more specific phrasings per canonical color
   ("navy blue shirt", "royal blue shirt", "sky blue shirt", ...) and
   their softmax probability mass is summed back into one canonical
   bucket ("blue"). This both gives CLIP easier, more separable prompts
   and directly implements "blue should also match navy/sky blue/royal
   blue" — the aggregation IS the similarity mapping.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# ── Canonical colors + prompt variants ────────────────────────────────────

CANONICAL_COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "gray",
    "orange", "purple", "pink", "brown", "beige",
]

# Every phrase here is used as a separate CLIP prompt; probabilities are
# summed back into the canonical key. This is what makes "navy" count as
# a "blue" match, "grey" count as "gray", etc., without needing intent
# extraction or retrieval changes.
_COLOR_VARIANTS: Dict[str, List[str]] = {
    "red": ["red", "maroon", "crimson", "dark red", "burgundy"],
    "blue": ["blue", "navy blue", "navy", "royal blue", "sky blue", "light blue", "dark blue", "denim blue"],
    "green": ["green", "olive green", "dark green", "teal green", "forest green"],
    "yellow": ["yellow", "golden yellow", "mustard yellow"],
    "black": ["black"],
    "white": ["white", "off white", "cream white"],
    "gray": ["gray", "grey", "silver", "charcoal gray", "charcoal grey"],
    "orange": ["orange", "burnt orange"],
    "purple": ["purple", "violet", "lavender"],
    "pink": ["pink", "hot pink", "light pink"],
    "brown": ["brown", "tan", "khaki", "khaki brown"],
    "beige": ["beige", "cream", "ivory"],
}

# Direct synonym -> canonical lookup, for canonicalizing a raw color word
# that arrives from a query (e.g. attribute_query.value == "grey").
_SYNONYM_TO_CANONICAL: Dict[str, str] = {}
for _canon, _variants in _COLOR_VARIANTS.items():
    _SYNONYM_TO_CANONICAL[_canon] = _canon
    for _v in _variants:
        _SYNONYM_TO_CANONICAL[_v] = _canon
# A few extra synonyms that aren't standalone CLIP prompts but should
# still canonicalize correctly if they show up in free text.
_SYNONYM_TO_CANONICAL.update({
    "grey": "gray",
    "navy": "blue",
    "teal": "green",
    "maroon": "red",
    "khaki": "brown",
    "tan": "brown",
    "cream": "beige",
    "ivory": "beige",
    "violet": "purple",
    "lavender": "purple",
})

# Flat prompt list + parallel list of which canonical bucket each belongs to.
_ALL_VARIANT_PHRASES: List[str] = []
_VARIANT_CANONICAL_INDEX: List[str] = []
for _canon, _variants in _COLOR_VARIANTS.items():
    for _v in _variants:
        _ALL_VARIANT_PHRASES.append(_v)
        _VARIANT_CANONICAL_INDEX.append(_canon)


def canonicalize_color(color: str) -> str:
    """Map any recognized color word/phrase to its canonical bucket.
    Unrecognized input is returned lowercased/stripped, unchanged."""
    key = color.strip().lower()
    return _SYNONYM_TO_CANONICAL.get(key, key)


# ── Torso-crop heuristic ──────────────────────────────────────────────────

_TORSO_TOP_FRAC = 0.12       # skip head/hair band
_TORSO_BOTTOM_FRAC = 0.55    # stop above legs/chair/desk
_TORSO_HPAD_FRAC = 0.08      # small horizontal pad so both shoulders stay in frame
_MIN_TORSO_SIDE = 8          # if the computed torso crop degenerates below this, fall back to full box

# Backward compatibility
COLORS = CANONICAL_COLORS
def extract_torso_crop(image: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Given a person's full YOLO bounding box, return a heuristic torso
    box (shoulders -> mid-chest) instead of the full head-to-feet box.

    This is intentionally crude — fixed fractions of the box, no pose
    estimation or segmentation — but it reliably drops most of the head,
    hair, legs, chair, and desk that dilute a full-body crop.

    Returns a box clamped to image bounds. Falls back to the original
    box if the computed torso region degenerates (e.g. extremely short
    detections where fractional cropping leaves near-zero height).
    """
    img_w, img_h = image.size
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return box

    ty1 = y1 + int(round(h * _TORSO_TOP_FRAC))
    ty2 = y1 + int(round(h * _TORSO_BOTTOM_FRAC))
    pad = int(round(w * _TORSO_HPAD_FRAC))
    tx1 = x1 - pad
    tx2 = x2 + pad

    tx1 = max(0, tx1)
    ty1 = max(0, ty1)
    tx2 = min(img_w - 1, tx2)
    ty2 = min(img_h - 1, ty2)

    if (tx2 - tx1) < _MIN_TORSO_SIDE or (ty2 - ty1) < _MIN_TORSO_SIDE:
        # Degenerate torso crop (e.g. tiny detection) — fall back to the
        # full box rather than handing CLIP a near-empty image.
        return box

    return (tx1, ty1, tx2, ty2)


# ── Canonical zero-shot color classification ──────────────────────────────

def classify_crop_canonical_colors(
    model, processor, crops: List[Image.Image], garment: str, device: str,
) -> List[Dict[str, float]]:
    """
    For each crop, run CLIP zero-shot classification against every color
    VARIANT phrase (not just canonical names), softmax across all
    variants, then sum probability mass back into canonical buckets.

    Returns one dict per crop: {canonical_color: aggregated_probability},
    covering all of CANONICAL_COLORS (values sum to ~1.0 per crop).
    """
    if not crops:
        return []

    prompts = [f"a photo of a person wearing a {phrase} {garment}" for phrase in _ALL_VARIANT_PHRASES]
    inputs = processor(text=prompts, images=crops, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits_per_image, dim=-1).cpu().numpy()  # [n_crops, n_variants]

    results: List[Dict[str, float]] = []
    for row in probs:
        bucket: Dict[str, float] = {c: 0.0 for c in CANONICAL_COLORS}
        for prob, canon in zip(row, _VARIANT_CANONICAL_INDEX):
            bucket[canon] += float(prob)
        results.append(bucket)
    return results


def top_color(canonical_probs: Dict[str, float]) -> Tuple[str, float]:
    """Return (canonical_color, prob) for the highest-probability bucket."""
    best_color = max(canonical_probs, key=canonical_probs.get)
    return best_color, canonical_probs[best_color]