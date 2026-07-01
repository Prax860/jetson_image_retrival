"""
Attribute-query parsing + verifier registry.

To add a new pluggable attribute (e.g. `verify_pose`):
    1. Write a class implementing AttributeVerifier.
    2. Add its trigger keyword(s) to the appropriate keyword map below
       (or add a new small parse branch if it doesn't fit color/object/
       zero-shot), and register the verifier singleton + type key.
No other file needs to change — retrieval.py and reranker.py only ever
call parse_attribute_query() / get_verifier().
"""

from __future__ import annotations

import re
from typing import Optional

from backend.app.services.verifiers.base import AttributeQuery, AttributeVerifier
from backend.app.services.color_utils import COLORS
from backend.app.services.verifiers.color_verifier import ColorAttributeVerifier
from backend.app.services.verifiers.object_verifier import OBJECT_KEYWORDS, CocoObjectVerifier
from backend.app.services.verifiers.clip_zero_shot_verifier import ClipZeroShotAttributeVerifier

_GARMENTS = [
    "shirt", "t-shirt", "tshirt", "jacket", "hoodie", "sweater",
    "coat", "top", "dress", "uniform", "vest", "kurta",
]

# Attributes with no COCO class, handled by CLIP zero-shot instead.
_ZERO_SHOT_KEYWORDS = ["helmet", "mask", "high-vis vest", "hi-vis vest", "hard hat"]

# Singletons — verifiers are stateless wrappers around shared model singletons
# (CLIP / YOLO), so one instance per process is enough.
_COLOR_VERIFIER = ColorAttributeVerifier()
_OBJECT_VERIFIER = CocoObjectVerifier()
_ZERO_SHOT_VERIFIER = ClipZeroShotAttributeVerifier()

_VERIFIER_BY_TYPE = {
    "color": _COLOR_VERIFIER,
    "object": _OBJECT_VERIFIER,
    "clip_zero_shot": _ZERO_SHOT_VERIFIER,
}


def parse_attribute_query(text_query: str) -> Optional[AttributeQuery]:
    """
    Inspect a free-text query and decide whether it names a verifiable
    attribute. Returns None for queries with nothing to verify (e.g.
    "show all persons", "camera 5") — callers must treat None as
    "skip verification, behave exactly as before" (backward compatible).
    """
    q = text_query.lower()

    # 1. Color + garment (e.g. "blue shirt")
    color = next((c for c in COLORS if re.search(rf"\b{c}\b", q)), None)
    if color:
        garment = next((g for g in _GARMENTS if re.search(rf"\b{g}\b", q)), "shirt")
        if garment in ("t-shirt", "tshirt"):
            garment = "t-shirt"
        return AttributeQuery(raw_query=text_query, attribute_type="color", value=color, garment=garment)

    # 2. COCO-recognizable object (e.g. "bottle", "phone", "backpack")
    for keyword, coco_class in OBJECT_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", q):
            return AttributeQuery(raw_query=text_query, attribute_type="object", value=coco_class)

    # 3. Non-COCO attribute -> CLIP zero-shot fallback (e.g. "helmet")
    for keyword in _ZERO_SHOT_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", q):
            return AttributeQuery(raw_query=text_query, attribute_type="clip_zero_shot", value=keyword)

    return None


def get_verifier(attribute_query: AttributeQuery) -> AttributeVerifier:
    return _VERIFIER_BY_TYPE[attribute_query.attribute_type]