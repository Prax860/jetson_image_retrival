"""
Generic object-presence verifier: "person holding a bottle",
"person with a backpack", "person using a phone", "person carrying a
laptop" — anything that maps to a COCO class YOLO already knows.

A "no detection" result only counts as confident absence (NO_MATCH) if
the image had enough resolution to give the detector a fair chance.
Below that, small/occluded objects are known to get missed — so a miss
on a low-res image is UNKNOWN, not NO_MATCH.
"""

from __future__ import annotations

from PIL import Image

from backend.app.core.logging import get_logger
from backend.app.services.detection import detect_objects
from backend.app.services.verifiers.base import AttributeQuery, AttributeVerifier, VerificationResult

logger = get_logger(__name__)

# query keyword -> canonical COCO class name
OBJECT_KEYWORDS = {
    "bottle": "bottle",
    "backpack": "backpack",
    "bag": "backpack",
    "phone": "cell phone",
    "mobile": "cell phone",
    "cellphone": "cell phone",
    "cell phone": "cell phone",
    "laptop": "laptop",
    "umbrella": "umbrella",
    "handbag": "handbag",
    "suitcase": "suitcase",
}

# Below this, a small/handheld object (bottle, phone) is genuinely hard
# for a detector to catch even when present — a miss isn't meaningful
# evidence of absence.
_RELIABLE_DETECTION_MIN_SIDE = 480


class CocoObjectVerifier(AttributeVerifier):
    """Verifies presence of a YOLO/COCO-recognizable object in the image."""

    def verify(self, image: Image.Image, attribute_query: AttributeQuery) -> VerificationResult:
        coco_class = attribute_query.value
        try:
            detections = detect_objects(image, [coco_class])
            hits = detections.get(coco_class, [])

            if hits:
                best_conf = max(conf for _box, conf in hits)
                return VerificationResult(
                    matched=True,
                    confidence=best_conf,
                    matched_count=len(hits),
                    detail=f"{coco_class} detected ({len(hits)}x)",
                )

            # No detection. Was the image good enough to trust that verdict?
            longer_side = max(image.size)
            if longer_side < _RELIABLE_DETECTION_MIN_SIDE:
                return VerificationResult(
                    matched=None,
                    detail=(
                        f"no {coco_class} detected, but image resolution "
                        f"{image.size} is low — a miss is plausible"
                    ),
                )

            return VerificationResult(
                matched=False, detail=f"no {coco_class} detected in a high-enough-resolution image"
            )

        except Exception as exc:
            logger.warning("CocoObjectVerifier failed, marking UNKNOWN: %s", exc)
            return VerificationResult(matched=None, detail=str(exc))