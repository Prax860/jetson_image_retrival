"""
Color-attribute verifier: "person wearing a blue shirt", etc.

Detects people in the candidate image (via detection.detect_persons_with_conf)
and zero-shot classifies each detection's TORSO region (not the full
head-to-feet crop) using canonical color matching (see color_utils.py),
so that "blue" also matches navy/sky blue/royal blue, "gray" also
matches grey, etc.

Distinguishes:
  - MATCH:    one or more torso crops clear _MATCH_MIN_PROB on the target
              canonical color. Checked BEFORE the general reliability
              floor. ALL matching people are collected (matched_people),
              not just the single best one.
  - NO_MATCH: every usable crop was confidently classified as SOME OTHER
              canonical color.
  - UNKNOWN:  no person was detected at all, or a crop's own top-color
              confidence was low, or the ORIGINAL detection (not the
              torso sub-crop) was too small to trust. None of these are
              evidence the color is absent.
"""

from __future__ import annotations

from typing import List, Tuple

from PIL import Image

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.services.color_utils import (
    canonicalize_color,
    classify_crop_canonical_colors,
    top_color,
)
from backend.app.services.detection import detect_persons_with_conf
from backend.app.services.rag import _load_clip
from backend.app.services.verifiers.base import AttributeQuery, AttributeVerifier, VerificationResult

logger = get_logger(__name__)

_CROP_MIN_SIDE = 32                     # upscale crops smaller than this before CLIP
_MATCH_MIN_PROB = 0.30                  # threshold on the TARGET canonical color's aggregated prob to accept MATCH
_CLASSIFIER_CONFIDENCE_FLOOR = 0.35     # top-1 canonical prob below this -> classifier itself is unsure -> UNKNOWN
_LOW_RES_ORIGINAL_SIDE = 40             # original (full-body) detection's min side below this -> unreliable -> UNKNOWN
_YOLO_MATCH_BONUS_WEIGHT = 0.35         # boosts final MATCH confidence when detector evidence is strong


def _prep_crop(crop: Image.Image) -> Image.Image:
    """Upscale very small crops before CLIP for a slightly more stable signal."""
    w, h = crop.size
    if min(w, h) >= _CROP_MIN_SIDE:
        return crop
    scale = _CROP_MIN_SIDE / max(1, min(w, h))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS
    return crop.resize(new_size, resample_method)


def _blend_match_confidence(color_prob: float, yolo_conf: float) -> float:
    """
    Combine canonical-color-classification confidence with YOLO detection
    confidence. Gives reranking an explicit boost when YOLO localization
    is strong, while still requiring CLIP to agree on the requested color.
    """
    color_prob = max(0.0, min(1.0, float(color_prob)))
    yolo_conf = max(0.0, min(1.0, float(yolo_conf)))
    blended = (1.0 - _YOLO_MATCH_BONUS_WEIGHT) * color_prob + _YOLO_MATCH_BONUS_WEIGHT * yolo_conf
    return max(0.0, min(1.0, blended))


class ColorAttributeVerifier(AttributeVerifier):
    """Verifies "does this image contain a person wearing <color> <garment>?"."""

    def verify(self, image: Image.Image, attribute_query: AttributeQuery) -> VerificationResult:
        cfg = get_settings()
        device = cfg.EMBEDDING_DEVICE
        color = canonicalize_color(attribute_query.value)
        garment = attribute_query.garment or "shirt"

        try:
            person_detections = detect_persons_with_conf(image)

            if not person_detections:
                # Zero detections is NOT proof of "no matching person" —
                # detectors miss small/occluded/low-res people constantly.
                return VerificationResult(matched=None, detail="no person detected — cannot verify")

            # Import here to avoid a module-level import cycle with
            # highlight.py, which also uses this heuristic.
            from backend.app.services.color_utils import extract_torso_crop

            # Keep each box alongside its torso crop and metadata so a
            # MATCH can report exactly which full-person boxes matched.
            crops_with_meta = []
            for box, yolo_conf in person_detections:
                x1, y1, x2, y2 = box
                orig_side = min(x2 - x1, y2 - y1)
                torso_box = extract_torso_crop(image, box)
                torso_crop = _prep_crop(image.crop(torso_box))
                crops_with_meta.append((torso_crop, orig_side, yolo_conf, box))

            model, processor = _load_clip()
            crops = [c for c, _side, _yolo_conf, _box in crops_with_meta]
            canonical_dists = classify_crop_canonical_colors(model, processor, crops, garment, device)

            matched_boxes: List[Tuple[int, int, int, int]] = []
            matched_confidences: List[float] = []
            any_unknown_crop = False
            any_confident_non_match = False

            for canonical_probs, (_crop, orig_side, yolo_conf, box) in zip(canonical_dists, crops_with_meta):
                # Low-res gate is based on the ORIGINAL full-person
                # detection size, not the (smaller) torso sub-crop —
                # that's the right signal for "was this detection even
                # big enough to trust at all."
                low_res = orig_side < _LOW_RES_ORIGINAL_SIDE

                target_prob = canonical_probs.get(color, 0.0)
                top_canon, top_prob = top_color(canonical_probs)

                # Check the target-color match FIRST, before applying the
                # general reliability floor. A crop whose aggregated
                # probability mass on the TARGET color clears
                # _MATCH_MIN_PROB is real positive evidence, even if the
                # classifier's overall top-1 confidence is unremarkable —
                # the floor exists to catch crops with no real opinion at
                # all, not to override a threshold-clearing hit on the
                # exact color being searched for.
                if not low_res and target_prob >= _MATCH_MIN_PROB:
                    blended_prob = _blend_match_confidence(target_prob, yolo_conf)
                    if blended_prob >= _MATCH_MIN_PROB:
                        matched_boxes.append(box)
                        matched_confidences.append(blended_prob)
                        continue

                low_confidence = top_prob < _CLASSIFIER_CONFIDENCE_FLOOR

                if low_res or low_confidence:
                    # Can't trust this crop either way — might be hiding
                    # the target color.
                    any_unknown_crop = True
                    continue

                if top_canon != color:
                    # Classifier was confident, and confident it's a
                    # DIFFERENT canonical color. Real evidence against a
                    # match for this specific crop.
                    any_confident_non_match = True
                else:
                    any_unknown_crop = True

            if matched_boxes:
                best_confidence = max(matched_confidences)
                return VerificationResult(
                    matched=True,
                    confidence=best_confidence,
                    best_confidence=best_confidence,
                    matched_count=len(matched_boxes),
                    matched_people=matched_boxes,
                    detail=f"{len(matched_boxes)} {color} {garment}(s) found",
                )

            if any_unknown_crop:
                return VerificationResult(
                    matched=None,
                    detail=f"{len(person_detections)} person(s), at least one low-confidence/low-res crop",
                )

            if any_confident_non_match:
                return VerificationResult(
                    matched=False,
                    detail=f"all {len(person_detections)} person(s) confidently classified as other colors",
                )

            return VerificationResult(matched=None, detail="inconclusive classification")

        except Exception as exc:
            logger.warning("ColorAttributeVerifier failed, marking UNKNOWN: %s", exc)
            return VerificationResult(matched=None, detail=str(exc))