"""
CLIP zero-shot binary verifier — fallback for attributes that don't map
to a COCO object class (e.g. "helmet", "mask", "high-vis vest").
Classifies the whole candidate image against a positive/negative prompt
pair.

Probability near 0.5 (neither confidently present nor confidently
absent) is treated as UNKNOWN, not NO_MATCH — CLIP's global pooled score
is a weak, diffuse signal for small attributes, so a coin-flip result
should not be read as confident evidence of absence.
"""

from __future__ import annotations

import torch
from PIL import Image

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.services.rag import _load_clip
from backend.app.services.verifiers.base import AttributeQuery, AttributeVerifier, VerificationResult

logger = get_logger(__name__)

_MATCH_MIN_PROB = 0.55     # positive_prob >= this -> confident MATCH
_NO_MATCH_MAX_PROB = 0.45  # positive_prob <= this -> confident NO_MATCH
                            # anything in between -> UNKNOWN (ambiguous)


class ClipZeroShotAttributeVerifier(AttributeVerifier):
    """Verifies "does this image show a person with <attribute>?" via CLIP."""

    def verify(self, image: Image.Image, attribute_query: AttributeQuery) -> VerificationResult:
        cfg = get_settings()
        device = cfg.EMBEDDING_DEVICE
        attribute = attribute_query.value

        try:
            model, processor = _load_clip()
            prompts = [
                f"a photo of a person wearing a {attribute}",
                f"a photo of a person not wearing a {attribute}",
            ]
            inputs = processor(text=prompts, images=[image], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits_per_image, dim=-1)[0].cpu().tolist()

            positive_prob = probs[0]

            if positive_prob >= _MATCH_MIN_PROB:
                return VerificationResult(
                    matched=True,
                    confidence=positive_prob,
                    matched_count=1,
                    detail=f"{attribute} present",
                )

            if positive_prob <= _NO_MATCH_MAX_PROB:
                return VerificationResult(matched=False, detail=f"{attribute} confidently absent")

            # Too close to call — genuinely ambiguous, not evidence of absence.
            return VerificationResult(matched=None, confidence=positive_prob, detail="ambiguous CLIP score")

        except Exception as exc:
            logger.warning("ClipZeroShotAttributeVerifier failed, marking UNKNOWN: %s", exc)
            return VerificationResult(matched=None, detail=str(exc))