"""
Attribute verification framework.

An AttributeQuery describes WHAT to check for in an image (parsed once
from the user's raw query). An AttributeVerifier checks whether a given
image satisfies that query. Adding a new attribute (bottle, phone,
backpack, helmet, pose, ...) means writing one new verifier and adding
one line to the registry — nothing in retrieval.py changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image


@dataclass
class AttributeQuery:
    """What the user asked to verify, extracted from the free-text query."""
    raw_query: str
    attribute_type: str          # e.g. "color", "object", "clip_zero_shot"
    value: str                   # canonical attribute value, e.g. "blue", "bottle", "helmet"
    garment: Optional[str] = None  # only meaningful for attribute_type == "color"


@dataclass
class VerificationResult:
    """
    Outcome of checking one image against one AttributeQuery.

    matched:
        True  -> MATCH: attribute confidently confirmed present.
        False -> NO_MATCH: attribute confidently confirmed absent. Only
                 return this when the verifier had a fair, high-confidence
                 look at the image and the attribute genuinely isn't there
                 — not merely "I didn't find it."
        None  -> UNKNOWN: verification was inconclusive (low resolution,
                 no detection where a miss is plausible, occlusion,
                 ambiguous classifier confidence, detector error, etc.).
                 This is NOT evidence of absence. Callers must keep the
                 image and must NOT penalize it relative to its original
                 CLIP score — see reranker.py.

    confidence:      best/representative confidence, meaningful only when
                      matched is True. Kept for backward compatibility —
                      mirrors best_confidence when only one is set.
    matched_count:    number of individually-matched detections (e.g.
                      distinct people confirmed to have the attribute).
    matched_people:   bounding boxes (x1, y1, x2, y2) of every detection
                      that matched, so downstream consumers (e.g.
                      highlighting) can box ALL of them, not just the
                      single best one. Empty list if not applicable
                      (e.g. non-person attributes, or no matches).
    best_confidence:  the highest per-detection confidence among
                      matched_people.
    detail:           human-readable explanation, useful for logging/debugging.
    """
    matched: Optional[bool]
    confidence: float = 0.0
    matched_count: int = 0
    matched_people: Optional[List[Tuple[int, int, int, int]]] = None
    best_confidence: float = 0.0
    detail: str = ""

    def __post_init__(self):
        if self.matched_people is None:
            self.matched_people = []
        # Keep confidence/best_confidence in sync for callers that only
        # know about one of the two fields (backward compatibility with
        # verifiers/consumers written before best_confidence existed).
        if self.best_confidence and not self.confidence:
            self.confidence = self.best_confidence
        elif self.confidence and not self.best_confidence:
            self.best_confidence = self.confidence


class AttributeVerifier(ABC):
    """Base class for all attribute verifiers."""

    @abstractmethod
    def verify(self, image: Image.Image, attribute_query: AttributeQuery) -> VerificationResult:
        ...