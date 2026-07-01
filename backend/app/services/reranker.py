"""
Generic verification + re-ranking stage, applied to CLIP's candidate list
only (never touches the full index).

Scoring per verification outcome:
    MATCH (True)     -> ranked before UNKNOWN, ordered by:
                         1) matched_count (desc)
                         2) verification confidence (desc)
                         3) CLIP score (desc, tie-breaker)
                         final_score is tier-separated (see _MATCH_TIER_OFFSET
                         below) so any downstream consumer that sorts or
                         displays by final_score alone still guarantees
                         MATCH ranks above UNKNOWN — not just list order.
    NO_MATCH (False) -> candidate discarded entirely. Only returned by a
                         verifier when it confidently confirmed the
                         attribute is absent — see verifiers/base.py.
    UNKNOWN (None)   -> ranked after all MATCH items and preserves original
                         CLIP candidate order exactly. UNKNOWN is NOT
                         evidence of absence (low resolution, missed
                         detection, occlusion, ambiguous classifier
                         confidence, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.services.verifiers.base import AttributeQuery, VerificationResult
from backend.app.services.verifiers.registry import get_verifier, parse_attribute_query

logger = get_logger(__name__)

T = TypeVar("T")

_DEFAULT_CLIP_WEIGHT = 0.6
_DEFAULT_VERIFICATION_WEIGHT = 0.4

# Tier separation for final_score. CLIP scores and verification confidence
# are expected in [0, 1]. Any MATCH candidate's final_score is guaranteed
# to land at or above _MATCH_TIER_OFFSET, while any UNKNOWN candidate's
# final_score is just its raw clip_score (<= ~1). This means ANY sort or
# display that keys off final_score alone — not just the list order this
# function returns — can never rank an UNKNOWN above a MATCH again.
#
# Within the MATCH tier, the weights below are ordered so the additive
# terms can never cross tiers into each other, preserving the same
# priority as the explicit sort key used on `matches`:
#   matched_count (desc) > confidence (desc) > clip_score (desc)
_MATCH_TIER_OFFSET = 1_000_000.0
_MATCH_COUNT_WEIGHT = 1_000.0
_MATCH_CONFIDENCE_WEIGHT = 10.0


@dataclass
class RerankedCandidate(Generic[T]):
    item: T
    clip_score: float
    verification: Optional[VerificationResult]
    final_score: float


def _weights() -> tuple[float, float]:
    cfg = get_settings()
    clip_w = getattr(cfg, "CLIP_WEIGHT", _DEFAULT_CLIP_WEIGHT)
    verify_w = getattr(cfg, "VERIFICATION_WEIGHT", _DEFAULT_VERIFICATION_WEIGHT)
    return clip_w, verify_w


def _effective_match_count(result: Optional[VerificationResult]) -> int:
    """
    Backward-compatible match-count accessor.

    Older verifiers may not set matched_count explicitly. In that case,
    treat any MATCH as count=1 so MATCH items are never demoted behind
    UNKNOWN due to missing metadata.
    """
    if result is None:
        return 0
    if result.matched is not True:
        return 0
    return result.matched_count if result.matched_count > 0 else 1


def maybe_rerank_by_attribute(
    query: str,
    candidates: List[T],
    get_image: Callable[[T], Optional["object"]],
    get_clip_score: Callable[[T], float],
) -> List[RerankedCandidate[T]]:
    """
    Run attribute verification + reranking over *candidates* IF the query
    names a verifiable attribute. If it doesn't, this is a no-op that
    preserves CLIP's original order exactly (backward compatible).

    Parameters
    ----------
    query          : raw user query text (verification target is parsed from this)
    candidates     : list of opaque candidate objects
    get_image      : callable(candidate) -> PIL.Image.Image | None
    get_clip_score : callable(candidate) -> float

    Returns
    -------
    List[RerankedCandidate], sorted so all MATCH items precede all UNKNOWN
    items (guaranteed both by list order AND by final_score tier
    separation — see module docstring). If no attribute was found in the
    query, verification is None for every item and final_score ==
    clip_score, preserving the original CLIP order.
    """
    attribute_query: Optional[AttributeQuery] = parse_attribute_query(query)

    if attribute_query is None:
        return [
            RerankedCandidate(item=c, clip_score=get_clip_score(c), verification=None, final_score=get_clip_score(c))
            for c in candidates
        ]

    clip_w, verify_w = _weights()
    verifier = get_verifier(attribute_query)
    logger.info(
        "Attribute verification active | type=%s value=%s garment=%s | candidates=%d",
        attribute_query.attribute_type, attribute_query.value, attribute_query.garment, len(candidates),
    )

    match_item_count = 0
    unknown_count = 0
    discarded_count = 0

    matches: List[RerankedCandidate[T]] = []
    unknowns: List[tuple[int, RerankedCandidate[T]]] = []

    for idx, candidate in enumerate(candidates):
        clip_score = get_clip_score(candidate)
        image = get_image(candidate)

        if image is None:
            result = VerificationResult(matched=None, detail="no image to inspect")
        else:
            result = verifier.verify(image, attribute_query)

        if result.matched is False:
            discarded_count += 1
            logger.debug("Discarding candidate — confidently absent: %s", result.detail)
            continue

        if result.matched is True:
            match_item_count += 1
            match_count = _effective_match_count(result)
            # Tier-safe score: always >= _MATCH_TIER_OFFSET, and within
            # the tier, strictly ordered by (matched_count, confidence,
            # clip_score) — matching the explicit sort key below term for
            # term, so a naive external sort-by-final_score can't disagree.
            final_score = (
                _MATCH_TIER_OFFSET
                + match_count * _MATCH_COUNT_WEIGHT
                + result.confidence * _MATCH_CONFIDENCE_WEIGHT
                + clip_score
            )
            matches.append(RerankedCandidate(
                item=candidate, clip_score=clip_score, verification=result, final_score=final_score,
            ))
        else:
            # UNKNOWN — keep CLIP's original ranking untouched, and keep
            # final_score strictly below _MATCH_TIER_OFFSET (clip_score is
            # expected in [0, 1]) so it can never outrank a MATCH no
            # matter how it's later sorted or displayed.
            unknown_count += 1
            final_score = clip_score
            unknowns.append((idx, RerankedCandidate(
                item=candidate, clip_score=clip_score, verification=result, final_score=final_score,
            )))

    # MATCH priority: matched_count desc, then verifier confidence desc,
    # then CLIP score desc as a final tie-breaker.
    matches.sort(
        key=lambda r: (
            _effective_match_count(r.verification),
            r.verification.confidence if r.verification is not None else 0.0,
            r.clip_score,
        ),
        reverse=True,
    )

    # UNKNOWN priority: preserve original CLIP order exactly.
    unknowns.sort(key=lambda t: t[0])
    reranked: List[RerankedCandidate[T]] = matches + [item for _idx, item in unknowns]

    logger.info(
        "Verification complete | matched=%d unknown=%d discarded=%d kept=%d",
        match_item_count, unknown_count, discarded_count, len(reranked),
    )
    return reranked