"""
Lightweight regex parsers for metadata fields that the LLM often misses.

Each function is independent and returns None if nothing is found.
They are called by services/intent.py AFTER the LLM produces its result,
and only for fields that are still None.

Parsers
-------
parse_camera_id(text)   → str | None     e.g. "cam2", "camera 3" → "2", "3"
parse_time_range(text)  → (str|None, str|None)   HH:MM 24-h bounds
parse_confidence(text)  → float | None   e.g. "above 80%" → 0.8
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# ── Camera ID ─────────────────────────────────────────────────────────────────

# Matches: "camera 2", "cam2", "cam_2", "CAM-3", "camera no 4", "camera #5"
_CAM_RE = re.compile(
    r"\bcam(?:era)?\s*[_\-#no\.]*\s*(\d+)\b",
    re.IGNORECASE,
)


def parse_camera_id(text: str) -> Optional[str]:
    """
    Extract the first camera number from *text*.

    Returns the bare digit string (e.g. "2") or None.
    """
    m = _CAM_RE.search(text)
    if m:
        result = m.group(1)
        logger.debug("query_parsers: camera_id match → %s", result)
        return result
    return None


# ── Time range ────────────────────────────────────────────────────────────────

# 12-h: "3 PM", "3:30 PM", "3:30PM"
_TIME_12H = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    re.IGNORECASE,
)

# 24-h explicit: "15:00", "09:30"
_TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

# Keywords that indicate direction
_AFTER_RE  = re.compile(r"\bafter\b",  re.IGNORECASE)
_BEFORE_RE = re.compile(r"\bbefore\b|\buntil\b|\btill\b", re.IGNORECASE)
_BETWEEN_RE = re.compile(r"\bbetween\b", re.IGNORECASE)
_AND_RE     = re.compile(r"\band\b",     re.IGNORECASE)


def _parse_single_time(text: str) -> Optional[str]:
    """Return the first time found in *text* as HH:MM (24-h), or None."""
    m = _TIME_12H.search(text)
    if m:
        h = int(m.group(1)) % 12
        mins = int(m.group(2)) if m.group(2) else 0
        if m.group(3).lower() == "pm":
            h += 12
        return f"{h:02d}:{mins:02d}"

    m = _TIME_24H.search(text)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"

    return None


def parse_time_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (time_after, time_before) from *text*.

    Handles:
        "after 3 PM"            → ("15:00", None)
        "before 6 PM"           → (None,    "18:00")
        "between 3 PM and 6 PM" → ("15:00", "18:00")
        "from 09:00 to 17:00"   → ("09:00", "17:00")

    Returns (None, None) if no time expression is found.
    """
    lower = text.lower()

    # "between X and Y" / "from X to Y"
    between_m = re.search(
        r"\b(?:between|from)\b\s+(.+?)\s+\b(?:and|to)\b\s+(.+?)(?:\s+|$)",
        lower,
        re.IGNORECASE,
    )
    if between_m:
        t1 = _parse_single_time(between_m.group(1))
        t2 = _parse_single_time(between_m.group(2))
        if t1 and t2:
            logger.debug("query_parsers: time range → %s – %s", t1, t2)
            return t1, t2

    # "after X"
    after_m = _AFTER_RE.search(lower)
    before_m = _BEFORE_RE.search(lower)

    time_after: Optional[str] = None
    time_before: Optional[str] = None

    if after_m:
        snippet = lower[after_m.end():]
        time_after = _parse_single_time(snippet)
        if time_after:
            logger.debug("query_parsers: time_after → %s", time_after)

    if before_m:
        snippet = lower[before_m.end():]
        time_before = _parse_single_time(snippet)
        if time_before:
            logger.debug("query_parsers: time_before → %s", time_before)

    return time_after, time_before


# ── Confidence ────────────────────────────────────────────────────────────────

# "above 80%", "over 0.9", "confidence 85", "at least 70%"
_CONF_RE = re.compile(
    r"\b(?:above|over|at\s+least|min(?:imum)?|confidence\s+(?:of\s+)?)\s*(\d+(?:\.\d+)?)\s*(%?)",
    re.IGNORECASE,
)


def parse_confidence(text: str) -> Optional[float]:
    """
    Extract a confidence lower bound from *text*.

    Returns a float in [0.0, 1.0] or None.
    """
    m = _CONF_RE.search(text)
    if not m:
        return None

    val = float(m.group(1))
    is_percent = bool(m.group(2))

    # Heuristic: if > 1.0 and no % sign, treat as percentage anyway
    if is_percent or val > 1.0:
        val /= 100.0

    val = max(0.0, min(1.0, val))
    logger.debug("query_parsers: confidence → %.2f", val)
    return val