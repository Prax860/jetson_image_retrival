"""
Helpers for working with camera identifiers.

The repository has legacy camera IDs in multiple shapes. These helpers infer
the storage shape from existing data and normalize new values to match it
without depending on the LLM output format.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable, Optional


_TRAILING_DIGITS_RE = re.compile(r"^(?P<prefix>.*?)(?P<digits>\d+)$")
_CAMERA_QUERY_RE = re.compile(r"(?:camera|cam)[\s_\-]*0*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CameraIdFormat:
    prefix: str = ""
    width: int = 0

    @property
    def is_numeric(self) -> bool:
        return not self.prefix and self.width == 0

    def describe(self) -> str:
        if self.is_numeric:
            return "numeric"
        return f"prefixed(prefix={self.prefix!r}, width={self.width})"


def infer_camera_id_format(samples: Iterable[object]) -> CameraIdFormat:
    """Infer the most likely storage format from observed camera IDs."""
    best = CameraIdFormat()
    best_score = -1

    for sample in samples:
        parsed = _split_camera_id(sample)
        if parsed is None:
            continue

        prefix, digits = parsed
        score = len(digits)
        if prefix:
            score += 100

        if score > best_score:
            best = CameraIdFormat(prefix=prefix, width=len(digits) if prefix else 0)
            best_score = score

    return best


def infer_existing_camera_id_format() -> CameraIdFormat:
    """Infer the storage format from the on-disk workspace state."""
    samples: list[str] = []

    meta_dir = Path("data") / "metadata"
    if meta_dir.exists():
        for path in meta_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            camera_id = payload.get("camera_id")
            if camera_id is not None and str(camera_id).strip():
                samples.append(str(camera_id))

    image_dir = Path("data") / "images"
    if image_dir.exists():
        for path in image_dir.iterdir():
            if path.is_dir():
                samples.append(path.name)

    return infer_camera_id_format(samples)


def normalize_camera_id(value: object, camera_format: CameraIdFormat | None = None) -> Optional[str]:
    """Normalize *value* to the requested storage format."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    parsed = _split_camera_id(text)
    if parsed is None:
        return text

    _, digits = parsed

    if camera_format is None or camera_format.is_numeric:
        return str(int(digits))

    return f"{camera_format.prefix}{digits.zfill(camera_format.width)}"


def parse_camera_id_from_query(query: str) -> Optional[str]:
    """Extract a camera ID candidate from a user query."""
    if not query:
        return None

    match = _CAMERA_QUERY_RE.search(query)
    if match:
        return match.group(1)

    stripped = query.strip()
    if stripped.isdigit():
        return stripped

    return None


def _split_camera_id(value: object) -> Optional[tuple[str, str]]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    match = _TRAILING_DIGITS_RE.match(text)
    if not match:
        return None

    return match.group("prefix"), match.group("digits")