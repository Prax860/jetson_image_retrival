"""
In-process verification result cache.

Attribute verification (YOLO detection + CLIP classification) is the
single most expensive step in the retrieval pipeline. Images in the
alert store are immutable once ingested — ingest.py always mints a new
record_id for a new image — so the result of "does image <record_id>
satisfy attribute query <type/value/garment>?" never changes. It's safe
to cache indefinitely for the life of the process.

This cache is intentionally simple: a size-bounded LRU dict guarded by
a lock. It is NOT persisted across process restarts (a restart just
means a cold cache, not incorrect results) and needs no invalidation
logic given the immutability guarantee above.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional, Tuple

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.services.verifiers.base import AttributeQuery, VerificationResult

logger = get_logger(__name__)

_DEFAULT_MAX_SIZE = 5000

_lock = threading.Lock()
_cache: "OrderedDict[Tuple, VerificationResult]" = OrderedDict()


def _max_size() -> int:
    cfg = get_settings()
    return max(1, int(getattr(cfg, "VERIFICATION_CACHE_MAX_SIZE", _DEFAULT_MAX_SIZE)))


def make_key(record_id: str, attribute_query: AttributeQuery) -> Tuple:
    """Build a stable cache key for one (image, attribute) pair."""
    return (record_id, attribute_query.attribute_type, attribute_query.value, attribute_query.garment)


def get(key: Tuple) -> Optional[VerificationResult]:
    with _lock:
        result = _cache.get(key)
        if result is not None:
            _cache.move_to_end(key)  # LRU touch
        return result


def set(key: Tuple, result: VerificationResult) -> None:
    with _lock:
        _cache[key] = result
        _cache.move_to_end(key)
        max_size = _max_size()
        while len(_cache) > max_size:
            _cache.popitem(last=False)  # evict least-recently-used


def clear() -> None:
    """Exposed for tests / manual cache invalidation if ever needed."""
    with _lock:
        _cache.clear()


def stats() -> dict:
    """Lightweight introspection, handy for a debug/health endpoint."""
    with _lock:
        return {"size": len(_cache), "max_size": _max_size()}