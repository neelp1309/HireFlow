"""Process-local, bounded TTL cache for recruiter searches.

The cache intentionally lives only in memory. Search bundles may contain candidate
identity for recruiter display, so HireFlow does not persist cached responses to disk.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from hireflow.models import CandidateProfile, JobRequirements

if TYPE_CHECKING:
    from hireflow.product.models import ProductSearchConfig

T = TypeVar("T")


class MemoryTTLCache(Generic[T]):
    """Small thread-safe LRU/TTL cache with no external service dependency."""

    def __init__(self, max_entries: int = 128, ttl_seconds: int = 900) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def build_search_cache_key(
    job: JobRequirements,
    candidates: list[CandidateProfile],
    config: "ProductSearchConfig",
    scoring_weights: dict[str, float],
) -> str:
    """Create a deterministic key from job-relevant content, not contact details."""
    payload = {
        "job_id": job.job_id,
        "job_matching_text": job.matching_text(),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "matching_text": candidate.matching_text(),
                "parser_version": candidate.parser_version,
                # Identity never affects ranking, but an opaque digest prevents a
                # cached recruiter-facing bundle from being reused after identity
                # metadata changes for the same candidate ID.
                "identity_digest": hashlib.sha256(
                    json.dumps(
                        {
                            "full_name": candidate.full_name,
                            "location": candidate.location,
                            "contact": candidate.contact.model_dump(mode="json") if candidate.contact else None,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            for candidate in sorted(candidates, key=lambda item: item.candidate_id)
        ],
        "config": config.model_dump(mode="json"),
        "weights": scoring_weights,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


SearchBundleCache = MemoryTTLCache[Any]

__all__ = ["MemoryTTLCache", "SearchBundleCache", "build_search_cache_key"]
