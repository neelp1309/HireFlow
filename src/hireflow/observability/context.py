"""Per-request context used by logs and API responses."""
from __future__ import annotations

from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("hireflow_request_id", default="-")


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()


__all__ = ["get_request_id", "set_request_id"]
