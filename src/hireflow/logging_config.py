"""Logging configuration for local development and deployment."""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RequestContextFilter(logging.Filter):
    """Attach request_id to every record without logging candidate PII."""

    def filter(self, record: logging.LogRecord) -> bool:
        from hireflow.observability.context import get_request_id

        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Minimal structured JSON formatter without an extra dependency."""

    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "request_id",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in self._reserved and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure human-readable console logs and JSON file logs."""
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "console",
            "stream": "ext://sys.stdout",
            "filters": ["request_context"],
        }
    }

    root_handlers = ["console"]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers["json_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json",
            "filename": str(log_dir / "hireflow.jsonl"),
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
            "filters": ["request_context"],
        }
        root_handlers.append("json_file")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_context": {"()": RequestContextFilter}},
            "formatters": {
                "console": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(message)s"
                },
                "json": {"()": JsonFormatter},
            },
            "handlers": handlers,
            "root": {"level": log_level, "handlers": root_handlers},
        }
    )
