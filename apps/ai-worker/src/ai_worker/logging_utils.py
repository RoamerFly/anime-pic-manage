"""Structured JSON Lines logging to stderr.

Stdout is reserved for the IPC stream, so logs must never be emitted there.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonLineFormatter(logging.Formatter):
    """Format each record as one machine-readable JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "task_id", "message_type", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                event[key] = value
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Configure the worker logger once and return it."""

    logger = logging.getLogger("anime_pic_ai_worker")
    logger.setLevel(level if isinstance(level, int) else level.upper())
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
    return logger
