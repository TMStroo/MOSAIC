"""Structured logging. One configuration function, imported by the CLI and API."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("experiment_id", "stage", "dataset_version", "duration_s", "rows"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        extra = ""
        for key in ("experiment_id", "stage", "rows", "duration_s"):
            if hasattr(record, key):
                extra += f" {key}={getattr(record, key)}"
        return super().format(record) + extra


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Idempotently configure root logging for MOSAIC."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root.handlers = [handler]
    root.setLevel(level.upper())
    for noisy in ("matplotlib", "py4j", "urllib3", "asyncio", "httpx", "lightgbm", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
