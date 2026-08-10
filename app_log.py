"""Local diagnostic log — the only place raw AI/BingX failure detail goes.

The dashboard and terminal only ever show a normalized error code + safe
message (see `error_codes.py`); the actual gateway response body, the full
request URL, and any other provider-side diagnostic text are written here
instead, for a human to inspect locally if something needs debugging.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FILE = Path(__file__).resolve().parent / "neurotrade.log"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

logger = logging.getLogger("neurotrade")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = RotatingFileHandler(_LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)


def log_ai_failure(model: str, mode: str, stage: str, detail: str, raw_text: str = "") -> None:
    """Record an AI parsing/validation failure. `stage` is e.g. "first_attempt"
    or "repair_attempt". `raw_text` is truncated — kept only for debugging,
    never shown in the UI.
    """
    snippet = (raw_text or "")[:2000]
    logger.info("AI_FAILURE model=%s mode=%s stage=%s detail=%s raw=%r", model, mode, stage, detail, snippet)


def log_ai_repair(model: str, mode: str) -> None:
    logger.info("AI_REPAIR_ATTEMPT model=%s mode=%s", model, mode)


def log_error(code: str, model: str, mode: str, detail: str) -> None:
    """Record any normalized-error-code failure with its real (unmasked) detail."""
    logger.warning("ERROR code=%s model=%s mode=%s detail=%s", code, model, mode, detail)
