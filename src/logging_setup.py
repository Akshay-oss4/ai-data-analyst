"""Application logging to a rotating file (logs/app.log). It records problems, never your data."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.config import PROJECT_ROOT

_LOG_DIR = PROJECT_ROOT / "logs"


def get_logger(name: str = "analyst") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    try:
        _LOG_DIR.mkdir(exist_ok=True)
        handler = RotatingFileHandler(_LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.NullHandler())
    return logger