"""Logging utilities — call setup_logging() once at pipeline startup."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", name: str = "hakim_ai") -> logging.Logger:
    """Configure and return the root pipeline logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger for a pipeline module."""
    return logging.getLogger(f"hakim_ai.{module_name}")