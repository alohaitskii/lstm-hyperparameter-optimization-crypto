"""Centralized logging configuration using the rich library for colored output."""
from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


_CONFIGURED = False


def setup_logger(name: str = "binance_lstm", level: str = "INFO") -> logging.Logger:
    """Initialize and return a logger. Safe to call multiple times."""
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        formatter = logging.Formatter("%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
        handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True

    return logger


def get_logger(name: str = "binance_lstm") -> logging.Logger:
    """Convenience accessor that ensures the logger is configured."""
    return setup_logger(name)


def add_file_handler(logger: logging.Logger, path: str | Path) -> None:
    """Attach a rotating-style file handler for persistent debug logs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(fh)
