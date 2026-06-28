"""Logging configuration utilities.

Provides project-standard logging setup used by entry points (train.py, evaluate.py).
Individual modules should use logging.getLogger(__name__) and never configure
the root logger themselves.

Reference: docs/module_12_repository_structure.md §5
"""

from __future__ import annotations

# standard library
import logging
from pathlib import Path

# local
from p2p_energy_trading.constants import LOG_DATE_FORMAT, LOG_FORMAT


def setup_logging(
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
    log_file: str = "training.log",
) -> None:
    """Configure the root logger with console and optional file output.

    Call this once at program entry (e.g., in train.py or evaluate.py).
    Modules should only use logging.getLogger(__name__).

    Args:
        level: Logging level for both console and file handlers.
        log_dir: Directory for log file. If None, only console logging.
        log_file: Name of the log file within log_dir.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / log_file, mode="a", encoding="utf-8"
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the project's standard naming convention.

    This is a convenience wrapper around logging.getLogger.
    Modules should prefer logging.getLogger(__name__) directly.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def set_log_level(level: int) -> None:
    """Change the log level of the root logger and all handlers.

    Args:
        level: New logging level (e.g., logging.DEBUG, logging.INFO).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
