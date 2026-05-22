# FILE: app/utils/logger.py

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_LOGGER_NAME = "ml_pipeline_debugger"

DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_LOG_DIRECTORY = "logs"

DEFAULT_LOG_FILENAME = "pipeline_debugger.log"

DEFAULT_MAX_LOG_SIZE_MB = 10

DEFAULT_BACKUP_COUNT = 5

LOG_FORMAT = (
    "[%(asctime)s] "
    "[%(levelname)s] "
    "[%(name)s] "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# LOGGER CACHE
# ============================================================================

_LOGGER_CACHE = {}

# ============================================================================
# LOG LEVEL RESOLUTION
# ============================================================================


def resolve_log_level(
    level: Optional[str] = None,
) -> int:
    """
    Stable log-level normalization.

    Prevents:
    - invalid environment configs
    - inconsistent logger states
    - silent logging failures
    """

    level = (
        level
        or os.getenv(
            "MLPD_LOG_LEVEL",
            DEFAULT_LOG_LEVEL,
        )
    )

    level = str(level).strip().upper()

    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return valid_levels.get(
        level,
        logging.INFO,
    )

# ============================================================================
# FORMATTERS
# ============================================================================


def build_formatter() -> logging.Formatter:
    """
    Stable formatter construction.
    """

    return logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

# ============================================================================
# HANDLERS
# ============================================================================


def build_stream_handler(
    log_level: int,
) -> logging.StreamHandler:
    """
    Streamlit/notebook-safe console handler.
    """

    handler = logging.StreamHandler(
        stream=sys.stdout
    )

    handler.setLevel(log_level)

    handler.setFormatter(
        build_formatter()
    )

    return handler


def build_file_handler(
    log_directory: str = DEFAULT_LOG_DIRECTORY,
    log_filename: str = DEFAULT_LOG_FILENAME,
    log_level: int = logging.INFO,
    max_size_mb: int = DEFAULT_MAX_LOG_SIZE_MB,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> RotatingFileHandler:
    """
    Stable rotating-file handler.

    Prevents:
    - runaway log growth
    - filesystem instability
    - production logging explosions
    """

    log_path = Path(log_directory)

    log_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_log_path = (
        log_path / log_filename
    )

    handler = RotatingFileHandler(
        filename=full_log_path,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )

    handler.setLevel(log_level)

    handler.setFormatter(
        build_formatter()
    )

    return handler

# ============================================================================
# LOGGER FACTORY
# ============================================================================


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
    *,
    enable_console: bool = True,
    enable_file: bool = True,
    log_directory: str = DEFAULT_LOG_DIRECTORY,
    log_filename: str = DEFAULT_LOG_FILENAME,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Centralized logger factory.

    Goals:
    - singleton-safe loggers
    - no duplicate handlers
    - streamlit-safe behavior
    - notebook-safe behavior
    - multiprocessing-safe formatting
    - deterministic configuration

    IMPORTANT:
    Prevents:
    - duplicate console spam
    - root logger pollution
    - recursive handler attachment
    """

    cache_key = (
        f"{name}|"
        f"{enable_console}|"
        f"{enable_file}|"
        f"{log_directory}|"
        f"{log_filename}|"
        f"{level}"
    )

    if cache_key in _LOGGER_CACHE:
        return _LOGGER_CACHE[
            cache_key
        ]

    logger = logging.getLogger(
        name
    )

    log_level = resolve_log_level(
        level
    )

    logger.setLevel(log_level)

    # ==========================================================
    # Prevent propagation to root logger
    # ==========================================================

    logger.propagate = False

    # ==========================================================
    # Prevent duplicate handlers
    # ==========================================================

    if logger.handlers:
        logger.handlers.clear()

    # ==========================================================
    # Console Handler
    # ==========================================================

    if enable_console:
        try:
            console_handler = (
                build_stream_handler(
                    log_level
                )
            )

            logger.addHandler(
                console_handler
            )

        except Exception:
            pass

    # ==========================================================
    # File Handler
    # ==========================================================

    if enable_file:
        try:
            file_handler = (
                build_file_handler(
                    log_directory=log_directory,
                    log_filename=log_filename,
                    log_level=log_level,
                )
            )

            logger.addHandler(
                file_handler
            )

        except Exception as error:
            logger.warning(
                "File logger initialization failed: %s",
                str(error),
            )

    # ==========================================================
    # Cache
    # ==========================================================

    _LOGGER_CACHE[
        cache_key
    ] = logger

    return logger

# ============================================================================
# EXCEPTION LOGGING
# ============================================================================


def log_exception(
    logger: logging.Logger,
    error: Exception,
    *,
    context: Optional[dict] = None,
    level: int = logging.ERROR,
) -> None:
    """
    Structured exception logging helper.

    Prevents:
    - inconsistent exception formatting
    - missing operational context
    - unreadable stack traces
    """

    context = context or {}

    logger.log(
        level,
        (
            "Exception occurred | "
            f"type={error.__class__.__name__} | "
            f"message={str(error)} | "
            f"context={context}"
        ),
        exc_info=True,
    )

# ============================================================================
# TELEMETRY LOGGING
# ============================================================================


def log_pipeline_stage(
    logger: logging.Logger,
    stage_name: str,
    *,
    duration_seconds: Optional[float] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Stable telemetry-stage logging.
    """

    extra = extra or {}

    message = (
        f"Pipeline stage completed | "
        f"stage={stage_name}"
    )

    if duration_seconds is not None:
        message += (
            f" | duration="
            f"{duration_seconds:.4f}s"
        )

    if extra:
        message += (
            f" | extra={extra}"
        )

    logger.info(message)

# ============================================================================
# ENVIRONMENT HELPERS
# ============================================================================


def is_debug_mode() -> bool:
    """
    Environment-safe debug detection.
    """

    debug_value = os.getenv(
        "MLPD_DEBUG",
        "false",
    )

    return (
        str(debug_value)
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )

# ============================================================================
# DEFAULT LOGGER
# ============================================================================

default_logger = get_logger()
