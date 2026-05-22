# FILE: tests/test_logger.py

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.utils.logger import (
    build_file_handler,
    build_formatter,
    build_stream_handler,
    get_logger,
    is_debug_mode,
    log_exception,
    log_pipeline_stage,
    resolve_log_level,
)

# ============================================================================
# LOG LEVEL TESTS
# ============================================================================


def test_resolve_log_level_default():
    """
    Default log level stability.
    """

    level = resolve_log_level()

    assert isinstance(
        level,
        int,
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_resolve_log_level_values(
    raw,
    expected,
):
    """
    Log-level normalization stability.
    """

    level = resolve_log_level(raw)

    assert level == expected


def test_invalid_log_level_fallback():
    """
    Invalid levels must fallback safely.
    """

    level = resolve_log_level(
        "INVALID"
    )

    assert level == logging.INFO


# ============================================================================
# FORMATTER TESTS
# ============================================================================


def test_build_formatter():
    """
    Formatter construction stability.
    """

    formatter = build_formatter()

    assert isinstance(
        formatter,
        logging.Formatter,
    )


# ============================================================================
# HANDLER TESTS
# ============================================================================


def test_build_stream_handler():
    """
    Stream handler stability.
    """

    handler = build_stream_handler(
        logging.INFO
    )

    assert isinstance(
        handler,
        logging.StreamHandler,
    )

    assert (
        handler.level
        == logging.INFO
    )


def test_build_file_handler(
    tmp_path,
):
    """
    File handler stability.
    """

    handler = build_file_handler(
        log_directory=str(
            tmp_path
        ),
        log_filename="test.log",
    )

    assert handler is not None


# ============================================================================
# LOGGER FACTORY TESTS
# ============================================================================


def test_get_logger_returns_logger():
    """
    Logger factory stability.
    """

    logger = get_logger(
        name="test_logger",
        enable_console=False,
        enable_file=False,
    )

    assert isinstance(
        logger,
        logging.Logger,
    )


def test_logger_no_duplicate_handlers():
    """
    Prevent duplicate handler regressions.
    """

    logger_1 = get_logger(
        name="duplicate_test",
        enable_console=False,
        enable_file=False,
    )

    handler_count_1 = len(
        logger_1.handlers
    )

    logger_2 = get_logger(
        name="duplicate_test",
        enable_console=False,
        enable_file=False,
    )

    handler_count_2 = len(
        logger_2.handlers
    )

    assert (
        handler_count_1
        == handler_count_2
    )


def test_logger_propagation_disabled():
    """
    Prevent root logger pollution.
    """

    logger = get_logger(
        name="propagation_test",
        enable_console=False,
        enable_file=False,
    )

    assert (
        logger.propagate
        is False
    )


# ============================================================================
# EXCEPTION LOGGING TESTS
# ============================================================================


def test_log_exception():
    """
    Exception logging must not crash.
    """

    logger = get_logger(
        name="exception_test",
        enable_console=False,
        enable_file=False,
    )

    try:
        raise ValueError(
            "Failure"
        )

    except Exception as error:
        log_exception(
            logger,
            error,
            context={
                "step": "testing"
            },
        )


# ============================================================================
# TELEMETRY TESTS
# ============================================================================


def test_log_pipeline_stage():
    """
    Pipeline-stage logging stability.
    """

    logger = get_logger(
        name="telemetry_test",
        enable_console=False,
        enable_file=False,
    )

    log_pipeline_stage(
        logger,
        stage_name="training",
        duration_seconds=1.25,
    )


# ============================================================================
# ENVIRONMENT TESTS
# ============================================================================


def test_is_debug_mode_returns_bool():
    """
    Debug-mode detection stability.
    """

    result = is_debug_mode()

    assert isinstance(
        result,
        bool,
    )


# ============================================================================
# REGRESSION SAFETY TESTS
# ============================================================================


def test_logger_name_stability():
    """
    Logger names must remain deterministic.
    """

    logger = get_logger(
        name="stable_logger",
        enable_console=False,
        enable_file=False,
    )

    assert (
        logger.name
        == "stable_logger"
    )


def test_logger_level_numeric():
    """
    Logger level must remain numeric.
    """

    logger = get_logger(
        name="level_test",
        enable_console=False,
        enable_file=False,
    )

    assert isinstance(
        logger.level,
        int,
    )


def test_logger_handlers_list():
    """
    Handlers must remain list-like.
    """

    logger = get_logger(
        name="handler_test",
        enable_console=False,
        enable_file=False,
    )

    assert isinstance(
        logger.handlers,
        list,
    )
