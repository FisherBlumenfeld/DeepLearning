"""Logging configuration for Assignment 5 using structlog."""

import logging
import sys
from typing import Any
import numpy as np
import structlog


def custom_serializer_processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Formats special types like NumPy arrays and floats for clean logging output."""
    for key, value in event_dict.items():
        if isinstance(value, (np.floating, float)):
            event_dict[key] = round(float(value), 4)
        elif isinstance(value, np.ndarray):
            event_dict[key] = f"array(shape={value.shape}, dtype={value.dtype})"
    return event_dict


def setup_logging(debug: bool = False) -> None:
    """Configures structlog and standard logging handlers."""
    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            custom_serializer_processor,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
