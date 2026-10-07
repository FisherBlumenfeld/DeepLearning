"""Evaluation entry point for hw04."""

import structlog

from .logging import configure_logging


def main() -> None:
    """CLI entry point for homework evaluation."""
    configure_logging()
    log = structlog.get_logger()
    log.info(
        "Evaluation not yet implemented for hw04. Implement evaluation logic in src/hw04/evaluate.py"
    )
