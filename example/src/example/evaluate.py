"""Evaluation entry point for example."""

import structlog

from .logging import configure_logging


def main() -> None:
    """CLI entry point for homework evaluation."""
    configure_logging()
    log = structlog.get_logger()
    log.info(
        "Evaluation not yet implemented for example. Implement evaluation logic in src/example/evaluate.py"
    )
