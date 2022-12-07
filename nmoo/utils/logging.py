# pylint: disable=global-statement

"""Logging related utilities"""

import sys
from typing import Optional

from loguru import logger as logging


def configure_logging(
    logging_level: str = "INFO", prefix: Optional[str] = None
):
    """Reconfigures the logging facility"""
    if prefix is not None and prefix[-1] != " ":
        prefix += " "
    else:
        prefix = ""
    logging.remove()
    logging.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            + "[<level>{level: <8}</level>] "
            + prefix
            + "<level>{message}</level>"
        ),
        level=logging_level,
        enqueue=True,
        colorize=True,
    )
