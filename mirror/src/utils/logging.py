import logging
import os
from datetime import datetime
from pathlib import Path

try:
    from user_config import get_data_version, LOG_LEVEL
except ImportError:

    def get_data_version():
        return "unknown"

    LOG_LEVEL = "INFO"


def setup_logging():
    """
    Set up logging with console and file output.
    Call this once in main.py and it applies to the entire codebase.
    """
    level = getattr(logging, LOG_LEVEL.upper())

    # Create log filename with timestamp and version
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_version = get_data_version()
    log_file = f"logs/atlas_{data_version}_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    return logging.getLogger(__name__)


def get_logger(name):
    """
    Get a logger for any module.

    Usage in any file:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return logging.getLogger(name)
