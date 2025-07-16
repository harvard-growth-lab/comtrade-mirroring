import logging
import os
from time import strftime

from user_config import LOG_LEVEL


def setup_logging():
    """Configure logging with both console and file output"""

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler (detailed format)
    data_version = get_data_version()
    log_file = (
        f"logs/atlas_processing_{data_version}_{strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)
