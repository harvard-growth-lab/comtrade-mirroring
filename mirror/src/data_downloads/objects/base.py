from user_config import LOG_LEVEL
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/data_downloads_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
            mode="a",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)
