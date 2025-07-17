import pandas as pd
import os
from src.utils.logging import get_logger

logger = get_logger(__name__)


def sitc_and_skip_processing(
    year: int, product_classification: str, download_type: str
) -> bool:
    if (
        product_classification in ["SITC", "S1", "S2", "S3"]
        and year > 1994
        and download_type == "by_classification"
    ):
        # use cleaned CCPY H0 data for SITC
        return True
    return False


def handle_product_classification(
    year: int, product_classification: str, download_type: str
) -> list[str]:
    """ """
    if (
        product_classification in ["SITC", "S1", "S2", "S3"]
        and download_type == "by_classification"
    ):
        return handle_by_classification_data(year)
    elif (
        product_classification in ["SITC", "S1", "S2", "S3"]
        and download_type == "as_reported"
    ):
        return [product_classification]
    else:
        return [product_classification]


def handle_by_classification_data(year: int) -> list[str]:
    """
    Special handling of SITC data when downloading data already converted by Comtrade
    Uses "by_classification" download type
    """
    classifications = []
    if year >= 1976 and year < 1995:
        classifications.append("S2")
    if year >= 1962 and year < 1976:
        classifications.append("S1")

    logger.debug(
        f"generating aggregations for the following classifications: {classifications}"
    )
    return classifications
