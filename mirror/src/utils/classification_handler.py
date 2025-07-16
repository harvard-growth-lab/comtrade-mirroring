import logging
import pandas as pd
import os

# logging.basicConfig(level=logging.INFO)


def sitc_and_skip_processing(
    year: int, product_classification: str, download_type: str
) -> bool:
    if (
        product_classification == "SITC"
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
    if product_classification == "SITC" and download_type == "by_classification":
        return get_classifications(year)
    elif product_classification == "SITC" and download_type == "as_reported":
        return ["S2"]
    else:
        return [product_classification]


def get_classifications(year: int) -> list[str]:
    """
    Based on year, generate list of all available classifications for that year
    """
    classifications = []
    if year >= 1976 and year < 1995:
        classifications.append("S2")
    if year >= 1962 and year < 1976:
        classifications.append("S1")

    logger.info(
        f"generating aggregations for the following classifications: {classifications}"
    )
    return classifications
