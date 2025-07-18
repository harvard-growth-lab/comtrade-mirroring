import os
from pathlib import Path
import sys
import argparse
import pandas as pd
from time import strftime, localtime
from datetime import datetime

from user_config import (
    get_paths_config,
    get_classifications,
    PROCESSING_STEPS,
    validate_config,
    print_config_summary,
    LOG_LEVEL,
    get_data_version,
)

from src.utils.CIF_calculations import compute_distance
from src.objects.base import AtlasCleaning
from src.table_objects.aggregate_trade import AggregateTrade
from src.table_objects.trade_analysis_cleaner import TradeAnalysisCleaner
from src.table_objects.trade_data_reconciler import TradeDataReconciler
from src.table_objects.country_country_product_year import CountryCountryProductYear
from src.utils.classification_handler import (
    sitc_and_skip_processing,
    handle_product_classification,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_ingestion_attrs(classification, start_year, end_year):
    """Create ingestion attributes for a specific classification and year range"""
    base_config = get_paths_config()

    return {
        "start_year": start_year,
        "end_year": end_year,
        "product_classification": classification,
        **base_config,
    }


def run_mirroring(ingestion_attrs):
    """
    Run the bilateral trade mirroring to generate reliable trade data.

    Outputs mirrored bilateral trade data:
    1. Data aggregation across classifications
    2. Reliability scores for country-level trade
    3. Country-level trade reconciliation
    4. Product-level trade reconciliation

    Parameters:
    - ingestion_attrs (dict): Configuration dictionary with required keys:
        - start_year (int): First year to process
        - end_year (int): Last year to process
        - product_classification (str): Trade classification system
        - downloaded_files_path (str): Path to raw data files
        - root_dir (str): Root directory path
    """
    logger.info(f"start time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
    start_year = ingestion_attrs["start_year"]
    end_year = ingestion_attrs["end_year"]
    product_classification = ingestion_attrs["product_classification"]
    downloaded_files_path = ingestion_attrs["downloaded_files_path"]

    base_obj = AtlasCleaning(**ingestion_attrs)
    aggregate_trade(base_obj, ingestion_attrs)
    if base_obj.missing_data:
        logger.error(
            "\nData not available for selected range. Skipping classification..."
        )
        return
    logger.info(f"Completed data aggregations")

    run_bilateral_mirroring_pipeline(ingestion_attrs)


def run_bilateral_mirroring_pipeline(ingestion_attrs):
    """
    Process a single year through the complete reconciliation pipeline
    """
    product_classification = ingestion_attrs["product_classification"]
    download_type = ingestion_attrs["download_type"]

    for year in range(ingestion_attrs["start_year"], ingestion_attrs["end_year"] + 1):
        logger.info(f"Mirroring trade data for {year}... for {product_classification}")
        if sitc_and_skip_processing(year, product_classification, download_type):
            continue
        product_classification = handle_product_classification(
            year, product_classification, download_type
        )[0]

        logger.debug(f"Beginning compute distance for year {year}")
        base_obj = AtlasCleaning(**ingestion_attrs)
        dist = pd.read_stata(base_obj.static_data_path / "dist_cepii.dta")
        df = compute_distance(base_obj, year, product_classification, dist)

        # cleaned country-country trade data with reporting quality metrics
        trade_discrepancy_analysis = TradeAnalysisCleaner(year, df, **ingestion_attrs)

        # country-country trade data with reconciled values and reliability scores
        country_trade_reconciler = TradeDataReconciler(
            year, trade_discrepancy_analysis.df, **ingestion_attrs
        )
        ccy = country_trade_reconciler.reconcile_country_country_estimates()

        if not sitc_and_skip_processing(year, product_classification, download_type):
            product_level_reconciler = CountryCountryProductYear(
                year, ccy, **ingestion_attrs
            )
            ccpy = (
                product_level_reconciler.reconcile_country_country_product_estimates()
            )
            product_level_reconciler.save_parquet(
                ccpy,
                f"final",
                f"{product_classification}_{year}",
                product_classification,
            )


def aggregate_trade(base_obj, ingestion_attrs):
    """
    Aggregate raw trade data for all years and classifications.
    """
    product_classification = ingestion_attrs["product_classification"]
    download_type = ingestion_attrs["download_type"]

    for year in range(ingestion_attrs["start_year"], ingestion_attrs["end_year"] + 1):
        if sitc_and_skip_processing(year, product_classification, download_type):
            continue
        classifications = handle_product_classification(
            year, product_classification, download_type
        )

        logger.debug(
            f"Aggregating data for {year} and these classifications {classifications}"
        )
        [
            AggregateTrade(year, product_class, **ingestion_attrs).run_aggregate_trade()
            for product_class in classifications
        ]


def clean_up_intermediate_files(ingestion_attrs):
    base = AtlasCleaning(**ingestion_attrs)
    base.cleanup_files_from_dir(base.intermediate_data_path.iterdir())
