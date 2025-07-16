import os
from pathlib import Path
import sys
import argparse
import pandas as pd
from time import strftime, localtime
from datetime import datetime

script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

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
from src.table_objects.complexity import Complexity
from src.table_objects.unilateral_services import UnilateralServices
from src.utils.classification_handler import (
    sitc_and_skip_processing,
    handle_product_classification,
)
from src.utils.logging import setup_logging


# # Set up logging based on config
# logging.basicConfig(level=getattr(logging, LOG_LEVEL))
# logger = logging.getLogger(__name__)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)


def create_ingestion_attrs(classification, start_year, end_year):
    """Create ingestion attributes for a specific classification and year range"""
    base_config = get_paths_config()

    return {
        "start_year": start_year,
        "end_year": end_year,
        "product_classification": classification,
        **base_config,
    }


def run_atlas_cleaning(ingestion_attrs):
    """
    Run the bilateral trade mirroring to generate reliable trade data.

    Builds and saves input tables for Atlas ingestion using a three-stage process:
    1. Data aggregation across classifications
    2. Country-level trade reconciliation
    3. Product-level trade reconciliation

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
    aggregate_trade(ingestion_attrs)
    logger.info(f"Completed data aggregations")

    run_bilateral_mirroring_pipeline(ingestion_attrs)
    # deletes all intermediate processing files
    clean_up_intermediate_files(ingestion_attrs)


def run_bilateral_mirroring_pipeline(ingestion_attrs):
    """
    Process a single year through the complete reconciliation pipeline
    """
    product_classification = ingestion_attrs["product_classification"]
    download_type = ingestion_attrs["download_type"]

    for year in range(ingestion_attrs["start_year"], ingestion_attrs["end_year"] + 1):
        logger.info(f"Beginning trade mirror {year}... for {product_classification}")
        if sitc_and_skip_processing(year, product_classification, download_type):
            continue
        product_classification = handle_product_classification(
            year, product_classification, download_type
        )[0]

        logger.info(f"Beginning compute distance for year {year}")
        base_obj = AtlasCleaning(**ingestion_attrs)
        dist = pd.read_stata(base_obj.static_data_path / "dist_cepii.dta")
        # dist = pd.read_stata(os.path.join("data", "static", "dist_cepii.dta"))
        df = compute_distance(base_obj, year, product_classification, dist)

        # cleaned country-country trade data with reporting quality metrics
        trade_discrepancy_analysis = TradeAnalysisCleaner(year, df, **ingestion_attrs)

        # country-country trade data with reconciled values and accuracy weights
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


def aggregate_trade(ingestion_attrs):
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

        logger.info(
            f"Aggregating data for {year} and these classifications {classifications}"
        )
        [
            AggregateTrade(year, product_class, **ingestion_attrs).run_aggregate_trade()
            for product_class in classifications
        ]


def clean_up_intermediate_files(ingestion_attrs):
    base = AtlasCleaning(**ingestion_attrs)
    base.cleanup_files_from_dir(base.intermediate_data_path.iterdir())


def main():
    """
    Main execution function that runs the Atlas cleaning pipeline
    based on configuration settings
    """
    parser = argparse.ArgumentParser(description="Run the Atlas clean")
    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="Validates config, prints configuration summary, and exits",
    )

    args = parser.parse_args()

    global logger
    logger = setup_logging()

    if args.config_summary:
        errors = validate_config()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  • {error}")
                sys.exit(0)
        print_config_summary()
        sys.exit(0)

    errors = validate_config()
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  • {error}")
        sys.exit(1)

    classifications = get_classifications()

    if not classifications:
        logger.error("No classifications selected! Check your config settings.")
        sys.exit(1)

    # Show what will be processed
    logger.info("=" * 60)
    logger.info("ATLAS PROCESSING STARTING")
    logger.info("=" * 60)
    logger.info(f"Data version: {get_data_version()}")
    logger.info(f"Processing {len(classifications)} classification(s)")

    for classification, start_year, end_year, description in classifications:
        years_count = end_year - start_year + 1
        logger.info(f"  • {description}: {start_year}-{end_year} ({years_count} years)")

    # Show processing steps
    enabled_steps = [step for step, enabled in PROCESSING_STEPS.items() if enabled]
    logger.info(f"Processing steps: {', '.join(enabled_steps)}")

    logger.info("=" * 60)

    # Process each classification
    total_start_time = datetime.now()

    for i, (classification, start_year, end_year, description) in enumerate(
        classifications, 1
    ):
        classification_start_time = datetime.now()
        logger.info(f"[{i}/{len(classifications)}] Starting {description}")

        # Create configuration for this classification
        ingestion_attrs = create_ingestion_attrs(classification, start_year, end_year)

        try:
            if PROCESSING_STEPS.get("run_cleaning", True):
                logger.info("Running cleaning pipeline...")
                run_atlas_cleaning(ingestion_attrs)

            classification_duration = datetime.now() - classification_start_time
            logger.info(f"Completed {description} in {classification_duration}")

        except Exception as e:
            logger.error(f"Error processing {classification}: {str(e)}", exc_info=True)
            raise

    total_duration = datetime.now() - total_start_time
    logger.info("=" * 60)
    logger.info(f"ATLAS PROCESSING COMPLETED in {total_duration}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
