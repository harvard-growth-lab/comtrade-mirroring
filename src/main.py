import os
from pathlib import Path
import sys
import argparse
import pandas as pd
from time import strftime, localtime
from datetime import datetime

script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from config.user_config import (
    get_paths_config,
    get_classifications,
    PROCESSING_STEPS,
    validate_config,
    print_config_summary,
    LOG_LEVEL,
    get_data_version,
)

from clean.utils.CIF_calculations import compute_distance
from clean.objects.base import AtlasCleaning
from clean.table_objects.aggregate_trade import AggregateTrade
from clean.table_objects.trade_analysis_cleaner import TradeAnalysisCleaner
from clean.table_objects.trade_data_reconciler import TradeDataReconciler
from clean.table_objects.country_country_product_year import CountryCountryProductYear
from clean.table_objects.complexity import Complexity
from clean.table_objects.unilateral_services import UnilateralServices
from clean.utils.classification_handler import (
    sitc_and_skip_processing,
    handle_product_classification,
)

import logging


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
    log_file = f"logs/atlas_processing_{data_version}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


# Set up logging based on config
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)


def create_ingestion_attrs(config, classification, start_year, end_year):
    """Create ingestion attributes for a specific classification and year range"""
    base_config = config.get_paths_config()

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
    logging.INFO(f"start time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
    start_year = ingestion_attrs["start_year"]
    end_year = ingestion_attrs["end_year"]
    product_classification = ingestion_attrs["product_classification"]
    downloaded_files_path = ingestion_attrs["downloaded_files_path"]

    base_obj = AtlasCleaning(**ingestion_attrs)
    aggregate_trade(ingestion_attrs)
    logging.INFO(f"Completed data aggregations")

    run_bilateral_mirroring_pipeline(ingestion_attrs)
    # deletes all intermediate processing files
    clean_up_intermediate_files(ingestion_attrs)


def run_complexity(ingestion_attrs):
    generate_complexity_metrics(ingestion_attrs)
    # deletes all intermediate complexity processing files
    clean_up_intermediate_files(ingestion_attrs)


def run_bilateral_mirroring_pipeline(ingestion_attrs):
    """
    Process a single year through the complete reconciliation pipeline
    """
    product_classification = ingestion_attrs["product_classification"]
    download_type = ingestion_attrs["download_type"]

    for year in range(ingestion_attrs["start_year"], ingestion_attrs["end_year"] + 1):
        logging.info(f"Beginning trade mirror {year}... for {product_classification}")
        if sitc_and_skip_processing(year, product_classification, download_type):
            continue
        product_classification = handle_product_classification(
            year, product_classification, download_type
        )[0]

        logging.info(f"Beginning compute distance for year {year}")
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

        logging.info(
            f"Aggregating data for {year} and these classifications {classifications}"
        )
        [
            AggregateTrade(year, product_class, **ingestion_attrs)
            for product_class in classifications
        ]


def generate_complexity_metrics(ingestion_attrs):
    """
    # handle complexity
    """
    product_classification = ingestion_attrs["product_classification"]
    download_type = ingestion_attrs["download_type"]
    for year in range(ingestion_attrs["start_year"], ingestion_attrs["end_year"] + 1):
        complexity = Complexity(year, **ingestion_attrs)
        complexity.save_parquet(
            complexity.df,
            "intermediate",
            f"{download_type}_{product_classification}_{year}_complexity",
        )
        del complexity.df
        logging.info(
            f"end time for {year}: {strftime('%Y-%m-%d %H:%M:%S', localtime())}"
        )

    complexity_all_years = complexity.intermediate_data_path.glob(
        f"{download_type}_{product_classification}_*_complexity.parquet"
    )
    # complexity_all_years = glob.glob(
    #     f"data/processed/{download_type}_{product_classification}_*_complexity.parquet"
    # )
    complexity_all = pd.concat(
        [pd.read_parquet(file) for file in complexity_all_years], axis=0
    )
    atlas_base_obj = AtlasCleaning(**ingestion_attrs)
    atlas_base_obj.save_parquet(
        complexity_all, "final", f"{product_classification}_cpy_all", "CPY"
    )
    del complexity_all


def run_unilateral_services(ingestion_attrs):
    unilateral_services = UnilateralServices(**ingestion_attrs)
    unilateral_services.save_parquet(
        unilateral_services.df, "final", f"unilateral_services", "Services"
    )
    del unilateral_services.df


def clean_up_intermediate_files(ingestion_attrs):
    base = AtlasCleaning(**ingestion_attrs)
    base.cleanup_files_from_dir(base.intermediate_data_path.iterdir())


import os
import sys
import argparse
import pandas as pd
import pyarrow as pq
import numpy as np
from time import strftime, localtime
from datetime import date, timedelta, datetime
from pathlib import Path

# Add the directory containing this script to Python path for imports
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Import configuration from config directory
sys.path.insert(0, str(script_dir))
from config.user_config import (
    get_paths_config,
    get_classifications,
    PROCESSING_STEPS,
    validate_config,
    print_config_summary,
    LOG_LEVEL,
    get_data_version,
)

# Import Atlas dependencies
from clean.utils.CIF_calculations import compute_distance
from clean.objects.base import AtlasCleaning
from clean.table_objects.aggregate_trade import AggregateTrade
from clean.table_objects.trade_analysis_cleaner import TradeAnalysisCleaner
from clean.table_objects.trade_data_reconciler import TradeDataReconciler
from clean.table_objects.country_country_product_year import CountryCountryProductYear
from clean.table_objects.complexity import Complexity
from clean.table_objects.unilateral_services import UnilateralServices
from clean.utils.classification_handler import (
    sitc_and_skip_processing,
    handle_product_classification,
)

import logging


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
    log_file = f"logs/atlas_processing_{data_version}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


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
    """
    start_year = ingestion_attrs["start_year"]
    end_year = ingestion_attrs["end_year"]
    product_classification = ingestion_attrs["product_classification"]

    logger.info(
        f"Starting Atlas cleaning for {product_classification} ({start_year}-{end_year})"
    )
    logger.info(f"Start time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

    base_obj = AtlasCleaning(**ingestion_attrs)
    aggregate_trade(ingestion_attrs)
    logger.info("Completed data aggregations")

    run_bilateral_mirroring_pipeline(ingestion_attrs)

    # Clean up intermediate files if configured
    if PROCESSING_STEPS.get("cleanup_intermediate", True):
        logger.info("Deleting intermediate processing files")
        clean_up_intermediate_files(ingestion_attrs)

    logger.info(f"Completed Atlas cleaning for {product_classification}")


def run_complexity(ingestion_attrs):
    product_classification = ingestion_attrs["product_classification"]
    logger.info(f"Starting complexity metrics generation for {product_classification}")

    generate_complexity_metrics(ingestion_attrs)
    logger.info(f"Completed complexity metrics for {product_classification}")

    # Clean up intermediate files if configured
    if PROCESSING_STEPS.get("cleanup_intermediate", True):
        logger.info("Deleting intermediate processing files")
        clean_up_intermediate_files(ingestion_attrs)


def run_bilateral_mirroring_pipeline(ingestion_attrs):
    """
    Process a single year through the complete reconciliation pipeline
    """
    product_classification = ingestion_attrs["product_classification"]
    download_type = ingestion_attrs["download_type"]

    for year in range(ingestion_attrs["start_year"], ingestion_attrs["end_year"] + 1):
        logger.info(
            f"Beginning trade mirror for year {year} ({product_classification})"
        )

        if sitc_and_skip_processing(year, product_classification, download_type):
            logger.info(f"Skipping year {year} for {product_classification}")
            continue

        product_classification = handle_product_classification(
            year, product_classification, download_type
        )[0]

        logger.debug(f"Computing distance matrix for year {year}")
        base_obj = AtlasCleaning(**ingestion_attrs)
        dist = pd.read_stata(base_obj.static_data_path / "dist_cepii.dta")
        df = compute_distance(base_obj, year, product_classification, dist)

        # cleaned country-country trade data with reporting quality metrics
        logger.debug(f"Running trade discrepancy analysis for {year}")
        trade_discrepancy_analysis = TradeAnalysisCleaner(year, df, **ingestion_attrs)

        # country-country trade data with reconciled values and accuracy weights
        logger.debug(f"Reconciling country-country trade data for {year}")
        country_trade_reconciler = TradeDataReconciler(
            year, trade_discrepancy_analysis.df, **ingestion_attrs
        )
        ccy = country_trade_reconciler.reconcile_country_country_estimates()

        if not sitc_and_skip_processing(year, product_classification, download_type):
            logger.debug(f"Reconciling product-level trade data for {year}")
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
            logger.info(f"Completed processing for year {year}")


def aggregate_trade(ingestion_attrs):
    """
    Aggregate raw trade data for all years and classifications.
    """
    product_classification = ingestion_attrs["product_classification"]
    download_type = ingestion_attrs["download_type"]

    for year in range(ingestion_attrs["start_year"], ingestion_attrs["end_year"] + 1):
        if sitc_and_skip_processing(year, product_classification, download_type):
            logger.debug(
                f"Skipping aggregation for year {year} ({product_classification})"
            )
            continue

        classifications = handle_product_classification(
            year, product_classification, download_type
        )

        logger.info(
            f"Aggregating data for {year} with classifications: {classifications}"
        )
        [
            AggregateTrade(year, product_class, **ingestion_attrs).run_aggregate_trade()
            for product_class in classifications
        ]


def generate_complexity_metrics(ingestion_attrs):
    """
    Generate complexity metrics for the specified classification and years
    """
    product_classification = ingestion_attrs["product_classification"]
    download_type = ingestion_attrs["download_type"]

    for year in range(ingestion_attrs["start_year"], ingestion_attrs["end_year"] + 1):
        logger.debug(f"Generating complexity metrics for {year}")
        complexity = Complexity(year, **ingestion_attrs)
        complexity.save_parquet(
            complexity.df,
            "intermediate",
            f"{download_type}_{product_classification}_{year}_complexity",
        )
        del complexity.df
        logger.debug(
            f"Completed complexity for {year} at {strftime('%Y-%m-%d %H:%M:%S', localtime())}"
        )

    logger.info("Consolidating complexity metrics across all years")
    complexity_all_years = complexity.intermediate_data_path.glob(
        f"{download_type}_{product_classification}_*_complexity.parquet"
    )
    complexity_all = pd.concat(
        [pd.read_parquet(file) for file in complexity_all_years], axis=0
    )
    atlas_base_obj = AtlasCleaning(**ingestion_attrs)
    atlas_base_obj.save_parquet(
        complexity_all, "final", f"{product_classification}_cpy_all", "CPY"
    )
    del complexity_all
    logger.info("Complexity consolidation completed")


def run_unilateral_services(ingestion_attrs):
    """
    Process unilateral services data
    """
    logger.info("Starting unilateral services processing")
    unilateral_services = UnilateralServices(**ingestion_attrs)
    unilateral_services.save_parquet(
        unilateral_services.df, "final", f"unilateral_services", "Services"
    )
    del unilateral_services.df
    logger.info("Unilateral services processing completed")


def clean_up_intermediate_files(ingestion_attrs):
    """
    Clean up intermediate processing files
    """
    logger.debug("Cleaning up intermediate files")
    base = AtlasCleaning(**ingestion_attrs)
    base.cleanup_files_from_dir(base.intermediate_data_path.iterdir())
    logger.debug("Intermediate file cleanup completed")


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

    # Print configuration summary if requested
    if args.config_summary:
        errors = validate_config()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  • {error}")
                sys.exit(0)
        print_config_summary()
        sys.exit(0)

    # Validate configuration
    errors = validate_config()
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  • {error}")
        sys.exit(1)

    # Get classifications to process
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

            if PROCESSING_STEPS.get("run_complexity", True):
                logger.info("Generating complexity metrics...")
                run_complexity(ingestion_attrs)

            classification_duration = datetime.now() - classification_start_time
            logger.info(f"Completed {description} in {classification_duration}")

        except Exception as e:
            logger.error(f"Error processing {classification}: {str(e)}", exc_info=True)
            raise

    # unilateral services processes once for all classifications
    if PROCESSING_STEPS.get("run_services", True):
        logger.info("Processing unilateral services...")
        # Use general config for services
        general_attrs = create_ingestion_attrs(None, None, 2023)
        run_unilateral_services(general_attrs)

    total_duration = datetime.now() - total_start_time
    logger.info("=" * 60)
    logger.info(f"ATLAS PROCESSING COMPLETED in {total_duration}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
