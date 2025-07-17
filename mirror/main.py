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

from src.objects.orchestration import (
    create_ingestion_attrs,
    run_mirroring,
    clean_up_intermediate_files,
)

# from src.utils.CIF_calculations import compute_distance
# from src.objects.base import AtlasCleaning
# from src.table_objects.aggregate_trade import AggregateTrade
# from src.table_objects.trade_analysis_cleaner import TradeAnalysisCleaner
# from src.table_objects.trade_data_reconciler import TradeDataReconciler
# from src.table_objects.country_country_product_year import CountryCountryProductYear
# from src.utils.classification_handler import (
#     sitc_and_skip_processing,
#     handle_product_classification,
# )
from src.utils.logging import setup_logging


pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)


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
    logger.info(f"BILATERAL MIRRORING STARTING")
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
        logger.info(f"[{i}/{len(classifications)}] \nStarting {description}\n")

        # Create configuration for this classification
        ingestion_attrs = create_ingestion_attrs(classification, start_year, end_year)

        try:
            if PROCESSING_STEPS.get("run_cleaning", True):
                logger.info("Running cleaning pipeline...")
                run_mirroring(ingestion_attrs)

            classification_duration = datetime.now() - classification_start_time
            logger.info(f"\nCompleted {description} in {classification_duration}")

        except Exception as e:
            logger.error(f"Error processing {classification}: {str(e)}", exc_info=True)

        try:
            if PROCESSING_STEPS.get("delete_intermediate_files", True):
                logger.info("Deleting intermediate processing files...")
                clean_up_intermediate_files(ingestion_attrs)
        except Exception as e:
            logger.error(
                f"Error deleting intermediate processing files: {str(e)}", exc_info=True
            )

    total_duration = datetime.now() - total_start_time
    logger.info("=" * 60)
    logger.info(f"BILATERAL MIRRORING COMPLETED in {total_duration}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logger = setup_logging()
    main()
