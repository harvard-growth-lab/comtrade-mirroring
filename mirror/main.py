import os
from pathlib import Path
import sys
import argparse
import pandas as pd
from datetime import datetime
import importlib

from mirror.src.objects.orchestration import (
    create_ingestion_attrs,
    run_mirroring,
    clean_up_intermediate_files,
)
from mirror.src.utils.logging import setup_logging
from mirror.src.objects.config_generator import ConfigGenerator

from mirror.src.utils.handle_config import get_data_version, print_config_summary, validate_config


def run(config_module):
    """
    Main execution function that runs the Atlas cleaning pipeline
    based on configuration settings
    """

    # Get config variables from the imported module
    classifications = config_module.classifications
    processing_steps = config_module.PROCESSING_STEPS
    test_mode = config_module.TEST_MODE
    paths = config_module.PATHS
    download_type = config_module.DOWNLOAD_TYPE
    data_version = config_module.DATA_VERSION

    data_version = get_data_version(data_version)

    print_config_summary(test_mode, data_version, classifications, processing_steps)

    errors = validate_config(paths, download_type, classifications)
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  • {error}")
        sys.exit(1)

    if not classifications:
        logger.error("No classifications selected! Check your config settings.")
        sys.exit(1)


    # Show what will be processed
    logger.info("=" * 60)
    logger.info(f"BILATERAL MIRRORING STARTING")
    logger.info("=" * 60)
    logger.info(f"Data version: {data_version}")
    logger.info(f"Processing {len(classifications)} classification(s)")

    for classification, start_year, end_year, description in classifications:
        years_count = end_year - start_year + 1
        logger.info(f"  • {description}: {start_year}-{end_year} ({years_count} years)")

    # Show processing steps
    enabled_steps = [step for step, enabled in processing_steps.items() if enabled]
    logger.info(f"Processing steps: {', '.join(enabled_steps)}")

    logger.info("=" * 60)

    # Process each classification
    total_start_time = datetime.now()

    base_attrs = {
        "data_version": data_version,
        "paths": paths,
        "download_type" : download_type
        }

    for i, (classification, start_year, end_year, description) in enumerate(
        classifications, 1
    ):
        classification_start_time = datetime.now()
        logger.info(f"[{i}/{len(classifications)}] \nStarting {description}\n")

        ingestion_attrs = create_ingestion_attrs(classification, start_year, end_year, base_attrs)
        try:
            if processing_steps.get("run_cleaning", True):
                logger.info("Running cleaning pipeline...")
                run_mirroring(ingestion_attrs)

            classification_duration = datetime.now() - classification_start_time
            logger.info(f"\nCompleted {description} in {classification_duration}")

        except Exception as e:
            logger.error(f"Error processing {classification}: {str(e)}", exc_info=True)

        try:
            if processing_steps.get("delete_intermediate_files", True):
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


def main():
    run(config_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description="Run Comtrade data processing with specified config"
    )
    parser.add_argument(
        "--config",
        choices=["user_config", "atlas_dev_config", "dev"],
        default="user_config",
        help="Config file to use (default: user_config)",
    )

    args = parser.parse_args()
    config_file = args.config

    # Generate Python config from YAML
    if config_file == "user_config":
        config_path = Path(f"{config_file}.yaml")
    else:
        config_path = Path("config") / f"{config_file}.yaml"
    generator = ConfigGenerator(config_path)
    generator.generate_python_config('config/generated_config.py')

    try:
        config_module = importlib.import_module("config.generated_config")
    except ImportError:
        raise ImportError(f"Config module '{config_file}' not found")

    logger = setup_logging(config_module.LOG_LEVEL, config_module.DATA_VERSION)
    logger.info(f"Using config: {args.config}")

    main()
