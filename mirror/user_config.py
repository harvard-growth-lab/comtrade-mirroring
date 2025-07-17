"""
Bilateral Trade Data Processing Configuration

This configuration file controls how the bilateral trade data cleaning pipeline runs.
Edit the settings below to match your environment and requirements.
"""

from pathlib import Path
import sys
from datetime import date, timedelta

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

"""
Data version - will be used as folder name for output
If None, auto-generates based on today's date
"""
DATA_VERSION = None  # e.g., "2024_12_01" or None for auto-generation

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Base paths - Update these to match your environment

# directory of aggregated data files
# Example: DOWNLOADED_FILES_PATH = (/data/as_reported/aggregated_by_year/parquet)
DOWNLOADED_FILES_PATH = (
    f"../../mirror/data/as_reported/aggregated_by_year/parquet"
)

# results directory path
FINAL_OUTPUT_PATH = (
    f"/path/to/output/data"
)


PATHS = {
    "downloaded_files_path": DOWNLOADED_FILES_PATH,
    "final_output_path": FINAL_OUTPUT_PATH,
}

# =============================================================================
# CLASSIFICATION VINTAGE & YEAR RANGE SELECTION
# =============================================================================

# MUST HAVE DATA DOWNLOADED FROM COMTRADE-DOWNLOADER 
# END YEAR FOR PROCESSING
END_YEAR = 2023

# Which trade classifications to process (leave True for the ones you want)
PROCESS_SITC1 = False  # SITC data from 1962-END_YEAR
PROCESS_SITC2 = False  # SITC data from 1976-END_YEAR
PROCESS_SITC3 = False  # SITC data from 1988-END_YEAR

PROCESS_HS92 = False  # HS92 data from 1992-END_YEAR
PROCESS_HS12 = False  # HS12 data from 2012-END_YEAR
PROCESS_HS96 = False
PROCESS_HS02 = False
PROCESS_HS07 = False
PROCESS_HS17 = False
PROCESS_HS22 = True

# Test mode - only process recent years (TEST_START_YEAR- END_YEAR)
TEST_MODE = True
# must be set for year classification vintage was released or later
TEST_START_YEAR = 2020

# =============================================================================
# DATA PROCESSING STEPS
# =============================================================================

PROCESSING_STEPS = {
    "run_cleaning": True,  # Main bilateral trade cleaning pipeline
    "delete_intermediate_files": False,  # Clean up intermediate files after processing
}

# =============================================================================
# LOGGING LEVEL
# =============================================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# determines type of data to download from Comtrade
# do not recommend changing this
DOWNLOAD_TYPE = "as_reported"  # alternative is "by_classification"

# =============================================================================
# PATH HANDLING
# =============================================================================

root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))


def get_data_version(data_version=DATA_VERSION):
    """Generate data version string if not manually specified"""
    if data_version:
        return data_version
    return f"{(date.today()).strftime('%Y_%m_%d')}"


def get_paths_config(data_version=DATA_VERSION):
    """Generate full path configuration based on download type and data version"""

    if data_version is None:
        data_version = get_data_version()
    final_output_path = (
        Path(PATHS["final_output_path"]) / data_version / "mirrored_output"
    )
    final_output_path.mkdir(exist_ok=True, parents=True)

    return {
        "downloaded_files_path": Path(PATHS["downloaded_files_path"]),
        "root_dir": str(root_dir),
        "final_output_path": final_output_path,
        "download_type": DOWNLOAD_TYPE,
    }


def get_classifications():
    """Get the list of classifications to process based on settings"""
    classifications = []

    if PROCESS_SITC1:
        start_year = TEST_START_YEAR if TEST_MODE else 1962
        classifications.append(("S1", start_year, END_YEAR, "SITC Revision 1"))

    if PROCESS_SITC2:
        start_year = TEST_START_YEAR if TEST_MODE else 1976
        classifications.append(("S2", start_year, END_YEAR, "SITC Revision 2"))

    if PROCESS_SITC3:
        start_year = TEST_START_YEAR if TEST_MODE else 1988
        classifications.append(("S3", start_year, END_YEAR, "SITC Revision 3"))


    if PROCESS_HS92:
        start_year = TEST_START_YEAR if TEST_MODE else 1992
        classifications.append(("H0", start_year, END_YEAR, "HS92"))

    if PROCESS_HS96:
        start_year = TEST_START_YEAR if TEST_MODE else 1996
        classifications.append(("H1", start_year, END_YEAR, "HS96"))

    if PROCESS_HS02:
        start_year = TEST_START_YEAR if TEST_MODE else 2002
        classifications.append(("H2", start_year, END_YEAR, "HS02"))

    if PROCESS_HS07:
        start_year = TEST_START_YEAR if TEST_MODE else 2007
        classifications.append(("H3", start_year, END_YEAR, "HS07"))

    if PROCESS_HS12:
        start_year = TEST_START_YEAR if TEST_MODE else 2012
        classifications.append(("H4", start_year, END_YEAR, "HS12"))

    if PROCESS_HS17:
        start_year = TEST_START_YEAR if TEST_MODE else 2017
        classifications.append(("H5", start_year, END_YEAR, "HS17"))

    if PROCESS_HS22:
        start_year = TEST_START_YEAR if TEST_MODE else 2022
        classifications.append(("H6", start_year, END_YEAR, "HS22"))
    return classifications


# =============================================================================
# VALIDATION
# =============================================================================


def validate_config():
    """Validate configuration settings"""
    errors = []

    # Check paths exist
    for path_name, path_value in PATHS.items():
        if not Path(path_value).exists():
            errors.append(f"Path does not exist: {path_name} = {path_value}")

    # Check download type
    if DOWNLOAD_TYPE not in ["by_classification", "as_reported"]:
        errors.append(f"Invalid DOWNLOAD_TYPE: {DOWNLOAD_TYPE}")

    # Check classifications
    valid_classifications = [
        "H0",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "SITC",
        "S1",
        "S2",
        "S3",
    ]

    for classification, start_year, end_year, desc in get_classifications():
        if classification not in valid_classifications:
            errors.append(f"Invalid classification: {classification}")
        if start_year > end_year:
            errors.append(
                f"Invalid year range for {classification}: {start_year} > {end_year}"
            )
        if end_year > date.today().year:
            errors.append(f"End year {end_year} is in the future for {classification}")

    return errors


# =============================================================================
# REPORTING / INFO
# =============================================================================


def print_config_summary():
    """Print a summary of current configuration"""
    print("=" * 60)
    print("BILATERAL MIRRORING CONFIGURATION")
    print("=" * 60)
    print(f"Data Version: {get_data_version()}")
    # print(f"Download Type: {DOWNLOAD_TYPE}")
    print(
        f"Test Mode: {'ON (2020-END_YEAR only)' if TEST_MODE else 'OFF (full year range)'}"
    )
    print()

    print("What will be processed:")
    classifications = get_classifications()
    if not classifications:
        print(
            "  ⚠️  Nothing selected! Turn on PROCESS_SITC, PROCESS_HS92, or PROCESS_HS12"
        )
    else:
        for classification, start_year, end_year, desc in classifications:
            print(f"  ✓ {desc}: {start_year}-{end_year}")
    print()

    print("Processing steps:")
    for step, enabled in PROCESSING_STEPS.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {step}")
    print()

    # Validation
    errors = validate_config()
    if errors:
        print("⚠️  Configuration Errors:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("✅ Configuration is valid")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
