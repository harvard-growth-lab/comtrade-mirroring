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

# Data version - will be used as folder name for output
# If None, will auto-generate based on yesterday's date
DATA_VERSION = None  # e.g., "2024_12_01" or None for auto-generation

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Base paths - Update these to match your environment
PATHS = {
    # top level directory of downloaded Comtrade files
    "downloaded_files_path": f"../../../../atlas/data/by_classification/aggregated_by_year/parquet",

    # results directory path
    "final_output_path": f"/n/hausmann_lab/lab/atlas/data/",
}

# =============================================================================
# WHAT TO PROCESS
# =============================================================================

# Which trade classifications to process (leave True for the ones you want)
PROCESS_SITC = False   # SITC data from 1962-2023
PROCESS_HS92 = False   # HS92 data from 1992-2023  
PROCESS_HS12 = True   # HS12 data from 2012-2023

PROCESS_HS96 = False
PROCESS_HS02 = False
PROCESS_HS07 = False
PROCESS_HS17 = False
PROCESS_HS22 = False

# Test mode - only process recent years (2020-2023) for faster testing
TEST_MODE = True

# =============================================================================
# PROCESSING OPTIONS
# =============================================================================

# Which processing steps to run
PROCESSING_STEPS = {
    "run_cleaning": True,        # Main bilateral trade cleaning pipeline
    "delete_intermediate_files": True, # Clean up intermediate files after processing
}

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Logging level
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# handles 
DOWNLOAD_TYPE = "as_reported"

# =============================================================================
# AUTOMATIC PATH GENERATION
# =============================================================================

root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

def get_data_version(data_version=DATA_VERSION):
    """Generate data version string if not manually specified"""
    if data_version:
        return data_version
    return f"rewrite_{(date.today() - timedelta(days=1)).strftime('%Y_%m_%d')}"


def get_paths_config(data_version=DATA_VERSION):
    """Generate full path configuration based on download type and data version"""
    
    if data_version is None:
        data_version = get_data_version()
    final_output_path = Path(PATHS["final_output_path"]) / data_version / "input"
    
    return {
        "downloaded_files_path": Path(PATHS['downloaded_files_path']),
        "root_dir": str(root_dir),
        "final_output_path": final_output_path,
        "download_type": DOWNLOAD_TYPE,
    }

def get_classifications():
    """Get the list of classifications to process based on settings"""
    classifications = []
    
    if PROCESS_SITC:
        start_year = 2020 if TEST_MODE else 1962
        classifications.append(("SITC", start_year, 2023, "SITC Revision 2"))
    
    if PROCESS_HS92:
        start_year = 2020 if TEST_MODE else 1992
        classifications.append(("H0", start_year, 2023, "HS92"))

    if PROCESS_HS96:
        start_year = 2020 if TEST_MODE else 1996
        classifications.append(("H1", start_year, 2023, "HS96"))

    if PROCESS_HS02:
        start_year = 2020 if TEST_MODE else 2002
        classifications.append(("H2", start_year, 2023, "HS96"))

    if PROCESS_HS07:
        start_year = 2020 if TEST_MODE else 2007
        classifications.append(("H3", start_year, 2023, "HS96"))
    
    if PROCESS_HS12:
        start_year = 2020 if TEST_MODE else 2012
        classifications.append(("H4", start_year, 2023, "HS12"))

    if PROCESS_HS17:
        start_year = 2020 if TEST_MODE else 2017
        classifications.append(("H5", start_year, 2023, "HS12"))

    if PROCESS_HS22:
        start_year = 2020 if TEST_MODE else 2022
        classifications.append(("H6", start_year, 2023, "HS12"))
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
    valid_classifications = ["H0", "H2", "H3", "H4", "H5", "H6", "SITC"]
    
    for classification, start_year, end_year, desc in get_classifications():
        if classification not in valid_classifications:
            errors.append(f"Invalid classification: {classification}")
        if start_year > end_year:
            errors.append(f"Invalid year range for {classification}: {start_year} > {end_year}")
        if end_year > date.today().year:
            errors.append(f"End year {end_year} is in the future for {classification}")
    
    return errors

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_config_summary():
    """Print a summary of current configuration"""
    print("=" * 60)
    print("ATLAS TRADE DATA PROCESSING CONFIGURATION")
    print("=" * 60)
    print(f"Data Version: {get_data_version()}")
    # print(f"Download Type: {DOWNLOAD_TYPE}")
    print(f"Test Mode: {'ON (2020-2023 only)' if TEST_MODE else 'OFF (full year range)'}")
    print()
    
    print("What will be processed:")
    classifications = get_classifications()
    if not classifications:
        print("  ⚠️  Nothing selected! Turn on PROCESS_SITC, PROCESS_HS92, or PROCESS_HS12")
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