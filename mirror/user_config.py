"""
Bilateral Trade Data Processing Configuration

This configuration file controls how the bilateral trade data cleaning pipeline runs.
Edit the settings below to match your environment and requirements.
"""

from pathlib import Path
import sys
from datetime import date, timedelta
from mirror.src.utils.handle_config import get_classifications_list

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

"""
Data version - will be used as folder name for output
If None, auto-generates based on today's date
"""
DATA_VERSION = "2025_09_24"  # e.g., "2024_12_01" or None for auto-generation

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
    f"/n/hausmann_lab/lab/atlas/data"
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
PROCESS_HS02 = True
PROCESS_HS07 = False
PROCESS_HS17 = False
PROCESS_HS22 = True

# Test mode - only process recent years (TEST_START_YEAR- END_YEAR)
TEST_MODE = True
# must be set for year classification vintage was released or later
TEST_START_YEAR = 2023

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
# CONFIGURATION DICTIONARIES
# =============================================================================

CLASSIFICATION_START_YEARS = {
    # Standard International Trade Classification (SITC)
    "S1": 1962,  # SITC Revision 1 (1962-present)
    "S2": 1976,  # SITC Revision 2 (1976-present)
    "S3": 1988,  # SITC Revision 3 (1988-present)
    # Harmonized System (HS) Classifications
    "H0": 1992,  # HS Combined (1992-present)
    "H1": 1996,  # HS 1992 vintage (1996-present)
    "H2": 2002,  # HS 2002 vintage (2002-present)
    "H3": 2007,  # HS 2007 vintage (2007-present)
    "H4": 2012,  # HS 2012 vintage (2012-present)
    "H5": 2017,  # HS 2017 vintage (2017-present)
    "H6": 2022,  # HS 2022 vintage (2022-present)
}

classifications_dict = {
    "HS92": PROCESS_HS92,
    "HS12": PROCESS_HS12,
    "HS96": PROCESS_HS96,
    "HS02": PROCESS_HS02,
    "HS07": PROCESS_HS07,
    "HS17": PROCESS_HS17,
    "HS22": PROCESS_HS22,
    "SITC1": PROCESS_SITC1,
    "SITC2": PROCESS_SITC2,
    "SITC3": PROCESS_SITC3,
}



if TEST_MODE:
    classifications = get_classifications_list(
        classifications_dict, 
                                               END_YEAR, 
                                               CLASSIFICATION_START_YEARS,
                                               TEST_START_YEAR
                                               )
else:
    classifications = get_classifications_list(
        classifications_dict, 
                                               END_YEAR, 
                                               CLASSIFICATION_START_YEARS,
                                               )



