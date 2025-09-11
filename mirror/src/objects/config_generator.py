"""
Configuration Generator
Reads YAML configuration and generates equivalent Python config file
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigGenerator:
    """Generate Python config files from YAML configuration"""
    
    def __init__(self, yaml_path: str):
        """Initialize with path to YAML config file"""
        self.yaml_path = Path(yaml_path)
        self.config_data = self._load_yaml()
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load and parse YAML configuration file"""
        try:
            with open(self.yaml_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML config file not found: {self.yaml_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
    
    def _map_classification_flags(self) -> Dict[str, bool]:
        """Map YAML classification settings to Python boolean flags"""
        classifications = self.config_data.get('classifications', {})
        
        return {
            'PROCESS_SITC1': classifications.get('sitc1', False),
            'PROCESS_SITC2': classifications.get('sitc2', False),
            'PROCESS_SITC3': classifications.get('sitc3', False),
            'PROCESS_HS92': classifications.get('hs92', False),
            'PROCESS_HS96': classifications.get('hs96', False),
            'PROCESS_HS02': classifications.get('hs02', False),
            'PROCESS_HS07': classifications.get('hs07', False),
            'PROCESS_HS12': classifications.get('hs12', False),
            'PROCESS_HS17': classifications.get('hs17', False),
            'PROCESS_HS22': classifications.get('hs22', False),
            'PROCESS_EB10': classifications.get('eb10', False),
        }
    
        return start_years
    
    def _map_processing_steps(self) -> Dict[str, bool]:
        """Map YAML processing steps to Python dictionary"""
        steps = self.config_data.get('mirror', {}).get('processing_steps', [])
        
        return {
            'run_cleaning': steps.get('run_cleaning', False),
            'delete_intermediate_files': steps.get('delete_intermediate_files', False),
        }
    
    def _get_paths(self) -> Dict[str, str]:
        """Extract path configuration from YAML"""
        mirror_config = self.config_data.get('mirror', {})
        paths_list = mirror_config.get('paths', [])
        
        # Convert list of dicts to single dict
        paths = {}
        for path_dict in paths_list:
            paths.update(path_dict)
        
        return {
            'downloaded_files_path': paths.get('downloaded_files_path', ''),
            'final_output_path': paths.get('final_output_path', ''),
        }
    
    def generate_python_config(self, output_path: str = 'config.py') -> None:
        """Generate Python configuration file from YAML data"""
        
        # Extract configuration values
        shared = self.config_data.get('shared', {})
        mirror = self.config_data.get('mirror', {})
        classifications = self.config_data.get('classifications', {})
        
        # Map values
        classification_flags = self._map_classification_flags()
        classification_start_years = self.config_data.get('classification_start_years')
        # classification_start_years = self._map_classification_start_years()
        processing_steps = self._map_processing_steps()
        paths = self._get_paths()
        
        # Generate Python config content
        config_content = self._generate_config_template(
            end_year=shared.get('end_year', 2023),
            log_level=shared.get('log_level', 'INFO'),
            data_version=mirror.get('data_version'),
            download_type=mirror.get('download_type', 'as_reported'),
            test_mode=mirror.get('test_mode', False),
            test_start_year=mirror.get('test_start_year', 2020),
            classification_flags=classification_flags,
            processing_steps=processing_steps,
            paths=paths,
            classification_start_years=classification_start_years,
        )
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(config_content)
        
        print(f"Generated Python config file: {output_path}")
    
    def _generate_config_template(self, **kwargs) -> str:
        """Generate the actual Python config file content"""
        
        classification_flags = kwargs['classification_flags']
        processing_steps = kwargs['processing_steps']
        paths = kwargs['paths']
        classification_start_years = kwargs['classification_start_years']
        
        return f'''"""
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
DATA_VERSION = f"{kwargs['data_version']}"  # e.g., "2024_12_01" or None for auto-generation

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Base paths - Update these to match your environment

# directory of aggregated data files
# Example: DOWNLOADED_FILES_PATH = (/data/as_reported/aggregated_by_year/parquet)
DOWNLOADED_FILES_PATH = (
    f"{paths['downloaded_files_path']}"
)

# results directory path
FINAL_OUTPUT_PATH = (
    f"{paths['final_output_path']}"
)


PATHS = {{
    "downloaded_files_path": DOWNLOADED_FILES_PATH,
    "final_output_path": FINAL_OUTPUT_PATH,
}}

# =============================================================================
# CLASSIFICATION VINTAGE & YEAR RANGE SELECTION
# =============================================================================

# MUST HAVE DATA DOWNLOADED FROM COMTRADE-DOWNLOADER 
# END YEAR FOR PROCESSING
END_YEAR = {kwargs['end_year']}

# Which trade classifications to process (leave True for the ones you want)
PROCESS_SITC1 = {classification_flags['PROCESS_SITC1']}  # SITC data from 1962-END_YEAR
PROCESS_SITC2 = {classification_flags['PROCESS_SITC2']}  # SITC data from 1976-END_YEAR
PROCESS_SITC3 = {classification_flags['PROCESS_SITC3']}  # SITC data from 1988-END_YEAR

PROCESS_HS92 = {classification_flags['PROCESS_HS92']}  # HS92 data from 1992-END_YEAR
PROCESS_HS12 = {classification_flags['PROCESS_HS12']}  # HS12 data from 2012-END_YEAR
PROCESS_HS96 = {classification_flags['PROCESS_HS96']}
PROCESS_HS02 = {classification_flags['PROCESS_HS02']}
PROCESS_HS07 = {classification_flags['PROCESS_HS07']}
PROCESS_HS17 = {classification_flags['PROCESS_HS17']}
PROCESS_HS22 = {classification_flags['PROCESS_HS22']}

PROCESS_EB10 = {classification_flags['PROCESS_EB10']}

# Test mode - only process recent years (TEST_START_YEAR- END_YEAR)
TEST_MODE = {kwargs['test_mode']}
# must be set for year classification vintage was released or later
TEST_START_YEAR = {kwargs['test_start_year']}

CLASSIFICATION_START_YEARS = {classification_start_years}

# =============================================================================
# DATA PROCESSING STEPS
# =============================================================================

PROCESSING_STEPS = {{
    "run_cleaning": {processing_steps['run_cleaning']},  # Main bilateral trade cleaning pipeline
    "delete_intermediate_files": {processing_steps['delete_intermediate_files']},  # Clean up intermediate files after processing
}}

# =============================================================================
# LOGGING LEVEL
# =============================================================================

LOG_LEVEL = "{kwargs['log_level']}"  # Options: DEBUG, INFO, WARNING, ERROR

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# determines type of data to download from Comtrade
# do not recommend changing this
DOWNLOAD_TYPE = "{kwargs['download_type']}"  # alternative is "by_classification"

# =============================================================================
# CONFIGURATION DICTIONARIES
# =============================================================================

classifications_dict = {{
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
    "EB10": PROCESS_EB10,
}}


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
'''
    
    def _format_classification_start_years(self, start_years: Dict[str, int]) -> str:
        """Format classification start years dictionary for Python code"""
        lines = []
        lines.append("    # Standard International Trade Classification (SITC)")
        for key in ['S1', 'S2', 'S3']:
            if key in start_years:
                year = start_years[key]
                comment = {
                    'S1': 'SITC Revision 1 (1962-present)',
                    'S2': 'SITC Revision 2 (1976-present)', 
                    'S3': 'SITC Revision 3 (1988-present)'
                }[key]
                lines.append(f'    "{key}": {year},  # {comment}')
        
        lines.append("    # Harmonized System (HS) Classifications")
        for key in ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            if key in start_years:
                year = start_years[key]
                comment = {
                    'H0': 'HS Combined (1992-present)',
                    'H1': 'HS 1992 vintage (1996-present)',
                    'H2': 'HS 2002 vintage (2002-present)',
                    'H3': 'HS 2007 vintage (2007-present)',
                    'H4': 'HS 2012 vintage (2012-present)',
                    'H5': 'HS 2017 vintage (2017-present)',
                    'H6': 'HS 2022 vintage (2022-present)'
                }[key]
                lines.append(f'    "{key}": {year},  # {comment}')
                
        for key in ['EB10']:
            if key in start_years:
                year = start_years[key]
                comment = {
                    'EB10': 'EBOPS Services (2005-present)',
                }[key]
                lines.append(f'    "{key}": {year},  # {comment}')

        
        return '\n'.join(lines)
