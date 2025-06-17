# Atlas Trade Data Processing Pipeline
A comprehensive data processing pipeline for cleaning and reconciling bilateral trade data from UN Comtrade for the Atlas of Economic Complexity.

# Overview
This pipeline processes raw bilateral trade data through a three-stage cleaning process:

1. Data Aggregation - Aggregate raw trade data across classifications and years
2. Country-level Reconciliation - Reconcile trade discrepancies between country pairs using mirroring techniques
3. Product-level Reconciliation - Generate final product-level trade data with complexity metrics

# Quick Start

**Prerequisites**

- Python 3.9+
- Poetry (for dependency management)
- Request a Federal Reserve Bank of St. Louis API key (for economic data)
    https://fred.stlouisfed.org/docs/api/api_key.html 

**Installation**

Clone the repository:
-`bash git clone <https://github.com/cid-harvard/atlas_cleaning.git>`
- `cd atlas_cleaning`

Install dependencies:
- `bash poetry install`

Set up environment variables:
- `bash export FRED_API_KEY="your_fred_api_key_here"`

Activate the environment:
- `bash poetry shell`

**Basic Usage**

Edit config/user_config.py to customize processing:

Which trade classifications to process

- `PROCESS_SITC = True`   # SITC data from 1962-2023
- `PROCESS_HS92 = True`   # HS92 data from 1992-2023  
- `PROCESS_HS12 = True`   # HS12 data from 2012-2023

Test mode - only process recent years (2020-2023) for faster testing
<br/>
`TEST_MODE = False`

**Processing steps**

| Step                    | Enabled   | Description   
| ----------------------- | ----------| -----------------------------------------------| 
| `run_cleaning`          | True      | # Main bilateral trade cleaning pipeline       | 
| `cleanup_intermediate`  | True      | # Clean up intermediate files after processing |


**Check configuration:**
`bash cd src`
`python main.py --config-summary`

**Run the full pipeline:**
`bash python main.py`







