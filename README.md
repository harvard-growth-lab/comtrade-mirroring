# Atlas Trade Data Processing Pipeline
A comprehensive data processing pipeline for cleaning and reconciling bilateral trade data from UN Comtrade for the Atlas of Economic Complexity.

# Overview
This pipeline processes raw bilateral trade data through a three-stage cleaning process:

1. Data Aggregation

- Reads raw bilateral trade data files from UN Comtrade
- Aggregates data by year, country pair, and product classification
- Handles different product classification systems (SITC, HS92, HS12)
- Standardizes data formats and codes across years

2. Trade Analysis & Country-level Reconciliation

- Distance Calculation: Computes geographic distances between countries using CEPII data
- Trade Discrepancy Analysis: Identifies and analyzes differences between reporter and partner country data
- Quality Assessment: Generates reliability metrics for each country's trade reporting
- Bilateral Reconciliation: Uses statistical mirroring techniques to reconcile conflicting trade reports
- Accuracy Weighting: Applies weights based on reporting quality to create best estimates

3. Product-level Reconciliation

- Product-level Processing: Extends country-level reconciliation to detailed product categories
- Bilateral Trade Matrices: Creates comprehensive bilateral trade datasets
- Data Validation: Performs consistency checks across different aggregation levels
- Final Output Generation: Produces clean, reconciled trade data ready for Atlas ingestion

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
- `TEST_MODE = False`

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







