# Comtrade Mirroring

Reconcile international trade data from UN Comtrade to produce mirrored bilateral trade data.

## What This Does

Transforms UN Comtrade data into clean bilateral trade statistics through a sophisticated mirroring process that reconciles discrepancies between exporter and importer reported values.

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/) for managing dependencies
- Premium UN Comtrade API key ([get one here](https://comtradeplus.un.org/))
- FRED API key ([get one here](https://fred.stlouisfed.org/docs/api/api_key.html))
- Comtrade data files (download from [comtrade-downloader](https://github.com/harvard-growth-lab/comtrade-downloader))

### Installation
```bash
git clone https://github.com/your-org/comtrade-mirroring.git
cd comtrade-mirroring
poetry install && poetry shell

# Set up environment variables
export COMTRADE_API_KEY="your_comtrade_key_here"
export FRED_API_KEY="your_fred_key_here"
```

## Quick Start

1. **Configure** processing settings in `user_config.py`
2. **Run** the pipeline: `python main.py`
3. **Find results** in your configured output directory

## Configuration

Edit `user_config.py`:

### Select Classifications to Process
```python
# Choose which trade classifications to process
PROCESS_SITC = False   # SITC data from 1962-END_YEAR
PROCESS_HS92 = True    # HS92 data from 1992-END_YEAR
PROCESS_HS12 = True    # HS12 data from 2012-END_YEAR

# Test mode - only process recent years
TEST_MODE = True
TEST_START_YEAR = 2020
END_YEAR = 2023
```

### Set Data Paths
```python
# Path to downloaded Comtrade files
DOWNLOADED_FILES_PATH = "/path/to/downloaded/comtrade/data"

# Results output directory
FINAL_OUTPUT_PATH = "/path/to/output/directory"
```

### Processing Steps
```python
PROCESSING_STEPS = {
    "run_cleaning": True,              # Main bilateral trade cleaning pipeline
    "delete_intermediate_files": True, # Clean up intermediate files
}
```

## Supported Classifications

**SITC (Standard International Trade Classification):**
- SITC Revision 1 (1962-current)
- SITC Revision 2 (1976-current)
- SITC Revision 3 (1988-current)

**HS (Harmonized System):**
- HS1992 (1992-current)
- HS1996 (1996-current) 
- HS2002 (2002-current)
- HS2007 (2007-current)
- HS2012 (2012-current)
- HS2017 (2017-current)
- HS2022 (2022-current)

## Output

Mirrored trade data saved as:
```
{FINAL_OUTPUT_PATH}/{DATA_VERSION}/mirrored_output/
├── H0/                    # HS92 bilateral trade data
│   ├── H0_2020.parquet
│   ├── H0_2021.parquet
│   └── ...
```

Each trade file contains: `year, exporter, importer, commoditycode, value_final, value_exporter, value_importer`

## How It Works

The mirroring pipeline consists of five processing steps. 

### 1. Preprocessing and trade aggregation

### 2. CIF-to-FOB adjustment

### 3. Compute country reliability scores

### 4. Country pair totals trade reconciliation

### 5. Product-level trade reconciliation


The final output provides reconciled trade values that combine exporter and importer reports based on the reporting country reliability scores.

## Repository Structure

```
comtrade-mirroring/
├── mirror/
│   ├── main.py                    # Main entry point
│   ├── user_config.py             # Configuration
│   ├── src/
│   │   ├── objects/
│   │   ├── table_objects/
│   │   └── utils/
    ├── logs/   
│   └── data/
│       ├── static/
├── pyproject.toml                # Python dependencies
└── README.md                     # This file
```

## Data Requirements

### Input Data Structure
The pipeline expects downloaded Comtrade data in this structure:
```
{DOWNLOADED_FILES_PATH}/
├── H0/                    # HS92 classification
│   ├── H0_2020.parquet
│   ├── H0_2021.parquet
│   └── ...
├── H4/                    # HS12 classification  
│   ├── H4_2020.parquet
│   └── ...
└── SITC/                  # SITC classification
    ├── SITC_2020.parquet
    └── ...
```

### Static Data Files
Required reference files (included in repository):
- Distance matrices (CEPII)
- Country concordances
- Product classification mappings
- Areas not specified listings

## Technical Details

### System Requirements
- **Memory**: 16GB+ RAM recommended for full processing
- **Storage**: 100GB+ available space for intermediate files
- **CPU**: Multi-core processor recommended


## License

Apache License, Version 2.0 - see LICENSE file.

## Citation

```bibtex
@Misc{comtrade_mirroring,
  author={Harvard Growth Lab},
  title={Comtrade Mirroring Pipeline},
  year={2025},
  howpublished = {\url{https://github.com/harvard-growth-lab/comtrade-mirroring}},
}
```





