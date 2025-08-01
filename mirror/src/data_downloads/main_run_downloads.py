import os
from pathlib import Path
from glob import glob
from user_config import get_classifications
from src.data_downloads.params.download_tables_conf import (
    dataverse_datasets,
    table_display_names,
)
from src.data_downloads.objects.data_preparer import (
    DataPreparer,
    DataPreparerConfig,
)
from src.data_downloads.objects.data_formatter import IDtoHumanReadableMapper
from user_config import PATHS


def run_data_preparer():
    config = DataPreparerConfig(PATHS)
    preparer = DataPreparer(config)
    vintages = [vin for (vin, _, _, _) in get_classifications()]
    preparer.prepare_bilateral_reported_trade(vintages)
    # preparer.prepare_conversion_weights()


if __name__ == "__main__":
    run_data_preparer()
