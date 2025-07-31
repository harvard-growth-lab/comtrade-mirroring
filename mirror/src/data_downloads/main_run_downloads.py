import os
import logging
from pathlib import Path
from glob import glob
from mirror.src.data_downloads.params.download_tables_conf import (
    dataverse_datasets,
    table_display_names,
)
from mirror.src.data_downloads.objects.data_preparer import (
    DataPreparer,
    DataPreparerConfig,
)
from mirror.src.data_downloads.objects.data_formatter import IDtoHumnaReadableMapper
from mirror.user_config import PATHS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_data_preparer():
    config = DataPreparerConfig(PATHS)
    human_readable_formatter = IDtoHumnaReadableMapper(config)
    preparer = DataPreparer(config)
    preparer.prepare_conversion_weights()
    preparer.prepare_bilateral_reported_trade()


if __name__ == "__main__":
    run_data_preparer()
