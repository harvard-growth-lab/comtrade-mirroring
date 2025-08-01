from src.data_downloads.objects.base import logger
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
from user_config import PATHS
from user_config import DATA_VERSION
from src.data_downloads.objects.data_formatter import (
    IDtoHumanReadableMapper,
    DataFormatter,
    FileMetaData,
)
from pathlib import Path
import pandas as pd
from src.data_downloads.params.download_tables_conf import (
    trade_data_cols_renamed,
    trade_data_optimized_dtypes,
    trade_data_optimized_pyarrow_dtypes,
)


class DataPreparerConfig:
    """
    Configures the data preparer.
    """

    FILE_TYPE = "parquet"

    def __init__(self, PATHS: dict):
        """
        Initializes the data preparer config.
        """
        self.common_data_path = Path(PATHS["common_data_path"])
        # load metadata
        self.metadata_path = self.common_data_path / "data_downloads"

        # add conversion weights to the common data path
        self.bilateral_reported_trade_path = (
            Path(PATHS["final_output_path"]) / DATA_VERSION / "mirrored_output"
        )
        self.conversion_weights_path = self.common_data_path / "conversion_weights"
        self.data_downloads_path = (
            Path(PATHS["final_output_path"])
            / DATA_VERSION
            / "mirrored_output"
            / "data_downloads"
        )
        self.set_paths(self.data_downloads_path)

    def set_paths(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)


class DataPreparer:
    """
    Converts raw trade data into user-friendly download files.

    Key responsibilities:
    1. Load country/product code mappings
    2. Process different types of files
    3. Generate metadata and documentation
    """

    CLASSIFICATION_SYSTEM = ["HS", "SITC"]
    CLASSIFICATION_VINTAGES = [
        "S1",
        "S2",
        "S3",
        "H0",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    ]

    # Hierarchy levels for each classification vintage
    HIERARCHY_LEVELS = {
        "HS": (1, 2, 4, 6),
        "SITC": (1, 2, 4),
    }

    def __init__(
        self,
        config: DataPreparerConfig,
    ):
        self.config = config
        self.metadata = FileMetaData(config)
        self.data_formatter = DataFormatter(config)
        self.human_readable_formatter = IDtoHumanReadableMapper(config)

    def prepare_data(self, vintage: str, table_name: str):
        pass

    def prepare_conversion_weights(self):
        pass

    def prepare_bilateral_reported_trade(self, vintages: List[str]) -> None:
        for vintage in vintages:
            self._process_vintage(vintage)

    def _process_vintage(self, vintage: str) -> None:

        logger.info(f"Preparing bilateral reported trade for vintage {vintage}")
        dir_path = self.config.bilateral_reported_trade_path / vintage
        files = list(dir_path.glob("*.parquet"))
        if not files:
            logger.error(f"No files found for vintage {vintage}")
            return
        schema = pa.schema(trade_data_optimized_pyarrow_dtypes)
        parquet_path = (
            self.config.data_downloads_path
            / f"{vintage}.parquet"
            # / self.metadata.get_metadata("bilateral_reported_trade")["file_name"]
        )
        self.stream_parquet_files(
            files,
            parquet_path,
            schema,
            trade_data_cols_renamed,
            trade_data_optimized_dtypes,
        )

    def stream_parquet_files(self, files, parquet_path, schema, col_names, dtypes):
        with pq.ParquetWriter(parquet_path, schema) as writer:
            for file in files:
                try:
                    df = pd.read_parquet(file)
                    df = df.rename(columns=col_names)
                    df = df.astype(dtypes)
                    table = pa.Table.from_pandas(df, schema=schema)
                    writer.write_table(table)
                    del df

                except FileNotFoundError:
                    logger.error(f"File {file} not found")
                    continue
                except Exception as e:
                    logger.error(f"Error reading file {file}: {e}")
                    continue
