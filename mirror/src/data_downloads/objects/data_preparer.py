from mirror.config.user_config import PATHS
from mirror.config.user_config import DATA_VERSION
from mirror.src.data_downloads.objects.data_formatter import IDtoHumnaReadableMapper
from pathlib import Path
import pandas as pd


class DataPreparerConfig:
    def __init__(self, PATHS):
        self.common_data_path = Path(PATHS["common_data_path"])
        # add conversion weights to the common data path
        self.bilateral_reported_trade_path = (
            Path(PATHS["final_output_path"]) / DATA_VERSION / "mirrored_output"
        )
        self.conversion_weights_path = self.common_data_path / "conversion_weights"

        # load metadata
        self.metadata_path = self.common_data_path / "data_downloads"


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
        self.human_readable_formatter = IDtoHumnaReadableMapper(config)

    def prepare_conversion_weights(self):
        pass

    def prepare_bilateral_reported_trade(self, vintages):

        for vintage in vintages:
            dir_path = self.config.bilateral_reported_trade_path / vintage
            files = dir_path.glob("*.parquet")
            trade_dfs = []
            for file in files:
                df = pd.read_parquet(file)
                df = self.human_readable_formatter.map_country_id_to_country_iso(df)
                trade_dfs.append(df)
            trade_df = pd.concat(trade_dfs)
            # format datatypes and columns
            # handle metadata
            trade_df.to_parquet(dir_path / "trade_df.parquet")
