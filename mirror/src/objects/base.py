"the parent object for the bilateral mirroring script"

import os
from os import path
from pathlib import Path
import pandas as pd
import typing
import glob
import pyarrow.parquet as pq
import shutil
from datetime import datetime
from src.utils.logging import get_logger

logger = get_logger(__name__)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)


class AtlasCleaning(object):
    # Classification names & levels
    PRODUCT_CLASSIFICATIONS = [
        "H0",
        "HS",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "S1",
        "S2",
        "S3",
        "ST",
    ]

    HIERARCHY_LEVELS = {
        "H0": (0, 2, 4, 6),
        "HS": (0, 2, 4, 6),
        "H1": (0, 2, 4, 6),
        "H2": (0, 2, 4, 6),
        "H3": (0, 2, 4, 6),
        "H4": (0, 2, 4, 6),
        "H5": (0, 2, 4, 6),
        "H6": (0, 2, 4, 6),
        "SITC": (0, 2, 4),
        "S1": (0, 2, 4),
        "S2": (0, 2, 4),
        "S3": (0, 2, 4),
        "ST": (0, 2, 4),
    }

    REGIONAL_GROUP_TYPES = ["world", "region", "subregion"]

    INGESTION_OUTPUT_FORMATS = ["parquet", "hdf5"]

    def __init__(
        self,
        start_year,
        end_year,
        downloaded_files_path,
        root_dir,
        final_output_path,
        product_classification,
        download_type,
    ):

        self.latest_data_year = (
            datetime.now().year - 2
            if datetime.now().month > 4
            else datetime.now().year - 3
        )

        # INPUTS
        self.download_type = download_type
        self.product_classification = product_classification

        self.downloaded_files_path = Path(downloaded_files_path)
        self.root_dir = Path(root_dir)
        self.data_path = self.root_dir / "data"
        self.final_output_path = Path(final_output_path)
        self.static_data_path = self.data_path / "static"
        self.intermediate_data_path = self.data_path / "intermediate"

        self.path_mapping = {
            "static": self.static_data_path,
            "intermediate": self.intermediate_data_path,
            "final": self.final_output_path,
        }
        self._setup_paths()
        self.product_class_system = self.get_product_class_system()

        self.fred_api_key = os.environ.get("FRED_API_KEY")
        if not self.fred_api_key:
            raise ValueError(
                "FRED API key required: pass fred_api_key or set FRED_API_KEY env var"
            )
        self.missing_data = False

        # data inputs
        self.dist_cepii = pd.read_stata(
            os.path.join(self.static_data_path, "dist_cepii.dta")
        )
        self.ans_partners = pd.read_csv(
            os.path.join(self.static_data_path, "areas_not_specified.csv")
        )

        self.df = None
        self.start_year = start_year
        self.end_year = end_year

    def get_attrs(self):
        return {
            "start_year": self.start_year,
            "end_year": self.end_year,
            "downloaded_files_path": self.downloaded_files_path,
            "root_dir": self.root_dir,
            "final_output_path": self.final_output_path,
            # used for comparison to atlas production data and generated data
            "comparison_file_path": self.comparison_file_path,
            "atlas_common_path": self.atlas_common_path,
            "product_classification": self.product_classification,
        }

    def _setup_paths(self):
        paths = [
            self.data_path,
            self.static_data_path,
            self.intermediate_data_path,
            self.final_output_path,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def get_product_class_system(self):
        if self.product_classification in ["S1", "S2", "S3", "SITC"]:
            return "SITC"
        else:
            return "HS"

    def load_parquet(self, data_folder, table_name: str):
        read_dir = self.path_mapping[data_folder]
        if read_dir.exists():
            df = pd.read_parquet(Path(read_dir / f"{table_name}.parquet"))
        else:
            raise ValueError("{data_folder} is not a valid data folder")
        return df

    def save_parquet(
        self,
        df,
        data_folder,
        table_name: str,
        parent_folder="",
    ):
        save_dir = self.path_mapping[data_folder]
        save_dir.mkdir(exist_ok=True)
        if data_folder == "final":
            save_dir = save_dir / parent_folder
            save_dir.mkdir(exist_ok=True)

        save_path = save_dir / f"{table_name}.parquet"
        df.to_parquet(save_path, index=False)

    def cleanup_files_from_dir(self, files):
        for file in files:
            if file.is_file():
                file.unlink()
