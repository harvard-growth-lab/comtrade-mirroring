"the parent object for Atlas Cleaning script"

import os
from os import path
from pathlib import Path
import pandas as pd
import typing
import glob
import pyarrow.parquet as pq
import logging
import shutil
from datetime import datetime

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)


class _AtlasCleaning(object):
    # Classification names & levels
    PRODUCT_CLASSIFICATIONS = ["H0", "HS", "S1", "S2", "ST"]

    HIERARCHY_LEVELS = {
        "H0": (0, 2, 4, 6),
        "HS": (0, 2, 4, 6),
        "H4": (0, 2, 4, 6),
        "H5": (0, 2, 4, 6),
        "SITC": (0, 2, 4),
        "S1": (0, 2, 4),
        "S2": (0, 2, 4),
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
        comparison_file_path,
        atlas_common_path,
        product_classification,
        download_type,
    ):
        self.latest_year = datetime.now().year - 2 if datetime.now().month > 4 else datetime.now().year - 3

        # INPUTS
        self.download_type = download_type
        self.product_classification = product_classification

        self.downloaded_files_path = Path(downloaded_files_path)
        self.root_dir = Path(root_dir)
        self.data_path = self.root_dir / "data"
        self.final_output_path = Path(final_output_path)
        # self.comparison_file_path = os.path.join(comparison_file_path)
        # self.atlas_common_path = os.path.join(atlas_common_path)
        self.raw_data_path = self.data_path / "raw"
        self.intermediate_data_path = self.data_path / "new_intermediate"
        self.processed_data_path = self.data_path / "processed"

        self.path_mapping = {
            "raw": self.raw_data_path,
            "intermediate": self.intermediate_data_path,
            "processed": self.processed_data_path,
            "final": self.final_output_path,
        }

        # data inputs
        self.dist_cepii = pd.read_stata(
            os.path.join(self.raw_data_path, "dist_cepii.dta")
        )
        self.ans_partners = pd.read_csv(
            os.path.join(self.raw_data_path, "areas_not_specified.csv")
        )

        self.df = None
        self.start_year = start_year
        self.end_year = end_year

        self.wdi_path = os.path.join(self.raw_data_path, "wdi_extended.dta")

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

    def cleanup_intermediate_files(self, force=False):
        """
        Delete all files and subdirectories in an intermediate file folder.

        Args:
            intermediate_folder (str or Path): Path to the intermediate files folder
            force (bool): If True, ignore errors and force deletion. Default False.

        Returns:
            bool: True if cleanup successful, False otherwise
        """
        try:
            # Check if folder exists
            if not self.intermediate_data_path.exists():
                logging.warning(
                    f"Intermediate folder does not exist: {self.intermediate_data_path}"
                )
                return True

            if not self.intermediate_data_path.is_dir():
                logging.error(f"Path is not a directory: {self.intermediate_data_path}")
                return False

            # Count items before deletion
            items_to_delete = list(self.intermediate_data_path.iterdir())
            item_count = len(items_to_delete)

            if item_count == 0:
                logging.info(
                    f"Intermediate folder is already empty: {self.intermediate_data_path}"
                )
                return True

            # Delete all contents
            deleted_count = 0
            for item in items_to_delete:
                try:
                    if item.is_file():
                        item.unlink()  # Delete file
                        deleted_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)  # Delete directory and contents
                        deleted_count += 1
                except Exception as e:
                    if force:
                        logging.warning(f"Failed to delete {item}, continuing: {e}")
                        continue
                    else:
                        logging.error(f"Failed to delete {item}: {e}")
                        return False

            logging.info(
                f"Successfully cleaned up {deleted_count}/{item_count} items from {folder_path}"
            )
            return True

        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            return False

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
        product_classification="",
    ):
        save_dir = self.path_mapping[data_folder]
        save_dir.mkdir(exist_ok=True)
        if product_classification == "":
            product_classification = self.product_classification
        if save_dir.name == "final":
            save_dir = save_dir / product_classification

        save_path = save_dir / f"{table_name}.parquet"
        df.to_parquet(save_path, index=False)

    def compare_files(
        self, skip=["classification", "services_bilateral", "services_unilateral"]
    ):
        """
        Compares two Parquet files for exact data match using pyarrow.

        Returns:
            dict (string: bool): key is file name, bool True if the file is a match, False otherwise.
        """
        comparison = {}

        for folder in [x[0] for x in os.walk(self.final_output_path)][1:]:
            if folder not in skip:
                for file in glob.glob(os.path.join(folder, "*.parquet")):
                    file_name = file.split("/")[-1]
                    df1 = pq.read_table(
                        os.path.join(self.prod_output_path, folder, file_name)
                    )
                    df2 = pq.read_table(
                        os.path.join(self.final_output_path, folder, file_name)
                    )
                    comparison[file_name] = df1.equals(df2)
        return comparison
