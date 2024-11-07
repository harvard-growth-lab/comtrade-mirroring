"the parent object for Atlas Cleaning script"

import os
from os import path
import pandas as pd
import typing
import glob
import pyarrow.parquet as pq


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
        prod_output_path,
        atlas_common_path,
        product_classification,
    ):
        # INPUTS
        self.product_classification = product_classification

        self.downloaded_files_path = downloaded_files_path
        self.root_dir = root_dir
        self.data_path = os.path.join(self.root_dir, "data")
        self.final_output_path = os.path.join(final_output_path)
        self.prod_output_path = os.path.join(prod_output_path)
        self.atlas_common_path = os.path.join(atlas_common_path)
        self.raw_data_path = os.path.join(self.data_path, "raw")
        self.intermediate_data_path = os.path.join(self.data_path, "intermediate")
        self.processed_data_path = os.path.join(self.data_path, "processed")

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

    def load_parquet(self, data_folder, table_name: str):
        read_dir = os.path.join(self.root_dir, "data", data_folder)
        df = pd.read_parquet(os.path.join(read_dir, f"{table_name}.parquet"))
        return df

    def save_parquet(self, df, data_folder, table_name: str):
        if data_folder == "final":
            save_dir = os.path.join(
                self.final_output_path, f"{self.product_classification}"
            )
        else:
            save_dir = os.path.join(self.data_path, data_folder)
        save_path = os.path.join(save_dir, f"{table_name}.parquet")
        df.to_parquet(save_path, index=False)

    def compare_files(self, skip=['classification', 'services_bilateral', 'services_unilateral']):
        """
        Compares two Parquet files for exact data match using pyarrow.

        Returns:
            dict (string: bool): key is file name, bool True if the file is a match, False otherwise.
        """
        comparison = {}

        for folder in [x[0] for x in os.walk(self.final_output_path)][1:]:
            if folder not in skip:
                for file in glob.glob(os.path.join(folder, "*.parquet")):
                    file_name = file.split('/')[-1]
                    df1 = pq.read_table(os.path.join(self.prod_output_path, folder, file_name))
                    df2 = pq.read_table(os.path.join(self.final_output_path, folder, file_name))
                    comparison[file_name] = df1.equals(df2)
        return comparison
                                      
            
