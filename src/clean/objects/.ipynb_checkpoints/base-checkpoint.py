"the parent object for Atlas Cleaning script"

import os
from os import path
import pandas as pd
import typing


class _AtlasCleaning(object):
    # Classification names & levels
    PRODUCT_CLASSIFICATIONS = ["H0", "HS", "S1", "S2", "ST"]
    
    HIERARCHY_LEVELS = {
        "H0": (0, 2, 4, 6),
        "HS": (0, 2, 4, 6),
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
        root_dir,
        product_classification
    ):
        # INPUTS
        self.root_dir = root_dir
        self.data_path = os.path.join(self.root_dir, "data")
        self.raw_data_path = os.path.join(self.data_path, "raw")
        self.intermediate_data_path = os.path.join(self.data_path, "intermediate")
        self.processed_data_path = os.path.join(self.data_path, "processed")
        
        # data inputs
        self.dist_cepii = pd.read_stata(os.path.join(self.raw_data_path, "dist_cepii.dta"))
        self.inflation = pd.read_parquet(
            os.path.join('data', 'intermediate', "inflation_index.parquet")
        )


        self.df = None
        self.start_year = start_year
        self.end_year = end_year
        
        self.product_classification = product_classification
        
        self.wdi_path = os.path.join(self.raw_data_path, "wdi_extended.dta")

    def load_parquet(
        self,
        data_folder,
        table_name: str
    ):
        read_dir = os.path.join(self.root_dir, 'data', data_folder)
        df = pd.read_parquet(os.path.join(read_dir, f"{table_name}.parquet"))
        return df

    def save_parquet(self, df, data_folder, table_name: str):
        save_dir = os.path.join(self.data_path, data_folder)
        save_path = os.path.join(save_dir, f"{table_name}.parquet")
        df.to_parquet(save_path, index=False)
