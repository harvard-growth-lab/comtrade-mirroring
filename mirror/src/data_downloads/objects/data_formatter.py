import pandas as pd
import os
import sys
from user_config import PATHS
from pathlib import Path
from src.data_downloads.objects.base import logger


class FileMetaData:
    def __init__(self, data_preparer_config: object):
        self.config = data_preparer_config
        self.metadata = {}

    def set_metadata(self, file_type: str, df: pd.DataFrame, vintage: str = None):
        start_year = df.year.min()
        end_year = df.year.max()
        self.metadata[file_type] = {
            "vintage": vintage,
            "start_year": start_year,
            "end_year": end_year,
            "file_name": f"{vintage}_{start_year}_{end_year}.parquet",
        }

    def get_metadata(self, file_type: str) -> dict:
        return self.metadata[file_type]


class DataFormatter:
    def __init__(self, DataPreparerConfig: object):
        self.config = DataPreparerConfig

    def rename_columns(self, df: pd.DataFrame, cols: dict) -> pd.DataFrame:
        df = df.rename(columns=cols)

        # Check if all target columns exist
        missing_cols = set(cols.values()) - set(df.columns)
        if missing_cols:
            raise KeyError(f"Columns not found after renaming: {missing_cols}")
        return df[list(cols.values())]

    def set_datatypes(self, df: pd.DataFrame, data_types: dict) -> pd.DataFrame:
        return df.astype(data_types)


class IDtoHumanReadableMapper:
    PRODUCT_COLS = ["product_id", "name_short_en", "code"]
    COUNTRY_COLS = ["country_id", "iso3_code"]

    def __init__(self, data_preparer_config: object):
        self.config = data_preparer_config

    def map_product_id_to_product_code(
        self, df: pd.DataFrame, vintage: str
    ) -> pd.DataFrame:
        mapped_products = pd.read_csv(
            self.config.common_data_path
            / "classification"
            / "product"
            / f"{vintage}.csv",
            usecols=self.PRODUCT_COLS,
        )
        df = df.merge(mapped_products, on="product_id", how="left")
        df = df.drop(columns=["product_id"])
        return df.rename(columns={"code": "product_code"})

    def map_country_id_to_country_iso(self, df: pd.DataFrame) -> pd.DataFrame:
        mapped_countries = pd.read_csv(
            self.config.common_data_path
            / "classification"
            / "location"
            / f"country.csv",
            usecols=self.COUNTRY_COLS,
        )
        df = df.merge(mapped_countries, on="country_id", how="left")
        df = df.drop(columns=["country_id"])
        return df.rename(columns={"iso3_code": "country_iso"})
