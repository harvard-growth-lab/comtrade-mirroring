import pandas as pd
import os
import sys


class DataFormatter:
    def __init__(self, DataPreparerConfig: object):
        self.config = DataPreparerConfig

    def order_columns(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df[cols]

    def set_datatypes(self, df: pd.DataFrame, data_types: dict) -> pd.DataFrame:
        return df.astype(data_types)


class IDtoHumnaReadableMapper:
    PRODUCT_COLS = ["product_id", "name_short_en", "code"]
    COUNTRY_COLS = ["country_id", "iso3_code"]

    def __init__(self, DataPreparerConfig: object):
        self.config = DataPreparerConfig

    def map_product_id_to_product_code(
        self, df: pd.DataFrame, vintage: str
    ) -> pd.DataFrame:
        mapped_products = pd.read_csv(
            self.atlas_common_data_path
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
            self.atlas_common_data_path
            / "classification"
            / "location"
            / f"country.csv",
            usecols=self.COUNTRY_COLS,
        )
        df = df.merge(mapped_countries, on="country_id", how="left")
        df = df.drop(columns=["country_id"])
        return df.rename(columns={"iso3_code": "country_iso"})
