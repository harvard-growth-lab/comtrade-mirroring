import sys
from os import path
import pandas as pd
from sys import argv
import logging

logging.basicConfig(level=logging.INFO)

from clean.objects.base import _AtlasCleaning


class DataLoader(_AtlasCleaning):
    columns = [
        "Year",
        "Aggregate Level",
        "Trade Flow Code",
        "Reporter",
        "Reporter ISO",
        "Partner",
        "Partner ISO",
        "Commodity Code",
        "Qty Unit Code",
        "Qty",
        "Trade Value (US$)",
    ]
    rename_cols = {
        "Year": "year",
        "Aggregate Level": "product_level",
        "Trade Flow Code": "trade_flow",
        "Reporter": "reporter",
        "Partner": "partner",
        "Reporter ISO": "reporter_iso",
        "Partner ISO": "partner_iso",
        "Commodity Code": "commodity_code",
        "Trade Value (US$)": "trade_value",
        "Qty Unit Code": "qty_unit_code",
        "Qty": "qty",
    }

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)

        self.df = pd.DataFrame()
        self.year = year
        self.df = self.get_comtrade(year)

        
    def get_comtrade(self, year):
        """
        outputs a dataframe for one year of Comtrade data from Comtrade Downloader script
        """
        df = pd.read_csv(
            path.join(self.raw_data_path, f"{self.product_classification}_{year}.zip"),
            compression="zip",
            usecols=self.columns,
            dtype={
                "Year": int,
                "Aggregate Level": int,
                "Trade Flow Code": int,
                "Reporter": str,
                "Reporter ISO": str,
                "Partner": str,
                "Partner ISO": str,
                "Commodity Code": str,
                "Qty Unit Code": int,
                "Qty": float,
                "Trade Value (US$)": int,
            },
        )
        df = df.rename(columns=self.rename_cols)

        df = df[df["product_level"].isin([0, 2, 3, 4, 6])]
        df = df[df["trade_flow"].isin([1, 2])]

        # TODO: confirm dtypes are string
        # cols_to_fix = ['reporter' , 'partner' ,  'reporter_iso' ,  'partner_iso', 'commoditycode']
        # for c in cols_to_fix:
        #     df.loc[:,c] = df[c].apply(str)

        df.loc[df["reporter"] == "Other Asia, nes", "reporter_iso"] = "TWN"
        df.loc[df["partner"] == "Other Asia, nes", "partner_iso"] = "TWN"
        df.loc[df["reporter_iso"] == "nan", "reporter_iso"] = "ANS"
        df.loc[df["partner_iso"] == "nan", "partner_iso"] = "ANS"
        return df.drop(["reporter", "partner"], axis=1)
