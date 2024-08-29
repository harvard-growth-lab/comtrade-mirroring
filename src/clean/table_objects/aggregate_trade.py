import sys
import numpy as np
import os
import pandas as pd
from sys import argv
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from clean.table_objects.base import _AtlasCleaning


class AggregateTrade(_AtlasCleaning):
    COLUMNS_DICT_COMPACTOR = {
        "period": "year",
        "digitLevel": "product_level",
        "flowCode": "trade_flow",
        "reporterISO3": "reporter_iso",
        "partnerISO3": "partner_iso",
        "cmdCode": "commodity_code",
        "primaryValue": "trade_value",
        "qty": "qty",
    }
    COLUMNS_DICT = {
        "Year": "year",
        "Aggregate Level": "product_level",
        "Trade Flow Code": "trade_flow",
        "Reporter": "reporter",
        "Partner": "partner",
        "Reporter ISO": "reporter_iso",
        "Partner ISO": "partner_iso",
        "Commodity Code": "commodity_code",
        "Trade Value (US$)": "trade_value",
        "Qty": "qty",
    }

    def __init__(self, year, product_class, **kwargs):
        super().__init__(**kwargs)

        # initialize object variables
        self.year = year
        self.unspecified_by_class = {
            "HS": "9999",
            "H0": "9999",
            "H4": "9999",
            "H5": "9999",
            "SITC": "9310",
            "S1": "9310",
            "S2": "9310",
            "ST": "9310",
        }
        self.product_class = product_class
        # load data
        self.df = self.load_comtrade_downloader_file()
        # conditional incase df is empty
        # moved to compactor
        self.ans_and_recode_other_asia_to_taiwan()
        self.check_commodity_code_length()

        logging.info(f"Size of raw comtrade dataframe {self.df.shape}")
        # filter and clean data
        self.filter_data()

        self.save_parquet(
            self.df, "intermediate", f"cleaned_{self.product_class}_{self.year}"
        )

        self.df = self.df[self.df["trade_flow"].isin([1, 2])]
        self.label_unspecified_products()

        logging.info(f"Size after unspecified products dataframe {self.df.shape}")
        self.handle_germany_reunification()

        # returns bilateral data
        df_0 = self.aggregate_data(0)
        df_4 = self.aggregate_data(4)

        self.df = df_0.merge(df_4, on=["importer", "exporter"], how="outer")

        # Process and integrate world trade data into the main dataset.
        self.aggregate_to_world_level()

        self.remove_outliers()

        self.df["year"] = self.year
        self.df = self.df[
            ["year", "exporter", "importer", "export_value_fob", "import_value_cif"]
        ]

        self.save_parquet(
            self.df, "intermediate", f"aggregated_{self.product_class}_{self.year}"
        )

    def load_comtrade_downloader_file(self):
        """
        outputs a dataframe for one year of Comtrade data from Comtrade Downloader script
        """
        df = pd.DataFrame()
        try:
            columns = self.COLUMNS_DICT
            df = pd.read_csv(
                os.path.join(
                    self.raw_data_path, f"{self.product_class}_{self.year}_FAIL.zip"
                ),
                usecols=self.COLUMNS_DICT.keys(),
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
            logging.info("using original csv file not from compactor")
            df.loc[df["Reporter"] == "Other Asia, nes", "Reporter ISO"] = "TWN"
            df.loc[df["Partner"] == "Other Asia, nes", "Partner ISO"] = "TWN"
            df = df.drop(columns=["Reporter", "Partner"])

        except FileNotFoundError:
            try:
                columns = self.COLUMNS_DICT_COMPACTOR
                df = pd.read_parquet(
                    os.path.join(
                        self.downloaded_files_path,
                        self.product_class,
                        f"{self.product_class}_{self.year}.parquet",
                    ),
                    columns=self.COLUMNS_DICT_COMPACTOR.keys(),
                )
            except:
                logging.error(f"{self.year} not stored as a parquet file")
                try:
                    columns = self.COLUMNS_DICT_COMPACTOR
                    df = pd.read_stata(
                        os.path.join(
                            self.downloaded_files_path,
                            self.product_class,
                            f"{self.product_class}_{self.year}.dta",
                        ),
                        columns=self.COLUMNS_DICT_COMPACTOR.keys(),
                    )
                except:
                    error_message = f"Data for classification class {self.product_class}-{self.year} not available. Nothing to aggregate"
                    # raise ValueError(error_message)
        df = df.dropna(axis=0, how="all")
        return df.rename(columns=columns)

    def filter_data(self):
        if self.product_class == "SITC":
            try:
                self.df["product_level"] = self.df["product_level"].astype(int)
            except:
                logging.info("failed to cast SITC product level as type int")
        self.df = self.df[
            self.df["product_level"].isin(self.HIERARCHY_LEVELS[self.product_class])
        ]
        # TODO how do I handle reimports and reexports (SEBA question)
        self.df.loc[:, "trade_flow"] = self.df["trade_flow"].replace(
            {"M": 1, "X": 2, "RM": 3, "RX": 4}
        )
        try:
            self.df["trade_flow"] = self.df["trade_flow"].astype(str).astype(int)
        except:
            print("unexpected unique trade_flow and not mapped to an integer")

    def ans_and_recode_other_asia_to_taiwan(self):
        """
        Accounts for Taiwan and Areas Not Specified

        //following along with comtrade reads.py
        """
        try:
            self.df.loc[self.df["reporter_iso"] == "S19", "reporter_iso"] = "TWN"
        except:
            logging.info("TWN did not report as S19")
        try:
            self.df.loc[self.df["partner_iso"] == "S19", "partner_iso"] = "TWN"
        except:
            logging.info("Countries did not report Taiwan as a partner")

        # from comtrade reads. py
        ans_partners = self.ans_partners["PartnerCodeIsoAlpha3"].tolist()
        self.df.loc[self.df["partner_iso"].isin(ans_partners), "partner_iso"] = "ANS"
        self.df.loc[self.df["partner_iso"].isna(), "partner_iso"] = "ANS"

    def check_commodity_code_length(self):
        mask = (
            self.df["commodity_code"].astype(str).str.len() < self.df["product_level"]
        )
        if not self.df[mask].empty:
            self.df["commodity_code"] = self.df.apply(
                lambda row: str(row["commodity_code"]).zfill(int(row["product_level"])),
                axis=1,
            )

        for level in self.HIERARCHY_LEVELS[self.product_class]:
            self.df[self.df.product_level == level]["commodity_code"] = self.df[
                self.df.product_level == level
            ]["commodity_code"].str.zfill(level)

    def label_unspecified_products(self):
        mask = (
            (self.df["partner_iso"] == "ANS")
            & (self.df["product_level"] == 4)
            & (
                self.df["commodity_code"].str[:4]
                == self.unspecified_by_class[self.product_class]
            )
        )
        self.df.loc[mask, "reporter_ansnoclas"] = self.df.loc[mask, "trade_value"]
        # self.df["reporter_ansnoclas"] = self.df["reporter_ansnoclas"]#.fillna(0)

    def handle_germany_reunification(self):
        # drop DEU/DDR trade because within country trade
        self.df = self.df[
            ~((self.df["reporter_iso"] == "DEU") & (self.df["partner_iso"] == "DDR"))
        ]
        self.df = self.df[
            ~((self.df["reporter_iso"] == "DDR") & (self.df["partner_iso"] == "DEU"))
        ]
        # DEU is current germany iso code
        self.df.loc[self.df["partner_iso"].isin(["DEU", "DDR"]), "partner_iso"] = "DEU"
        self.df.loc[
            self.df["reporter_iso"].isin(["DEU", "DDR"]), "reporter_iso"
        ] = "DEU"
        # set USSR to Russia
        self.df.loc[self.df["partner_iso"].isin(["RUS", "SUN"]), "partner_iso"] = "RUS"
        self.df.loc[
            self.df["reporter_iso"].isin(["RUS", "SUN"]), "reporter_iso"
        ] = "RUS"
        # south africa union (ZA1) is south africa (ZAF)
        self.df.loc[
            self.df["reporter_iso"].isin(["ZA1"]), "reporter_iso"
        ] = "ZAF"
        self.df.loc[
            self.df["partner_iso"].isin(["ZA1"]), "partner_iso"
        ] = "ZAF"


    def aggregate_data(self, level):
        """
        Aggregate trade data for product levels 0 and 4, creating a merged dataset of import and export values.

        Combines data from both product levels into a single DataFrame.
        """
        df = self.df[self.df.product_level == level]
        df = (
            df.groupby(
                ["year", "trade_flow", "product_level", "reporter_iso", "partner_iso"]
            )
            .agg({"trade_value": "sum", "reporter_ansnoclas": "sum"})
            .reset_index()
        )

        df = df[
            [
                "reporter_iso",
                "partner_iso",
                "trade_value",
                "reporter_ansnoclas",
                "trade_flow",
            ]
        ]

        # generates one obs per unique pair of reporter and partner for both
        # trade_value and reporter_ansnoclas
        df = df.pivot_table(
            index=["reporter_iso", "partner_iso"],
            columns="trade_flow",
            values=["trade_value", "reporter_ansnoclas"],
            # fill_value=0,
        ).reset_index()

        df.columns = [
            "_".join(str(i) for i in col).rstrip("_") if col[1] else col[0]
            for col in df.columns.values
        ]

        # table for reporter who is an exporter
        reporting_exporter = df[
            [
                "reporter_iso",
                "partner_iso",
                "trade_value_2",
                "reporter_ansnoclas_2",
            ]
        ].rename(
            columns={
                "reporter_iso": "exporter",
                "partner_iso": "importer",
                "trade_value_2": f"exports_{level}",
                "reporter_ansnoclas_2": f"exp2ansnoclas_{level}",
            }
        )

        # table for reporter who is an importer
        reporting_importer = df[
            [
                "reporter_iso",
                "partner_iso",
                "trade_value_1",
                "reporter_ansnoclas_1",
            ]
        ].rename(
            columns={
                "reporter_iso": "importer",
                "partner_iso": "exporter",
                "trade_value_1": f"imports_{level}",
                "reporter_ansnoclas_1": f"imp2ansnoclas_{level}",
            }
        )
        assert not reporting_importer.duplicated(
            subset=["importer", "exporter"]
        ).any(), "reporting exporter is not a unique pair"
        assert not reporting_exporter.duplicated(
            subset=["importer", "exporter"]
        ).any(), "reporting importer is not a unique pair"
        # need outer so we don't lose WLD

        df = reporting_importer.merge(
            reporting_exporter, on=["importer", "exporter"], how="outer"
        )

        # drops data where all rows are zero
        df = df[
            ~(
                (df[f"imports_{level}"] == 0)
                & (df[f"imp2ansnoclas_{level}"] == 0)
                & (df[f"exports_{level}"] == 0)
                & (df[f"exp2ansnoclas_{level}"] == 0)
            )
        ]
        return df

    def aggregate_to_world_level(self):
        """
        Process and integrate world trade data into the main dataset.

        Raises:
        AssertionError: If there are duplicate exporters or importers in the world trade data.
        """
        exp_to_world = self.df[self.df.importer == "WLD"][
            ["exporter", "exports_0"]
        ].rename(columns={"exports_0": "total_exports"})
        imp_to_world = self.df[self.df.exporter == "WLD"][
            ["importer", "imports_0"]
        ].rename(columns={"imports_0": "total_imports"})

        self.df = self.df[((self.df.importer != "WLD") & (self.df.exporter != "WLD"))]
        assert exp_to_world["exporter"].is_unique
        self.df = self.df.merge(exp_to_world, on="exporter", how="left")
        assert imp_to_world["importer"].is_unique
        self.df = self.df.merge(imp_to_world, on="importer", how="left")

    def remove_outliers(self):
        """
        Adjusts export and import values for countries claiming unrealistically high trade volumes.

        Calculates export (FOB) and import (CIF) values as the mean of 0-digit and 4-digit product levels

        Removes entries with negligible trade values.
        """
        self.df["ratio_exp"] = (
            (self.df[f"exp2ansnoclas_4"] / self.df["total_exports"]).astype(float)
            # .fillna(0.0)
        )
        self.df["ratio_imp"] = (
            (self.df[f"imp2ansnoclas_4"] / self.df["total_imports"]).astype(float)
            # .fillna(0.0)
        )

        for direction in ["exports", "imports"]:
            for product_level in [0, 4]:
                # subtract if areas not specified is greater than 25% of country trade to world
                self.df[f"{direction}_{product_level}"] = np.where(
                    self.df[f"ratio_{direction[:3]}"] > 0.25,
                    self.df[f"{direction}_{product_level}"]
                    - self.df[f"{direction[:3]}2ansnoclas_4"],
                    self.df[f"{direction}_{product_level}"],
                )
        self.df["export_value_fob"] = self.df[["exports_0", f"exports_4"]].mean(axis=1)
        self.df["import_value_cif"] = self.df[["imports_0", f"imports_4"]].mean(axis=1)
        # evaluate removing this, filtering out less than $1,000
        self.df[self.df[["export_value_fob", "import_value_cif"]].max(axis=1) >= 1_000]
