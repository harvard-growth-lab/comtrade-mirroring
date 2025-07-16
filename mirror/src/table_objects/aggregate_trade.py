import sys
import numpy as np
import os
import pandas as pd
from sys import argv
import logging
import numpy as np

from src.objects.base import AtlasCleaning
from src.utils.handle_iso_codes_recoding import (
    handle_ans_and_other_asia_to_taiwan_recoding,
    standardize_historical_country_codes,
)

logging.basicConfig(level=logging.INFO)


class AggregateTrade(AtlasCleaning):
    COLUMNS_DICT_COMPACTOR = {
        "digitLevel": "product_level",
        "flowCode": "trade_flow",
        "reporterISO3": "reporter_iso",
        "partnerISO3": "partner_iso",
        "cmdCode": "commodity_code",
        "primaryValue": "trade_value",
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

    def run_aggregate_trade(self) -> None:
        self.df = self.load_downloaded_trade_file()
        self.df, self.ans_partners = handle_ans_and_other_asia_to_taiwan_recoding(
            self.df, self.ans_partners
        )
        self.enforce_commodity_code_length()

        self.filter_data()

        if self.product_class in ["S1", "S2"]:
            self.product_class = "SITC"
        self.save_parquet(
            self.df, "intermediate", f"{self.product_class}_{self.year}_preprocessed"
        )

        self.df = self.df[self.df["trade_flow"].isin([1, 2])]
        self.flag_unspecified_products()

        self.df = standardize_historical_country_codes(self.df)

        # returns bilateral data
        df_0 = self.create_bilateral_trade_matrix(0)
        df_4 = self.create_bilateral_trade_matrix(4)
        self.df = df_0.merge(df_4, on=["importer", "exporter"], how="outer")

        self.integrate_world_totals()
        self.adjust_trade_values_for_data_quality()

        self.df["year"] = self.year
        self.df = self.df[
            ["year", "exporter", "importer", "export_value_fob", "import_value_cif"]
        ]
        self.save_parquet(
            self.df, "intermediate", f"{self.product_class}_{self.year}_aggregated"
        )
        del self.df

    def load_downloaded_trade_file(self) -> pd.DataFrame:
        """
        loads year of aggregated trade data downloaded from Comtrade's API
        in a given product classification vintage
        """
        df = pd.DataFrame()
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
            raise ValueError(
                f"Data for classification class {self.product_class}-{self.year} not available. Nothing to aggregate"
            )

        df = df.dropna(axis=0, how="all")
        return df.rename(columns=columns)

    def filter_data(self) -> None:
        """
        Updates data in place. Filters trade data by trade flows and product detail hierarchy level based on product
        classification

        Trade Flow Mapping:
        Maps string codes to integer values:
            - "M" (Import) → 1
            - "X" (Export) → 2
            - "RM" (Re-import) → 3
            - "RX" (Re-export) → 4

        """
        if self.product_class == "SITC":
            try:
                self.df["product_level"] = self.df["product_level"].astype(int)
            except (ValueError, TypeError) as e:
                logger.error("failed to cast SITC product level as type int")

        self.df = self.df[
            self.df["product_level"].isin(self.HIERARCHY_LEVELS[self.product_class])
        ]
        trade_flow_mapping = {"M": 1, "X": 2, "RM": 3, "RX": 4}

        self.df["trade_flow"] = self.df["trade_flow"].astype(str)
        self.df.loc[:, "trade_flow"] = self.df["trade_flow"].map(trade_flow_mapping)
        self.df["trade_flow"] = self.df["trade_flow"].astype("int8")

    def enforce_commodity_code_length(self) -> None:
        """
        Enforces product level integer and commodity code length alignment
        """
        codes = self.df.loc[:, "commodity_code"].astype(str)
        levels = self.df.loc[:, "product_level"].astype(int)

        padded_codes = [
            code.zfill(0 if code == "TOTAL" else level)
            for code, level in zip(codes, levels)
        ]
        self.df.loc[:, "commodity_code"] = padded_codes

    def flag_unspecified_products(self) -> None:
        """
        Insert new trade value column handling all Area Not Specified trade partners with
        an unspecified product.

        In this case there isn't information about the trade partner
        or the product being traded
        """
        mask = (
            (self.df["partner_iso"] == "ANS")
            & (self.df["product_level"] == 4)
            & (
                self.df["commodity_code"].str[:4]
                == self.unspecified_by_class[self.product_class]
            )
        )
        self.df.loc[mask, "reporter_ansnoclas"] = self.df["trade_value"]

    def create_bilateral_trade_matrix(self, product_level: int) -> pd.DataFrame:
        """
        Aggregate trade data into bilateral trade flows at specified product level.

        Creates a bilateral trade matrix where each row represents a unique
        importer-exporter pair with both reported imports and exports. This allows
        for comparison of mirror statistics (same trade flow reported by both
        trading partners).

        Trade flow codes:
        - Flow 1: Imports (reporter is importer)
        - Flow 2: Exports (reporter is exporter)

        Input(s):
            product_level : int Product classification level to aggregate (e.g., 2, 4, 6 for HS codes)
        """
        df = self.df[self.df.product_level == product_level]
        df = (
            df.groupby(["trade_flow", "product_level", "reporter_iso", "partner_iso"])
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
        df = df.pivot_table(
            index=["reporter_iso", "partner_iso"],
            columns="trade_flow",
            values=["trade_value", "reporter_ansnoclas"],
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
                "trade_value_2": f"exports_{product_level}",
                "reporter_ansnoclas_2": f"exp2ansnoclas_{product_level}",
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
                "trade_value_1": f"imports_{product_level}",
                "reporter_ansnoclas_1": f"imp2ansnoclas_{product_level}",
            }
        )
        assert not reporting_importer.duplicated(
            subset=["importer", "exporter"]
        ).any(), "reporting exporter is not a unique pair"
        assert not reporting_exporter.duplicated(
            subset=["importer", "exporter"]
        ).any(), "reporting importer is not a unique pair"

        # outer so we don't lose World (WLD)
        df = reporting_importer.merge(
            reporting_exporter, on=["importer", "exporter"], how="outer"
        )
        return df[
            ~(
                (df[f"imports_{product_level}"] == 0)
                & (df[f"imp2ansnoclas_{product_level}"] == 0)
                & (df[f"exports_{product_level}"] == 0)
                & (df[f"exp2ansnoclas_{product_level}"] == 0)
            )
        ]

    def integrate_world_totals(self) -> None:
        """

        Add world-level trade totals to bilateral trade data and remove world aggregates.

        This method processes the dataset to:
        1. Remove rows where either partner is "WLD" (world aggregate)
        2. Extract total exports by country (from exporter-to-world records)
        3. Extract total imports by country (from importer-from-world records)
        4. Merge these totals back into the bilateral trade dataset

        The resulting dataset contains bilateral trade flows with each
        country's total trade volumes.
        """
        self.df = self.df[((self.df.importer != "WLD") & (self.df.exporter != "WLD"))]

        exp_to_world = self.df[self.df.importer == "WLD"][
            ["exporter", "exports_0"]
        ].rename(columns={"exports_0": "total_exports"})

        assert exp_to_world["exporter"].is_unique
        self.df = self.df.merge(exp_to_world, on="exporter", how="left")

        imp_to_world = self.df[self.df.exporter == "WLD"][
            ["importer", "imports_0"]
        ].rename(columns={"imports_0": "total_imports"})

        assert imp_to_world["importer"].is_unique
        self.df = self.df.merge(imp_to_world, on="importer", how="left")

    def adjust_trade_values_for_data_quality(self) -> None:
        """
        The adjustments address common issues in international trade data where
        countries report significant trade with unspecified partner areas, which
        can inflate bilateral trade statistics and create misleading patterns.

        This method performs several data quality adjustments:
            1. Calculates the ratio of "area not specified" (ANS) trade to total trade
            2. Adjusts trade values downward when ANS ratios exceed 25% threshold
            3. Computes final trade values as mean of product levels 0 and 4
            4. Filters out bilateral pairs with negligible trade volumes
        """
        self.df["ratio_exp"] = (
            self.df[f"exp2ansnoclas_4"] / self.df["total_exports"]
        ).astype(float)
        trade_flows = ["imports", "exports"]

        self.df["ratio_imp"] = (
            self.df[f"imp2ansnoclas_4"] / self.df["total_imports"]
        ).astype(float)

        for direction in ["imports", "exports"]:
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
        # drops trade under $1,000
        self.df[self.df[["export_value_fob", "import_value_cif"]].max(axis=1) >= 1_000]
