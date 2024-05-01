import sys
import numpy as np
from os import path
import pandas as pd
from sys import argv
import logging

logging.basicConfig(level=logging.INFO)

from clean.objects.base import _AtlasCleaning


class TradeAggregator(_AtlasCleaning):
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
        "Qty Unit Code": "qty_unit_code",
        "Qty": "qty",
    }

    def __init__(self, year, product_classification, **kwargs):
        super().__init__(**kwargs)

        self.year = year
        self.product_classification = product_classification
        self.unspecified_by_class = {"HS": '9999',
                                     "H0": '9999',
                                     "S1": "9310", 
                                     "S2": "9310", 
                                     "ST": "9310"
                                    }

        df = self.load_data()
        if df.empty:
            logging.info(
                f"Data for classification class {self.product_classification} not available. Nothing to aggregate"
            )
            return None

        df = self.clean_data(df)

        # returns bilateral data
        df = self.aggregate_data(df)

        # generates exports to world and imports to world
        exp_to_world = df[df.importer == "WLD"][["exporter", "exports_0"]].rename(columns={"exports_0": "exp2WLD"})
        imp_to_world = df[df.exporter == "WLD"][["importer", "imports_0"]].rename(columns={"imports_0": "imp2WLD"})

        df = df[(df.importer != "WLD") & (df.exporter != "WLD")]
        assert exp_to_world["exporter"].is_unique
        df = df.merge(exp_to_world, on="exporter", how="left")
        assert imp_to_world["importer"].is_unique
        df = df.merge(imp_to_world, on="importer", how="left")

        df = self.remove_outliers(df)

        df["year"] = self.year
        df = df[["year", "exporter", "importer", "exportvalue_fob", "importvalue_cif"]]
        df = df.sort_values(by=["exporter", "importer"])
        self.save_parquet(
            df, "intermediate", f"{self.year}_{self.product_classification}"
        )

    def load_data(self):
        """
        outputs a dataframe for one year of Comtrade data from Comtrade Downloader script
        """
        try:
            df = pd.read_csv(
                path.join(
                    self.raw_data_path, f"{self.product_classification}_{self.year}.zip"
                ),
                compression="zip",
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
        except FileNotFoundError:
            return pd.DataFrame()
        return df.rename(columns=self.COLUMNS_DICT)

    
    def clean_data(self, df):
        """ """
        logging.info(
            f"Cleaning.. > {self.year} and classification = {self.product_classification}"
        )
        
        df = df[df["product_level"].isin(self.HIERARCHY_LEVELS[self.product_classification])]
        df = df[df["trade_flow"].isin([1, 2])]

        # recodes Other Asia to Taiwan
        df.loc[df["reporter"] == "Other Asia, nes", "reporter_iso"] = "TWN"
        df.loc[df["partner"] == "Other Asia, nes", "partner_iso"] = "TWN"
        df[["reporter_iso", "partner_iso"]] = df[["reporter_iso", "partner_iso"]].fillna(value="ANS")
        df = df.drop(["reporter", "partner"], axis=1)

        # make sure commodity_code is the correct product length
        # mask = df["commodity_code"].astype(str).str.len() < df["product_level"]
        # if not df[mask].empty:
        #     df["commodity_code"] = df.apply(
        #         lambda row: row["commodity_code"].zfill(row["product_level"]), axis=0
        #     )
        
        for level in self.HIERARCHY_LEVELS[self.product_classification]:
            df[df.product_level == level]["commodity_code"] = df[df.product_level == level]["commodity_code"].zfill(level)
        
        # labels unspecified products 
        mask = (
            (df["partner_iso"] == "ANS")
            & (df["product_level"] == 4)
            & (df["commodity_code"].str[:4] == self.unspecified_by_class[self.product_classification])
        )
        df.loc[mask, "reporter_ansnoclas"] = df.loc[mask, "trade_value"]
        df["reporter_ansnoclas"] = df["reporter_ansnoclas"].fillna(0)
        
        # handles Germany (reunification) and Russia, drop DEU/DDR trade because within country trade 
        df = df[~((df["reporter_iso"] == "DEU") & (df["partner_iso"] == "DDR"))]
        df = df[~((df["reporter_iso"] == "DDR") & (df["partner_iso"] == "DEU"))]
        # DEU is current germany iso code
        df.loc[df["partner_iso"].isin(["DEU", "DDR"]), "partner_iso"] = "DEU"
        df.loc[df["reporter_iso"].isin(["DEU", "DDR"]), "reporter_iso"] = "DEU"
        # set USSR to Russia 
        df.loc[df["partner_iso"].isin(["RUS", "SUN"]), "partner_iso"] = "RUS"
        df.loc[df["reporter_iso"].isin(["RUS", "SUN"]), "reporter_iso"] = "RUS"
        return df

    
    def aggregate_data(self, df):
        """
        extract unique pair of importer and exporter by import and export trade values
        for both product level: [0, 1]
        """
        df = (
            df.groupby(
                ["year", "trade_flow", "product_level", "reporter_iso", "partner_iso"]
            )
            .agg({"trade_value": "sum", "reporter_ansnoclas": "sum"})
            .reset_index()
        )

        df[["trade_value", "reporter_ansnoclas"]] = df[["trade_value", "reporter_ansnoclas"]].astype("float")
        
        dfs = {0: pd.DataFrame(), 4: pd.DataFrame()}
        # product level 0 and 4
        for level in dfs.keys:
            df_pl = df[df.product_level == level]
            df_pl = df_pl[
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
            df_pl = df_pl.pivot_table(
                index=["reporter_iso", "partner_iso"],
                columns="trade_flow",
                values=["trade_value", "reporter_ansnoclas"],
                fill_value=0,
            ).reset_index()

            df_pl.columns = [
                "_".join(str(i) for i in col).rstrip("_") if col[1] else col[0]
                for col in df_pl.columns.values
            ]

            # table for reporter who is an exporter
            reporting_exporter = df_pl[
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
            reporting_importer = df_pl[
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
            merged_df = reporting_importer.merge(
                reporting_exporter, on=["importer", "exporter"], how="outer"
            )

            merged_df[
                [
                    f"imports_{level}",
                    f"imp2ansnoclas_{level}",
                    f"exports_{level}",
                    f"exp2ansnoclas_{level}",
                ]
            ] = (
                merged_df[
                    [
                        f"imports_{level}",
                        f"imp2ansnoclas_{level}",
                        f"exports_{level}",
                        f"exp2ansnoclas_{level}",
                    ]
                ]
                .astype(float)
                .fillna(0.0)
            )

            # drops data where all rows are zero
            merged_df = merged_df[~((merged_df[f"imports_{level}"] == 0) & (merged_df[f"imp2ansnoclas_{level}"] == 0) & (merged_df[f"exports_{level}"] == 0) & (merged_df[f"exp2ansnoclas_{level}"] == 0))]
                                                                          
            dfs[level] = merged_df
        return dfs[0].merge(dfs[4], on=["importer", "exporter"])

    
    def remove_outliers(self, df):
        """ """
        df["ratio_exp"] = (df[f"exp2ansnoclas_4"] / df["exp2WLD"]).astype(float).fillna(0.0)
        df["ratio_imp"] = (df[f"imp2ansnoclas_4"] / df["imp2WLD"]).astype(float).fillna(0.0)

        for direction in ['exports', 'imports']:
            for product_level in [0, 4]:
                # drop any country claiming exports/imports greater than 25% of global trade
                df[f"{direction}_{product_level}"] = np.where(df[f"ratio_{direction[:3]}"] > 0.25, df[f"exports_{product_level}"] - df["exp2ansnoclas_4"], df[f"{direction}_{product_level}"])
        
        df["exportvalue_fob"] = df[["exports_0", f"exports_4"]].mean(axis=1)
        df["importvalue_cif"] = df[["imports_0", f"imports_4"]].mean(axis=1)
        return df[df[['exportvalue_fob', 'importvalue_cif']].max(axis=1) >= 1_000]
