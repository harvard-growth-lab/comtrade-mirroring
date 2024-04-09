import sys
import numpy as np
from os import path
import pandas as pd
from sys import argv
import logging

logging.basicConfig(level=logging.INFO)

from clean.objects.base import _AtlasCleaning


class TradeAggregator(_AtlasCleaning):
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

    def __init__(self, year, product_classification, **kwargs):
        super().__init__(**kwargs)

        self.year = year
        self.product_classification = product_classification

        df = self.load_data()
        if df.empty:
            logging.info(
                f"Data for classification class {self.product_classification} not available. Nothing to aggregate"
            )
            return None
        
        df = self.clean_data(df)
        
        # unique pairs for product level 0 & 4
        df = self.aggregate_data(df)

        exp_to_world, imp_to_world = self.to_world(df)

        df = df[(df["importer"] != "WLD") & (df["exporter"] != "WLD")]
        assert exp_to_world["exporter"].is_unique
        df = df.merge(exp_to_world, on="exporter", how="left")
        assert imp_to_world["importer"].is_unique
        df = df.merge(imp_to_world, on="importer", how="left")
        
        import pdb
        pdb.set_trace()

        df = self.remove_outliers(df)
        
        import pdb
        pdb.set_trace()

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
        except FileNotFoundError:
            return pd.DataFrame()
        df = df.rename(columns=self.rename_cols)

        df = df[df["product_level"].isin([0, 2, 3, 4, 6])]
        df = df[df["trade_flow"].isin([1, 2])]

        df.loc[df["reporter"] == "Other Asia, nes", "reporter_iso"] = "TWN"
        df.loc[df["partner"] == "Other Asia, nes", "partner_iso"] = "TWN"
        df[["reporter_iso", "partner_iso"]] = df[["reporter_iso", "partner_iso"]].fillna(value="ANS")
        df = df.drop(["reporter", "partner"], axis=1)
        return df

    
    def clean_data(self, df):
        """ """
        logging.info(
            f"Cleaning.. > {self.year} and classification = {self.product_classification}"
        )

        # make sure commodity_code is the correct product length
        mask = df['commodity_code'].astype(str).str.len() < df['product_level']
        if not df[mask].empty:
            df['commodity_code'] = df.apply(lambda row: row['commodity_code'].zfill(row['product_level']), axis=0)
        
        if self.product_classification in ["H0", "HS"]:
            # areas not specified
            logging.info("accounting for areas not specified")
            mask = (df['partner_iso'] == "ANS") & (df['product_level'] == 4) & (df['commodity_code'].str[:4] == "9999")
            df.loc[mask, 'reporter_ansnoclas'] = df.loc[mask, 'trade_value']
            df['reporter_ansnoclas'] = df['reporter_ansnoclas'].fillna(0)
            
            
        elif self.product_classification in ["S1", "S2", "ST"]:
            # areas not specified
            logging.info("accounting for areas not specified")
            mask = (df['partner_iso'] == "ANS") & (df['product_level'] == 4) & (df['commodity_code'].str[:4] == "9310")
            df.loc[mask, 'reporter_ansnoclas'] = df.loc[mask, 'trade_value']
            df['reporter_ansnoclas'] = df['reporter_ansnoclas'].fillna(0)
        
        # handles Germany (reunification) and Russia
        # drop if reporter and partner are DEU and DDR, trading with itself
        df = df[~((df["reporter_iso"] == "DEU") & (df["partner_iso"] == "DDR"))]
        df = df[~((df["reporter_iso"] == "DDR") & (df["partner_iso"] == "DEU"))]
        df.loc[df["partner_iso"].isin(["DEU", "DDR"]), "partner_iso"] = "DEU"
        df.loc[df["reporter_iso"].isin(["DEU", "DDR"]), "reporter_iso"] = "DEU"
        df.loc[df["partner_iso"].isin(["RUS", "SUN"]), "partner_iso"] = "RUS"
        df.loc[df["reporter_iso"].isin(["RUS", "SUN"]), "reporter_iso"] = "RUS"

        df = (
            df.groupby(
                ["year", "trade_flow", "product_level", "reporter_iso", "partner_iso"]
            )
            .agg({"trade_value": "sum", "reporter_ansnoclas": "sum"})
            .reset_index()
        )
        # recast float trade_value reporter_ansnoclas, force
        df["trade_value"] = df["trade_value"].astype("float")
        df["reporter_ansnoclas"] = df["reporter_ansnoclas"].astype("float")
        return df

    
    def aggregate_data(self, df):
        """ 
        extract unique pair of importer and exporter by import and export trade values
        for both product level: [0, 1]
        """
        dfs = []
        # product level 0 and 4
        for level in [0, 4]:
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
            ]
            reporting_exporter = reporting_exporter.rename(
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
            ]
            reporting_importer = reporting_importer.rename(
                columns={
                    "reporter_iso": "importer",
                    "partner_iso": "exporter",
                    "trade_value_1": f"imports_{level}",
                    "reporter_ansnoclas_1": f"imp2ansnoclas_{level}",
                }
            )
            assert not reporting_importer.duplicated(subset=["importer", "exporter"]).any(), "reporting exporter is not a unique pair"
            assert not reporting_exporter.duplicated(subset=["importer", "exporter"]).any(), "reporting importer is not a unique pair"
            # need outer so we don't lose WLD
            merged_df = reporting_importer.merge(
                reporting_exporter, on=["importer", "exporter"], how="outer")
            
            merged_df[[f"imports_{level}",
                       f"imp2ansnoclas_{level}",
                       f"exports_{level}",
                       f"exp2ansnoclas_{level}",
                      ]] = merged_df[[f"imports_{level}",
                       f"imp2ansnoclas_{level}",
                       f"exports_{level}",
                       f"exp2ansnoclas_{level}",
                      ]].astype(float).fillna(0.)

            # calculates rowwise sum
            merged_df["temp"] = merged_df[
                [
                    f"imports_{level}",
                    f"imp2ansnoclas_{level}",
                    f"exports_{level}",
                    f"exp2ansnoclas_{level}",
                ]
            ].sum(axis=1)

            # drops rows temp sums to zero summing to zero
            merged_df = merged_df[merged_df.temp != 0.]
            merged_df.drop(columns="temp", inplace=True)
            dfs.append(merged_df)
            
        # merge product_level zero with product_level four
        bilateral_data = dfs[0].merge(dfs[1], on=["importer", "exporter"])
        return bilateral_data

    def to_world(self, df):
        """
        zero digit values
        """
        exp_to_world = df[df.importer == "WLD"]
        exp_to_world = exp_to_world[["exporter", "exports_0"]]
        exp_to_world = exp_to_world.rename(columns={"exports_0": "exp2WLD"})
        imp_to_world = df[df.exporter == "WLD"]
        imp_to_world = imp_to_world[["importer", "imports_0"]]
        imp_to_world = imp_to_world.rename(columns={"imports_0": "imp2WLD"})
        return exp_to_world, imp_to_world

    def remove_outliers(self, df):
        """ 
        """
        df["ratio_exp"] = df[f"exp2ansnoclas_4"] / df["exp2WLD"]
        df["ratio_imp"] = df[f"imp2ansnoclas_4"] / df["imp2WLD"]
        df[['ratio_exp', 'ratio_imp']] = df[['ratio_exp', 'ratio_imp']].astype(float).fillna(0.0)
        
        # exports with ratio greater than .25
        import pdb
        pdb.set_trace()
        df["exports_0"] = np.where(df["ratio_exp"] > 0.25, 
                                   df['exports_0'] - df['exp2ansnoclas_4'], df["exports_0"])
        df["exports_4"] = np.where(df["ratio_exp"] > 0.25, 
                           df['exports_4'] - df['exp2ansnoclas_4'], df["exports_4"])

        # imports with ratio greater than .25
        df["imports_0"] = np.where(df["ratio_imp"] > 0.25, 
                           df['exports_0'] - df['exp2ansnoclas_4'], df["imports_0"])
        df["imports_4"] = np.where(df["ratio_imp"] > 0.25, 
                           df['exports_4'] - df['exp2ansnoclas_4'], df["imports_4"])

        df["exportvalue_fob"] = df[["exports_0", f"exports_4"]].mean(
            axis=1
        )
        df["importvalue_cif"] = df[["imports_0", f"imports_4"]].mean(
            axis=1
        )
        df["temp"] = df[["exportvalue_fob", f"importvalue_cif"]].max(axis=1)
        import pdb
        pdb.set_trace()
        df = df[df["temp"] != 1]
        df = df[df["temp"] >= 1_000]
        return df
