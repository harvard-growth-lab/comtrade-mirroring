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

        df = self.get_comtrade()
        if df.empty:
            logging.info(
                f"Data for classification class {self.product_classification} not available. Nothing to aggregate"
            )
            return None
        df = self.clean_data(df)
        import pdb
        pdb.set_trace()
        if self.product_classification in ["H0", "HS"]:
            self.detail_level = 4
            reporter_ansnoclas = tradevalue if partner_iso == "ANS" and commoditycode[:4] == "9999" else None# TODO generate ANS value
        elif self.product_classification in ["S1", "S0", "ST"]:
            self.detail_level = 4
            # TODO generate ANS value
        else:
            logging.info("incorrect classification")
        df = self.aggregate_data(df)

        exp_to_world, imp_to_world = self.to_world(df)
        df = df[(df["importer"] != "WLD") & (df["exporter"] != "WLD")]
        assert exp_to_world["exporter"].is_unique
        df = df.merge(exp_to_world, on="exporter", how="left")
        assert imp_to_world["importer"].is_unique
        df = df.merge(imp_to_world, on="importer", how="left")
        self.filter_data(df)
        df["year"] = self.year
        df = df[["year", "exporter", "importer", "exportvalue_fob", "importvalue_cif"]]
        # TODO format fob and cif
        # df[['exportvalue_fob', 'importvalue_cif']] = df[['exportvalue_fob',
        # 'importvalue_cif']].apply(lambda x: '{:15.0f}'.format(x) if pd.notnull(x) else '')
        df = df.sort_values(by=["exporter", "importer"])
        self.save_parquet(
            df, "intermediate", f"{self.year}_{self.product_classification}"
        )

    def get_comtrade(self):
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
        df[["reporter_iso", "partner_iso"]] = df[
            ["reporter_iso", "partner_iso"]
        ].fillna(value="ANS")
        df = df.drop(["reporter", "partner"], axis=1)
        return df
        # self.save_parquet(df, "intermediate", f"{self.year}_{self.product_classification}")

    def clean_data(self, df):
        """ """
        logging.info(
            f"Cleaning.. > {self.year} and classification = {self.product_classification}"
        )

        # Data manipulation based on 'classification'
        # This is a simplification, adapt based on actual logic and conditions in Stata script
        if self.product_classification in ["H0", "HS"]:
            # Add "00" prefix to 'commoditycode' where the length of 'commoditycode' is 4 and 'product_level' is 6
            df["commodity_code"] = np.where(
                (df["commodity_code"].str.len() == 4) & (df["product_level"] == 6),
                "00" + df["commodity_code"],
                df["commodity_code"],
            )
            # Add "0" prefix to 'commoditycode' where the length of 'commoditycode' is 5 and 'product_level' is 6
            df["commodity_code"] = np.where(
                (df["commodity_code"].str.len() == 5) & (df["product_level"] == 6),
                "0" + df["commodity_code"],
                df["commodity_code"],
            )
            # TODO check should match to product_level detail being filtered at?
            df["reporter_ansnoclass"] = df.trade_value.where(
                (df.partner_iso == "ANS")
                & (df.commodity_code.str.slice(0, 4) == "9999")
            )

        elif self.product_classification in ["S1", "S1", "ST"]:
            # Add "00" prefix to 'commoditycode' where the length of 'commoditycode' is 2 and 'product_level' is 4
            df["commodity_code"] = np.where(
                (df["commodity_code"].str.len() == 2) & (df["product_level"] == 4),
                "00" + df["commodity_code"],
                df["commodity_code"],
            )
            # Add "0" prefix to 'commoditycode' where the length of 'commoditycode' is 3 and 'product_level' is 4
            df["commodity_code"] = np.where(
                (df["commodity_code"].str.len() == 3) & (df["product_level"] == 4),
                "0" + df["commodity_code"],
                df["commodity_code"],
            )
            # TODO follow up if this filter for anything
            df["reporter_ansnoclass"] = df.trade_value.where(
                (df.partner_iso == "ANS") & (df.commodity_code == "9310")
            )

        # handles Germany (reunification) and Russia
        # drop if reporter and partner are DEU and DDR, trading with itself
        df = df[~((df["reporter_iso"] == "DEU") & (df["partner_iso"] == "DDR"))]
        df = df[~((df["reporter_iso"] == "DDR") & (df["partner_iso"] == "DEU"))]
        df.loc[df["partner_iso"].isin(["DEU", "DDR"]), "partner_iso"] = "DEU"
        df.loc[df["reporter_iso"].isin(["DEU", "DDR"]), "reporter_iso"] = "DEU"
        df.loc[df["partner_iso"].isin(["RUS", "SUN"]), "partner_iso"] = "RUS"
        df.loc[df["reporter_iso"].isin(["RUS", "SUN"]), "reporter_iso"] = "RUS"

        # compress
        # collapse (sum) trade_value reporter_ansnoclas , by( year trade_flow product_level reporter_iso partner_iso )
        df = (
            df.groupby(
                ["year", "trade_flow", "product_level", "reporter_iso", "partner_iso"]
            )
            .agg({"trade_value": "sum", "reporter_ansnoclass": "sum"})
            .reset_index()
        )
        # recast float trade_value reporter_ansnoclas, force
        df["trade_value"] = df["trade_value"].astype("float")
        df["reporter_ansnoclass"] = df["reporter_ansnoclass"].astype("float")
        return df

    def aggregate_data(self, df):
        """ """
        dfs = []
        for level in [0, self.detail_level]:
            df_pl = df[df.product_level == level]
            df_pl = df_pl[
                [
                    "reporter_iso",
                    "partner_iso",
                    "trade_value",
                    "reporter_ansnoclass",
                    "trade_flow",
                ]
            ]

            # trade_flow column becomes two cols indicating the trade_flow's trade_value
            df_pl = df_pl.pivot_table(
                index=["reporter_iso", "partner_iso"],
                columns="trade_flow",
                values=["trade_value", "reporter_ansnoclass"],
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
                    "reporter_ansnoclass_2",
                ]
            ]
            reporting_exporter = reporting_exporter.rename(
                columns={
                    "reporter_iso": "exporter",
                    "partner_iso": "importer",
                    "trade_value_2": f"exports_{level}",
                    "reporter_ansnoclass_2": f"exp2ansnoclass_{level}",
                }
            )

            # table for reporter who is an importer
            reporting_importer = df_pl[
                [
                    "reporter_iso",
                    "partner_iso",
                    "trade_value_1",
                    "reporter_ansnoclass_1",
                ]
            ]
            reporting_importer = reporting_importer.rename(
                columns={
                    "reporter_iso": "importer",
                    "partner_iso": "exporter",
                    "trade_value_1": f"imports_{level}",
                    # TODO: why is this called imp2 not imp1
                    "reporter_ansnoclass_1": f"imp2ansnoclass_{level}",
                }
            )
            # merge 1:1 exporter importer using `temp', nogen
            if reporting_importer.duplicated(subset=["importer", "exporter"]).any():
                print("reporting exporter is not a unique pair")
            if reporting_exporter.duplicated(subset=["importer", "exporter"]).any():
                print("reporting importer is not a unique pair")
            merged_df = reporting_importer.merge(
                reporting_exporter, on=["importer", "exporter"]
            )
            # egen temp = rowtotal( imports_six imp2ansnoclass_six exports_six exp2ansnoclass_six )
            merged_df["temp"] = merged_df[
                [
                    f"imports_{level}",
                    f"imp2ansnoclass_{level}",
                    f"exports_{level}",
                    f"exp2ansnoclass_{level}",
                ]
            ].sum(axis=1)
            # drops rows summing to zero
            merged_df = merged_df[merged_df["temp"] != 0]
            merged_df.drop(columns="temp", inplace=True)
            dfs.append(merged_df)

        df = dfs[0].merge(dfs[1], on=["importer", "exporter"])
        cols = [
            "exporter",
            "importer",
            f"exports_0",
            f"exports_{self.detail_level}",
            f"exp2ansnoclass_{self.detail_level}",
            "imports_0",
            f"imports_{self.detail_level}",
            f"imp2ansnoclass_{self.detail_level}",
        ]
        return df[cols]

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

    def filter_data(self, df):
        """ """
        df["ratio_exp"] = df[f"exp2ansnoclass_{self.detail_level}"] / df["exp2WLD"]
        df["ratio_imp"] = df[f"imp2ansnoclass_{self.detail_level}"] / df["imp2WLD"]

        df.loc[df["ratio_exp"] > 0.25, "exports_0"] = (
            df["exports_0"] - df[f"exp2ansnoclass_{self.detail_level}"]
        )
        df.loc[df["ratio_exp"] > 0.25, f"exports_{self.detail_level}"] = (
            df[f"exports_{self.detail_level}"]
            - df[f"exp2ansnoclass_{self.detail_level}"]
        )
        df.loc[df["ratio_imp"] > 0.25, "imports_0"] = (
            df["imports_0"] - df[f"exp2ansnoclass_{self.detail_level}"]
        )
        # TODO: code substracts exp2ansnoclass_six not imp2ansnoclass_six?
        df.loc[df["ratio_imp"] > 0.25, f"imports_{self.detail_level}"] = (
            df[f"imports_{self.detail_level}"]
            - df[f"exp2ansnoclass_{self.detail_level}"]
        )

        df["exportvalue_fob"] = df[["exports_0", f"exports_{self.detail_level}"]].mean(
            axis=1
        )
        df["importvalue_cif"] = df[["imports_0", f"imports_{self.detail_level}"]].mean(
            axis=1
        )
        df["temp"] = df[["imports_0", f"imports_{self.detail_level}"]].max(axis=1)
        df = df[df["temp"] != 0]
        df = df[df["temp"] > 1_000]
