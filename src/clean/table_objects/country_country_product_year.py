import pandas as pd
from clean.table_objects.base import _AtlasCleaning
import os
import numpy as np
from time import gmtime, strftime, localtime


import logging
import dask.dataframe as dd
import cProfile

logging.basicConfig(level=logging.INFO)


class CountryCountryProductYear(_AtlasCleaning):
    OIL = {
        "H0": "270900",
        "H4": "270900",
        "H5": "270900",
        "S1": "3230",
        "S2": "3230",
        "SITC": "3230",
    }

    SPECIALIZED_COMMODITY_CODES_BY_CLASS = {
        "H0": ["XXXXXX", "999999"],
        "H4": ["XXXXXX", "999999"],
        "H5": ["XXXXXX", "999999"],
        "S1": ["XXXX", "9999"],
        "S2": ["XXXX", "9999"],
        "SITC": ["XXXX", "9999"],
    }
    TRADE_DATA_DISCREPANCIES = 0
    NOT_SPECIFIED = 1
    # CIF_RATIO = 0.075

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)
        print(f"starting ccpy: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
        self.product_classification = kwargs["product_classification"]
        self.year = year

        if self.product_classification == "SITC" and year > 1994:
            self.df = pd.read_parquet(
                os.path.join(
                    self.final_output_path, "SITC", f"SITC_{self.year}.parquet"
                )
            )
            self.handle_venezuela()

        else:
            # load data
            self.df = self.load_parquet(
                f"intermediate", f"cleaned_{self.product_classification}_{self.year}"
            )
            self.df = self.df.rename(columns={"commodity_code": "commoditycode"})
            if self.product_classification[:1] == "H":
                self.df.loc[
                    self.df["commoditycode"].str[:4] == "9999", "commoditycode"
                ] = "999999"

            accuracy = self.load_parquet(
                "intermediate", f"{self.product_classification}_{self.year}_accuracy"
            )

            # prepare the data
            self.filter_and_clean_data()
            logging.info("check after filter for WLD, nan, NAN")

            logging.info("ccpy: filtered and cleaned data")
            all_ccpy, self.npairs, self.nprod = self.setup_trade_analysis_framework(
                accuracy
            )

            logging.info("ccpy: set up trade analysis framework")

            # calculate the value of exports for each country pair and product
            self.generate_trade_value_matrix(accuracy, all_ccpy)

            self.trade_score = self.assign_trade_scores()
            accuracy = accuracy[accuracy.importer != accuracy.exporter]
            accuracy = accuracy.set_index(["exporter", "importer"])

            cc_trade_totals = self.assign_accuracy_scores(accuracy)
            logging.info("ccpy: assigned accuracy scores")

            self.calculate_final_trade_value()
            logging.info("ccpy: calculated final trade val")

            self.reweight_final_trade_value(accuracy)
            logging.info("ccpy: reweighted")

            # final processing
            self.filter_and_handle_trade_data_discrepancies()
            logging.info("ccpy: handle trade data discrepancies")

            self.handle_not_specified()

            self.df["year"] = self.year
            self.df = self.df.rename(
                columns={
                    "reweighted_value": "value_final",
                    "export_value": "value_exporter",
                    "import_value": "value_importer",
                }
            )
            self.handle_venezuela()

            self.df[["value_final", "value_exporter", "value_importer"]] = self.df[
                ["value_final", "value_exporter", "value_importer"]
            ].fillna(0)
            self.df = self.df[
                [
                    "year",
                    "exporter",
                    "importer",
                    "commoditycode",
                    "value_final",
                    "value_exporter",
                    "value_importer",
                ]
            ]

    def filter_and_clean_data(self):
        """
        Filter trade data to include only level 6 products and remove invalid entries.

        This function:
        1. Keeps only products at level 6 for HS and 4 for SITC
        2. Removes entries with commodity code "TOTAL"
        3. Removes entries where reporter_iso or partner_iso is in ["WLD", "NAN", "nan"]

        TODO:
        - Repetitive country code cleaning done by base
        - ie: Handle Germany/Russia unification
        """
        if self.product_classification not in ["SITC"]:
            self.df = self.df[self.df.product_level == 6]
        else:
            self.df = self.df[self.df.product_level == 4]

        # drop commodity code totals, WLD, na
        self.df = self.df[self.df.commoditycode != "TOTAL"]
        self.df = self.df[
            ~((self.df.partner_iso.isna()) | (self.df.reporter_iso.isna()))
        ]
        self.df = self.df[
            ~((self.df.partner_iso == "WLD") | (self.df.reporter_iso == "WLD"))
        ]
        self.df = self.df[~(self.df.reporter_iso == self.df.partner_iso)]

    def setup_trade_analysis_framework(self, accuracy):
        """
        Prepare the data structure for trade analysis by creating indices and a comprehensive dataset
        ensuring all potential trade relationships are accounted for
        """
        # generate index on unique country pairs
        logging.info("setup trade analysis")
        accuracy["idpair"] = accuracy.groupby(["exporter", "importer"]).ngroup()
        country_pairs = accuracy[["idpair", "exporter", "importer"]]

        # generate index for each product
        self.df["idprod"] = self.df.groupby(["commoditycode"]).ngroup()
        products = self.df[["idprod", "commoditycode"]].drop_duplicates(
            subset=["idprod", "commoditycode"]
        )
        nprod = products.count().idprod

        # set this at the country pairs level and only include products traded between both?
        multi_index = pd.MultiIndex.from_product(
            [
                country_pairs["exporter"].unique(),
                country_pairs["importer"].unique(),
            ],
            names=["reporter_iso", "partner_iso"],
        )

        all_ccpy = (
            pd.DataFrame(index=multi_index)
            .query("reporter_iso != partner_iso")
            .reset_index()
        ).drop_duplicates()

        npairs = all_ccpy[["reporter_iso", "partner_iso"]].drop_duplicates().shape[0]
        return all_ccpy, npairs, nprod

    def generate_trade_value_matrix(self, accuracy, all_ccpy):
        """
        Generate a matrix of trade values for either exports or imports.

        Inputs:
            trade_flow (str): Either "exports" or "imports" to specify the direction of trade.

        Returns:
            pd.DataFrame: A pivoted DataFrame where:
                - Index: (reporter, partner) country pairs
                - Columns: commodity codes
                - Values: trade values for the specified flow
        """
        self.df = self.df[
            [
                "trade_flow",
                "reporter_iso",
                "partner_iso",
                "commoditycode",
                "trade_value",
            ]
        ]

        self.df = self.df[~(self.df.trade_value < 1_000)]

        self.df = (
            self.df.groupby(
                ["trade_flow", "reporter_iso", "partner_iso", "commoditycode"]
            )
            .agg("sum")
            .reset_index()
        )

        re = self.df[self.df.trade_flow == 2]
        ri = self.df[self.df.trade_flow == 1]

        # import pdb
        # pdb.set_trace()

        self.df = re.merge(
            ri,
            left_on=["reporter_iso", "partner_iso", "commoditycode"],
            right_on=["partner_iso", "reporter_iso", "commoditycode"],
            how="outer",
            suffixes=("_reporting_exp", "_reporting_imp"),
        )
        # import pdb
        # pdb.set_trace()

        # Handle asymmetrical imports/exports
        self.df["reporter_iso_reporting_exp"] = self.df[
            "reporter_iso_reporting_exp"
        ].combine_first(self.df["partner_iso_reporting_imp"])
        self.df["partner_iso_reporting_exp"] = self.df[
            "partner_iso_reporting_exp"
        ].combine_first(self.df["reporter_iso_reporting_imp"])

        # import pdb
        # pdb.set_trace()

        self.df = self.df.drop(
            columns=[
                "trade_flow_reporting_exp",
                "trade_flow_reporting_imp",
                "reporter_iso_reporting_imp",
                "partner_iso_reporting_imp",
            ]
        )
        self.df = self.df.rename(
            columns={
                "trade_value_reporting_exp": "export_value",
                "trade_value_reporting_imp": "import_value",
                "reporter_iso_reporting_exp": "exporter",
                "partner_iso_reporting_exp": "importer",
            }
        )

        self.df[["export_value", "import_value"]] = self.df[
            ["export_value", "import_value"]
        ].fillna(0)

        self.df = self.df.merge(
            accuracy[
                [
                    "importer",
                    "exporter",
                    "cif_ratio",
                    "exporter_weight",
                    "importer_weight",
                ]
            ],
            on=["importer", "exporter"],
            how="left",
        )
        self.df["import_value"] = self.df["import_value"] * (1 - self.df["cif_ratio"])
        self.df = self.df.drop(columns=["cif_ratio"])
        self.df = self.df.fillna(0.0)

    def assign_trade_scores(self):
        """
        at commodity bilateral level
        score of 4 if reporters provides positive imports and exports
        score of 2 if reporter only provides positive imports
        score of 1 if reporter only provides positive exports
        """
        self.df["trade_score"] = (
            1
            * (
                (1 * (self.df["export_value"] > 0) + 1 * (self.df["import_value"] > 0))
                > 1
            )
            + 1 * (self.df["export_value"] > 0)
            + 2 * (self.df["import_value"] > 0)
        )

    def assign_accuracy_scores(self, accuracy):
        """ """
        # score of 4 if exporter and importer weight both > 0
        # score of 2 if importer weight only > 0
        # score of 1 if exporter weight only > 0
        self.df["accuracy"] = (
            1
            * (
                (
                    1 * (self.df["exporter_weight"] > 0)
                    + 1 * (self.df["importer_weight"] > 0)
                )
                > 1
            )
            + 1 * ((self.df["exporter_weight"] > 0))
            + 2 * ((self.df["importer_weight"] > 0))
        )

    def calculate_final_trade_value(self):
        """
        Calculate the final trade value based on a set of conditions that consider
        the trade score, accuracy score, export values, import values, and weights.

        The final value is determined as follows:

        1. When both trade score and accuracy score are 4:
           - Use a weighted average of export and import values.

        2. For other combinations of trade and accuracy scores:
            - When trade score is 4 but accuracy score is 0, use the average of export and import values
            - When either score is 0, use the available value (import or export) based on the non-zero score
        """
        # import pdb
        # pdb.set_trace()
        self.df["final_value"] = (
            (
                (
                    (self.df["exporter_weight"] * self.df["export_value"])
                    + ((1 - self.df["importer_weight"]) * self.df["import_value"])
                )
                * ((self.df["trade_score"] == 4) * (self.df["accuracy"] == 4))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 2) * (self.df["accuracy"] == 2))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 2) * (self.df["accuracy"] == 4))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 1) * (self.df["accuracy"] == 1))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 1) * (self.df["accuracy"] == 4))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 4) * (self.df["accuracy"] == 2))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 4) * (self.df["accuracy"] == 1))
            )
            + (
                0.5
                * (self.df["export_value"] + self.df["import_value"])
                * ((self.df["trade_score"] == 4) * (self.df["accuracy"] == 0))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 2) * (self.df["accuracy"] == 0))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 1) * (self.df["accuracy"] == 0))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 2) * (self.df["accuracy"] == 1))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 1) * (self.df["accuracy"] == 2))
            )
        )
        # import pdb
        # pdb.set_trace()
        self.df = self.df.drop(columns=["trade_score", "accuracy"])

    def reweight_final_trade_value(self, accuracy):
        """
        Adjusts final trade values to reconcile discrepancies with reported total trade figures

        Reweights trade values across commodities to match reported totals while preserving relative proportions.

        Adds a 'trade data discrepancies' category to account for large unexplained differences.
        """
        ccpy_trade_total = (
            self.df.groupby(["exporter", "importer"])
            .agg({"final_value": "sum"})
            .reset_index()
            .rename(columns={"final_value": "ccpy_trade"})
            .fillna(0)
        )
        ccy_trade_total = (
            accuracy.groupby(["exporter", "importer"])
            .agg({"final_trade_value": "sum"})
            .reset_index()[["exporter", "importer", "final_trade_value"]]
            .rename(columns={"final_trade_value": "ccy_trade"})
            .fillna(0)
        )

        reweight_df = ccpy_trade_total.merge(
            ccy_trade_total, on=["exporter", "importer"], how="outer"
        ).fillna(0)
        reweight_df = reweight_df.rename(columns={"commodity_code": "commoditycode"})

        # determine if data trade discrepancies
        reweight_df["trade_discrep1"] = (
            (
                ((reweight_df["ccy_trade"] / reweight_df["ccpy_trade"]) > 1.20).astype(
                    int
                )
                + (
                    (reweight_df["ccy_trade"] - reweight_df["ccpy_trade"])
                    > (2.5 * 10**7)
                ).astype(int)
                + (reweight_df["ccy_trade"] > 10**8).astype(int)
            )
            == 3
        ).astype(int)
        reweight_df["trade_discrep2"] = (
            (
                (reweight_df["ccy_trade"] > 10**8).astype(int)
                + (reweight_df["ccpy_trade"] < 10**5).astype(int)
            )
            == 2
        ).astype(int)

        # trade data discprepancies present
        reweight_df["is_discrepancy"] = (
            (reweight_df["trade_discrep1"] + reweight_df["trade_discrep2"]) > 0
        ).astype(int)

        reweight_df["discrep_val"] = (
            reweight_df["ccy_trade"] - reweight_df["ccpy_trade"]
        ) * reweight_df["is_discrepancy"]

        reweight_df["value_less_discrep"] = (
            reweight_df["ccy_trade"] - reweight_df["discrep_val"]
        )

        self.df.loc[self.df.final_value > 1000, "reweighted_value"] = self.df[
            "final_value"
        ]
        # self.df = self.df[self.df['final_value']>1000]

        self.df["reweighted_value_ratio"] = self.df["reweighted_value"] / (
            self.df.groupby(["exporter", "importer"])["reweighted_value"].transform(
                "sum"
            )
        )

        self.df = self.df.merge(
            reweight_df[["exporter", "importer", "value_less_discrep"]],
            on=["exporter", "importer"],
            how="outer",
        )
        self.df["reweighted_value"] = (
            self.df["reweighted_value_ratio"] * self.df["value_less_discrep"]
        )

        self.df = self.df.drop(columns=["value_less_discrep", "reweighted_value_ratio"])

        reweight_df = reweight_df[["exporter", "importer", "discrep_val"]]
        discrep_vals = reweight_df[reweight_df.discrep_val > 0]
        discrep_vals["commoditycode"] = self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[
            self.product_classification
        ][self.TRADE_DATA_DISCREPANCIES]
        discrep_vals = discrep_vals.rename(columns={"discrep_val": "reweighted_value"})
        self.df = pd.concat([self.df, discrep_vals], axis=0)

    def filter_and_handle_trade_data_discrepancies(self):
        """
        Clean trade data by removing rows without meaningful values and fill missing commodity codes.

        This function:
        1. Removes rows where all of 'final_value', 'import_value', and 'export_value' are zero.
        2. Removes rows where all values are null.
        3. Fills missing commodity codes with trade data discrepancy code based on the current product classification.
        """
        # drop rows that don't have data
        self.df = self.df.dropna(
            subset=["final_value", "reweighted_value", "import_value", "export_value"],
            how="all",
        )
        self.df.loc[
            (
                self.df[
                    ["final_value", "reweighted_value", "import_value", "export_value"]
                ]
                != 0
            ).any(axis=1)
        ]

        self.df.loc[
            self.df.commoditycode.isna(), "commoditycode"
        ] = self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[self.product_classification][
            self.TRADE_DATA_DISCREPANCIES
        ]

    def handle_not_specified(self):
        """
        Handle trade data with unspecified commodity codes and filter out exporters where
        ratio of 'not specified' trade to total trade for each exporter ratio exceeds 1/3 (33.33%)
        unspecified trade.
        """

        not_specified_val = self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[
            self.product_classification
        ][self.NOT_SPECIFIED]

        mask = (self.df["importer"] == "ANS") & (
            self.df["commoditycode"] == not_specified_val
        )
        self.df.loc[mask, "not_specified"] = self.df.loc[mask, "reweighted_value"]
        self.df["reweighted_value_temp"] = self.df["reweighted_value"]
        self.df.loc[mask, "reweighted_value_temp"] = np.nan

        not_specified_df = (
            self.df[["exporter", "reweighted_value_temp", "not_specified"]]
            .groupby("exporter")
            .agg("sum")
            .reset_index()
        )

        drop_exporter = (
            not_specified_df[
                (
                    not_specified_df["not_specified"]
                    / not_specified_df["reweighted_value_temp"]
                )
                > (1 / 3)
            ]["exporter"]
            .unique()
            .tolist()
        )

        if drop_exporter:
            logging.info("dropping exporter")
            self.df[~(self.df.exporter.isin(drop_exporter))]

    def handle_venezuela(self):
        """
        Comtrade stopped patching trade data for Venezuela starting in 2020.

        As part of the cleaning the Growth Lab patches Venezuela's exports for Crude Petroleum.
        The value is calculated by determining oil production less country's oil consumption
        using the price per barrel from the https://www.energyinst.org/statistical-review
        """
        self.df = self.df[
            ~(
                (self.df.exporter == "VEN")
                & (self.df.importer == "ANS")
                & (self.df.commoditycode == self.OIL[self.product_classification])
            )
        ]

        ven_opec = pd.read_csv("data/ven_fix/venezuela_270900_exports.csv")
        ven_opec = ven_opec[ven_opec.year == self.year]
        ven_opec = ven_opec.astype({"year": "int64"})
        ven_opec["commoditycode"] = self.OIL[self.product_classification]
        if ven_opec.empty and self.year > 2019:
            raise ValueError(
                f"Need to add the export value for oil in {self.year} for Venezuela"
            )
        self.df = pd.concat([self.df, ven_opec], axis=0, ignore_index=True, sort=False)
