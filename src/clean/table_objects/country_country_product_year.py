import pandas as pd
from clean.table_objects.base import _AtlasCleaning
import os
import numpy as np

import logging
import dask.dataframe as dd
import cProfile

logging.basicConfig(level=logging.INFO)


class CountryCountryProductYear(_AtlasCleaning):
    SPECIALIZED_COMMODITY_CODES_BY_CLASS = {
        "H0": ["XXXXXX", "999999"],
        "H4": ["XXXXXX", "999999"],
        "S1": ["XXXX", "9999"],
        "S2": ["XXXX", "9999"],
    }
    TRADE_DATA_DISCREPANCIES = 0
    NOT_SPECIFIED = 1
    CIF_RATIO = 0.075

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)

        self.product_classification = kwargs["product_classification"]
        self.year = year

        # load data
        self.df = self.load_parquet(f"raw/{self.product_classification}", f"{self.product_classification}_{self.year}")
        # leaving filter for quick testing purposes
        # self.df = self.df[
        #     (self.df.reporter_iso.isin(["SAU", "IND", "CHL", "VEN", "ZWE"]))
        #     & (self.df.partner_iso.isin(["SAU", "IND", "CHL," "VEN", "ZWE"]))
        # ]
        accuracy = self.load_parquet("processed", f"accuracy")
        # accuracy = accuracy[
        #     accuracy.value_final >= 100_000
        # ]
        # accuracy = accuracy[
        #     (accuracy.exporter.isin(["SAU", "IND", "CHL", "VEN", "ZWE"]))
        #     & (accuracy.importer.isin(["SAU", "IND", "CHL", "VEN", "ZWE"]))
        # ]

        # prepare the data
        self.filter_and_clean_data()
        self.all_ccpy, self.npairs, self.nprod = self.setup_trade_analysis_framework(
            accuracy
        )

        # calculate the value of exports for each country pair and product
        self.exports_matrix = self.generate_trade_value_matrix("exports")
        self.imports_matrix = self.generate_trade_value_matrix("imports")
        # self.imports_matrix = self.imports_matrix * (1 - self.CIF_RATIO)
        self.imports_matrix = self.imports_matrix.swaplevel().sort_index()

        self.trade_score = self.assign_trade_scores()
        cc_trade_totals, self.accuracy_scores = self.assign_accuracy_scores(accuracy)

        # prep matrices for trade value logic
        self.prepare_for_matrix_multiplication(accuracy)

        self.calculate_final_trade_value()
        self.reweight_final_trade_value(cc_trade_totals)

        # final processing
        self.filter_and_handle_trade_data_discrepancies()
        self.handle_not_specified()

        self.df["year"] = self.year

    def filter_and_clean_data(self):
        """
        Filter trade data to include only level 6 products and remove invalid entries.

        This function:
        1. Keeps only products at level 6
        2. Removes entries with commodity code "TOTAL"
        3. Removes entries where reporter_iso or partner_iso is in ["WLD", "NAN", "nan"]

        TODO:
        - Repetitive country code cleaning done by base
        - ie: Handle Germany/Russia unification
        """
        self.df = self.df[self.df.product_level == 6]

        # TODO: reivew if any initial repetitive country code cleaning
        # TODO: repeat deal with germany/russia unification

        # drop commodity code totals
        self.df = self.df[self.df.commodity_code != "TOTAL"]
        drop_values = ["WLD", "NAN", "nan"]
        self.df = self.df[
            ~self.df.apply(
                lambda row: row["reporter_iso"] in drop_values
                or row["partner_iso"] in drop_values,
                axis=1,
            )
        ]

    def setup_trade_analysis_framework(self, accuracy):
        """ 
        Prepare the data structure for trade analysis by creating indices and a comprehensive dataset
        ensuring all potential trade relationships are accounted for
        """
        # generate index on unique country pairs
        accuracy["idpair"] = accuracy.groupby(["exporter", "importer"]).ngroup()
        country_pairs = accuracy[["idpair", "exporter", "importer"]]

        # generate index for each product
        self.df["idprod"] = self.df.groupby(["commodity_code"]).ngroup()
        products = self.df[["idprod", "commodity_code"]].drop_duplicates(
            subset=["idprod", "commodity_code"]
        )
        nprod = products.count().idprod

        # set this at the country pairs level and only include products traded between both?
        multi_index = pd.MultiIndex.from_product(
            [
                country_pairs["exporter"].unique(),
                country_pairs["importer"].unique(),
                products["commodity_code"].unique(),
            ],
            names=["exporter", "importer", "commodity_code"],
        )

        all_ccpy = (
            pd.DataFrame(index=multi_index).query("importer != exporter").reset_index()
        ).drop_duplicates()

        npairs = all_ccpy[["exporter", "importer"]].drop_duplicates().shape[0]
        return all_ccpy, npairs, nprod

    def generate_trade_value_matrix(self, trade_flow):
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
        if trade_flow == "exports":
            reporter = "exporter"
            partner = "importer"
            df = self.df[self.df["trade_flow"] == 2][
                ["reporter_iso", "partner_iso", "commodity_code", "trade_value"]
            ]
        elif trade_flow == "imports":
            reporter = "importer"
            partner = "exporter"
            df = self.df[self.df["trade_flow"] == 1][
                ["reporter_iso", "partner_iso", "commodity_code", "trade_value"]
            ]

        df.columns = [reporter, partner, "commodity_code", f"{trade_flow}_value"]
        #  all products and all country pairs
        df = self.all_ccpy.merge(
            df, on=[reporter, partner, "commodity_code"], how="left"
        )
        df = df.fillna(0.0)
        return df.pivot(
            index=[reporter, partner],
            columns="commodity_code",
            values=f"{trade_flow}_value",
        )

    def assign_trade_scores(self):
        """
        at commodity bilateral level
        score of 4 if reporter provides positive imports and exports
        score of 2 if reporter only provides positive imports
        score of 1 if reporter only provides positive exports
        """
        return pd.DataFrame(
            1 * ((1 * (self.exports_matrix > 0) + 1 * (self.imports_matrix > 0)) > 1)
            + 1 * (self.exports_matrix > 0)
            + 2 * (self.imports_matrix > 0)
        )

    def assign_accuracy_scores(self, accuracy):
        """ """
        country_pairs_index = self.exports_matrix.index
        accuracy = (
            accuracy.set_index(["exporter", "importer"]).reindex(country_pairs_index)
            # .reset_index()
        )

        cc_trade_total = np.array(accuracy["final_trade_value"].values.reshape(-1, 1))

        # country pair weighted accuracy
        exporter_weight = np.array(accuracy["exporter_weight"].values.reshape(-1, 1))
        importer_weight = np.array(accuracy["importer_weight"].values.reshape(-1, 1))
        # .swaplevel().sort_index().values.reshape(-1, 1))

        # score of 4 if exporter and importer weight both > 0
        # score of 2 if importer weight only > 0
        # score of 1 if exporter weight only > 0
        accuracy_scores = (
            1 * ((1 * (exporter_weight > 0) + 1 * (importer_weight > 0)) > 1)
            + 1 * ((exporter_weight > 0))
            + 2 * ((importer_weight > 0))
        )
        accuracy_scores = np.ones((self.npairs, self.nprod)) * accuracy_scores
        return cc_trade_total, accuracy_scores

    def prepare_for_matrix_multiplication(self, accuracy):
        """
        Prepare data for matrix multiplication by reshaping and melting dataframes,
        merging trade data, and handling NaN values.
        """
        self.weight_matrix = np.array(accuracy["weight"].values.reshape(-1, 1))
        self.weight_matrix = np.ones((self.npairs, self.nprod)) * self.weight_matrix

        # melt the dataframes
        self.imports_matrix = pd.melt(
            self.imports_matrix.reset_index(),
            id_vars=["importer", "exporter"],
            var_name="commodity_code",
            value_name="import_value",
        )

        self.exports_matrix = pd.melt(
            self.exports_matrix.reset_index(),
            id_vars=["importer", "exporter"],
            var_name="commodity_code",
            value_name="export_value",
        )

        self.df = self.imports_matrix.merge(
            self.exports_matrix,
            on=["exporter", "importer", "commodity_code"],
            how="left",
        )

        self.weight_matrix = self.weight_matrix.reshape(-1, 1)
        self.export_values = self.df["export_value"].values.reshape(-1, 1)
        self.import_values = self.df["import_value"].values.reshape(-1, 1)

        self.trade_score = pd.melt(
            self.trade_score.reset_index(),
            id_vars=["exporter", "importer"],
            var_name="commodity_code",
            value_name="trade_score",
        )
        self.trade_score = self.trade_score["trade_score"].values.reshape(-1, 1)

        # Replace NaN with 0.0 for numerical stability
        for array in [
            self.trade_score,
            self.export_values,
            self.import_values,
            self.weight_matrix,
        ]:
            np.nan_to_num(array, copy=False, nan=0.0)

        self.accuracy_scores = self.accuracy_scores.reshape(-1, 1)

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
        self.df["final_value"] = (
            (
                (
                    (self.weight_matrix * self.export_values)
                    + ((1 - self.weight_matrix) * self.import_values)
                )
                * ((self.trade_score == 4) * (self.accuracy_scores == 4))
            )
            + (
                self.import_values
                * ((self.trade_score == 2) * (self.accuracy_scores == 2))
            )
            + (
                self.import_values
                * ((self.trade_score == 2) * (self.accuracy_scores == 4))
            )
            + (
                self.export_values
                * ((self.trade_score == 1) * (self.accuracy_scores == 1))
            )
            + (
                self.export_values
                * ((self.trade_score == 1) * (self.accuracy_scores == 4))
            )
            + (
                self.import_values
                * ((self.trade_score == 4) * (self.accuracy_scores == 2))
            )
            + (
                self.export_values
                * ((self.trade_score == 4) * (self.accuracy_scores == 1))
            )
            + (
                0.5
                * (self.export_values + self.import_values)
                * ((self.trade_score == 4) * (self.accuracy_scores == 0))
            )
            + (
                self.import_values
                * ((self.trade_score == 2) * (self.accuracy_scores == 0))
            )
            + (
                self.export_values
                * ((self.trade_score == 1) * (self.accuracy_scores == 0))
            )
            + (
                self.import_values
                * ((self.trade_score == 2) * (self.accuracy_scores == 1))
            )
            + (
                self.export_values
                * ((self.trade_score == 1) * (self.accuracy_scores == 2))
            )
        )

    def reweight_final_trade_value(self, trade_total):
        """
        Adjusts final trade values to reconcile discrepancies with reported total trade figures

        Reweights trade values across commodities to match reported totals while preserving relative proportions.

        Adds an 'unspecified' category to account for large unexplained differences.
        """
        cc_scored_value_est = (
            self.df.groupby(["importer", "exporter"])["final_value"]
            .sum()
            .values.reshape(-1, 1)
        )

        # determine if data trade discrepancies
        case_1 = 1 * (
            (
                np.where((trade_total / cc_scored_value_est) > 1.20, 1, 0)
                + np.where((trade_total - cc_scored_value_est) > 2.5 * 10**7, 1, 0)
                + np.where(trade_total > 10**8, 1, 0)
            )
            == 3
        )
        case_2 = 1 * (
            (
                np.where(trade_total > 10**8, 1, 0)
                + np.where(cc_scored_value_est < 10**5, 1, 0)
                == 2
            )
        )

        xxxx = 1 * ((case_1 + case_2) > 0)
        value_xxxx = (trade_total - cc_scored_value_est) * (xxxx == 1)
        value_reweight = trade_total - value_xxxx

        # proportionally reweight products for each country country pair
        self.df["final_value"] = self.df["final_value"].where(
            self.df["final_value"] >= 1000, 0
        )
        trade_value_matrix = self.df.pivot(
            index=["exporter", "importer"],
            columns="commodity_code",
            values="final_value",
        )

        partner_trade = trade_value_matrix.sum(axis=1)
        trade_value_matrix = trade_value_matrix.div(partner_trade, axis=0)

        trade_value_matrix = trade_value_matrix * value_reweight.reshape(-1, 1)
        trade_value_matrix.loc[:, "value_xxxx"] = value_xxxx

        trade_value_matrix = pd.melt(
            trade_value_matrix.reset_index(),
            id_vars=["exporter", "importer"],
            var_name="commodity_code",
            value_name="final_value",
        )

        self.df = self.df.drop(columns=["final_value"]).merge(
            trade_value_matrix,
            on=["exporter", "importer", "commodity_code"],
            how="left",
        )

    def filter_and_handle_trade_data_discrepancies(self):
        """
        Clean trade data by removing rows without meaningful values and fill missing commodity codes.

        This function:
        1. Removes rows where all of 'final_value', 'import_value', and 'export_value' are zero.
        2. Removes rows where all values are null.
        3. Fills missing commodity codes with trade data discrepancy code based on the current product classification.
        """
        # drop rows that don't have data
        self.df = self.df.loc[
            (self.df[["final_value", "import_value", "export_value"]] != 0.0).any(
                axis=1
            )
            & self.df.notnull().any(axis=1)
        ]

        self.df["commodity_code"] = self.df["commodity_code"].fillna(
            self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[self.product_classification][
                self.TRADE_DATA_DISCREPANCIES
            ]
        )

    def handle_not_specified(self):
        """
        Handle trade data with unspecified commodity codes and filter out exporters where
        atio of 'not specified' trade to total trade for each exporter ratio exceeds 1/3 (33.33%)
        unspecified trade.
        """
        not_specified_val = self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[
            self.product_classification
        ][self.NOT_SPECIFIED]

        not_specified_df = self.df.copy(deep=True)
        not_specified_df["not_specified"] = self.df.apply(
            lambda x: (
                x["final_value"]
                if x["commodity_code"] == not_specified_val and x["importer"] == "ANS"
                else 0
            ),
            axis=1,
        )

        # stata VR translated to final_value
        # need to confirm if ANS should be filtered from earlier
        not_specified_df.loc[
            (self.df["commodity_code"] == not_specified_val)
            & (self.df["importer"] == "ANS"),
            "final_value",
        ] = None

        not_specified_df = not_specified_df.groupby("exporter", as_index=False).agg(
            {"not_specified": "sum", "final_value": "sum"}
        )
        not_specified_df["not_specified_trade_ratio"] = (
            not_specified_df["not_specified"] / not_specified_df["final_value"]
        )

        countries_with_too_many_ns = (
            not_specified_df.loc[
                not_specified_df["not_specified_trade_ratio"] > 1 / 3, "exporter"
            ]
            .unique()
            .tolist()
        )

        return self.df[~self.df.exporter.isin(countries_with_too_many_ns)]
