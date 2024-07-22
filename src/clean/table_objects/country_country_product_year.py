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
    SPECIALIZED_COMMODITY_CODES_BY_CLASS = {
        "H0": ["XXXXXX", "999999"],
        "H4": ["XXXXXX", "999999"],
        "S1": ["XXXX", "9999"],
        "S2": ["XXXX", "9999"],
    }
    TRADE_DATA_DISCREPANCIES = 0
    NOT_SPECIFIED = 1
    # CIF_RATIO = 0.075

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)
        print(f"starting ccpy: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
        self.product_classification = kwargs["product_classification"]
        self.year = year

        # load data

        self.df = self.load_parquet(
            f"intermediate", f"cleaned_{self.product_classification}_{self.year}"
        )
        # logging.info(f"number of exporters {len(self.df.exporter.unique())}")
        # leaving filter for quick testing purposes
        # self.df = self.df[
        #     (
        #         self.df.reporter_iso.isin(
        #             ["USA", "BEL", "SAU", "IND", "CHL", "VEN", "ZWE", "ABW", "CAN"]
        #         )
        #     )
        #     & (
        #         self.df.partner_iso.isin(
        #             ["USA", "BEL", "SAU", "IND", "CHL," "VEN", "ZWE", "ABW", "CAN"]
        #         )
        #     )
        # ]
        accuracy = self.load_parquet("processed", f"accuracy_{self.year}")
        # accuracy = accuracy[
        #     accuracy.value_final >= 100_000
        # ]
        # accuracy = accuracy[
        #     (
        #         accuracy.exporter.isin(
        #             ["USA", "BEL", "SAU", "IND", "CHL", "VEN", "ZWE", "ABW", "CAN"]
        #         )
        #     )
        #     & (
        #         accuracy.importer.isin(
        #             ["USA", "BEL", "SAU", "IND", "CHL", "VEN", "ZWE", "ABW", "CAN"]
        #         )
        #     )
        # ]
        # prepare the data
        self.filter_and_clean_data()
        logging.info("check after filter for WLD, nan, NAN")

        logging.info("ccpy: filtered and cleaned data")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
        self.all_ccpy, self.npairs, self.nprod = self.setup_trade_analysis_framework(
            accuracy
        )
        logging.info("ccpy: set up trade analysis framework")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

        # calculate the value of exports for each country pair and product
        self.exports_matrix = self.generate_trade_value_matrix("exports", accuracy)
        self.imports_matrix = self.generate_trade_value_matrix("imports", accuracy)
        # merge in cif ratio
        logging.info("ccpy: genered import and export matrices")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
        self.imports_matrix = self.imports_matrix.reset_index().merge(
            accuracy[["exporter", "importer", "cif_ratio"]],
            on=["importer", "exporter"],
            how="left",
        )

        # account for cif
        cif_factor = 1 - self.imports_matrix["cif_ratio"]
        self.imports_matrix = self.imports_matrix.set_index(["importer", "exporter"])
        self.imports_matrix = self.imports_matrix.drop(columns=["cif_ratio"])
        # import pdb
        # pdb.set_trace()
        self.imports_matrix = self.imports_matrix.multiply(
            np.array(cif_factor).reshape(-1, 1), axis=0
        )
        logging.info("ccpy: accounted for cif")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

        # swap importer, exporter to exporter, importer to merge with exports matrix
        self.imports_matrix = self.imports_matrix.swaplevel()

        self.trade_score = self.assign_trade_scores()

        country_pairs_index = self.exports_matrix.index
        accuracy = accuracy[accuracy.importer != accuracy.exporter]
        accuracy = accuracy.set_index(
            ["exporter", "importer"]
        )  # .reindex(country_pairs_index)

        cc_trade_totals, self.accuracy_scores = self.assign_accuracy_scores(accuracy)
        logging.info("ccpy: assigned accuracy scores")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

        # prep matrices for trade value logic
        self.prepare_for_matrix_multiplication(accuracy)
        logging.info("ccpy: prepped for matrix multiplication")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

#         import pdb

#         pdb.set_trace()
        self.calculate_final_trade_value()
        logging.info("ccpy: calculated final trade val")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

#         import pdb

#         pdb.set_trace()

        self.reweight_final_trade_value(cc_trade_totals)
        logging.info("ccpy: reweighted")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
#         import pdb

#         pdb.set_trace()

        # final processing
        self.filter_and_handle_trade_data_discrepancies()
        logging.info("ccpy: handle trade data discrepancies")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
#         import pdb

#         pdb.set_trace()

        self.handle_not_specified()
        logging.info("ccpy: handled not specified")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
#         import pdb

#         pdb.set_trace()

        self.df["year"] = self.year
        # self.save_parquet(self.df, "processed", f"country_country_product_year_{self.year}")

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
        # drop commodity code totals
        self.df = self.df[self.df.commodity_code != "TOTAL"]
        values_to_drop = ["WLD", "NAN", "nan"]
        columns = ["reporter_iso", "partner_iso"]
        mask = ~self.df[columns].isin(values_to_drop).any(axis=1)
        self.df = self.df[mask]

    def setup_trade_analysis_framework(self, accuracy):
        """
        Prepare the data structure for trade analysis by creating indices and a comprehensive dataset
        ensuring all potential trade relationships are accounted for
        """
        # generate index on unique country pairs
        logging.info("setup trade analysis")
        # import pdb
        # pdb.set_trace()
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

    def generate_trade_value_matrix(self, trade_flow, accuracy):
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
        trade_value_df = self.df[
            [
                "trade_flow",
                "reporter_iso",
                "partner_iso",
                "commodity_code",
                "trade_value",
            ]
        ]
        if trade_flow == "exports":
            reporter, partner = "exporter", "importer"
            trade_value_df = trade_value_df[trade_value_df["trade_flow"] == 2]
        elif trade_flow == "imports":
            reporter, partner = "importer", "exporter"
            trade_value_df = trade_value_df[trade_value_df["trade_flow"] == 1]

        trade_value_df = trade_value_df.rename(
            columns={
                "reporter_iso": reporter,
                "partner_iso": partner,
                "trade_value": f"{trade_flow}_value",
            }
        ).drop(columns=["trade_flow"])
        #  all products and all country pairs
        trade_value_df = trade_value_df.set_index(
            ["exporter", "importer", "commodity_code"]
        )
        
        trade_value_df = self.all_ccpy.set_index(
            ["exporter", "importer", "commodity_code"]
        ).join(trade_value_df, how="left")
        trade_value_df = trade_value_df.reset_index()

        trade_value_df = trade_value_df.fillna(0.0)
        return trade_value_df.pivot(
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
        # function to unravel, need to remove importer==exporter
        # import pdb
        # pdb.set_trace()
        cc_trade_total = accuracy["final_trade_value"]
        exporter_weight = accuracy["exporter_weight"]
        importer_weight = accuracy["importer_weight"]

        # score of 4 if exporter and importer weight both > 0
        # score of 2 if importer weight only > 0
        # score of 1 if exporter weight only > 0
        accuracy_scores = (
            1
            * (
                (
                    1 * (exporter_weight > 0)
                    + 1 * (importer_weight > 0)
                )
                > 1
            )
            + 1 * ((exporter_weight > 0))
            + 2 * ((importer_weight > 0))
        )
        # accuracy_scores = np.ones((self.npairs, self.nprod)) * accuracy_scores
        return cc_trade_total, accuracy_scores

    def prepare_for_matrix_multiplication(self, accuracy):
        """
        Prepare data for matrix multiplication by reshaping and melting dataframes,
        merging trade data, and handling NaN values.
        """
#         import pdb

#         pdb.set_trace()
        self.weight_matrix = accuracy[["weight"]].unstack().unstack()

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

        self.import_values = (
            self.imports_matrix.T.unstack()
            .unstack()
            .set_index(["exporter", "importer", "commodity_code"])
        )
        self.export_values = (
            self.exports_matrix.T.unstack()
            .unstack()
            .set_index(["exporter", "importer", "commodity_code"])
        )

        self.trade_score = pd.melt(
            self.trade_score.reset_index(),
            id_vars=["exporter", "importer"],
            var_name="commodity_code",
            value_name="trade_score",
        )
        self.trade_score = (
            self.trade_score.T.unstack()
            .unstack()
            .set_index(["exporter", "importer", "commodity_code"])
        )

        # Replace NaN with 0.0 for numerical stability
        self.trade_score = self.trade_score.fillna(0.0)
        self.export_values = self.export_values.fillna(0.0)
        self.import_values = self.import_values.fillna(0.0)
        self.weight_matrix = self.weight_matrix.fillna(0.0)

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

#         import pdb

#         pdb.set_trace()

        final_value = (
            (
                (
                    (self.weight_matrix["weight"] * self.export_values["export_value"])
                    + (
                        (1 - self.weight_matrix["weight"])
                        * self.import_values["import_value"]
                    )
                )
                * ((self.trade_score["trade_score"] == 4) * (self.accuracy_scores == 4))
            )
            + (
                self.import_values["import_value"]
                * ((self.trade_score["trade_score"] == 2) * (self.accuracy_scores == 2))
            )
            + (
                self.import_values["import_value"]
                * ((self.trade_score["trade_score"] == 2) * (self.accuracy_scores == 4))
            )
            + (
                self.export_values["export_value"]
                * ((self.trade_score["trade_score"] == 1) * (self.accuracy_scores == 1))
            )
            + (
                self.export_values["export_value"]
                * ((self.trade_score["trade_score"] == 1) * (self.accuracy_scores == 4))
            )
            + (
                self.import_values["import_value"]
                * ((self.trade_score["trade_score"] == 4) * (self.accuracy_scores == 2))
            )
            + (
                self.export_values["export_value"]
                * ((self.trade_score["trade_score"] == 4) * (self.accuracy_scores == 1))
            )
            + (
                0.5
                * (
                    self.export_values["export_value"]
                    + self.import_values["import_value"]
                )
                * ((self.trade_score["trade_score"] == 4) * (self.accuracy_scores == 0))
            )
            + (
                self.import_values["import_value"]
                * ((self.trade_score["trade_score"] == 2) * (self.accuracy_scores == 0))
            )
            + (
                self.export_values["export_value"]
                * ((self.trade_score["trade_score"] == 1) * (self.accuracy_scores == 0))
            )
            + (
                self.import_values["import_value"]
                * ((self.trade_score["trade_score"] == 2) * (self.accuracy_scores == 1))
            )
            + (
                self.export_values["export_value"]
                * ((self.trade_score["trade_score"] == 1) * (self.accuracy_scores == 2))
            )
        )
        final_value.name = "final_value"
        self.df = (
            self.df.set_index(["importer", "exporter", "commodity_code"])
            .join(final_value)
            .reset_index()
        )

    def reweight_final_trade_value(self, trade_total):
        """
        Adjusts final trade values to reconcile discrepancies with reported total trade figures

        Reweights trade values across commodities to match reported totals while preserving relative proportions.

        Adds an 'unspecified' category to account for large unexplained differences.
        """
        by_commodity_code = self.df.pivot(columns=['commodity_code'], index=['importer', 'exporter'], values='final_value')
        cc_estimated_trade_val = by_commodity_code.sum(axis=1)
        cc_estimated_trade_val = cc_estimated_trade_val.replace(0, np.nan)

        # determine if data trade discrepancies
        case_1 = (
            (
                ((trade_total / cc_estimated_trade_val) > 1.20).astype(int)
                + ((trade_total - cc_estimated_trade_val) > 2.5 * 10**7).astype(int)
                + (trade_total > 10**8).astype(int)
            )
            == 3).astype(int)
        case_2 = (
            (
                (trade_total > 10**8).astype(int)
                + (cc_estimated_trade_val < 10**5).astype(int)
            )
                == 2).astype(int)

        xxxx = ((case_1 + case_2) > 0).astype(int)
        value_xxxx = (trade_total - cc_estimated_trade_val) * (xxxx == 1)
        value_reweight = trade_total - value_xxxx
        
                
        reweighted = by_commodity_code - by_commodity_code * ((by_commodity_code < 1000).astype(int))
        reweighted = reweighted.div(reweighted.sum(axis=1).replace(0, np.nan), axis=0)
                
        reweighted = reweighted.reset_index().set_index(['exporter','importer'])
        reweighted = reweighted.mul(value_reweight, axis=0)
        reweighted = reweighted.fillna(0)
        reweighted = reweighted.reset_index().melt(id_vars=['exporter', 'importer'], value_name = 'reweighted_value')
        
        self.df = self.df.merge(reweighted, on=['exporter', 'importer', 'commodity_code'], how='left')
        

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
            (self.df[["reweighted_value", "import_value", "export_value"]] != 0.0).any(
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

        self.df.loc[:, "not_specified"] = np.where(
            (self.df.importer == "ANS") & (self.df.commodity_code == not_specified_val),
            self.df["reweighted_value"],
            0,
        )

        self.df.loc[:, "reweighted_value"] = np.where(
            (self.df.importer == "ANS") & (self.df.commodity_code == not_specified_val),
            np.nan,
            self.df["reweighted_value"],
        )

        # stata VR translated to final_value
        # need to confirm if ANS should be filtered from earlier
        # self.df.loc[
        #     (self.df["commodity_code"] == not_specified_val)
        #     & (self.df["importer"] == "ANS"),
        #     "final_value",
        # ] = np.nan

        # TODO group by may become an issue, memory?
        grouped = self.df.groupby("exporter", as_index=False).agg(
            {"not_specified": "sum", "reweighted_value": "sum"}
        )
        grouped["not_specified_trade_ratio"] = (
            grouped["not_specified"] / grouped["reweighted_value"].replace(0, np.nan)
        )

        countries_with_too_many_ns = (
            grouped.loc[grouped["not_specified_trade_ratio"] > 1 / 3, "exporter"]
            .unique()
            .tolist()
        )

        return self.df[~self.df.exporter.isin(countries_with_too_many_ns)]
