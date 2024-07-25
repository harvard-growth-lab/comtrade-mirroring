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
        self.df = self.df[
            (
                self.df.reporter_iso.isin(
                    ["USA", "BEL", "SAU", "IND", "CHL", "VEN", "ZWE", "ABW", "CAN"]
                )
            )
            & (
                self.df.partner_iso.isin(
                    ["USA", "BEL", "SAU", "IND", "CHL," "VEN", "ZWE", "ABW", "CAN"]
                )
            )
        ]
        accuracy = self.load_parquet("processed", f"accuracy_{self.year}")
        # accuracy = accuracy[
        #     accuracy.value_final >= 100_000
        # ]
        accuracy = accuracy[
            (
                accuracy.exporter.isin(
                    ["USA", "BEL", "SAU", "IND", "CHL", "VEN", "ZWE", "ABW", "CAN"]
                )
            )
            & (
                accuracy.importer.isin(
                    ["USA", "BEL", "SAU", "IND", "CHL", "VEN", "ZWE", "ABW", "CAN"]
                )
            )
        ]
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
        self.generate_trade_value_matrix(accuracy)
        # self.imports_matrix = self.generate_trade_value_matrix("imports", accuracy)
        # merge in cif ratio
        logging.info("ccpy: genered trade vals matrix with cif ratio applied")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

        # swap importer, exporter to exporter, importer to merge with exports matrix
        # self.imports_matrix = self.imports_matrix.swaplevel()

        self.trade_score = self.assign_trade_scores()

        accuracy = accuracy[accuracy.importer != accuracy.exporter]
        accuracy = accuracy.set_index(["exporter", "importer"])

        cc_trade_totals = self.assign_accuracy_scores(accuracy)
        logging.info("ccpy: assigned accuracy scores")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

        # prep matrices for trade value logic
        # self.prepare_for_matrix_multiplication(accuracy)
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

        # final processing
        self.filter_and_handle_trade_data_discrepancies()
        logging.info("ccpy: handle trade data discrepancies")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
        import pdb
        pdb.set_trace()

        self.handle_not_specified()
        logging.info("ccpy: handled not specified")
        print(f"time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
        import pdb
        pdb.set_trace()

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
            names=["reporter_iso", "partner_iso", "commodity_code"],
        )

        all_ccpy = (
            pd.DataFrame(index=multi_index)
            .query("reporter_iso != partner_iso")
            .reset_index()
        ).drop_duplicates()

        npairs = all_ccpy[["reporter_iso", "partner_iso"]].drop_duplicates().shape[0]
        return all_ccpy, npairs, nprod

    
    def generate_trade_value_matrix(self, accuracy):
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
        self.df = self.df[['trade_flow', 'reporter_iso', 'partner_iso', 'commodity_code', 'trade_value']]
        self.df = self.df.groupby(['reporter_iso', 'partner_iso', 'commodity_code', 'trade_flow']).agg('sum').reset_index()
        re = self.df[self.df.trade_flow==2]
        ri = self.df[self.df.trade_flow==1]
        
        self.df = re.merge(ri, on=['reporter_iso', 'partner_iso', 'commodity_code'], how='outer', suffixes=('_reporting_exp', '_reporting_imp'))
        self.df = self.df.drop(columns=["trade_flow_reporting_exp", "trade_flow_reporting_imp"])
        self.df = self.df.rename(columns={"trade_value_reporting_exp": "export_value", "trade_value_reporting_imp": "import_value", "reporter_iso": "exporter", "partner_iso": "importer"})
#         import pdb
#         pdb.set_trace()
#         # self.df.loc[self.df["trade_flow"] == 2, "export_value"] = self.df["trade_value"]
#         # self.df.loc[self.df["trade_flow"] == 1, "import_value"] = self.df["trade_value"]
#         # self.df = self.df.drop(columns=['trade_value'])
#         self.df = self.df.groupby(['reporter_iso', 'partner_iso', 'commodity_code']).agg('sum')
#         # self['trade_score'] = self['trade_flow']
#         self.df.loc[self.df['trade_flow'] == 3, 'trade_score'] = 4 
#         self.df.loc[self.df['trade_flow'] == 1, 'trade_score'] = 2
#         self.df.loc[self.df['trade_flow'] == 2, 'trade_score'] = 1

#         self.df = self.df.reset_index()
#         self.df = self.df.merge(self.df, left_on=['reporter_iso', 'partner_iso', 'commodity_code'], right_on=['partner_iso', 'reporter_iso', 'commodity_code'], how='left', suffixes=('_country_1', '_country_2'))
        
        self.df = self.df.merge(
            accuracy[["importer", "exporter", "cif_ratio", 'exporter_weight', 'importer_weight']],
            on=["importer", "exporter"],
            how="left",
        )
        self.df["import_value"] = self.df["import_value"] * (1 - self.df["cif_ratio"])
        self.df = self.df.drop(columns=['cif_ratio'])
        # self.df = self.df.drop(columns=["trade_value", "product_level"])

        # return (
        #     self.df.merge(
        #         self.all_ccpy,
        #         on=["reporter_iso", "partner_iso", "commodity_code"],
        #         how="right",
        #     )
        #     .fillna(0.0)
        # )

    def assign_trade_scores(self):
        """
        at commodity bilateral level
        score of 4 if reporter provides positive imports and exports
        score of 2 if reporter only provides positive imports
        score of 1 if reporter only provides positive exports
        """
        import pdb
        pdb.set_trace()
        self.df['trade_score'] = (
            1
            * (
                (
                    1 * (self.df["export_value"] > 0)
                    + 1 * (self.df["import_value"] > 0)
                )
                > 1
            )
            + 1 * (self.df["export_value"] > 0)
            + 2 * (self.df["import_value"] > 0)
        )

    def assign_accuracy_scores(self, accuracy):
        """ """
        # function to unravel, need to remove importer==exporter
        cc_trade_total = accuracy["final_trade_value"]
        exporter_weight = accuracy["exporter_weight"]
        importer_weight = accuracy["importer_weight"]

        # score of 4 if exporter and importer weight both > 0
        # score of 2 if importer weight only > 0
        # score of 1 if exporter weight only > 0
        self.df['accuracy'] = (
            1 * ((1 * (self.df['exporter_weight'] > 0) + 1 * (self.df['importer_weight'] > 0)) > 1)
            + 1 * ((self.df['exporter_weight'] > 0))
            + 2 * ((self.df['importer_weight'] > 0))
        )
        return cc_trade_total

    def prepare_for_matrix_multiplication(self, accuracy):
        """
        Prepare data for matrix multiplication by reshaping and melting dataframes,
        merging trade data, and handling NaN values.
        """
        self.trade_values_matrix = self.trade_values_matrix.reset_index()
        self.accuracy_scores.name = "accuracy"
        for metric, df in [
            ("accuracy", self.accuracy_scores.to_frame()),
            ("weight", accuracy[["weight"]]),
        ]:
            if df.index.names == ["exporter", "importer"]:
                exporter_df = df.rename(columns={metric: f"exporter_{metric}"})
                importer_df = df.swaplevel().rename(
                    columns={f"{metric}": f"importer_{metric}"}
                )

                self.trade_values_matrix = self.trade_values_matrix.merge(
                    exporter_df.reset_index().rename(
                        columns={"exporter": "reporter_iso", "importer": "partner_iso"}
                    ),
                    on=["reporter_iso", "partner_iso"],
                    how="left",
                )

                self.trade_values_matrix = self.trade_values_matrix.merge(
                    importer_df.reset_index().rename(
                        columns={"importer": "reporter_iso", "exporter": "partner_iso"}
                    ),
                    on=["reporter_iso", "partner_iso"],
                    how="left",
                )

            else:
                raise ValueError("Expected exporter, importer multiindex")

        self.trade_score = self.trade_score.rename(columns={0: "trade_score"})
        self.trade_values_matrix = self.trade_values_matrix.merge(
            self.trade_score.reset_index(),
            on=["reporter_iso", "partner_iso", "commodity_code"],
            how="left",
        )

        # Replace NaN with 0.0 for numerical stability
        self.trade_values_matrix = self.trade_values_matrix.fillna(0)
        # self.trade_values_matrix = self.trade_values_matrix.groupby('
        self.trade_values_matrix = self.trade_values_matrix.set_index(
            ["reporter_iso", "partner_iso", "commodity_code"]
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
        self.df['final_value'] = (
            (
                (
                    (
                        self.df["exporter_weight"]
                        * self.df["export_value"]
                    )
                    + (
                        (1 - self.df["importer_weight"])
                        * self.df["import_value"]
                    )
                )
                * (
                    (self.df["trade_score"] == 4)
                    * (self.df["accuracy"] == 4)
                )
            )
            + (
                self.df["import_value"]
                * (
                    (self.df["trade_score"] == 2)
                    * (self.df["accuracy"] == 2)
                )
            )
            + (
                self.df["import_value"]
                * (
                    (self.df["trade_score"] == 2)
                    * (self.df["accuracy"] == 4)
                )
            )
            + (
                self.df["export_value"]
                * (
                    (self.df["trade_score"] == 1)
                    * (self.df["accuracy"] == 1)
                )
            )
            + (
                self.df["export_value"]
                * (
                    (self.df["trade_score"] == 1)
                    * (self.df["accuracy"] == 4)
                )
            )
            + (
                self.df["import_value"]
                * (
                    (self.df["trade_score"] == 4)
                    * (self.df["accuracy"] == 2)
                )
            )
            + (
                self.df["export_value"]
                * (
                    (self.df["trade_score"] == 4)
                    * (self.df["accuracy"] == 1)
                )
            )
            + (
                0.5
                * (
                    self.df["export_value"]
                    + self.df["import_value"]
                )
                * (
                    (self.df["trade_score"] == 4)
                    * (self.df["accuracy"] == 0)
                )
            )
            + (
                self.df["import_value"]
                * (
                    (self.df["trade_score"] == 2)
                    * (self.df["accuracy"] == 0)
                )
            )
            + (
                self.df["export_value"]
                * (
                    (self.df["trade_score"] == 1)
                    * (self.df["accuracy"] == 0)
                )
            )
            + (
                self.df["import_value"]
                * (
                    (self.df["trade_score"] == 2)
                    * (self.df["accuracy"] == 1)
                )
            )
            + (
                self.df["export_value"]
                * (
                    (self.df["trade_score"] == 1)
                    * (self.df["accuracy"] == 2)
                )
            )
        )

        # clean up memory
        # try:
        #     del self.trade_score
        #     del self.accuracy_scores
        # except:
        #     logging.error("can't delete object")

        # final_value.name = "final_value"
        # self.trade_values_matrix = self.trade_values_matrix.reset_index().merge(
        #     final_value.to_frame(),
        #     on=["reporter_iso", "partner_iso", "commodity_code"],
        #     how="left",
        # )

    def reweight_final_trade_value(self, trade_total):
        """
        Adjusts final trade values to reconcile discrepancies with reported total trade figures

        Reweights trade values across commodities to match reported totals while preserving relative proportions.

        Adds an 'unspecified' category to account for large unexplained differences.
        """

        # self.trade_values_matrix = self.trade_values_matrix.rename(
        #     columns={'reporter_iso' : 'exporter',
        #               'partner_iso' : 'importer'}).set_index(['exporter', 'importer', 'commodity_code'])
            
        cc_estimated_trade_val = self.df.groupby(['exporter', 'importer']).agg('sum')['final_value']
        cc_estimated_trade_val = cc_estimated_trade_val.replace(0, np.nan)

        # determine if data trade discrepancies
        case_1 = (
            (
                ((trade_total / cc_estimated_trade_val) > 1.20).astype(int)
                + ((trade_total - cc_estimated_trade_val) > 2.5 * 10**7).astype(int)
                + (trade_total > 10**8).astype(int)
            )
            == 3
        ).astype(int)
        case_2 = (
            (
                (trade_total > 10**8).astype(int)
                + (cc_estimated_trade_val < 10**5).astype(int)
            )
            == 2
        ).astype(int)

        xxxx = ((case_1 + case_2) > 0).astype(int)
        value_xxxx = (trade_total - cc_estimated_trade_val) * (xxxx == 1)
        value_reweight = trade_total - value_xxxx

        self.df = self.df.set_index(['exporter', 'importer', 'commodity_code'])
        reweighted = self.df[['final_value']] - self.df[['final_value']] * (
            (self.df[['final_value']] < 1000).astype(int)
        )
        reweighted = reweighted.div(reweighted.sum(axis=1).replace(0, np.nan), axis=0)

        # reweighted = reweighted.reset_index().set_index(["exporter", "importer"])
        reweighted = reweighted.mul(value_reweight, axis=0)
        reweighted = reweighted.fillna(0)
        # reweighted = reweighted.reset_index().melt(
        #     id_vars=["exporter", "importer"], value_name="reweighted_value"
        # )
        reweighted = reweighted.rename(columns={'final_value': 'reweighted_value'})
        self.df = self.df.reset_index().merge(
            reweighted, on=["exporter", "importer", "commodity_code"], how="left"
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
            (self.df[["final_value", "reweighted_value", "import_value", "export_value"]] != 0.0).any(
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
        # self.df.loc[
        #     (self.df["commodity_code"] == not_specified_val)
        #     & (self.df["importer"] == "ANS"),
        #     "final_value",
        # ] = np.nan

        grouped = self.df.groupby("exporter", as_index=False).agg(
            {"not_specified": "sum", "reweighted_value": "sum"}
        )
        grouped["not_specified_trade_ratio"] = grouped["not_specified"] / grouped[
            "reweighted_value"
        ].replace(0, np.nan)

        countries_with_too_many_ns = (
            grouped.loc[grouped["not_specified_trade_ratio"] > 1 / 3, "exporter"]
            .unique()
            .tolist()
        )

        self.df = self.df[~self.df.exporter.isin(countries_with_too_many_ns)]
