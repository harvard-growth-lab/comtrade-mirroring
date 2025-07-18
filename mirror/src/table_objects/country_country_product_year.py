import pandas as pd
from src.objects.base import AtlasCleaning
import os
from pathlib import Path
import numpy as np
from time import gmtime, strftime, localtime
from src.utils.country_edge_cases import handle_ven_oil, handle_sau_9999_category
from src.objects.concordance_table import ConcordanceTable
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CountryCountryProductYear(AtlasCleaning):
    """
    Applies country-level reliability scores to product-specific trade flows

    Function:
        - Loads product-level trade data
        - Merges with country reliability scores from step 2
        - Reconciles export/import discrepancies at commodity level
        - Reweights values to match country totals
    """

    EXPORT_FLOW_CODE = 2
    IMPORT_FLOW_CODE = 1
    OIL = {
        "H0": "270900",
        "H1": "270900",
        "H2": "270900",
        "H3": "270900",
        "H6": "270900",
        "H4": "270900",
        "H5": "270900",
        "S1": "3230",
        "S2": "3230",
        "SITC": "3230",
    }

    SPECIALIZED_COMMODITY_CODES_BY_CLASS = {
        "H0": ["XXXXXX", "999999"],
        "H1": ["XXXXXX", "999999"],
        "H2": ["XXXXXX", "999999"],
        "H3": ["XXXXXX", "999999"],
        "H6": ["XXXXXX", "999999"],
        "H4": ["XXXXXX", "999999"],
        "H5": ["XXXXXX", "999999"],
        "S1": ["XXXX", "9999"],
        "S2": ["XXXX", "9999"],
        "SITC": ["XXXX", "9999"],
    }
    TRADE_DATA_DISCREPANCIES = 0
    NOT_SPECIFIED = 1

    def __init__(self, year, ccy, **kwargs):
        super().__init__(**kwargs)
        print(f"starting ccpy: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
        self.product_classification = kwargs["product_classification"]
        self.year = year
        self.ccy = ccy

    def reconcile_country_country_product_estimates(self) -> pd.DataFrame:
        """ """
        self.df = self.load_parquet(
            f"intermediate", f"{self.product_classification}_{self.year}_preprocessed"
        )
        self.df = self.df.rename(columns={"commodity_code": "commoditycode"})
        if self.product_classification.startswith("H"):
            self.df.loc[self.df["commoditycode"].str[:4] == "9999", "commoditycode"] = (
                "999999"
            )

        # prepare the data
        self.filter_and_clean_data()

        logger.debug("ccpy: filtered and cleaned data")
        logger.debug("ccpy: set up trade analysis framework")

        # calculate the value of exports for each country pair and product
        self.generate_trade_value_matrix()

        self.trade_score = self.assign_trade_scores()
        self.ccy = self.ccy[self.ccy.importer != self.ccy.exporter]
        self.ccy = self.ccy.set_index(["exporter", "importer"])

        cc_trade_totals = self.assign_reliability_scores()
        logger.debug("ccpy: assigned reliability scores")

        self.calculate_final_trade_value()
        logger.debug("ccpy: calculated final trade val")

        self.reweight_final_trade_value()
        logger.debug("ccpy: reweighted")

        # final processing
        self.filter_and_handle_trade_data_discrepancies()
        logger.debug("ccpy: handle trade data discrepancies")

        self.handle_not_specified()
        if self.year == 2023:
            self.df = handle_sau_9999_category(self.df)

        self.df["year"] = self.year
        self.df = self.df.rename(
            columns={
                "reweighted_value": "value_final",
                "export_value": "value_exporter",
                "import_value": "value_importer",
            }
        )
        if self.year > 2019:
            ven_opec = handle_ven_oil(
                self.year,
                self.df,
                self.OIL[self.product_classification],
                self.static_data_path,
            )
            self.df = pd.concat(
                [self.df, ven_opec], axis=0, ignore_index=True, sort=False
            )

        self.df[["value_final", "value_exporter", "value_importer"]] = self.df[
            ["value_final", "value_exporter", "value_importer"]
        ].fillna(0)

        ccpy_final_cols = [
            "year",
            "exporter",
            "importer",
            "commoditycode",
            "value_final",
            "value_exporter",
            "value_importer",
        ]
        self.df = self.df[ccpy_final_cols]
        if self.download_type == "by_classification":
            self.handle_comtrade_converted_sitc()
        return self.df

    def filter_and_clean_data(self) -> None:
        """
        Filter trade data to include only level 6 products and remove invalid entries.

        This function:
        1. Keeps only products at level 6 for HS and 4 for SITC
        2. Removes entries with commodity code "TOTAL"
        3. Removes entries where reporter_iso or partner_iso is in ["WLD", "NAN", "nan"]
        """
        if self.product_class_system not in ["SITC"]:
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

    def generate_trade_value_matrix(self) -> None:
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

        reported_exports = self.df[self.df.trade_flow == self.EXPORT_FLOW_CODE]
        reported_imports = self.df[self.df.trade_flow == self.IMPORT_FLOW_CODE]

        self.df = reported_exports.merge(
            reported_imports,
            left_on=["reporter_iso", "partner_iso", "commoditycode"],
            right_on=["partner_iso", "reporter_iso", "commoditycode"],
            how="outer",
            suffixes=("_reporting_exp", "_reporting_imp"),
        )

        # Handle asymmetrical imports/exports
        self.df["reporter_iso_reporting_exp"] = self.df[
            "reporter_iso_reporting_exp"
        ].combine_first(self.df["partner_iso_reporting_imp"])
        self.df["partner_iso_reporting_exp"] = self.df[
            "partner_iso_reporting_exp"
        ].combine_first(self.df["reporter_iso_reporting_imp"])

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
            self.ccy[
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

    def assign_trade_scores(self) -> None:
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

    def assign_reliability_scores(self) -> None:
        """ """
        # score of 4 if exporter and importer weight both > 0
        # score of 2 if importer weight only > 0
        # score of 1 if exporter weight only > 0
        self.df["reliability"] = (
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

    def calculate_final_trade_value(self) -> None:
        """
        Calculate the final trade value based on a set of conditions that consider
        the trade score, reliability score, export values, import values, and weights.

        The final value is determined as follows:

        1. When both trade score and reliability score are 4:
           - Use a weighted average of export and import values.

        2. For other combinations of trade and reliability scores:
            - When trade score is 4 but reliability score is 0, use the average of
            export and import values
            - When either score is 0, use the available value (import or export)
            based on the non-zero score
        """
        self.df["final_value"] = (
            (
                (
                    (self.df["exporter_weight"] * self.df["export_value"])
                    + ((1 - self.df["importer_weight"]) * self.df["import_value"])
                )
                * ((self.df["trade_score"] == 4) * (self.df["reliability"] == 4))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 2) * (self.df["reliability"] == 2))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 2) * (self.df["reliability"] == 4))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 1) * (self.df["reliability"] == 1))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 1) * (self.df["reliability"] == 4))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 4) * (self.df["reliability"] == 2))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 4) * (self.df["reliability"] == 1))
            )
            + (
                0.5
                * (self.df["export_value"] + self.df["import_value"])
                * ((self.df["trade_score"] == 4) * (self.df["reliability"] == 0))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 2) * (self.df["reliability"] == 0))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 1) * (self.df["reliability"] == 0))
            )
            + (
                self.df["import_value"]
                * ((self.df["trade_score"] == 2) * (self.df["reliability"] == 1))
            )
            + (
                self.df["export_value"]
                * ((self.df["trade_score"] == 1) * (self.df["reliability"] == 2))
            )
        )
        self.df = self.df.drop(columns=["trade_score", "reliability"])

    def reweight_final_trade_value(self) -> None:
        """
        Adjusts final trade values to reconcile discrepancies with reported total trade figures

        Reweights trade values across commodities to match reported totals while preserving
        relative proportions.

        Adds a 'trade data discrepancies' category to account for large unexplained differences.
        """
        product_level_trade_total = (
            self.df.groupby(["exporter", "importer"])
            .agg({"final_value": "sum"})
            .reset_index()
            .rename(columns={"final_value": "ccpy_trade"})
            .fillna(0)
        )
        country_level_trade_total = (
            self.ccy.groupby(["exporter", "importer"])
            .agg({"final_trade_value": "sum"})
            .reset_index()[["exporter", "importer", "final_trade_value"]]
            .rename(columns={"final_trade_value": "ccy_trade"})
            .fillna(0)
        )

        reweight_df = product_level_trade_total.merge(
            country_level_trade_total, on=["exporter", "importer"], how="outer"
        ).fillna(0)
        reweight_df = reweight_df.rename(columns={"commodity_code": "commoditycode"})

        # determine if data trade discrepancies
        reweight_df["reporting_inconsistency_type_1"] = (
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
        reweight_df["reporting_inconsistency_type_2"] = (
            (
                (reweight_df["ccy_trade"] > 10**8).astype(int)
                + (reweight_df["ccpy_trade"] < 10**5).astype(int)
            )
            == 2
        ).astype(int)

        # trade data discprepancies present
        reweight_df["has_trade_data_discrepancy"] = (
            (
                reweight_df["reporting_inconsistency_type_1"]
                + reweight_df["reporting_inconsistency_type_2"]
            )
            > 0
        ).astype(int)

        reweight_df["trade_discrepancy_value"] = (
            reweight_df["ccy_trade"] - reweight_df["ccpy_trade"]
        ) * reweight_df["has_trade_data_discrepancy"]

        reweight_df["value_less_discrep"] = (
            reweight_df["ccy_trade"] - reweight_df["trade_discrepancy_value"]
        )

        self.df.loc[self.df.final_value > 1000, "reweighted_value"] = self.df[
            "final_value"
        ]

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

        reweight_df = reweight_df[["exporter", "importer", "trade_discrepancy_value"]]
        trade_discrepancy_values = reweight_df[
            reweight_df.trade_discrepancy_value > 0
        ].copy()
        trade_discrepancy_values.loc[:, "commoditycode"] = (
            self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[self.product_classification][
                self.TRADE_DATA_DISCREPANCIES
            ]
        )
        trade_discrepancy_values = trade_discrepancy_values.rename(
            columns={"trade_discrepancy_value": "reweighted_value"}
        )
        self.df = pd.concat([self.df, trade_discrepancy_values], axis=0)

    def filter_and_handle_trade_data_discrepancies(self) -> None:
        """
        Clean trade data by removing rows without meaningful values and fill missing
        commodity codes.

        This function:
        1. Removes rows where all of 'final_value', 'import_value', and 'export_value' are zero.
        2. Removes rows where all values are null.
        3. Fills missing commodity codes with trade data discrepancy code based on the
        current product classification.
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

        self.df.loc[self.df.commoditycode.isna(), "commoditycode"] = (
            self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[self.product_classification][
                self.TRADE_DATA_DISCREPANCIES
            ]
        )

    def handle_not_specified(self) -> None:
        """
        Handle trade data with unspecified commodity codes and filter out exporters
        where ratio of 'not specified' trade to total trade for each exporter ratio
        exceeds 1/3 (33.33%) unspecified trade.
        """

        not_specified_val = self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[
            self.product_classification
        ][self.NOT_SPECIFIED]

        not_specified_mask = (self.df["importer"] == "ANS") & (
            self.df["commoditycode"] == not_specified_val
        )
        self.df.loc[not_specified_mask, "not_specified"] = self.df.loc[
            not_specified_mask, "reweighted_value"
        ]
        self.df["reweighted_value_temp"] = self.df["reweighted_value"]
        self.df.loc[not_specified_mask, "reweighted_value_temp"] = np.nan

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
            logger.debug("dropping exporter")
            self.df[~(self.df.exporter.isin(drop_exporter))]

    def handle_comtrade_converted_sitc(self) -> None:
        """
        Use Comtrade Conversion table on mirror bilateral product level trade
        data and harmonize SITC to rev 2 for by_classification data
        """
        concordance_obj = ConcordanceTable(self.static_data_path)
        if (
            self.product_classification in ["SITC", "S1", "S2", "S3"]
            and self.year <= 1975
        ):
            # convert S1 to S2; only save SITC rev 2 therefore update self.df
            self.df = concordance_obj.run_conversion(self.df, "S1", "S2")

        elif self.product_classification == "H0":
            # convert H0 to SITC rev 2; will save SITC and H0 therefore new SITC df
            sitc_df = concordance_obj.run_conversion(self.df, "H0", "S2")
            self.save_parquet(sitc_df, "final", f"SITC_{self.year}", "SITC")
        else:
            return
