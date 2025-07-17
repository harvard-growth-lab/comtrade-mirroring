import pandas as pd
from src.objects.base import AtlasCleaning
import numpy as np
from sklearn.decomposition import PCA
import copy
from typing import Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TradeDataReconciler(AtlasCleaning):
    MAX_ITERATIONS = 25
    APPLY_STANDARDIZATION = False
    APPLY_LOG_TRANSFORMATION = False
    # weight exporter_reliability and importer_reliability in single measure
    USE_PCA_COMBINATION = False
    PERCENTILES_TO_CALC = [0.10, 0.25, 0.50, 0.75, 0.90]
    MIN_TRADE_VAL = 10**3

    def __init__(self, year, ccy, **kwargs):
        super().__init__(**kwargs)

        self.year = year
        # constant dollars with atlas year as base year
        self.ccy = ccy
        self.df = pd.DataFrame()

    def reconcile_country_country_estimates(self) -> pd.DataFrame:
        ccy_reliability = self.compute_reliability_scores()
        reliability_percentiles = self.calculate_reliability_percentiles(
            ccy_reliability
        )
        self.calculate_weights(reliability_percentiles)
        self.calculate_estimated_value(reliability_percentiles)
        self.finalize_output()
        return self.df

    def compute_reliability_scores(self) -> pd.DataFrame:
        """
        This function converges to stable reliability scores that reflect
        each country's reliability in trade reporting relative to its partners.

        The reliability scores are based on the consistency of trade reporting between
        countries and the number of trade partners each country has. Countries
        with more consistent reporting and more trade partners tend to receive higher
        reliability scores.

        Algorithm Overview:
            1. Build discrepancy matrices from bilateral trade data
            2. Initialize reliability scores to 1.0
            3. Iteratively refine scores using a probabilistic approach
            4. Apply optional transformations (log, normalization)
            5. Combine exporter/importer scores using averaging or PCA
            6. Return consolidated reliability metrics
        """
        self.ncountries = self.ccy["exporter"].nunique()

        # if both countries did not report trade with each other than reporting_discrep is zero
        exporter_discrepancy_matrix = self.ccy.pivot(
            index="exporter", columns="importer", values="reporting_discrepancy"
        ).fillna(0)
        importer_discrepancy_matrix = self.ccy.pivot(
            index="importer", columns="exporter", values="reporting_discrepancy"
        ).fillna(0)

        exporter_trade_partner_count = self.ccy.groupby("exporter")[
            "exporter_nflows"
        ].first()
        nflows_imp = self.ccy.groupby("importer")["importer_nflows"].first()

        iso_index = exporter_trade_partner_count.index

        # initialize reliability score to one
        exporter_reliability = pd.DataFrame(index=iso_index)
        exporter_reliability["exporter_reliability"] = 1
        importer_reliability = pd.DataFrame(index=iso_index)
        importer_reliability["importer_reliability"] = 1

        for i in range(0, self.MAX_ITERATIONS):
            # @ is element-wise multiplication
            exporter_probability = 1 / (
                (
                    importer_discrepancy_matrix
                    @ importer_reliability["importer_reliability"]
                )
                / exporter_trade_partner_count
            )
            importer_probability = 1 / (
                (
                    exporter_discrepancy_matrix
                    @ exporter_reliability["exporter_reliability"]
                )
                / nflows_imp
            )

            importer_reliability["importer_reliability"] = importer_probability
            exporter_reliability["exporter_reliability"] = exporter_probability

        exporter_discrepancy_matrix = (
            exporter_discrepancy_matrix.sum(axis=1) / self.ncountries
        )
        importer_discrepancy_matrix = (
            importer_discrepancy_matrix.sum(axis=1) / self.ncountries
        )

        exporter_reliability = self.apply_transformations(exporter_reliability)
        importer_reliability = self.apply_transformations(importer_reliability)
        reliability_score = self.combine_scores(
            exporter_reliability, importer_reliability
        )

        return self.construct_result_df(
            exporter_trade_partner_count,
            nflows_imp,
            exporter_discrepancy_matrix,
            importer_discrepancy_matrix,
            exporter_reliability,
            importer_reliability,
            reliability_score,
        )

    def apply_transformations(self, reliability_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log and standardization transformations if enabled
        """
        if self.APPLY_LOG_TRANSFORMATION:
            reliability_df = np.ln(reliability_df)

        if self.APPLY_STANDARDIZATION:
            reliability_df = (
                reliability_df - reliability_df.mean()
            ) / reliability_df.std()
        return reliability_df

    def combine_scores(
        self, exporter_reliability: pd.DataFrame, importer_reliability: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine exporter and importer scores using averaging or PCA.
        """
        if not self.USE_PCA_COMBINATION:
            reliability_score = (
                exporter_reliability["exporter_reliability"]
                + importer_reliability["importer_reliability"]
            ) / 2

        elif self.USE_PCA_COMBINATION:
            reliability_score = PCA().fit_transform(
                exporter_reliability, importer_reliability
            )

        if self.APPLY_STANDARDIZATION:
            reliability_score = (
                reliability_score - reliability_score.mean()
            ) / reliability_score.std()
        return reliability_score

    def construct_result_df(
        self,
        exporter_trade_partner_count: pd.Series,
        nflows_imp: pd.Series,
        exporter_discrepancy_matrix: pd.Series,
        importer_discrepancy_matrix: pd.Series,
        exporter_reliability: pd.DataFrame,
        importer_reliability: pd.DataFrame,
        reliability_score: pd.Series,
    ) -> pd.DataFrame:
        """
        Construct the final result dataframe with all metrics.
        """
        importer_discrepancy_matrix = importer_discrepancy_matrix.rename(
            "importer_discrepancy_matrix"
        )
        exporter_discrepancy_matrix = exporter_discrepancy_matrix.rename(
            "exporter_discrepancy_matrix"
        )
        reliability_score = reliability_score.rename("acc_final")
        # rename indices to concat columns
        ccy_reliability = pd.concat(
            [
                exporter_trade_partner_count,
                nflows_imp.rename(index={"importer": "exporter"}),
                exporter_discrepancy_matrix,
                importer_discrepancy_matrix.rename(index={"importer": "exporter"}),
                exporter_reliability["exporter_reliability"],
                importer_reliability.rename(index={"importer": "exporter"})[
                    "importer_reliability"
                ],
                reliability_score,
            ],
            axis=1,
        )

        ccy_reliability["year"] = self.year
        ccy_reliability = ccy_reliability.reset_index().rename(columns={"index": "iso"})
        return ccy_reliability

    def calculate_reliability_percentiles(
        self, ccy_reliability: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate reliability percentiles for exporters and importers by
        merging nominal trade data with reliability scores and computing
        statistical percentiles.

        Returns:
            - exporter_reliability_percentiles
            - importer_reliability_percentiles
        """
        nominal_dollars_df = self.load_parquet(
            "intermediate",
            f"{self.product_classification}_{self.year}_ccy_nominal_dollars",
        )
        self.combine_ccy_with_reliability_scores(nominal_dollars_df, ccy_reliability)

        # unique entity tags to avoid double-counting in percentile calculations
        for entity in ["exporter", "importer"]:
            self.df[f"tag_{entity[0]}"] = (~self.df[entity].duplicated()).astype(int)

        # remove trade values less than 1000, fob
        self.df.loc[
            self.df["import_value_fob"] < self.MIN_TRADE_VAL, "import_value_fob"
        ] = np.nan
        self.df.loc[
            self.df["export_value_fob"] < self.MIN_TRADE_VAL, "export_value_fob"
        ] = np.nan

        # calculating percentiles grouped by unique exporter and then importer
        exporter_reliability_percentiles = (
            self.df[self.df.tag_e == 1]["exporter_reliability_score"]
            .quantile(self.PERCENTILES_TO_CALC)
            .round(3)
        )
        importer_reliability_percentiles = (
            self.df[self.df.tag_i == 1]["importer_reliability_score"]
            .quantile(self.PERCENTILES_TO_CALC)
            .round(3)
        )

        columns_to_cast = [
            "exporter_reliability_score",
            "acc_imp_for_exporter",
            "acc_exp_for_importer",
            "importer_reliability_score",
        ]
        for col in columns_to_cast:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        return pd.concat(
            [exporter_reliability_percentiles, importer_reliability_percentiles], axis=1
        )

    def calculate_weights(self, reliability_percentiles: pd.DataFrame) -> None:
        """
        Calculate softmax weights and binary weight flag based on reliability scores.

         - 'exporter_weight': Binary flag (1 if above threshold, 0 otherwise)
        - 'importer_weight': Binary flag (1 if above threshold, 0 otherwise)

        """
        # gives higher weight to exporters with better reliability relative to importers
        self.df["weight"] = np.exp(self.df["exporter_reliability_score"]) / (
            np.exp(self.df["exporter_reliability_score"])
            + np.exp(self.df["importer_reliability_score"])
        )

        exporter_threshold = reliability_percentiles.loc[0.10][
            "exporter_reliability_score"
        ]
        importer_threshold = reliability_percentiles.loc[0.10][
            "importer_reliability_score"
        ]

        # binary inclusion flags for countries above reliability thresholds
        self.df["exporter_weight"] = (
            self.df["exporter_reliability_score"].notna()
            & (self.df["exporter_reliability_score"] > exporter_threshold)
        ).astype(int)

        self.df["importer_weight"] = (
            self.df["importer_reliability_score"].notna()
            & (self.df["importer_reliability_score"] > importer_threshold)
        ).astype(int)

    def combine_ccy_with_reliability_scores(
        self, ccy: pd.DataFrame, ccy_reliability: pd.DataFrame
    ) -> None:
        """
        Merges trade data (ccy) with reliability data (ccy_reliability) for
        both exporters and importers
        """
        self.df = ccy.merge(
            ccy_reliability[
                ["iso", "exporter_reliability", "importer_reliability"]
            ].rename(
                columns={
                    "exporter_reliability": "exporter_reliability_score",
                    "importer_reliability": "acc_imp_for_exporter",
                }
            ),
            left_on=["exporter"],
            right_on=["iso"],
            how="left",
        ).drop(columns=["iso"])

        self.df = self.df.merge(
            ccy_reliability[
                ["iso", "exporter_reliability", "importer_reliability"]
            ].rename(
                columns={
                    "exporter_reliability": "acc_exp_for_importer",
                    "importer_reliability": "importer_reliability_score",
                }
            ),
            left_on=["importer"],
            right_on=["iso"],
            how="left",
            suffixes=("", "_for_importer"),
        ).drop(columns=["iso"])
        self.df = self.df[self.df.importer != self.df.exporter]

    def calculate_estimated_value(self, reliability_percentiles: pd.DataFrame) -> None:
        """
        Calculate estimated trade values using a hierarchical approach based on
        reliability scores.

        The estimation follows this priority order:
            1. Weighted average of import/export values (when both reliability scores and values exist)
            2. Single value selection based on relative reliability percentiles
            3. Maximum value when both countries meet weight criteria
            4. Import value when only importer meets weight criteria
            5. Export value when only exporter meets weight criteria
            6. Minimum of available values as fallback
        """
        self.df["est_trade_value"] = np.nan

        percentiles = self.extract_percentile_thresholds(reliability_percentiles)
        self.apply_weighted_average()
        self.apply_percentile_based_selection(percentiles)
        self.apply_weight_based_selection()
        self.apply_minimum_fallback()

    def extract_percentile_thresholds(
        self, reliability_percentiles: pd.DataFrame
    ) -> dict:
        """Extract percentile thresholds into a dictionary for ease of use."""
        return {
            "exporter_90": reliability_percentiles.loc[
                0.90, "exporter_reliability_score"
            ],
            "exporter_75": reliability_percentiles.loc[
                0.75, "exporter_reliability_score"
            ],
            "exporter_50": reliability_percentiles.loc[
                0.50, "exporter_reliability_score"
            ],
            "exporter_25": reliability_percentiles.loc[
                0.25, "exporter_reliability_score"
            ],
            "importer_90": reliability_percentiles.loc[
                0.90, "importer_reliability_score"
            ],
            "importer_75": reliability_percentiles.loc[
                0.75, "importer_reliability_score"
            ],
            "importer_50": reliability_percentiles.loc[
                0.50, "importer_reliability_score"
            ],
            "importer_25": reliability_percentiles.loc[
                0.25, "importer_reliability_score"
            ],
        }

    def apply_weighted_average(self) -> None:
        """Apply weighted average when both reliability scores and trade values are available."""
        mask = (
            self.df["importer_reliability_score"].notna()
            & self.df["exporter_reliability_score"].notna()
            & self.df["import_value_fob"].notna()
            & self.df["export_value_fob"].notna()
        )

        self.df.loc[mask, "est_trade_value"] = self.df.loc[
            mask, "export_value_fob"
        ] * self.df.loc[mask, "weight"] + self.df.loc[mask, "import_value_fob"] * (
            1 - self.df.loc[mask, "weight"]
        )

    def apply_percentile_based_selection(self, percentiles: dict) -> None:
        """Apply single value selection based on reliability percentile comparisons."""
        # Base condition: reliability scores available but no estimated value yet
        base_mask = (
            self.df["importer_reliability_score"].notna()
            & self.df["exporter_reliability_score"].notna()
            & self.df["est_trade_value"].isna()
        )

        # Define selection rules: (condition, value_column)
        selection_rules = [
            # High importer reliability, low exporter reliability
            (
                (self.df["exporter_reliability_score"] < percentiles["exporter_50"])
                & (self.df["importer_reliability_score"] >= percentiles["importer_90"]),
                "import_value_fob",
            ),
            # High exporter reliability, low importer reliability
            (
                (self.df["exporter_reliability_score"] >= percentiles["exporter_90"])
                & (self.df["importer_reliability_score"] < percentiles["importer_50"]),
                "export_value_fob",
            ),
            # Very high importer reliability, very low exporter reliability
            (
                (self.df["exporter_reliability_score"] < percentiles["exporter_25"])
                & (self.df["importer_reliability_score"] >= percentiles["importer_75"]),
                "import_value_fob",
            ),
            # Very high exporter reliability, very low importer reliability
            (
                (self.df["exporter_reliability_score"] >= percentiles["exporter_75"])
                & (self.df["importer_reliability_score"] < percentiles["importer_25"]),
                "export_value_fob",
            ),
        ]

        # Apply each rule in order
        for condition, value_column in selection_rules:
            mask = base_mask & condition & self.df["est_trade_value"].isna()
            self.df.loc[mask, "est_trade_value"] = self.df.loc[mask, value_column]

    def apply_weight_based_selection(self) -> None:
        """Apply selection based on weight criteria."""
        # Both countries meet weight criteria - use maximum value
        mask = (
            self.df["est_trade_value"].isna()
            & (self.df["exporter_weight"] == 1)
            & (self.df["importer_weight"] == 1)
        )
        self.df.loc[mask, "est_trade_value"] = self.df.loc[
            mask, ["export_value_fob", "import_value_fob"]
        ].max(axis=1)

        # Only importer meets weight criteria
        mask = self.df["est_trade_value"].isna() & (self.df["importer_weight"] == 1)
        self.df.loc[mask, "est_trade_value"] = self.df.loc[mask, "import_value_fob"]

        # Only exporter meets weight criteria
        mask = self.df["est_trade_value"].isna() & (self.df["exporter_weight"] == 1)
        self.df.loc[mask, "est_trade_value"] = self.df.loc[mask, "export_value_fob"]

    def apply_minimum_fallback(self) -> None:
        """Apply minimum value as final fallback for remaining missing values."""
        mask = self.df["est_trade_value"].isna()
        self.df.loc[mask, "est_trade_value"] = self.df.loc[
            mask, ["import_value_fob", "export_value_fob"]
        ].min(axis=1)

    def finalize_output(self) -> None:
        """
        Rename columns and return dataframe with mirrored trade values
        """
        self.df = self.df.rename(
            columns={
                "export_value_fob": "export_value",
                "import_value_fob": "import_value",
                "est_trade_value": "final_trade_value",
            }
        )

        # Select and reorder columns
        columns_to_keep = [
            # "year",
            "exporter",
            "importer",
            "export_value",
            "import_value",
            "final_trade_value",
            "cif_ratio",
            "weight",
            "exporter_weight",
            "importer_weight",
            "exporter_reliability_score",
            "importer_reliability_score",
        ]
        self.df = self.df[columns_to_keep]
