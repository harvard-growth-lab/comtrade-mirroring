import pandas as pd
from src.objects.base import AtlasCleaning
import numpy as np
from sklearn.decomposition import PCA
import copy
from typing import Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)

# logging.basicConfig(level=logging.INFO)


class TradeDataReconciler(AtlasCleaning):
    MAX_ITERATIONS = 25
    APPLY_STANDARDIZATION = False
    APPLY_LOG_TRANSFORMATION = False
    # weight exporter_accuracy and importer_accuracy in single measure
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
        ccy_accuracy = self.compute_accuracy_scores()
        accuracy_percentiles = self.calculate_accuracy_percentiles(ccy_accuracy)
        self.calculate_weights(accuracy_percentiles)
        self.calculate_estimated_value(accuracy_percentiles)
        self.finalize_output()
        return self.df

    def compute_accuracy_scores(self) -> pd.DataFrame:
        """
        This function converges to stable accuracy scores that reflect
        each country's reliability in trade reporting relative to its partners.

        The accuracy scores are based on the consistency of trade reporting between
        countries and the number of trade partners each country has. Countries
        with more consistent reporting and more trade partners tend to receive higher
        accuracy scores.

        Algorithm Overview:
            1. Build discrepancy matrices from bilateral trade data
            2. Initialize accuracy scores to 1.0
            3. Iteratively refine scores using a probabilistic approach
            4. Apply optional transformations (log, normalization)
            5. Combine exporter/importer scores using averaging or PCA
            6. Return consolidated accuracy metrics
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

        # initialize accuracy to one
        exporter_accuracy = pd.DataFrame(index=iso_index)
        exporter_accuracy["exporter_accuracy"] = 1
        importer_accuracy = pd.DataFrame(index=iso_index)
        importer_accuracy["importer_accuracy"] = 1

        for i in range(0, self.MAX_ITERATIONS):
            # @ is element-wise multiplication
            exporter_probability = 1 / (
                (importer_discrepancy_matrix @ importer_accuracy["importer_accuracy"])
                / exporter_trade_partner_count
            )
            importer_probability = 1 / (
                (exporter_discrepancy_matrix @ exporter_accuracy["exporter_accuracy"])
                / nflows_imp
            )

            importer_accuracy["importer_accuracy"] = importer_probability
            exporter_accuracy["exporter_accuracy"] = exporter_probability

        exporter_discrepancy_matrix = (
            exporter_discrepancy_matrix.sum(axis=1) / self.ncountries
        )
        importer_discrepancy_matrix = (
            importer_discrepancy_matrix.sum(axis=1) / self.ncountries
        )

        exporter_accuracy = self.apply_transformations(exporter_accuracy)
        importer_accuracy = self.apply_transformations(importer_accuracy)
        accuracy_score = self.combine_scores(exporter_accuracy, importer_accuracy)

        return self.construct_result_df(
            exporter_trade_partner_count,
            nflows_imp,
            exporter_discrepancy_matrix,
            importer_discrepancy_matrix,
            exporter_accuracy,
            importer_accuracy,
            accuracy_score,
        )

    def apply_transformations(self, accuracy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log and standardization transformations if enabled
        """
        if self.APPLY_LOG_TRANSFORMATION:
            accuracy_df = np.ln(accuracy_df)

        if self.APPLY_STANDARDIZATION:
            accuracy_df = (accuracy_df - accuracy_df.mean()) / accuracy_df.std()
        return accuracy_df

    def combine_scores(
        self, exporter_accuracy: pd.DataFrame, importer_accuracy: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine exporter and importer scores using averaging or PCA.
        """
        if not self.USE_PCA_COMBINATION:
            accuracy_score = (
                exporter_accuracy["exporter_accuracy"]
                + importer_accuracy["importer_accuracy"]
            ) / 2

        elif self.USE_PCA_COMBINATION:
            accuracy_score = PCA().fit_transform(exporter_accuracy, importer_accuracy)

        if self.APPLY_STANDARDIZATION:
            accuracy_score = (
                accuracy_score - accuracy_score.mean()
            ) / accuracy_score.std()
        return accuracy_score

    def construct_result_df(
        self,
        exporter_trade_partner_count: pd.Series,
        nflows_imp: pd.Series,
        exporter_discrepancy_matrix: pd.Series,
        importer_discrepancy_matrix: pd.Series,
        exporter_accuracy: pd.DataFrame,
        importer_accuracy: pd.DataFrame,
        accuracy_score: pd.Series,
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
        accuracy_score = accuracy_score.rename("acc_final")
        # rename indices to concat columns
        ccy_accuracy = pd.concat(
            [
                exporter_trade_partner_count,
                nflows_imp.rename(index={"importer": "exporter"}),
                exporter_discrepancy_matrix,
                importer_discrepancy_matrix.rename(index={"importer": "exporter"}),
                exporter_accuracy["exporter_accuracy"],
                importer_accuracy.rename(index={"importer": "exporter"})[
                    "importer_accuracy"
                ],
                accuracy_score,
            ],
            axis=1,
        )

        ccy_accuracy["year"] = self.year
        ccy_accuracy = ccy_accuracy.reset_index().rename(columns={"index": "iso"})
        return ccy_accuracy

    def calculate_accuracy_percentiles(
        self, ccy_accuracy: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate accuracy percentiles for exporters and importers by
        merging nominal trade data with accuracy scores and computing
        statistical percentiles.

        Returns:
            - exporter_accuracy_percentiles
            - importer_accuracy_percentiles
        """
        nominal_dollars_df = self.load_parquet(
            "intermediate",
            f"{self.product_classification}_{self.year}_ccy_nominal_dollars",
        )
        self.combine_ccy_with_accuracy_scores(nominal_dollars_df, ccy_accuracy)

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
        exporter_accuracy_percentiles = (
            self.df[self.df.tag_e == 1]["exporter_accuracy_score"]
            .quantile(self.PERCENTILES_TO_CALC)
            .round(3)
        )
        importer_accuracy_percentiles = (
            self.df[self.df.tag_i == 1]["importer_accuracy_score"]
            .quantile(self.PERCENTILES_TO_CALC)
            .round(3)
        )

        columns_to_cast = [
            "exporter_accuracy_score",
            "acc_imp_for_exporter",
            "acc_exp_for_importer",
            "importer_accuracy_score",
        ]
        for col in columns_to_cast:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        return pd.concat(
            [exporter_accuracy_percentiles, importer_accuracy_percentiles], axis=1
        )

    def calculate_weights(self, accuracy_percentiles: pd.DataFrame) -> None:
        """
        Calculate softmax weights and binary weight flag based on accuracy scores.

         - 'exporter_weight': Binary flag (1 if above threshold, 0 otherwise)
        - 'importer_weight': Binary flag (1 if above threshold, 0 otherwise)

        """
        # gives higher weight to exporters with better accuracy relative to importers
        self.df["weight"] = np.exp(self.df["exporter_accuracy_score"]) / (
            np.exp(self.df["exporter_accuracy_score"])
            + np.exp(self.df["importer_accuracy_score"])
        )

        exporter_threshold = accuracy_percentiles.loc[0.10]["exporter_accuracy_score"]
        importer_threshold = accuracy_percentiles.loc[0.10]["importer_accuracy_score"]

        # binary inclusion flags for countries above accuracy thresholds
        self.df["exporter_weight"] = (
            self.df["exporter_accuracy_score"].notna()
            & (self.df["exporter_accuracy_score"] > exporter_threshold)
        ).astype(int)

        self.df["importer_weight"] = (
            self.df["importer_accuracy_score"].notna()
            & (self.df["importer_accuracy_score"] > importer_threshold)
        ).astype(int)

    def combine_ccy_with_accuracy_scores(
        self, ccy: pd.DataFrame, ccy_accuracy: pd.DataFrame
    ) -> None:
        """
        Merges trade data (ccy) with accuracy data (ccy_accuracy) for
        both exporters and importers
        """
        self.df = ccy.merge(
            ccy_accuracy[["iso", "exporter_accuracy", "importer_accuracy"]].rename(
                columns={
                    "exporter_accuracy": "exporter_accuracy_score",
                    "importer_accuracy": "acc_imp_for_exporter",
                }
            ),
            left_on=["exporter"],
            right_on=["iso"],
            how="left",
        ).drop(columns=["iso"])

        self.df = self.df.merge(
            ccy_accuracy[["iso", "exporter_accuracy", "importer_accuracy"]].rename(
                columns={
                    "exporter_accuracy": "acc_exp_for_importer",
                    "importer_accuracy": "importer_accuracy_score",
                }
            ),
            left_on=["importer"],
            right_on=["iso"],
            how="left",
            suffixes=("", "_for_importer"),
        ).drop(columns=["iso"])
        self.df = self.df[self.df.importer != self.df.exporter]

    def calculate_estimated_value(self, accuracy_percentiles: pd.DataFrame) -> None:
        """
        Calculate estimated trade values using a hierarchical approach based on
        accuracy scores.

        The estimation follows this priority order:
            1. Weighted average of import/export values (when both accuracy scores and values exist)
            2. Single value selection based on relative accuracy percentiles
            3. Maximum value when both countries meet weight criteria
            4. Import value when only importer meets weight criteria
            5. Export value when only exporter meets weight criteria
            6. Minimum of available values as fallback
        """
        self.df["est_trade_value"] = np.nan

        percentiles = self.extract_percentile_thresholds(accuracy_percentiles)
        self.apply_weighted_average()
        self.apply_percentile_based_selection(percentiles)
        self.apply_weight_based_selection()
        self.apply_minimum_fallback()

    def extract_percentile_thresholds(self, accuracy_percentiles: pd.DataFrame) -> dict:
        """Extract percentile thresholds into a dictionary for ease of use."""
        return {
            "exporter_90": accuracy_percentiles.loc[0.90, "exporter_accuracy_score"],
            "exporter_75": accuracy_percentiles.loc[0.75, "exporter_accuracy_score"],
            "exporter_50": accuracy_percentiles.loc[0.50, "exporter_accuracy_score"],
            "exporter_25": accuracy_percentiles.loc[0.25, "exporter_accuracy_score"],
            "importer_90": accuracy_percentiles.loc[0.90, "importer_accuracy_score"],
            "importer_75": accuracy_percentiles.loc[0.75, "importer_accuracy_score"],
            "importer_50": accuracy_percentiles.loc[0.50, "importer_accuracy_score"],
            "importer_25": accuracy_percentiles.loc[0.25, "importer_accuracy_score"],
        }

    def apply_weighted_average(self) -> None:
        """Apply weighted average when both accuracy scores and trade values are available."""
        mask = (
            self.df["importer_accuracy_score"].notna()
            & self.df["exporter_accuracy_score"].notna()
            & self.df["import_value_fob"].notna()
            & self.df["export_value_fob"].notna()
        )

        self.df.loc[mask, "est_trade_value"] = self.df.loc[
            mask, "export_value_fob"
        ] * self.df.loc[mask, "weight"] + self.df.loc[mask, "import_value_fob"] * (
            1 - self.df.loc[mask, "weight"]
        )

    def apply_percentile_based_selection(self, percentiles: dict) -> None:
        """Apply single value selection based on accuracy percentile comparisons."""
        # Base condition: accuracy scores available but no estimated value yet
        base_mask = (
            self.df["importer_accuracy_score"].notna()
            & self.df["exporter_accuracy_score"].notna()
            & self.df["est_trade_value"].isna()
        )

        # Define selection rules: (condition, value_column)
        selection_rules = [
            # High importer accuracy, low exporter accuracy
            (
                (self.df["exporter_accuracy_score"] < percentiles["exporter_50"])
                & (self.df["importer_accuracy_score"] >= percentiles["importer_90"]),
                "import_value_fob",
            ),
            # High exporter accuracy, low importer accuracy
            (
                (self.df["exporter_accuracy_score"] >= percentiles["exporter_90"])
                & (self.df["importer_accuracy_score"] < percentiles["importer_50"]),
                "export_value_fob",
            ),
            # Very high importer accuracy, very low exporter accuracy
            (
                (self.df["exporter_accuracy_score"] < percentiles["exporter_25"])
                & (self.df["importer_accuracy_score"] >= percentiles["importer_75"]),
                "import_value_fob",
            ),
            # Very high exporter accuracy, very low importer accuracy
            (
                (self.df["exporter_accuracy_score"] >= percentiles["exporter_75"])
                & (self.df["importer_accuracy_score"] < percentiles["importer_25"]),
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
            "exporter_accuracy_score",
            "importer_accuracy_score",
        ]
        self.df = self.df[columns_to_keep]
