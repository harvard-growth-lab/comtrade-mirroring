import pandas as pd
from clean.table_objects.base import _AtlasCleaning
import os
import numpy as np
from sklearn.decomposition import PCA
import logging
import copy

logging.basicConfig(level=logging.INFO)


# generates a country country year table
class Accuracy(_AtlasCleaning):
    anorm = 0  # Normalize the score
    alog = 0  # Apply logs
    af = 0  # weight exporter_accuracy and importer_accuracy in single measure

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)

        # Set parameters
        # self.ncountries = ncountries
        self.year = year
        self.df = pd.DataFrame()

        # load data
        nominal_dollars_df = self.load_parquet("intermediate", "ccy_nominal_dollars")
        self.ccy = self.load_parquet(f"processed", f"country_country_year_{year}")

        # Compute accuracy scores, called temp.dta in stata
        ccy_accuracy = self.compute_accuracy_scores()

        (
            exporter_accuracy_percentiles,
            importer_accuracy_percentiles,
        ) = self.calculate_accuracy_percentiles(nominal_dollars_df, ccy_accuracy)

        # Estimate trade values
        self.calculate_weights(
            exporter_accuracy_percentiles, importer_accuracy_percentiles
        )

        self.calculate_estimated_value(
            exporter_accuracy_percentiles, importer_accuracy_percentiles
        )

        self.finalize_output()

    def compute_accuracy_scores(self):
        """
        Compute accuracy scores for exporters and importers based on trade reporting discrepancies.

        The accuracy scores are based on the consistency of trade reporting between countries and the number
        of trade partners each country has. Countries with more consistent reporting and more trade partners
        tend to receive higher accuracy scores.

        This function performs the following steps:
            1. Creates a matrix of reporting discrepancies between exporters and importers.
            2. Converts the discrepancy matrix and trade flow counts to numpy arrays for efficient computation.
            3. Iteratively calculates accuracy scores for exporters and importers using a probabilistic method.
            4. Applies optional logarithmic transformation and normalization to the accuracy scores.
            5. Combines exporter and importer accuracy scores into a final accuracy score.
            6. Optionally applies Principal Component Analysis (PCA) for dimension reduction.
            7. Normalizes the final accuracy score if specified.
        """
        exporters = pd.DataFrame(self.ccy["exporter"].unique())
        importers = pd.DataFrame(self.ccy["importer"].unique())
        self.ncountries = len(exporters)

        # prepare matrices to maintain indices
        # stata name: es_ij: exporters, is_ij: importers
        trdiscrep_exp = self.ccy.pivot(
            index="exporter", columns="importer", values="reporting_discrepancy"
        ).fillna(0)
        trdiscrep_imp = self.ccy.pivot(
            index="importer", columns="exporter", values="reporting_discrepancy"
        ).fillna(0)

        nflows_exp = self.ccy.groupby("exporter")["exporter_nflows"].first()
        nflows_imp = self.ccy.groupby("importer")["importer_nflows"].first()

        iso_index = nflows_exp.index

        # initialize accuracy to one
        exporter_accuracy = pd.DataFrame(index=iso_index)
        exporter_accuracy["exporter_accuracy"] = 1
        importer_accuracy = pd.DataFrame(index=iso_index)
        importer_accuracy["importer_accuracy"] = 1

        for i in range(0, 25):  # 25 try
            # @ is element-wise multiplication
            exporter_probability = 1 / (
                (trdiscrep_imp @ importer_accuracy["importer_accuracy"]) / nflows_exp
            )
            importer_probability = 1 / (
                (trdiscrep_exp @ exporter_accuracy["exporter_accuracy"]) / nflows_imp
            )

            importer_accuracy["importer_accuracy"] = importer_probability
            exporter_accuracy["exporter_accuracy"] = exporter_probability

        # unstack
        trdiscrep_exp = (
            trdiscrep_exp.stack()
            .reset_index()
            .groupby("exporter")
            .agg("sum")
            .drop(columns=["importer"])
            / self.ncountries
        )
        trdiscrep_exp = trdiscrep_exp.rename(columns={0: "trdiscrep_exp"})

        trdiscrep_imp = (
            trdiscrep_imp.stack()
            .reset_index()
            .groupby("importer")
            .agg("sum")
            .drop(columns=["exporter"])
            / self.ncountries
        )
        trdiscrep_imp = trdiscrep_imp.rename(
            columns={0: "trdiscrep_imp"}, index={"importer": "exporter"}
        )
        exporters = exporters.rename(columns={0: "exporter"})

        # fix some df has single exporter for year 2015
        if self.alog == 1:
            exporter_accuracy = np.ln(exporter_accuracy)
            importer_accuracy = np.ln(importer_accuracy)
        if self.anorm == 1:
            exporter_accuracy = (
                exporter_accuracy - exporter_accuracy.mean()
            ) / exporter_accuracy.std()
            importer_accuracy = (
                importer_accuracy - importer_accuracy.mean()
            ) / importer_accuracy.std()

        if self.af == 0:
            # this is set to run
            logging.info("calculating mean of exporter accuracy and importer accuracy")
            accuracy_score = (
                exporter_accuracy["exporter_accuracy"]
                + importer_accuracy["importer_accuracy"] / 2
            )

        elif self.af == 1:
            accuracy_score = PCA().fit_transform(exporter_accuracy, importer_accuracy)

        if self.anorm == 1:
            accuracy_score = (
                accuracy_score - accuracy_score.mean()
            ) / accuracy_score.std()

        ccy_accuracy = pd.concat(
            [
                nflows_exp,
                nflows_imp.rename(index={"importer": "exporter"}),
                trdiscrep_exp["trdiscrep_exp"],
                trdiscrep_imp.rename(index={"importer": "exporter"})["trdiscrep_imp"],
                exporter_accuracy["exporter_accuracy"],
                importer_accuracy.rename(index={"importer": "exporter"})[
                    "importer_accuracy"
                ],
                accuracy_score,
            ],
            axis=1,
        )

        ccy_accuracy["year"] = self.year
        ccy_accuracy = ccy_accuracy.reset_index().rename(
            columns={"index": "iso", 0: "acc_final"}
        )
        return ccy_accuracy

    def calculate_accuracy_percentiles(self, ccy, ccy_accuracy):
        """ """
        self.df = ccy.merge(
            # merging exporter accuracy scores
            ccy_accuracy[["iso", "exporter_accuracy", "importer_accuracy"]].rename(
                columns={
                    # stata exporter_A_e
                    "exporter_accuracy": "exporter_accuracy_score",
                    # stata exporter_A_i
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
                    # stata importer_A_e
                    "exporter_accuracy": "acc_exp_for_importer",
                    # stata importer_A_i
                    "importer_accuracy": "importer_accuracy_score",
                }
            ),
            left_on=["importer"],
            right_on=["iso"],
            how="left",
            suffixes=("", "_for_importer"),
        ).drop(columns=["iso"])

        self.df = self.df[self.df.importer != self.df.exporter]

        for entity in ["exporter", "importer"]:
            self.df[f"tag_{entity[0]}"] = (~self.df[entity].duplicated()).astype(int)

        # remove trade values less than 1000, fob
        # self.df.loc[self.df["import_value_fob"] < 1000, "import_value_fob"] = 0.0
        # self.df.loc[self.df["export_value_fob"] < 1000, "export_value_fob"] = 0.0

        # calculating percentiles grouped by unique exporter and then importer
        exporter_accuracy_percentiles = (
            self.df[self.df.tag_e == 1]["exporter_accuracy_score"]
            .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            .round(3)
        )
        importer_accuracy_percentiles = (
            self.df[self.df.tag_i == 1]["importer_accuracy_score"]
            .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
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
        return exporter_accuracy_percentiles, importer_accuracy_percentiles

    def calculate_weights(
        self, exporter_accuracy_percentiles, importer_accuracy_percentiles
    ):
        """ """
        self.df["weight"] = np.exp(self.df["exporter_accuracy_score"]) / (
            np.exp(self.df["exporter_accuracy_score"])
            + np.exp(self.df["importer_accuracy_score"])
        )

        # include set of countries
        self.df = self.df.assign(
            exporter_weight=np.where(
                (self.df.exporter_accuracy_score.notna())
                & (
                    self.df.exporter_accuracy_score
                    > exporter_accuracy_percentiles[0.10]
                ),
                1,
                0,
            ),
            importer_weight=np.where(
                (self.df.importer_accuracy_score.notna())
                & (
                    self.df.importer_accuracy_score
                    > importer_accuracy_percentiles[0.10]
                ),
                1,
                0,
            ),
        )

        # self.df["discrep"] = np.exp(
        #     np.abs(np.log(self.df["export_value_fob"] / self.df["import_value_fob"]))
        # )
        # self.df["discrep"] = self.df["discrep"].fillna(0.0)

    def calculate_estimated_value(self, export_percentiles, import_percentiles):
        """
        Series of filtered data based on Nan values with applied conditions to determine
        estimated trade value. Uses accuracy scores and relative percentage of imports and exports
        """
        # est trade value only if accuracy scores and trade values are not nan
        filtered_df = (
            (self.df["importer_accuracy_score"].notna())
            & (self.df["exporter_accuracy_score"].notna())
            & (self.df["import_value_fob"].notna())
            & (self.df["export_value_fob"].notna())
        )
        self.df.loc[filtered_df, "est_trade_value"] = self.df.loc[
            filtered_df, "export_value_fob"
        ] * self.df.loc[filtered_df, "weight"] + self.df.loc[
            filtered_df, "import_value_fob"
        ] * (
            1 - self.df.loc[filtered_df, "weight"]
        )

        # est trade value only if accuracy scores are not nan
        filtered_df = (
            (self.df["importer_accuracy_score"].notna())
            & (self.df["exporter_accuracy_score"].notna())
            & self.df["est_trade_value"].isna()
        )

        # conditions to determine est_trade_value
        conditions = [
            (self.df["exporter_accuracy_score"] < export_percentiles[0.50])
            & (self.df["importer_accuracy_score"] >= import_percentiles[0.90]),
            (self.df["exporter_accuracy_score"] >= export_percentiles[0.90])
            & (self.df["importer_accuracy_score"] < import_percentiles[0.50]),
            (self.df["exporter_accuracy_score"] < export_percentiles[0.25])
            & (self.df["importer_accuracy_score"] >= import_percentiles[0.75]),
            (self.df["exporter_accuracy_score"] >= export_percentiles[0.75])
            & (self.df["importer_accuracy_score"] < import_percentiles[0.25]),
            (self.df["exporter_weight"] == 1) & (self.df["importer_weight"] == 1),
        ]

        replacement_values = [
            self.df["import_value_fob"],
            self.df["export_value_fob"],
            self.df["import_value_fob"],
            self.df["export_value_fob"],
            self.df[["import_value_fob", "export_value_fob"]].max(axis=1),
        ]

        result = np.select(
            condlist=conditions,
            choicelist=replacement_values,
            default=self.df["est_trade_value"],
        )
        self.df.loc[filtered_df, "est_trade_value"] = result[filtered_df]

        # remaining est trade value with nan
        filtered_df = self.df["est_trade_value"].isna()
        importer_mask = filtered_df & (self.df["importer_weight"] == 1)
        exporter_mask = filtered_df & (self.df["exporter_weight"] == 1)

        self.df.loc[importer_mask, "est_trade_value"] = self.df.loc[
            importer_mask, "import_value_fob"
        ]
        self.df.loc[exporter_mask, "est_trade_value"] = self.df.loc[
            exporter_mask, "export_value_fob"
        ]

        # fill remaining NaNs with import_value_fob
        self.df["est_trade_value"] = self.df["est_trade_value"].fillna(
            self.df["import_value_fob"]
        )

        # Replace est trade value 0 with NaN
        self.df.loc[self.df["est_trade_value"] == 0, "est_trade_value"] = np.nan

        # self.df = self.df.drop(columns=["discrep"])

        # Calculate mintrade and update estvalue
        self.df["min_trade"] = self.df[["export_value_fob", "import_value_fob"]].min(
            axis=1
        )
        self.df.loc[
            (self.df["min_trade"].notna()) & (self.df["est_trade_value"].isna()),
            "est_trade_value",
        ] = self.df["min_trade"]

    def finalize_output(self):
        """ """
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
