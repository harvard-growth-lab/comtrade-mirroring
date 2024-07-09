import pandas as pd
from clean.objects.base import _AtlasCleaning
import os
import numpy as np
from sklearn.decomposition import PCA
import logging
import copy

logging.basicConfig(level=logging.INFO)


# generates a country country year table
class CountryCountryYear(_AtlasCleaning):
    niter = 25  # Iterations A_e
    trade_value_threshold = 10**4
    population_threshold = 0.5 * 10**6  
    trade_flow_threshold = 10  # 30
    flow_limit = 10**4  # Minimum value to assume that there is a flow
    vfile = 1
    anorm = 0  # Normalize the score
    alog = 0  # Apply logs
    af = 0  # Combine A_e and A_i in single measure
    seed = 1  # Initial value for the A's
    CPI_BASE_YEAR = 2010
    CIF_RATIO = 0.075

    def __init__(self, df, year, **kwargs):
        super().__init__(**kwargs)

        # Set parameters
        self.year = year
        self.df = df.copy(deep=True)

        # Step 1: Prepare economic indicators
        cpi, population = self.add_economic_indicators()
        cpi = self.inflation_adjustment(cpi)

        # Step 2: Clean and filter data
        self.clean_data()
        logging.info(f"after clean data {self.df.shape}")

        # merge data to have all possible combinations for exporter, importer
        all_combinations_ccy_index = pd.MultiIndex.from_product(
            [
                self.df["exporter"].unique(),
                self.df["importer"].unique(),
            ],
            names=["exporter", "importer"],
        )
        all_combinations_ccy = pd.DataFrame(
            index=all_combinations_ccy_index
        ).reset_index()

        self.df = all_combinations_ccy.merge(
            self.df, on=["exporter", "importer"], how="left"
        )
        
        self.filter_by_population_threshold(population)
        logging.info(f"shape after filter by population  {self.df.shape}")
        ccy = self.df.copy(deep=True)
        self.compare_base_year_trade_values()
        logging.info(f"shape after trade values comparison  {self.df.shape}")


        # Step 3: Calculate trade statistics
        self.calculate_trade_reporting_discrepancy()
        self.filter_by_trade_flows()
        ncountries = self.df["exporter"].nunique()
        self.calculate_trade_percentages()
        self.normalize_trade_flows()
        

        # Step 4: Compute accuracy scores
        ccy_accuracy = self.compute_accuracy_scores(ncountries)
    
        (
            ccy_accuracy,
            exporter_accuracy_percentiles,
            importer_accuracy_percentiles,
        ) = self.calculate_accuracy_percentiles(ccy, ccy_accuracy)

        # Step 5: Estimate trade values
        ccy_accuracy = self.calculate_weights(ccy_accuracy, exporter_accuracy_percentiles, importer_accuracy_percentiles)

        ccy_accuracy = self.calculate_estimated_value(
            ccy_accuracy, exporter_accuracy_percentiles, importer_accuracy_percentiles
        )

        # Step 6: Finalize output
        self.finalize_output(ccy_accuracy)


    def clean_data(self):
        self.df = self.df.dropna(subset=["exporter", "importer"])
        self.df = self.df[
            ~(
                (self.df.exporter.isin(["WLD", "ANS"]))
                | (self.df.importer.isin(["WLD", "ANS"]))
            )
        ]
        logging.info(f"shape of df {self.df.shape}")
        self.df = self.df[self.df.exporter != self.df.importer]
        self.df = self.df[
            self.df[["import_value_fob", "export_value_fob"]].max(axis=1) >= self.trade_value_threshold
        ]
        logging.info(f"shape of df {self.df.shape}")

        # TODO: ensures cif_ratio is never greater than .20
        # this does nothing while CIF RATIO set as a constant and not using compute_distance()
        self.df["cif_ratio"] = (
            self.df["import_value_cif"] / self.df["import_value_fob"]
        ) - 1
        self.df["cif_ratio"] = self.df["cif_ratio"].apply(
            lambda val: min(val, 0.20) if pd.notnull(val) else val
        )
        # saved as temp_accuracy.dta in stata
        # self.df.to_parquet(f"data/intermediate/ccy_{year}.parquet")

    def filter_by_population_threshold(self, population: pd.DataFrame()):
        """
        Drop all exporter and importers with populations below the population limit
        """
        population = population[population.year == self.year].drop(columns=["year"])
        countries_under_threshold = population[population.imf_pop < self.population_threshold][
            "iso"
        ].tolist()
        self.df = self.df[
            ~(
                (self.df.exporter.isin(countries_under_threshold))
                | (self.df.importer.isin(countries_under_threshold))
            )
        ]

    def filter_by_trade_flows(self):
        """
        Count the number of trade partners (trade flows) for each country  if
        country has trade partners below the trade flow threshold remove the country
        from the data as both an importer and exporter. Validate each remaining
        country is both an exporter and an importer
        """
        # trade flow count method
        for trade_flow in ["importer", "exporter"]:
            for t in range(1, 6):
                self.df["nflows"] = (
                    # if neither exports or imports are zero than count as a flow
                    (
                        (self.df["exports_const_usd"] != 0.0)
                        | (self.df["imports_const_usd"] != 0.0)
                    )
                    .groupby(self.df[trade_flow])
                    .transform("sum")
                )

                small_nflows = set(
                    self.df.loc[
                        self.df["nflows"] < self.trade_flow_threshold, trade_flow
                    ].tolist()
                )
                # Drop exporter or importer that has trade flows below threshold
                if small_nflows:
                    for iso in small_nflows:
                        self.df = self.df[
                            ~(
                                (self.df["exporter"] == iso)
                                | (self.df["importer"] == iso)
                            )
                        ]
        import pdb
        pdb.set_trace()

        # confirms all countries included have data as importer and exporter if not, then drop the country
        importer_only, exporter_only = set(self.df["importer"].unique()) - set(
            self.df["exporter"].unique()
        ), set(self.df["exporter"].unique()) - set(self.df["importer"].unique())

        if importer_only or exporter_only:
            self.df = self.df[
                ~(
                    self.df["exporter"].isin(exporter_only)
                    | self.df["importer"].isin(importer_only)
                )
            ]
        self.df = self.df.drop(columns=["nflows"])

        # leave for testing
        assert (
            self.df["exporter"].nunique() == self.df["importer"].nunique()
        ), f"Number of exporters does not equal number of importers"

        
    def compare_base_year_trade_values(self):
        """
        Convert all trade dollars to base year (2010 - US) and zero out trade between countries
        below the flow limit
        """
        cpi_index_base = self.inflation[self.inflation.year == self.year]
        self.df["cpi_index_base"] = cpi_index_base.cpi_index_base.iloc[0]

        # converts exports, import values to constant dollar values
        for col in ["export_value_fob", "import_value_fob"]:
            self.df[col] = self.df[col] / self.df.cpi_index_base

        self.df = self.df.drop(
            columns=["cpi_index_base", "import_value_cif", "cif_ratio"]
        )
        # in stata v_e and v_i
        self.df = self.df.rename(
            columns={
                "export_value_fob": "exports_const_usd",
                "import_value_fob": "imports_const_usd",
            }
        ).fillna(0.0)

        # trade below threshold is zeroed
        self.df.loc[
            self.df.exports_const_usd < self.flow_limit, "exports_const_usd"
        ] = 0.0
        self.df.loc[
            self.df.imports_const_usd < self.flow_limit, "imports_const_usd"
        ] = 0.0

        # Filter rows
        self.df = self.df.groupby("exporter").filter(
            lambda row: (row["exports_const_usd"] > 0).sum() > 0
        )
        self.df = self.df.groupby("importer").filter(
            lambda row: (row["imports_const_usd"] > 0).sum() > 0
        )

    def calculate_trade_reporting_discrepancy(self):
        """
        Takes the absolute value of the difference in export and imports
        and divides by sum of imports and exports
        """
        # in stata s_ij
        self.df["reporting_discrepancy"] = (
            (abs(self.df["exports_const_usd"] - self.df["imports_const_usd"]))
            / (self.df["exports_const_usd"] + self.df["imports_const_usd"])
        ).fillna(0.0)

    def calculate_trade_percentages(self):
        """ """
        # calculate the mean of each row, exclude import or export value when equal to zero
        self.df["trade_flow_average"] = pd.DataFrame(
            {
                "exports": np.where(
                    self.df["exports_const_usd"] != 0,
                    self.df["exports_const_usd"],
                    np.nan,
                ),
                "imports": np.where(
                    self.df["imports_const_usd"] != 0,
                    self.df["imports_const_usd"],
                    np.nan,
                ),
            }
        ).mean(axis=1)

        # percentage of countries total exports
        self.df["export_percentage"] = np.maximum(
            self.df["trade_flow_average"]
            / self.df.groupby("exporter")["trade_flow_average"].transform("sum"),
            0.0,
        )
        # percentage of countries total imports
        self.df["import_percentage"] = np.maximum(
            self.df["trade_flow_average"]
            / self.df.groupby("importer")["trade_flow_average"].transform("sum"),
            0.0,
        )
        self.df = self.df.drop(columns=["trade_flow_average"])

    def normalize_trade_flows(self):
        """ """
        # count trade flows for each country as an importer and exporter
        for trade_flow in ["importer", "exporter"]:
            self.df[f"{trade_flow}_nflows"] = (
                (
                    (self.df["exports_const_usd"] != 0.0)
                    | (self.df["imports_const_usd"] != 0.0)
                )
                .groupby(self.df[trade_flow])
                .transform("sum")
            )
            # average normalized trade imbalance by trade flow
            self.df[f"reporting_discrepancy_{trade_flow}_avg"] = self.df.groupby(
                trade_flow
            )["reporting_discrepancy"].transform("mean")

            # divide the total trade discrepancy by importer and exporter by the number of respective trade flows
            self.df[f"reporting_discrepancy_{trade_flow}_total"] = (
                self.df.groupby(trade_flow)[
                    f"reporting_discrepancy_{trade_flow}_avg"
                ].transform("sum")
                / self.df[f"{trade_flow}_nflows"]
            )

    def compute_accuracy_scores(self, ncountries):
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
        exporters = self.df["exporter"].unique()
        exporter_to_idx = {exp: idx for idx, exp in enumerate(exporters)}

        # prepare matrices to maintain indices
        # stata name: es_ij: exporters, is_ij: importers
        reporting_discrepancy = self.df.pivot(
            index="exporter", columns="importer", values="reporting_discrepancy"
        ).fillna(0)
        reporting_discrepancy = reporting_discrepancy.reindex(
            # columns=exporters
            index=exporters, columns=exporters, fill_value=0
        )

        # Convert to numpy arrays
        trdiscrep_exp = reporting_discrepancy.values
        trdiscrep_imp = trdiscrep_exp.T

        nflows_exp = (
            self.df.groupby("exporter")["exporter_nflows"].first().values.reshape(-1, 1)
        )
        nflows_imp = (
            self.df.groupby("importer")["importer_nflows"]
            .first()
            .reindex(exporters)
            .values.reshape(-1, 1)
        )
        
        
        # initialize accuracy to one
        exporter_accuracy = np.ones((ncountries, 1))
        importer_accuracy = np.ones((ncountries, 1))

        for _ in range(0, 25):
            # @ is element-wise multiplication
            exporter_accuracy_probability = 1 / np.divide(
                (trdiscrep_exp @ importer_accuracy), nflows_exp
            )
            importer_accuracy_probability = 1 / np.divide(
                (trdiscrep_imp @ exporter_accuracy), nflows_imp
            )
            
            importer_accuracy = importer_accuracy_probability
            exporter_accuracy = exporter_accuracy_probability

        import pdb
        pdb.set_trace()
        
        trdiscrep_exp = (np.sum(trdiscrep_exp, axis=1) / ncountries).reshape(-1, 1)
        trdiscrep_imp = (np.sum(trdiscrep_imp, axis=1) / ncountries).reshape(-1, 1)
        
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
            accuracy_score = np.mean([exporter_accuracy, importer_accuracy], axis=0)

        elif self.af == 1:
            accuracy_score = PCA().fit_transform(exporter_accuracy, importer_accuracy)

        if self.anorm == 1:
            accuracy_score = (
                accuracy_score - accuracy_score.mean()
            ) / accuracy_score.std()

        # combine np arrays into pandas
        year_array = np.full(ncountries, self.year).reshape(-1, 1)

        cy_accuracy = pd.DataFrame(
            np.hstack(
                [
                    year_array,
                    exporters.reshape(-1, 1),
                    nflows_exp,
                    nflows_imp,
                    trdiscrep_exp,
                    trdiscrep_imp,
                    exporter_accuracy,
                    importer_accuracy,
                    accuracy_score,
                ]
            ),
            columns=[
                "year",
                "iso",
                "nflows_exp",
                "nflows_imp",
                "trdiscrep_exp",
                "trdiscrep_imp",
                "acc_exp",
                "acc_imp",
                "acc_final",
            ],
        )
        import pdb
        pdb.set_trace()
        cy_accuracy.to_parquet("data/intermediate/accuracy_new.parquet")
        return cy_accuracy

    
    def calculate_accuracy_percentiles(self, ccy, ccy_accuracy):
        """ """
        # earlier ccy
        ccy_acc = ccy.merge(
            ccy_accuracy[["year", "iso", "acc_exp", "acc_imp"]].rename(
                columns={
                    "acc_exp": "exporter_accuracy_score",
                    "acc_imp": "acc_imp_for_exporter",
                }
            ),
            left_on=["year", "exporter"],
            right_on=["year", "iso"],
            how="left",
        ).drop(columns=["iso"])

        ccy_acc = ccy_acc.merge(
            ccy_accuracy[["year", "iso", "acc_exp", "acc_imp"]].rename(
                columns={
                    "acc_exp": "acc_exp_for_importer",
                    "acc_imp": "importer_accuracy_score",
                }
            ),
            left_on=["year", "importer"],
            right_on=["year", "iso"],
            how="left",
            suffixes=("", "_for_importer"),
        ).drop(columns=["iso"])
        
        import pdb
        pdb.set_trace()

        ccy_acc = ccy_acc[ccy_acc.importer != ccy_acc.exporter]

        for entity in ["exporter", "importer"]:
            ccy_acc[f"tag_{entity[0]}"] = (~ccy_acc[entity].duplicated()).astype(int)

        # remove trade values less than 1000, fob
        ccy_acc.loc[ccy_acc["import_value_fob"] < 1000, "import_value_fob"] = 0.0
        ccy_acc.loc[ccy_acc["export_value_fob"] < 1000, "export_value_fob"] = 0.0

        # calculating percentiles grouped by unique exporter and then importer
        exporter_accuracy_percentiles = (
            ccy_acc[ccy_acc.tag_e == 1]["exporter_accuracy_score"]
            .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            .round(3)
        )
        importer_accuracy_percentiles = (
            ccy_acc[ccy_acc.tag_i == 1]["importer_accuracy_score"]
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
            ccy_acc[col] = pd.to_numeric(ccy_acc[col], errors="coerce")
        return ccy_acc, exporter_accuracy_percentiles, importer_accuracy_percentiles

            
    def calculate_weights(self, ccy_acc, exporter_accuracy_percentiles, importer_accuracy_percentiles):
        """ 
        """
        ccy_acc["weight"] = np.exp(ccy_acc["exporter_accuracy_score"]) / (
            np.exp(ccy_acc["exporter_accuracy_score"])
            + np.exp(ccy_acc["importer_accuracy_score"])
        )

        # include set of countries
        ccy_acc = ccy_acc.assign(
            weight_exporter=np.where(
                (ccy_acc.exporter_accuracy_score.notna())
                & (ccy_acc.exporter_accuracy_score > exporter_accuracy_percentiles[0.10]),
                1,
                0,
            ),
            weight_importer=np.where(
                (ccy_acc.importer_accuracy_score.notna())
                & (ccy_acc.importer_accuracy_score > importer_accuracy_percentiles[0.10]),
                1,
                0,
            ),
        )
        import pdb
        pdb.set_trace()

        ccy_acc["discrep"] = np.exp(
            np.abs(np.log(ccy_acc["export_value_fob"] / ccy_acc["import_value_fob"]))
        )
        ccy_acc["discrep"] = ccy_acc["discrep"].replace(np.nan, 99)
        return ccy_acc

        
    def calculate_estimated_value(self, df, export_percentiles, import_percentiles):
        """
        Series of conditions to determine estimated trade value
        """
        logging.info("Estimating total trade flows between countries")

        df["est_trade_value"] = np.where(
            (df.exporter_accuracy_score.notna())
            & (df.importer_accuracy_score.notna())
            & (df.import_value_fob.notna())
            & (df.export_value_fob.notna()),
            df.export_value_fob * df.weight + df.import_value_fob * (1 - df.weight),
            np.nan,
        )

        # conditions to determine est_trade_value
        # TODO: loop through and filter dataset so not running notna() each time
        df.loc[
            (df["exporter_accuracy_score"] < export_percentiles[0.50])
            & (df["importer_accuracy_score"] >= import_percentiles[0.90])
            & (df["importer_accuracy_score"].notna())
            & (df["exporter_accuracy_score"].notna())
            & df["est_trade_value"].isna(),
            "est_trade_value",
        ] = df["import_value_fob"]

        df.loc[
            (df["exporter_accuracy_score"] >= export_percentiles[0.90])
            & (df["importer_accuracy_score"] < import_percentiles[0.50])
            & (df["importer_accuracy_score"].notna())
            & (df["exporter_accuracy_score"].notna())
            & df["est_trade_value"].isna(),
            "est_trade_value",
        ] = df["export_value_fob"]

        df.loc[
            (df["exporter_accuracy_score"] < export_percentiles[0.25])
            & (df["importer_accuracy_score"] >= import_percentiles[0.75])
            & (df["importer_accuracy_score"].notna())
            & (df["exporter_accuracy_score"].notna())
            & df["est_trade_value"].isna(),
            "est_trade_value",
        ] = df["import_value_fob"]

        df.loc[
            (df["exporter_accuracy_score"] >= export_percentiles[0.75])
            & (df["importer_accuracy_score"] < import_percentiles[0.25])
            & (df["importer_accuracy_score"].notna())
            & (df["exporter_accuracy_score"].notna())
            & df["est_trade_value"].isna(),
            "est_trade_value",
        ] = df["export_value_fob"]

        df.loc[
            (df["importer_accuracy_score"].notna())
            & (df["exporter_accuracy_score"].notna())
            & (df["weight_exporter"] == 1)
            & (df["weight_importer"] == 1)
            & df["est_trade_value"].isna(),
            "est_trade_value",
        ] = df[["import_value_fob", "export_value_fob"]].max(axis=1)

        df.loc[
            (df["weight_importer"] == 1) & df["est_trade_value"].isna(),
            "est_trade_value",
        ] = df["import_value_fob"]
        df.loc[
            (df["weight_exporter"] == 1) & df["est_trade_value"].isna(),
            "est_trade_value",
        ] = df["export_value_fob"]
        df.loc[df["est_trade_value"].isna(), "est_trade_value"] = df["import_value_fob"]
        df.loc[df["est_trade_value"] == 0, "est_trade_value"] = np.nan

        df = df.drop(columns=["discrep"])

        # Calculate mintrade and update estvalue
        df["min_trade"] = df[["export_value_fob", "import_value_fob"]].min(
            axis=1
        )
        df.loc[
            (df["min_trade"].notna()) & (df["est_trade_value"].isna()),
            "est_trade_value",
        ] = df["min_trade"]
        return df
        
    def finalize_output(self, df):
        """
        """
        df = df.rename(
            columns={
                "export_value_fob": "export_value",
                "import_value_fob": "import_value",
                "est_trade_value": "final_trade_value",
            }
        )

        # Select and reorder columns
        columns_to_keep = [
            "year",
            "exporter",
            "importer",
            "export_value",
            "import_value",
            "final_trade_value",
            # "cif_ratio",
            "weight",
            "weight_exporter",
            "weight_importer",
            "exporter_accuracy_score",
            "importer_accuracy_score",
        ]
        df = df[columns_to_keep]
        logging.info("PAUSE BEFORE SAVING NEW")
        import pdb
        pdb.set_trace()

        # Save the DataFrame to a file
        output_path = os.path.join(
            self.intermediate_data_path, f"weights_{self.year}_new.parquet"
        )
        df.to_parquet(output_path)

        
    def add_economic_indicators(self):
        """
        population and cpi from wdi
        # TODO: convert to IMF data, use wdi initially to compare values
        """
        wdi = (
            pd.read_stata(
                self.wdi_path, columns=["year", "iso", "fp_cpi_totl_zg", "sp_pop_totl"]
            )
            .rename(columns={"fp_cpi_totl_zg": "cpi"})
            .reset_index(drop=True)
        )
        # price index
        wdi_cpi = wdi[(wdi.year >= 1962) & (wdi.iso == "USA")].drop(
            ["sp_pop_totl"], axis=1
        )

        # population
        wdi_pop = wdi.drop(["cpi"], axis=1)

        imf_pop = pd.read_csv(
            os.path.join(self.raw_data_path, "imf_data.csv"),
            usecols=["code", "year", "population"],
        ).rename(columns={"population": "imf_pop"})

        # if empty fill in with imf population data
        pop = wdi_pop.merge(
            imf_pop, left_on=["iso", "year"], right_on=["code", "year"], how="outer"
        ).drop(columns=["code"])
        return wdi_cpi, pop

    def inflation_adjustment(self, cpi):
        """ """
        logging.info("Inflation adjustment")
        cpi = cpi.reset_index(drop=True)

        for i, row in cpi.iterrows():
            if i == 0:
                cpi.at[i, "cpi_index"] = 100.0
            else:
                cpi.at[i, "cpi_index"] = cpi.iloc[i - 1]["cpi_index"] * (
                    1 + row.cpi / 100
                )

        # sets base year at 2010
        base_year_cpi_index = cpi.loc[cpi.year == self.CPI_BASE_YEAR, "cpi_index"].iloc[
            0
        ]
        cpi["cpi_index_base"] = cpi["cpi_index"] / base_year_cpi_index

        cpi.to_parquet(
            os.path.join(self.intermediate_data_path, "inflation_index.parquet")
        )
        return cpi
