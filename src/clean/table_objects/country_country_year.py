import pandas as pd
from clean.objects.base import _AtlasCleaning
import os
import numpy as np
from sklearn.decomposition import PCA
import logging
import copy

logging.basicConfig(level=logging.INFO)


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

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)

        # Set parameters
        self.year = year
        self.df = self.load_parquet(
            os.path.join(f"intermediate"),
            f"{self.product_classification}_{self.year}",
        )

        # Clean and filter data
        self.clean_data()

        # temp_accuracy in stata
        nominal_dollars_df = self.df.copy(deep=True)

        nominal_dollars_df = self.limit_cif_markup(nominal_dollars_df)
        # # save intermediate ccy file (saved as temp_accuracy.dta in stata file)
        self.save_parquet(
            nominal_dollars_df,
            "intermediate",
            f"{self.product_classification}_ccy_nominal_dollars",
        )

        # read in economic indicators
        population, fred = self.add_economic_indicators()

        # merge data to have all possible combinations for exporter, importer
        all_combinations_ccy_index = pd.MultiIndex.from_product(
            [
                [self.year],
                self.df["exporter"].unique(),
                self.df["importer"].unique(),
            ],
            names=["year", "exporter", "importer"],
        )
        all_combinations_ccy = (
            pd.DataFrame(index=all_combinations_ccy_index)
            .query("importer != exporter")
            .reset_index()
            .drop_duplicates()
        )

        self.df = all_combinations_ccy.merge(
            self.df, on=["year", "exporter", "importer"], how="left"
        )
        self.df = self.df.drop(columns=["year"])

        self.filter_by_population_threshold(population)

        self.compare_base_year_trade_values(fred)

        # Calculate trade statistics
        self.calculate_trade_reporting_discrepancy()

        self.filter_by_trade_flows()

        self.calculate_trade_percentages()

        self.normalize_trade_flows()

    def clean_data(self):
        self.df = self.df[~((self.df.exporter == "WLD") | (self.df.importer == "WLD"))]
        self.df = self.df[~((self.df.exporter == "nan") | (self.df.importer == "nan"))]

        self.df = self.df[~((self.df.exporter == "ANS") & (self.df.importer == "ANS"))]
        self.df = self.df[self.df.exporter != self.df.importer]
        # drop trade values less than trade value threshold
        self.df = self.df[
            self.df[["import_value_fob", "export_value_fob"]].max(axis=1, skipna=True)
            >= self.trade_value_threshold
        ]

    def limit_cif_markup(self, df):
        """
        ensures cif_ratio is never greater than .20
        """
        df["cif_ratio"] = (df["import_value_cif"] / self.df["import_value_fob"]) - 1
        df["cif_ratio"] = df.groupby("exporter")["cif_ratio"].transform("mean")
        logging.info("review CIF ratio from compute distance")
        df["cif_ratio"] = df["cif_ratio"].apply(
            lambda val: min(val, 0.20) if pd.notnull(val) else val
        )
        df.loc[df.cif_ratio.isna(), "cif_ratio"] = 0.20
        return df

    def filter_by_population_threshold(self, population: pd.DataFrame()):
        """
        Drop all exporter and importers with populations below the population limit
        to reduce noise when determining accuracy scores
        """

        population = population[population.year == self.year].drop(columns=["year"])
        population.loc[population["imf_pop"].isna(), "imf_pop"] = population["wdi_pop"]
        population = population.rename(columns={"imf_pop": "imf_wdi_pop"})

        countries_under_threshold = population[
            population.imf_wdi_pop < self.population_threshold
        ]["iso"].tolist()

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

        assert (
            self.df["exporter"].nunique() == self.df["importer"].nunique()
        ), f"Number of exporters does not equal number of importers"

    def compare_base_year_trade_values(self, fred):
        """
        Convert all trade dollars to base year (2010 - US) and zero out trade between countries
        below the flow limit
        """
        # converts exports, import values to constant dollar values
        for col in ["export_value_fob", "import_value_fob"]:
            self.df[col] = (
                self.df[col] / fred[fred.year == self.year]["deflator"].iloc[0]
            )

        self.df[["export_value_fob", "import_value_fob", "import_value_cif"]] = self.df[
            ["export_value_fob", "import_value_fob", "import_value_cif"]
        ].fillna(0)

        self.df = self.df.drop(
            columns=["import_value_cif"]  # , "cif_ratio"] "cpi_index_base",
        )
        # in stata v_e and v_i
        self.df = self.df.rename(
            columns={
                "export_value_fob": "exports_const_usd",
                "import_value_fob": "imports_const_usd",
            }
        )

        # trade below threshold is zeroed
        self.df.loc[
            self.df.exports_const_usd < self.flow_limit, "exports_const_usd"
        ] = 0.0
        self.df.loc[
            self.df.imports_const_usd < self.flow_limit, "imports_const_usd"
        ] = 0.0

        self.df = self.df.groupby("exporter").filter(
            lambda row: (row["exports_const_usd"] > 0).sum() > 0
        )
        self.df = self.df.groupby("importer").filter(
            lambda row: (row["imports_const_usd"] > 0).sum() > 0
        )

    def calculate_trade_reporting_discrepancy(self):
        """
        Takes the absolute value of the difference in export and imports
        and divides by sum of imports and exports replaces nans with 0
        """
        # in stata s_ij, should be fob and
        self.df["reporting_discrepancy"] = (
            (abs(self.df["exports_const_usd"] - self.df["imports_const_usd"]))
            / (self.df["exports_const_usd"] + self.df["imports_const_usd"])
        ).fillna(0)

    def calculate_trade_percentages(self):
        """
        calculate the mean of each row, exclude import or export value when equal to zero
        """

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

    def add_economic_indicators(self):
        """
        population and produce price index from FRED (st. louis)
        """
        fred = pd.read_csv(
            os.path.join(self.atlas_common_path, "fred", "data", "fred_ppiidc.csv")
        )
        logging.info(
            f"base year set to {fred.atlas_base_year.unique()}, should be same as atlas data year"
        )
        fred = fred[["year", "deflator"]]
        fred = fred[fred.year >= 1962]

        # population
        wdi_pop = (
            pd.read_stata(self.wdi_path, columns=["year", "iso", "sp_pop_totl"])
            .reset_index(drop=True)
            .rename(columns={"sp_pop_totl": "wdi_pop"})
        )

        imf_pop = pd.read_csv(
            os.path.join(self.raw_data_path, "imf_data.csv"),
            usecols=["code", "year", "population"],
        ).rename(columns={"population": "imf_pop"})

        pop = imf_pop.merge(
            wdi_pop, left_on=["code", "year"], right_on=["iso", "year"], how="outer"
        ).drop(columns=["code"])
        return pop, fred
