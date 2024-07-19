import pandas as pd
from clean.table_objects.base import _AtlasCleaning
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

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)

        # Set parameters
        self.year = year
        self.df = self.load_parquet(
            os.path.join(f"intermediate", self.product_classification),
            f"{self.product_classification}_{self.year}",
        )

        # Prepare economic indicators
        cpi, population = self.add_economic_indicators()
        cpi = self.inflation_adjustment(cpi)

        # Clean and filter data
        self.clean_data()

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
        # temp_accuracy in stata
        self.apply_relative_cif_markup()
        # save intermediate ccy file (saved as temp_accuracy.dta in stata file)
        self.save_parquet(self.df, "intermediate", "ccy_cif_markup")

        self.filter_by_population_threshold(population)
        self.compare_base_year_trade_values()

        # Calculate trade statistics
        self.calculate_trade_reporting_discrepancy()
        self.filter_by_trade_flows()
        self.calculate_trade_percentages()
        self.normalize_trade_flows()

    def clean_data(self):
        self.df = self.df.dropna(subset=["exporter", "importer"])
        self.df = self.df[
            ~(
                (self.df.exporter.isin(["WLD", "ANS"]))
                | (self.df.importer.isin(["WLD", "ANS"]))
            )
        ]
        self.df = self.df[self.df.exporter != self.df.importer]
        # drop trade values less than trade value threshold
        self.df = self.df[
            self.df[["import_value_fob", "export_value_fob"]].max(axis=1)
            >= self.trade_value_threshold
        ]

    def apply_relative_cif_markup(self):
        """ """
        # ensures cif_ratio is never greater than .20
        self.df["cif_ratio"] = (
            self.df["import_value_cif"] / self.df["import_value_fob"]
        ) - 1
        logging.info("review CIF ratio from compute distance")
        self.df["cif_ratio"] = self.df["cif_ratio"].apply(
            lambda val: min(val, 0.20) if pd.notnull(val) else val
        )

    def filter_by_population_threshold(self, population: pd.DataFrame()):
        """
        Drop all exporter and importers with populations below the population limit
        """
        population = population[population.year == self.year].drop(columns=["year"])
        countries_under_threshold = population[
            population.imf_pop < self.population_threshold
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
            columns=["cpi_index_base", "import_value_cif"]  # , "cif_ratio"]
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
        # in stata s_ij, should be fob and 
        logging.info("***** review why not FOB *******")
        # import pdb
        # pdb.set_trace()
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

        self.save_parquet(cpi, "intermediate", "inflation_index")
        # cpi.to_parquet(
        #     os.path.join(self.intermediate_data_path, "inflation_index.parquet")
        # )
        return cpi
