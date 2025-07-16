import pandas as pd
from src.objects.base import AtlasCleaning
from src.utils.api_handler import IMFData, WDIData
import os
from pathlib import Path
import numpy as np
import copy
from fredapi import Fred
from datetime import datetime
from src.utils.logging import get_logger

logger = get_logger(__name__)

pd.set_option("future.no_silent_downcasting", True)


class TradeAnalysisCleaner(AtlasCleaning):
    MAX_EXPORTER_CIF_RATIO = 0.2
    # Producer Price Index by Commodity: Industrial Commodities
    FRED_SERIES_ID = "PPIIDC"
    FRED_INDEX_MONTH = 12
    SITC_START_YEAR = 1962

    # filter out countries below thresholds
    POPULATION_THRESHOLD = 0.5 * 10**6
    TRADE_FLOW_THRESHOLD = 10
    TRADE_VALUE_MIN_THRESHOLD = 10**4  # Minimum value to assume that there is a flow
    MAX_ITERATIONS = 6

    def __init__(
        self, year: int, df: pd.DataFrame, fred_api_key=None, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.year = year
        self.df = df

        self.filter_bilateral_trade_data()

        nominal_dollars_df = self.df.copy(deep=True)
        self.cap_cif_markup_ratio(nominal_dollars_df)

        fred = self.fetch_ppiidc_deflators()
        population = self.fetch_population_data()

        # merge data to have all possible combinations for exporter, importer
        self.expand_to_all_country_pairs()

        # thresholds filter countries prior to calculating reliability scores
        # to prevent anomalies from skewing country data
        self.filter_by_population_threshold(population)
        # deflate values to atlas data year for improved comparison
        self.deflate_trade_values_to_latest_year(fred)
        self.filter_countries_below_trade_value_threshold()
        self.calculate_trade_reporting_discrepancy()
        self.filter_countries_below_trade_flow_threshold()
        self.ensure_bidirectional_presence()

        self.calculate_trade_share_percentages()
        self.calculate_country_reporting_quality_metrics()

    def filter_bilateral_trade_data(self) -> None:
        """
        Remove invalid trade records and filter for meaningful bilateral trade flows.

        Filters the trade data to exclude:
        - World aggregate records (where exporter or importer is 'WLD')
        - Records with missing country codes (where exporter or importer is 'nan')
        - Self-trade records (where exporter equals importer)
        - Areas not specified
        - Trade flows below the minimum value threshold
        """
        self.df = self.df[~((self.df.exporter == "WLD") | (self.df.importer == "WLD"))]
        self.df = self.df[~((self.df.exporter == "nan") | (self.df.importer == "nan"))]

        self.df = self.df[~((self.df.exporter == "ANS") & (self.df.importer == "ANS"))]
        self.df = self.df[self.df.exporter != self.df.importer]
        # drop trade values less than trade value threshold
        self.df = self.df[
            self.df[["import_value_fob", "export_value_fob"]].max(axis=1, skipna=True)
            >= self.TRADE_VALUE_MIN_THRESHOLD
        ]

    def cap_cif_markup_ratio(self, df: pd.DataFrame) -> None:
        """
        Caps the mean CIF ratio for a single exporter and replaces
        missing CIF ratios with the capped ratio value

        Saves nominal trade value data before country filters
        for processing downstream
        """
        df["cif_ratio"] = (df["import_value_cif"] / df["import_value_fob"]) - 1
        df["cif_ratio"] = df.groupby("exporter")["cif_ratio"].transform("mean")
        df["cif_ratio"] = df["cif_ratio"].clip(upper=self.MAX_EXPORTER_CIF_RATIO)
        df.loc[df.cif_ratio.isna(), "cif_ratio"] = self.MAX_EXPORTER_CIF_RATIO
        self.save_parquet(
            df,
            "intermediate",
            f"{self.product_classification}_{self.year}_ccy_nominal_dollars",
        )

    def fetch_ppiidc_deflators(self) -> pd.DataFrame:
        """
        Calculate inflation deflators using FRED Producer Price Index data.

        Fetches the Producer Price Index for Industrial Commodities (PPIIDC) from
        the Federal Reserve Economic Data (FRED) API and calculates deflators
        relative to the latest year as the base year.
        """
        # https://fred.stlouisfed.org/series/PPIIDC
        # Producer Price Index by Commodity: Industrial Commodities
        fred = Fred(self.fred_api_key)

        try:
            ppiidc_series = fred.get_series_latest_release(self.FRED_SERIES_ID)
        except Exception as e:
            raise ValueError(f"Failed to fetch FRED series {self.FRED_SERIES_ID}: {e}")

        df = pd.DataFrame(
            {"date": ppiidc_series.index, "ppiidc_index": ppiidc_series.values}
        )

        # use December (12) index
        df = df[df["date"].dt.month == self.FRED_INDEX_MONTH]
        df["year"] = df["date"].dt.year
        base = df.loc[df["date"].dt.year == self.latest_data_year, "ppiidc_index"].iloc[
            0
        ]
        df["base_year"] = self.latest_data_year
        df["deflator"] = df["ppiidc_index"] / base
        df = df[["year", "deflator"]]
        return df[df.year >= self.SITC_START_YEAR]

    def fetch_population_data(self):
        """
        population and produce price index from FRED (st. louis)
        """
        wdi_obj = WDIData(self.latest_data_year)
        wdi_pop = wdi_obj.query_for_wdi_indicators({"SP.POP.TOTL": "population"})
        wdi_pop = wdi_pop.rename(columns={"population": "wdi_pop"})

        imf_obj = IMFData(self.latest_data_year)
        imf_pop = imf_obj.query_imf_api(["LP"])
        imf_pop = imf_pop.rename(columns={"population": "imf_pop"})

        return imf_pop.merge(wdi_pop, on=["iso3_code", "year"], how="outer")

    def expand_to_all_country_pairs(self) -> None:
        """
        Creates a complete matrix of all possible exporter-importer pairs for a given
        year, including pairs with no reported trade. This ensures the dataset has
        consistent dimensions regardless of which country pairs actually traded.
        """
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

    def filter_by_population_threshold(self, population: pd.DataFrame) -> None:
        """
        Drop all exporter and importers with populations below the population limit
        to reduce noise when determining reliability scores

        Uses IMF population data with WDI as fallback.
        """
        population = population[population.year == self.year].drop(columns=["year"])
        population["imf_pop"] = (
            population["imf_pop"]
            .fillna(population["wdi_pop"])
            .infer_objects(copy=False)
            .astype("float64")
        )

        population = population.rename(columns={"imf_pop": "imf_wdi_pop"})

        countries_under_threshold = population[
            population.imf_wdi_pop < self.POPULATION_THRESHOLD
        ]["iso3_code"].tolist()

        self.df = self.df[
            ~(
                (self.df.exporter.isin(countries_under_threshold))
                | (self.df.importer.isin(countries_under_threshold))
            )
        ]

    def filter_countries_below_trade_flow_threshold(self) -> None:
        """
        Remove countries with insufficient trade relationships.

        Iteratively filters out countries that have fewer trade partners than the
        minimum threshold. Validate each remaining country is both an exporter and
        an importer
        """
        # Iteratively remove countries with insufficient partners
        for trade_flow in ["importer", "exporter"]:
            for _ in range(1, self.MAX_ITERATIONS):
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
                        self.df["nflows"] < self.TRADE_FLOW_THRESHOLD, trade_flow
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

        self.df = self.df.drop(columns=["nflows"])

    def ensure_bidirectional_presence(self) -> None:
        """
        Remove countries that only appear as exporter or importer
        """
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
        assert (
            self.df["exporter"].nunique() == self.df["importer"].nunique()
        ), f"Number of exporters does not equal number of importers"

    def deflate_trade_values_to_latest_year(self, fred: pd.DataFrame) -> None:
        """
        Convert nominal trade values to constant USD and apply flow thresholds.

        Deflates trade values to constant dollars using FRED price deflators,
        applies minimum flow thresholds, and removes countries with no qualifying trade.
        """
        # converts exports, import values to constant dollar values
        for col in ["export_value_fob", "import_value_fob"]:
            self.df[col] = (
                self.df[col]
                / fred[fred.year == self.latest_data_year]["deflator"].iloc[0]
            )

        self.df[["export_value_fob", "import_value_fob", "import_value_cif"]] = self.df[
            ["export_value_fob", "import_value_fob", "import_value_cif"]
        ].fillna(0)

        self.df = self.df.drop(columns=["import_value_cif"])
        self.df = self.df.rename(
            columns={
                "export_value_fob": "exports_const_usd",
                "import_value_fob": "imports_const_usd",
            }
        )

    def filter_countries_below_trade_value_threshold(self) -> None:
        """
        Remove countries with no trade flows above threshold.
        """
        self.df.loc[
            self.df.exports_const_usd < self.TRADE_VALUE_MIN_THRESHOLD,
            "exports_const_usd",
        ] = 0.0
        self.df.loc[
            self.df.imports_const_usd < self.TRADE_VALUE_MIN_THRESHOLD,
            "imports_const_usd",
        ] = 0.0

        self.df = self.df.groupby("exporter").filter(
            lambda row: (row["exports_const_usd"] > 0).sum() > 0
        )
        self.df = self.df.groupby("importer").filter(
            lambda row: (row["imports_const_usd"] > 0).sum() > 0
        )

    def calculate_trade_reporting_discrepancy(self) -> None:
        """
        Computes a measure of reporting inconsistency between trading partners by
        comparing the exporter's reported FOB value with the importer's reported
        FOB value for the same trade flow. The discrepancy is normalized by total
        trade to make it comparable across different trade volumes.

        The formula is:
            reporting_discrepancy = |exports - imports| / (exports + imports)

        """
        self.df["reporting_discrepancy"] = (
            (abs(self.df["exports_const_usd"] - self.df["imports_const_usd"]))
            / (self.df["exports_const_usd"] + self.df["imports_const_usd"])
        ).fillna(0)

    def calculate_trade_share_percentages(self) -> None:
        """
        Calculate bilateral trade shares using averaged trade flows.

        Computes each trade flow's share of a country's total trade by first
        averaging reported export and import values (excluding zeros), then
        calculating what percentage this represents of each country's total trade.

        This method addresses reporting discrepancies by using the average of
        non-zero reported values rather than relying solely on one country's report.
        """
        self.df["trade_flow_average"] = (
            self.df[["exports_const_usd", "imports_const_usd"]]
            .replace(0, np.nan)
            .mean(axis=1)
        )
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

    def calculate_country_reporting_quality_metrics(self) -> None:
        """
        Calculate reporting quality metrics for each country as exporter and importer.

        Computes normalized measures of how consistently each country reports trade
        data compared to their trading partners. This helps identify countries with
        systematic reporting issues.

        For each country in both roles (exporter/importer), calculates:
            1. Number of active trade relationships (flows)
            2. Average reporting discrepancy across all partners
            3. Total normalized discrepancy (sum of discrepancies / number of flows)
        """
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
