import pandas as pd
from clean.table_objects.base import _AtlasCleaning
import os
import numpy as np
from sklearn.decomposition import PCA
import logging
import copy
from pathlib import Path
import requests
import datetime as datetime
import io

logging.basicConfig(level=logging.INFO)


class DataLoader(_AtlasCleaning):
    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)
        
    
    def load_natural_resources(self):
        """
        keep year iso eci oppval in_atlas diversity avubiquity total_exports
        """
        df = pd.read_parquet(
            Path(self.final_output_path) / "CPY" / "SITC_cpy_all.parquet"
        )
        nat_res = pd.read_stata(Path(self.raw_data_path) / "sitc_nrproducts.dta")
        nat_res.loc[nat_res.nrproducts.isna(), "nrproducts"] = 0

        df = df.merge(nat_res, on="commoditycode", how="left")

        df.loc[df.nrproducts == 1, "net_exports"] = (
            df["export_value"] - df["import_value"]
        ).clip(lower=0)

        df["total_exports"] = df.groupby(["year", "exporter"])[
            "export_value"
        ].transform("sum")
        df["diversity"] = df.groupby(["year", "exporter"])["mcp"].transform("count")
        df["ubiquity"] = df.groupby(["year", "commoditycode"])["mcp"].transform("count")
        df.loc[:, "avubiquity"] = df["ubiquity"] * df["mcp"] / df["diversity"]
        df["avubiquity"] = df.groupby(["year", "exporter"])["avubiquity"].transform(
            "sum"
        )

        df = (
            df.groupby(["year", "exporter"])
            .agg(
                {
                    "net_exports": "sum",
                    "eci": "first",
                    "oppval": "first",
                    "inatlas": "first",
                    "diversity": "first",
                    "avubiquity": "first",
                }
            )
            .reset_index()
        )

        df["eci_rank"] = df.groupby(["year"])["eci"].rank(ascending=False)
        # df.loc[df.inatlas==1, 'eci_rank_inatlas'] = df.groupby(["year"])['eci'].rank(ascending=False)
        df = df.rename(columns={"code":"exporter"})
        return df

    def load_wdi_data(self):
        """
        keep year idc iso ny_gdp_pcap_kd eci inatlas oppval
        """
        return pd.read_csv(
            Path(self.atlas_common_path) / "wdi_indicators" / "data" / "wdi_data.csv"
        )

    def load_population_forecast(self):
        """
        keep year idc iso ny_gdp_pcap_kd eci inatlas oppval
        """
        cols = {
            "Year": "year",
            "ISO3 Alpha-code": "iso",
            "Total Population, as of 1 January (thousands)": "population",
        }
        response = self.request_un_pop_data()
        pop_df = pd.read_excel(
            io.BytesIO(response.content),
            sheet_name="Estimates",
            header=16,
            usecols=cols.keys(),
        )
        pop_df = pop_df.rename(columns=cols)
        # pop_df["year"] = pop_df.year.astype("int16")
        projections_df = pd.read_excel(
            io.BytesIO(response.content),
            sheet_name="Medium variant",
            header=16,
            usecols=cols.keys(),
        )
        projections_df = projections_df.rename(columns=cols)

        df = pd.concat([pop_df, projections_df], axis=0)
        return df[~df.iso.isna()]

    def request_un_pop_data(self):
        """ """
        this_year = datetime.datetime.now().year
        for year in range(this_year - 2, this_year + 1):
            un_pop_download_link = f"https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/1_General/WPP{year}_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx"
            r = requests.get(un_pop_download_link)
            if r.status_code == 200:
                return r
        raise ValueError("Link for UN Population Forecasts is broken")

    def handle_natural_resource_complexity(self, nat_res):
        df = (
            df.groupby(["year", "iso"])
            .agg(
                {
                    "export_value": "sum",
                    "eci": "first",
                    "oppval": "first",
                    "in_atlas": "first",
                    "diversity": "first",
                    "avubiquity": "first",
                }
            )
            .reset_index()
        )

        return wdi
