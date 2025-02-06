import pandas as pd
from clean.table_objects.base import _AtlasCleaning
from clean.table_objects.load_data import DataLoader
import os
import numpy as np
from sklearn.decomposition import PCA
import logging
import copy
import numpy

logging.basicConfig(level=logging.INFO)


class GrowthProjections(_AtlasCleaning):
    FORWARD_YEAR = 10

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)

        self.year = year
        self.data_loader = DataLoader(self.year, **kwargs)

    def regression_forecast(self, forecast_year=8):
        """
        for regression need:
            - gdppc_const (lnypc)
            - growth10
            - deltaNRrealexports
            - pghat
            - eci
            - oppval
            - eci_opp
        """
        # take natural log of gdppc_const
        # only atlas
        # only keep years, ending in forecast year (60 - 2010s)
        # gen growth10 = 1 * (  (gdppc_const_year_2018/gdppc_const_year_2008)^(1/10)-1)
        wdi.set_index("year", inplace=True)
        wdi.reset_index(inplace=True)

    def calc_growth_rate(self, df):
        """
        stata's growth10
        """
        wdi = self.data_loader.load_wdi_data().rename(columns={"code": "exporter"})
        growth_df = pd.DataFrame()
        if self.year + 10 < 2022:
            # gen deltaNRrealexports= ( (f10.nnrexppc/f10.deflactor) - (nnrexppc/deflactor) ) / ny_gdp_pcap_kd
            pivot_df = wdi.pivot_table(
                index=["exporter"], columns="year", values=["gdppc_const"]
            )

            growth_df[self.year] = (
                pivot_df["gdppc_const"][self.year + self.FORWARD_YEAR]
                / pivot_df["gdppc_const"][self.year]
            ) ** (1 / 10) - 1
            growth_df = growth_df.reset_index().rename(
                columns={self.year: f"gdppc_growth_{self.FORWARD_YEAR}"}
            )
            growth_df["year"] = self.year
        else:
            growth_df.loc[
                growth_df.year == self.year, f"gdppc_growth_{self.FORWARD_YEAR}"
            ] = np.nan

    def calc_pop_growth(self):
        """
        stata's pghat
        """
        wdi = self.data_loader.load_wdi_data()
        wdi = wdi[["code", "year", "population"]].rename(columns={"code": "iso"})
        un_pop = self.data_loader.load_population_forecast()
        df = wdi.merge(un_pop, on=["iso", "year"], how="left", suffixes=("_wdi", "_un"))
        # gen pghat =1* ( (f10.pop_hat/pop_hat)^(1/10)-1)
        pop_growth_df = pd.DataFrame()
        if self.year + 10 < 2022:
            pivot_df = df.pivot_table(
                index=["iso"], columns="year", values=["population_un"]
            )
            pop_growth_df[self.year] = 1 * (
                (
                    pivot_df["population_un"][self.year + self.FORWARD_YEAR]
                    / pivot_df["population_un"][self.year]
                )
                ** (1 / self.FORWARD_YEAR)
                - 1
            )
            pop_growth_df = pop_growth_df.reset_index().rename(
                columns={self.year: f"pop_growth_{self.FORWARD_YEAR}"}
            )
            pop_growth_df["year"] = self.year
        else:
            pop_growth_df["year"] = self.year
            pop_growth_df.loc[
                pop_growth_df.year == self.year, f"pop_growth_{self.FORWARD_YEAR}"
            ] = np.nan
        return pop_growth_df

    def calc_natural_resources_per_cap(self):
        """
        stata's deltaNRrealexports
        calculates natural resource 10-change in exports normalized by gdp per cap
        """
        wdi = self.data_loader.load_wdi_data()
        nr = self.data_loader.load_natural_resources()
        df = nr.merge(
            wdi, left_on=["year", "exporter"], right_on=["year", "code"], how="left"
        )
        # gen nnrexppc= nr_net_exports/ (ny_gdp_mktp_cd/ny_gdp_pcap_cd)
        df.loc[:, "nat_res_exportspc"] = df["net_exports"] / (df["gdp"] / df["gdppc"])
        # net_nat_res_exp_perc_to_gdppc = (nr_net_exports / gdppc) / population
        df.loc[:, "net_nat_res_exp_perc_to_gdppc"] = (
            df["net_exports"] / df["gdppc"]
        ) / df["population"]
        # log_net_nat_resrouce_ppc = ln(nr_net_exports / (gdp / gdppc))
        df.loc[:, "ln_net_nat_res_exp_perc_to_gdppc"] = np.log(
            df["net_exports"] / (df["gdp"] / df["gdppc"])
        )

        df.loc[:, "deflator"] = df["gdppc"] / df["gdppc_const"]
        nat_res_change_df = pd.DataFrame()

        if self.year + 10 < 2022:
            # gen deltaNRrealexports= ( (f10.nnrexppc/f10.deflactor) - (nnrexppc/deflactor) ) / ny_gdp_pcap_kd
            pivot_df = df.pivot_table(
                index=["exporter"],
                columns="year",
                values=["nat_res_exportspc", "deflator", "gdppc_const"],
            )

            nat_res_change_df[self.year] = (
                (
                    pivot_df["nat_res_exportspc"][self.year + self.FORWARD_YEAR]
                    / pivot_df["deflator"][self.year + self.FORWARD_YEAR]
                )
                - (
                    pivot_df["nat_res_exportspc"][self.year]
                    / pivot_df["deflator"][self.year]
                )
            ) / pivot_df["gdppc_const"][self.year]

            nat_res_change_df = nat_res_change_df.reset_index().rename(
                columns={self.year: "nat_res_change"}
            )
            nat_res_change_df["year"] = self.year

        else:
            nat_res_change_df["year"] = self.year
            nat_res_change_df.loc[
                nat_res_change_df.year == self.year, "nat_res_change"
            ] = np.nan
        return nat_res_change_df
    

    def complexity_metrics(self):
        """
        """
        pass
