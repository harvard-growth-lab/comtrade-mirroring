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
        
        wdi = self.data_loader.load_wdi_data().rename(columns={"code": "exporter"})
        
        growth_rate_df = self.calc_growth_rate(wdi)
        pop_rate_df = self.calc_pop_growth()
        nr = self.data_loader.load_natural_resources()
        natres_rate_df = self.calc_natural_resources_per_cap(wdi, nr)
        
        nr.loc[:, 'eci_oppval']= nr['eci'] * nr['oppval']
        wdi.loc[:, 'ln_gdppc_const'] = np.log(wdi['gdppc_const'])
        
        self.df = wdi[['year','exporter','ln_gdppc_const']].merge(growth_rate_df, on=['year','exporter'], how='left').merge(pop_rate_df, on=['year','exporter'], how='left').merge(natres_rate_df, on=['year','exporter'], how='left').merge(nr[['year','exporter','eci', 'oppval', 'eci_oppval']], on=['year','exporter'], how='left')
        
        # self.df = self.df[self.df.year>1969]
        
        
        # import pdb
        # pdb.set_trace()
        
        
        


    def detect_outliers(self):
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
        for digit_year in range(0,10):
            # hold on to only years ending in digit_year
            df_filtered = filter_year_pattern(df, digit_year, forecast_year)
                 
    
    
    def filter_year_pattern(df, j, forecast_year):
        return df[
            (df['year'] == 1960 + j) |
            (df['year'] == 1970 + j) |
            (df['year'] == 1980 + j) |
            (df['year'] == 1990 + j) |
            (df['year'] == 2000 + j) |
            (df['year'] == 2010 + j) |
            (df['year'] == forecast_year)
        ]


    def calc_growth_rate(self, wdi):
        """
        stata's growth10
        """
        df = wdi[['year', 'exporter','gdppc_const']].set_index('year')
        # gen deltaNRrealexports= ( (f10.nnrexppc/f10.deflactor) - (nnrexppc/deflactor) ) / ny_gdp_pcap_kd
        df.loc[:, f"gdppc_growth_{self.FORWARD_YEAR}"] = (df.groupby('exporter')["gdppc_const"].shift(-10)/df['gdppc_const']) ** (1 / 10) - 1
        return df.reset_index()

    def calc_pop_growth(self):
        """
        stata's pghat
        """
        wdi_population = self.data_loader.load_wdi_data()
        wdi_population = wdi_population[["code", "year", "population"]].rename(columns={"code": "exporter"})
        un_pop = self.data_loader.load_population_forecast()
        un_pop = un_pop.rename(columns={"iso":"exporter"})
        
        df = wdi_population.merge(un_pop, on=["exporter", "year"], how="left", suffixes=("_wdi", "_un"))
        # TODO: handle missing un population figures
        df = df[['year', 'exporter','population_un']].set_index('year')
        # gen pghat =1* ( (f10.pop_hat/pop_hat)^(1/10)-1)
        df.loc[:, f"pop_growth_{self.FORWARD_YEAR}"] = 1 * (
                (
                    df.groupby('exporter')["population_un"].shift(-10)
                    / df["population_un"]
                )
                ** (1 / self.FORWARD_YEAR)
                - 1
            )
        
        return df.reset_index()

    
    def calc_natural_resources_per_cap(self, wdi, nr):
        """
        stata's deltaNRrealexports
        calculates natural resource 10-change in exports normalized by gdp per cap
        """
        df = nr.merge(
            wdi, on=["year", "exporter"], how="left"
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
        df = df[['year', 'exporter', 'nat_res_exportspc', 'gdppc_const' ,'deflator']].set_index('year')
        df = df.sort_values(['year','exporter'])

        df.loc[:, f"nr_growth_{self.FORWARD_YEAR}"] = (
                (
                    df.groupby('exporter')["nat_res_exportspc"].shift(-10)
                    / df.groupby('exporter')["deflator"].shift(-10)
                )
                - (
                    df["nat_res_exportspc"]
                    / df["deflator"]
                )
            ) / df["gdppc_const"]

        return df.reset_index()
    

    def complexity_metrics(self):
        """
        """
        pass
