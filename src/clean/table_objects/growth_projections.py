import pandas as pd
from clean.objects.base import _AtlasCleaning
from clean.objects.load_data import DataLoader
import os
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import logging
import copy
import numpy
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse as RMSE



logging.basicConfig(level=logging.INFO)


class GrowthProjections(_AtlasCleaning):
    # FORECAST_YEAR = 2023
    FORWARD_YEAR = 10
    FEATURES = ['ln_gdppc_const','pop_growth_10', 'nr_growth_10', 'eci', 'oppval',  'eci_oppval']
    DEPENDENT_VARIABLE = 'gdppc_growth_10'
    THRESHOLD = 2.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.forecast_year = self.end_year
        self.data_loader = DataLoader(**kwargs)
        
        imf = self.data_loader.load_imf_data().rename(columns={"code": "iso3_code"})
        # CUBA missing from IMF pull in from WDI
        wdi = self.data_loader.load_wdi_data().rename(columns={"code": "iso3_code"})
        
        self.world_indicators = pd.concat([wdi, imf[imf.iso3_code=="TWN"]])
        
        # validated
        growth_rate_df = self.calc_growth_rate()
        
        # validated
        pop_rate_df = self.calc_pop_growth()
        
        nr = self.data_loader.load_natural_resources().rename(columns={"exporter": "iso3_code"})
        # validated
        natres_rate_df = self.calc_natural_resources_per_cap(nr)
        
        
        nr.loc[:, 'eci_oppval']= nr['eci'] * nr['oppval']
        self.world_indicators.loc[:, 'ln_gdppc_const'] = np.log(self.world_indicators['gdppc_const'])
        
        self.df = self.world_indicators[['year','iso3_code','ln_gdppc_const']].merge(growth_rate_df, on=['year','iso3_code'], how='left').merge(pop_rate_df, on=['year','iso3_code'], how='left').merge(natres_rate_df, on=['year','iso3_code'], how='left').merge(nr[['year','iso3_code','eci', 'oppval', 'eci_oppval']], on=['year','iso3_code'], how='left')
        
        self.df = self.df[self.df.year>1969]
        self.df = self.filter_to_in_rankings(self.df)
        
        self.df_res = pd.DataFrame()
        for digit_year in range(0,10):
            
            pred_df, X, y = self.select_regression_data(digit_year)

            X, y = self.remove_outliers(digit_year, X, y)

            results, X, y = self.run_growth_projection_regression(digit_year, X, y)
            
            self.predict_future_growth(digit_year, pred_df, results, X, y)

        self.calc_aggregated_final_growth_projection()
        self.append_forecast_year_growth()
        
        


    def select_regression_data(self, digit_year):
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
        # hold on to only years ending in digit_year

        # data selection
        dff = self.filter_year_pattern(digit_year)
        print(f"shape of filtered data frame {dff.shape}")
        dff['year'] = dff.year.astype('int')
        dff['dummy_year'] = dff.loc[:, 'year']
        dff = pd.get_dummies(dff, columns=['dummy_year'], drop_first=True, dtype='int')
        # forecast year data
        pred_df = dff[dff.year == self.forecast_year]

        X = dff[dff.year.astype(str).str.endswith(str(digit_year)) | (dff.year == self.forecast_year)].copy()
        y = X[self.DEPENDENT_VARIABLE]

        # get data that will be used in regression 
        X = sm.add_constant(X)
        model = sm.OLS(y, X[['const'] + self.FEATURES], missing='drop')
        print(f"model results from identifying data sample {model.fit().summary()}")
        
        X = X[X.index.isin(model.data.row_labels) | (X.year == self.forecast_year)]
        return pred_df, X, X[self.DEPENDENT_VARIABLE]
            
    def remove_outliers(self, digit_year, X, y):
        
        self.dummy_vars = [#f"dummy_year_197{digit_year}", 
              f"dummy_year_198{digit_year}", 
              f"dummy_year_199{digit_year}",
              f"dummy_year_200{digit_year}",
              f"dummy_year_201{digit_year}"]
        #self.dummy_vars.remove(f"dummy_year_197{digit_year}")

        X = sm.add_constant(X)
        
        res = sm.OLS(y, X[['const'] + self.FEATURES + self.dummy_vars], missing='drop').fit()
        print(f"model results from finding outliers {res.summary()}")
        
        valid_obs = ~np.isnan(y) & ~X[['const'] + self.FEATURES + self.dummy_vars].isna().any(axis=1)
        X.loc[valid_obs, 'predicted'] = res.predict(X.loc[valid_obs, ['const'] + self.FEATURES + self.dummy_vars])
        X.loc[valid_obs, 'abs_difference'] = abs(X.loc[valid_obs, 'predicted'] - X.loc[valid_obs, self.DEPENDENT_VARIABLE])
        rmse = np.sqrt(res.mse_resid)

        X.loc[:, "exceeds_threshold"] = X['abs_difference'] > (rmse * self.THRESHOLD)
        
        # stata finds 9 outside of threshold and python finds 15, all of 9 inclusive in 15
        X = X[~(X.exceeds_threshold==True)]

        y = X[self.DEPENDENT_VARIABLE]
        X = sm.add_constant(X)
        return X, y


    def run_growth_projection_regression(self, digit_year, X, y):
        self.reg_features = self.FEATURES.copy()
        self.reg_features.remove('pop_growth_10')
        gp_model = sm.OLS(y, X[['const'] + self.reg_features + self.dummy_vars], missing='drop')

        historical_X = X.loc[gp_model.data.row_labels]
        historical_y = y.loc[gp_model.data.row_labels]
        groups_used = historical_X['iso3_code']
        res = gp_model.fit(cov_type='cluster', cov_kwds={'groups': groups_used})        
        print(f"model results from running gp regression {res.summary()}")
        
        return res, historical_X, historical_y


    def predict_future_growth(self, digit_year, pred_df, res, X, y):
        # TODO determine appropriate baseline year
        try:
            baseline = res.params.get(f"dummy_year_201{digit_year}", 0)
            if baseline == 0:
                baseline = res.params.get(f"dummy_year_200{digit_year}", 0)
        except:
            baseline = 0
        
        X.loc[:, 'predicted_gdppc'] = res.predict(X[['const'] + self.reg_features + self.dummy_vars])
        X.loc[:, 'abs_difference_gdppc'] = abs(X['predicted']  - y)
        X['digit_year'] = digit_year
        rmse = RMSE(y, X['predicted_gdppc'])
        
        pred_df.loc[:, 'const'] = 1
        
        pred_df.loc[pred_df.year==self.forecast_year, 'predicted_gdppc'] = res.predict(pred_df[['const'] + self.reg_features + self.dummy_vars]) + baseline

        pred_df.loc[pred_df.year==self.forecast_year, 'predicted_gdppc'] = res.predict(pred_df[['const'] + self.reg_features + self.dummy_vars]) + baseline
        pred_df['digit_year'] = digit_year
        
        self.df_res = pd.concat([self.df_res, X, pred_df])
        dummy_cols = [col for col in self.df_res.columns if col.startswith('dummy_')]
        self.df_res = self.df_res.drop(columns=dummy_cols)
        

    def calc_aggregated_final_growth_projection(self):
        
        self.df_res['diff']=100*(self.df_res['gdppc_growth_10']-self.df_res['predicted_gdppc'])
        self.df_res.loc[self.df_res['year']==self.forecast_year,'diff']=0
        self.df_res.loc[self.df_res['year'] == self.forecast_year, 'temp1'] = self.df_res.loc[self.df_res['year'] == self.forecast_year, 'predicted_gdppc']
        self.df_res.loc[self.df_res['year'] == self.forecast_year, 'temp2'] = self.df_res.loc[self.df_res['year'] == self.forecast_year, 'pop_growth_10']
        
        self.df_res['point_est']=self.df_res.groupby(['digit_year','iso3_code'])['temp1'].transform('mean')
        
        self.df_res['pop_est']=self.df_res.groupby(['digit_year','iso3_code'])['temp2'].transform('mean')
                
        self.df_res['estimate'] = self.df_res['point_est'] + (self.df_res['diff']/100)
        self.df_res['gdppoint'] = 100 * ((1 + self.df_res['point_est'] ) * ( 1 + self.df_res['pop_est']) - 1)
        self.df_res['gdpestimate'] = 100 * ((1 + self.df_res['estimate'] ) * ( 1 + self.df_res['pop_est']) - 1)
        # replace laggrowthmkt10 = .  if year!=`fyear'
        self.df_res['meangdpest']=self.df_res.groupby(['digit_year','iso3_code'])['gdpestimate'].transform('mean')
        self.df_res['meanoverall']=self.df_res.groupby(['iso3_code'])['gdpestimate'].transform('mean')
        
        
    def append_forecast_year_growth(self):
        growth_proj = pd.read_csv(Path(self.atlas_common_path) / "growth_projections" / "growth_projections.csv")
        if self.forecast_year in growth_proj.year.unique():
            growth_proj = growth_proj[~growth_proj.year==self.forecast_year]
        forecast_year = self.df_res[self.df_res.year==self.forecast_year][['iso3_code', 'meanoverall', 'year']].drop_duplicates(subset=['iso3_code', 'meanoverall', 'year'])
        forecast_year = forecast_year.rename(columns={"iso3_code":"abbrv", 
                                                      "meanoverall": "growth_proj",
                                                      "year":"year"})
        
        self.df = pd.concat([growth_proj, forecast_year])
                               
    
    def filter_to_in_rankings(self, df):
        rank = pd.read_csv(
            Path(self.atlas_common_path) / "static_files" / "data" / "in_rankings.csv"
        )
        df = df.merge(rank, on='iso3_code', how='left')
        return df[df.in_rankings==True]
        
    
    def filter_year_pattern(self, digit_year):
        return self.df[
            (self.df['year'] == 1960 + digit_year) |
            (self.df['year'] == 1970 + digit_year) |
            (self.df['year'] == 1980 + digit_year) |
            (self.df['year'] == 1990 + digit_year) |
            (self.df['year'] == 2000 + digit_year) |
            (self.df['year'] == 2010 + digit_year) |
            (self.df['year'] == self.forecast_year)
        ]


    def calc_growth_rate(self):
        """
        stata's growth10
        """
        df = self.world_indicators[['year', 'iso3_code','gdppc_const']].set_index('year')
        # gen deltaNRrealexports= ( (f10.nnrexppc/f10.deflactor) - (nnrexppc/deflactor) ) / ny_gdp_pcap_kd
        df.loc[:, f"gdppc_growth_{self.FORWARD_YEAR}"] = (df.groupby('iso3_code')["gdppc_const"].shift(-10)/df['gdppc_const']) ** (1 / 10) - 1
        return df.reset_index()

    def calc_pop_growth(self):
        """
        stata's pghat
        """
        population = self.world_indicators[["iso3_code", "year", "population"]].rename(columns={"code": "iso3_code"})
        un_pop = self.data_loader.load_population_forecast()
        un_pop['population'] = un_pop['population'] * 1_000
        un_pop = un_pop.rename(columns={"iso":"iso3_code"})
        
        df = un_pop.merge(population, on=["iso3_code", "year"], how="left", suffixes=("_imf", "_un"))
        
        df['population_un'] = df.population_un.astype('float') 
        df['population_imf'] = df.population_imf.astype('float') 
        df.loc[df.population_un.isna(), 'population_un'] = df['population_imf']
        
        df = df[['year', 'iso3_code','population_un']].set_index('year')
        # gen pghat =1* ( (f10.pop_hat/pop_hat)^(1/10)-1)
                        
        df[f"pop_growth_{self.FORWARD_YEAR}"] = df.groupby('iso3_code')['population_un'].transform(lambda x: ((x.shift(-10)/x)**(1/10) -1))
        
        return df.reset_index()

    
    def calc_natural_resources_per_cap(self, nr):
        """
        stata's deltaNRrealexports
        calculates natural resource 10-change in exports normalized by gdp per cap
        """
        df = nr.merge(
            self.world_indicators, on=["year", "iso3_code"], how="left"
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
        df = df[['year', 'iso3_code', 'nat_res_exportspc', 'gdppc_const' ,'deflator']].set_index('year')
        df = df.sort_values(['year','iso3_code'])

        df.loc[:, f"nr_growth_{self.FORWARD_YEAR}"] = (
                (
                    df.groupby('iso3_code')["nat_res_exportspc"].shift(-10)
                    / df.groupby('iso3_code')["deflator"].shift(-10)
                )
                - (
                    df["nat_res_exportspc"]
                    / df["deflator"]
                )
            ) / df["gdppc_const"]

        df = df.reset_index()
        df[f"nr_growth_{self.FORWARD_YEAR}"] = df[f"nr_growth_{self.FORWARD_YEAR}"].fillna(0)
        return df[['year', 'iso3_code',f"nr_growth_{self.FORWARD_YEAR}"]]