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
    FORWARD_YEAR = 10
    FEATURES = ['ln_gdppc_const','pop_growth_10', 'nr_growth_10', 'eci', 'oppval',  'eci_oppval']
    DEPENDENT_VARIABLE = 'gdppc_growth_10'
    THRESHOLD = 2.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_loader = DataLoader(**kwargs)
        
        wdi = self.data_loader.load_wdi_data().rename(columns={"code": "exporter"})
        
        growth_rate_df = self.calc_growth_rate(wdi)
        pop_rate_df = self.calc_pop_growth()
        nr = self.data_loader.load_natural_resources()
        natres_rate_df = self.calc_natural_resources_per_cap(wdi, nr)
        
        nr.loc[:, 'eci_oppval']= nr['eci'] * nr['oppval']
        wdi.loc[:, 'ln_gdppc_const'] = np.log(wdi['gdppc_const'])
        
        self.df = wdi[['year','exporter','ln_gdppc_const']].merge(growth_rate_df, on=['year','exporter'], how='left').merge(pop_rate_df, on=['year','exporter'], how='left').merge(natres_rate_df, on=['year','exporter'], how='left').merge(nr[['year','exporter','eci', 'oppval', 'eci_oppval']], on=['year','exporter'], how='left')
        
        self.df = self.df[self.df.year>1969]
        self.df = self.filter_to_in_rankings()
        
        self.forecast_year= 2021
        self.df_res = pd.DataFrame()
        for digit_year in range(0,10):
            
            pred_df, X, y = self.select_regression_data(digit_year)
            X, y = self.remove_outliers(digit_year, X, y)
            results, X, y = self.run_growth_projection_regression(digit_year, X, y)
            self.predict_future_growth(digit_year, pred_df, results, X, y)
            
        import pdb
        pdb.set_trace()

        self.calc_aggregated_final_growth_projection()
        
        


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
        model = sm.OLS(y, X[['const'] + self.FEATURES + self.dummy_vars], missing='drop')
        res = model.fit()
        print(f"model results from finding outliers {res.summary()}")
        
        X.loc[:, 'predicted'] = res.predict(X[['const'] + self.FEATURES + self.dummy_vars])
        X.loc[:, 'predicted'] = res.predict(X[['const'] + self.FEATURES + self.dummy_vars])
        X.loc[:, 'abs_difference'] = abs(X['predicted']  - y)
        rmse = RMSE(X[self.DEPENDENT_VARIABLE], X['predicted'])
        X.loc[:, "exceeds_threshold"] = X['abs_difference'] > (rmse * self.THRESHOLD)
        X = X[~(X.exceeds_threshold==True)]

        y = X[self.DEPENDENT_VARIABLE]
        X = sm.add_constant(X)
        return X, y


    def run_growth_projection_regression(self, digit_year, X, y):
        self.reg_features = self.FEATURES.copy()
        self.reg_features.remove('pop_growth_10')
        gp_model = sm.OLS(y, X[['const'] + self.reg_features + self.dummy_vars], missing='drop')
#             model_index = gp_model.data.row_labels

        historical_X = X.loc[gp_model.data.row_labels]
        historical_y = y.loc[gp_model.data.row_labels]
        groups_used = historical_X['iso3_code']
        # res = gp_model.fit(cov_type='cluster', cov_kwds={'groups': X['iso3_code']})
        res = gp_model.fit(cov_type='cluster', cov_kwds={'groups': groups_used})        
        print(f"model results from running gp regression {res.summary()}")
        
        baseline_year = f"dummy_year_201{self.forecast_year}"
        # take the decade Fixed Effect of the latest dummy
        return res, historical_X, historical_y


    def predict_future_growth(self, digit_year, pred_df, res, X, y):
        coeff = res.params[f"dummy_year_201{digit_year}"]
        X.loc[:, 'predicted_gdppc'] = res.predict(X[['const'] + self.reg_features + self.dummy_vars])
        X.loc[:, 'abs_difference_gdppc'] = abs(X['predicted']  - y)
        rmse = RMSE(y, X['predicted_gdppc'])
        
        pred_df.loc[:, 'const'] = 1
        pred_df.loc[pred_df.year==2021, 'predicted_gdppc'] = res.predict(pred_df[['const'] + self.reg_features + self.dummy_vars]) + coeff
        # df = pd.concat([historical_X, pred_df])
        pred_df['digit_year'] = digit_year
        pred_df['r2'] = res.rsquared

        # Calculate total growth (tg)
        pred_df['total_growth'] = ((1 + pred_df['predicted_gdppc']/1) * (1 + pred_df['pop_growth_10']/1) - 1)
        dummy_cols = [col for col in pred_df.columns if col.startswith('dummy_')]
        pred_df = pred_df.drop(columns=dummy_cols)

        self.df_res = pd.concat([self.df_res, pred_df])
        
    def calc_aggregated_final_growth_projection(self):
        self.df_res['diff']=100*(self.df_res['gdppc_growth_10']-self.df_res['predicted_gdppc'])
        self.df_res.loc[self.df_res.year==2021,'diff']=0
        df.loc[df.year==2021,'temp1']=df['predicted_gdppc']
        df.loc[df.year==2021,'temp2']=df['pop_growth_10']
        df['point_est']=df.groupby(['digit_year','iso3_code'])['temp1'].transform('mean')
        df['pop_est']=df.groupby(['digit_year','iso3_code'])['temp2'].transform('mean')
        
        df['estimate'] = df['point_est'] + (df['diff']/100)
        
#         gen gdppoint = 100*(( 1+pointest/1)*(1+pop/1) -1 )
#         gen gdpestimate = 100*(( 1+estimate/1)*(1+pop/1) -1 )
#         replace laggrowthmkt10 = .  if year!=`fyear'
#         egen meangdpest = mean(gdpestimate), by(baseyear iso)
#         egen meanoverall = mean(gdpestimate), by( iso)
#         format gdp* diff* mean* laggrowthmkt10 pghat %9.3fc

#     cd $path
#     save "$path/atlas_growth_forecasts_$byear.dta", replace 	

        import pdb
        pdb.set_trace()
                 
    
    def filter_to_in_rankings(self):
        df = pd.read_parquet(Path(self.raw_data_path) / "growth_projection_countries.parquet")
        self.df = df.merge(self.df, left_on='iso3_code', right_on='exporter', how='left')
        return self.df[self.df.in_rankings==True]
        
    
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
        
        df = un_pop.merge(wdi_population, on=["exporter", "year"], how="left", suffixes=("_wdi", "_un"))
        
        # TODO: handle missing un population figures
        df['population_un'] = df.population_un.astype('float') 
        df['population_wdi'] = df.population_wdi.astype('float') 
        df.loc[df.population_un.isna(), 'population_un'] = df['population_wdi']
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

        df = df.reset_index()
        df[f"nr_growth_{self.FORWARD_YEAR}"] = df[f"nr_growth_{self.FORWARD_YEAR}"].fillna(0)
        return df[['year', 'exporter',f"nr_growth_{self.FORWARD_YEAR}"]]
    

    def complexity_metrics(self):
        """
        """
        pass
