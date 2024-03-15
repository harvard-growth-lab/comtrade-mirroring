import sys
from os import path
import pandas as pd
from sys import argv
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

from clean.objects.base import _AtlasCleaning


class TradeAggregator(_AtlasCleaning):
    def __init__(self, df, year, **kwargs):
        super().__init__(**kwargs)

        self.df = df
        self.year = year
        self.clean_data()

        if self.product_classification in ["H0", "HS"]:
            self.detail_level = 6
        elif self.product_classification in ["S1", "S2", "ST"]:
            self.detail_level = 4
        else:
            logging.info("incorrect classification")
        self.aggregate_data()
        
        exp_to_world, imp_to_world = self.to_world()
        self.df = self.df[(self.df['importer'] != "WLD") & (self.df['exporter'] != "WLD")]
        assert exp_to_world['exporter'].is_unique
        self.df = self.df.merge(exp_to_world, on = "exporter", how="left")
        assert imp_to_world['importer'].is_unique
        self.df = self.df.merge(imp_to_world, on = "importer", how="left")
        self.filter_data()
        self.df['year'] = self.year
        self.df = self.df[['year', 'exporter', 'importer', 'exportvalue_fob', 'importvalue_cif']]
        #TODO format fob and cif
        # self.df[['exportvalue_fob', 'importvalue_cif']] = self.df[['exportvalue_fob', 
                                                                   # 'importvalue_cif']].apply(lambda x: '{:15.0f}'.format(x) if pd.notnull(x) else '')
        self.df = self.df.sort_values(by=['exporter', 'importer'])


    def clean_data(self):
        """ """
        logging.info(f"Cleaning.. > {self.year} and classification = {self.product_classification}")

        # Data manipulation based on 'classification'
        # This is a simplification, adapt based on actual logic and conditions in Stata script
        if self.product_classification in ["H0", "HS"]:
            # Add "00" prefix to 'commoditycode' where the length of 'commoditycode' is 4 and 'product_level' is 6
            self.df["commodity_code"] = np.where(
                (self.df["commodity_code"].str.len() == 4)
                & (self.df["product_level"] == 6),
                "00" + self.df["commodity_code"],
                self.df["commodity_code"],
            )
            # Add "0" prefix to 'commoditycode' where the length of 'commoditycode' is 5 and 'product_level' is 6
            self.df["commodity_code"] = np.where(
                (self.df["commodity_code"].str.len() == 5)
                & (self.df["product_level"] == 6),
                "0" + self.df["commodity_code"],
                self.df["commodity_code"],
            )
            self.df["reporter_ansnoclass"] = self.df.trade_value.where(
                (self.df.partner_iso == "ANS")
                & (self.df.product_level == 4)
                & (self.df.commodity_code.str.slice(0, 4) == "9999")
            )

        elif self.product_classification in ["S1", "S1", "ST"]:
            # Add "00" prefix to 'commoditycode' where the length of 'commoditycode' is 2 and 'product_level' is 4
            self.df["commodity_code"] = np.where(
                (self.df["commodity_code"].str.len() == 2)
                & (self.df["product_level"] == 4),
                "00" + self.df["commodity_code"],
                self.df["commodity_code"],
            )
            # Add "0" prefix to 'commoditycode' where the length of 'commoditycode' is 3 and 'product_level' is 4
            self.df["commodity_code"] = np.where(
                (self.df["commodity_code"].str.len() == 3)
                & (self.df["product_level"] == 4),
                "0" + self.df["commodity_code"],
                self.df["commodity_code"],
            )
            #TODO follow up if this filter for anything
            self.df["reporter_ansnoclass"] = self.df.trade_value.where(
                (self.df.partner_iso == "ANS")
                & (self.df.product_level == 4)
                & (self.df.commodity_code == "9310")
            )

        # handles Germany (reunification) and Russia
        # drop if reporter and partner are DEU and DDR, trading with itself
        self.df = self.df[
            ~((self.df["reporter_iso"] == "DEU") & (self.df["partner_iso"] == "DDR"))
        ]
        self.df = self.df[
            ~((self.df["reporter_iso"] == "DDR") & (self.df["partner_iso"] == "DEU"))
        ]
        self.df.loc[self.df["partner_iso"].isin(["DEU", "DDR"]), "partner_iso"] = "DEU"
        self.df.loc[
            self.df["reporter_iso"].isin(["DEU", "DDR"]), "reporter_iso"
        ] = "DEU"
        self.df.loc[self.df["partner_iso"].isin(["RUS", "SUN"]), "partner_iso"] = "RUS"
        self.df.loc[
            self.df["reporter_iso"].isin(["RUS", "SUN"]), "reporter_iso"
        ] = "RUS"

        # compress
        # collapse (sum) trade_value reporter_ansnoclas , by( year trade_flow product_level reporter_iso partner_iso )
        self.df = (
            self.df.groupby(
                ["year", "trade_flow", "product_level", "reporter_iso", "partner_iso"]
            )
            .agg({"trade_value": "sum", "reporter_ansnoclass": "sum"})
            .reset_index()
        )
        # recast float trade_value reporter_ansnoclas, force
        self.df["trade_value"] = self.df["trade_value"].astype(
            "float"
        )
        self.df["reporter_ansnoclass"] = self.df[
            "reporter_ansnoclass"
        ].astype("float")
        
        
    def aggregate_data(self):
        """
        """
        dfs = []
        for level in [0, self.detail_level]:
            df = self.df[self.df.product_level == level]
            df = df[["reporter_iso", "partner_iso", "trade_value", "reporter_ansnoclass", "trade_flow"]]

            # trade_flow column becomes two cols indicating the trade_flow's trade_value
            df = df.pivot_table(index=['reporter_iso', 'partner_iso'], 
                                      columns='trade_flow', 
                                      values=['trade_value', 'reporter_ansnoclass'], 
                                      fill_value=0).reset_index()
            df.columns = ['_'.join(str(i) for i in col).rstrip('_') 
                                    if col[1] else col[0] for col in df.columns.values]

            # table for reporter who is an exporter
            reporting_exporter = df[['reporter_iso', 'partner_iso', 'trade_value_2', 'reporter_ansnoclass_2']]
            reporting_exporter = reporting_exporter.rename(columns={'reporter_iso': 'exporter', 
                                                                    'partner_iso': 'importer',
                                                                    'trade_value_2': f'exports_{level}',
                                                                    'reporter_ansnoclass_2':
                                                                    f'exp2ansnoclass_{level}'})

            # table for reporter who is an importer
            reporting_importer = df[['reporter_iso', 'partner_iso', 'trade_value_1', 'reporter_ansnoclass_1']]
            reporting_importer = reporting_importer.rename(columns={'reporter_iso': 'importer', 
                                                                    'partner_iso': 'exporter',
                                                                    'trade_value_1': f'imports_{level}',
                                                                    # TODO: why is this called imp2 not imp1
                                                                    'reporter_ansnoclass_1': 
                                                                    f'imp2ansnoclass_{level}'})
            #merge 1:1 exporter importer using `temp', nogen
            if reporting_importer.duplicated(subset=['importer', 'exporter']).any():
                print("reporting exporter is not a unique pair")
            if reporting_exporter.duplicated(subset=['importer', 'exporter']).any():
                print("reporting importer is not a unique pair")
            merged_df = reporting_importer.merge(reporting_exporter, on=['importer', 'exporter'])
            # egen temp = rowtotal( imports_six imp2ansnoclass_six exports_six exp2ansnoclass_six )
            merged_df['temp'] = merged_df[[f'imports_{level}', 
                                           f'imp2ansnoclass_{level}', 
                                           f'exports_{level}', f'exp2ansnoclass_{level}']].sum(axis=1)
            # drops rows summing to zero 
            merged_df = merged_df[merged_df['temp'] != 0]
            merged_df.drop(columns='temp', inplace=True)
            dfs.append(merged_df)
            
        self.df = dfs[0].merge(dfs[1], on=['importer', 'exporter'])
        cols = ['exporter', 'importer', f'exports_0', 
                f'exports_{self.detail_level}', f'exp2ansnoclass_{self.detail_level}', 
                'imports_0', f'imports_{self.detail_level}', f'imp2ansnoclass_{self.detail_level}']
        self.df = self.df[cols]
    
    
    def to_world(self):
        """
        """
        exp_to_world = self.df[self.df.importer == "WLD"]  
        exp_to_world = exp_to_world[['exporter', 'exports_0']]
        exp_to_world = exp_to_world.rename(columns={'exports_0': 'exp2WLD'})
        imp_to_world = self.df[self.df.exporter == "WLD"]  
        imp_to_world = imp_to_world[['importer', 'imports_0']]
        imp_to_world = imp_to_world.rename(columns={'imports_0': 'imp2WLD'})
        return exp_to_world, imp_to_world
                            
                            
    def filter_data(self):
        """
        """
        self.df['ratio_exp'] = self.df[f'exp2ansnoclass_{self.detail_level}'] / self.df['exp2WLD']
        self.df['ratio_imp'] = self.df[f'imp2ansnoclass_{self.detail_level}'] / self.df['imp2WLD']
                            
        self.df.loc[self.df['ratio_exp'] > 0.25, 'exports_0'] = self.df['exports_0'] - self.df[f'exp2ansnoclass_{self.detail_level}']
        self.df.loc[self.df['ratio_exp'] > 0.25, f'exports_{self.detail_level}'] = self.df[f'exports_{self.detail_level}'] - self.df[f'exp2ansnoclass_{self.detail_level}']
        self.df.loc[self.df['ratio_imp'] > 0.25, 'imports_0'] = self.df['imports_0'] - self.df[f'exp2ansnoclass_{self.detail_level}']
        #TODO: code substracts exp2ansnoclass_six not imp2ansnoclass_six?
        self.df.loc[self.df['ratio_imp'] > 0.25, f'imports_{self.detail_level}'] = self.df[f'imports_{self.detail_level}'] - self.df[f'exp2ansnoclass_{self.detail_level}']

        self.df['exportvalue_fob'] = self.df[['exports_0', f'exports_{self.detail_level}']].mean(axis=1)
        self.df['importvalue_cif'] = self.df[['imports_0', f'imports_{self.detail_level}']].mean(axis=1)
        self.df['temp'] = self.df[['imports_0', f'imports_{self.detail_level}']].max(axis=1)
        self.df = self.df[self.df['temp'] != 0]
        self.df = self.df[self.df['temp'] >1_000]