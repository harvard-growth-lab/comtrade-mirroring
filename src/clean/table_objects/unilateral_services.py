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


class UnilateralServices(_AtlasCleaning):
    def __init__(self, year, **kwargs):
        
        START_YEAR = 1980
        
        super().__init__(**kwargs)
        
        self.load_wdi_services_data()
        self.df = self.df.fillna(0)
        self.df = self.df.rename(columns={'code': 'iso'})
        self.preprocess_flows("export")
        self.preprocess_flows("import")
        exports = self.isolate_trade_flows("export")
        imports = self.isolate_trade_flows("import")
        self.df = exports.merge(imports, on=['year','iso','services'],how='outer',suffixes=('_export','_import'))
        
                
        
    def load_wdi_services_data(self):
        """
        keep year idc iso ny_gdp_pcap_kd eci inatlas oppval
        """
        self.df = pd.read_csv(
            Path(self.atlas_common_path) / "wdi_indicators" / "data" / "wdi_service_data.csv"
        )
                              
                              
    def preprocess_flows(self, trade_flow):
        """
        """
        share = [col for col in self.df.columns if col.endswith(f'{trade_flow}_share')]
        for col in share:
            val = col.replace('share','value')
            self.df.loc[:, val] = ( self.df[col] / 100 ) * self.df[f'services_{trade_flow}_value']
        self.df.loc[:,f'unspecified_{trade_flow}_value'] = ((100 - self.df[share].sum(axis=1)) /100) * self.df[f'services_{trade_flow}_value']
        self.df.loc[self.df[f'unspecified_{trade_flow}_value']<0, f'unspecified_{trade_flow}_value'] = 0
        self.df = self.df.drop(columns=share)
        
        
    def isolate_trade_flows(self, trade_flow):
        df = self.df.copy()
        if trade_flow == 'export':
            cols = [col for col in df.columns if f"{trade_flow}" in col and col != f'services_export_value']
            renamed_cols = [col.replace(f'_export_value', '') for col in cols]
        else:
            cols = [col for col in df.columns if f"{trade_flow}" in col and col != f'services_import_value']
            renamed_cols = [col.replace(f'_import_value', '') for col in cols]
        df = df.rename(columns=dict(zip(cols, renamed_cols)))
        return pd.melt(df,id_vars=['iso', 'year'],value_vars=renamed_cols,var_name='services',value_name='value')
                              
                    
        
        
        
        
        
# https://data.worldbank.org/indicator/BX.GSR.NFSV.CD
        
# https://data.worldbank.org/indicator/BX.GSR.TRVL.ZS (travel and tourism services) %exports
# https://data.worldbank.org/indicator/BX.GSR.CCIS.ZS (ICT services) %exports
# https://data.worldbank.org/indicator/BX.GSR.INSF.ZS (insurance and financial) %exports
# https://data.worldbank.org/indicator/BM.GSR.TRAN.ZS (transport services) % imports
        
# https://data.worldbank.org/indicator/BM.GSR.CMCP.ZS
    
    
    
