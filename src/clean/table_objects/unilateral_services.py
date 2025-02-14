import pandas as pd
from clean.objects.base import _AtlasCleaning
from clean.objects.load_data import DataLoader

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
    SERVICES_START_YEAR = 1980
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.data_loader = DataLoader(**kwargs)
        
        self.df = self.data_loader.load_wdi_services_data()
        self.df = self.df.fillna(0)
        self.df = self.df.rename(columns={'code': 'iso'})
        self.preprocess_flows("export")
        self.preprocess_flows("import")
        exports = self.isolate_trade_flows("export")
        imports = self.isolate_trade_flows("import")
        self.df = exports.merge(imports, on=['year','iso','services'],how='outer',suffixes=('_export','_import'))
        self.df = self.df.rename(columns={"value_export":"export_value","value_import":"import_value"})
        products = self.combine_services_and_goods()
        self.handle_complexity()
                                      
                              
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
    
    def combine_services_and_goods(self):
        """
        """
        goods = self.data_loader.load_sitc_cpy_complexity()
        grouped_goods = goods.groupby(['year','exporter']).agg({'export_value':'sum'})
        grouped_goods.loc[:,'total_exports']=grouped_goods.groupby('year')['export_value'].transform('sum')
        grouped_goods.loc[:,'wtshare']=(grouped_goods['export_value']/grouped_goods['total_exports'])*100
        grouped_goods= grouped_goods.drop(columns='total_exports').reset_index()
        
        services = self.df.copy()
        services['totals'] = services[services['services'] != 'unspecified'].groupby(['year', 'iso'])['export_value'].transform('sum')
        services.loc[:,'share']=services['export_value']/services['totals']
        services['sumshare'] = services.groupby(['year','iso'])['share'].transform('sum')
        # following stata code
        services['nodata'] = services['totals'] > 1
        services=services.drop_duplicates(subset=['iso','year'], keep='first')
        services = services.drop(columns=['services','export_value','import_value','share']).rename(columns={'iso':"exporter"})
        
        df = grouped_goods.merge(services, on=['exporter','year'],how='left', suffixes=('_goods','_services'))
        del grouped_goods, services
        return df
        
    def handle_complexity(self):
        # self.df = self.df.set_index('year')
        for year in range(self.SERVICES_START_YEAR, self.end_year + 1):
            lag_year = year - 1
            df = self.df[self.df.year.isin([year, lag_year])].copy()
            df.loc[df.export_value==0, 'export_value'] = np.nan
            df = df.groupby(['iso','services']).agg({"export_value":"mean"})
            df['year'] = year
            df=df.reset_index().rename(columns={'iso':'exporter'})
            df=df[~(df.services=="unspecified")]
            df = df.rename(columns={"services":"commodity_code"})
            # dataservices 
            avg_service_exports = df.copy()
            df = df.groupby(['exporter','year']).agg({"export_value":"sum"})
            df=df.rename(columns={"export_value":"total_services"})
            df=df[~((df.total_services.isna()) | (df.total_services==0))]
            
            goods = self.data_loader.load_sitc_cpy_complexity()
            goods = goods[goods.year.isin([year, lag_year])].copy()
            goods=goods[~((goods.commoditycode=="XXXX") | (goods.commoditycode=="9310"))]
            goods = goods[['exporter', 'commoditycode', 'export_value', 'pci']].rename(columns={"pci":"old_pci"})
            goods.loc[:,'commoditycode'] = goods['commoditycode'].str[:2]
            goods = goods.groupby(['exporter','commoditycode']).agg({"export_value":"sum","old_pci":"mean"})
            
            # identify small flows
            tot_goods = goods.groupby(['commoditycode']).agg({"export_value":"sum"})
            tot_goods = tot_goods.sort_values(by='export_value')
            tot_goods['cumulative_sum'] = tot_goods['export_value'].cumsum()
            tot_goods['share'] = 100* (tot_goods['cumulative_sum'] / tot_goods['export_value'].max())
            small_flows =tot_goods[(tot_goods.share<.05)]['commoditycode'].unique().tolist()
            del tot_goods
            
            goods = goods[~(goods.commoditycode.isin(small_flows))]
            goods['year'] = year
            
            
            import pdb
            pdb.set_trace()
            
            
        
        
                              
                    
        
        
        
        
        
# https://data.worldbank.org/indicator/BX.GSR.NFSV.CD
        
# https://data.worldbank.org/indicator/BX.GSR.TRVL.ZS (travel and tourism services) %exports
# https://data.worldbank.org/indicator/BX.GSR.CCIS.ZS (ICT services) %exports
# https://data.worldbank.org/indicator/BX.GSR.INSF.ZS (insurance and financial) %exports
# https://data.worldbank.org/indicator/BM.GSR.TRAN.ZS (transport services) % imports
        
# https://data.worldbank.org/indicator/BM.GSR.CMCP.ZS
    
    
    
