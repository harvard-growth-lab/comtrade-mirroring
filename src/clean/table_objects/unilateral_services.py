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
from ecomplexity import ecomplexity


logging.basicConfig(level=logging.INFO)


class UnilateralServices(_AtlasCleaning):
    SERVICES_START_YEAR = 1980
    SERVICES = ['comms', 'finance', 'transport', 'travel']
    
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
        self.complexity = self.handle_complexity()
        self.update_pci_to_avg()
        
        import pdb
        pdb.set_trace()
                                      
                              
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
        services_complexity = pd.DataFrame()
        for year in range(self.SERVICES_START_YEAR, self.end_year + 1):
            lag_year = year - 1
            df = self.df[self.df.year.isin([year, lag_year])].copy()
            df.loc[df.export_value==0, 'export_value'] = np.nan
            df = df.groupby(['iso','services']).agg({"export_value":"mean"})
            df['year'] = year
            df=df.reset_index().rename(columns={'iso':'exporter'})
            df=df[~(df.services=="unspecified")]
            df = df.rename(columns={"services":"commoditycode"})
            # dataservices 
            avg_service_exports = df.copy()
            df = df.groupby(['exporter','year']).agg({"export_value":"sum"}).reset_index()
            df=df.rename(columns={"export_value":"total_services"})
            df=df[~((df.total_services.isna()) | (df.total_services==0))]
            
            goods = self.data_loader.load_sitc_cpy_complexity()
            goods = goods[goods.year.isin([year, lag_year])].copy()
            goods=goods[~((goods.commoditycode=="XXXX") | (goods.commoditycode=="9310"))]
            goods = goods[['exporter', 'commoditycode', 'export_value', 'pci']].rename(columns={"pci":"old_pci"})
            goods.loc[:,'commoditycode'] = goods['commoditycode'].str[:2]
            goods = goods.groupby(['exporter','commoditycode']).agg({"export_value":"sum","old_pci":"mean"}).reset_index()
            
            # identify small flows
            tot_goods = goods.groupby(['commoditycode']).agg({"export_value":"sum"}).reset_index()
            tot_goods = tot_goods.sort_values(by='export_value')
            tot_goods['cumulative_sum'] = tot_goods['export_value'].cumsum()
            tot_goods['share'] = 100* (tot_goods['cumulative_sum'] / tot_goods['export_value'].max())
            small_flows =tot_goods[(tot_goods.share<.05)]['commoditycode'].unique().tolist()
            del tot_goods
            
            goods = goods[~(goods.commoditycode.isin(small_flows))]
            goods['year'] = year
            
            products = pd.concat([goods, avg_service_exports], axis=0)
            df = products.merge(df, on=['exporter','year'], how='inner')
            df['nflows']=df.groupby('exporter')['export_value'].transform('count')
            df = df[~(df.nflows<=5)]
            
            # handle complexity
            # handle nans?
            # py-ecomplexity package requires time variable
            df['year'] = 10
            trade_cols ={'loc':'exporter', 'prod':'commoditycode', 'val':'export_value', 'time':'year'}
            cdata = ecomplexity(df, trade_cols)
            
            df = cdata.groupby('commoditycode').agg({"pci":"first","old_pci":"first", "ubiquity":"first"}).reset_index()
            df['nexporters'] = cdata['exporter'].nunique()
            
            mean_pci = df.describe()['pci'].loc['mean']
            std_pci= df.describe()['pci'].loc['std']
            df.loc[:, 'pci'] = (df['pci'] - mean_pci) / std_pci
            
            # ask Seba,validate above a threshold?
            corr = df[['pci','old_pci']].corr()
            
            # keep commoditycode pci  oldpci ubiquity
            df = df[df.commoditycode.isin(self.SERVICES)]
            df = df.drop(columns='old_pci')
            df['year'] = year
                        
            services_complexity = pd.concat([services_complexity, df])
        return services_complexity
    
    def update_pci_to_avg(self):
        self.complexity = self.complexity.set_index(['year', 'commoditycode']).drop(columns='nexporters')
        self.complexity['pci_3yr_avg'] = self.complexity.groupby('commoditycode')['pci'].transform(lambda x: (x + x.shift(1) + x.shift(-1))/3)
        self.complexity=self.complexity.reset_index()
        
        self.df = self.df.rename(columns={"services":"commoditycode"})
        self.df = self.df.merge(self.complexity, on=['year','commoditycode'], how='outer')
        self.df['export_value'] = self.df.export_value.fillna(0)
        self.df['import_value'] = self.df.import_value.fillna(0)        
        
                              
                    
        
        
        
        
        
# https://data.worldbank.org/indicator/BX.GSR.NFSV.CD
        
# https://data.worldbank.org/indicator/BX.GSR.TRVL.ZS (travel and tourism services) %exports
# https://data.worldbank.org/indicator/BX.GSR.CCIS.ZS (ICT services) %exports
# https://data.worldbank.org/indicator/BX.GSR.INSF.ZS (insurance and financial) %exports
# https://data.worldbank.org/indicator/BM.GSR.TRAN.ZS (transport services) % imports
        
# https://data.worldbank.org/indicator/BM.GSR.CMCP.ZS
    
    
    
