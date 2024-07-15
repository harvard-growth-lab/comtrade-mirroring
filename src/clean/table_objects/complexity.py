import pandas as pd
from clean.table_objects.base import _AtlasCleaning
import os
import numpy as np

import logging
import dask.dataframe as dd
import cProfile
from ecomplexity import ecomplexity
from ecomplexity import proximity


logging.basicConfig(level=logging.INFO)


# complexity table
class Complexity(_AtlasCleaning):
    NOISY_TRADE = {"S2": ["9310","9610","9710","9999","XXXX"],
                    "H0": ["7108","9999","XXXX"],
                    "H4": ["7108","9999","XXXX"]}

    
    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)
        self.product_class = kwargs["product_classification"]
        self.year = year
        
        # load data 
        aux_stats = pd.read_csv(os.path.join(self.raw_data_path, "auxiliary_statistics.csv"), sep="\t")
        reliable_exporters = pd.read_stata(os.path.join(self.raw_data_path, "obs_atlas.dta"))
        # Import trade data from CID Atlas
    
        self.df = pd.read_parquet(
            f"data/processed/country_country_product_{self.year}.parquet"
        )
        self.df = self.df[["exporter", "importer", "commodity_code", "final_value"]]
        self.df = self.df.rename(columns={"final_value": "export_value"})
        # aggregate to four digit level
        self.df['commodity_code'] = self.df['commodity_code'].astype(str).str[:4]
        
        # show the import and export value for each exporter
        self.df = (
            self.df.groupby(["exporter", "importer", "commodity_code"]).sum().reset_index()
        )
        imports = self.df.copy(deep=True)
        imports = imports.drop(columns=['exporter']).rename(columns={"export_value": "import_value",
                                                                    "importer": "exporter"})
        imports = imports.groupby(["exporter", "commodity_code"]).sum().reset_index()
        
        self.df = self.df.drop(columns=["importer"])
        self.df = self.df.groupby(["exporter", "commodity_code"]).sum().reset_index()
        self.df = self.df.merge(imports, on=["exporter", "commodity_code"], how="left")
        
        # fillin all combinations of exporter and commodity code
        
        self.df = self.df.merge(aux_stats[['exporter', 'population', 'gdp_pc']], on=["exporter"], how="left")
        self.df = self.df.fillna(0.0)
        
        self.df['reliable'] = self.df['exporter'].isin(reliable_exporters.exporter).astype(bool)
        # reliable=False for "SYR", "GNQ", used to be HKG (now added to Atlas)
        # reliable=True for "ARM","BHR","CYP","MMR","SWZ","TGO","BFA" "COD","LBR","SDN","SGP"
        
        total_by_country = self.df[['exporter', 'export_value']].groupby('exporter').agg('sum').reset_index()
        total_by_commodity = self.df[['commodity_code', 'export_value']].groupby('commodity_code').agg('sum').reset_index()
        drop_countries = total_by_country[total_by_country.export_value==0.0]['exporter'].tolist() 
        drop_commodities = total_by_commodity[total_by_commodity.export_value==0.0]['commodity_code'].tolist()
        if drop_countries or drop_commodities:
            self.df = self.df[~self.df.exporter.isin(drop_countries)]
            self.df = self.df[~self.df.commodity_code.isin(drop_commodities)]
            
        # save all countries, 207 countries
        # self.save_parquet(self.df, "intermediate", "all_countries")
        
        # only reliable countries, subset of 123 countries
        self.df = self.df[self.df.reliable==True]
        
        self.df = self.df.groupby(['exporter', 'commodity_code']).agg({
                'export_value': 'sum',
                'population': 'first',
                'gdp_pc': 'first'
            }).reset_index()
        
        # drop unknown trade
        self.df = self.df[~self.df.commodity_code.isin(self.NOISY_TRADE[self.product_classification])]
        
        # mcp matrix, rca of 1 and greater
        self.df['by_commodity_code'] = self.df.groupby('commodity_code')['export_value'].transform('sum')
        self.df['by_exporter'] = self.df.groupby('exporter')['export_value'].transform('sum')
        # rca calculation, a commodity's percentage of a country export basket in comparison to the 
        # export value for the product in global trade 
        self.df['rca'] = (self.df['export_value'] / self.df['by_exporter']) / (self.df['by_commodity_code'] / self.df['export_value'].sum())

        self.df['mcp'] = np.where(self.df['rca'] >= 1, 1, 0)
        
        import pdb
        pdb.set_trace()
        
        # filter data
        # Herfindahl-Hirschman Index Calculation
        self.df['HH_index'] = (self.df['export_value'] / self.df.groupby('commodity_code')['export_value'].transform('sum'))**2
        self.df = self.df.groupby('commodity_code').agg('sum')
        export_totals = self.df.export_value.sum()
        self.df['share'] = 100 * (self.df['export_value'] / self.df.export_value.sum())
        effective_exporters = 1 / self.df['HH_index']
        

        
        # pass mcp matrix into Shreyas's ecomplexity package
        
        
        # col names for complexity package
        trade_cols = {
            "time": "year",
            "loc": "exporter",
            "prod": "commodity_code",
            "val": "export_value",
        }

        # calculate complexity
        complexity_df = ecomplexity(self.df[['year', 'exporter', 'commodity_code', 'export_value']], 
                                    trade_cols,
                                   presence_test="manual")

        # calculate proximity
        proximity_df = proximity(self.df, trade_cols)

        # impute for all countries with product restrictions
        # presence of each country across products (m matrix for all countries)
        # calculate avg pci of a countries exports by using previous complexity matrix
        #
