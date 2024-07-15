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
        
        import pdb
        pdb.set_trace()
        
        self.df['reliable'] = self.df['exporter'].isin(reliable_exporters.exporter).astype(bool)
        # country specific 
        # reliable=False for "SYR", "GNQ", used to be HKG (now added to Atlas)
        # reliable=True for "ARM","BHR","CYP","MMR","SWZ","TGO","BFA" "COD","LBR","SDN","SGP"
        
        # totals by country
        # totals by product
        # save fulldata
                                          

        # separate into more reliable countries (~140 countries)

        # remove noisy commodity codes

        # col names for complexity package
        trade_cols = {
            "time": "year",
            "loc": "exporter",
            "prod": "commodity_code",
            "val": "export_value",
        }

        # calculate complexity
        complexity_df = ecomplexity(self.df, trade_cols)

        # calculate proximity
        proximity_df = proximity(self.df, trade_cols)

        # impute for all countries with product restrictions
        # presence of each country across products (m matrix for all countries)
        # calculate avg pci of a countries exports by using previous complexity matrix
        #
