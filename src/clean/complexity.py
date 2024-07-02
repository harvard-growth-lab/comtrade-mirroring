import pandas as pd
from clean.objects.base import _AtlasCleaning
import os
import numpy as np

import logging
import dask.dataframe as dd
import cProfile
from ecomplexity import ecomplexity
from ecomplexity import proximity



logging.basicConfig(level=logging.INFO)


# CCPY country country product year table
class complexity_calculations(_AtlasCleaning):


    def __init__(self, year, product_classification, **kwargs):
        super().__init__(**kwargs)
        
        # Import trade data from CID Atlas
        self.df = pd.read_parquet("data/intermediate/country_country_product_year.parquet")
        self.df = self.df[['year','exporter','commodity_code','export_value']]
        
        self.df = self.df.groupby(['year', 'exporter', 'commodity_code']).sum().reset_index()

        # col names for complexity package
        trade_cols = {'time':'year', 'loc':'exporter', 'prod':'commodity_code', 'val':'export_value'}
        
        # calculate complexity
        complexity_df = ecomplexity(self.df, trade_cols)

        # calculate proximity 
        proximity_df = proximity(self.df, trade_cols)
