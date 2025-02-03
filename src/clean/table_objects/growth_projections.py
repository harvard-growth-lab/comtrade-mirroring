import pandas as pd
from clean.table_objects.base import _AtlasCleaning
import os
import numpy as np
from sklearn.decomposition import PCA
import logging
import copy

logging.basicConfig(level=logging.INFO)


class GrowthProjections(_AtlasCleaning):
    
        def __init__(self, year, **kwargs):
        super().__init__(**kwargs)
        
        
        def regression_forecast(self, forecast_year=8):
            """
            for regression need: 
                - gdppc_const (lnypc)
                - growth10  
                - deltaNRrealexports
                - eci 
                - oppval 
                - pghat 
                - eci_opp
                
            """
            # take natural log of gdppc_const
            # only atlas
            # only keep years, ending in forecast year (60 - 2010s)
            # gen growth10 = 1 * (  (gdppc_const_year_2018/gdppc_const_year_2008)^(1/10)-1)
            wdi.set_index('year', inplace=True)
            wdi.loc[:, 'growth10'] = ((wdi['gdppc_const'].shift(-10) / wdi['gdppc_const']) ** (1/10) - 1)
            wdi.reset_index(inplace=True)
            
            
        
        
        
        
        
        
        
        


        