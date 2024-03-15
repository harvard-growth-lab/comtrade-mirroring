import logging
from os import path

from clean.objects.base import _AtlasCleaning
from clean.load_data import DataLoader
from clean.aggregate_trade import TradeAggregator

logging.basicConfig(level=logging.INFO)


def run_atlas_cleaning(ingestion_attrs):
    """
    Run the Bustos-Yildirm method to generate reliable trade data. Builds and saves input tables for
    Atlas ingestion

    Parameters:
    - ingestion_attrs (dict): A dictionary containing attributes necessary for ingestion.
        Required keys:
            - start_year (int): The latest year of data.
            - end_year (int): Data coverage from the latest year.
            - product_classification (str)
            - root_dir (str): root directory path
            - data_dir (str): data is stored across three folders: raw, intermediate, processed
    """
    product_classification = ingestion_attrs['product_classification']
    for year in range(start_year, end_year + 1):
        # load data set for one year
        DL = DataLoader(year, **ingestion_attrs).save_parquet("intermediate", 
                                                              f"{product_classification}_{year}")
        #TODO: load in parquet file
        TA = TradeAggregator(DL.df, year, **ingestion_attrs)
        

    # Step 1, Do File
    # Aggregate and clean up country data
    # CountryProductYear("1", **ingestion_attrs).save_parquet("country_product_year_1")
    # CountryProductYear("2", **ingestion_attrs).save_parquet("country_product_year_2")


if __name__ == "__main__":
    ingestion_attrs = {
        "start_year": 2015,
        "end_year": 2015,
        "product_classification": "H0",
        "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    }
    run_atlas_cleaning(ingestion_attrs)
