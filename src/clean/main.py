import os
import sys
import argparse
from glob import glob
import pandas as pd
import pyarrow as pq
from scipy.stats.mstats import winsorize
import numpy as np
from time import gmtime, strftime, localtime
import cProfile
import glob
from datetime import date, timedelta, datetime
from clean.utils.CIF_calculations import compute_distance

from clean.objects.base import _AtlasCleaning
from clean.table_objects.aggregate_trade import AggregateTrade
from clean.utils.classification_handler import get_classifications, merge_classifications
from clean.table_objects.country_country_year import CountryCountryYear
from clean.table_objects.accuracy import Accuracy
from clean.table_objects.country_country_product_year import CountryCountryProductYear
from clean.table_objects.complexity import Complexity
from clean.objects.concordance_table import ConcordanceTable
from clean.table_objects.unilateral_services import UnilateralServices

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CIF_RATIO = 0.075

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)

parser = argparse.ArgumentParser(description="Run the Atlas clean")
parser.add_argument(
    "-V", "--version", type=str, help="Data folder name to store cleaned output data"
)

parser.add_argument(
    "-D",
    "--download-type",
    type=str,
    help="from comtrade get data type: [as_reported, by_classification]",
)

args = parser.parse_args()

data_version = (
    args.version
    if args.version
    else f"rewrite_{(date.today() - timedelta(days=1)).strftime('%Y_%m_%d')}"
)

download_type = args.download_type if args.download_type else "by_classification"


# use and manipulate to run sections interactively
ingestion_attrs = {
    "start_year": 2015,
    "end_year": 2020,
    "downloaded_files_path": f"../../../../atlas/data/{download_type}/aggregated_by_year/parquet",
    # "root_dir": "/Users/ELJ479/projects/atlas_cleaning/src",
    "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    "final_output_path": f"/n/hausmann_lab/lab/atlas/data/{data_version}/input",
    # used for comparison to atlas production data and generated data
    "comparison_file_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2024_11_18/input",
    "atlas_common_path": "/n/hausmann_lab/lab/atlas/atlas-common-data/atlas_common_data",
    "product_classification": "H0",
    "download_type": "by_classification",
}

ingestion_attrs_base = {
    "downloaded_files_path": f"../../../../atlas/data/by_classification/aggregated_by_year/parquet",
    "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    "final_output_path": f"/n/hausmann_lab/lab/atlas/data/{data_version}/input",
    "comparison_file_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2024_11_18/input",
    "atlas_common_path": "/n/hausmann_lab/lab/atlas/atlas-common-data/atlas_common_data",
    "download_type": "by_classification",
}


ingestion_attrs_converted_base = {
    "downloaded_files_path": f"../../../../atlas/data/as_reported/converted_aggregated_by_year/parquet",
    "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    "final_output_path": f"/n/hausmann_lab/lab/atlas/data/{data_version}/input",
    "comparison_file_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2024_11_18/input",
    "atlas_common_path": "/n/hausmann_lab/lab/atlas/atlas-common-data/atlas_common_data",
    "download_type": "as_reported",
}


ingestion_attrs_H0 = {
    "start_year": 1995,
    "end_year": 2023,
    "product_classification": "H0",
}

ingestion_attrs_H4 = {
    "start_year": 2012,
    "end_year": 2023,
    "product_classification": "H4",
}

ingestion_attrs_H5 = {
    "start_year": 2017,
    "end_year": 2023,
    "product_classification": "H5",
}

ingestion_attrs_SITC = {
    "start_year": 1962,
    "end_year": 2023,
    "product_classification": "SITC",
}

general_ingestion_attrs = {
    "start_year": None,
    "end_year": 2023,
    "product_classification": None,
}

# ingestion_attrs_temp = {
#     "start_year": 1995,
#     "end_year": 2023,
#     "product_classification": "SITC",
# }


def run_atlas_cleaning(ingestion_attrs):
    """
    Run the Bustos-Yildirm method to generate reliable trade data. Builds and saves input tables for
    Atlas ingestion

    Parameters:
    - ingestion_attrs (dict): A dictionary containing attributes necessary for ingestion.
        Required keys:
            - start_year (int): The latest year of data.
            - end_year (int): Data coverage from the latest year.
            - root_dir (str): root directory path
    """
    print(f"start time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")
    start_year = ingestion_attrs["start_year"]
    end_year = ingestion_attrs["end_year"]
    product_classification = ingestion_attrs["product_classification"]
    downloaded_files_path = ingestion_attrs["downloaded_files_path"]

    # load data
    dist = pd.read_stata(os.path.join("data", "raw", "dist_cepii.dta"))
    
    for year in range(start_year, end_year + 1):
        if product_classification == "SITC" and year > 1994 and download_type == "by_classification":
            # use cleaned CCPY H0 data for SITC
            continue
        elif product_classification == "SITC" and download_type == "by_classification":
            classifications = get_classifications(year)
        elif product_classification == "SITC" and download_type == "as_reported":
            classifications = ["S2"]
        else:
            classifications = [product_classification]

        logging.info(
            f"Aggregating data for {year} and these classifications {classifications}"
        )
        [
            AggregateTrade(year, product_class, **ingestion_attrs)
            for product_class in classifications
        ]

    logging.info(f"Completed data aggregations")

    for year in range(start_year, end_year + 1):
        logging.info(f"Beginning {year}... for {product_classification}")
        if product_classification == "SITC" and year > 1994 and download_type == "by_classification":
            # use cleaned CCPY H0 data for SITC
            continue
        elif product_classification == "SITC" and download_type == "by_classification":
            product_classification = get_classifications(year)[0]


        # compute distance requires three years of aggregated data
        logging.info(f"Beginning compute distance for year {year}")
        df = compute_distance(year, product_classification, dist)

        # country country year intermediate file, passed into CCY object
        df.to_parquet(
            os.path.join(
                ingestion_attrs["root_dir"],
                "data",
                "intermediate",
                f"{product_classification}_{year}.parquet",
            ),
            index=False,
        )
        del df

        ccy = CountryCountryYear(year, **ingestion_attrs)

        ccy.save_parquet(
            ccy.df,
            "intermediate",
            f"{product_classification}_{year}_country_country_year",
        )
        del ccy.df

        accuracy = Accuracy(year, **ingestion_attrs)
        logging.info("confirm CIF ratio column is present")

        accuracy.save_parquet(
            accuracy.df, "intermediate", f"{product_classification}_{year}_accuracy"
        )
        del accuracy.df

        ccpy = CountryCountryProductYear(year, **ingestion_attrs)

        ccpy.save_parquet(
            ccpy.df,
            "processed",
            f"{download_type}_{product_classification}_{year}_country_country_product_year",
        )
        ccpy.save_parquet(ccpy.df, "final", f"{product_classification}_{year}")

        if product_classification == "SITC" and year <= 1975 and download_type == "by_classification":
            sitcv2_ccpy = ConcordanceTable(ccpy.df, "S1", "S2")
            ccpy.save_parquet(
                sitcv2_ccpy.df,
                "processed",
                f"{download_type}_SITC_{year}_country_country_product_year",
            )
            ccpy.save_parquet(sitcv2_ccpy.df, "final", f"SITC_{year}", "SITC")
            del sitcv2_ccpy.df


        # handle SITC CCPY by running H0 through conversion table
        elif product_classification == "H0" and download_type == "by_classification":
            sitc_ccpy = ConcordanceTable(ccpy.df, product_classification, "S2")
            ccpy.save_parquet(
                sitc_ccpy.df,
                "processed",
                f"{download_type}_SITC_{year}_country_country_product_year",
            )
            ccpy.save_parquet(sitc_ccpy.df, "final", f"SITC_{year}", "SITC")
            del sitc_ccpy.df
        del ccpy.df

    # handle complexity
    for year in range(start_year, end_year + 1):
        # complexity files
        complexity = Complexity(year, **ingestion_attrs)

        complexity.save_parquet(
            complexity.df,
            "processed",
            f"{download_type}_{product_classification}_{year}_complexity",
        )
        del complexity.df

        logging.info(
            f"end time for {year}: {strftime('%Y-%m-%d %H:%M:%S', localtime())}"
        )

    complexity_all_years = glob.glob(
        f"data/processed/{download_type}_{product_classification}_*_complexity.parquet"
    )
    complexity_all = pd.concat(
        [pd.read_parquet(file) for file in complexity_all_years], axis=0
    )

    atlas_base_obj = _AtlasCleaning(**ingestion_attrs)

    atlas_base_obj.save_parquet(
        complexity_all, "final", f"{product_classification}_cpy_all", "CPY"
    )
    del complexity_all


def run_unilateral_services(ingestion_attrs):
    unilateral_services = UnilateralServices(**ingestion_attrs)
    unilateral_services.save_parquet(
        unilateral_services.df, "final", f"unilateral_services", "Services"
    )
    del unilateral_services.df



    # comparison = complexity.compare_files()
    # logging.info(f"review of compared files {comparison}")




if __name__ == "__main__":
    # for testing sections and manipulating the attrs directly
    # run_atlas_cleaning(ingestion_attrs)
    
    if data_version == "as_reported":
    
        ingestion_attrs_H0.update(ingestion_attrs_converted_base)
        ingestion_attrs_SITC.update(ingestion_attrs_converted_base)
        ingestion_attrs_H4.update(ingestion_attrs_converted_base)
        general_ingestion_attrs.update(ingestion_attrs_converted_base)
        
    elif data_version == "by_classification":
        
        ingestion_attrs_H0.update(ingestion_attrs_base)
        ingestion_attrs_SITC.update(ingestion_attrs_base)
        ingestion_attrs_H4.update(ingestion_attrs_base)
        ingestion_attrs_H5.update(ingestion_attrs_base)
        general_ingestion_attrs.update(ingestion_attrs_base)


    run_atlas_cleaning(ingestion_attrs_H0)
    run_atlas_cleaning(ingestion_attrs_H4)
    run_atlas_cleaning(ingestion_attrs_SITC)
    # run_atlas_cleaning(ingestion_attrs_H5)
    run_atlas_cleaning(ingestion_attrs)
    
    run_unilateral_services(general_ingestion_attrs)
