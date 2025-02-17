import os
import sys
from glob import glob
import pandas as pd
import pyarrow as pq
from scipy.stats.mstats import winsorize
import numpy as np
from time import gmtime, strftime, localtime
import cProfile
import glob
from datetime import date, timedelta, datetime

from clean.objects.base import _AtlasCleaning
from clean.table_objects.aggregate_trade import AggregateTrade
from clean.utils import get_classifications, merge_classifications
from clean.table_objects.country_country_year import CountryCountryYear
from clean.table_objects.accuracy import Accuracy
from clean.table_objects.country_country_product_year import CountryCountryProductYear
from clean.table_objects.complexity import Complexity
from clean.objects.concordance_table import ConcordanceTable
from clean.table_objects.unilateral_services import UnilateralServices

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CIF_RATIO = 0.075

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)

ingestion_attrs = {
    "start_year": 1995,
    "end_year": 2023,
    "downloaded_files_path": "../../../../atlas/data/by_classification/aggregated_by_year/parquet",
    # "root_dir": "/Users/ELJ479/projects/atlas_cleaning/src",
    "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    "final_output_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2025_01_15/input",
    # used for comparison to atlas production data and generated data
    "comparison_file_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2024_11_18/input",
    "atlas_common_path": "/n/hausmann_lab/lab/atlas/atlas-common-data/atlas_common_data",
    "product_classification": "H0",
}

ingestion_attrs_H0 = {
    "start_year": 1995,
    "end_year": 2023,
    "downloaded_files_path": "../../../../atlas/data/by_classification/aggregated_by_year/parquet",
    "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    "final_output_path": f"/n/hausmann_lab/lab/atlas/data/rewrite_{(datetime.now() - timedelta(days=1)).strftime('%Y_%m_%d')}/input",
    "comparison_file_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2024_11_18/input",
    "atlas_common_path": "/n/hausmann_lab/lab/atlas/atlas-common-data/atlas_common_data",
    "product_classification": "H0",
}

ingestion_attrs_H4 = {
    "start_year": 2012,
    "end_year": 2023,
    "downloaded_files_path": "../../../../atlas/data/by_classification/aggregated_by_year/parquet",
    "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    "final_output_path": f"/n/hausmann_lab/lab/atlas/data/rewrite_{(datetime.now() - timedelta(days=1)).strftime('%Y_%m_%d')}/input",
    "comparison_file_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2024_11_18/input",
    "atlas_common_path": "/n/hausmann_lab/lab/atlas/atlas-common-data/atlas_common_data",
    "product_classification": "H4",
}

ingestion_attrs_H5 = {
    "start_year": 2017,
    "end_year": 2023,
    "downloaded_files_path": "../../../../atlas/data/by_classification/aggregated_by_year/parquet",
    "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    "final_output_path": f"/n/hausmann_lab/lab/atlas/data/rewrite_{(datetime.now() - timedelta(days=1)).strftime('%Y_%m_%d')}/input",
    "comparison_file_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2024_11_18/input",
    "atlas_common_path": "/n/hausmann_lab/lab/atlas/atlas-common-data/atlas_common_data",
    "product_classification": "H5",
}

ingestion_attrs_SITC = {
    "start_year": 1965,
    "end_year": 2023,
    "downloaded_files_path": "../../../../atlas/data/by_classification/aggregated_by_year/parquet",
    "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
    "final_output_path": f"/n/hausmann_lab/lab/atlas/data/rewrite_{(datetime.now() - timedelta(days=1)).strftime('%Y_%m_%d')}/input",
    "comparison_file_path": "/n/hausmann_lab/lab/atlas/data/rewrite_2024_11_18/input",
    "atlas_common_path": "/n/hausmann_lab/lab/atlas/atlas-common-data/atlas_common_data",
    "product_classification": "SITC",
}



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
        if product_classification == "SITC" and year > 1994:
            # use cleaned CCPY H0 data for SITC
            continue
        elif product_classification == "SITC":
            classifications = get_classifications(year)
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
        # if product_classification == "SITC":
        #     product_classification = get_classifications(year)[0]
        if product_classification=="SITC" and year > 1994:
            # use cleaned CCPY H0 data for SITC
            continue
            
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
            f"{product_classification}_{year}_country_country_product_year",
        )
        ccpy.save_parquet(ccpy.df, "final", f"{product_classification}_{year}")

        # handle SITC CCPY by running H0 through conversion table
        if product_classification == "H0":
            converted_table = ConcordanceTable(ccpy.df, product_classification, "S2")
            ccpy.save_parquet(
                converted_table.df,
                "processed",
                f"SITC_{year}_country_country_product_year",
            )
            ccpy.save_parquet(converted_table.df, "final", f"SITC_{year}", "SITC")
            del converted_table.df
        del ccpy.df

        # complexity files
        complexity = Complexity(year, **ingestion_attrs)

        complexity.save_parquet(
            complexity.df,
            "processed",
            f"{product_classification}_{year}_complexity",
        )
        del complexity.df

        logging.info(
            f"end time for {year}: {strftime('%Y-%m-%d %H:%M:%S', localtime())}"
        )

    complexity_all_years = glob.glob(
        f"data/processed/{product_classification}_*_complexity.parquet"
    )
    complexity_all = pd.concat(
        [pd.read_parquet(file) for file in complexity_all_years], axis=0
    )
    
    atlas_base_obj = _AtlasCleaning(**ingestion_attrs)
    atlas_base_obj.save_parquet(
        df=complexity_all,
        data_folder="processed",
        table_name=f"{product_classification}_complexity_all",
    )
    atlas_base_obj.save_parquet(
        complexity_all, "final", f"{product_classification}_cpy_all", "CPY"
    )
    del complexity_all
    
    # unilateral_services = UnilateralServices(**ingestion_attrs)
    # unilateral_services.save_parquet(
    #         unilateral_services.df,
    #         "final",
    #         f"unilateral_services",
    #         "Services"
    #     )


    # comparison = complexity.compare_files()
    # logging.info(f"review of compared files {comparison}")


def compute_distance(year, product_classification, dist):
    """
    based on distances compute cost of cif as a percentage of import_value_fob
    """
    df = pd.read_parquet(
        f"data/intermediate/{product_classification}_{year}_aggregated.parquet"
    )
    # lag and lead
    df_lag_lead = pd.DataFrame()
    for wrap_year in [year - 1, year + 1]:
        try:
            df_lag_lead = pd.read_parquet(
                f"data/intermediate/{product_classification}_{wrap_year}_aggregated.parquet"
            )
        except FileNotFoundError:
            logging.error(f"Didn't download year: {wrap_year}")

        df = pd.concat([df, df_lag_lead])

    dist.loc[dist["exporter"] == "ROU", "exporter"] = "ROM"
    dist.loc[dist["importer"] == "ROU", "exporter"] = "ROM"

    df = df.merge(dist, on=["exporter", "importer"], how="left")

    df.loc[df["exporter"] == "ROM", "exporter"] = "ROU"
    df.loc[df["importer"] == "ROM", "exporter"] = "ROU"

    df["lndist"] = np.log(df["distwces"])
    df.loc[(df["lndist"].isna()) & (df["dist"].notna()), "lndist"] = np.log(df["dist"])
    df["oneplust"] = df["import_value_cif"] / df["export_value_fob"]
    df["lnoneplust"] = np.log(df["import_value_cif"] / df["export_value_fob"])
    df["tau"] = np.nan

    compute_dist_df = df.copy(deep=True)
    # select the greater of the two, either the top 1% or 1,000,000
    exp_p1 = max(compute_dist_df["export_value_fob"].quantile(0.01), 10**6)
    imp_p1 = max(compute_dist_df["import_value_cif"].quantile(0.01), 10**6)
    # use to set min boundaries
    compute_dist_df = compute_dist_df[
        ~(compute_dist_df["export_value_fob"] < exp_p1)
        | (compute_dist_df["import_value_cif"] < imp_p1)
    ]
    compute_dist_df["lnoneplust"] = winsorize(
        compute_dist_df["lnoneplust"], limits=[0.1, 0.1]
    )
    compute_dist_df[["lnoneplust", "lndist", "contig"]] = compute_dist_df[
        ["lnoneplust", "lndist", "contig"]
    ].fillna(0.0)

    stata_code = """
        egen int idc_o = group(exporter)
        egen int idc_d = group(importer)
        reghdfe lnoneplust lndist contig, abs(year#idc_o year#idc_d)

        loc c = _coef[_cons]
        loc c_se = _se[_cons]
        loc beta_dist = _coef[lndist]
        loc se_dist = _se[lndist]
        loc beta_contig = _coef[contig]
        """
    output = run_stata_code(compute_dist_df, stata_code)
    # Extract the coefficients and standard errors
    coefficients = output["e(b)"][0]
    std_errors = np.sqrt(np.diag(output["e(V)"]))

    res = {
        "c": coefficients[2],
        "c_se": std_errors[2],
        "beta_dist": coefficients[0],
        "se_dist": std_errors[0],
        "beta_contig": coefficients[1],
    }

    df.loc[df["year"] == year, "tau"] = (
        res["c"]
        + (res["beta_dist"] * df["lndist"])
        + (res["beta_contig"] * df["contig"])
    )

    # clean up compute dist df
    del compute_dist_df

    df.loc[(df["year"] == year) & (df["tau"] < 0) & (df["tau"].notna()), "tau"] = 0
    df.loc[(df["year"] == year) & (df["tau"] > 0.2) & (df["tau"].notna()), "tau"] = 0.2
    tau_mean = df[df["year"] == year]["tau"].mean()
    df.loc[(df["year"] == year) & (df["tau"].isna()), "tau"] = tau_mean

    df = df[df.year == year]

    df.loc[abs(df["lnoneplust"]) < 0.05, "import_value_fob"] = df["import_value_cif"]

    df.loc[
        ((df["lnoneplust"] > 0) | (df["lnoneplust"].isna()))
        & (df["import_value_fob"].isna()),
        "import_value_fob",
    ] = df["import_value_cif"] * (1 - df["tau"])

    df.loc[
        (df["lnoneplust"] < 0) & (df["import_value_fob"].isna()), "import_value_fob"
    ] = df["import_value_cif"]

    return df[
        [
            "year",
            "exporter",
            "importer",
            "export_value_fob",
            "import_value_cif",
            "import_value_fob",
        ]
    ]


def run_stata_code(df, stata_code):
    # Initialize Stata
    sys.path.append("/n/sw/stata-17/utilities")
    from pystata import config

    config.init("se")
    from pystata import stata

    stata.pdataframe_to_data(df, force=True)
    stata.run(stata_code)
    return stata.get_ereturn()


if __name__ == "__main__":
    # run_atlas_cleaning(ingestion_attrs)
    run_atlas_cleaning(ingestion_attrs_H0)
    # run_atlas_cleaning(ingestion_attrs_SITC)
    # run_atlas_cleaning(ingestion_attrs_H4)
    # run_atlas_cleaning(ingestion_attrs_H5)
