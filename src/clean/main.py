import logging
import os
import sys
from glob import glob
import pandas as pd
from scipy.stats.mstats import winsorize
import numpy as np
from time import gmtime, strftime, localtime
import cProfile


from clean.table_objects.base import _AtlasCleaning
from clean.table_objects.aggregate_trade import AggregateTrade
from clean.utils import get_classifications, merge_classifications
from clean.table_objects.country_country_year import CountryCountryYear
from clean.table_objects.accuracy import Accuracy
from clean.table_objects.country_country_product_year import CountryCountryProductYear
from clean.table_objects.complexity import Complexity

logging.basicConfig(level=logging.INFO)
CIF_RATIO = 0.075

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)


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
    downloaded_files_path = ingestion_attrs['downloaded_files_path']
    
    # load data 
    dist = pd.read_stata(os.path.join('data', 'raw', "dist_cepii.dta"))
    
    for year in range(start_year - 1, end_year + 2):
        # file_name = f"data/intermediate/cleaned_{product_classification}_{year}.parquet"
        # if not os.path.isfile(file_name):
        
        # get possible classifications based on year
        logging.info("removing classifications step until can confirm with Seba")
        classifications = [product_classification]
        # classifications = get_classifications(year)
        logging.info(f"Aggregating data for {year} and these classifications {classifications}")
        [AggregateTrade(year, product_class, **ingestion_attrs) for product_class in classifications]
    
    logging.info("Completed data aggregations, starting next loop")
    for year in range(start_year, end_year + 1):
        # depending on year, merge multiple classifications, take median of values
        df = merge_classifications(year, ingestion_attrs["root_dir"])
        # compute distance requires three years of aggregated data
        logging.info(f"Beginning compute distance for year {year}")
        df = compute_distance(df, year, product_classification, dist)
        
        os.makedirs(
            os.path.join(
                ingestion_attrs["root_dir"], "data", "intermediate", product_classification
            ),
            exist_ok=True,
        )
        # country country year intermediate file, passed into CCY object
        df.to_parquet(
            os.path.join(
                ingestion_attrs["root_dir"],
                "data",
                "intermediate",
                product_classification,
                f"{product_classification}_{year}.parquet",
            ),
            index=False,
        )
        ccy = CountryCountryYear(year, **ingestion_attrs)
        ccy.save_parquet(ccy.df, 'processed', f'country_country_year_{year}')
        
        accuracy = Accuracy(year, **ingestion_attrs)
        logging.info("confirm CIF ratio column is present")
        accuracy.save_parquet(accuracy.df, 'processed', f'accuracy_{year}')
        
        # cProfile.run('run_ccpy()')
        # cProfile.run('CountryCountryProductYear(year, **ingestion_attrs)')
        # logging.info("finished running cprofile")
        ccpy = CountryCountryProductYear(year, **ingestion_attrs)
        ccpy.save_parquet(ccpy.df, 'processed', f'country_country_product_{year}')

        # complexity files
        print(f"end time: {strftime('%Y-%m-%d %H:%M:%S', localtime())}")

        
def run_ccpy():
    return CountryCountryProductYear(year, **ingestion_attrs)


def compute_distance(df, year, product_classification, dist):
    """
    based on distances compute cost of cif as a percentage of import_value_fob
    """
    # lag and lead
    df_lag_lead = pd.DataFrame()
    for wrap_year in [year - 1, year + 1]:
        try:
            df_lag_lead = pd.read_parquet(f"data/intermediate/aggregated_{product_classification}_{wrap_year}.parquet")
        except FileNotFoundError:
            logging.error(f"Didn't download year: {wrap_year}")
        df = pd.concat([df, df_lag_lead])

    dist.loc[dist["exporter"] == "ROU", "exporter"] = "ROM"
    dist.loc[dist["importer"] == "ROU", "exporter"] = "ROM"

    df = df.merge(dist, on=["exporter", "importer"], how="left")
    
    df.loc[df["exporter"] == "ROM", "exporter"] = "ROU"
    df.loc[df["importer"] == "ROM", "exporter"] = "ROU"


    df["lndist"] = np.log(df["distwces"])
    df.loc[(df['lndist'].isna()) & (df['dist'].notna()), 'lndist'] = np.log(df['dist'])
    df["oneplust"] = df["import_value_cif"] / df["export_value_fob"]
    df["lnoneplust"] = np.log(df["import_value_cif"] / df["export_value_fob"])
    df["tau"] = np.nan
        
    # select the greater of the two, either the top 1% or 1,000,000
    exp_p1 = max(df["export_value_fob"].quantile(0.01), 10**6)
    imp_p1 = max(df["import_value_cif"].quantile(0.01), 10**6)
    # use to set min boundaries
    df = df[~
        (df["export_value_fob"] < exp_p1) | (df["import_value_cif"] < imp_p1)
    ]
    df['lnoneplust'] = winsorize(df['lnoneplust'], limits=[0.1, 0.1])
    df[['lnoneplust', 'lndist', 'contig']] = df[['lnoneplust', 'lndist', 'contig']].fillna(0.0)

    stata_code = '''
        egen int idc_o = group(exporter)
        egen int idc_d = group(importer)
        reghdfe lnoneplust lndist contig, abs(year#idc_o year#idc_d)

        loc c = _coef[_cons]
        loc c_se = _se[_cons]
        loc beta_dist = _coef[lndist]
        loc se_dist = _se[lndist]
        loc beta_contig = _coef[contig]
        '''
    output = run_stata_code(df, stata_code)
    # Extract the coefficients and standard errors
    coefficients = output['e(b)'][0]
    std_errors = np.sqrt(np.diag(output['e(V)']))

    res = {
        'c': coefficients[2],
        'c_se': std_errors[2],
        'beta_dist': coefficients[0],
        'se_dist': std_errors[0],
        'beta_contig': coefficients[1]
    }
    tau_replace = res['c'] + (res['beta_dist'] * df['lndist']) + (res['beta_contig'] * df['contig'])
    df.loc[df['year'] == year, 'tau'] = tau_replace
    df.loc[(df['year'] == year) & (df['tau'] < 0) & (df['tau'].notna()), 'tau'] = 0
    df.loc[(df['year'] == year) & (df['tau'] > .2) & (df['tau'].notna()), 'tau'] = 0.2
    tau_mean = df[df['year']==year]['tau'].mean()
    df.loc[(df['year'] == year) & (df['tau'].isna()), 'tau'] = tau_mean

    df = df[df.year==year]
    df.loc[df['lnoneplust'] > 0, 'import_value_fob'] = df['import_value_cif']*(1 - df['tau'])
    df.loc[df['lnoneplust'] < 0, 'import_value_fob'] = df['import_value_cif']
    return df[['year', 'exporter', 'importer', 'export_value_fob', 'import_value_cif', 'import_value_fob']]


def run_stata_code(df, stata_code):
    # Initialize Stata
    sys.path.append('/n/sw/stata-17/utilities')
    from pystata import config
    config.init('se')
    from pystata import stata

    stata.pdataframe_to_data(df, force=True)
    stata.run(stata_code)
    return stata.get_ereturn()


if __name__ == "__main__":
    ingestion_attrs = {
        "start_year": 2019,
        "end_year": 2022,
        "downloaded_files_path": "../../../../*data_tools_for_GL/compactor_output/atlas_update/",
        # "root_dir": "/Users/ELJ479/projects/atlas_cleaning/src",
        "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
        # "root_dir": "/media/psf/AllFiles/Users/ELJ479/projects/atlas_cleaning/src",
        "product_classification": "H0",
    }
    run_atlas_cleaning(ingestion_attrs)
