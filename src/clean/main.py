import logging
import os
import sys
from glob import glob
import pandas as pd
from scipy.stats.mstats import winsorize
import numpy as np

from clean.table_objects.base import _AtlasCleaning
from clean.table_objects.aggregate_trade import AggregateTrade
from clean.utils import get_classifications, merge_classifications
from clean.table_objects.country_country_year import CountryCountryYear
from clean.table_objects.accuracy import Accuracy
from clean.table_objects.country_country_product_year import CountryCountryProductYear
from clean.table_objects.complexity import Complexity


# connect to stata to set cif ratio
sys.path.append('/n/sw/stata-17/utilities')
from pystata import config, stata
config.init('se')

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
    start_year = ingestion_attrs["start_year"]
    end_year = ingestion_attrs["end_year"]
    product_classification = ingestion_attrs["product_classification"]
    
    # load data 
    dist = pd.read_stata(os.path.join('data', 'raw', "dist_cepii.dta"))
                         
    for year in range(start_year, end_year + 1):
        # get possible classifications based on year
        classifications = get_classifications(year)

        # try:
        #     list(
        #         map(
        #             lambda product_class: AggregateTrade(year, **ingestion_attrs),
        #             classifications,
        #         )
        #     )
        # except ValueError as e:
        #     logging.error(f"Downloader file not found, skipping {self.year}")
        
        # depending on year, merge multiple classifications, take median of values
        df = merge_classifications(year, ingestion_attrs["root_dir"])

        # place holder for the cost of insurance/freight ==1.08 
        # TODO: replace with compute distance function
        compute_distance(df, dist, start_year, end_year)
        df["import_value_fob"] = df["import_value_cif"] * (1 - CIF_RATIO)

        os.makedirs(
            os.path.join(
                ingestion_attrs["root_dir"], "data", "raw", product_classification
            ),
            exist_ok=True,
        )
        # save file as totals_raw
        df.to_parquet(
            os.path.join(
                ingestion_attrs["root_dir"],
                "data",
                "raw",
                product_classification,
                f"ccy_raw_{year}.parquet",
            ),
            index=False,
        )

        ccy = CountryCountryYear(df, year, **ingestion_attrs)
        ccy.save_parquet(ccy.df, 'processed', f'country_country_year_{year}')
        
        accuracy = Accuracy(ccy.ncountries, year, **ingestion_attrs)
        accuracy.save_parquet(accuracy.df, 'processed', f'accuracy_{year}')
        
        ccpy = CountryCountryProductYear(year, **ingestion_attrs)
        ccpy.save_parquet(ccpy.df, 'processed', f'country_country_product_{year}')
                

    # TODO: concat all years
    # concat all total_raw files for all years
    # TODO: need to distinguish by requested classification
    ccy_list = glob(
        os.path.join(
            ingestion_attrs["root_dir"],
            "data",
            "raw",
            product_classification,
            "ccy_raw_*.parquet",
        )
    )
    ccy_df = pd.concat(map(pd.read_parquet, ccy_list), ignore_index=True)



def compute_distance(df, dist, start_year, end_year):
    """
    TODO: not validated
    currently not called in data and using place with CIF RATIO set to 7.25%
    """
    df = pd.read_parquet("data/intermediate/2015_compute_dist_syn.parquet")
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
    
    logging.info("Calculating CIF/FOB correction")
    # compute for each year
    
    for year in range(start_year, end_year + 1):
        lag = year - 1
        lead = year + 1
        # dataframe with only these three years
        df = df[df.year.isin([lag, year, lead])]
        # select the greater of the two, either the top 1% or 1,000,000
        exp_p1 = max(df["export_value_fob"].quantile(0.01), 10**6)
        imp_p1 = max(df["import_value_cif"].quantile(0.01), 10**6)
        # use to set min boundaries
        df = df[~
            (df["export_value_fob"] < exp_p1) | (df["import_value_cif"] < imp_p1)
        ]
        df['lnoneplust'] = winsorize(df['lnoneplust'], limits=[0.1, 0.1])
        df[['lnoneplust', 'lndist', 'contig']] = df[['lnoneplust', 'lndist', 'contig']].fillna(0.0)
        stata.pdataframe_to_data(df, force=True)
        stata_code = '''
            egen int idc_o = group(exporter)
            egen int idc_d = group(importer)
            reghdfe lnoneplust lndist contig, abs(year#idc_o year#idc_d)
            
            loc c = _coef[_cons]
            loc c_se = _se[_cons]
            loc beta_dist = _coef[lndist]
            loc se_dist = _se[lndist]
            loc beta_contig = _coef[contig]
            
            display "`c'"
            display "`c_se'"
            display "`beta_dist'"
            display "`se_dist'"
            display "`beta_contig'"

            '''
        stata.run(stata_code)
        output = stata.get_ereturn()
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
        
        import pdb
        pdb.set_trace()



        
        # reghdfe, no python equivalent, https://regpyhdfe.readthedocs.io/en/latest/regpyhdfe.html
        target = "lnoneplust"
        predictors = ["lndist", "contig"]
        absorb_ids = ["year", "idcode"]
        cluster_ids = ["year"]

        
        
        # remove outliers, option to use winsorize with scipy
        # TODO: confirm chop off bottom 10% and top 10%
        df_3["lnoneplust"] = df_3["lnoneplust"].clip(
            lower=temp_df["lnoneplust"].quantile(0.1),
            upper=temp_df["lnoneplust"].quantile(0.9),
        )
        # df_3 = winsorize(df_3, (.1, .1))
        # https://scorreia.com/software/reghdfe/quickstart.html

        # reghdfe lnoneplust  lndist contig,  abs(year#idc_o year#idc_d)
        
        # https://regpyhdfe.readthedocs.io/en/latest/regpyhdfe.html
        regpyhdfe(
            df_3,
            target=["lnoneplust", "lndist"],
            predictors=["contig"],
            absorb_ids=[],
            cluster_ids=[],
            drop_singletons=True,
            intercept=False,
        )[source]

        results = model.fit()

        c = results.params["Intercept"]
        c_se = results.std_errors["Intercept"]
        beta_dist = results.params["lndist"]
        se_dist = results.std_errors["lndist"]
        beta_contig = results.params["contig"]

        df.loc[df["year"] == y, "tau"] = (
            c + (beta_dist * df["_lndist"]) + (beta_contig * df["_contig"])
        )
        df.loc[(df["year"] == y) & (df["tau"] < 0) & (df["tau"].notnull()), "tau"] = 0
        df.loc[
            (df["year"] == y) & (df["tau"] > 0.2) & (df["tau"].notnull()), "tau"
        ] = 0.2
        df.loc[df["year"] == y, "tau"] = df.loc[df["year"] == y, "tau"].mean()

        print(df.columns)
        df["import_value_fob"] = np.nan
        df.loc[
            (df["import_value_fob"] == import_value_cif)
            & (df["tau"] < 0)
            & (df["tau"].notnull()),
            "tau",
        ] = 0

        # small difference
        mask_small_diff = df["import_value_fob"].isnull() & (
            df["lnoneplust"].abs() < 0.05
        )
        df.loc[mask_small_diff, "import_value_fob"] = df.loc[
            mask_small_diff, "import_value_cif"
        ]

        # positive
        mask_positive = df["import_value_fob"].isnull() & (df["lnoneplust"] > 0)
        df.loc[mask_positive, "import_value_fob"] = df.loc[
            mask_positive, "import_value_cif"
        ] * (1 - df.loc[mask_positive, "tau"])

        mask_negative = df["import_value_fob"].isnull() & (df["lnoneplust"] < 0)
        df.loc[mask_negative, "import_value_fob"] = df.loc[
            mask_negative, "import_value_cif"
        ]

        df = df[
            [
                "year",
                "exporter",
                "importer",
                "export_value_fob",
                "import_value_cif",
                "import_value_fob",
            ]
        ]
        df = df.sort_values(["year", "exporter", "importer"])
        
            


if __name__ == "__main__":
    ingestion_attrs = {
        "start_year": 2015,
        "end_year": 2015,
        # "root_dir": "/Users/ELJ479/projects/atlas_cleaning/src",
        "root_dir": "/n/hausmann_lab/lab/atlas/bustos_yildirim/atlas_stata_cleaning/src",
        # "root_dir": "/media/psf/AllFiles/Users/ELJ479/projects/atlas_cleaning/src",
        "product_classification": "H0",
    }
    run_atlas_cleaning(ingestion_attrs)
