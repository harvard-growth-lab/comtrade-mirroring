import logging
import os
from glob import glob
import pandas as pd

# from scipy.stats.mstats import winsorize
from clean.objects.base import _AtlasCleaning
from clean.aggregate_trade import AggregateTrade
from clean.utils import get_classifications, merge_classifications

from clean.country_country_year import CountryCountryYear

logging.basicConfig(level=logging.INFO)
CIF_RATIO = .075


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
    for year in range(start_year, end_year + 1):
        
        # get possible classifications based on year
        classifications = get_classifications(year)
                
        list(
            map(
                lambda product_class: AggregateTrade(
                    year, **ingestion_attrs
                ),
                classifications,
            )
        )
        # depending on year, merge multiple classifications and then takes median of values
        df = merge_classifications(year, ingestion_attrs["root_dir"])
        
        # expect insurance/freight to be approximately 1.08 of imports_fob
        # TODO: replace with compute distance function
        df["import_value_fob"] = df["import_value_cif"] * (1 - CIF_RATIO)

        os.makedirs(os.path.join(ingestion_attrs["root_dir"], "data", "raw", product_classification), exist_ok=True)
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
        

    # concat all total_raw files for all years
    # TODO: need to distinguish by requested classification
    ccy_list = glob(
        os.path.join(
            ingestion_attrs["root_dir"], "data", "raw", product_classification, "ccy_raw_*.parquet"
        )
    )
    ccy_df = pd.concat(map(pd.read_parquet, ccy_list), ignore_index=True)

    # TODO: compute_distance(ccy_df, start_year, end_year)
    # ccy_df.to_csv(
    #     os.path.join(
    #         ingestion_attrs["root_dir"], "data", "intermediate", f"ccy_{start_year}_{end_year}.csv"
    #     ),
    #     index=False,
    # )
    


def compute_distance(df, start_year, end_year):
    """
    TODO: not validated
    currently not called in data and using place with CIF RATIO set to 7.25%
    """
    # TODO: confirm handling of Romania
    # for dist_cepii replace ROM with ROU
    self.dist_cepii.loc[dist["exporter"] == "ROM", "exporter"] = "ROU"
    self.dist_cepii.loc[dist["importer"] == "ROM", "exporter"] = "ROU"
    df = df.merge(self.dist_cepii, on=["importer", "exporter"], how="left")
    df["lndist"] = np.log(df["distwces"])
    df.loc[df["lndist"].isna() & df["dist"].notna(), "lndist"] = np.log(df["dist"])
    df["oneplust"] = df["import_value_cif"] / df["export_value_fob"]
    df["lnoneplust"] = np.log(df["import_value_cif"] / df["export_value_fob"])
    df["tau"] = np.nan
    logging.info("Calculating CIF/FOB correction")
    # compute for each year
    for year in range(start_year, end_year + 1):
        df_3 = df[(df["year"] >= year - 1) & (df["year"] <= year + 1)].copy()
        # select the greater of the two, either the top 1% or 1,000,000
        exp_p1 = max(df_3["export_value_fob"].quantile(0.01), 10**6)
        exp_p1 = max(df_3["import_value_cif"].quantile(0.01), 10**6)
        # use to set min boundaries
        df_3 = df_3[
            (df_3["export_value_fob"] >= exp_p1) & (df_3["import_value_cif"] >= imp_p1)
        ]
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
        "product_classification": "H0"
    }
    run_atlas_cleaning(ingestion_attrs)
