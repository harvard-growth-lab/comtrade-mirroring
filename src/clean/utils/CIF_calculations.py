from scipy.stats.mstats import winsorize
import sys
import pandas as pd
import numpy as np
import pyfixest as pf

import logging
logging.basicConfig(level=logging.INFO)

TRADE_FLOOR = 10**6
TRADE_VALUE_FLOOR_PERCENTILE = 0.01
TAU_UPPER_LIMIT = 0.2
CIF_FOB_EQUIVALENCE_THRESHOLD = 0.05


def compute_distance(year: int, product_classification: str, 
                     dist: pd.DataFrame) -> pd.DataFrame:
    """
    Based on geographic distances betweeen trade partners (dist df) estimate the 
    cost of insurance freight (CIF) as a percentage of FOB (Free on Board) import value
    
    loads trade data, merges with distance data, estimates 
    trade costs using regression, and adjusts import values accordingly.

    Notes:
        - Handles Romania country code conversion (ROU <-> ROM)
        - Uses winsorization (10% from each tail) for outlier treatment
        - Filters out trades below 1st percentile or $1M threshold
        - tau is the trade cost rate; tau = (CIF - FOB) / FOB
        - Uses high dimensional fixed effects regression for trade cost estimation
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

    # "ROM" was the ISO code for Romania until 2002
    dist = standardize_romania_codes(dist)
    df = df.merge(dist, on=["exporter", "importer"], how="left")
    df = modernize_romania_codes(df)

    df["lndist"] = np.log(df["distwces"])
    df.loc[(df["lndist"].isna()) & (df["dist"].notna()), "lndist"] = np.log(df["dist"])
    df["oneplust"] = df["import_value_cif"] / df["export_value_fob"]
    df["lnoneplust"] = np.log(df["import_value_cif"] / df["export_value_fob"])
    df["tau"] = np.nan

    compute_dist_df = df.copy(deep=True)
    # select the greater of the two, either the top 1% or 1,000,000
    # filter out small trade values that will skew the regression 
    exp_p1 = max(compute_dist_df["export_value_fob"].quantile(TRADE_VALUE_FLOOR_PERCENTILE), TRADE_FLOOR)
    imp_p1 = max(compute_dist_df["import_value_cif"].quantile(TRADE_VALUE_FLOOR_PERCENTILE), TRADE_FLOOR)
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

    res = compute_reghdfe(compute_dist_df)

    df.loc[df["year"] == year, "tau"] = (
        res["c"]
        + (res["beta_dist"] * df["lndist"])
        + (res["beta_contig"] * df["contig"])
    )

    del compute_dist_df

    # trade costs can't be negative
    df.loc[(df["year"] == year) & (df["tau"] < 0) & (df["tau"].notna()), "tau"] = 0
    df.loc[(df["year"] == year) & (df["tau"] > TAU_UPPER_LIMIT) & (df["tau"].notna()), "tau"] = TAU_UPPER_LIMIT
    tau_mean = df[df["year"] == year]["tau"].mean()
    df.loc[(df["year"] == year) & (df["tau"].isna()), "tau"] = tau_mean

    df = df[df.year == year]

    # when trade costs are small treat CIF and FOB as equivalent, close enough
    df.loc[abs(df["lnoneplust"]) < CIF_FOB_EQUIVALENCE_THRESHOLD, "import_value_fob"] = df["import_value_cif"]

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

def compute_reghdfe(df: pd.DataFrame) -> dict[str, float]:
    df['idc_o'] = pd.Categorical(df['exporter']).codes
    df['idc_d'] = pd.Categorical(df['importer']).codes

    model = pf.feols(
        'lnoneplust ~ 1 + lndist + contig | year^idc_o + year^idc_d',
        data=df,
        vcov='iid'
    )

    coeff = model.coef()
    se = model.se()

    y_resid = df['lnoneplust'].values
    X = df[['lndist', 'contig']].values
    pred_no_fe = X @ np.array([model.coef()['lndist'], model.coef()['contig']])
    implicit_constant = (y_resid - pred_no_fe).mean()

    return {
        "c": implicit_constant,
        "beta_dist": coeff.iloc[0],
        "se_dist": se.iloc[0],
        "beta_contig": coeff.iloc[1],
    }


def standardize_romania_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Romania country codes from ROU to ROM for distance matching."""
    df.loc[df["exporter"] == "ROU", "exporter"] = "ROM"
    df.loc[df["importer"] == "ROU", "importer"] = "ROM"
    return df


def modernize_romania_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Revert Romania country codes from ROM back to ROU."""
    df.loc[df["exporter"] == "ROM", "exporter"] = "ROU"
    df.loc[df["importer"] == "ROM", "importer"] = "ROU"
    return df
