from scipy.stats.mstats import winsorize
import os
import sys
import pandas as pd
import numpy as np
import pyfixest as pf

import logging
logging.basicConfig(level=logging.INFO)


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

    res = compute_reghdfe(compute_dist_df)

    
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

def compute_reghdfe(df):
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