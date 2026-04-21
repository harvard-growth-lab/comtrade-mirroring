import pandas as pd
from mirror.src.utils.logging import get_logger
import atlas_common_data

logger = get_logger(__name__)


def standardize_historical_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize historical country ISO codes to their modern equivalents.

    The function modifies the DataFrame in-place by:
    - Filtering out trade records between DEU and DDR (considered internal trade)
    - Mapping legacy ISO codes to their current standard equivalents
    """

    # remove because trade with self
    df = df[~((df["reporter_iso"] == "DEU") & (df["partner_iso"] == "DDR"))]
    df = df[~((df["reporter_iso"] == "DDR") & (df["partner_iso"] == "DEU"))]

    for country in ['reporter_iso', 'partner_iso']:
        df.loc[df[country].isin(["DDR"]), country] = "DEU"
        df.loc[df[country].isin(["SUN"]), country] = "RUS"
        df.loc[df[country].isin(["ZA1"]), country] = "ZAF"
        df.loc[df[country].isin(["ESH"]), country] = "MAR"
        df.loc[df[country].isin(["PRI", "VIR"]), country] = "USA"
        df.loc[df[country].isin(["GUF", "MYT", "REU", "MTQ", "GLP"]), country] = "FRA"
        df.loc[df[country].isin(["VDR"]), country] = "VNM"
        df.loc[df[country].isin(["YMD"]), country] = "YEM"
        df.loc[df[country].isin(["PCZ"]), country] = "PAN"
    return df

def include_country_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    countries = atlas_common_data.load_countries()
    wld_row = pd.DataFrame([{"iso3_code": "WLD"}])
    countries = pd.concat([countries, wld_row], ignore_index=True)
    if countries.iso3_code.nunique() != 235:
        raise ValueError("wrong number of countries, update atlas common data")
    df = df.merge(countries['iso3_code'], left_on='reporter_iso',right_on='iso3_code',how='right')
    df = df.merge(countries['iso3_code'], left_on='partner_iso',right_on='iso3_code',how='right')
    if df.reporter_iso.nunique() >= 235 and df.partner_iso.nunique() >= 235:
        raise ValueError("wrong number of countries, merge failed")
    return df

def handle_ans_and_other_asia_to_taiwan_recoding(
    df: pd.DataFrame, ans_partners: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns updated dataframe and list of ANS partners.

    - Reclassify iso code S19 to Taiwan
    - Loads list of ANS (Areas Not Specified) partners and reclassifies these to a single ANS code
    """
    try:
        df.loc[df["reporter_iso"] == "S19", "reporter_iso"] = "TWN"
    except:
        logger.debug("TWN did not report as S19")
    try:
        df.loc[df["partner_iso"] == "S19", "partner_iso"] = "TWN"
    except:
        logger.debug("Countries did not report Taiwan as a partner")

    ans_partners = ans_partners["PartnerCodeIsoAlpha3"].tolist()
    df.loc[df["partner_iso"].isin(ans_partners), "partner_iso"] = "ANS"
    df.loc[df["partner_iso"].isna(), "partner_iso"] = "ANS"        
    return df, ans_partners


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
