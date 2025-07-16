import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)


def standardize_historical_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize historical country ISO codes to their modern equivalents.

    The function modifies the DataFrame in-place by:
    - Filtering out trade records between DEU and DDR (considered internal trade)
    - Mapping legacy ISO codes to their current standard equivalents
        - Consolidating German country codes (DEU/DDR) to modern Germany (DEU)
        - Consolidating Soviet Union codes (RUS/SUN) to Russia (RUS)
        - Consolidating South African Union code (ZA1) to South Africa (ZAF)

    Country code mappings:
    - DEU, DDR → DEU (Germany)
    - RUS, SUN → RUS (Russia/Soviet Union)
    - ZA1 → ZAF (South Africa)
    """
    df = df[~((df["reporter_iso"] == "DEU") & (df["partner_iso"] == "DDR"))]
    df = df[~((df["reporter_iso"] == "DDR") & (df["partner_iso"] == "DEU"))]

    df.loc[df["partner_iso"].isin(["DEU", "DDR"]), "partner_iso"] = "DEU"
    df.loc[df["reporter_iso"].isin(["DEU", "DDR"]), "reporter_iso"] = "DEU"

    df.loc[df["partner_iso"].isin(["RUS", "SUN"]), "partner_iso"] = "RUS"
    df.loc[df["reporter_iso"].isin(["RUS", "SUN"]), "reporter_iso"] = "RUS"

    df.loc[df["reporter_iso"].isin(["ZA1"]), "reporter_iso"] = "ZAF"
    df.loc[df["partner_iso"].isin(["ZA1"]), "partner_iso"] = "ZAF"
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
        logger.info("TWN did not report as S19")
    try:
        df.loc[df["partner_iso"] == "S19", "partner_iso"] = "TWN"
    except:
        logger.info("Countries did not report Taiwan as a partner")

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
