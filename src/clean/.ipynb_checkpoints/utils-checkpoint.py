import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)


def get_classifications(year):
    """
    Based on year, generate list of all available classifications for that year
    """
    classifications = []
    if year >= 1976 and year < 1995:
        classifications.append("S2")
    if year >= 1988 and year <= 2003:
        classifications.append("S3")
    if year >= 1994:
        classifications.append("H0")
        classifications.append("HS")
    if year <= 2003:
        classifications.append("ST")
    if year <= 1985:
        classifications.append("S1")

    logging.info(
        f"generating aggregations for the following classifications: {classifications}"
    )
    return classifications


def merge_classifications(year: str, root_dir: str) -> pd.DataFrame():
    """
    Based on year, merge comtrade classifications and then take median export, import values
    """
    merge_conditions = [
        (year >= 1976 and year < 1995, f"{year}_S2.parquet"),
        (year >= 1995, f"{year}_H0.parquet"),
        (year >= 1995, f"{year}_HS.parquet"),
        (year >= 1985 and year <= 2003, f"{year}_S3.parquet"),
        (year <= 2003, f"{year}_ST.parquet"),
    ]

    df = pd.DataFrame()
    for condition, file in merge_conditions:
        if df.empty:
            try:
                logging.info("reading in file")
                df = pd.read_parquet(
                    os.path.join(root_dir, "data", "intermediate", file)
                )
            except FileNotFoundError:
                continue
        else:
            try:
                df = df.merge(
                    pd.read_parquet(
                        os.path.join(
                            root_dir,
                            "data",
                            "intermediate",
                            file,
                        )
                    ),
                    on=["year", "exporter", "importer"],
                    how="left",
                )
            except FileNotFoundError:
                continue

    df.astype({"importer": str, "exporter": str}).dtypes
    df["export_value_fob"] = df.filter(like="export_value_fob").median(axis=1)
    df["import_value_cif"] = df.filter(like="import_value_cif").median(axis=1)
    return df[["year", "exporter", "importer", "export_value_fob", "import_value_cif"]]
