import logging
import pandas as pd
import os
logging.basicConfig(level=logging.INFO)


def get_classifications(year):
    """
    """
    classifications = []
    if year >= 1976 and year < 1995:
        print("adding to classifications")
        classifications.append("S2")
    if year >= 1988 and year <= 2003:
        print("adding to classifications")
        classifications.append("S3")
    if year >= 1994:
        print("adding to classifications")
        classifications.append("H0")
        classifications.append("HS")
    if year <= 2003:
        print("adding to classifications")
        classifications.append("ST")
    if year <= 1985:
        print("adding to classifications")
        classifications.append("S1")

    logging.info(
        f"generating aggregations for the following classifications: {classifications}"
    )
    return classifications


def merge_classifications(year):
    """
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
                    os.path.join(
                        ingestion_attrs["root_dir"], "data", "intermediate", file
                    )
                )
            except FileNotFoundError:
                continue
        else:
            try:
                df = df.merge(
                    pd.read_parquet(
                        os.path.join(
                            ingestion_attrs["root_dir"],
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
    return df

