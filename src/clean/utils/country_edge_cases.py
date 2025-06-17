import pandas as pd
import logging


def handle_venezuela(year, df, ven_oil_code):
    """
    Comtrade stopped patching trade data for Venezuela starting in 2020.

    As part of the cleaning the Growth Lab patches Venezuela's exports for Crude Petroleum.
    The value is calculated by determining oil production less country's oil consumption
    using the price per barrel from the https://www.energyinst.org/statistical-review
    """
    reported_total = df[
        (
            (df.exporter == "VEN")
            & (df.commoditycode == ven_oil_code)
        )
    ]["value_final"].sum()
    ven_opec = pd.read_csv("data/ven_fix/venezuela_270900_exports.csv")
    ven_opec = ven_opec[ven_opec.year == year]
    if ven_opec.empty and year > 2019:
        raise ValueError(
            f"Need to add the export value for oil in {year} for Venezuela"
        )
    # the difference of total Venezuela exports, subtracts trade value
    # if anyone imports did report Venezuela oil trade
    ven_opec["value_final"] = ven_opec["value_final"] - reported_total
    ven_opec = ven_opec.astype({"year": "int64"})
    ven_opec["commoditycode"] = ven_oil_code
    return ven_opec


def handle_9999_saudi_reporting(df: pd.DataFrame) -> pd.DataFrame:
    """"
    handle 9999 reporting from Saudi for Atlas Year 2023
    """
    logging.info("updating Saudi's 2023 9999 trade value to oil")
    logging.info(
        f"Saudi's 99999 export trade value: {df[(df.exporter=='SAU')&(df.commoditycode=='999999')]['export_value'].sum()}"
    )

    df.loc[
        (df.exporter == "SAU") & (df.commoditycode == "999999"),
        "commoditycode",
    ] = "270900"
    return df

