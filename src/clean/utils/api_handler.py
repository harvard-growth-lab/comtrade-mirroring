import json
import requests
import pandas as pd
import os
from pathlib import Path

class IMFData:
    IMF_INDICATORS = {
        "NGDP_R" : "gdp_const",
        "NGDPD": "gdp",
        "NGDP_RPCH": "gdp_const_growth",
        "PPPGDP": "gdp_ppp",
        "NGDPDPC": "gdppc",
        "PPPPC": "gdppc_ppp",
        "LP": "population",
        "BCA": "current_account",
        "PCPIPCH": "inflation_rate",
    }

    MULTIPLIERS = {
        "NGDPD": 1_000_000_000,
        "PPPGDP": 1_000_000_000,
        "LP": 1_000_000,
        "BCA": 1_000_000_000,
    }

    # These percentile thresholds mirror the results from the WDI data
    INCOME_LEVEL_PERCENTILES = {
        "low": 0,
        "lower middle": 0.12,
        "upper middle": 0.39,
        "high": 0.66,
    }

    def __init__(
        self,
        latest_atlas_year,
    ):
        self.latest_atlas_year = latest_atlas_year


    def query_imf_api(self, fields: list, country_codes=[]):
        fields = "/".join(fields)
        if country_codes:
            country_codes = "/".join(country_codes)

        df = pd.json_normalize(
            json.loads(
                requests.get(
                    "https://www.imf.org/external/datamapper/api/v1"
                    f"/{fields}/{country_codes}"
                ).text
            )["values"]
        ).T.reset_index()
        df = df.rename(columns={"code":"iso3_code"})
        df[["indicator", "iso3_code", "year"]] = df["index"].str.split(".", expand=True)
        df = (
            df.drop(columns="index")
            .rename(columns={0: "value"})
            .astype({"indicator": str, "iso3_code": str, "year": int, "value": float})
        )

        df = df.pivot_table(
            values="value", index=["year", "iso3_code"], columns="indicator"
        ).reset_index()

        # Some variables are in millions/billions, multiply to get raw values
        for col in df.columns:
            if col in self.MULTIPLIERS.keys():
                df[col] = df[col].multiply(self.MULTIPLIERS[col])

        # Rename columns from IMF specification to our database names
        df = df.rename(columns=self.IMF_INDICATORS, errors="ignore")
        return df



class WDIData:

    WDI_INDICATORS = {
        # GDP Current $
        "NY.GDP.MKTP.CD": "gdp",
        "NY.GDP.PCAP.CD": "gdppc",
        "NY.GDP.MKTP.PP.CD": "gdp_ppp",
        "NY.GDP.PCAP.PP.CD": "gdppc_ppp",
        # GDP Constant $
        "NY.GDP.MKTP.KD": "gdp_const",
        "NY.GDP.PCAP.KD": "gdppc_const",
        "NY.GDP.MKTP.PP.KD": "gdp_ppp_const",
        "NY.GDP.PCAP.PP.KD": "gdppc_ppp_const",
        # GNI
        # "NY.GNP.MKTP.CD": "gni",
        # "NY.GNP.MKTP.KD": "gni_const",
        # Population
        "SP.POP.TOTL": "population",
        # Balance of Payments
        # "BN.GSR.MRCH.CD": "net_trade_goods",
        # "BN.GSR.GNFS.CD": "net_trade_goods_services",
        "BN.CAB.XOKA.CD": "current_account",
    }
    WDI_SERVICE_INDICATORS = {
        # Total Services
        "BX.GSR.NFSV.CD": "services_export_value",
        "BM.GSR.NFSV.CD": "services_import_value",

        # Travel & Tourism
        "BX.GSR.TRVL.ZS": "travel_export_share",
        "BM.GSR.TRVL.ZS": "travel_import_share",

        # Insurance & Financial
        "BX.GSR.INSF.ZS": "finance_export_share",
        "BM.GSR.INSF.ZS": "finance_import_share",

        # Transport
        "BX.GSR.TRAN.ZS": "transport_export_share",
        "BM.GSR.TRAN.ZS": "transport_import_share",

        # Communications & Computer
        "BM.GSR.CMCP.ZS": "comms_import_share",
        "BX.GSR.CMCP.ZS": "comms_export_share"

        # ICT (commented out)
        # "BX.GSR.CCIS.ZS": "ict_export_share",
        # "BM.GSR.CCIS.ZS": "ict_import_share",
    }

    def __init__(
        self,
        latest_atlas_year,
    ):
        self.latest_atlas_year = latest_atlas_year


    def query_for_wdi_indicators(self, indicators, file_name="wdi_data"):
        indicator_string = ";".join(indicators.keys())

        # CSV is an option, but download comes in a zip of multiple files
        # Excel comes in a workbook where data broken into tabs of Data or Metadata
        url = (
            f"http://api.worldbank.org/v2/countries/all/indicator/{indicator_string}"
            "?source=2&downloadformat=excel&dataformat=list"
        )

        # Get dict of all sheets in workbook
        tabs = pd.read_excel(
            url,
            sheet_name=None,
            header=None,
            names=["country_name", "code", "indicator_name", "indicator", "year", "value"],
        )

        dfs = []

        for name, df in tabs.items():
            # Can be multiple "Data" or "Data (n)" tabs
            if name == "Data":
                # First four rows of first tab are info about download
                df.drop(index=[0, 1, 2, 3], inplace=True)
                dfs.append(df)
            elif name[:4] == "Data":
                # Drop their header row in favor of ours passed in read_excel
                df.drop(index=0, inplace=True)
                dfs.append(df)

        # Combine data tabs, pivot, and save to csv
        df = pd.concat(dfs)
        df = df.rename(columns={"code":"iso3_code"})
        df = df[["iso3_code", "indicator", "year", "value"]]
        
        df.indicator = df.indicator.map(indicators)    

        df = df.set_index(["iso3_code", "year"]).pivot(columns="indicator")
        df.columns = df.columns.droplevel()
        df = df.reset_index()
        return df