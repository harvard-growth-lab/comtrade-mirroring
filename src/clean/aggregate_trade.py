import sys
from os import path
import pandas as pd
from sys import argv

from clean.objects.base import _AtlasCleaning


class TradeAggregator(_AtlasCleaning):
    def __init__(self, df, year, **kwargs):
        super().__init__(**kwargs)

        self.df = df
        self.year = year
        self.clean_data()


    def clean_data(self):
        """ """
        print(f"   > {year} and classification = {self.product_classification}")

        # Data manipulation based on 'classification'
        # This is a simplification, adapt based on actual logic and conditions in Stata script
        if self.product_classificatin in ["H0", "HS"]:
            # Add "00" prefix to 'commoditycode' where the length of 'commoditycode' is 4 and 'product_level' is 6
            self.df["commodity_code"] = np.where(
                (self.df["commodity_code"].str.len() == 4)
                & (self.df["product_level"] == 6),
                "00" + self.df["commodity_code"],
                self.df["commodity_code"],
            )
            # Add "0" prefix to 'commoditycode' where the length of 'commoditycode' is 5 and 'product_level' is 6
            self.df["commodity_code"] = np.where(
                (self.df["commodity_code"].str.len() == 5)
                & (self.df["product_level"] == 6),
                "0" + self.df["commodity_code"],
                self.df["commodity_code"],
            )
            self.df["reporter_ansnoclas"] = self.df.trade_value.where(
                (self.df.partner_iso == "ANS")
                & (self.df.product_level == 4)
                & (self.df.commodity_code.str.slice(0, 4) == "9999")
            )

        elif classification in ["S1", "S1", "ST"]:
            # Add "00" prefix to 'commoditycode' where the length of 'commoditycode' is 2 and 'product_level' is 4
            self.df["commodity_code"] = np.where(
                (self.df["commodity_code"].str.len() == 2)
                & (self.df["product_level"] == 4),
                "00" + self.df["commodity_code"],
                self.df["commodity_code"],
            )
            # Add "0" prefix to 'commoditycode' where the length of 'commoditycode' is 3 and 'product_level' is 4
            self.df["commodity_code"] = np.where(
                (self.df["commodity_code"].str.len() == 3)
                & (self.df["product_level"] == 4),
                "0" + self.df["commodity_code"],
                self.df["commodity_code"],
            )
            self.df["reporter_ansnoclas"] = self.df.trade_value.where(
                (self.df.partner_iso == "ANS")
                & (self.df.product_level == 4)
                & (self.df.commodity_code == "9310")
            )

        # handles Germany (reunification) and Russia
        # drop if reporter and partner are DEU and DDR, trading with itself
        self.df = self.df[
            ~((self.df["reporter_iso"] == "DEU") & (self.df["partner_iso"] == "DDR"))
        ]
        self.df = self.df[
            ~((self.df["reporter_iso"] == "DDR") & (self.df["partner_iso"] == "DEU"))
        ]
        self.df.loc[self.df["partner_iso"].isin(["DEU", "DDR"]), "partner_iso"] = "DEU"
        self.df.loc[
            self.df["reporter_iso"].isin(["DEU", "DDR"]), "reporter_iso"
        ] = "DEU"
        self.df.loc[self.df["partner_iso"].isin(["RUS", "SUN"]), "partner_iso"] = "RUS"
        self.df.loc[
            self.df["reporter_iso"].isin(["RUS", "SUN"]), "reporter_iso"
        ] = "RUS"

        # compress
        # collapse (sum) tradevalue reporter_ansnoclas , by( year tradeflow product_level reporter_iso partner_iso )
        self.df = (
            self.df.groupby(
                ["year", "tradeflow", "product_level", "reporter_iso", "partner_iso"]
            )
            .agg({"tradevalue": "sum", "reporter_ansnoclas": "sum"})
            .reset_index()
        )
        # recast float tradevalue reporter_ansnoclas, force
        self.df["tradevalue"] = self.df["tradevalue"].astype(
            "float"
        )
        self.df["reporter_ansnoclas"] = self.df[
            "reporter_ansnoclas"
        ].astype("float")


#     # Loop over years and classifications, calling the cleandata function
#     for year in range(startyear, finalyear + 1):
#         print(f"> Doing year = {year}")

#         # Example classifications processing, add as necessary
#         if 1976 <= year < 1995:
#             classification = "S2"
#             cleaned_data = cleandata(classification, year)
#             # Save cleaned data to file, replace `trade_S2` with actual path or filename
#             cleaned_data.to_csv(
#                 os.path.join(path, f"Totals_raw_{year}.csv"), index=False
#             )

#         # Continue with other classification and year conditions

#     # After processing all years and classifications, you might want to merge, manipulate,
#     # and analyze the cleaned datasets similarly to how it's done in the Stata script.
