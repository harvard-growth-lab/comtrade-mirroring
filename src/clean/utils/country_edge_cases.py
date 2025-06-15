import pandas as pd


def handle_venezuela(self):
    """
    Comtrade stopped patching trade data for Venezuela starting in 2020.

    As part of the cleaning the Growth Lab patches Venezuela's exports for Crude Petroleum.
    The value is calculated by determining oil production less country's oil consumption
    using the price per barrel from the https://www.energyinst.org/statistical-review
    """
    reported_total = self.df[
        (
            (self.df.exporter == "VEN")
            & (self.df.commoditycode == self.OIL[self.product_classification])
        )
    ]["value_final"].sum()

    ven_opec = pd.read_csv("data/ven_fix/venezuela_270900_exports.csv")
    ven_opec = ven_opec[ven_opec.year == self.year]
    if ven_opec.empty and self.year > 2019:
        raise ValueError(
            f"Need to add the export value for oil in {self.year} for Venezuela"
        )
    # the difference of total Venezuela exports, subtracts trade value
    # if anyone imports did report Venezuela oil trade
    ven_opec["value_final"] = ven_opec["value_final"] - reported_total
    ven_opec = ven_opec.astype({"year": "int64"})
    ven_opec["commoditycode"] = self.OIL[self.product_classification]
    self.df = pd.concat([self.df, ven_opec], axis=0, ignore_index=True, sort=False)
