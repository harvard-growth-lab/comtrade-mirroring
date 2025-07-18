from pathlib import Path
import regex as re
from datetime import datetime
import pandas as pd


class ConcordanceTable():
    """
    Uses Concordance Table provided by Comtrade to convert cleaned CCPY 6digit product codes
    to a target classification code.

    This is used when comtrade trade data is downloaded by classification code
    """

    clcodes = {
        "HS92": ["H0", "HS92"],
        "SITC2": ["S2", "SITC2"],
        "SITC1": ["S1", "SITC1"],
    }
    clcode_length = {"HS92": 6, "SITC2": 4, "SITC1": 4}
    trade_val_cols = ["value_final", "value_exporter", "value_importer"]

    


    def __init__(self, static_data_path : Path):
        self.concordance_table = pd.read_excel(Path(static_data_path) / 
                                               "HS-SITC-BEC Correlations_2022.xlsx",
                dtype={"HS92": str, "SITC1": str, "SITC2": str},
            )
        
    def run_conversion(self, df : pd.DataFrame, 
                       classification_code:  str, 
                       target_classification_code : str) -> pd.DataFrame:
        self.df = df
        self.classification_code = classification_code
        self.target_classification_code = target_classification_code

        self._validate_request()
        self._prep_data_request()
        # run by year
        self._run_concordance()
        return self.df

    def _validate_request(self) -> None:
        def get_clcode_from_input(input_value):
            for key, value_list in self.clcodes.items():
                if input_value in value_list:
                    return key
            raise ValueError(
                f"classification code: {self.classification_code} not valid"
            )

        self.classification_code = get_clcode_from_input(self.classification_code)
        self.target_classification_code = get_clcode_from_input(
            self.target_classification_code
        )

    def _prep_data_request(self) -> None:
        """ """
        self.concordance_table = self.concordance_table[
            [self.classification_code, self.target_classification_code]
        ]
        if self.classification_code.startswith("S"):
            self.concordance_table[self.classification_code] = self.concordance_table[
                self.classification_code
            ].apply(lambda x: x[:4] if len(str(x)) == 5 else x)

        if self.target_classification_code.startswith("S"):
            self.concordance_table[
                self.target_classification_code
            ] = self.concordance_table[self.target_classification_code].apply(
                lambda x: x[:4] if len(str(x)) == 5 else x
            )

        self.concordance_table = self.concordance_table.drop_duplicates()
        for col in [self.classification_code, self.target_classification_code]:
            self.concordance_table[col] = self.concordance_table[col].apply(
                lambda x: str(int(x)) if pd.notna(x) else x
            )

        for clcode in [self.classification_code, self.target_classification_code]:
            # nans are dropped
            self.concordance_table = self.concordance_table[
                ~self.concordance_table[clcode].isna()
            ]
            # codes can have a leading zero
            mask = self.concordance_table[clcode].str.len() < self.clcode_length[clcode]
            self.concordance_table.loc[mask, clcode] = self.concordance_table.loc[
                mask, clcode
            ].str.zfill(self.clcode_length[clcode])
            # a concordance table can match to more than one digit length
            mask = self.concordance_table[clcode].str.len() > self.clcode_length[clcode]
            self.concordance_table.loc[mask, clcode] = self.concordance_table.loc[
                mask, clcode
            ].str[:-1]
            if len(self.concordance_table[clcode].str.len().unique()) > 1:
                raise ValueError(
                    f"More than one product digit level for {clcode}: {self.concordance_table[clcode].str.len().unique()}"
                )

            self.concordance_table = self.concordance_table.drop_duplicates()
        self.df = self.df.rename(columns={"commoditycode": self.classification_code})

    def _run_concordance(self) -> None:
        """ """
        self.concordance_table["cl_counts"] = self.concordance_table.groupby(
            self.classification_code
        )[self.classification_code].transform("count")
        self.concordance_table["eql_distribution"] = (
            1 / self.concordance_table["cl_counts"]
        )

        if (
            self.classification_code == "HS92"
            and self.target_classification_code == "SITC2"
        ):
            self.df.loc[
                self.df[self.classification_code].str.startswith("9999"),
                self.classification_code,
            ] = "999999"
        self.df = self.df.merge(
            self.concordance_table, on=[self.classification_code], how="left"
        )
        for trade_val in self.trade_val_cols:
            self.df.loc[:, trade_val] = self.df.eql_distribution * self.df[trade_val]

        if (
            self.classification_code == "HS92"
            and self.target_classification_code == "SITC2"
        ):
            self.df.loc[
                self.df[self.classification_code] == "XXXXXX",
                self.target_classification_code,
            ] = "XXXX"

        self.df = self.df.drop(columns=["eql_distribution", self.classification_code])
        self.df[["value_final", "value_exporter", "value_importer"]] = self.df[
            ["value_final", "value_exporter", "value_importer"]
        ].round(0)

        self.df = (
            self.df.groupby(
                ["year", "exporter", "importer", self.target_classification_code]
            )
            .agg(
                {"value_final": "sum", "value_exporter": "sum", "value_importer": "sum"}
            )
            .reset_index()
        )
        self.df = self.df.rename(
            columns={self.target_classification_code: "commoditycode"}
        )
