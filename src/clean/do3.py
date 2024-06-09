import pandas as pd
from clean.objects.base import _AtlasCleaning
import os
import numpy as np

# from sklearn.decomposition import PCA
import logging
import dask.dataframe as dd
import cProfile


logging.basicConfig(level=logging.INFO)


# CCPY country country product year table
class do3(_AtlasCleaning):

    SPECIALIZED_COMMODITY_CODES_BY_CLASS = {
        "H0": ["XXXXXX", "999999"],
        "H4": ["XXXXXX", "999999"],
        "S1": ["XXXX", "9999"],
        "S2": ["XXXX", "9999"],
    }
    TRADE_DATA_DISCREPANCIES = 0
    NOT_SPECIFIED = 1

    def __init__(self, year, product_classification, **kwargs):
        super().__init__(**kwargs)
        # cProfile.run()

        self.product_classification = product_classification
        self.year = year

        # Set parameters
        self.df = pd.read_parquet(
            os.path.join(
                self.raw_data_path, f"{self.product_classification}_{self.year}.parquet"
            )
        )

        # TODO: temp to reduce data set size
        self.df = self.df[self.df.product_level == 6]
        self.df = self.df[
            (self.df.reporter_iso.isin(["SAU", "IND", "CHL"]))
            & (self.df.partner_iso.isin(["SAU", "IND", "CHL"]))
        ]

        # creating country pairs and id of products

        # initial repetitive country code cleaning
        # repeat deal with germany/russia unification

        # drop commodity code totals
        self.df = self.df[self.df.commodity_code != "TOTAL"]
        drop_values = ["WLD", "NAN", "nan"]
        self.df = self.df[
            ~self.df.apply(
                lambda row: row["reporter_iso"] in drop_values
                or row["partner_iso"] in drop_values,
                axis=1,
                # meta=(None, "bool"),
            )
        ]

        # exports
        logging.info("exports table")

        ccy_attractiveness = pd.read_parquet(
            f"data/intermediate/weights_{self.year}.parquet"
        )
        ccy_attractiveness = ccy_attractiveness[
            (ccy_attractiveness.exporter.isin(["SAU", "IND", "CHL"]))
            & (ccy_attractiveness.importer.isin(["SAU", "IND", "CHL"]))
        ]

        ccy_attractiveness = ccy_attractiveness[
            ccy_attractiveness.value_final >= 100_000
        ]
        # generate idpairs
        cif_ratio = 0.8

        logging.info("set cif ratio")

        # Step 1: Index on country pair groups and product groups
        ccy_attractiveness["idpair"] = ccy_attractiveness.groupby(
            ["exporter", "importer"]
        ).ngroup()
        country_pairs = ccy_attractiveness[["idpair", "exporter", "importer"]]
        npairs = country_pairs.count().idpair
        self.df["idprod"] = self.df.groupby(["commodity_code"]).ngroup()
        products = self.df[["idprod", "commodity_code"]].drop_duplicates(
            subset=["idprod", "commodity_code"]
        )
        nprod = products.count().idprod

        multi_index = pd.MultiIndex.from_product(
            [
                country_pairs["exporter"],
                country_pairs["importer"],
                products["commodity_code"],
            ],
            names=["importer", "exporter", "commodity_code"],
        )

        all_pairs_products = (
            pd.DataFrame(index=multi_index).query("importer != exporter").reset_index()
        )

        all_pairs_products["idpair"] = pd.factorize(
            all_pairs_products[["exporter", "importer"]].apply(tuple, axis=1)
        )[0]

        # Step 2: Calculate the value of exports for each country pair and product
        exports = self.df[self.df["trade_flow"] == 2][
            ["reporter_iso", "partner_iso", "commodity_code", "trade_value"]
        ]
        exports.columns = ["exporter", "importer", "commodity_code", "export_value"]
        exports = all_pairs_products.merge(
            exports, on=["exporter", "importer", "commodity_code"], how="left"
        )

        exports = (
            exports.groupby(["exporter", "importer", "commodity_code"])["export_value"]
            .sum()
            .reset_index()
        )

        # exports = exports.merge(products, on="commodity_code", how="right")

        # Step 3: Calculate the value of imports for each country pair and product
        imports = self.df[self.df["trade_flow"] == 1][
            ["reporter_iso", "partner_iso", "commodity_code", "trade_value"]
        ]
        imports.columns = ["exporter", "importer", "commodity_code", "import_value"]
        imports = all_pairs_products.merge(
            imports, on=["exporter", "importer", "commodity_code"], how="left"
        )

        imports = (
            imports.groupby(["exporter", "importer", "commodity_code"])["import_value"]
            .sum()
            .reset_index()
        )
        imports.columns = ["exporter", "importer", "commodity_code", "import_value"]
        # imports = imports.merge(products, on="commodity_code", how="left")

        # trade reconciliation
        exports_matrix = exports.pivot(
            index=["importer", "exporter"],
            columns="commodity_code",
            values="export_value",
        ).fillna(0.0)

        imports_matrix = imports.pivot(
            index=["importer", "exporter"],
            columns="commodity_code",
            values="import_value",
        ).fillna(0.0)

        # multiply imports by (1 - cif_ratio)
        # TODO: confirm may need to be array of cif_ratio, why 1?
        imports_matrix = imports_matrix * (1 - cif_ratio)

        # flag indicators based on cases,
        trdata = (
            # positive exports and positive imports => 1
            1 * ((1 * (exports_matrix > 0) + 1 * (imports_matrix > 0)) > 1)
            # positive exports => 1
            + 1 * ((exports_matrix > 0))
            # positive imports => 2
            + 2 * ((imports_matrix > 0))
        )

        final_value = np.array(ccy_attractiveness["value_final"])
        # country pair attractiveness
        w_e = np.array(ccy_attractiveness["w_e"])  # .values.reshape(-1, 1))
        w_e_0 = np.array(ccy_attractiveness["w_e_0"])  # .values.reshape(-1, 1))
        w_i_0 = np.array(ccy_attractiveness["w_e_0"])  # .values.reshape(-1, 1))

        accuracy = (
            # attractiveness exports and attractiveness imports => 1
            1 * ((1 * (w_e_0 > 0) + 1 * (w_i_0 > 0)) > 1)
            + 1 * ((w_e_0 > 0))
            + 2 * ((w_i_0 > 0))
        )

        accuracy_array = accuracy.reshape(-1, 1)
        accuracy_matrix = np.ones((npairs, nprod)) * accuracy_array
        # accurary_array = accuracy_matrix.reshape(-1, 1)

        w_e = np.array(ccy_attractiveness["w_e"].values.reshape(-1, 1))
        w_e_matrix = np.ones((npairs, nprod)) * w_e
        # w_e_array = w_e_matrix.reshape(-1, 1)
        # size of array dictacted by number country pair ids, number product ids
        VF = (
            ((w_e_matrix * exports_matrix) + ((1 - w_e_matrix) * imports_matrix))
            * ((trdata == 4) * (accuracy_matrix == 4))
            + (imports_matrix * ((trdata == 2) * (accuracy_matrix == 2)))
            + (imports_matrix * ((trdata == 2) * (accuracy_matrix == 4)))
            + (exports_matrix * ((trdata == 1) * (accuracy_matrix == 1)))
            + (exports_matrix * ((trdata == 1) * (accuracy_matrix == 4)))
            + (imports_matrix * ((trdata == 4) * (accuracy_matrix == 2)))
            + (exports_matrix * ((trdata == 4) * (accuracy_matrix == 1)))
            + (
                0.5
                * (exports_matrix + imports_matrix)
                * ((trdata == 4) * (accuracy_matrix == 0))
            )
            + (imports_matrix * ((trdata == 2) * (accuracy_matrix == 0)))
            + (exports_matrix * ((trdata == 1) * (accuracy_matrix == 0)))
            + (imports_matrix * ((trdata == 2) * (accuracy_matrix == 1)))
            + (exports_matrix * ((trdata == 1) * (accuracy_matrix == 2)))
        )

        # reweight VF
        VR = self.reweight(VF, final_value, nprod)

        # melt the dataframes
        melted_imports_matrix = pd.melt(
            imports_matrix.reset_index(),
            id_vars=["importer", "exporter"],
            var_name="commodity_code",
            value_name="import_value",
        )
        melted_exports_matrix = pd.melt(
            exports_matrix.reset_index(),
            id_vars=["importer", "exporter"],
            var_name="commodity_code",
            value_name="export_value",
        )
        melted_VR = pd.melt(
            VR.reset_index(),
            id_vars=["importer", "exporter"],
            var_name="commodity_code",
            value_name="VR",
        )

        df = pd.merge(
            melted_VR,
            melted_imports_matrix,
            on=["exporter", "importer", "commodity_code"],
            how="left",
        )
        df = pd.merge(
            df,
            melted_exports_matrix,
            on=["exporter", "importer", "commodity_code"],
            how="left",
        )

        # drop rows that don't have data
        df = df.loc[
            (df[["VR", "import_value", "export_value"]] != 0.0).any(axis=1)
            & df.notnull().any(axis=1)
        ]

        df["commodity_code"] = df["commodity_code"].fillna(
            self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[self.product_classification][
                self.TRADE_DATA_DISCREPANCIES
            ]
        )

        not_specified_val = self.SPECIALIZED_COMMODITY_CODES_BY_CLASS[
            self.product_classification
        ][self.NOT_SPECIFIED]

        df_ns_handle = df.copy(deep=True)
        df_ns_handle["not_specified"] = df.apply(
            lambda x: (
                x["VR"]
                if x["commodity_code"] == not_specified_val and x["importer"] == "ANS"
                else 0
            ),
            axis=1,
        )
        df_ns_handle.loc[
            (df["commodity_code"] == not_specified_val) & (df["importer"] == "ANS"),
            "VR",
        ] = None

        df_ns_handle = df_ns_handle.groupby("exporter", as_index=False).agg(
            {"not_specified": "sum", "VR": "sum"}
        )
        df_ns_handle["not_specified_trade_ratio"] = (
            df_ns_handle["not_specified"] / df_ns_handle["VR"]
        )

        countries_with_too_many_ns = (
            df_ns_handle.loc[
                df_ns_handle["not_specified_trade_ratio"] > 1 / 3, "exporter"
            ]
            .unique()
            .tolist()
        )

        df = df[~df.exporter.isin(countries_with_too_many_ns)]

        df["year"] = self.year

        df.to_parquet(
            os.path.join(
                self.intermediate_data_path, "country_country_product_year.parquet"
            )
        )

    def reweight(self, VF, value_final, Nprod):
        """ """
        logging.info("REWEIGHTING...")
        sumVF = np.sum(VF, axis=1)

        case_1 = (
            np.where((value_final / sumVF) > 1.20, 1, 0)
            + np.where((value_final - sumVF) > 2.5 * 10**7, 1, 0)
            + np.where(value_final > 10**8, 1, 0)
        ) == 3
        case_2 = (
            np.where(value_final > 10**8, 1, 0) + np.where(sumVF < 10**5, 1, 0) == 2
        )
        xxxx = (case_1 + case_2) > 0

        # if cases are true, the difference of valuefinal and sumVF
        value_xxxx = (value_final - sumVF) * (xxxx == 1)
        value_reweight = value_final - value_xxxx

        # clear out VF less than 1_000
        VR = VF - VF * (VF < 1000)

        VR = VR.div(np.sum(VR, axis=1), axis=0, level=["importer", "exporter"])

        # align indices
        VR_aligned, value_reweight_aligned = VR.align(
            value_reweight, axis=0, level=[0, 1]
        )
        VR = VR_aligned.mul(value_reweight_aligned, axis=0)

        VR.loc[:, "value_xxxx"] = value_xxxx
        VR = VR.fillna(0.0)
        return VR
