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

    def __init__(self, year, product_classification, **kwargs):
        super().__init__(**kwargs)
        # cProfile.run()

        self.product_classification = product_classification
        self.year = year

        # Set parameters
        self.df = dd.read_parquet(
            os.path.join(
                self.raw_data_path, f"{self.product_classification}_{self.year}.parquet"
            )
        )

        # TODO: temp to reduce data set size
        self.df = self.df[self.df.product_level == 2]
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
                meta=(None, "bool"),
            )
        ]

        # exports
        logging.info("exports table")

        weights = dd.read_parquet(f"data/intermediate/weights_{self.year}.dta")
        weights = weights[
            (weights.exporter.isin(["SAU", "IND", "CHL"]))
            & (weights.importer.isin(["SAU", "IND", "CHL"]))
        ]

        weights = weights[weights.value_final >= 100_000]
        # generate idpairs
        weights["cif_ratio"] = 0.8

        logging.info("set cif ratio")
        
        import pdb
        pdb.set_trace()

        # Create a DataFrame with all country pairs and products
        country_pairs = weights.drop_duplicates(subset=['importer', 'exporter'])[['importer', 'exporter']]
        products = self.df["commodity_code"].unique()
        
        # static file? 
        rows = pd.MultiIndex.from_product([range(len(country_pairs)), products],
                                          names=['pair_id', 'commodity_code'])
        all_cp = pd.DataFrame(index=rows).reset_index()
        all_cp = all_cp.merge(
            country_pairs.compute(), left_on="pair_id", right_index=True
        )

        # Calculate the value of exports for each country pair and product
        exports = (
            self.df[self.df["trade_flow"] == 2]
            .groupby(["reporter_iso", "partner_iso", "commodity_code"])["trade_value"]
            .sum()
            .reset_index()
        ).compute()

        exports.columns = ["exporter", "importer", "commodity_code", "export_value"]
        exports_all_cp = all_cp.merge(
            exports, on=["exporter", "importer", "commodity_code"], how="left"
        )

        # Calculate the value of imports for each country pair and product
        imports = (
            self.df[self.df["trade_flow"] == 1]
            .groupby(["partner_iso", "reporter_iso", "commodity_code"])["trade_value"]
            .sum()
            .reset_index()
        ).compute()
        imports.columns = ["exporter", "importer", "commodity_code", "import_value"]
        import pdb

        pdb.set_trace()

        imports_all_cp = all_cp.merge(
            imports, on=["exporter", "importer", "commodity_code"], how="left"
        )
        
        import pdb
        pdb.set_trace()

        # Step 4: Merge the w_e values from the weights DataFrame
        all_cp = all_cp.merge(
            weights[["exporter", "importer", "w_e_0"]].compute(),
            on=["exporter", "importer"],
            how="left",
        )

        import pdb

        pdb.set_trace()

        # Step 5: Fill missing values with 0 and select the desired columns
        all_cp = all_cp.fillna(0)
        all_cp = all_cp[
            [
                "exporter",
                "importer",
                "commodity_code",
                "export_value",
                "import_value",
                "w_e_0",
            ]
        ]

        import pdb

        pdb.set_trace()

        Me = all_cp["export_value"].to_numpy()
        Mi = all_cp["import_value"].to_numpy()
        w_e_0 = all_cp["w_e_0"].to_numpy()

        # Perform trade reconciliation calculations using matrix operations
        trdata = (1 * (((Me > 0) + (Mi > 0)) > 1)) + (1 * ((Me > 0))) + (2 * ((Mi > 0)))

        accuracy = (
            (1 * (((w_e_0 > 0) + (w_i_0 > 0)) > 1))
            + (1 * ((w_e_0 > 0)))
            + (2 * ((w_i_0 > 0)))
        )

        accuracy = np.ones((Nidpair, Nidprod)) * accuracy
        accuracy = accuracy.reshape(-1, 1)

        w_e = np.ones((Nidpair, Nidprod)) * w_e
        w_e = w_e.reshape(-1, 1)

        VF = (
            ((w_e * Me) + ((1 - w_e) * Mi)) * ((trdata == 4) * (accuracy == 4))
            + (Mi * ((trdata == 2) * (accuracy == 2)))
            + (Mi * ((trdata == 2) * (accuracy == 4)))
            + (Me * ((trdata == 1) * (accuracy == 1)))
            + (Me * ((trdata == 1) * (accuracy == 4)))
            + (Mi * ((trdata == 4) * (accuracy == 2)))
            + (Me * ((trdata == 4) * (accuracy == 1)))
            + (0.5 * (Me + Mi) * ((trdata == 4) * (accuracy == 0)))
            + (Mi * ((trdata == 2) * (accuracy == 0)))
            + (Me * ((trdata == 1) * (accuracy == 0)))
            + (Mi * ((trdata == 2) * (accuracy == 1)))
            + (Me * ((trdata == 1) * (accuracy == 2)))
        )

        # logging.info(f"size of df {self.df.shape}")
        # logging.info(f"size of weights table {weights.shape}")

    #     # list of all products
    #     self.df["idprod"] = self.df.map_partitions(
    #         lambda df: pd.factorize(df["commodity_code"])[0]
    #     )

    #     products = self.df[["commodity_code", "idprod"]].drop_duplicates()
    #     num_products = products.idprod.max()

    #     logging.info("generated list of all products")
    #     # list of all country pairs
    #     weights["idpair"] = weights.map_partitions(
    #         lambda df: df.groupby(["exporter", "importer"]).ngroup(),
    #         meta=("idpair", "int64"),
    #     )
    #     country_pairs = weights[["exporter", "importer", "idpair"]]

    #     logging.info("list of all country pairs")

    #     # merge onto exports table
    #     exports = exports.merge(country_pairs, on=["exporter", "importer"], how="inner")
    #     exports = exports.merge(products, on=["commodity_code"], how="inner").compute()
    #     import pdb

    #     pdb.set_trace()
    #     # merge onto imports table
    #     imports = imports.merge(country_pairs, on=["exporter", "importer"], how="inner")
    #     imports = imports.merge(products, on=["commodity_code"], how="inner").compute()

    #     import pdb

    #     pdb.set_trace()
    #     # filled matrix

    #     # trade reconciliation
    #     Mi = Mi * (1 - cif_ratio)
    #     del cif_ratio
    #     Me = Me.reshape(-1, 1)
    #     Mi = Mi.reshape(-1, 1)
    #     w_e = w_e.reshape(-1, 1)

    # def loadingMs(idc, idp, val, Npair, Nprod):
    #     M = np.zeros((Npair, Nprod))
    #     for i in range(len(idc)):
    #         r = idc[i] - 1  # Adjust index to 0-based
    #         c = idp[i] - 1  # Adjust index to 0-based
    #         M[r, c] = val[i]
    #     M = np.where(np.isnan(M), 0, M)  # Replace missing values with 0
    #     return M
