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
                # raw data set with Country Country Product Year
                self.raw_data_path,
                f"{self.product_classification}_{self.year}.parquet",
            )
        )

        # TODO: temp to reduce data set size
        self.df = self.df[self.df.product_level == 6]
        self.df = self.df[
            (self.df.reporter_iso.isin(["SAU", "IND", "CHL", "VEN", "ZWE"]))
            & (self.df.partner_iso.isin(["SAU", "IND", "CHL," "VEN", "ZWE"]))
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

        ccy_attractiveness = pd.read_parquet(  # pd.read_parquet(
            # TODO using weights file generated from seba's file, not python output
            f"data/intermediate/weights_{self.year}.parquet"
        )  # .parquet"
        ccy_attractiveness = ccy_attractiveness[
            (ccy_attractiveness.exporter.isin(["SAU", "IND", "CHL", "VEN", "ZWE"]))
            & (ccy_attractiveness.importer.isin(["SAU", "IND", "CHL", "VEN", "ZWE"]))
        ]

        # ccy_attractiveness = ccy_attractiveness[
        #     ccy_attractiveness.value_final >= 100_000
        # ]
        # generate idpairs
        cif_ratio = 0.2
        logging.info("set cif ratio")
        

        # generate index on unique country pairs
        ccy_attractiveness["idpair"] = ccy_attractiveness.groupby(
            ["exporter", "importer"]
        ).ngroup()
        country_pairs = ccy_attractiveness[["idpair", "exporter", "importer"]]
        
        # generate index for each product
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
            names=["exporter", "importer", "commodity_code"],
        )

        all_pairs_products = (
            pd.DataFrame(index=multi_index).query("importer != exporter").reset_index()
        ).drop_duplicates()
        
        npairs = all_pairs_products[['exporter', 'importer']].drop_duplicates().shape[0]

        # Step 2: Calculate the value of exports for each country pair and product
        exports = self.df[self.df["trade_flow"] == 2][
            ["reporter_iso", "partner_iso", "commodity_code", "trade_value"]
        ]
        # the reporter is the exporter
        exports.columns = ["exporter", "importer", "commodity_code", "export_value"]

        # export matrix has all products and all country pairs
        exports = all_pairs_products.merge(
            exports, on=["exporter", "importer", "commodity_code"], how="left"
        )

        # calculate the value of imports for each country pair and product
        imports = self.df[self.df["trade_flow"] == 1][
            ["reporter_iso", "partner_iso", "commodity_code", "trade_value"]
        ]
        imports.columns = ["importer", "exporter", "commodity_code", "import_value"]
        imports = all_pairs_products.merge(
            imports, on=["importer", "exporter", "commodity_code"], how="left"
        )

        # trade reconciliation
        exports_matrix = exports.fillna(0.0)
        imports_matrix = imports.fillna(0.0)

        exports_matrix = exports.pivot(
            index=["exporter", "importer"],
            columns="commodity_code",
            values="export_value",
        )

        imports_matrix = imports.pivot(
            index=["importer", "exporter"],
            columns="commodity_code",
            values="import_value",
        )
        
        # multiply imports by (1 - cif_ratio)
        # TODO: confirm may need to be array of cif_ratio, why 1?
        imports_matrix = imports_matrix * (1 - cif_ratio)
        imports_matrix = imports_matrix.swaplevel().sort_index()
        
        # at commodity bilateral level
        # score of 4 if reporter provides positive imports and exports
        # score of 2 if reporter only provides positive imports
        # score of 1 if reporter only provides positive exports
        trdata = pd.DataFrame(
            1 * ((1 * (exports_matrix > 0) + 1 * (imports_matrix > 0)) > 1)
            + 1 * (exports_matrix > 0)
            + 2 * (imports_matrix > 0)
        )

        country_pairs_index = exports_matrix.index

        ccy_attractiveness = (
            ccy_attractiveness.set_index(["exporter", "importer"])
            .reindex(country_pairs_index)
            # .reset_index()
        )

        # country pair attractiveness
        weight_exporter = np.array(ccy_attractiveness["weight_exporter"].values.reshape(-1, 1))
        weight_importer = np.array(ccy_attractiveness["weight_importer"].values.reshape(-1, 1)) 
                #.swaplevel().sort_index().values.reshape(-1, 1))

        # score of 4 if exporter and importer weight both > 0 
        # score of 2 if importer weight only > 0
        # score of 1 if exporter weight only > 0 
        accuracy = (
            1 * ((1 * (weight_exporter > 0) + 1 * (weight_importer > 0)) > 1)
            + 1 * ((weight_exporter > 0))
            + 2 * ((weight_importer > 0))
        )

        # accuracy_array = accuracy.reshape(-1, 1)
        accuracy_matrix = np.ones((npairs, nprod)) * accuracy

        weight = np.array(ccy_attractiveness["weight"].values.reshape(-1, 1))
        weight_matrix = np.ones((npairs, nprod)) * weight
                
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

        df = melted_imports_matrix.merge(
            melted_exports_matrix,
            on=["exporter", "importer", "commodity_code"],
            how="left",
        )
                
        # df['final_value'] = (.5 * df['export_value']) + (.5 * df['import_value'])
                        
        weights = weight_matrix.reshape(-1,1)
        export_values = df['export_value'].values.reshape(-1,1)
        import_values = df['import_value'].values.reshape(-1,1)
        trdata_melted = pd.melt(trdata.reset_index(),
                         id_vars=['exporter', 'importer'],  
                         var_name='commodity_code', 
                         value_name='trade_score')
        
        import pdb
        pdb.set_trace()
        
        trdata = trdata_melted['trade_score'].values.reshape(-1,1)
        
        # accuracy_melted = pd.melt(accuracy_matrix.reset_index(),
        #          id_vars=['exporter', 'importer'],  
        #          var_name='commodity_code', 
        #          value_name='accuracy_score')

        accuracy = accuracy_matrix.reshape(-1,1) 
        
        import pdb
        pdb.set_trace()
                
        df['final_value']  = (
            # if trdata and accuracy are characterized as four then multiply e and i by weights respectively
            ((weights * export_values) + ((1 - weights) * import_values)) * ((trdata == 4) * (accuracy == 4))
            # only an import value (none)
            + (import_values * ((trdata == 2) *(accuracy == 2)))
            # only reported an import value 
            + (import_values * ((trdata == 2) * (accuracy == 4)))
            # only reported export values (none)
            + (export_values * ((trdata == 1) * (accuracy == 1)))
            # 
            + (export_values * ((trdata == 1) * (accuracy == 4)))
            + (import_values * ((trdata == 4) * (accuracy == 2)))
            + (export_values * ((trdata == 4) * (accuracy == 1)))
            + (0.5 * (export_values + import_values) * ((trdata == 4) * (accuracy == 0)))
            + (import_values * ((trdata == 2) * (accuracy == 0)))
            + (export_values * ((trdata == 1) * (accuracy == 0)))
            + (import_values * ((trdata == 2) * (accuracy == 1)))
            + (export_values * ((trdata == 1) * (accuracy == 2)))
        )

        import pdb
        pdb.set_trace()
        
        # reweight VF
        # VR = VF
        # VR = self.reweight(df['final_value'], cc_trade_total, nprod)
                
        # drop rows that don't have data
        df = df.loc[
            (df[["final_value", "import_value", "export_value"]] != 0.0).any(axis=1)
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
                x["final_value"]
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
            {"not_specified": "sum", "final_value": "sum"}
        )
        df_ns_handle["not_specified_trade_ratio"] = (
            df_ns_handle["not_specified"] / df_ns_handle["final_value"]
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

    def reweight(self, VF, trade_total, Nprod):
        """ """
        logging.info("REWEIGHTING...")
        cc_products_sum = np.sum(VF, axis=1)

        # determine if data trade discrepancies
        case_1 = 1 * (
            (
                np.where((trade_total / cc_traded_products_sum) > 1.20, 1, 0)
                + np.where((trade_total - cc_traded_products_sum) > 2.5 * 10**7, 1, 0)
                + np.where(trade_total > 10**8, 1, 0)
            )
            == 3
        )
        case_2 = 1 * (
            (
                np.where(value_final > 10**8, 1, 0) + np.where(cc_traded_products_sum < 10**5, 1, 0)
                == 2
            )
        )

        xxxx = 1 * ((case_1 + case_2) > 0)
        value_xxxx = (value_final - cc_traded_products_sum) * (xxxx == 1)
        value_reweight = value_final - value_xxxx

        # proportionally reweight products for each country country pair
        VR = VF - (VF * 1 * (VF < 1000))
        VR = VR / np.sum(VR, axis=1).to_numpy().reshape(-1, 1)
        VR = VR * value_reweight.to_numpy().reshape(-1, 1)

        VR = VR.fillna(0.0)
        VR.loc[:, "value_xxxx"] = value_xxxx
        return VR
