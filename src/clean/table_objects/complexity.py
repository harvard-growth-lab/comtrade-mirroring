import pandas as pd
from clean.table_objects.base import _AtlasCleaning
import os
import numpy as np

import logging
import dask.dataframe as dd
import cProfile
from ecomplexity import ecomplexity
from ecomplexity import proximity


logging.basicConfig(level=logging.INFO)


# complexity table
class Complexity(_AtlasCleaning):
    NOISY_TRADE = {
        "SITC": ["9310", "9610", "9710", "9999", "XXXX"],
        "S2": ["9310", "9610", "9710", "9999", "XXXX"],
        "H0": ["7108", "9999", "XXXX"],
        "H4": ["7108", "9999", "XXXX"],
        "H5": ["7108", "9999", "XXXX"],
    }

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)
        self.product_class = kwargs["product_classification"]
        self.year = year

        # load data
        aux_stats = pd.read_csv(
            os.path.join(self.raw_data_path, "auxiliary_statistics.csv"), sep="\t"
        )
        reliable_exporters = pd.read_stata(
            os.path.join(self.raw_data_path, "obs_atlas.dta")
        )
        # Import trade data from CID Atlas

        # self.df = pd.read_parquet(
        #     f"data/processed/{self.product_class}_{self.year}_country_country_product_year.parquet"
        # )
        logging.info("RUNNING STATA INPUTS")
        self.df = pd.read_stata("data/raw/H0_ccpy_2015.dta")
        
        try:
            self.df = self.df.rename(columns={"commodity_code":"commoditycode"})
        except:
            logging.info("update ccpy, can remove commodity code rename")
        self.df = self.df[["exporter", "importer", "commoditycode", "value_final"]]
        self.df = self.df.rename(columns={"value_final": "export_value"})
        # aggregate to four digit level
        self.df["commoditycode"] = self.df["commoditycode"].astype(str).str[:4]

        # show the import and export value for each exporter
        self.df = (
            self.df.groupby(["exporter", "importer", "commoditycode"])
            .sum()
            .reset_index()
        )

        imports = self.df.copy(deep=True)
        imports = imports[['importer', 'commoditycode', 'export_value']].groupby(['importer', 'commoditycode']).agg('sum').reset_index()
        imports = imports.rename(
            columns={"export_value": "import_value", 
                     "importer": "exporter"}
        )
        
        self.df = self.df[['exporter', 'commoditycode', 'export_value']].groupby(["exporter", "commoditycode"]).agg('sum').reset_index()


        self.df = self.df.merge(imports, on=["exporter", "commoditycode"], how="outer")
        # fill in all combinations of exporter and commodity code
        
        all_combinations_cp = pd.DataFrame(index=(pd.MultiIndex.from_product(
            [
                self.df["exporter"].unique(),
                self.df["commoditycode"].unique(),
            ],
            names=["exporter", "commoditycode"],
        )))

        self.df = all_combinations_cp.merge(
            self.df, on=["exporter", "commoditycode"], how="left"
        )

        self.df[['import_value', 'export_value']] = self.df[['import_value', 'export_value']].fillna(0.0)

        aux_stats = aux_stats[aux_stats.year == self.year]

        self.df = self.df.merge(
            aux_stats[["exporter", "population", "gdp_pc"]], on=["exporter"], how="left"
        )
        
        self.df['population'] = self.df['population'].fillna(0)

        self.df["inatlas"] = (
            self.df["exporter"].isin(reliable_exporters.exporter).astype(int)
        )
        
        self.df.loc[self.df['exporter'].isin(["SYR","HKG","GNQ"]),'inatlas'] = 0
        self.df.loc[self.df['exporter'].isin(["ARM","BHR","CYP","MMR","SWZ","TGO","BFA", "COD","LBR","SDN","SGP"]),'inatlas'] = 1

        total_by_country = (
            self.df[["exporter", "export_value"]]
            .groupby("exporter")
            .agg("sum")
            .reset_index()
        )
        total_by_commodity = (
            self.df[["commoditycode", "export_value"]]
            .groupby("commoditycode")
            .agg("sum")
            .reset_index()
        )

        drop_countries = total_by_country[total_by_country.export_value == 0.0][
            "exporter"
        ].tolist()
        drop_commodities = total_by_commodity[total_by_commodity.export_value == 0.0][
            "commoditycode"
        ].tolist()
        if drop_countries or drop_commodities:
            self.df = self.df[~self.df.exporter.isin(drop_countries)]
            self.df = self.df[~self.df.commoditycode.isin(drop_commodities)]

        # save all countries, 207 countries, stata fulldata
        # validated to full data
        self.save_parquet(self.df, "intermediate", f"{self.product_classification}_{self.year}_complexity_all_countries")

        # only reliable countries, subset of 123 countries        
        self.df = self.df[self.df.inatlas == 1]        
        
        self.df = (
            self.df.groupby(["exporter", "commoditycode"])
            .agg({"export_value": "sum", "population": "first", "gdp_pc": "first"})
            .reset_index()
        )

        # drop unknown trade
        self.df = self.df[
            ~self.df.commoditycode.isin(self.NOISY_TRADE[self.product_classification])
        ]
        
        # preserve section of stata
        self.save_parquet(self.df, "intermediate", f"{self.product_classification}_{self.year}_before_mcp")
        mcp = self.df.copy(deep=True)
        
        
        # mcp matrix, rca of 1 and greater
        mcp['rca'] = mcp["export_value"] / (mcp.groupby("exporter")['export_value'].transform('sum'))/ (mcp.groupby("commoditycode")['export_value'].transform('sum') / mcp.export_value.sum())
        
        mcp["mcp"] = np.where(mcp["rca"] >= 1, 1, 0)
        
        # Herfindahl-Hirschman Index Calculation
        mcp["HH_index"] = (
            mcp["export_value"]
            / (mcp.groupby("commoditycode")["export_value"].transform("sum"))
        ) ** 2
        # mcp becomes the count of cases where rca>=1 for each commoditycode

        # validated through RCA 
        logging.info("save to review rca values")
        self.save_parquet(mcp, "intermediate", f"{self.product_classification}_{self.year}_mcp_rca")
        
        mcp = (
            mcp[["commoditycode", "export_value", "HH_index", "mcp"]]
            .groupby("commoditycode")
            .agg("sum")
            .reset_index()
        )
        mcp["share"] = 100 * (mcp["export_value"] / mcp.export_value.sum())
        mcp = mcp.sort_values(by=["export_value"])
        mcp["cumul_share"] = mcp["share"].cumsum()
        mcp["eff_exporters"] = 1 / mcp["HH_index"]
        mcp = mcp.sort_values(by=["cumul_share"])

        # generate flags:
        mcp["flag_for_small_share"] = np.where(mcp["cumul_share"] <= 0.025, 1, 0)
        mcp["flag_for_few_exporters"] = np.where(mcp["eff_exporters"] <= 2, 1, 0)
        mcp["flag_for_low_ubiquity"] = np.where(mcp["mcp"] <= 2, 1, 0)

        mcp["exclude_flag"] = mcp[["flag_for_small_share", "flag_for_few_exporters", "flag_for_low_ubiquity"]].sum(axis=1)
        
        # )
        mcp["exclude_flag"] = (mcp["exclude_flag"] > 0).astype(int)
        mcp.loc[mcp["export_value"] < 1, "exclude_flag"] = 1
        
        # mcp matrix, VALDIDATED through this point
        logging.info("save intermediate mcp matrix")
        self.save_parquet(mcp, "intermediate", f"{self.product_classification}_{self.year}_mcp")

        # dropping products                 
        drop_products_list = (
            mcp[mcp.exclude_flag == 1]["commoditycode"].unique().tolist()
        )
        # drop least traded products
        self.df = self.df[~self.df["commoditycode"].isin(drop_products_list)]
        
        self.df["year"] = self.year
        
        # self.df = self.df.rename(columns = {"mcp": "mcp_input"})
        # pass mcp matrix into Shreyas's ecomplexity package
        trade_cols = {
            "time": "year",
            "loc": "exporter",
            "prod": "commoditycode",
            "val": "export_value",
            # "val": "mcp_input",
        }

        # calculate complexity, not mcp matrix
        logging.info("Calculating the complexity of selected countries and products")
        complexity_df = ecomplexity(
            self.df[["year", "exporter", "commoditycode", "export_value"]],
            # self.df[["year", "exporter", "commoditycode", "mcp_input"]],
            trade_cols,
            # presence_test="manual",
        )
        
        # ecomplexity output
        complexity_df = complexity_df.drop(columns=['year'])
        proximity_df = proximity(self.df, trade_cols)
        
        df_gdppc = self.df[["exporter", "gdp_pc"]].groupby("exporter").agg("first").T

        df_rca = complexity_df[["exporter", "commoditycode", "rca"]].pivot(
            values="rca", index="commoditycode", columns="exporter"
        )

        # todo: review values
        prody = (df_rca / df_rca.sum(axis=0)).mul(df_gdppc.iloc[0], axis=1)
        prody = prody.sum(axis=0)

        # RELIABLE COUNTRIES: pci, rca, eci
        df_pci = complexity_df[["exporter", "commoditycode", "pci"]].pivot(
            values="pci", index="commoditycode", columns="exporter"
        )
        # mata eci1 = (rca1:>=1):* pci1'

        df_eci = (df_rca >= 1).astype(int) * df_pci
        # mata eci1 = eci1 :/ rowsum( (rca1:>=1))
        df_eci = df_eci.div((df_rca >= 1).astype(int).sum(axis=1), axis=0)

        # shape is cols of commodity by rows of country exporters
        # mata eci1 = J(rows(rca1),cols(rca1),1) :* eci1
        df_eci_reliable = (
            pd.DataFrame(1, index=df_eci.index, columns=df_eci.columns) * df_eci
        )

        # egen byte tagp = tag(commoditycode)
        # qui sum pci1 if tagp
        # replace pci1 = (pci1 - r(mean))/r(sd)
        df_pci_reliable = (df_pci - np.mean(df_pci.iloc[:, 0])) / np.std(
            df_pci.iloc[:, 0]
        )
        keep_commodity_list = self.df.commoditycode.unique().tolist()

        # save  `selecteddata', replace
        # exporter cc exportval pop gdppc rca1 M density eci pci diversity ubiquity coi cog
        import pdb
        pdb.set_trace()
        
        # primary key exporter+commodity_code
        # exporter commoditycode export_value population gdp_pc rca1 M density1 eci1 pci1 diversity ubiquity coi cog
        # reliable countries added
        logging.info(
            f"shape of complexity df before reliable concat {complexity_df.shape}"
        )
        
        # import pdb
        # pdb.set_trace()
        
        dfs_reliable = {"rca_reliable": df_rca, "eci_reliable": df_eci_reliable, "pci_reliable": df_pci_reliable}
        
        import pdb
        pdb.set_trace()

        
        reliable_df = pd.concat([df.add_prefix(f"{name}_") for name, df in dfs_reliable.items()], join="inner", axis=1).reset_index()    
        
        import pdb
        pdb.set_trace()
        
        reliable_df = pd.wide_to_long(
            reliable_df,
            stubnames=list(dfs_reliable.keys()),
            i="commoditycode",
            j="exporter",
            sep="_",
            suffix="[A-Z]+"
        )
        
        complexity_df = complexity_df.merge(reliable_df, on=['exporter', 'commoditycode'], how='left').set_index(["exporter", "commoditycode"])


        # ALL COUNTRIES, drop least traded products
        all_countries = self.load_parquet("intermediate", f"{self.product_classification}_{self.year}_complexity_all_countries")[
            ["exporter", "commoditycode", "export_value", "import_value"]
        ]
        logging.info(f"all countries {all_countries.shape}")
        all_countries = all_countries[
            all_countries.commoditycode.isin(keep_commodity_list)
        ]
        logging.info(f"all countries {all_countries.shape} after dropped commodities")
        # fill in so all exporters match to all remaining commodity codes
        # fill na with zero

        num_commodities = len(keep_commodity_list)

        export_value_df = all_countries.pivot(
            values="export_value", columns="commoditycode", index="exporter"
        )
        df_rca_all = (export_value_df.div(export_value_df.sum(axis=1), axis=0)) / (
            export_value_df.sum(axis=0) / all_countries.export_value.sum()
        )

        mcp_all = (df_rca_all >= 1).astype(int)
        # mata eci2 = mcp :* pci1'
        df_eci_all = mcp_all.mul(df_pci_reliable.T, axis=1)
        # mata eci2 = rowsum(eci2) :/ rowsum(mcp)
        df_eci_all = (df_eci_all.sum(axis=1)).div(mcp_all.sum(axis=1))
        # mata eci2 = J(rows(eci2),rows(kp1d),1) :* eci2
        df_eci_all = (
            pd.DataFrame(1, index=df_eci_all.index, columns=df_rca.index)
        ).mul(df_eci_all, axis=0)

        # mata expy = rowsum((export_value:/rowsum(export_value))  :* prody)
        expy = (
            export_value_df.div(export_value_df.sum(axis=1), axis=0).mul(prody, axis=0)
        ).sum(axis=1)
        # mata expy = J(rows(export_value),cols(export_value),1) :* expy
        expy = pd.DataFrame(
            1, index=export_value_df.index, columns=export_value_df.columns
        ).mul(expy, axis=0)
        # mata prody = J(rows(export_value),cols(export_value),1) :* prody
        prody = pd.DataFrame(
            1, index=export_value_df.index, columns=export_value_df.columns
        ).mul(prody, axis=0)
        
        # import pdb
        # pdb.set_trace()
        # complexity_df = complexity_df.merge(prody, on=['exporter'], how='left')

        dfs_allc = {"prody": prody.T, "rca_all_countries": df_rca_all.T, "eci_all_countries": df_eci_all.T}
        logging.info(
            f"shape of complexity df before all countries {complexity_df.shape}"
        )
        
        allc_df = pd.concat([df.add_prefix(f"{name}_") for name, df in dfs_allc.items()], join="inner", axis=1).reset_index()
                
        
        import pdb
        pdb.set_trace()
        
        allc_df = pd.wide_to_long(
            allc_df,
            stubnames=list(dfs_allc.keys()),
            i="commoditycode",
            j="exporter",
            sep="_",
            suffix="[A-Z]+"
        )

        complexity_df = complexity_df.merge(allc_df, on=['exporter', 'commoditycode'], how='left').reset_index()
        import pdb
        pdb.set_trace()
        # complexity_df = all_countries.merge(complexity_df, on=['exporter', 'commoditycode'], how='left')
        
        complexity_df[f"tag_e"] = (~complexity_df["exporter"].duplicated()).astype(int)
        # replace eci = eci2-mean / std
        exporter_eci = complexity_df[complexity_df.tag_e == 1]["eci"]
        # is this only if fillna? TODO
        
        # import pdb
        # pdb.set_trace()
        
        complexity_df["eci_all_countries"] = (
            complexity_df["eci_all_countries"] - np.mean(exporter_eci)
        ) / np.std(exporter_eci)

        # All COUNTRIES, ALL PRODUCTS
        allcp = self.load_parquet("intermediate", f"{self.product_classification}_{self.year}_complexity_all_countries")[
            ["exporter", "commoditycode", "export_value", "gdp_pc", "import_value"]
        ]
        allcp[["export_value", "import_value"]] = allcp[
            ["export_value", "import_value"]
        ].fillna(0)
        all_products = set(allcp.commoditycode.tolist())
        num_products = len(all_products)

        # package into a function
        # mata export_value = colshape(export_value,`ni')
        export_value_allcp = allcp.pivot(
            values="export_value", columns="commoditycode", index="exporter"
        )
        # mata `income2use' = colshape(`income2use',`ni')
        gdppc_allcp = allcp.pivot(
            values="gdp_pc", columns="commoditycode", index="exporter"
        )
        # mata export_value[.,`ni'] = J(rows(export_value),1,0)
        export_value_allcp.iloc[:, -1] = 0
        # mata rca3 = (export_value :/ rowsum(export_value)) :/ (colsum(export_value):/sum(export_value))
        df_rca_allcp = (
            export_value_allcp.div(export_value_allcp.sum(axis=1), axis=0)
        ) / (export_value_allcp.sum(axis=0) / allcp.export_value.sum())

        # mata mcp3 = (rca3:>=1)
        mcp_allcp = (df_rca_allcp >= 1).astype(int)

        # mata pci3 = (rca3:>=1) :* eci2[.,1]
        pci_allcp = (df_rca_allcp >= 1).mul(df_eci_all.iloc[:, 0], axis=0)
        # mata pci3 = colsum(pci3)'
        # mata pci3 = pci3 :/ colsum(rca3:>=1)'
        pci_allcp = pci_allcp.sum(axis=0).div(
            ((df_rca_allcp >= 1).astype(int)).sum(axis=0).replace(0, np.nan)
        )
        pci_allcp = pd.DataFrame(
            1, index=df_rca_allcp.index, columns=df_rca_allcp.columns
        ).mul(pci_allcp.T)

        # mata prody3 = (rca3:/colsum(rca3)) :* `income2use'
        prody_allcp = df_rca_allcp.div(df_rca_allcp.sum(axis=0), axis=1).mul(
            gdppc_allcp
        )
        # prody_allcp = prody_allcp.sum(axis = 0)
        # mata prody3 = colsum(prody3)
        # mata prody3 = J(rows(export_value),cols(export_value),1) :* prody3
        prody_allcp = pd.DataFrame(
            1, index=export_value_allcp.index, columns=export_value_allcp.columns
        ).mul(prody_allcp.sum(axis=0))

        logging.info("Creating the product space for all countries & all products")
        # should these have any indices?

        # mata C = M'*M
        country = mcp_allcp.T @ mcp_allcp
        # mata S = J(Nps,Ncs,1)*M
        space = (
            pd.DataFrame(1, index=mcp_allcp.columns, columns=mcp_allcp.index)
            @ mcp_allcp
        )
        product_x = country.div(space)
        product_y = country.div(space.T)

        # mata proximity = (P1+P2 - abs(P1-P2))/2 - I(Nps)
        proximity_allcp = (
            product_x + product_y - abs(product_x + product_y) / 2
        ) - np.identity(mcp_allcp.shape[1])
        # mata density3 = proximity' :/ (J(Nps,Nps,1) * proximity')
        proximity_allcp = proximity_allcp.fillna(0)
        density_allcp = proximity_allcp.T.div(
            np.dot(
                np.ones((mcp_allcp.shape[1], mcp_allcp.shape[1]), dtype=int),
                proximity_allcp.T.values,
            )
        )
        # mata density3 = M * density3
        density_allcp = mcp_allcp @ density_allcp
        # mata opportunity_value =  ((density3:*(1 :- M)):*pci3)*J(Nps,Nps,1)
        opportunity_value = ((density_allcp.mul(1 - mcp_allcp)).mul(pci_allcp)).fillna(
            0.0
        ) @ (pd.DataFrame(1, index=mcp_allcp.columns, columns=mcp_allcp.columns))

        mcp_rows = mcp_allcp.shape[0]
        mcp_cols = mcp_allcp.shape[1]

        opportunity_gain = (np.ones((mcp_rows, mcp_cols), dtype=int) - mcp_allcp).mul(
            (np.ones((mcp_rows, mcp_cols), dtype=int) - mcp_allcp).fillna(0)
            @ (
                proximity_allcp
                * (
                    (
                        pci_allcp.iloc[0,].div(
                            (
                                (
                                    proximity_allcp @ np.ones((mcp_cols, 1), dtype=int)
                                ).iloc[:, 0]
                            ).replace(0, np.nan),
                            axis=0,
                        )
                    ).values.reshape(-1, 1)
                    @ np.ones((1, mcp_cols), dtype=int)
                )
            ).fillna(0)
        )

        # local rca3 pci3 M density3 prody3 opportunity_value
        
        
        dfs_allcp = {
            "rca_allcp": df_rca_allcp.T,
            "pci_allcp": pci_allcp.T,
            "mcp_allcp": mcp_allcp.T,
            "density_allcp": density_allcp.T,
            "expy": expy.T,
            "prody_allcp": prody_allcp.T,
            "opportunity_value": opportunity_value.T,
            "opportunity_gain": opportunity_gain.T,
        }
                
        df_allcp = pd.concat([df.add_prefix(f"{name}_") for name, df in dfs_allcp.items()], join="inner", axis=1).reset_index()
        
        import pdb
        pdb.set_trace()

        
        df_allcp = pd.wide_to_long(
            df_allcp,
            stubnames=list(dfs_allcp.keys()),
            i="commoditycode",
            j="exporter",
            sep="_",
            suffix="[A-Z]+",
        ).reset_index()
                              
        # import pdb
        # pdb.set_trace()
                            
        # complexity_df = complexity_df.set_index(["exporter", "commoditycode"])
        import pdb
        pdb.set_trace()
        complexity_df = complexity_df.merge(df_allcp, on=['exporter', 'commoditycode'], how='left')
        

        # foreach j in eci1 eci2 expy
        # egen temp = mean(`j'), by(exporter)
        # replace `j' = temp if `j'==.
        # complexity_df = complexity_df.rename(
        #     columns={"level_0": "exporter", "level_1": "commoditycode"}
        # )
        # complexity_df = allcp.merge(complexity_df, on=['exporter', 'commoditycode'])
        logging.info(f"shape of complexity df after all c and p {complexity_df.shape}")

        complexity_df[
            ["eci_reliable", "eci_all_countries", "expy"]
        ] = complexity_df.groupby("exporter")[
            ["eci_reliable", "eci_all_countries", "expy"]
        ].transform(
            lambda x: x.fillna(x.mean())
        )

        # replace pci3 = (pci3 - r(mean))/r(sd) if pci3!=.
        mean_pci = np.mean(
            complexity_df.groupby("commoditycode")["pci_allcp"].agg("first")
        )
        stdev_pci = np.std(
            complexity_df.groupby("commoditycode")["pci_allcp"].agg("first")
        )
        complexity_df["pci_allcp"] = (complexity_df["pci_allcp"] - mean_pci) / stdev_pci

        # replace opportunity_value = (opportunity_value-r(mean))/r(sd) if opportunity_value!=.
        mean_oppval = np.mean(
            complexity_df.groupby("exporter")["opportunity_value"].agg("first")
        )
        stdev_oppval = np.std(
            complexity_df.groupby("exporter")["opportunity_value"].agg("first")
        )
        complexity_df["opportunity_value"] = (
            complexity_df["opportunity_value"] - mean_pci
        ) / stdev_pci

        logging.info("combine variables")
        measures = {
            "rca": ["rca_reliable", "rca_all_countries", "rca_allcp"],
            "eci": ["eci_reliable", "eci_all_countries"],
            "pci": ["pci_reliable", "pci_allcp"],
            "prody": ["prody_allcp"],
            "density": ["density_allcp"],  # density output directly from ecomplexity
            "oppval": ["coi", "opportunity_value"],
            "oppgain": ["cog", "opportunity_gain"],
        }

        for measure, replacement_vals in measures.items():
            # import pdb
            # pdb.set_trace()
            complexity_df[measure] = complexity_df[replacement_vals].bfill(axis=1).iloc[:,0]
            complexity_df = complexity_df.drop(columns=replacement_vals)

        # rename M mcp
        complexity_df = complexity_df.drop(columns=["mcp", "mcp_input"]).rename(
            columns={"mcp_allcp": "mcp"}
        )

        # cap gen distance = 1 - density
        complexity_df["distance"] = 1 - complexity_df["density"]
        complexity_df = complexity_df.drop(columns=["density"])
        
        import pdb
        pdb.set_trace()
        
        # drop any countries with export value == 0
        zero_val_exporters = complexity_df.groupby('exporter')['export_value'].sum() == 0
        if not zero_val_exporters.empty:
            complexity_df = complexity_df[~(complexity_df['exporter'].isin(zero_val_exporters['exporters'].tolist()))]
        
        # complexity_df.loc[complexity_df.groupby("exporter")['export_value'].transform('sum')==0
        # by_exporter = (
        #     complexity_df[["exporter", "export_value"]]
        #     .groupby("exporter")
        #     .agg("sum")
        #     .reset_index()
        # )
        # drop_countries = by_exporter[by_exporter.export_value == 0][
        #     "exporter"
        # ].to_list()
        # complexity_df = complexity_df[~complexity_df.exporter.isin(drop_countries)]

        # drop noisy commodity_codes
        complexity_df =  complexity_df[
            ~complexity_df.commoditycode.isin(
                self.NOISY_TRADE[self.product_classification]
            )
        ]
        
        self.df = complexity_df.copy()
        
        self.df = self.df.rename(columns={'commoditycode': 'commoditycode'})
        columns_to_keep = ['exporter', 'commoditycode', 'export_value', 'import_value', 'rca', 'mcp', 'eci', 'pci', 'oppval', 'oppgain', 'distance', 'prody', 'expy', 'gdp_pc']
        self.df = self.df[columns_to_keep]
        
        float32_columns = ['export_value', 'rca', 'eci', 'pci', 'oppval', 'oppgain', 'distance', 'prody', 'expy', 'gdp_pc']
        for col in float32_columns:
            self.df[col] = self.df[col].astype('float32')
        self.df['mcp'] = self.df['mcp'].astype('int8')
        self.df['inatlas'] = 1
        self.df['year'] = self.year
        self.df = self.df[['year', 'exporter', 'commoditycode', 'inatlas', 'export_value', 'import_value', 'rca', 'mcp', 'eci', 'pci', 'oppval', 'oppgain', 'distance', 'prody', 'expy', 'gdp_pc']]
