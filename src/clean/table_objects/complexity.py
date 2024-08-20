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
        # # logging.info("RUNNING STATA INPUTS")
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

        # save all countries, 207 countries, stata fulldata, VALIDATED
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
        
        self.save_parquet(self.df[["year", "exporter", "commoditycode", "export_value"]], "intermediate", f"{self.product_classification}_{self.year}_ecomplexity_input")
        
        reliable_df = ecomplexity(
            self.df[["year", "exporter", "commoditycode", "export_value"]],
            # self.df[["year", "exporter", "commoditycode", "mcp_input"]],
            trade_cols,
            # presence_test="manual",
        )
        

        non_normalized_pci = pd.read_parquet("data/intermediate/H0_2015_nonnorm_pci_val.parquet")
        non_normalized_pci = non_normalized_pci.rename(columns={"pci":"pci_nonnorm"})
        
        reliable_df = reliable_df.merge(non_normalized_pci[['exporter', 'commoditycode', 'pci_nonnorm']], on=['exporter', 'commoditycode'], how='left')
        reliable_df = reliable_df.rename(columns={"pci":"pci_normalized", "pci_nonnorm":"pci"})
        
        # complexity matrix, VALIDATED OUTPUT
        logging.info("save complexity matrix")
        self.save_parquet(reliable_df, "intermediate", f"{self.product_classification}_{self.year}_complexitytest")
        
        # ecomplexity output
        reliable_df = reliable_df.drop(columns=['year'])
        proximity_df = proximity(self.df, trade_cols)
        
        self.df = self.df.merge(reliable_df.drop(columns='export_value'), on=['exporter', 'commoditycode'], how='left')

        # MATA variables: pci1, rca1, eci1 gets renamed to rca, pci, eci, density
        # df_rca = self.df[["exporter", "commoditycode", "rca"]].pivot(
        #      values="rca", index="commoditycode", columns="exporter").fillna(0)
        
        # reliable country data
        self.df['rca_reliable'] = self.df.rca.fillna(0)
        # a matrix shape of 1134 (number of commodities) in mata
        prody = self.df.groupby('commoditycode').apply(lambda x: (x['rca_reliable'] * x['gdp_pc'] / x['rca'].sum()).sum())
        
        self.df = self.df.rename(columns={"eci": "eci_reliable"})
        self.df['eci_normalized'] = np.where(self.df['rca_reliable'] >= 1, self.df['pci'], 0)
        rca_count = self.df.groupby('exporter')['rca_reliable'].apply(lambda x: (x>=1).sum())
        self.df = self.df.merge(rca_count.rename("rca_count"), on=['exporter'], how='left')
        self.df['eci_normalized'] = self.df['eci_normalized'] / self.df['rca_count']
        self.df.drop(columns='rca_count')
        
        pci = self.df.groupby('commoditycode')['pci_normalized'].agg('first')
        self.df['pci_reliable'] = ( self.df['pci_normalized'] - pci.mean() ) / pci.std()

        # VALIDATED through selected data
        self.save_parquet(self.df, "intermediate", f"{self.product_classification}_{self.year}_reliable_countries")

        keep_commodity_list = self.df.commoditycode.unique().tolist()

        # ALL COUNTRIES, Drop least traded products
        # reload full data for all countries
        all_countries = self.load_parquet("intermediate", f"{self.product_classification}_{self.year}_complexity_all_countries")[
            ["exporter", "commoditycode", "export_value", "import_value"]
        ]
        logging.info(f"all countries {all_countries.shape}")
        all_countries = all_countries[
            all_countries.commoditycode.isin(keep_commodity_list)
        ]
        logging.info(f"all countries {all_countries.shape} after dropped commodities")
        # fill in so all exporters match to all remaining commodity codes
        combinations = pd.DataFrame(index=(pd.MultiIndex.from_product(
            [
                all_countries["exporter"].unique(),
                all_countries["commoditycode"].unique(),
            ],
            names=["exporter", "commoditycode"],
        )))
        all_countries = combinations.merge(all_countries, on=['exporter', 'commoditycode'], how='left')
        # fill na with zero
        all_countries['export_value'] = all_countries['export_value'].fillna(0)
        
        num_commodities = len(keep_commodity_list)

        all_countries['rca'] = all_countries['export_value'].div(all_countries.groupby('exporter')['export_value'].transform('sum')).div(all_countries.groupby('commoditycode')['export_value'].transform('sum').div(all_countries['export_value'].sum()))

        all_countries['mcp'] = np.where(all_countries['rca'] >= 1, 1, 0)
        
        # COME BACK TO THIS, NEED NON NORMALIZED PCI VALUE
        all_countries = all_countries.merge(self.df[['commoditycode', 'pci_reliable', 'pci']].drop_duplicates(), on=['commoditycode'], how='left')
                
        all_countries = all_countries.sort_values(by=['exporter', 'commoditycode'])
        
        import pdb
        pdb.set_trace()

        all_countries['eci'] = all_countries['mcp'] * all_countries['pci']
        # all_countries['eci'] = all_countries['mcp'] * all_countries['pci_reliable']
        # grouped by exporter
        all_countries['eci'] = all_countries.groupby('exporter')['eci'].transform('sum') / (all_countries.groupby('exporter')['mcp'].transform('sum'))
        
        prody.name = 'prody'
        all_countries = all_countries.merge(prody, on=['commoditycode'],  how='left')
        
        all_countries['expy'] = (all_countries['export_value'] / (all_countries.groupby('exporter')['export_value'].transform('sum'))) * all_countries['prody']

        all_countries['expy'] = all_countries.groupby('exporter')['expy'].transform('sum')
        # self.df = all_countries.merge(self.df, on=['exporter', 'commoditycode'], how='left', suffixes=('', '_all_countries'))

        eci = all_countries.groupby('exporter')['eci'].agg('first')
        all_countries['eci'] = ( all_countries['eci'] - eci.mean() ) / eci.std()
        

        # FIX PCI with non-normalized value
        # save selecteddata
        self.save_parquet(all_countries, "intermediate", f"{self.product_classification}_{self.year}_all_countries")

        # All COUNTRIES, ALL PRODUCTS
        all_cp = self.load_parquet("intermediate", f"{self.product_classification}_{self.year}_complexity_all_countries")[
            ["exporter", "commoditycode", "export_value", "gdp_pc", "import_value"]
        ]
                                  
        combinations = pd.DataFrame(index=(pd.MultiIndex.from_product(
            [
                all_cp["exporter"].unique(),
                all_cp["commoditycode"].unique(),
            ],
            names=["exporter", "commoditycode"],
        )))
        all_cp = combinations.merge(all_cp, on=['exporter', 'commoditycode'], how='left')
        
        all_cp[["export_value", "import_value"]] = all_cp[
            ["export_value", "import_value"]
        ].fillna(0)
        
        ## fill in commoditycode 'XXXX' with all zeroes (last column with all zeroes)
        all_cp.loc[all_cp.commoditycode=="XXXX", 'export_value'] = 0
        
        all_cp = all_cp.sort_values(by=["exporter", "commoditycode"])
        
        all_cp['rca'] = ( all_cp['export_value'] / all_cp.groupby('exporter')['export_value'].transform('sum') ) / ( all_cp.groupby('commoditycode')['export_value'].transform('sum') / all_cp['export_value'].sum() )
         
        all_cp['mcp'] = np.where(all_cp['rca'] >= 1, 1, 0)
        
        import pdb
        pdb.set_trace()
        
        all_cp = all_cp.merge(all_countries[['exporter', 'eci']].drop_duplicates(), on=['exporter'], how='outer')
        all_cp = all_cp.rename(columns={"eci": "eci_all_countries"})
        
        import pdb
        pdb.set_trace()
        
        all_cp['pci'] = all_cp['mcp'] * all_cp['eci_all_countries']
        all_cp['pci'] = all_cp.groupby('commoditycode')['pci'].transform('sum') / all_cp.groupby('commoditycode')['mcp'].transform('sum')
        
        all_cp['prody'] = ( all_cp['rca'] / all_cp.groupby('commoditycode')['rca'].transform('sum') ) * all_cp['gdp_pc']
        all_cp['prody'] = all_cp.groupby('commoditycode')['prody'].transform('sum')
        
        # import pdb
        # pdb.set_trace()
        
        # STILL NEED UPDATED PCI, non normalized
        self.save_parquet(all_cp, "intermediate", f"{self.product_classification}_{self.year}_all_countries_all_products")  
                   
        logging.info("Creating the product space for all countries & all products")
        # mata C = M'*M
        all_cp_mcp = all_cp.pivot(index='exporter', columns='commoditycode', values='mcp')
        country = all_cp_mcp.T @ all_cp_mcp
        
        # mata S = J(Nps,Ncs,1)*M
        space = (
            pd.DataFrame(1, index=all_cp_mcp.columns, columns=all_cp_mcp.index)
            @ all_cp_mcp
        )
        product_x = country.div(space)
        product_y = country.div(space.T)

        # mata proximity = (P1+P2 - abs(P1-P2))/2 - I(Nps)
        all_cp_proximity = (
            product_x + product_y - abs(product_x + product_y) / 2
        ) - np.identity(all_cp_mcp.shape[1])
        # mata density3 = proximity' :/ (J(Nps,Nps,1) * proximity')
        all_cp_proximity = all_cp_proximity.fillna(0)
        all_cp_density = all_cp_proximity.T.div(
            np.dot(
                np.ones((all_cp_mcp.shape[1], all_cp_mcp.shape[1]), dtype=int),
                all_cp_proximity.T.values,
            )
        )
        all_cp_pci = all_cp.pivot(index='exporter', columns='commoditycode', values='pci')
        # mata density3 = M * density3
        all_cp_density = all_cp_mcp @ all_cp_density
        # mata opportunity_value =  ((density3:*(1 :- M)):*pci3)*J(Nps,Nps,1)
        opportunity_value = ((all_cp_density.mul(1 - all_cp_mcp)).mul(all_cp_pci)).fillna(
            0.0
        ) @ (pd.DataFrame(1, index=all_cp_mcp.columns, columns=all_cp_mcp.columns))

        mcp_rows, mcp_cols = all_cp_mcp.shape

        opportunity_gain = (np.ones((mcp_rows, mcp_cols), dtype=int) - all_cp_mcp).mul(
            (np.ones((mcp_rows, mcp_cols), dtype=int) - all_cp_mcp).fillna(0)
            @ (
                all_cp_proximity
                * (
                    (
                        all_cp_pci.iloc[0,].div(
                            (
                                (
                                    all_cp_proximity @ np.ones((mcp_cols, 1), dtype=int)
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
                
        all_cp_metrics = {
            "density": all_cp_density.T,
            "opportunity_value": opportunity_value.T,
            "opportunity_gain": opportunity_gain.T,
        }
                
        all_cp_metrics_df = pd.concat([df.add_prefix(f"{name}_") for name, df in all_cp_metrics.items()], join="inner", axis=1).reset_index()
        
        all_cp_metrics_df = pd.wide_to_long(
            all_cp_metrics_df,
            stubnames=list(all_cp_metrics.keys()),
            i="commoditycode",
            j="exporter",
            sep="_",
            suffix="[A-Z]+",
        ).reset_index()
                              
        all_cp = all_cp.merge(all_cp_metrics_df, on=['exporter', 'commoditycode'], how='left')
        
        
        logging.info(f"shape of complexity df after all c and p {reliable_df.shape}")
        # merge all the data sets self.df, all_countries, all_cp
        
        self.df = self.df.rename(columns={'mcp':'mcp_reliable'})
        use_cols = ['exporter', 'commoditycode','diversity', 'ubiquity', 'mcp_reliable', 'eci_reliable', 'pci','density', 'coi', 'cog', 'rca', 'rca_reliable', 'eci_normalized', 'pci_reliable']
        self.df = all_cp.merge(self.df[use_cols], on=['exporter', 'commoditycode'], how='left', suffixes=('_all_cp',''))
        
        # all_countries = all_countries.rename(columns={"rca":"rca_all_countries",
        #                                               "mcp": "mcp_all_countries",
        #                                               "eci":"eci_all_countries",
        #                                               "prody": "prody_all_countries",
        #                                               "expy": "expy_all_countries"
        use_cols = ['exporter', 'commoditycode', 'rca', 'mcp', 'eci', 'prody', 'expy']
        self.df = self.df.merge(all_countries[use_cols], on=['exporter', 'commoditycode'], how='left', suffixes=('','_all_countries'))
        
        self.df[
            ["eci_reliable", "eci_all_countries", "expy"]
        ] = self.df.groupby("exporter")[
            ["eci_reliable", "eci_all_countries", "expy"]
        ].transform(
            lambda x: x.fillna(x.mean())
        )

        # replace pci3 = (pci3 - r(mean))/r(sd) if pci3!=.
        pci = self.df.groupby("commoditycode")["pci_all_cp"].agg("first")

        self.df["pci_all_cp"] = np.where(self.df["pci_all_cp"].isna(), (self.df["pci_all_cp"] - np.mean(pci)) / np.std(pci), self.df["pci_all_cp"])
        
        # replace opportunity_value = (opportunity_value-r(mean))/r(sd) if opportunity_value!=. 
        opp_val = self.df.groupby("exporter")["opportunity_value"].agg("first")
        self.df["opportunity_value"] = np.where(self.df["opportunity_value"].isna(), (self.df["opportunity_value"] - np.mean(opp_val)) / np.std(opp_val), self.df["opportunity_value"])
        
        

        logging.info("combine variables")
        measures = {
            "rca": ["rca_reliable", "rca_all_countries", "rca_all_cp"],
            "eci": ["eci_reliable", "eci_all_countries"],
            "pci": ["pci_reliable", "pci_all_cp"],
            "prody": ["prody_all_countries"],
            "density": ["density_all_cp"],  # density output directly from ecomplexity
            "oppval": ["coi", "opportunity_value"],
            "oppgain": ["cog", "opportunity_gain"],
        }

        for measure, replacement_vals in measures.items():
            # import pdb
            # pdb.set_trace()
            self.df[measure] = self.df[replacement_vals].bfill(axis=1).iloc[:,0]
            self.df = self.df.drop(columns=replacement_vals)

        # rename M mcp
        self.df = self.df.rename(
            columns={"mcp_all_cp": "mcp"}
        )

        # cap gen distance = 1 - density
        self.df["distance"] = 1 - self.df["density"]
        self.df = self.df.drop(columns=["density"])
        
        # drop any countries with export value == 0
        # zero_val_exporters = self.df.groupby('exporter')['export_value'].sum() == 0
        # if not zero_val_exporters.empty:
        #     self.df = self.df[~(self.df['exporter'].isin(zero_val_exporters['exporter'].tolist()))]
        
        # self.df.loc[self.df.groupby("exporter")['export_value'].transform('sum')==0
        # by_exporter = (
        #     self.df[["exporter", "export_value"]]
        #     .groupby("exporter")
        #     .agg("sum")
        #     .reset_index()
        # )
        # drop_countries = by_exporter[by_exporter.export_value == 0][
        #     "exporter"
        # ].to_list()
        # self.df = self.df[~self.df.exporter.isin(drop_countries)]

        # drop noisy commodity_codes
        self.df =  self.df[
            ~self.df.commoditycode.isin(
                self.NOISY_TRADE[self.product_classification]
            )
        ]
                
        columns_to_keep = ['exporter', 'commoditycode', 'export_value', 'import_value', 'rca', 'mcp', 'eci', 'pci', 'oppval', 'oppgain', 'distance', 'prody', 'expy', 'gdp_pc']
        self.df = self.df[columns_to_keep]
        
        float32_columns = ['export_value', 'rca', 'eci', 'pci', 'oppval', 'oppgain', 'distance', 'prody', 'expy', 'gdp_pc']
        for col in float32_columns:
            self.df[col] = self.df[col].astype('float32')
        self.df['mcp'] = self.df['mcp'].astype('int8')
        self.df['inatlas'] = 1
        self.df['year'] = self.year
        self.df = self.df[['year', 'exporter', 'commoditycode', 'inatlas', 'export_value', 'import_value', 'rca', 'mcp', 'eci', 'pci', 'oppval', 'oppgain', 'distance', 'prody', 'expy', 'gdp_pc']]
