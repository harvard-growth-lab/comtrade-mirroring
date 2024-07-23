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
        "S2": ["9310", "9610", "9710", "9999", "XXXX"],
        "H0": ["7108", "9999", "XXXX"],
        "H4": ["7108", "9999", "XXXX"],
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

        self.df = pd.read_parquet(
            f"data/processed/country_country_product_{self.year}.parquet"
        )
        self.df = self.df[["exporter", "importer", "commodity_code", "final_value"]]
        self.df = self.df.rename(columns={"final_value": "export_value"})
        # aggregate to four digit level
        self.df["commodity_code"] = self.df["commodity_code"].astype(str).str[:4]

        # show the import and export value for each exporter
        self.df = (
            self.df.groupby(["exporter", "importer", "commodity_code"])
            .sum()
            .reset_index()
        )
        
        imports = self.df.copy(deep=True)
        imports = imports.drop(columns=["exporter"]).rename(
            columns={"export_value": "import_value", "importer": "exporter"}
        )
        imports = imports.groupby(["exporter", "commodity_code"]).sum().reset_index()

        self.df = self.df.drop(columns=["importer"])
        
        self.df = self.df.merge(imports, on=["exporter", "commodity_code"], how="left")
        self.df = self.df.groupby(["exporter", "commodity_code"]).sum().reset_index()
        
        # fillin all combinations of exporter and commodity code, all combinations
        # may filter ... 
        
        aux_stats = aux_stats[aux_stats.year==self.year]

        self.df = self.df.merge(
            aux_stats[["exporter", "population", "gdp_pc"]], on=["exporter"], how="left"
        )
        self.df = self.df.fillna(0.0)

        self.df["reliable"] = (
            self.df["exporter"].isin(reliable_exporters.exporter).astype(bool)
        )
        # reliable=False for "SYR", "GNQ", used to be HKG (now added to Atlas)
        # reliable=True for "ARM","BHR","CYP","MMR","SWZ","TGO","BFA" "COD","LBR","SDN","SGP"

        total_by_country = (
            self.df[["exporter", "export_value"]]
            .groupby("exporter")
            .agg("sum")
            .reset_index()
        )
        total_by_commodity = (
            self.df[["commodity_code", "export_value"]]
            .groupby("commodity_code")
            .agg("sum")
            .reset_index()
        )
                
        drop_countries = total_by_country[total_by_country.export_value == 0.0][
            "exporter"
        ].tolist()
        drop_commodities = total_by_commodity[total_by_commodity.export_value == 0.0][
            "commodity_code"
        ].tolist()
        if drop_countries or drop_commodities:
            self.df = self.df[~self.df.exporter.isin(drop_countries)]
            self.df = self.df[~self.df.commodity_code.isin(drop_commodities)]
            
        # save all countries, 207 countries, stata fulldata
        self.save_parquet(self.df, "intermediate", "complexity_all_countries")

        # only reliable countries, subset of 123 countries
        self.df = self.df[self.df.reliable == True]

        self.df = (
            self.df.groupby(["exporter", "commodity_code"])
            .agg({"export_value": "sum", "population": "first", "gdp_pc": "first"})
            .reset_index()
        )

        # drop unknown trade
        self.df = self.df[
            ~self.df.commodity_code.isin(self.NOISY_TRADE[self.product_classification])
        ]

        # mcp matrix, rca of 1 and greater

        self.df["by_commodity_code"] = self.df.groupby("commodity_code")[
            "export_value"
        ].transform("sum")
        self.df["by_exporter"] = self.df.groupby("exporter")["export_value"].transform(
            "sum"
        )
        # rca calculation, a commodity's percentage of a country export basket in comparison to the
        # export value for the product in global trade
        self.df["rca"] = (self.df["export_value"] / self.df["by_exporter"]) / (
            self.df["by_commodity_code"] / self.df["export_value"].sum()
        )
        # self.df matrix binary?
        self.df["mcp"] = np.where(self.df["rca"] >= 1, 1, 0)
        mcp = self.df.copy(deep=True)

        # Herfindahl-Hirschman Index Calculation
        mcp["HH_index"] = (
            mcp["export_value"]
            / mcp.groupby("commodity_code")["export_value"].transform("sum")
        ) ** 2
        # mcp becomes the count of cases where rca>=1 for each commoditycode

        mcp = (
            mcp[["commodity_code", "export_value", "HH_index", "mcp"]]
            .groupby("commodity_code")
            .agg("sum")
            .reset_index()
        )
        mcp["share"] = 100 * (mcp["export_value"] / mcp.export_value.sum())
        mcp = mcp.sort_values(by=["export_value"])
        mcp["cumul_share"] = mcp["share"].cumsum()
        mcp["eff_exporters"] = 1 / mcp["HH_index"]

        # generate flags:
        mcp["flag_for_small_share"] = np.where(mcp["cumul_share"] <= 0.025, 1, 0)
        mcp["flag_for_few_exporters"] = np.where(mcp["eff_exporters"] <= 2, 1, 0)
        mcp["flag_for_low_ubiquity"] = np.where(mcp["mcp"] <= 2, 1, 0)

        mcp["exclude_flag"] = (
            mcp["flag_for_small_share"]
            + mcp["flag_for_few_exporters"]
            + mcp["flag_for_low_ubiquity"]
        )
        mcp["exclude_flag"] = (mcp["exclude_flag"] > 0).astype(int)
        mcp.loc[mcp["export_value"] < 1, "exclude_flag"] = 1

        drop_products_list = (
            mcp[mcp.exclude_flag == 1]["commodity_code"].unique().tolist()
        )
        # drop least traded products
        self.df = self.df[~self.df["commodity_code"].isin(drop_products_list)]
        self.df["year"] = self.year

        # pass mcp matrix into Shreyas's ecomplexity package
        trade_cols = {
            "time": "year",
            "loc": "exporter",
            "prod": "commodity_code",
            "val": "export_value",
        }

        # calculate complexity, not mcp matrix
        logging.info("Calculating the complexity of selected countries and products")
        complexity_df = ecomplexity(
            self.df[["year", "exporter", "commodity_code", "export_value"]],
            trade_cols,
            # presence_test="manual",
        )

        # calculate proximity
        proximity_df = proximity(self.df, trade_cols)
        
        df_gdppc = self.df[['exporter', 'gdp_pc']].groupby('exporter').agg('first').T
        
        df_rca = complexity_df[['exporter', 'commodity_code', 'rca']].pivot(values='rca', index='commodity_code', columns='exporter')
        
        # todo: review values
        prody = (df_rca / df_rca.sum(axis=0)).mul(df_gdppc.iloc[0], axis=1)
        prody = prody.sum(axis=0)        
        
        # RELIABLE COUNTRIES: pci, rca, eci 
        df_pci = complexity_df[['exporter', 'commodity_code', 'pci']].pivot(values='pci', index='commodity_code', columns='exporter')
        # mata eci1 = (rca1:>=1):* pci1'
        
        df_eci = (df_rca >= 1).astype(int) * df_pci
        # mata eci1 = eci1 :/ rowsum( (rca1:>=1))
        df_eci = df_eci.div( (df_rca >= 1).astype(int).sum(axis=1), axis=0)
        
        # shape is cols of commodity by rows of country exporters
        # mata eci1 = J(rows(rca1),cols(rca1),1) :* eci1
        df_eci_reliable = (pd.DataFrame(1, index=df_eci.index, columns=df_eci.columns) * df_eci)
        
                
        # egen byte tagp = tag(commoditycode)
        # qui sum pci1 if tagp
        # replace pci1 = (pci1 - r(mean))/r(sd)
        df_pci_reliable = (df_pci - np.mean(df_pci.iloc[:,0])) / np.std(df_pci.iloc[:,0])
        keep_commodity_list = self.df.commodity_code.unique().tolist()
        
        
        # save  `selecteddata', replace 
        # primary key exporter+commodity_code
        # exporter commoditycode export_value population gdp_pc rca1 M density1 eci1 pci1 diversity ubiquity coi cog
        # reliable countries added 
        cols = [df_eci_reliable, df_pci_reliable]
        complexity_df = complexity_df.set_index(['exporter', 'commodity_code'])
        complexity_df.index.names = ['exporter','commodity_code']
        complexity_df = pd.concat([complexity_df] + [col.unstack() for col in cols], axis=1)
        rename_cols = list(complexity_df.columns[:-2]) + ['eci_reliable', 'pci_reliable']
        complexity_df.columns = rename_cols
        
        # ALL COUNTRIES, drop least traded products
        all_countries = self.load_parquet("intermediate", "complexity_all_countries")[['exporter', 'commodity_code', 'export_value']]
        logging.info(f"all countries {all_countries.shape}")
        all_countries = all_countries[all_countries.commodity_code.isin(keep_commodity_list)]
        logging.info(f"all countries {all_countries.shape} after dropped commodities")
        # fill in so all exporters match to all remaining commodity codes 
        # fill na with zero
        
        
        num_commodities = len(keep_commodity_list)
        
        export_value_df = all_countries.pivot(values='export_value', columns='commodity_code', index='exporter')
        df_rca_all = ( export_value_df.div(export_value_df.sum(axis=1), axis=0) ) / ( export_value_df.sum(axis=0) / all_countries.export_value.sum() )

        mcp_all = (df_rca_all >= 1).astype(int)
        # mata eci2 = mcp :* pci1'
        df_eci_all = mcp_all.mul(df_pci_reliable.T, axis=1)
        # mata eci2 = rowsum(eci2) :/ rowsum(mcp)
        df_eci_all = ( df_eci_all.sum(axis = 1) ).div(mcp_all.sum(axis=1) )
        # mata eci2 = J(rows(eci2),rows(kp1d),1) :* eci2
        df_eci_all = (pd.DataFrame(1, index=df_eci_all.index, columns=df_rca.index)).mul(df_eci_all, axis=0)
        
        # mata expy = rowsum((export_value:/rowsum(export_value))  :* prody)
        expy = (export_value_df.div(export_value_df.sum(axis=1), axis=0 ).mul( prody, axis=0 ) ).sum(axis=1)
        # mata expy = J(rows(export_value),cols(export_value),1) :* expy
        expy = pd.DataFrame(1, index=export_value_df.index, columns=export_value_df.columns).mul( expy, axis=0)
        # mata prody = J(rows(export_value),cols(export_value),1) :* prody
        prody = pd.DataFrame(1, index=export_value_df.index, columns=export_value_df.columns).mul( prody, axis=0)
        
        cols = [df_eci_all, df_rca_all]
        complexity_df = pd.concat([complexity_df] + [col.unstack() for col in cols], axis=1)
        rename_cols = list(complexity_df.columns[:-2]) + ['eci_all_countries', 'pci_all_countries']
        complexity_df.columns = rename_cols
        
        # update eci for all countries 
        complexity_df = complexity_df.reset_index()
        complexity_df = complexity_df.rename(columns={"level_0": "exporter", "level_1": "commodity_code"})
        complexity_df[f"tag_e"] = (~complexity_df['exporter'].duplicated()).astype(int)
        # replace eci = eci2-mean / std
        exporter_eci = complexity_df[complexity_df.tag_e==1]['eci']
        # is this only if fillna? TODO
        complexity_df['eci_all_countries'] =  (complexity_df['eci_all_countries'] - np.mean(exporter_eci)) / np.std(exporter_eci)
        

        # All COUNTRIES, ALL PRODUCTS
        allcp = self.load_parquet("intermediate", "complexity_all_countries")[['exporter', 'commodity_code', 'export_value', 'gdp_pc', 'import_value']]
        allcp[['export_value', 'import_value']] = allcp[['export_value', 'import_value']].fillna(0)
        all_products = set(allcp.commodity_code.tolist())
        num_products = len(all_products)
        
        # package into a function
        # mata export_value = colshape(export_value,`ni')
        export_value_allcp = allcp.pivot(values='export_value', columns='commodity_code', index='exporter')
        # mata `income2use' = colshape(`income2use',`ni')
        gdppc_allcp = allcp.pivot(values='gdp_pc', columns='commodity_code', index='exporter')
        # mata export_value[.,`ni'] = J(rows(export_value),1,0)
        export_value_allcp.iloc[:, -1] = 0
        # mata rca3 = (export_value :/ rowsum(export_value)) :/ (colsum(export_value):/sum(export_value))
        df_rca_allcp = ( export_value_allcp.div(export_value_allcp.sum(axis=1), axis=0) ) / ( export_value_allcp.sum(axis=0) / allcp.export_value.sum() )

        # mata mcp3 = (rca3:>=1)
        mcp_allcp = (df_rca_allcp >= 1).astype(int)
        
        # mata pci3 = (rca3:>=1) :* eci2[.,1]
        pci_allcp = (df_rca_allcp >= 1).mul(df_eci_all.iloc[:, 0], axis = 0)
        # mata pci3 = colsum(pci3)'
        # mata pci3 = pci3 :/ colsum(rca3:>=1)'
        pci_allcp = pci_allcp.sum(axis=0).div( ((df_rca_allcp >= 1).astype(int)).sum(axis = 0) )
        pci_allcp = pd.DataFrame(1, index=df_rca_allcp.index, columns=df_rca_allcp.columns).mul(pci_allcp.T)
        
        # mata prody3 = (rca3:/colsum(rca3)) :* `income2use'
        prody_allcp = df_rca_allcp.div( df_rca_allcp.sum(axis = 0), axis = 1 ).mul(gdppc_allcp)
        # prody_allcp = prody_allcp.sum(axis = 0) 
        # mata prody3 = colsum(prody3)
        # mata prody3 = J(rows(export_value),cols(export_value),1) :* prody3
        prody_allcp = pd.DataFrame(1, index=export_value_allcp.index, columns=export_value_allcp.columns).mul(prody_allcp.sum(axis = 0) )
                          
        logging.info("Creating the product space for all countries & all products")
        # should these have any indices?
        
        # mata C = M'*M
        country = mcp_allcp.T @ mcp_allcp 
        # mata S = J(Nps,Ncs,1)*M
        space = pd.DataFrame(1, index=mcp_allcp.columns, columns=mcp_allcp.index) @ mcp_allcp
        product_x = country.div(space)
        product_y = country.div(space.T)
        
        # mata proximity = (P1+P2 - abs(P1-P2))/2 - I(Nps)
        proximity_allcp = (product_x + product_y - abs(product_x + product_y)/2) - np.identity(mcp_allcp.shape[1])
        # mata density3 = proximity' :/ (J(Nps,Nps,1) * proximity')
        proximity_allcp = proximity_allcp.fillna(0)
        density_allcp = proximity_allcp.T.div( np.dot(np.ones((mcp_allcp.shape[1], mcp_allcp.shape[1]), dtype=int), proximity_allcp.T.values) )    
        # mata density3 = M * density3
        density_allcp = mcp_allcp @ density_allcp
        # mata opportunity_value =  ((density3:*(1 :- M)):*pci3)*J(Nps,Nps,1)
        opportunity_value = ( ( density_allcp.mul( 1 - mcp_allcp) ) .mul(pci_allcp) ).fillna(0.) @ (pd.DataFrame(1, index=mcp_allcp.columns, columns=mcp_allcp.columns))
        
        
        mcp_rows = mcp_allcp.shape[0]
        mcp_cols = mcp_allcp.shape[1]
        
        
        opportunity_gain = ( np.ones((mcp_rows, mcp_cols), dtype=int) - mcp_allcp ).mul(
        ( np.ones((mcp_rows, mcp_cols), dtype=int) - mcp_allcp ).fillna(0) @ (
            proximity_allcp * (
                (pci_allcp.iloc[0,].div
                 (
                    (
                        (proximity_allcp @ np.ones((mcp_cols,1), dtype=int)).iloc[:, 0]
                    ), axis= 0
                )
                ).values.reshape(-1,1) 
                @
                np.ones((1,mcp_cols), dtype=int) 
            )
        ).fillna(0)
        )
        
        import pdb
        pdb.set_trace()
        
        # local rca3 pci3 M density3 prody3 opportunity_value 
     
        cols = [df_rca_allcp, pci_allcp, mcp_allcp, density_allcp, expy, prody_allcp, opportunity_value]
        complexity_df = complexity_df.set_index(['exporter', 'commodity_code'])
        complexity_df.index.names = ['exporter','commodity_code']
        complexity_df = pd.concat([complexity_df] + [col.unstack() for col in cols], axis = 1)
        rename_cols = list(complexity_df.columns[:-7]) + ['rca_allcp', 'pci_allcp', 'mcp_allcp', 'density_allcp', 'expy', 'prody_allcp', 'opportunity_value']
        complexity_df.columns = rename_cols
        complexity_df = complexity_df.reset_index()
        
        import pdb
        pdb.set_trace()
        
        # foreach j in eci1 eci2 expy
            # egen temp = mean(`j'), by(exporter)
            # replace `j' = temp if `j'==.
        complexity_df = complexity_df.rename(columns={"level_0": "exporter", "level_1": "commodity_code"})
        
        complexity_df[['eci_reliable', 'eci_all_countries']] = complexity_df.groupby("exporter")[['eci_reliable', 'eci_all_countries', 'expy']].transform(lambda x: x.fillna(x.mean()))
        
        # replace pci3 = (pci3 - r(mean))/r(sd) if pci3!=. 
        # 
        df['pci_all_countries'] = df.groupby('exporter')['pci_all_countries'].transform(lambda x: (x - x.mean()) / x.std() if len(x) > 1 else x)

        # replace opportunity_value = (opportunity_value-r(mean))/r(sd) if opportunity_value!=. 
        df['opportunity_value'] = df.groupby('commodity_code')['opportunity_value'].transform(lambda x: (x - x.mean()) / x.std() if len(x) > 1 else x)
        
        logging.info("combine variables")
        
        complexity_df[['rca', 'rca_reliable', 'rca_all_countries', 'rca_allcp']] = complexity_df[['rca', 'rca_reliable', 'rca_all_countries', 'rca_allcp']].bfill(axis=1)
        
        import pdb
        pdb.set_trace()
        
        
        

        
        
          



        i = 0

        # impute for all countries with product restrictions
        # presence of each country across products (m matrix for all countries)
        # calculate avg pci of a countries exports by using previous complexity matrix
        #
