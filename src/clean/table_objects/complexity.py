import pandas as pd
from clean.table_objects.base import _AtlasCleaning
import os
import numpy as np

import logging
import dask.dataframe as dd
import cProfile

# using a cloned repo in order to return a non normalized pci value
# intermediate_pci_val
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

        self.load_data()

        self.preprocess_data()
        self.filter_countries_and_noisy_commodities()

        mcp = self.calculate_mcp()
        self.filter_least_traded_products(mcp)
        # delete mcp?

        self.calculate_ecomplexity()

        prody = self.most_reliable_country_metrics()

        all_countries = self.all_countries_less_least_traded_product_metrics(prody)

        all_cp = self.all_countries_all_products_metrics(all_countries)

        all_cp = self.growth_opportunity_metrics(all_cp)

        self.merge_data(all_cp, all_countries)

        self.standardize_metrics()

        self.backfill_complexity_by_data_reliability()

        self.prepare_output_data()

    def load_data(self):
        self.aux_stats = pd.read_csv(
            os.path.join(self.raw_data_path, "auxiliary_statistics.csv"), sep="\t"
        )
        self.reliable_exporters = pd.read_stata(
            os.path.join(self.raw_data_path, "obs_atlas.dta")
        )

        # TODO: include functionality for generating SITC ccpy with ~800 products
        if self.product_class == "SITC":
            self.handle_sitc()
        else:
            self.df = pd.read_parquet(
                f"data/processed/{self.product_class}_{self.year}_country_country_product_year.parquet"
            )

    def handle_sitc(self):
        """
        Due to product incompatibility across different released version of SITC,
        GL uses a baseline of ~800 products
        """
        self.df = pd.read_parquet(
            os.path.join(self.final_output_path, "SITC", f"SITC_{self.year}.parquet")
        )
        if "value_final" not in self.df.columns:
            self.df["value_final"] = self.df["export_value"]
            self.df = self.df.rename(
                columns={
                    "export_value": "value_exporter",
                    "import_value": "value_importer",
                }
            )
            self.df.to_parquet(
                os.path.join(
                    self.final_output_path, "SITC", f"SITC_{self.year}.parquet"
                )
            )

    def preprocess_data(self):
        """
        Manipulates final trade value to display, export and import value.
        Rectangularizes trade data and fills in nan values with zeroes. This step
        is a preprocessing step before using Growth Lab's py-ecomplexity package
        """
        # try:
        #     self.df = self.df.rename(columns={"commodity_code": "commoditycode"})
        # except:
        #     logging.info("update ccpy, can remove commodity code rename")

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
        imports = (
            imports[["importer", "commoditycode", "export_value"]]
            .groupby(["importer", "commoditycode"])
            .agg("sum")
            .reset_index()
        )
        imports = imports.rename(
            columns={"export_value": "import_value", "importer": "exporter"}
        )

        self.df = (
            self.df[["exporter", "commoditycode", "export_value"]]
            .groupby(["exporter", "commoditycode"])
            .agg("sum")
            .reset_index()
        )

        self.df = self.df.merge(imports, on=["exporter", "commoditycode"], how="outer")

        # TODO: fix so not returning self.df
        self.df = self.rectangularize_data(self.df)
        self.df[["import_value", "export_value"]] = self.df[
            ["import_value", "export_value"]
        ].fillna(0.0)

    def filter_countries_and_noisy_commodities(self):
        """
        Filter country and product list to highest level of data quality based on historical
        precedent.

        Based on a predetermined list of the historically most-reliable country reporters only
        include most reliable reporters. Remove traditionally noisy commodity codes.

        Return filtered dataframe
        """
        self.aux_stats = self.aux_stats[self.aux_stats.year == self.year]

        self.df = self.df.merge(
            self.aux_stats[["exporter", "population", "gdp_pc"]],
            on=["exporter"],
            how="left",
        )

        self.df["population"] = self.df["population"].fillna(0)

        self.df["inatlas"] = (
            self.df["exporter"].isin(self.reliable_exporters.exporter).astype(int)
        )

        # TODO: update the raw input data obs_atlas.dta file so changes aren't made in code
        self.df.loc[self.df["exporter"].isin(["SYR", "HKG", "GNQ"]), "inatlas"] = 0
        self.df.loc[
            self.df["exporter"].isin(
                [
                    "ARM",
                    "BHR",
                    "CYP",
                    "MMR",
                    "SWZ",
                    "TGO",
                    "BFA",
                    "COD",
                    "LBR",
                    "SDN",
                    "SGP",
                ]
            ),
            "inatlas",
        ] = 1

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

        # save all countries, 207 countries, will use later on
        self.save_parquet(
            self.df,
            "intermediate",
            f"{self.product_classification}_{self.year}_complexity_all_countries",
        )

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

    def calculate_mcp(self):
        """
        Calculates the revealed comparative advantage, we use Balassa’s definition,
        which says that a country is an effective exporter of a product if it exports
        more than its “fair share,” or a share that is at least equal to the share of total
        world trade that the product represents (RCA greater than 1) and groups the data
        by commodity codes in order to count the number of rca>=1 products
        """
        # preserve
        df = self.df.copy(deep=True)

        # mcp matrix, rca of 1 and greater
        df["rca"] = (
            df["export_value"]
            / (df.groupby("exporter")["export_value"].transform("sum"))
            / (
                df.groupby("commoditycode")["export_value"].transform("sum")
                / df.export_value.sum()
            )
        )

        df["mcp"] = np.where(df["rca"] >= 1, 1, 0)

        # Herfindahl-Hirschman Index Calculation
        df["HH_index"] = (
            df["export_value"]
            / (df.groupby("commoditycode")["export_value"].transform("sum"))
        ) ** 2

        # df becomes the count of cases where rca>=1 for each commoditycode
        return (
            df[["commoditycode", "export_value", "HH_index", "mcp"]]
            .groupby("commoditycode")
            .agg("sum")
            .reset_index()
        )

    def filter_least_traded_products(self, df):
        """
        Removes products based on three criteria:

            1. products share of world trade is less than 2.5%
            2. based on the Herfindahl-Hirschman Index effective exporters are 2 or less
            3. number of countries with a revealed comparative advantage for the product is 2 or less

        Returns filtered data frame
        """
        df["share"] = 100 * (df["export_value"] / df.export_value.sum())
        df = df.sort_values(by=["export_value"])
        df["cumul_share"] = df["share"].cumsum()
        df["eff_exporters"] = 1 / df["HH_index"]
        df = df.sort_values(by=["cumul_share"])

        # generate flags:
        df["flag_for_small_share"] = np.where(df["cumul_share"] <= 0.025, 1, 0)
        df["flag_for_few_exporters"] = np.where(df["eff_exporters"] <= 2, 1, 0)
        df["flag_for_low_ubiquity"] = np.where(df["mcp"] <= 2, 1, 0)

        df["exclude_flag"] = df[
            ["flag_for_small_share", "flag_for_few_exporters", "flag_for_low_ubiquity"]
        ].sum(axis=1)

        df["exclude_flag"] = (df["exclude_flag"] > 0).astype(int)
        df.loc[df["export_value"] < 1, "exclude_flag"] = 1

        # dropping products
        drop_products_list = df[df.exclude_flag == 1]["commoditycode"].unique().tolist()

        # drop least traded products
        self.df = self.df[~self.df["commoditycode"].isin(drop_products_list)]

    def calculate_ecomplexity(self):
        self.df["year"] = self.year

        # pass export value matrix into Shreyas's ecomplexity package
        trade_cols = {
            "time": "year",
            "loc": "exporter",
            "prod": "commoditycode",
            "val": "export_value",
            # "val": "mcp_input",
        }

        # calculate complexity
        logging.info("Calculating the complexity of selected countries and products")
        reliable_df = ecomplexity(
            self.df[["year", "exporter", "commoditycode", "export_value"]],
            # self.df[["year", "exporter", "commoditycode", "mcp_input"]],
            trade_cols,
            # presence_test="manual",
        )

        pci_df = ecomplexity(
            self.df[["year", "exporter", "commoditycode", "export_value"]],
            # self.df[["year", "exporter", "commoditycode", "mcp_input"]],
            trade_cols,
            output_normalized_pci=False,
        )

        pci_df = pci_df.rename(columns={"pci": "pci_nonnorm"})
        reliable_df = reliable_df.merge(
            pci_df[["exporter", "commoditycode", "pci_nonnorm"]],
            on=["exporter", "commoditycode"],
            how="left",
        )
        reliable_df = reliable_df.rename(
            columns={"pci": "pci_normalized", "pci_nonnorm": "pci"}
        )

        # ecomplexity output
        reliable_df = reliable_df.drop(columns=["year"])
        # proximity_df = proximity(self.df, trade_cols)

        self.df = self.df.merge(
            reliable_df.drop(columns="export_value"),
            on=["exporter", "commoditycode"],
            how="left",
        )

    def rectangularize_data(self, df):
        """
        Manipulates df shape for matrix multiplication. Each exporter has has row for each
        commodity code
        """

        combinations = pd.DataFrame(
            index=(
                pd.MultiIndex.from_product(
                    [
                        df["exporter"].unique(),
                        df["commoditycode"].unique(),
                    ],
                    names=["exporter", "commoditycode"],
                )
            )
        )
        df = combinations.merge(df, on=["exporter", "commoditycode"], how="left")
        # fill na with zero
        # df["export_value"] = all_countries["export_value"].fillna(0)
        return df

    def most_reliable_country_metrics(self):
        """
        Calculates prody, pci, and rca for the most reliable reporter countries
        """
        # MATA variables: pci1, rca1, eci1 gets renamed to rca, pci, eci, density

        # reliable country data
        self.df["rca_reliable"] = self.df.rca.fillna(0)
        # a matrix shape of 1134 (number of commodities) in mata
        prody = self.df.groupby("commoditycode").apply(
            lambda x: (x["rca_reliable"] * x["gdp_pc"] / x["rca"].sum()).sum()
        )

        self.df = self.df.rename(columns={"eci": "eci_reliable"})
        self.df["eci_normalized"] = np.where(
            self.df["rca_reliable"] >= 1, self.df["pci"], 0
        )
        rca_count = self.df.groupby("exporter")["rca_reliable"].apply(
            lambda x: (x >= 1).sum()
        )
        self.df = self.df.merge(
            rca_count.rename("rca_count"), on=["exporter"], how="left"
        )
        self.df["eci_normalized"] = self.df["eci_normalized"] / self.df["rca_count"]
        self.df.drop(columns="rca_count")

        pci = self.df.groupby("commoditycode")["pci_normalized"].agg("first")
        self.df["pci_reliable"] = (self.df["pci_normalized"] - pci.mean()) / pci.std()
        return prody

    def all_countries_less_least_traded_product_metrics(self, prody):
        keep_commodity_list = self.df.commoditycode.unique().tolist()

        # reload full data for all countries
        all_countries = self.load_parquet(
            "intermediate",
            f"{self.product_classification}_{self.year}_complexity_all_countries",
        )[["exporter", "commoditycode", "export_value", "import_value"]]

        all_countries = all_countries[
            all_countries.commoditycode.isin(keep_commodity_list)
        ]

        all_countries = self.rectangularize_data(all_countries)
        all_countries["export_value"] = all_countries["export_value"].fillna(0)

        num_commodities = len(keep_commodity_list)

        all_countries["rca"] = (
            all_countries["export_value"]
            .div(all_countries.groupby("exporter")["export_value"].transform("sum"))
            .div(
                all_countries.groupby("commoditycode")["export_value"]
                .transform("sum")
                .div(all_countries["export_value"].sum())
            )
        )

        all_countries["mcp"] = np.where(all_countries["rca"] >= 1, 1, 0)

        all_countries = all_countries.merge(
            self.df[["commoditycode", "pci_reliable", "pci"]].drop_duplicates(),
            on=["commoditycode"],
            how="left",
        )

        all_countries = all_countries.sort_values(by=["exporter", "commoditycode"])

        all_countries["eci"] = all_countries["mcp"] * all_countries["pci"]
        # all_countries['eci'] = all_countries['mcp'] * all_countries['pci_reliable']
        # grouped by exporter
        all_countries["eci"] = all_countries.groupby("exporter")["eci"].transform(
            "sum"
        ) / (all_countries.groupby("exporter")["mcp"].transform("sum"))

        prody.name = "prody"
        all_countries = all_countries.merge(prody, on=["commoditycode"], how="left")

        all_countries["expy"] = (
            all_countries["export_value"]
            / (all_countries.groupby("exporter")["export_value"].transform("sum"))
        ) * all_countries["prody"]

        all_countries["expy"] = all_countries.groupby("exporter")["expy"].transform(
            "sum"
        )

        eci = all_countries.groupby("exporter")["eci"].agg("first")

        # validated eci2, eci2 in mata is different than eci2 in stata
        all_countries["eci_all_countries"] = (
            all_countries["eci"] - eci.mean()
        ) / eci.std()

        return all_countries

    def all_countries_all_products_metrics(self, all_countries):
        """ """
        all_cp = self.load_parquet(
            "intermediate",
            f"{self.product_classification}_{self.year}_complexity_all_countries",
        )[["exporter", "commoditycode", "export_value", "gdp_pc", "import_value"]]

        all_cp = self.rectangularize_data(all_cp)
        all_cp[["export_value", "import_value"]] = all_cp[
            ["export_value", "import_value"]
        ].fillna(0)

        ## fill in commoditycode 'XXXX' with all zeroes (last column with all zeroes)
        all_cp.loc[all_cp.commoditycode == "XXXX", "export_value"] = 0

        # all_cp = all_cp.sort_values(by=["exporter", "commoditycode"])

        all_cp["rca"] = (
            all_cp["export_value"]
            / all_cp.groupby("exporter")["export_value"].transform("sum")
        ) / (
            all_cp.groupby("commoditycode")["export_value"].transform("sum")
            / all_cp["export_value"].sum()
        )

        all_cp["mcp"] = np.where(all_cp["rca"] >= 1, 1, 0)
        all_cp.loc[all_cp.commoditycode == "XXXX", "mcp"] = 1

        all_cp = all_cp.merge(
            all_countries[["exporter", "eci"]].drop_duplicates(),
            on=["exporter"],
            how="outer",
        )

        # all_cp = all_cp.sort_values(by=["exporter", "commoditycode"])

        # eci comes from mata, not the all_countries eci
        all_cp["pci"] = all_cp["mcp"] * all_cp["eci"]

        all_cp["pci"] = all_cp.groupby("commoditycode")["pci"].transform(
            "sum"
        ) / all_cp.groupby("commoditycode")["mcp"].transform("sum")

        all_cp["prody"] = (
            all_cp["rca"] / all_cp.groupby("commoditycode")["rca"].transform("sum")
        ) * all_cp["gdp_pc"]
        all_cp["prody"] = all_cp.groupby("commoditycode")["prody"].transform("sum")
        return all_cp

    def growth_opportunity_metrics(self, all_cp):
        """ """
        logging.info("Creating the product space for all countries & all products")
        # mata C = M'*M
        all_cp_mcp = all_cp.pivot(
            index="exporter", columns="commoditycode", values="mcp"
        )
        all_cp_pci = all_cp.pivot(
            index="exporter", columns="commoditycode", values="pci"
        )

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
            product_x + product_y - abs(product_x - product_y)
        ) / 2 - np.identity(all_cp_mcp.shape[1])

        # mata density3 = proximity' :/ (J(Nps,Nps,1) * proximity')
        # all_cp_proximity = all_cp_proximity.fillna(0)
        all_cp_density = all_cp_proximity.T.div(
            np.dot(
                np.ones((all_cp_mcp.shape[1], all_cp_mcp.shape[1]), dtype=int),
                all_cp_proximity.T.values,
            )
        )
        # mata density3 = M * density3
        all_cp_density = all_cp_mcp @ all_cp_density
        # mata opportunity_value =  ((density3:*(1 :- M)):*pci3)*J(Nps,Nps,1)
        opportunity_value = ((all_cp_density.mul(1 - all_cp_mcp)).mul(all_cp_pci)) @ (
            pd.DataFrame(1, index=all_cp_mcp.columns, columns=all_cp_mcp.columns)
        )

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

        all_cp_metrics = {
            "density": all_cp_density.T,
            "opportunity_value": opportunity_value.T,
            "opportunity_gain": opportunity_gain.T,
        }

        all_cp_metrics_df = pd.concat(
            [df.add_prefix(f"{name}_") for name, df in all_cp_metrics.items()],
            join="inner",
            axis=1,
        ).reset_index()

        all_cp_metrics_df = pd.wide_to_long(
            all_cp_metrics_df,
            stubnames=list(all_cp_metrics.keys()),
            i="commoditycode",
            j="exporter",
            sep="_",
            suffix="[A-Z]+",
        ).reset_index()

        all_cp = all_cp.merge(
            all_cp_metrics_df, on=["exporter", "commoditycode"], how="left"
        )
        all_cp = all_cp.rename({"prody": "prody_allcp", "density": "density_allcp"})
        return all_cp

    def merge_data(self, all_cp, all_countries):
        """ """
        self.df = self.df.rename(
            columns={
                "density": "density_reliable",
                "coi": "coi_reliable",
                "cog": "cog_reliable",
            }
        )
        reliable_cols = [
            "exporter",
            "commoditycode",
            "density_reliable",
            "rca_reliable",
            "eci_reliable",
            "pci_reliable",
            "coi_reliable",
            "cog_reliable",
        ]

        all_cp = all_cp.rename(
            columns={
                "rca": "rca_allcp",
                "pci": "pci_allcp",
                "density": "density_allcp",
                "prody": "prody_allcp",
                "opportunity_value": "opportunity_value_allcp",
                "opportunity_gain": "opportunity_gain_allcp",
                "mcp": "mcp_allcp",
            }
        )

        all_cp = all_cp.drop(columns=["eci"])

        self.df = all_cp.merge(
            self.df[reliable_cols], on=["exporter", "commoditycode"], how="outer"
        )

        all_countries = all_countries.drop(
            columns=["eci", "mcp", "pci_reliable", "pci"]
        )
        all_countries = all_countries.rename(
            columns={
                "rca": "rca_all_countries",
                "prody": "prody_all_countries",
                "expy": "expy_all_countries",
            }
        )

        use_cols = [
            "exporter",
            "commoditycode",
            "rca_all_countries",
            "eci_all_countries",
            "prody_all_countries",
            "expy_all_countries",
        ]

        self.df = self.df.merge(
            all_countries[use_cols], on=["exporter", "commoditycode"], how="outer"
        )

    def standardize_metrics(self):
        """ """
        for col in ["eci_reliable", "eci_all_countries", "expy_all_countries"]:
            self.df["mean_val"] = self.df.groupby("exporter")[col].transform("mean")
            self.df.loc[self.df[col].isna(), col] = self.df["mean_val"]
            self.df = self.df.drop(columns="mean_val")

        # replace pci3 = (pci3 - r(mean))/r(sd) if pci3!=.
        pci = self.df.groupby("commoditycode")["pci_allcp"].agg("first")
        self.df.loc[self.df["pci_allcp"].notna(), "pci_allcp"] = (
            self.df["pci_allcp"] - np.mean(pci)
        ) / np.std(pci)

        # replace opportunity_value = (opportunity_value-r(mean))/r(sd) if opportunity_value!=.
        # standardize opportunity_value
        opp_val = self.df.groupby("exporter")["opportunity_value_allcp"].agg("first")
        self.df.loc[
            self.df["opportunity_value_allcp"].notna(), "opportunity_value_allcp"
        ] = (self.df["opportunity_value_allcp"] - np.mean(opp_val)) / np.std(opp_val)

    def backfill_complexity_by_data_reliability(self):
        """ """
        measures = {
            "rca": ["rca_reliable", "rca_all_countries", "rca_allcp"],
            "eci": ["eci_reliable", "eci_all_countries"],
            "pci": ["pci_reliable", "pci_allcp"],
            "prody": ["prody_all_countries", "prody_allcp"],
            "expy": ["expy_all_countries"],
            "density": [
                "density_reliable",
                "density_allcp",
            ],  # density output directly from ecomplexity
            "oppval": ["coi_reliable", "opportunity_value_allcp"],
            "oppgain": ["cog_reliable", "opportunity_gain_allcp"],
        }

        for measure, replacement_vals in measures.items():
            self.df[measure] = self.df[replacement_vals].bfill(axis=1).iloc[:, 0]
            self.df = self.df.drop(columns=replacement_vals)

    def prepare_output_data(self):
        """
        Align data to meet expected data formats, dropping unecessary columns
        and enforcing data types
        """
        # rename M mcp
        self.df = self.df.rename(columns={"mcp_allcp": "mcp"})

        # cap gen distance = 1 - density
        self.df["distance"] = 1 - self.df["density"]
        self.df = self.df.drop(columns=["density"])

        columns_to_keep = [
            "exporter",
            "commoditycode",
            "export_value",
            "import_value",
            "gdp_pc",
            "rca",
            "mcp",
            "eci",
            "pci",
            "oppval",
            "oppgain",
            "distance",
            "prody",
            "expy",
        ]
        self.df = self.df[columns_to_keep]

        float32_columns = [
            "export_value",
            "rca",
            "eci",
            "pci",
            "oppval",
            "oppgain",
            "distance",
            "prody",
            "expy",
            "gdp_pc",
        ]
        for col in float32_columns:
            self.df[col] = self.df[col].astype("float32")
        self.df["mcp"] = self.df["mcp"].astype("int8")
        self.df["inatlas"] = 1
        self.df["year"] = self.year
        self.df = self.df[
            [
                "year",
                "exporter",
                "commoditycode",
                "inatlas",
                "export_value",
                "import_value",
                "rca",
                "mcp",
                "eci",
                "pci",
                "oppval",
                "oppgain",
                "distance",
                "prody",
                "expy",
                "gdp_pc",
            ]
        ]
