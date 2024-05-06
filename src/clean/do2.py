import pandas as pd
from clean.objects.base import _AtlasCleaning
import os
import numpy as np
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)


class do2(_AtlasCleaning):
    niter = 25  # Iterations A_e
    rnflows = 10  # 30
    flow_limit = 10**4  # Minimum value to assume that there is a flow
    vfile = 1
    poplimit = 0.5 * 10**6  # Only include countries with population above this number
    anorm = 0  # Normalize the score
    alog = 0  # Apply logs
    af = 0  # Combine A_e and A_i in single measure
    seed = 1  # Initial value for the A's

    def __init__(self, year, product_classification, **kwargs):
        super().__init__(**kwargs)

        # Set parameters
        self.df = pd.DataFrame()

        # prep wdi data
        # TODO: convert to IMF data, use wdi initially to compare values
        wdi = pd.read_stata(
            self.wdi_path, columns=["year", "iso", "fp_cpi_totl_zg", "sp_pop_totl"]
        )
        wdi_cpi = wdi[(wdi.year >= 1962) & (wdi.iso == "USA")].drop(
            ["sp_pop_totl"], axis=1
        )
        wdi_cpi = wdi_cpi.rename(columns={"fp_cpi_totl_zg": "cpi"})
        wdi_cpi = wdi_cpi.reset_index(drop=True)

        wdi_pop = wdi.drop(["fp_cpi_totl_zg"], axis=1)
        imf_pop = pd.read_csv(
            os.path.join(self.raw_data_path, "imf_data.csv"),
            usecols=["code", "year", "population"],
        ).rename(columns={"population": "imf_pop"})
        pop = wdi_pop.merge(
            imf_pop, left_on=["iso", "year"], right_on=["code", "year"], how="outer"
        ).drop(columns=["code"])
        # if empty fill in with imf population data

        # totals from dofile1, concats all years
        totals = pd.read_csv(
            os.path.join(self.intermediate_data_path, f"Totals_RAW_trade.csv")
        )

        wdi_cpi = self.inflation_adjustment(wdi_cpi)
        weights = []
        start_year = 2015
        end_year = 2015
        for year in range(start_year, end_year + 1):
            year_totals = totals[totals.year == year]
            year_totals = year_totals.dropna(subset=["exporter", "importer"])
            year_totals = year_totals[
                ~(
                    (year_totals.exporter.isin(["WLD", "ANS"]))
                    | (year_totals.importer.isin(["WLD", "ANS"]))
                )
            ]
            year_totals = year_totals[year_totals.exporter != year_totals.importer]
            year_totals = year_totals[
                year_totals[["importvalue_fob", "exportvalue_fob"]].max(axis=1) >= 10**4
            ]

            # TODO: if calculate cif_ratio by distance and existing values then implement
            # checking for max value from dofile2

            # complete dataframe to have all combinations for year, exporter, importer
            years = year_totals["year"].unique()
            exporters = year_totals["exporter"].unique()
            importers = year_totals["importer"].unique()
            all_combinations = pd.MultiIndex.from_product(
                [years, exporters, importers], names=["year", "exporter", "importer"]
            )
            filled_df = pd.DataFrame(index=all_combinations).reset_index()

            year_totals = filled_df.merge(
                year_totals, on=["year", "exporter", "importer"], how="left"
            )
            pop_year = pop[pop.year == year].drop(columns=["year"])

            # population limit of 1M for atlas cleaning inclusion
            # merge importer population data
            year_totals = (
                year_totals.merge(
                    pop_year, left_on="importer", right_on="iso", how="left"
                )
                .rename(
                    columns={
                        "sp_pop_totl": "wdi_pop_importer",
                        "imf_pop": "imf_pop_importer",
                    }
                )
                .drop(columns=["iso"])
            )
            # merge exporter population data
            year_totals = (
                year_totals.merge(
                    pop_year, left_on="exporter", right_on="iso", how="left"
                )
                .rename(
                    columns={
                        "sp_pop_totl": "wdi_pop_exporter",
                        "imf_pop": "imf_pop_exporter",
                    }
                )
                .drop(columns=["iso"])
            )

            # cutoff any importer/exporter below poplimit threshold
            year_totals = year_totals[
                year_totals[["wdi_pop_exporter", "imf_pop_exporter"]].max(axis=1)
                > self.poplimit
            ]
            year_totals = year_totals[
                year_totals[["wdi_pop_importer", "imf_pop_importer"]].max(axis=1)
                > self.poplimit
            ]
            # after cutoffs implemented, then drop population data
            year_totals = year_totals.drop(
                columns=[
                    "wdi_pop_exporter",
                    "imf_pop_exporter",
                    "wdi_pop_importer",
                    "imf_pop_importer",
                ]
            )

            # matrices
            inflation = pd.read_parquet(
                os.path.join(self.intermediate_data_path, "inflation_index.parquet")
            )
            df = year_totals.merge(
                inflation[["year", "cpi", "cpi_index", "cpi_index_base"]],
                on="year",
                how="inner",
            )
            df = df.assign(
                **{
                    col: df[col] / df.cpi_index_base
                    for col in ["exportvalue_fob", "importvalue_cif", "importvalue_fob"]
                }
            )
            # ensure export and import values are floats
            df = df.drop(columns="cpi_index_base").rename(
                columns={"exportvalue_fob": "v_e", "importvalue_fob": "v_i"}
            )
            # trade below threshold is zeroed
            df.loc[df.v_e < self.flow_limit, "v_e"] = 0.0
            df.loc[df.v_i < self.flow_limit, "v_i"] = 0.0
            df = df.groupby("exporter").filter(lambda x: (x["v_e"] > 0).sum() > 0)
            df = df.groupby("importer").filter(lambda x: (x["v_i"] > 0).sum() > 0)

            # deviation scores
            # trade imbalance normalized by total trade
            df["s_ij"] = (
                (abs(df["v_e"] - df["v_i"])) / (df["v_e"] + df["v_i"])
            ).fillna(0.0)

            df.v_e = df.v_e.fillna(0.0)
            df.v_i = df.v_i.fillna(0.0)

            for trade_flow in ["importer", "exporter"]:
                for t in range(1, 6):
                    # Calculate nflows
                    df["nflows"] = (
                        ((df["v_e"] != 0.0) | (df["v_i"] != 0.0))
                        .groupby(df[trade_flow])
                        .transform("sum")
                    )

                    # Get the list of countries with nflows < rnflows
                    listctry = (
                        df.loc[df["nflows"] < self.rnflows, trade_flow]
                        .unique()
                        .tolist()
                    )

                    # Drop rows where exporter or importer has trade flow below threshold
                    for i in listctry:
                        df = df[~((df["exporter"] == i) | (df["importer"] == i))]

            # check and ensure match for each importer and exporter
            missing_trade_flow = np.setdiff1d(
                df.importer.unique().tolist(), df.exporter.unique().tolist()
            )
            if missing_trade_flow:
                df = df[
                    ~(
                        df["exporter"].isin(missing_trade_flow)
                        | df["importer"].isin(missing_trade_flow)
                    )
                ]

            df = df.assign(
                temp_flow_avg=df[["v_e", "v_i"]].replace(0, np.nan).mean(axis=1)
            )
            df["temp_exporter_sums"] = df.groupby("exporter")[
                "temp_flow_avg"
            ].transform("sum")
            df["temp_importer_sums"] = df.groupby("importer")[
                "temp_flow_avg"
            ].transform("sum")

            # percentage of countries total exports
            df["perc_e"] = (df["temp_flow_avg"] / df["temp_exporter_sums"]).clip(
                lower=0
            )
            # percentage of countries total imports
            df["perc_i"] = (df["temp_flow_avg"] / df["temp_importer_sums"]).clip(
                lower=0
            )
            df = df.filter(regex="^(?!temp).*")

            max_exporter_flows = df.groupby(["exporter"])["exporter"].count().max()
            max_importer_flows = df.groupby(["importer"])["importer"].count().max()

            # generate metrics
            for trade_flow in ["importer", "exporter"]:
                df[f"nflows_{trade_flow}"] = (
                    ((df["v_e"] != 0) | (df["v_i"] != 0))
                    .groupby(df[trade_flow])
                    .transform("sum")
                )
                # average normalized trade imbalance by trade flow
                df[f"s_ij_{trade_flow}"] = df.groupby(trade_flow)["s_ij"].transform(
                    "mean"
                )
                # sum average by
                df[f"avs_ij_{trade_flow}"] = df.groupby(trade_flow)[
                    f"s_ij_{trade_flow}"
                ].transform("sum")

                # avg normalized trade imbalance for each flow
                df[f"avs_ij_{trade_flow}"] = (
                    df[f"avs_ij_{trade_flow}"] / df[f"nflows_{trade_flow}"]
                )

            df["is_ij"] = df["s_ij"].copy()
            df = df.rename(
                columns={
                    "s_ij": "es_ij",
                    "nflows_exporter": "en_ij",
                    "nflows_importer": "in_ij",
                }
            )

            # TODO: ask about Muhammed's code section (MAY)

            # iterations refine Attractiveness of importer and exporter
            # based on normalized trade imbalance and number of trade partners
            df["A_e"] = 1
            df["A_i"] = 1
            for i in range(0, 25):
                df["prA_e"] = df["es_ij"] * df["A_e"] / df["en_ij"]
                df["prA_i"] = df["is_ij"] * df["A_i"] / df["in_ij"]
                df["A_e"] = df["prA_e"]
                df["A_i"] = df["prA_i"]
            df = df.drop(columns=["prA_e", "prA_i"])

            df["es_ij"] = df["es_ij"].sum() / max_exporter_flows
            df["is_ij"] = df["is_ij"].sum() / max_importer_flows

            df["tag"] = df.groupby("exporter").cumcount() == 0
            df = df[df["tag"]]
            df = df.rename(columns={"exporter": "iso"})
            df = df[
                ["year", "iso", "A_e", "A_i", "en_ij", "in_ij", "es_ij", "is_ij"]
            ]  # , 'sigmas']]

            df = df.rename(
                columns={
                    "en_ij": "nflows_e",
                    "in_ij": "nflows_i",
                    "es_ij": "av_es",
                    "is_ij": "av_is",
                }
            )

            # fix some df has single exporter for year 2015

            if self.alog == 1:
                df["A_e"] = np.ln(df["A_e"])
                df["A_i"] = np.ln(df["A_i"])
            if self.anorm == 1:
                df["A_e"] = df["A_e"] - df["A_e"].mean() / df["A_e"].std()

            if self.af == 0:
                df["A_f"] = df[["A_e", "A_i"]].mean(axis=1)
            elif self.af == 1:
                df["A_f"] = PCA().fit_transform(df[["A_e", "A_i"]])

            if self.anorm == 1:
                df["A_f"] = df["A_f"] - df["A_f"].mean() / df["A_f"].std()

            df.sort_values(by="A_f", ascending=False)
            # noi list year iso A_e A_i A_f  if _n<=10
            df.to_stata("data/intermediate/attractiveness.dta")

            # TODO: need to add in sigmas
            # merge cpi index with exporters by year
            # TODO: requires many to one, there are many A_e,A_i values for year/exporter
            merged = year_totals.merge(
                df[["year", "iso", "A_e", "A_i"]],  # "sigmas"]],
                left_on=["year", "exporter"],
                right_on=["year", "iso"],
                how="left",
            ).drop(columns=["iso"])

            merged.rename(
                columns={
                    "A_e": "exporter_A_e",
                    "A_i": "exporter_A_i",
                    # "sigmas": "exporter_sigmas",
                },
                inplace=True,
            )

            merged = merged.merge(
                df[["year", "iso", "A_e", "A_i"]],  # "sigmas"]],
                left_on=["year", "importer"],
                right_on=["year", "iso"],
                how="left",
            ).drop(columns=["iso"])

            merged.rename(
                columns={
                    "A_e": "importer_A_e",
                    "A_i": "importer_A_i",
                    # "sigmas": "importer_sigmas",
                },
                inplace=True,
            )

            merged[merged.importer != merged.exporter]

            merged["tag_e"] = merged.duplicated("exporter", keep="first").astype(int)
            merged["tag_i"] = merged.duplicated("importer", keep="first").astype(int)

            # Data Cleaning
            merged.loc[merged["importvalue_fob"] < 1000, "importvalue_fob"] = np.nan
            merged.loc[merged["exportvalue_fob"] < 1000, "exportvalue_fob"] = np.nan

            # Calculating Percentiles
            percentiles_e = merged[merged["tag_e"] == 1]["exporter_A_e"].describe(
                percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
            )
            percentiles_i = merged[merged["tag_i"] == 1]["importer_A_i"].describe(
                percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
            )

            # Weight Calculation
            merged["w_e"] = np.exp(merged["exporter_A_e"]) / (
                np.exp(merged["exporter_A_e"]) + np.exp(merged["importer_A_i"])
            )

            # Output the percentiles
            logging.info(
                f":: exporter A_e :: 10, 25, 50, 75, 95 = {percentiles_e['10%']} & {percentiles_e['25%']} & {percentiles_e['50%']} & {percentiles_e['75%']} & {percentiles_e['90%']}"
            )
            logging.info(
                f":: importer A_i :: 10, 25, 50, 75, 95 = {percentiles_i['10%']} & {percentiles_i['25%']} & {percentiles_i['50%']} & {percentiles_i['75%']} & {percentiles_i['90%']}"
            )

            # include set of countries
            merged = merged.assign(
                w_e_0=np.where(
                    (merged.exporter_A_e.notna())
                    & (merged.exporter_A_e > percentiles_e["10%"]),
                    1,
                    0,
                ),
                w_i_0=np.where(
                    (merged.exporter_A_i.notna())
                    & (merged.exporter_A_i > percentiles_i["10%"]),
                    1,
                    0,
                ),
            )

            merged["discrep"] = np.exp(
                np.abs(np.log(merged["exportvalue_fob"] / merged["importvalue_fob"]))
            )
            merged["discrep"] = merged["discrep"].replace(np.nan, 99)

            df = self.calculate_estimated_value(merged, percentiles_e, percentiles_i)
            df = df.drop(columns=["discrep"])

            # Calculate total_value and share_exporter
            df["total_value"] = df.groupby(["year", "exporter"])["est_value"].transform(
                "sum"
            )
            df["share_exporter"] = df["est_value"] / df["total_value"]
            # drop flows
            df = df.drop(
                df.loc[
                    (df["share_exporter"] > 0.75)
                    & (df["share_exporter"].notna())
                    & (df["total_value"] > 10 ^ 7)
                    & (df["importer_A_i"] < percentiles_i["50%"])
                    & (df["importer_A_e"].notna())
                    & ((df["exporter"] != "BRN") & (df["importer"] != "MYS"))
                    & ((df["exporter"] != "DJI") & (df["importer"] != "SAU")),
                    "est_value",
                ].index
            )
            # df = df[~df.est_value.isna()]

            # Calculate mintrade and update estvalue
            df["mintrade"] = df[["exportvalue_fob", "importvalue_fob"]].min(axis=1)
            df.loc[(df["mintrade"].notna()) & (df["est_value"].isna()), "est_value"] = (
                df["mintrade"]
            )

            # Rename columns
            df = df.rename(
                columns={
                    "exportvalue_fob": "value_exporter",
                    "importvalue_fob": "value_importer",
                    "est_value": "value_final",
                }
            )

            # Select and reorder columns
            columns_to_keep = [
                "year",
                "exporter",
                "importer",
                "value_exporter",
                "value_importer",
                "value_final",
                # "cif_ratio",
                "w_e_0",
                "w_i_0",
                "importer_A_e",
                "importer_A_i",
            ]
            df = df[columns_to_keep]

            # Save the DataFrame to a file
            weights.append(df)
            output_path = os.path.join(
                self.intermediate_data_path, f"weights_{year}.dta"
            )
            df.to_parquet(output_path)
        weights_years_total = pd.concat(weights)
        output_path = os.path.join(
            self.processed_data_path, f"weights_{start_year}-{end_year}.dta"
        )
        weights_years_total.to_parquet(output_path)

    def calculate_estimated_value(self, df, perc_e, perc_i):
        """
        Series of conditions to determine estimated trade value
        """
        logging.info("Estimating total trade flows between countries")
        df["est_value"] = np.where(
            (df["exporter_A_e"].notna())
            & (df["importer_A_i"].notna())
            & (df["exportvalue_fob"].notna()),
            df["exportvalue_fob"] * df["w_e"] + df["importvalue_fob"] * (1 - df["w_e"]),
            np.nan,
        )

        conditions = [
            (df["exporter_A_e"] < perc_e["50%"])
            & (df["importer_A_i"] >= perc_i["90%"]),
            (df["exporter_A_e"] >= perc_e["90%"])
            & (df["importer_A_i"] < perc_i["50%"]),
            (df["exporter_A_e"] < perc_e["25%"])
            & (df["importer_A_i"] >= perc_i["75%"]),
            (df["exporter_A_e"] >= perc_e["75%"])
            & (df["importer_A_i"] < perc_i["25%"]),
            (df["w_e_0"] == 1) & (df["w_i_0"] == 1),
            (df["w_i_0"] == 1),
            (df["w_e_0"] == 1),
            df["est_value"].isna(),
        ]

        choices = [
            df["importvalue_fob"],
            df["exportvalue_fob"],
            df["importvalue_fob"],
            df["exportvalue_fob"],
            df[["importvalue_fob", "exportvalue_fob"]].max(axis=1),
            df["importvalue_fob"],
            df["exportvalue_fob"],
            df["importvalue_fob"],
        ]

        for condition, choice in zip(conditions, choices):
            df.loc[df["est_value"].isna() & condition, "est_value"] = choice
        return df

    def inflation_adjustment(self, wdi_cpi):
        """ """
        for i, row in wdi_cpi.iterrows():
            if i == 0:
                wdi_cpi.at[i, "cpi_index"] = 100.0
            else:
                wdi_cpi.at[i, "cpi_index"] = wdi_cpi.iloc[i - 1]["cpi_index"] * (
                    1 + row.cpi / 100
                )

        # sets base year at 2010
        base_year_cpi_index = wdi_cpi.loc[wdi_cpi.year == 2010, "cpi_index"].iloc[0]
        wdi_cpi["cpi_index_base"] = wdi_cpi["cpi_index"] / base_year_cpi_index

        # temp_accuracy
        wdi_cpi.to_parquet(
            os.path.join(self.intermediate_data_path, "inflation_index.parquet")
        )
        return wdi_cpi
