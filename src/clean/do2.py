import pandas as pd
from clean.objects.base import _AtlasCleaning
import os
import numpy as np
from sklearn.decomposition import PCA


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

        for year in range(2015, 2015 + 1):
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

            import pdb

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
            )

            merged.rename(
                columns={
                    "A_e": "exporter_A_e",
                    "A_i": "exporter_A_i",
                    # "sigmas": "exporter_sigmas",
                },
                inplace=True,
            )

            import pdb

            pdb.set_trace()

            merged = merged.merge(
                df[["year", "importer", "A_e", "A_i"]],  # "sigmas"]],
                left_on=["year", "importer"],
                right_on=["year", "iso"],
                how="left",
            )

            import pdb

            pdb.set_trace()

            merged.rename(
                columns={
                    "A_e": "importer_A_e",
                    "A_i": "importer_A_i",
                    # "sigmas": "importer_sigmas",
                },
                inplace=True,
            )

            # line 453

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


#             # Drop rows with max values under 10K
#             temp = data[["importvalue_fob", "exportvalue_fob"]].max(axis=1)
#             data = data[temp >= 10**4]


#             # Fill in missing combinations of year, exporter, and importer
#             data = data.set_index(["year", "exporter", "importer"]).unstack(fill_value=0).stack().reset_index()

#             # Merge population data for exporters and importers
#             for i in ["exporter", "importer"]:
#                 pop_data = pd.read_stata(f"temp_pop{i}.dta")
#                 data = data.merge(pop_data, on=["year", i], how="left")

#             # Handle countries without population data in WDI
#             for n in ["YUG", "DDR", "SUN", "CSK", "ANS", "TWN"]:
#                 data.loc[data[i] == n, f"pop_{i}"] = 10**7

#             data = data[~data["exporter"].isin(["MAC", "nan", "NAN", "ANS"])]
#             data = data[~data["importer"].isin(["MAC", "nan", "NAN", "ANS"])]

#             # Keep only countries with population above the limit
#             data = data[data["pop_exporter"] > poplimit]
#             data = data[data["pop_importer"] > poplimit]
#             data = data.drop(columns=["pop_exporter", "pop_importer"])

#             # Fill in missing combinations of year, exporter, and importer again
#             data = data.set_index(["year", "exporter", "importer"]).unstack(fill_value=0).stack().reset_index()

#             # Merge the inflation index data
#             index_data = pd.read_stata("inflation_adjusted.parquet")
#             data = data.merge(index_data, on="year", how="left")

#             # Adjust trade values using the inflation index
#             for j in ["exportvalue_fob", "importvalue_fob", "importvalue_cif"]:
#                 data[j] = data[j] / data["index"]
#                 data[j] = data[j].fillna(0)

#             data = data.drop(columns=["index"])
#             data = data.rename(columns={"exportvalue_fob": "v_e", "importvalue_fob": "v_i"})

#             # Replace trade values below the limit with 0
#             for j in ["e", "i"]:
#                 data[f"v_{j}"] = data[f"v_{j}"].where(data[f"v_{j}"] >= limit, 0)

#             # Drop exporters and importers with no positive trade flows
#             data = data.groupby("exporter").filter(lambda x: (x["v_e"] > 0).any())
#             data = data.groupby("importer").filter(lambda x: (x["v_i"] > 0).any())
#             data = data.sort_values(["exporter", "importer"])
#             data = data[["exporter", "importer", "v_e", "v_i"]]

#             # Calculate deviation scores
#             data["s_ij"] = (data["v_e"] - data["v_i"]).abs() / (data["v_e"] + data["v_i"])
#             data["s_ij"] = data["s_ij"].fillna(0)

#             # Eliminate exporters and importers below a certain threshold of flows
#             for direction in ["exporter", "importer"]:
#                 for t in range(1, 6):
#                     nflows = data.groupby(direction)[["v_e", "v_i"]].transform(lambda x: (x != 0).any(axis=1)).astype(int).sum()
#                     listctry = nflows[nflows < rnflows].index.tolist()
#                     data = data[~data[direction].isin(listctry)]

#                     # Ensure that importers and exporters are the same countries
#                     data["temp"] = 0
#                     listctry = data[direction].unique().tolist()
#                     if direction == "exporter":
#                         data.loc[data["importer"].isin(listctry), "temp"] = 1
#                     if direction == "importer":
#                         data.loc[data["exporter"].isin(listctry), "temp"] = 1
#                     data = data[data["temp"] == 1]
#                     data = data.drop(columns=["temp"])

#             # Calculate probability vectors p_e and p_i
#             data["temp1"] = data["v_e"].where(data["v_e"] != 0)
#             data["temp2"] = data["v_i"].where(data["v_i"] != 0)
#             data["temp3"] = data[["temp1", "temp2"]].mean(axis=1)
#             data["temp4"] = data.groupby("exporter")["temp3"].transform("sum")
#             data["temp5"] = data.groupby("importer")["temp3"].transform("sum")
#             data["p_e"] = (data["temp3"] / data["temp4"]).clip(lower=0)
#             data["p_i"] = (data["temp3"] / data["temp5"]).clip(lower=0)
#             data = data.drop(columns=["temp1", "temp2", "temp3", "temp4", "temp5"])

#             # Calculate additional statistics for exporters and importers
#             for direction in ["exporter", "importer"]:
#                 data[f"nflows_{direction}"] = data.groupby(direction)[["v_e", "v_i"]].transform(lambda x: (x != 0).any(axis=1)).astype(int).sum()
#                 data[f"s_ij_{direction}"] = data.groupby(direction)["s_ij"].transform("mean")
#                 data[f"avs_ij_{direction}"] = data[f"s_ij_{direction}"] / data[f"nflows_{direction}"]

#             # Assign numeric indices to exporters and importers
#             data["exp"] = pd.factorize(data["exporter"])[0] + 1
#             data["imp"] = pd.factorize(data["importer"])[0] + 1

#             # Prepare data for Mata operations
#             N = data["exp"].max()
#             es_ij = pd.pivot_table(data, values="s_ij", index="exp", columns="imp", fill_value=0).values
#             en_ij = data.groupby("exp")["nflows_exporter"].first().values.reshape(-1, 1)
#             p_e = data.groupby("exp")["p_e"].first().values.reshape(-1, 1)
#             is_ij = pd.pivot_table(data, values="s_ij", index="imp", columns="exp", fill_value=0).values
#             in_ij = data.groupby("imp")["nflows_importer"].first().values.reshape(-1, 1)
#             p_i = data.groupby("imp")["p_i"].first().values.reshape(-1, 1)

#             # Initialize A_e and A_i matrices
#             A_e = np.full((N, 1), seed)
#             A_i = A_e.copy()

#             # Perform the iterative estimation
#             for i in range(niter):
#                 prA_e = 1 / ((es_ij @ A_i) / en_ij)
#                 prA_i = 1 / ((is_ij @ A_e) / in_ij)
#                 A_e = prA_e
#                 A_i = prA_i

#             # Calculate additional statistics
#             es_ij = es_ij.sum(axis=1) / N
#             is_ij = is_ij.sum(axis=1) / N

#             # Prepare the output data
#             output_data = data[["exporter", "year"]].drop_duplicates()
#             output_data["A_e"] = A_e.ravel()
#             output_data["A_i"] = A_i.ravel()
#             output_data["nflows_e"] = en_ij.ravel()
#             output_data["nflows_i"] = in_ij.ravel()
#             output_data["av_es"] = es_ij
#             output_data["av_is"] = is_ij

#             # Apply log transformation and normalization if specified
#             for j in ["A_e", "A_i"]:
#                 if alog == 1:
#                     output_data[j] = np.log(output_data[j])
#                 if anorm == 1:
#                     output_data[j] = (output_data[j] - output_data[j].mean()) / output_data[j].std()

#             # Combine A_e and A_i into a single measure A_f if specified
#             if af == 0:
#                 output_data["A_f"] = output_data[["A_e", "A_i"]].mean(axis=1)
#             if af == 1:
#                 from sklearn.decomposition import PCA
#                 pca = PCA(n_components=1)
#                 output_data["A_f"] = pca.fit_transform(output_data[["A_e", "A_i"]])

#             if anorm == 1:
#                 output_data["A_f"] = (output_data["A_f"] - output_data["A_f"].mean()) / output_data["A_f"].std()

#             # Save the temporary results
#             output_data.sort_values("A_f", ascending=False).head(10)
#             output_data.to_stata("temp_r.dta", index=False)

#             # Merge the temporary results with the accuracy data
#             accuracy_data = pd.read_stata("temp_accuracy.dta")
#             accuracy_data = accuracy_data.rename(columns={"exporter": "iso"})
#             accuracy_data = accuracy_data.merge(output_data, on=["year", "iso"], how="left", suffixes=("", "_exporter"))
#             accuracy_data = accuracy_data.rename(columns={"iso": "exporter", "A_e": "exporter_A_e", "A_i": "exporter_A_i"})
#             accuracy_data = accuracy_data.rename(columns={"importer": "iso"})
#             accuracy_data = accuracy_data.merge(output_data, on=["year", "iso"], how="left", suffixes=("", "_importer"))
#             accuracy_data = accuracy_data.rename(columns={"iso": "importer", "A_e": "importer_A_e", "A_i": "importer_A_i"})
#             accuracy_data = accuracy_data.sort_values(["year", "exporter", "importer"])

#             # Drop rows where exporter and importer are the same
#             accuracy_data = accuracy_data[accuracy_data["exporter"] != accuracy_data["importer"]]

#             # Calculate percentiles for exporter and importer scores
#             percentiles = accuracy_data.groupby("exporter")["exporter_A_e"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
#             exp_percentiles = percentiles.loc[:, ["10%", "25%", "50%", "75%", "90%"]].to_dict(orient="index")
#             percentiles = accuracy_data.groupby("importer")["importer_A_i"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
#             imp_percentiles = percentiles.loc[:, ["10%", "25%", "50%", "75%", "90%"]].to_dict(orient="index")

#             # Calculate weights based on exporter and importer scores
#             accuracy_data["w_e"] = np.exp(accuracy_data["exporter_A_e"]) / (np.exp(accuracy_data["exporter_A_e"]) + np.exp(accuracy_data["importer_A_i"]))

#             # Determine countries to include based on exporter and importer scores
#             accuracy_data["w_e_0"] = (accuracy_data["exporter_A_e"].notnull()) & (accuracy_data["exporter_A_e"] > accuracy_data["exporter"].map(exp_percentiles).apply(lambda x: x["10%"]))
#             accuracy_data["w_i_0"] = (accuracy_data["importer_A_i"].notnull()) & (accuracy_data["importer_A_i"] > accuracy_data["importer"].map(imp_percentiles).apply(lambda x: x["10%"]))

#             # Estimate total trade flows between countries
#             accuracy_data["discrep"] = np.exp(np.abs(np.log(accuracy_data["exportvalue_fob"] / accuracy_data["importvalue_fob"])))
#             accuracy_data["discrep"] = accuracy
