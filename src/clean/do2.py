import pandas as pd
from clean.objects.base import _AtlasCleaning


class do2(_AtlasCleaning):
    def __init__(self, year, product_classification, **kwargs):
        super().__init__(**kwargs)
            
        # Set parameters
        niter = 25  # Iterations A_e
        rnflows = 10  # 30
        limit = 10**4  # Minimum value to assume that there is a flow
        vfile = 1
        poplimit = 0.5 * 10**6  # Only include countries with population above this number
        anorm = 0  # Normalize the score
        alog = 0  # Apply logs
        af = 0  # Combine A_e and A_i in single measure
        seed = 1  # Initial value for the A's
        
        self.wdi_data = pd.read_stata(self.wdi_path)
        self.wdi_data = self.wdi_data[(self.wdi_data.year >= 1962) & self.wdi_data.iso == 'USA')]
        self.wdi_data = self.wdi_data.rename(columns={"fp_cpi_totl_zg": "dcpi"})
        wdi = wdi.reset_index(drop=True)
        # self.wdi_data['cpi_index'] = 0.0

        self.inflation_adjustment()


    def inflation_adjustment(self):
        """
        """
        for i, row in self.wdi_data.iterrows():
            if i == 0:
                wdi.at[i, 'cpi_index'] = 100.0
            wdi.at[i, 'cpi_index'] = wdi.iloc[i - 1]['cpi_index']*(1+row.cpi/100)
        
        # sets base year at 2010
        base_year_cpi_index = wdi.loc[wdi.year == 2010].index[0]['cpi_index']
        self.wdi['cpi_index_base'] = wdi['cpi_index']/base_year_cpi_index
        
        # if cpi_index is empty than set cpi_index to be previous year's cpi_index
        pd.to_parquet(os.path.join(self.intermediate_data_path,'data/intermediate/inflation_index.parquet'))

            
#         # Read the trade data and determine the start and end years
#         trade_data = pd.read_stata("Totals_trade.dta")
#         start_year = trade_data["year"].min()
#         end_year = trade_data["year"].max()

#         # Perform the main loop over each year
#         for y in range(start_year, end_year + 1):
#             # Load the trade data for the current year
#             data = pd.read_stata(Path(path) / "Totals_trade.dta")
#             data = data[data["year"] == y]
#             data = data[~data["exporter"].isin(["WLD", "nan"])]
#             data = data[~data["importer"].isin(["WLD", "nan"])]
#             data = data[~((data["exporter"] == "ANS") & (data["importer"] == "ANS"))]
#             data = data[data["exporter"] != data["importer"]]

#             # Drop rows with max values under 10K
#             temp = data[["importvalue_fob", "exportvalue_fob"]].max(axis=1)
#             data = data[temp >= 10**4]

#             # Calculate the CIF ratio
#             cif_ratio = data["importvalue_cif"] / data["importvalue_fob"] - 1
#             cif_ratio = cif_ratio.groupby([data["exporter"], data["importer"]]).mean()
#             cif_ratio = cif_ratio.clip(upper=0.20)
#             cif_ratio = cif_ratio.reset_index()
#             cif_ratio = cif_ratio.rename(columns={0: "cif_ratio"})
#             cif_ratio["year"] = y
#             cif_ratio[["year", "exporter", "importer", "cif_ratio", "importvalue_fob", "exportvalue_fob"]].to_stata("temp_accuracy.dta", index=False)

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
