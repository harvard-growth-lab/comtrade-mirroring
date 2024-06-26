import pandas as pd
from clean.objects.base import _AtlasCleaning
import os
import numpy as np
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)


# generates a country country year table
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
    CPI_BASE_YEAR = 2010

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

        # country country year from dofile1, all years
        # all_ccy = pd.read_csv(
        all_ccy = pd.read_stata(
            # os.path.join(self.intermediate_data_path, f"Totals_RAW_trade.csv")
            os.path.join(
                self.intermediate_data_path, f"totals_raw_trade_stata_output.dta"
            )
        )

        wdi_cpi = self.inflation_adjustment(wdi_cpi)
        weights = []
        start_year = 2015
        end_year = 2015
        for year in range(start_year, end_year + 1):
            ccy = all_ccy[all_ccy.year == year]
            ccy = ccy.dropna(subset=["exporter", "importer"])
            ccy = ccy[
                ~(
                    (ccy.exporter.isin(["WLD", "ANS"]))
                    | (ccy.importer.isin(["WLD", "ANS"]))
                )
            ]
            ccy = ccy[ccy.exporter != ccy.importer]
            ccy = ccy[
                ccy[["importvalue_fob", "exportvalue_fob"]].max(axis=1) >= 10**4
            ]

            # TODO: if calculate cif_ratio by distance and existing values then implement
            # checking for max value from dofile2
            ccy["cif_ratio"] = (ccy["importvalue_cif"] / ccy["importvalue_fob"]) - 1
            ccy["cif_ratio"] = ccy["cif_ratio"].apply(
                lambda val: min(val, 0.2) if pd.notnull(val) else val
            )
            # save as temp_accuracy.dta
            ccy.to_parquet(f"data/intermediate/ccy_{year}.parquet")

            # complete dataframe to have all combinations for year, exporter, importer
            every_ccy_index = pd.MultiIndex.from_product(
                [
                    ccy["year"].unique(),
                    ccy["exporter"].unique(),
                    ccy["importer"].unique(),
                ],
                names=["year", "exporter", "importer"],
            )
            every_ccy = pd.DataFrame(index=every_ccy_index).reset_index()

            ccy = every_ccy.merge(ccy, on=["year", "exporter", "importer"], how="left")

            # merge importer population data
            ccy = (
                ccy.merge(
                    pop[pop.year == year].drop(columns=["year"]),
                    left_on="importer",
                    right_on="iso",
                    how="left",
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
            ccy = (
                ccy.merge(
                    pop[pop.year == year].drop(columns=["year"]),
                    left_on="exporter",
                    right_on="iso",
                    how="left",
                )
                .rename(
                    columns={
                        "sp_pop_totl": "wdi_pop_exporter",
                        "imf_pop": "imf_pop_exporter",
                    }
                )
                .drop(columns=["iso"])
            )
            # cutoff any importer/exporter below poplimit
            ccy = ccy[ccy.wdi_pop_exporter > self.poplimit]
            ccy = ccy[ccy.wdi_pop_importer > self.poplimit]

            # after cutoffs implemented, then drop population data
            ccy = ccy.drop(
                columns=[
                    "wdi_pop_exporter",
                    "imf_pop_exporter",
                    "wdi_pop_importer",
                    "imf_pop_importer",
                ]
            )

            # fillin year exporter importer
            # TODO: generate a fill in / rectangularize method for table object

            # cpi index base is set from 2010 for the US
            inflation = pd.read_parquet(
                os.path.join(self.intermediate_data_path, "inflation_index.parquet")
            )

            df = ccy.merge(
                inflation[["year", "cpi_index_base"]],
                on="year",
                how="inner",
            )

            # converts exports, import values to constant dollar values
            df = df.assign(
                **{
                    col: df[col] / df.cpi_index_base
                    for col in ["exportvalue_fob", "importvalue_fob"]
                }
            )
            df = df.fillna(0.0)
            df = df.drop(
                columns=["cpi_index_base", "importvalue_cif", "cif_ratio"]
            ).rename(
                # in stata v_e and v_i
                columns={
                    "exportvalue_fob": "exports_const_usd",
                    "importvalue_fob": "imports_const_usd",
                }
            )

            # trade below threshold is zeroed
            df.loc[df.exports_const_usd < self.flow_limit, "exports_const_usd"] = 0.0
            df.loc[df.imports_const_usd < self.flow_limit, "imports_const_usd"] = 0.0

            df = df.groupby("exporter").filter(
                lambda row: (row["exports_const_usd"] > 0).sum() > 0
            )
            df = df.groupby("importer").filter(
                lambda row: (row["imports_const_usd"] > 0).sum() > 0
            )

            # difference in trade reporting
            # in stata s_ij
            df["trade_discrepancy"] = (
                (abs(df["exports_const_usd"] - df["imports_const_usd"]))
                / (df["exports_const_usd"] + df["imports_const_usd"])
            ).fillna(0.0)

            # trade flow count method
            for trade_flow in ["importer", "exporter"]:
                for t in range(1, 6):
                    df["nflows"] = (
                        # if neither exports or imports are zero than count as a flow
                        (
                            (df["exports_const_usd"] != 0.0)
                            | (df["imports_const_usd"] != 0.0)
                        )
                        .groupby(df[trade_flow])
                        .transform("sum")
                    )

                    small_nflows = set(
                        df.loc[df["nflows"] < self.rnflows, trade_flow].tolist()
                    )
                    # Drop exporter or importer that has trade flows below threshold
                    if small_nflows:
                        for iso in small_nflows:
                            df = df[
                                ~((df["exporter"] == iso) | (df["importer"] == iso))
                            ]

            # confirms all countries included have data as importer and exporter if not, then drop the country
            importer_only, exporter_only = set(df["importer"].unique()) - set(
                df["exporter"].unique()
            ), set(df["exporter"].unique()) - set(df["importer"].unique())

            if importer_only or exporter_only:
                df = df[
                    ~(
                        df["exporter"].isin(exporter_only)
                        | df["importer"].isin(importer_only)
                    )
                ]
            df = df.drop(columns=["nflows"])
            
            # leave for testing
            assert (
                df["exporter"].nunique()  == df["importer"].nunique() 
            ), f"Number of exporters does not equal number of importers"

            # number of unique countries
            ncountries = df["exporter"].nunique() 

            # calculate the mean of each row, exclude import or export value when equal to zero
            df["temp_flow_avg"] = pd.DataFrame(
                {
                    "exports": np.where(
                        df["exports_const_usd"] != 0, df["exports_const_usd"], np.nan
                    ),
                    "imports": np.where(
                        df["imports_const_usd"] != 0, df["imports_const_usd"], np.nan
                    ),
                }
            ).mean(axis=1)

            # percentage of countries total exports
            df["perc_e"] = np.maximum(
                df["temp_flow_avg"]
                / df.groupby("exporter")["temp_flow_avg"].transform("sum"),
                0.0,
            )
            # percentage of countries total imports
            df["perc_i"] = np.maximum(
                df["temp_flow_avg"]
                / df.groupby("importer")["temp_flow_avg"].transform("sum"),
                0.0,
            )
            df = df.drop(columns=["temp_flow_avg"])

            for trade_flow in ["importer", "exporter"]:
                df[f"nflows_{trade_flow}"] = (
                    (
                        (df["exports_const_usd"] != 0.0)
                        | (df["imports_const_usd"] != 0.0)
                    )
                    .groupby(df[trade_flow])
                    .transform("sum")
                )
                # average normalized trade imbalance by trade flow
                df[f"trade_discrepancy_{trade_flow}_avg"] = df.groupby(trade_flow)[
                    "trade_discrepancy"
                ].transform("mean")

                # divide the total trade discrepancy by importer and exporter by the number of respective trade flows
                df[f"trade_discrepancy_{trade_flow}_total"] = (
                    df.groupby(trade_flow)[
                        f"trade_discrepancy_{trade_flow}_avg"
                    ].transform("sum")
                    / df[f"nflows_{trade_flow}"]
                )


            # TODO: ask about Muhammed's code section (MAY)
            
            # keep iso to keep country id
            # iso = df['exporter'].drop_duplicates().values #.reshape(-1,1)
            # iso_to_index = {code: index for index, code in enumerate(iso)}

            exporters = df['exporter'].unique()
            exporter_to_idx = {exp: idx for idx, exp in enumerate(exporters)}
            
            # prepare matrices to maintain indices
            # stata name: es_ij: exporters, is_ij: importers
            trade_discrepancy = df.pivot(index='exporter', columns='importer', values='trade_discrepancy').fillna(0)
            trade_discrepancy = trade_discrepancy.reindex(index=exporters, columns=exporters, fill_value=0)
            
            # Convert to numpy arrays
            trdiscrep_exp = trade_discrepancy.values
            trdiscrep_imp = trdiscrep_exp.T
            
            nflows_exp = df.groupby('exporter')['nflows_exporter'].first().values.reshape(-1,1)
            nflows_imp = df.groupby('importer')['nflows_importer'].first().reindex(exporters).values.reshape(-1,1)

            # based on normalized trade imbalance and number of trade partners
            # initialize accuracy metric to one
            # should end up being the same?
            accuracy_exp = np.ones((ncountries, 1))
            accuracy_imp = np.ones((ncountries, 1))

            for _ in range(0, 25):
                # @ is element-wise multiplication
                prob_accuracy_exp = 1 / np.divide((trdiscrep_exp @ accuracy_imp), nflows_exp)
                prob_accuracy_imp = 1 / np.divide((trdiscrep_imp @ accuracy_exp), nflows_imp)
                accuracy_imp = prob_accuracy_imp
                accuracy_exp = prob_accuracy_exp
                
            trdiscrep_exp = (np.sum(trdiscrep_exp, axis=1) / ncountries).reshape(-1,1)
            trdiscrep_imp = (np.sum(trdiscrep_imp, axis=1) / ncountries).reshape(-1,1)

            # fix some df has single exporter for year 2015
            if self.alog == 1:
                accuracy_exp = np.ln(accuracy_exp)
                accuracy_imp = np.ln(accuracy_imp)
            if self.anorm == 1:
                accuracy_exp = (accuracy_exp - accuracy_exp.mean()) / accuracy_exp.std()
                accuracy_imp = (accuracy_imp - accuracy_imp.mean()) / accuracy_imp.std()

            if self.af == 0:
                accuracy_final = np.mean([accuracy_exp, accuracy_imp], axis=0)
                # df["A_f"] = df[["A_e", "A_i"]].mean(axis=1)
                
            elif self.af == 1:
                accuracy_final = PCA().fit_transform(accuracy_exp, accuracy_imp)

            if self.anorm == 1:
                accuracy_final = (accuracy_final - accuracy_final.mean()) / accuracy_final.std()
                        
            # combine np arrays into pandas 
            year_array = np.full(ncountries, year).reshape(-1,1)
            
            cy_accuracy = pd.DataFrame(np.hstack([year_array, exporters.reshape(-1,1), nflows_exp, nflows_imp, trdiscrep_exp, trdiscrep_imp, accuracy_exp, accuracy_imp, accuracy_final]),
                                       columns=['year', 'iso', 'nflows_exp', 'nflows_imp', 'trdiscrep_exp', 'trdiscrep_imp', 'acc_exp', 'acc_imp', 'acc_final'])

            cy_accuracy.to_parquet("data/intermediate/accuracy.parquet")
            
            # TODO: need to add in sigmas
            # merge cpi index with exporters by year
            # TODO: requires many to one, there are many A_e,A_i values for year/exporter
            ccy_acc = ccy.merge(
                cy_accuracy[["year", "iso", "acc_exp", "acc_imp"]].rename(columns={
                    "acc_exp": "acc_exp_for_exporter",
                    "acc_imp": "acc_imp_for_exporter"
                }),  # "sigmas"]],
                left_on=["year", "exporter"],
                right_on=["year", "iso"],
                how="left",
            ).drop(columns=["iso"])

            ccy_acc = ccy_acc.merge(
                cy_accuracy[["year", "iso", "acc_exp", "acc_imp"]].rename(columns={
                    "acc_exp": "acc_exp_for_importer",
                    "acc_imp": "acc_imp_for_importer"
                }),  # "sigmas"]],
                left_on=["year", "importer"],
                right_on=["year", "iso"],
                how="left",
                suffixes=('', '_for_importer'),
            ).drop(columns=["iso"])
            
            ccy_acc = ccy_acc[ccy_acc.importer != ccy_acc.exporter]
            
            for entity in ['exporter', 'importer']:
                ccy_acc[f'tag_{entity[0]}'] = (~ccy_acc[entity].duplicated()).astype(int)

            # merged["tag_e"] = merged.duplicated("exporter", keep="first").astype(int)
            # merged["tag_i"] = merged.duplicated("importer", keep="first").astype(int)

            # remove trade values less than 1000
            ccy_acc.loc[ccy_acc["importvalue_fob"] < 1000, "importvalue_fob"] = 0.0
            ccy_acc.loc[ccy_acc["exportvalue_fob"] < 1000, "exportvalue_fob"] = 0.0

            # calculating percentiles grouped by unique exporter and then importer
            percentiles_e = ccy_acc[ccy_acc.tag_e == 1]["acc_exp_for_exporter"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).round(3)
            percentiles_i = ccy_acc[ccy_acc.tag_i == 1]["acc_imp_for_importer"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).round(3)
            
            columns_to_cast = ['acc_exp_for_exporter', 
                                  'acc_imp_for_exporter', 
                                  'acc_exp_for_importer', 
                                  'acc_imp_for_importer']

            for col in columns_to_cast:
                ccy_acc[col] = pd.to_numeric(ccy_acc[col], errors='coerce')


            # Weight Calculation
            # attractiveness of the exporter
            ccy_acc["weight"] = np.exp(ccy_acc["acc_exp_for_exporter"]) / (
                np.exp(ccy_acc["acc_exp_for_exporter"]) + np.exp(ccy_acc["acc_imp_for_importer"])
            )
            
            # Output the percentiles
            # logging.info(
            #     f":: exporter A_e :: 10, 25, 50, 75, 95 = {percentiles_e['10%']} & {percentiles_e['25%']} & {percentiles_e['50%']} & {percentiles_e['75%']} & {percentiles_e['90%']}"
            # )
            # logging.info(
            #     f":: importer A_i :: 10, 25, 50, 75, 95 = {percentiles_i['10%']} & {percentiles_i['25%']} & {percentiles_i['50%']} & {percentiles_i['75%']} & {percentiles_i['90%']}"
            # )

            # include set of countries
            ccy_acc = ccy_acc.assign(
                weight_exporter =np.where(
                    (ccy_acc.acc_exp_for_exporter.notna())
                    & (ccy_acc.acc_exp_for_exporter > percentiles_e[.10]),
                    1,
                    0,
                ),
                weight_importer=np.where(
                    (ccy_acc.acc_imp_for_importer.notna())
                    & (ccy_acc.acc_imp_for_importer > percentiles_i[.10]),
                    1,
                    0,
                ),
            )
            
            ccy_acc["discrep"] = np.exp(
                np.abs(np.log(ccy_acc["exportvalue_fob"] / ccy_acc["importvalue_fob"]))
            )
            
            ccy_acc["discrep"] = ccy_acc["discrep"].replace(np.nan, 99)

            
            df = self.calculate_estimated_value(ccy_acc, percentiles_e, percentiles_i)
            df = df.drop(columns=["discrep"])

            # Calculate mintrade and update estvalue
            df["min_trade"] = df[["exportvalue_fob", "importvalue_fob"]].min(axis=1)
            df.loc[
                (df["min_trade"].notna()) & (df["est_value"].isna()), "est_value"
            ] = df["min_trade"]

            # Rename columns
            df = df.rename(
                columns={
                    "exportvalue_fob": "export_value",
                    "importvalue_fob": "import_value",
                    "est_value": "final_value",
                }
            )

            # Select and reorder columns
            columns_to_keep = [
                "year",
                "exporter",
                "importer",
                "export_value",
                "import_value",
                "final_value",
                # "cif_ratio",
                "weight",
                "weight_exporter",
                "weight_importer",
                "acc_exp_for_exporter",
                "acc_imp_for_importer",
            ]
            df = df[columns_to_keep]

            # Save the DataFrame to a file
            output_path = os.path.join(
                self.intermediate_data_path, f"weights_{year}.parquet"
            )
            df.to_parquet(output_path)
            weights.append(df)

            
        weights_years_total = pd.concat(weights)
        output_path = os.path.join(
            self.processed_data_path, f"weights_{start_year}-{end_year}.parquet"
        )
        weights_years_total.to_parquet(output_path)

        
    def calculate_estimated_value(self, df, perc_e, perc_i):
        """
        Series of conditions to determine estimated trade value
        """
        logging.info("Estimating total trade flows between countries")
        
        df["est_value"] = np.where(
            (df.acc_exp_for_exporter.notna())
            & (df.acc_imp_for_importer.notna())
            & (df.importvalue_fob.notna())
            & (df.exportvalue_fob.notna()),
            df.exportvalue_fob * df.weight + df.importvalue_fob * (1 - df.weight),
            np.nan,
        )
        
        # conditions to determine est_value
        df.loc[(df['acc_exp_for_exporter'] < perc_e[.50]) & (df['acc_imp_for_importer'] >= perc_i[.90]) & (df['acc_imp_for_importer'].notna()) & (df['acc_exp_for_exporter'].notna()) & df['est_value'].isna(), 'est_value'] = df['importvalue_fob']
        df.loc[(df['acc_exp_for_exporter'] >= perc_e[.90]) & (df['acc_imp_for_importer'] < perc_i[.50]) & (df['acc_imp_for_importer'].notna()) & (df['acc_exp_for_exporter'].notna()) & df['est_value'].isna(), 'est_value'] = df['exportvalue_fob']
        df.loc[(df['acc_exp_for_exporter'] < perc_e[.25]) & (df['acc_imp_for_importer'] >= perc_i[.75]) & (df['acc_imp_for_importer'].notna()) & (df['acc_exp_for_exporter'].notna()) & df['est_value'].isna(), 'est_value'] = df['importvalue_fob']
        df.loc[(df['acc_exp_for_exporter'] >= perc_e[.75]) & (df['acc_imp_for_importer'] < perc_i[.25]) & (df['acc_imp_for_importer'].notna()) & (df['acc_exp_for_exporter'].notna()) & df['est_value'].isna(), 'est_value'] = df['exportvalue_fob']
        df.loc[(df['acc_imp_for_importer'].notna()) & (df['acc_exp_for_exporter'].notna()) & (df['weight_exporter'] == 1) & (df['weight_importer'] == 1) & df['est_value'].isna(), 'est_value'] = df[['importvalue_fob', 'exportvalue_fob']].max(axis=1)
        df.loc[(df['weight_importer'] == 1) & df['est_value'].isna(), 'est_value'] = df['importvalue_fob']
        df.loc[(df['weight_exporter'] == 1) & df['est_value'].isna(), 'est_value'] = df['exportvalue_fob']
        df.loc[df['est_value'].isna(), 'est_value'] = df['importvalue_fob']
        df.loc[df['est_value'] == 0, 'est_value'] = np.nan        
       
        
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
        base_year_cpi_index = wdi_cpi.loc[
            wdi_cpi.year == self.CPI_BASE_YEAR, "cpi_index"
        ].iloc[0]
        wdi_cpi["cpi_index_base"] = wdi_cpi["cpi_index"] / base_year_cpi_index

        wdi_cpi.to_parquet(
            os.path.join(self.intermediate_data_path, "inflation_index.parquet")
        )
        return wdi_cpi
