"the parent object for Atlas Cleaning script"

from os import path
import typing


class _AtlasCleaning(object):
    # Classification names & levels
    PRODUCT_CLASSIFICATIONS = ["H0", "HS", "S1", "S2", "ST"]

    HIERARCHY_LEVELS = {
        "hs92": (1, 2, 4, 6),
        "hs12": (1, 2, 4, 6),
        "sitc": (1, 2, 4),
        "services": (1, 2, 4, 6),
    }

    REGIONAL_GROUP_TYPES = ["world", "region", "subregion"]

    INGESTION_OUTPUT_FORMATS = ["parquet", "hdf5"]

    def __init__(
        self,
        start_year,
        end_year,
        product_classification,
        root_dir,
    ):
        # INPUTS
        self.root_dir = root_dir
        self.data_path = path.join(self.root_dir, "data")
        self.raw_data_path = path.join(self.data_path, "raw")
        self.intermediate_data_path = path.join(self.data_path, "intermediate")
        self.processed_data_path = path.join(self.data_path, "processed")

        self.df = None
        self.start_year = start_year
        self.end_year = end_year
        self.product_classification = product_classification

    def load_parquet(
        self,
        table_name: str,
        schema: typing.Optional[str] = None,
    ):
        if schema is not None:
            read_dir = os.path.join(self.root_dir, schema)
        else:
            read_dir = os.path.join(self.root_dir)

        df = pd.read_parquet(os.path.join(read_dir, f"{table_name}.parquet"))
        if self.limit_data_coverage and "year" in df.columns:
            if schema == "sitc":
                df = df[df.year >= (self.latest_year - 25)]
            else:
                df = df[df.year >= self.earliest_year]

        return df

    def save_parquet(self, data_folder, table_name: str):
        save_dir = path.join(self.data_path, data_folder)
        save_path = path.join(save_dir, f"{table_name}.parquet")

        self.df.to_parquet(save_path, index=False)
