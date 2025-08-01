import pandas as pd
import requests
from src.data_downloads.params.download_tables_conf import dataverse_datasets


class Metadata:
    def __init__(self, data_preparer_config: object):
        self.config = data_preparer_config
        # self.dataverse_datasets = self.config.dataverse_datasets
        self.DATAVERSE_API_URL = "https://dataverse.harvard.edu/api"
        DATAVERSE_FILE_FIELDS = {
            "id": "dv_file_id",
            "filename": "dv_file_name",
            "filesize": "dv_file_size",
            "publicationDate": "dv_publication_date",
        }
        TABLE_METADATA_COLUMN_RENAME = {
            "file_name": "table_name",
            "file_data_type": "data_type",
            "file_repo": "repo",
            "file_facet": "facet",
        }

    def human_readable_file_size(self, bytes: int) -> str:
        if bytes < 1024:
            return f"{bytes} B"
        elif bytes < 1024 * 1024:
            return f"{bytes / 1024:.2f} KB"
        elif bytes < 1024 * 1024 * 1024:
            return f"{bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{bytes / (1024 * 1024 * 1024):.2f} GB"

    def fetch_dataverse_file_metadata(self) -> pd.DataFrame:

        dataset_dfs = []

        for dataset, config in dataverse_datasets.items():
            doi = config["doi"]
            version = config.get("version")

            if (version == "latest") or (version is None):
                version = ":latest"
            elif version == "draft":
                version = ":draft"
            elif version == "latest-published":
                version = ":latest-published"

            dataset_files = (
                requests.get(
                    f"{self.DATAVERSE_API_URL}/datasets/:persistentId/"
                    f"versions/{version}"
                    f"?persistentId=doi:{doi}"
                )
                .json()
                .get("data")
                .get("files")
            )

            dataset_files = pd.DataFrame(
                [
                    {
                        k: f.get("dataFile").get(k)
                        for k in self.DATAVERSE_FILE_FIELDS.keys()
                    }
                    for f in dataset_files
                ]
            ).rename(columns=self.DATAVERSE_FILE_FIELDS)
            dataset_files["dataset"] = dataset
            dataset_files["doi"] = doi

            dataset_dfs.append(dataset_files)

        df = pd.concat(dataset_dfs)

        # Format file size from number of bytes into something human readable
        df.dv_file_size = df.dv_file_size.apply(self.human_readable_file_size)

        df["file_name_stub"] = df.dv_file_name.str.split(".").str[0]

        return df
