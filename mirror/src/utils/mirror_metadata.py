import json
import subprocess
from pathlib import Path

import pandas as pd

from mirror.src.utils.logging import get_logger

logger = get_logger(__name__)


def write_mirror_metadata(final_output_path, downloaded_files_path, classifications, end_year):
    """Write mirror run metadata to the mirrored_output folder."""
    comtrade_report_path = (
        Path(downloaded_files_path).parent.parent
        / "atlas_download_reports"
        / "comtrade_data_version.csv"
    )
    comtrade_meta = pd.read_csv(comtrade_report_path).tail(1).iloc[0].to_dict()

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
        ).decode().strip()
    except Exception:
        git_sha = None

    classifications_dict = {
        desc.lower().replace(" ", "_"): start_year
        for _cls, start_year, _end_year, desc in classifications
    }

    metadata = {
        **comtrade_meta,
        "mirror_git_sha": git_sha,
        "classifications": classifications_dict,
        "end_year": end_year,
    }

    output_path = Path(final_output_path) / "mirror_metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Mirror metadata written to {output_path}")
