"""
Tests for src/utils/handle_config.py

Covers:
  - get_classifications_list(): existing function
  - load_config(): new function that replaces ConfigGenerator + importlib flow

Run with:
    pytest mirror/tests/test_handle_config.py
"""

import types
import pytest
import yaml

from mirror.src.utils.handle_config import get_classifications_list, load_config


# =============================================================================
# Fixtures
# =============================================================================

START_YEARS = {
    "H0": 1995,
    "H1": 1996,
    "H2": 2002,
    "H3": 2007,
    "H4": 2012,
    "H5": 2017,
    "H6": 2022,
    "S1": 1962,
    "S2": 1962,
    "S3": 1988,
    "EB10": 2010,
}

ALL_DISABLED = {
    "HS92": False,
    "HS96": False,
    "HS02": False,
    "HS07": False,
    "HS12": False,
    "HS17": False,
    "HS22": False,
    "SITC1": False,
    "SITC2": False,
    "SITC3": False,
    "EB10": False,
}


def _enable(classification_key):
    """Return classifications_dict with only one entry enabled."""
    d = dict(ALL_DISABLED)
    d[classification_key] = True
    return d


def _make_yaml(tmp_path, overrides=None):
    """Write a minimal valid config YAML to a temp file and return its path."""
    base = {
        "shared": {"end_year": 2024, "log_level": "INFO"},
        "classifications": {
            "hs22": True,
            "hs17": False,
            "hs12": False,
            "hs07": False,
            "hs02": False,
            "hs96": False,
            "hs92": False,
            "sitc1": False,
            "sitc2": False,
            "sitc3": False,
            "eb10": False,
        },
        "classification_start_years": dict(START_YEARS),
        "mirror": {
            "data_version": "2026_03_09",
            "download_type": "as_reported",
            "paths": [
                {"downloaded_files_path": "/data/input"},
                {"final_output_path": "/data/output"},
            ],
            "processing_steps": {
                "run_cleaning": True,
                "delete_intermediate_files": False,
            },
            "test_mode": False,
            "test_start_year": 2022,
        },
    }
    if overrides:
        # Shallow-merge top-level keys; caller is responsible for nesting
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v

    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.dump(base))
    return path


# =============================================================================
# get_classifications_list
# =============================================================================


class TestGetClassificationsList:
    def test_all_disabled_returns_empty(self):
        result = get_classifications_list(ALL_DISABLED, 2024, START_YEARS)
        assert result == []

    # --- individual classifications ---

    @pytest.mark.parametrize(
        "key, code, description",
        [
            ("SITC1", "S1", "SITC Revision 1"),
            ("SITC2", "S2", "SITC Revision 2"),
            ("SITC3", "S3", "SITC Revision 3"),
            ("HS92",  "H0", "HS92"),
            ("HS96",  "H1", "HS96"),
            ("HS02",  "H2", "HS02"),
            ("HS07",  "H3", "HS07"),
            ("HS12",  "H4", "HS12"),
            ("HS17",  "H5", "HS17"),
            ("HS22",  "H6", "HS22"),
            ("EB10",  "EB10", "EB10"),
        ],
    )
    def test_single_classification_code_and_description(self, key, code, description):
        result = get_classifications_list(_enable(key), 2024, START_YEARS)
        assert len(result) == 1
        assert result[0][0] == code
        assert result[0][3] == description

    @pytest.mark.parametrize(
        "key, expected_start",
        [
            ("SITC1", START_YEARS["S1"]),
            ("SITC2", START_YEARS["S2"]),
            ("SITC3", START_YEARS["S3"]),
            ("HS92",  START_YEARS["H0"]),
            ("HS96",  START_YEARS["H1"]),
            ("HS02",  START_YEARS["H2"]),
            ("HS07",  START_YEARS["H3"]),
            ("HS12",  START_YEARS["H4"]),
            ("HS17",  START_YEARS["H5"]),
            ("HS22",  START_YEARS["H6"]),
            ("EB10",  START_YEARS["EB10"]),
        ],
    )
    def test_normal_mode_uses_start_year_dict(self, key, expected_start):
        result = get_classifications_list(_enable(key), 2024, START_YEARS)
        assert result[0][1] == expected_start

    @pytest.mark.parametrize("key", ALL_DISABLED.keys())
    def test_test_mode_overrides_start_year(self, key):
        test_start = 2020
        result = get_classifications_list(
            _enable(key), 2024, START_YEARS, test_start_year=test_start
        )
        assert result[0][1] == test_start

    def test_end_year_is_passed_through(self):
        result = get_classifications_list(_enable("HS22"), 2022, START_YEARS)
        assert result[0][2] == 2022

    def test_multiple_enabled_returns_multiple_entries(self):
        d = dict(ALL_DISABLED)
        d["HS22"] = True
        d["SITC1"] = True
        result = get_classifications_list(d, 2024, START_YEARS)
        assert len(result) == 2
        codes = [r[0] for r in result]
        assert "H6" in codes
        assert "S1" in codes

    def test_all_enabled_returns_eleven_entries(self):
        all_on = {k: True for k in ALL_DISABLED}
        result = get_classifications_list(all_on, 2024, START_YEARS)
        assert len(result) == 11


# =============================================================================
# load_config
# =============================================================================


class TestLoadConfig:
    # --- attribute presence ---

    def test_returns_namespace_with_required_attributes(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        for attr in [
            "DATA_VERSION",
            "LOG_LEVEL",
            "DOWNLOAD_TYPE",
            "PATHS",
            "PROCESSING_STEPS",
            "TEST_MODE",
            "classifications",
        ]:
            assert hasattr(cfg, attr), f"Missing attribute: {attr}"

    # --- scalar values ---

    def test_data_version(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.DATA_VERSION == "2026_03_09"

    def test_log_level(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.LOG_LEVEL == "INFO"

    def test_download_type(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.DOWNLOAD_TYPE == "as_reported"

    def test_test_mode_false(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.TEST_MODE is False

    def test_test_mode_true(self, tmp_path):
        path = _make_yaml(tmp_path, {"mirror": {"test_mode": True, "test_start_year": 2022}})
        cfg = load_config(path)
        assert cfg.TEST_MODE is True

    # --- paths ---

    def test_paths_is_flat_dict(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert isinstance(cfg.PATHS, dict)
        assert "downloaded_files_path" in cfg.PATHS
        assert "final_output_path" in cfg.PATHS

    def test_paths_values(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.PATHS["downloaded_files_path"] == "/data/input"
        assert cfg.PATHS["final_output_path"] == "/data/output"

    # --- processing steps ---

    def test_processing_steps_is_dict(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert isinstance(cfg.PROCESSING_STEPS, dict)

    def test_processing_steps_values(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.PROCESSING_STEPS["run_cleaning"] is True
        assert cfg.PROCESSING_STEPS["delete_intermediate_files"] is False

    # --- classifications list ---

    def test_classifications_is_list(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert isinstance(cfg.classifications, list)

    def test_one_enabled_classification_produces_one_entry(self, tmp_path):
        # default YAML has only hs22=true
        cfg = load_config(_make_yaml(tmp_path))
        assert len(cfg.classifications) == 1

    def test_enabled_classification_has_correct_code(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.classifications[0][0] == "H6"  # hs22 → H6

    def test_enabled_classification_uses_start_year_dict(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.classifications[0][1] == START_YEARS["H6"]

    def test_enabled_classification_uses_end_year(self, tmp_path):
        cfg = load_config(_make_yaml(tmp_path))
        assert cfg.classifications[0][2] == 2024

    def test_test_mode_uses_test_start_year(self, tmp_path):
        overrides = {
            "mirror": {
                "test_mode": True,
                "test_start_year": 2020,
                "data_version": "2026_03_09",
                "download_type": "as_reported",
                "paths": [
                    {"downloaded_files_path": "/data/input"},
                    {"final_output_path": "/data/output"},
                ],
                "processing_steps": {
                    "run_cleaning": True,
                    "delete_intermediate_files": False,
                },
            }
        }
        path = _make_yaml(tmp_path, overrides)
        cfg = load_config(path)
        assert cfg.classifications[0][1] == 2020

    def test_multiple_classifications_enabled(self, tmp_path):
        overrides = {
            "classifications": {
                "hs22": True,
                "hs17": True,
                "hs12": False,
                "hs07": False,
                "hs02": False,
                "hs96": False,
                "hs92": False,
                "sitc1": False,
                "sitc2": False,
                "sitc3": False,
                "eb10": False,
            }
        }
        cfg = load_config(_make_yaml(tmp_path, overrides))
        assert len(cfg.classifications) == 2

    def test_no_classifications_enabled_returns_empty_list(self, tmp_path):
        overrides = {
            "classifications": {k: False for k in [
                "hs22", "hs17", "hs12", "hs07", "hs02",
                "hs96", "hs92", "sitc1", "sitc2", "sitc3", "eb10"
            ]}
        }
        cfg = load_config(_make_yaml(tmp_path, overrides))
        assert cfg.classifications == []

    # --- YAML key → classification code mapping ---

    @pytest.mark.parametrize(
        "yaml_key, expected_code",
        [
            ("hs92",  "H0"),
            ("hs96",  "H1"),
            ("hs02",  "H2"),
            ("hs07",  "H3"),
            ("hs12",  "H4"),
            ("hs17",  "H5"),
            ("hs22",  "H6"),
            ("sitc1", "S1"),
            ("sitc2", "S2"),
            ("sitc3", "S3"),
            ("eb10",  "EB10"),
        ],
    )
    def test_yaml_key_maps_to_correct_classification_code(self, tmp_path, yaml_key, expected_code):
        disabled = {k: False for k in [
            "hs22", "hs17", "hs12", "hs07", "hs02",
            "hs96", "hs92", "sitc1", "sitc2", "sitc3", "eb10"
        ]}
        disabled[yaml_key] = True
        cfg = load_config(_make_yaml(tmp_path, {"classifications": disabled}))
        assert len(cfg.classifications) == 1
        assert cfg.classifications[0][0] == expected_code

    # --- data_version edge cases ---

    def test_data_version_none_passes_through(self, tmp_path):
        overrides = {
            "mirror": {
                "data_version": None,
                "download_type": "as_reported",
                "paths": [
                    {"downloaded_files_path": "/data/input"},
                    {"final_output_path": "/data/output"},
                ],
                "processing_steps": {
                    "run_cleaning": True,
                    "delete_intermediate_files": False,
                },
                "test_mode": False,
                "test_start_year": 2022,
            }
        }
        cfg = load_config(_make_yaml(tmp_path, overrides))
        assert cfg.DATA_VERSION is None

    # --- error handling ---

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed bracket")
        with pytest.raises((ValueError, yaml.YAMLError)):
            load_config(bad)
