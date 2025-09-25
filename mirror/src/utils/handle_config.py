import sys
from pathlib import Path
from datetime import date




def get_data_version(data_version):
    """Generate data version string if not manually specified"""
    if data_version and data_version is not None and data_version != 'None':
        return data_version
    return f"{(date.today()).strftime('%Y_%m_%d')}"

def get_paths_config(base_attrs):
    """Generate full path configuration based on download type and data version"""

    if base_attrs['data_version'] is None:
        data_version = get_data_version(base_attrs['data_version'])
    final_output_path = (
        Path(base_attrs['paths']["final_output_path"]) / base_attrs['data_version'] / "mirrored_output"
    )
    final_output_path.mkdir(exist_ok=True, parents=True)

    root_dir = Path(__file__).parent.parent.parent.absolute()
    sys.path.insert(0, str(root_dir))


    return {
        "downloaded_files_path": Path(base_attrs['paths']["downloaded_files_path"]),
        "root_dir": str(root_dir),
        "final_output_path": final_output_path,
        "download_type": base_attrs['download_type'],
    }


def get_classifications_list(classifications_dict, end_year, start_year_dict, test_start_year=None) -> list:
    """Get the list of classifications to process based on settings"""
    classifications = []

    if classifications_dict["SITC1"]:
        start_year = test_start_year if test_start_year else start_year_dict['S1']
        classifications.append(("S1", start_year, end_year, "SITC Revision 1"))

    if classifications_dict["SITC2"]:
        start_year = test_start_year if test_start_year else start_year_dict['S2']
        classifications.append(("S2", start_year, end_year, "SITC Revision 2"))

    if classifications_dict["SITC3"]:
        start_year = test_start_year if test_start_year else start_year_dict['S3']
        classifications.append(("S3", start_year, end_year, "SITC Revision 3"))


    if classifications_dict["HS92"]:
        start_year = test_start_year if test_start_year else start_year_dict['H0']
        classifications.append(("H0", start_year, end_year, "HS92"))

    if classifications_dict["HS96"]:
        start_year = test_start_year if test_start_year else start_year_dict['H1']
        classifications.append(("H1", start_year, end_year, "HS96"))

    if classifications_dict["HS02"]:
        start_year = test_start_year if test_start_year else start_year_dict['H2']
        classifications.append(("H2", start_year, end_year, "HS02"))

    if classifications_dict["HS07"]:
        start_year = test_start_year if test_start_year else start_year_dict['H3']
        classifications.append(("H3", start_year, end_year, "HS07"))

    if classifications_dict["HS12"]:
        start_year = test_start_year if test_start_year else start_year_dict['H4']
        classifications.append(("H4", start_year, end_year, "HS12"))

    if classifications_dict["HS17"]:
        start_year = test_start_year if test_start_year else start_year_dict['H5']
        classifications.append(("H5", start_year, end_year, "HS17"))

    if classifications_dict["HS22"]:
        start_year = test_start_year if test_start_year else start_year_dict['H6']
        classifications.append(("H6", start_year, end_year, "HS22"))
        
        
    if classifications_dict["EB10"]:
        start_year = test_start_year if test_start_year else start_year_dict['EB10']
        classifications.append(("EB10", start_year, end_year, "EB10"))

    return classifications


# =============================================================================
# VALIDATION
# =============================================================================


def validate_config(paths, download_type, classifications_list):
    """Validate configuration settings"""
    errors = []

    # Check paths exist
    for path_name, path_value in paths.items():
        if not Path(path_value).exists():
            errors.append(f"Path does not exist: {path_name} = {path_value}")

    # Check download type
    if download_type not in ["by_classification", "as_reported"]:
        errors.append(f"Invalid DOWNLOAD_TYPE: {download_type}")

    # Check classifications
    valid_classifications = [
        "H0",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "SITC",
        "S1",
        "S2",
        "S3",
        "EB10"
    ]

    for classification, start_year, end_year, desc in classifications_list:
        if classification not in valid_classifications:
            errors.append(f"Invalid classification: {classification}")
        if start_year > end_year:
            errors.append(
                f"Invalid year range for {classification}: {start_year} > {end_year}"
            )
        if end_year > date.today().year:
            errors.append(f"End year {end_year} is in the future for {classification}")

    if errors:
        print("⚠️  Configuration Errors:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("✅ Configuration is valid")
    return errors


# =============================================================================
# REPORTING / INFO
# =============================================================================


def print_config_summary(test_mode, data_version, classifications_list, processing_steps):
    """Print a summary of current configuration"""
    print("=" * 60)
    print("BILATERAL MIRRORING CONFIGURATION")
    print("=" * 60)
    print(f"Data Version: {get_data_version(data_version)}")
    print(
        f"Test Mode: {'ON (2020-END_YEAR only)' if test_mode else 'OFF (full year range)'}"
    )
    print()

    if not classifications_list:
        print(
            "  ⚠️  Nothing selected! Turn on PROCESS_SITC, PROCESS_HS92, or PROCESS_HS12"
        )
    else:
        for classification, start_year, end_year, desc in classifications_list:
            print(f"  ✓ {desc}: {start_year}-{end_year}")
    print()

    print("Processing steps:")
    for step, enabled in processing_steps.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {step}")
    print()
    print("=" * 60)
    return