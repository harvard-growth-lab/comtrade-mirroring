import pyarrow as pa

dataverse_datasets = {
    "conversion_weights": {"doi": "10.7910/DVN/6AADMR", "version": "latest"},
    "bilateral_reported_trade": {"doi": "10.7910/DVN/5NGVOB", "version": "latest"},
}

classification_translation_dict = {
    "S1": "SITCRev1",
    "S2": "SITCRev2",
    "S3": "SITCRev3",
    "H0": "HS1992",
    "H1": "HS1996",
    "H2": "HS2002",
    "H3": "HS2007",
    "H4": "HS2012",
    "H5": "HS2017",
    "H6": "HS2022",
}

trade_data_cols_renamed = {
    "year": "year",
    "exporter": "exporter_iso",
    "importer": "importer_iso",
    "commoditycode": "product_code",
    "value_final": "imputed_value",
    "value_exporter": "value_reported_by_exporter",
    "value_importer": "value_reported_by_importer",
}

trade_data_optimized_dtypes = {
    "year": "int16",
    "exporter_iso": "category",
    "importer_iso": "category",
    "product_code": "category",
    "imputed_value": "float32",
    "value_reported_by_exporter": "float32",
    "value_reported_by_importer": "float32",
}

trade_data_optimized_pyarrow_dtypes = [
    pa.field("year", pa.int16()),
    pa.field("exporter_iso", pa.dictionary(pa.int32(), pa.string())),
    pa.field("importer_iso", pa.dictionary(pa.int32(), pa.string())),
    pa.field("product_code", pa.dictionary(pa.int32(), pa.string())),
    pa.field("imputed_value", pa.float32()),
    pa.field("value_reported_by_exporter", pa.float32()),
    pa.field("value_reported_by_importer", pa.float32()),
]


table_display_names = {
    "location_country": "Country Classification",
    "location_group": "Regional Classification",
    "product_hs92": "HS92 Product Classification",
    "product_hs12": "HS12 Product Classification",
    "product_sitc": "SITC Product Classification",
    "product_services_unilateral": "Services Product Classification",
    "growth_proj_eci_rankings": "Complexity Rankings & Growth Projections",
    "umap_layout_hs92": "Product Space Layout",
    "top_edges_hs92": "Product Space Related Edges",
}

trade_data_table_display_names = {
    "CCPY": "Country Trade by Partner and Product",
    "CPY": "Country Trade by Product",
    "CCY": "Country Trade by Partner",
    "CY": "Total Trade by Country",
    "PY": "Total Trade by Product",
}

file_repo_display_order = (
    "rankings",
    "hs92",
    "hs12",
    "sitc",
    "services_unilateral",
    "classification",
    "product_space",
)

file_product_classifications = ["hs92", "hs12", "sitc", "services_unilateral"]

file_facet_display_order = ("CPY", "CY", "PY", "CCY", "CCPY")
