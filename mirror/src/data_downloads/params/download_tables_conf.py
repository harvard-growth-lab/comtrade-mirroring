dataverse_datasets = {
    "conversion_weights": {"doi": "10.7910/DVN/6AADMR", "version": "latest"},
    "bilateral_reported_trade": {"doi": "10.7910/DVN/5NGVOB", "version": "latest"},
}

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
