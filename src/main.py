from schema_detector import detect_schema

profile = detect_schema("../data/titanic.csv", use_ollama=True)

# Get all columns suitable for chart axes
numeric_cols = profile.by_type("numeric")

# Get columns that can serve as color groupings
color_cols = profile.with_role("color")

# Get the Parquet path for DuckDB to query later
parquet = profile.parquet_path  # → ".cache/titanic.parquet"
