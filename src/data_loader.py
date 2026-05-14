"""
data_loader.py
--------------
Handles all data I/O for the dashboard.

Responsibilities:
  - Parse an uploaded CSV (via DuckDB for speed)
  - Cache the result as Parquet
  - Expose a clean interface for the rest of the app to query data
    without touching raw CSV again

Usage:
    from data_loader import load_csv, query_parquet, get_dataframe

    df, parquet_path = load_csv("/tmp/upload/titanic.csv")
    result_df        = query_parquet(parquet_path, "SELECT Sex, AVG(Age) FROM data GROUP BY Sex")
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Max rows returned to the UI for preview tables
PREVIEW_ROWS = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_csv(csv_path: str) -> tuple[pd.DataFrame, str]:
    """
    Read a CSV file via DuckDB and cache it as Parquet.

    Returns
    -------
    (DataFrame, parquet_path)

    Why DuckDB instead of pd.read_csv:
      - Parallel, vectorised CSV parsing — ~3-5x faster on files > 10k rows
      - read_csv_auto detects delimiter, quoting, encoding automatically
      - SAMPLE_SIZE=10000 ensures type inference uses a large representative sample
    """
    log.info(f"Loading CSV: {csv_path}")
    con = duckdb.connect()

    try:
        df: pd.DataFrame = con.execute(
            f"SELECT * FROM read_csv_auto('{csv_path}', SAMPLE_SIZE=10000)"
        ).df()
    except Exception as e:
        log.error(f"DuckDB failed to read CSV: {e} — falling back to pandas")
        df = pd.read_csv(csv_path)
    finally:
        con.close()

    parquet_path = _cache_parquet(df, csv_path)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df, parquet_path


def get_dataframe(parquet_path: str) -> pd.DataFrame:
    """
    Read the cached Parquet file back into a DataFrame.
    Use this instead of re-parsing the CSV in callbacks.
    """
    return pd.read_parquet(parquet_path)


def query_parquet(parquet_path: str, sql: str) -> pd.DataFrame:
    """
    Run a SQL query directly against the Parquet cache via DuckDB.
    The table alias inside SQL must be 'data'.

    Example
    -------
    query_parquet(path, "SELECT neighbourhood_group, AVG(price) FROM data GROUP BY 1")
    """
    con = duckdb.connect()
    try:
        # Register the Parquet file as a virtual table called 'data'
        con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{parquet_path}')")
        result = con.execute(sql).df()
        return result
    except Exception as e:
        log.error(f"DuckDB query failed: {e}\nSQL: {sql}")
        return pd.DataFrame()
    finally:
        con.close()


def preview(parquet_path: str, n: int = PREVIEW_ROWS) -> pd.DataFrame:
    """Return the first n rows for the data preview table in the UI."""
    return query_parquet(parquet_path, f"SELECT * FROM data LIMIT {n}")


def file_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_parquet(df: pd.DataFrame, csv_path: str) -> str:
    stem = Path(csv_path).stem
    parquet_path = str(CACHE_DIR / f"{stem}.parquet")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    log.info(f"Cached → {parquet_path} ({file_size_mb(parquet_path):.2f} MB)")
    return parquet_path
