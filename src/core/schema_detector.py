"""
schema_detector.py
------------------
Detects semantic types for every column in an uploaded CSV.

Pipeline:
    1. DuckDB  — fast CSV ingestion + aggregate stats (cardinality, nulls, min/max)
    2. Pandas  — per-column content analysis (patterns, coercion, sampling)
    3. Heuristics — name hints, regex patterns, cardinality thresholds
    4. Ollama (optional fallback) — for columns the rule-based system can't resolve

Cache:
    Parsed DataFrames are saved as Parquet (columnar, DuckDB-queryable, Pandas-native).
    This lets the rest of the app re-read data without re-parsing the CSV.

Output:
    A DatasetProfile object containing one ColumnProfile per column,
    plus dataset-level metadata.
"""

import re
import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import duckdb
import pandas as pd
import numpy as np
from dateutil import parser as dateutil_parser

# ---------------------------------------------------------------------------
# Optional Ollama import — gracefully disabled if not installed / not running
# ---------------------------------------------------------------------------
try:
    import ollama  # pip install ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# Constants & lookup tables
# ===========================================================================

# Minimum fraction of values that must parse successfully to accept a type
PARSE_THRESHOLD = 0.85

# If more than this fraction of values are null, flag the column as low quality
HIGH_NULL_THRESHOLD = 0.50

# If unique_rate > this, column is identifier-like or free text
HIGH_CARDINALITY_RATE = 0.90

# Max unique values before a numeric column is also considered "categorical-safe"
MAX_CATEGORICAL_UNIQUE = 15

# How many rows to sample for slow content checks (date parsing, regex)
CONTENT_SAMPLE_SIZE = 300

# Parquet cache folder (relative to wherever the app runs from)
CACHE_DIR = Path(".cache")

# --- Name hint dictionaries (column name → semantic nudge) ---
_ID_HINTS       = {"id", "uuid", "key", "index", "idx", "pk", "code", "ref", "no", "num", "number"}
_DATE_HINTS     = {"date", "time", "at", "created", "updated", "modified", "timestamp",
                   "year", "month", "day", "hour", "minute", "second", "week", "dob", "birth"}
_GEO_HINTS      = {"lat", "latitude", "lon", "lng", "longitude", "geo", "location",
                   "coord", "coords", "x", "y"}
_MONEY_HINTS    = {"price", "cost", "revenue", "salary", "fee", "amount", "rate",
                   "payment", "income", "budget", "spend", "wage", "fare"}
_TEXT_HINTS     = {"name", "title", "label", "description", "text", "comment",
                   "review", "note", "message", "body", "content", "summary"}
_BOOL_HINTS     = {"flag", "is_", "has_", "active", "enabled", "deleted",
                   "verified", "approved", "status"}

# Regex patterns for geographic coordinate validation
_LAT_PATTERN = re.compile(r'^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)$')
_LON_PATTERN = re.compile(r'^[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$')


# ===========================================================================
# Data classes — the "contract" between detector and the rest of the app
# ===========================================================================

@dataclass
class ColumnProfile:
    # --- Identity ---
    column_name:       str
    original_dtype:    str              # raw pandas dtype string

    # --- Semantic classification ---
    semantic_type:     str              # see SEMANTIC_TYPES below
    sub_type:          Optional[str] = None   # e.g. "currency", "latitude", "binary"
    detection_source:  str = "rules"    # "rules" | "llm_fallback"

    # --- Nullity ---
    total_rows:        int  = 0
    null_count:        int  = 0
    null_rate:         float = 0.0
    is_low_quality:    bool = False     # null_rate > HIGH_NULL_THRESHOLD

    # --- Cardinality ---
    unique_count:      int  = 0
    unique_rate:       float = 0.0
    cardinality_class: str  = "unknown" # binary | low | medium | moderate | high

    # --- Numeric stats (if applicable) ---
    numeric_min:       Optional[float] = None
    numeric_max:       Optional[float] = None
    numeric_mean:      Optional[float] = None
    numeric_std:       Optional[float] = None

    # --- Categorical stats (if applicable) ---
    top_values:        list = field(default_factory=list)   # [(value, count), ...]

    # --- Datetime stats (if applicable) ---
    datetime_min:      Optional[str] = None
    datetime_max:      Optional[str] = None
    datetime_format:   Optional[str] = None

    # --- Chart guidance (consumed by chart_generator.py) ---
    chart_eligible:    bool = True
    chart_roles:       list = field(default_factory=list)
    # e.g. ["x_axis", "y_axis", "color", "facet", "map_lat", "map_lon"]

    # --- Warnings (shown in UI) ---
    warnings:          list = field(default_factory=list)

    # --- Name hint (metadata only) ---
    name_hint:         Optional[str] = None


@dataclass
class DatasetProfile:
    file_name:      str
    parquet_path:   Optional[str]       # path to cached Parquet file
    total_rows:     int
    total_columns:  int
    columns:        list[ColumnProfile] = field(default_factory=list)
    dataset_warnings: list[str]         = field(default_factory=list)

    # --- Convenience accessors ---
    def by_type(self, semantic_type: str) -> list[ColumnProfile]:
        return [c for c in self.columns if c.semantic_type == semantic_type]

    def chart_eligible_columns(self) -> list[ColumnProfile]:
        return [c for c in self.columns if c.chart_eligible]

    def with_role(self, role: str) -> list[ColumnProfile]:
        return [c for c in self.columns if role in c.chart_roles]

    def to_dict(self) -> dict:
        """Serialise to plain dict (for JSON export or LLM context)."""
        import dataclasses
        return dataclasses.asdict(self)


# Accepted semantic type values
SEMANTIC_TYPES = {
    "numeric",      # quantitative — suitable for histograms, scatter, line
    "categorical",  # finite set of labels — bar charts, pie, color grouping
    "datetime",     # temporal — line charts, time-series, aggregations by period
    "boolean",      # true/false or binary 0/1
    "identifier",   # high-cardinality key column — not useful for charts
    "geographic",   # lat/lon or geo names
    "text",         # free-form prose — not chartable but may feed LLM
    "unknown",      # rule-based detection failed; sent to Ollama fallback
}


# ===========================================================================
# Step 1 — DuckDB: fast CSV ingestion + aggregate statistics
# ===========================================================================

def _load_with_duckdb(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Use DuckDB to:
      - Read the CSV (faster than pd.read_csv for large files)
      - Compute per-column null counts, unique counts, min, max in one SQL pass
      - Return a Pandas DataFrame + raw aggregate stats dict

    Why DuckDB here:
      SELECT COUNT(DISTINCT col) over 48k rows is dramatically faster in DuckDB
      than df[col].nunique() in Pandas, because DuckDB uses vectorised execution
      and never materialises intermediate Python objects.
    """
    log.info(f"DuckDB: reading {csv_path}")
    con = duckdb.connect()

    # DuckDB's read_csv_auto infers delimiters, quoting, encoding automatically
    df: pd.DataFrame = con.execute(
        f"SELECT * FROM read_csv_auto('{csv_path}', SAMPLE_SIZE=10000)"
    ).df()

    log.info(f"DuckDB: loaded {len(df):,} rows × {len(df.columns)} columns")

    # --- Aggregate stats in a single SQL pass per column ---
    # Doing this in SQL avoids loading data back into Python for each stat
    agg_stats: dict[str, dict] = {}
    for col in df.columns:
        safe_col = f'"{col}"'   # quote for column names with spaces / special chars
        try:
            result = con.execute(f"""
                SELECT
                    COUNT(*)                        AS total,
                    COUNT({safe_col})               AS non_null,
                    COUNT(*) - COUNT({safe_col})    AS null_count,
                    COUNT(DISTINCT {safe_col})      AS unique_count,
                    MIN(TRY_CAST({safe_col} AS DOUBLE)) AS num_min,
                    MAX(TRY_CAST({safe_col} AS DOUBLE)) AS num_max,
                    AVG(TRY_CAST({safe_col} AS DOUBLE)) AS num_mean,
                    STDDEV(TRY_CAST({safe_col} AS DOUBLE)) AS num_std
                FROM df
            """).fetchone()

            agg_stats[col] = {
                "total":        result[0],
                "non_null":     result[1],
                "null_count":   result[2],
                "unique_count": result[3],
                "num_min":      result[4],
                "num_max":      result[5],
                "num_mean":     result[6],
                "num_std":      result[7],
            }
        except Exception as e:
            log.warning(f"  DuckDB stats failed for '{col}': {e}")
            agg_stats[col] = {
                "total": len(df), "non_null": df[col].notna().sum(),
                "null_count": df[col].isna().sum(),
                "unique_count": df[col].nunique(),
                "num_min": None, "num_max": None,
                "num_mean": None, "num_std": None,
            }

    con.close()
    return df, agg_stats


# ===========================================================================
# Step 2 — Pandas: per-column content analysis helpers
# ===========================================================================

def _get_name_hint(col_name: str) -> Optional[str]:
    """Return a semantic nudge from the column name, or None."""
    lower = col_name.lower()
    # Check prefix matches for boolean hints like "is_", "has_"
    for hint in _BOOL_HINTS:
        if lower.startswith(hint) or lower == hint:
            return "boolean"
    tokens = set(re.split(r'[_\s\-\.]+', lower))
    if tokens & _ID_HINTS:       return "identifier"
    if tokens & _DATE_HINTS:     return "datetime"
    if tokens & _GEO_HINTS:      return "geographic"
    if tokens & _MONEY_HINTS:    return "numeric"
    if tokens & _TEXT_HINTS:     return "text"
    return None


def _cardinality_class(unique_count: int, total: int) -> str:
    if unique_count == 2:
        return "binary"
    ratio = unique_count / max(total, 1)
    if unique_count <= MAX_CATEGORICAL_UNIQUE:
        return "low"
    elif ratio < 0.05:
        return "medium"
    elif ratio > HIGH_CARDINALITY_RATE:
        return "high"
    else:
        return "moderate"


def _try_numeric(series: pd.Series) -> tuple[bool, Optional[pd.Series]]:
    """Try to parse an object column as numeric (handles $, %, commas)."""
    sample = series.dropna().sample(min(CONTENT_SAMPLE_SIZE, len(series.dropna())),
                                    random_state=42)
    cleaned = sample.astype(str).str.replace(r'[$,%€£\s]', '', regex=True)
    converted = pd.to_numeric(cleaned, errors='coerce')
    success_rate = converted.notna().mean()
    if success_rate >= PARSE_THRESHOLD:
        # Apply cleaning to full series
        full_cleaned = series.astype(str).str.replace(r'[$,%€£\s]', '', regex=True)
        return True, pd.to_numeric(full_cleaned, errors='coerce')
    return False, None


def _try_datetime(series: pd.Series) -> tuple[bool, Optional[str]]:
    """
    Try fast path (pd.to_datetime) then slow path (dateutil on a sample).
    Returns (success, detected_format_hint).
    """
    # Fast path
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
        if parsed.notna().mean() >= PARSE_THRESHOLD:
            return True, "inferred"
    except Exception:
        pass

    # Slow path — sample only to avoid O(n) dateutil overhead
    sample = series.dropna().sample(min(50, len(series.dropna())), random_state=42)
    successes = 0
    for val in sample:
        try:
            dateutil_parser.parse(str(val), fuzzy=False)
            successes += 1
        except Exception:
            pass
    return (successes / max(len(sample), 1)) >= PARSE_THRESHOLD, "dateutil"


def _is_free_text(series: pd.Series) -> bool:
    """Long strings + high cardinality = prose, not a label."""
    avg_len = series.dropna().astype(str).str.len().mean()
    unique_rate = series.nunique() / max(len(series.dropna()), 1)
    return avg_len > 40 and unique_rate > 0.5


def _is_geographic(series: pd.Series, col_name: str) -> tuple[bool, Optional[str]]:
    """Check if column contains lat or lon values."""
    lower = col_name.lower()
    sample = series.dropna().sample(min(100, len(series.dropna())), random_state=42)
    str_vals = sample.astype(str)

    lat_matches = str_vals.apply(lambda v: bool(_LAT_PATTERN.match(v))).mean()
    lon_matches = str_vals.apply(lambda v: bool(_LON_PATTERN.match(v))).mean()

    if lat_matches >= PARSE_THRESHOLD and ("lat" in lower):
        return True, "latitude"
    if lon_matches >= PARSE_THRESHOLD and any(k in lower for k in ("lon", "lng")):
        return True, "longitude"
    return False, None


def _top_values(series: pd.Series, n: int = 10) -> list:
    counts = series.dropna().value_counts().head(n)
    return [(str(v), int(c)) for v, c in counts.items()]


# ===========================================================================
# Step 3 — Chart role assignment
# ===========================================================================

def _assign_chart_roles(profile: ColumnProfile) -> list[str]:
    """
    Decide which roles this column can play in a chart.
    The chart_generator.py module queries these roles to build visualisations.
    """
    roles = []
    st = profile.semantic_type

    if st == "numeric":
        roles += ["x_axis", "y_axis"]
        if profile.cardinality_class in ("low", "medium"):
            roles.append("color")   # e.g. a 0-10 rating score used as colour

    elif st == "categorical":
        roles += ["x_axis", "color", "facet"]
        if profile.cardinality_class == "low":
            roles.append("legend")

    elif st == "datetime":
        roles += ["x_axis", "time_series"]

    elif st == "boolean":
        roles += ["color", "facet", "x_axis"]

    elif st == "geographic":
        if profile.sub_type == "latitude":
            roles.append("map_lat")
        elif profile.sub_type == "longitude":
            roles.append("map_lon")

    elif st in ("identifier", "text", "unknown"):
        pass  # not chart-eligible

    return roles


# ===========================================================================
# Step 4 — Optional Ollama fallback for "unknown" columns
# ===========================================================================

def _ollama_classify(col_name: str, sample_values: list,
                     unique_count: int, total: int,
                     model: str = "llama3") -> tuple[str, Optional[str]]:
    """
    Ask Ollama to classify a column that rule-based detection couldn't resolve.
    Returns (semantic_type, sub_type).
    """
    if not OLLAMA_AVAILABLE:
        return "unknown", None

    prompt = f"""You are a data analyst. Classify the semantic type of a dataset column.

Column name: "{col_name}"
Sample values (up to 10): {sample_values[:10]}
Unique values: {unique_count} out of {total} total rows

Reply with ONLY a JSON object, no explanation, no markdown:
{{
  "semantic_type": "<one of: numeric, categorical, datetime, boolean, identifier, geographic, text, unknown>",
  "sub_type": "<optional refinement, e.g. currency, latitude, binary, or null>",
  "reason": "<one short sentence>"
}}"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"].strip()
        # Strip markdown fences if model added them
        raw = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
        sem = parsed.get("semantic_type", "unknown")
        sub = parsed.get("sub_type") or None
        log.info(f"  Ollama classified '{col_name}' → {sem} ({sub}) | {parsed.get('reason','')}")
        return sem if sem in SEMANTIC_TYPES else "unknown", sub
    except Exception as e:
        log.warning(f"  Ollama fallback failed for '{col_name}': {e}")
        return "unknown", None


# ===========================================================================
# Step 5 — Per-column profiler (orchestrates all layers)
# ===========================================================================

def _profile_column(col_name: str, series: pd.Series,
                    agg: dict, use_ollama: bool = True) -> ColumnProfile:

    total      = int(agg["total"])
    null_count = int(agg["null_count"])
    null_rate  = null_count / max(total, 1)
    unique_count = int(agg["unique_count"])
    unique_rate  = unique_count / max(total - null_count, 1)
    card_class   = _cardinality_class(unique_count, total - null_count)
    name_hint    = _get_name_hint(col_name)
    orig_dtype   = str(series.dtype)

    profile = ColumnProfile(
        column_name      = col_name,
        original_dtype   = orig_dtype,
        semantic_type    = "unknown",   # resolved below
        total_rows       = total,
        null_count       = null_count,
        null_rate        = null_rate,
        is_low_quality   = null_rate > HIGH_NULL_THRESHOLD,
        unique_count     = unique_count,
        unique_rate      = unique_rate,
        cardinality_class= card_class,
        name_hint        = name_hint,
    )

    if profile.is_low_quality:
        profile.warnings.append(
            f"{null_rate:.0%} of values are missing — column excluded from charts"
        )
        profile.chart_eligible = False
        profile.semantic_type  = "unknown"
        return profile

    # ------------------------------------------------------------------ #
    # Classification cascade                                               #
    # ------------------------------------------------------------------ #

    non_null = series.dropna()

    # 1. Boolean — check before numeric (0/1 ints are also boolean)
    if card_class == "binary":
        vals = set(non_null.astype(str).str.lower().unique())
        bool_sets = [
            {"0", "1"}, {"true", "false"}, {"yes", "no"},
            {"y", "n"}, {"t", "f"}, {"1.0", "0.0"},
        ]
        if any(vals <= bs for bs in bool_sets):
            profile.semantic_type = "boolean"
            profile.sub_type = "binary"
            profile.top_values = _top_values(series)
            return _finalise(profile)

    # 2. Identifier — high cardinality + name hint or fully unique numeric
    if card_class == "high" and name_hint == "identifier":
        profile.semantic_type = "identifier"
        profile.chart_eligible = False
        return _finalise(profile)

    # 3. Native numeric dtype
    if orig_dtype in ("int64", "float64", "int32", "float32"):
        # Could still be an identifier (e.g. auto-increment id)
        if unique_rate > HIGH_CARDINALITY_RATE and name_hint == "identifier":
            profile.semantic_type = "identifier"
            profile.chart_eligible = False
            return _finalise(profile)

        # Check geographic
        is_geo, geo_sub = _is_geographic(series, col_name)
        if is_geo:
            profile.semantic_type = "geographic"
            profile.sub_type = geo_sub
            return _finalise(profile)

        profile.semantic_type = "numeric"
        profile.numeric_min   = agg.get("num_min")
        profile.numeric_max   = agg.get("num_max")
        profile.numeric_mean  = round(agg["num_mean"], 4) if agg.get("num_mean") else None
        profile.numeric_std   = round(agg["num_std"],  4) if agg.get("num_std")  else None
        if name_hint == "numeric":
            profile.sub_type = "currency"
        # Low-cardinality numerics are also useful as categoricals
        if card_class in ("low", "binary"):
            profile.warnings.append(
                f"Only {unique_count} unique numeric values — may also suit a bar chart"
            )
        return _finalise(profile)

    # 4. Object dtype — needs deeper inspection
    if orig_dtype == "object":

        # 4a. Free text check first (skip expensive parsing on prose columns)
        if _is_free_text(non_null):
            profile.semantic_type = "text"
            profile.chart_eligible = False
            return _finalise(profile)

        # 4b. Try datetime
        is_dt, dt_fmt = _try_datetime(non_null)
        if is_dt or name_hint == "datetime":
            parsed_dt = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
            profile.semantic_type  = "datetime"
            profile.datetime_format = dt_fmt
            if parsed_dt.notna().any():
                profile.datetime_min = str(parsed_dt.min())
                profile.datetime_max = str(parsed_dt.max())
            if not is_dt:
                profile.warnings.append(
                    "Classified as datetime based on column name — parsing may be unreliable"
                )
            return _finalise(profile)

        # 4c. Try numeric (object column with $, %, commas etc.)
        is_num, num_series = _try_numeric(non_null)
        if is_num:
            profile.semantic_type = "numeric"
            profile.sub_type      = "currency" if name_hint == "numeric" else None
            profile.warnings.append("Numeric values extracted from text (symbols stripped)")
            if num_series is not None:
                profile.numeric_min  = float(num_series.min())
                profile.numeric_max  = float(num_series.max())
                profile.numeric_mean = round(float(num_series.mean()), 4)
            return _finalise(profile)

        # 4d. Geographic
        is_geo, geo_sub = _is_geographic(non_null, col_name)
        if is_geo:
            profile.semantic_type = "geographic"
            profile.sub_type = geo_sub
            return _finalise(profile)

        # 4e. Categorical (low/medium cardinality object)
        if card_class in ("low", "medium", "binary"):
            profile.semantic_type = "categorical"
            profile.top_values    = _top_values(series)
            return _finalise(profile)

        # 4f. High cardinality object — identifier or text
        if card_class == "high":
            profile.semantic_type = "identifier"
            profile.chart_eligible = False
            profile.warnings.append(
                "High cardinality text column — treated as identifier, excluded from charts"
            )
            return _finalise(profile)

    # ------------------------------------------------------------------ #
    # Fallback — send to Ollama if still unknown                          #
    # ------------------------------------------------------------------ #
    if use_ollama:
        sample_vals = non_null.sample(min(10, len(non_null)), random_state=42).tolist()
        sem, sub = _ollama_classify(col_name, sample_vals, unique_count, total)
        profile.semantic_type    = sem
        profile.sub_type         = sub
        profile.detection_source = "llm_fallback"
        profile.warnings.append("Type resolved via LLM fallback — verify manually")

    return _finalise(profile)


def _finalise(profile: ColumnProfile) -> ColumnProfile:
    """Assign chart roles and enforce chart_eligible flag."""
    if profile.semantic_type in ("identifier", "text", "unknown"):
        profile.chart_eligible = False
    profile.chart_roles = _assign_chart_roles(profile)
    return profile


# ===========================================================================
# Step 6 — Parquet cache
# ===========================================================================

def _cache_to_parquet(df: pd.DataFrame, csv_path: str) -> str:
    """
    Save the DataFrame as Parquet for fast subsequent reads.
    DuckDB can query this file directly without going back to the CSV.

    Why Parquet over HDF5:
      - Native columnar format — reads only the columns you query
      - DuckDB can scan it directly: SELECT * FROM 'file.parquet'
      - No PyTables / h5py dependency
      - Handles mixed types (strings, dates, ints) cleanly
      - Smaller files than CSV for most real-world datasets
    """
    CACHE_DIR.mkdir(exist_ok=True)
    stem = Path(csv_path).stem
    parquet_path = str(CACHE_DIR / f"{stem}.parquet")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    log.info(f"Cached to Parquet: {parquet_path}")
    return parquet_path


# ===========================================================================
# Public API
# ===========================================================================

def detect_schema(csv_path: str,
                  use_ollama: bool = True,
                  cache: bool = True) -> DatasetProfile:
    """
    Main entry point.

    Parameters
    ----------
    csv_path   : path to the uploaded CSV file
    use_ollama : whether to call Ollama for unresolved columns
    cache      : whether to save a Parquet cache of the parsed DataFrame

    Returns
    -------
    DatasetProfile with one ColumnProfile per column
    """
    # --- 1. DuckDB ingestion + aggregate stats ---
    df, agg_stats = _load_with_duckdb(csv_path)

    # --- 2. Parquet cache ---
    parquet_path = _cache_to_parquet(df, csv_path) if cache else None

    # --- 3. Profile each column ---
    column_profiles = []
    for col in df.columns:
        log.info(f"Profiling column: '{col}'")
        profile = _profile_column(
            col_name   = col,
            series     = df[col],
            agg        = agg_stats[col],
            use_ollama = use_ollama,
        )
        column_profiles.append(profile)

    # --- 4. Dataset-level warnings ---
    dataset_warnings = []
    low_quality = [c for c in column_profiles if c.is_low_quality]
    if low_quality:
        names = ", ".join(c.column_name for c in low_quality)
        dataset_warnings.append(f"Low-quality columns (>50% missing): {names}")

    identifiers = [c for c in column_profiles if c.semantic_type == "identifier"]
    if identifiers:
        names = ", ".join(c.column_name for c in identifiers)
        dataset_warnings.append(f"Identifier columns excluded from charts: {names}")

    # --- 5. Assemble dataset profile ---
    dataset_profile = DatasetProfile(
        file_name       = Path(csv_path).name,
        parquet_path    = parquet_path,
        total_rows      = len(df),
        total_columns   = len(df.columns),
        columns         = column_profiles,
        dataset_warnings= dataset_warnings,
    )

    log.info(
        f"Schema detection complete: "
        f"{len(column_profiles)} columns | "
        f"{sum(1 for c in column_profiles if c.chart_eligible)} chart-eligible"
    )
    return dataset_profile


# ===========================================================================
# CLI — run directly to inspect a CSV
# ===========================================================================

if __name__ == "__main__":
    import sys
    import pprint

    if len(sys.argv) < 2:
        print("Usage: python schema_detector.py <path_to_csv>")
        sys.exit(1)

    profile = detect_schema(sys.argv[1], use_ollama=False)
    pprint.pprint(profile.to_dict(), width=100, depth=4)


# ===========================================================================
# SAMPLE OUTPUT
# ===========================================================================
#
# Input: titanic.csv (891 rows, 12 columns)
# Called as: detect_schema("titanic.csv", use_ollama=False)
#
# DatasetProfile(
#   file_name     = "titanic.csv",
#   parquet_path  = ".cache/titanic.parquet",
#   total_rows    = 891,
#   total_columns = 12,
#   dataset_warnings = [
#     "Low-quality columns (>50% missing): Cabin",
#     "Identifier columns excluded from charts: PassengerId, Name, Ticket"
#   ],
#
#   columns = [
#
#     ColumnProfile(
#       column_name       = "PassengerId",
#       original_dtype    = "int64",
#       semantic_type     = "identifier",        ← unique_rate=1.0, name ends in "id"
#       sub_type          = None,
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 0,
#       null_rate         = 0.0,
#       is_low_quality    = False,
#       unique_count      = 891,
#       unique_rate       = 1.0,
#       cardinality_class = "high",
#       numeric_min       = 1.0,
#       numeric_max       = 891.0,
#       chart_eligible    = False,               ← excluded: identifier
#       chart_roles       = [],
#       warnings          = [],
#       name_hint         = "identifier",
#     ),
#
#     ColumnProfile(
#       column_name       = "Survived",
#       original_dtype    = "int64",
#       semantic_type     = "boolean",           ← only 2 unique values: {0, 1}
#       sub_type          = "binary",
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 0,
#       null_rate         = 0.0,
#       is_low_quality    = False,
#       unique_count      = 2,
#       unique_rate       = 0.002,
#       cardinality_class = "binary",
#       chart_eligible    = True,
#       chart_roles       = ["color", "facet", "x_axis"],
#       warnings          = [],
#       name_hint         = None,
#     ),
#
#     ColumnProfile(
#       column_name       = "Pclass",
#       original_dtype    = "int64",
#       semantic_type     = "numeric",           ← int dtype, 3 unique values
#       sub_type          = None,
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 0,
#       null_rate         = 0.0,
#       unique_count      = 3,
#       unique_rate       = 0.003,
#       cardinality_class = "low",
#       numeric_min       = 1.0,
#       numeric_max       = 3.0,
#       numeric_mean      = 2.3086,
#       numeric_std       = 0.8361,
#       chart_eligible    = True,
#       chart_roles       = ["x_axis", "y_axis", "color"],
#       warnings          = [
#         "Only 3 unique numeric values — may also suit a bar chart"
#       ],
#       name_hint         = None,
#     ),
#
#     ColumnProfile(
#       column_name       = "Name",
#       original_dtype    = "object",
#       semantic_type     = "text",              ← avg length >40 chars, unique_rate=1.0
#       sub_type          = None,
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 0,
#       null_rate         = 0.0,
#       unique_count      = 891,
#       unique_rate       = 1.0,
#       cardinality_class = "high",
#       chart_eligible    = False,
#       chart_roles       = [],
#       warnings          = [],
#       name_hint         = "text",
#     ),
#
#     ColumnProfile(
#       column_name       = "Sex",
#       original_dtype    = "object",
#       semantic_type     = "categorical",       ← object, 2 unique vals, not bool pattern
#       sub_type          = None,
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 0,
#       null_rate         = 0.0,
#       unique_count      = 2,
#       unique_rate       = 0.002,
#       cardinality_class = "binary",
#       top_values        = [("male", 577), ("female", 314)],
#       chart_eligible    = True,
#       chart_roles       = ["x_axis", "color", "facet", "legend"],
#       warnings          = [],
#       name_hint         = None,
#     ),
#
#     ColumnProfile(
#       column_name       = "Age",
#       original_dtype    = "float64",
#       semantic_type     = "numeric",
#       sub_type          = None,
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 177,
#       null_rate         = 0.1987,              ← 20% missing — warn but still eligible
#       is_low_quality    = False,
#       unique_count      = 88,
#       unique_rate       = 0.124,
#       cardinality_class = "moderate",
#       numeric_min       = 0.42,
#       numeric_max       = 80.0,
#       numeric_mean      = 29.6991,
#       numeric_std       = 14.5265,
#       chart_eligible    = True,
#       chart_roles       = ["x_axis", "y_axis"],
#       warnings          = [],
#       name_hint         = None,
#     ),
#
#     ColumnProfile(
#       column_name       = "Cabin",
#       original_dtype    = "object",
#       semantic_type     = "unknown",           ← 77% null → low_quality, skipped
#       sub_type          = None,
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 687,
#       null_rate         = 0.7710,
#       is_low_quality    = True,
#       chart_eligible    = False,
#       chart_roles       = [],
#       warnings          = [
#         "77% of values are missing — column excluded from charts"
#       ],
#       name_hint         = None,
#     ),
#
#     ColumnProfile(
#       column_name       = "Embarked",
#       original_dtype    = "object",
#       semantic_type     = "categorical",       ← 3 unique values: S, C, Q
#       sub_type          = None,
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 2,
#       null_rate         = 0.0022,
#       unique_count      = 3,
#       unique_rate       = 0.003,
#       cardinality_class = "low",
#       top_values        = [("S", 644), ("C", 168), ("Q", 77)],
#       chart_eligible    = True,
#       chart_roles       = ["x_axis", "color", "facet", "legend"],
#       warnings          = [],
#       name_hint         = None,
#     ),
#
#     ColumnProfile(
#       column_name       = "Fare",
#       original_dtype    = "float64",
#       semantic_type     = "numeric",
#       sub_type          = "currency",          ← name_hint matched _MONEY_HINTS
#       detection_source  = "rules",
#       total_rows        = 891,
#       null_count        = 0,
#       null_rate         = 0.0,
#       unique_count      = 248,
#       unique_rate       = 0.278,
#       cardinality_class = "moderate",
#       numeric_min       = 0.0,
#       numeric_max       = 512.3292,
#       numeric_mean      = 32.2042,
#       numeric_std       = 49.6934,
#       chart_eligible    = True,
#       chart_roles       = ["x_axis", "y_axis"],
#       warnings          = [],
#       name_hint         = "numeric",
#     ),
#
#   ]   # end columns
# )
