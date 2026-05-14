"""
stats_engine.py
---------------
Computes all statistical summaries needed by both the chart generator
and the insight engine.

Everything here is deterministic — no LLM calls.
The insight_engine.py will consume these results and pass them to Ollama.

Usage:
    from stats_engine import compute_stats
    from schema_detector import detect_schema

    profile = detect_schema("titanic.csv")
    stats   = compute_stats(df, profile)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from schema_detector import DatasetProfile, ColumnProfile

log = logging.getLogger(__name__)


# ===========================================================================
# Output data classes
# ===========================================================================


@dataclass
class NumericStats:
    column: str
    count: int
    null_rate: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    outlier_count: int  # values outside 1.5×IQR
    outlier_rate: float


@dataclass
class CategoricalStats:
    column: str
    count: int
    null_rate: float
    unique_count: int
    top_values: list  # [(label, count, pct), ...]
    entropy: float  # Shannon entropy — high = evenly distributed


@dataclass
class DatetimeStats:
    column: str
    min: str
    max: str
    span_days: int
    most_common_period: Optional[str]  # "daily" | "monthly" | "yearly" | None


@dataclass
class CorrelationPair:
    col_a: str
    col_b: str
    pearson_r: float
    strength: str  # "strong" | "moderate" | "weak"
    direction: str  # "positive" | "negative"


@dataclass
class DatasetStats:
    numeric: list[NumericStats] = field(default_factory=list)
    categorical: list[CategoricalStats] = field(default_factory=list)
    datetime: list[DatetimeStats] = field(default_factory=list)
    correlations: list[CorrelationPair] = field(default_factory=list)
    # High-level summary strings consumed by insight_engine
    summary_bullets: list[str] = field(default_factory=list)


# ===========================================================================
# Numeric stats
# ===========================================================================


def _numeric_stats(series: pd.Series, profile: ColumnProfile) -> NumericStats:
    clean = pd.to_numeric(series, errors="coerce").dropna()

    q25, q75 = clean.quantile([0.25, 0.75])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outliers = clean[(clean < lower) | (clean > upper)]

    return NumericStats(
        column=profile.column_name,
        count=int(clean.count()),
        null_rate=profile.null_rate,
        mean=round(float(clean.mean()), 4),
        median=round(float(clean.median()), 4),
        std=round(float(clean.std()), 4),
        min=round(float(clean.min()), 4),
        max=round(float(clean.max()), 4),
        q25=round(float(q25), 4),
        q75=round(float(q75), 4),
        skewness=round(float(clean.skew()), 4),
        kurtosis=round(float(clean.kurt()), 4),
        outlier_count=len(outliers),
        outlier_rate=round(len(outliers) / max(len(clean), 1), 4),
    )


# ===========================================================================
# Categorical stats
# ===========================================================================


def _shannon_entropy(counts: pd.Series) -> float:
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def _categorical_stats(series: pd.Series, profile: ColumnProfile) -> CategoricalStats:
    clean = series.dropna()
    counts = clean.value_counts()
    total = len(clean)
    top = [
        (str(v), int(c), round(c / total * 100, 1)) for v, c in counts.head(10).items()
    ]

    return CategoricalStats(
        column=profile.column_name,
        count=total,
        null_rate=profile.null_rate,
        unique_count=int(series.nunique()),
        top_values=top,
        entropy=round(_shannon_entropy(counts), 4),
    )


# ===========================================================================
# Datetime stats
# ===========================================================================


def _datetime_stats(
    series: pd.Series, profile: ColumnProfile
) -> Optional[DatetimeStats]:
    try:
        parsed = pd.to_datetime(
            series, infer_datetime_format=True, errors="coerce"
        ).dropna()
        if parsed.empty:
            return None
        span = (parsed.max() - parsed.min()).days
        # Infer granularity from median gap between sorted values
        diffs = parsed.sort_values().diff().dropna()
        median_gap = diffs.median()
        if median_gap.days < 2:
            period = "daily"
        elif median_gap.days < 10:
            period = "weekly"
        elif median_gap.days < 40:
            period = "monthly"
        else:
            period = "yearly"

        return DatetimeStats(
            column=profile.column_name,
            min=str(parsed.min().date()),
            max=str(parsed.max().date()),
            span_days=span,
            most_common_period=period,
        )
    except Exception as e:
        log.warning(f"Datetime stats failed for '{profile.column_name}': {e}")
        return None


# ===========================================================================
# Correlations
# ===========================================================================


def _compute_correlations(
    df: pd.DataFrame, numeric_profiles: list[ColumnProfile]
) -> list[CorrelationPair]:
    if len(numeric_profiles) < 2:
        return []

    cols = [p.column_name for p in numeric_profiles]
    numeric_df = df[cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric_df.corr(method="pearson")

    pairs = []
    seen = set()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            key = tuple(sorted([col_a, col_b]))
            if key in seen:
                continue
            seen.add(key)
            r = corr_matrix.loc[col_a, col_b]
            if pd.isna(r):
                continue
            abs_r = abs(r)
            strength = (
                "strong" if abs_r > 0.7 else ("moderate" if abs_r > 0.4 else "weak")
            )
            direction = "positive" if r > 0 else "negative"
            if abs_r > 0.4:  # only keep meaningful correlations
                pairs.append(
                    CorrelationPair(
                        col_a=col_a,
                        col_b=col_b,
                        pearson_r=round(float(r), 4),
                        strength=strength,
                        direction=direction,
                    )
                )

    # Sort by absolute correlation strength descending
    return sorted(pairs, key=lambda p: abs(p.pearson_r), reverse=True)


# ===========================================================================
# Summary bullets (consumed by insight_engine)
# ===========================================================================


def _build_summary_bullets(
    dataset: DatasetProfile,
    num_stats: list[NumericStats],
    cat_stats: list[CategoricalStats],
    correlations: list[CorrelationPair],
) -> list[str]:
    bullets = []

    # Dataset shape
    bullets.append(
        f"Dataset has {dataset.total_rows:,} rows and {dataset.total_columns} columns."
    )

    # Missing data
    high_null = [
        c for c in dataset.columns if 0.05 < c.null_rate <= HIGH_NULL_THRESHOLD
    ]
    if high_null:
        names = ", ".join(f"{c.column_name} ({c.null_rate:.0%})" for c in high_null)
        bullets.append(f"Columns with notable missing values: {names}.")

    # Numeric highlights
    for ns in num_stats:
        if ns.skewness > 1.5:
            bullets.append(
                f"'{ns.column}' is heavily right-skewed (skewness={ns.skewness}) — "
                f"most values cluster near {ns.median}, but outliers reach {ns.max}."
            )
        elif ns.skewness < -1.5:
            bullets.append(f"'{ns.column}' is left-skewed (skewness={ns.skewness}).")
        if ns.outlier_rate > 0.05:
            bullets.append(
                f"'{ns.column}' has {ns.outlier_count} outliers "
                f"({ns.outlier_rate:.1%} of values)."
            )

    # Categorical highlights
    for cs in cat_stats:
        if cs.top_values:
            top_label, top_count, top_pct = cs.top_values[0]
            if top_pct > 60:
                bullets.append(
                    f"'{cs.column}' is dominated by '{top_label}' ({top_pct}% of rows)."
                )

    # Correlation highlights
    for pair in correlations[:3]:  # top 3
        bullets.append(
            f"'{pair.col_a}' and '{pair.col_b}' show a {pair.strength} "
            f"{pair.direction} correlation (r={pair.pearson_r})."
        )

    return bullets


HIGH_NULL_THRESHOLD = 0.50


# ===========================================================================
# Public API
# ===========================================================================


def compute_stats(df: pd.DataFrame, profile: DatasetProfile) -> DatasetStats:
    """
    Compute all statistics for a dataset.

    Parameters
    ----------
    df      : the full DataFrame (from data_loader.get_dataframe)
    profile : DatasetProfile from schema_detector.detect_schema

    Returns
    -------
    DatasetStats — consumed by chart_generator and insight_engine
    """
    num_stats = []
    cat_stats = []
    dt_stats = []

    for col_profile in profile.columns:
        if col_profile.is_low_quality or not col_profile.chart_eligible:
            continue

        col = col_profile.column_name
        series = df[col]

        if col_profile.semantic_type in ("numeric", "boolean"):
            try:
                num_stats.append(_numeric_stats(series, col_profile))
            except Exception as e:
                log.warning(f"Numeric stats failed for '{col}': {e}")

        elif col_profile.semantic_type == "categorical":
            try:
                cat_stats.append(_categorical_stats(series, col_profile))
            except Exception as e:
                log.warning(f"Categorical stats failed for '{col}': {e}")

        elif col_profile.semantic_type == "datetime":
            result = _datetime_stats(series, col_profile)
            if result:
                dt_stats.append(result)

    numeric_profiles = [p for p in profile.columns if p.semantic_type == "numeric"]
    correlations = _compute_correlations(df, numeric_profiles)

    bullets = _build_summary_bullets(profile, num_stats, cat_stats, correlations)

    return DatasetStats(
        numeric=num_stats,
        categorical=cat_stats,
        datetime=dt_stats,
        correlations=correlations,
        summary_bullets=bullets,
    )
