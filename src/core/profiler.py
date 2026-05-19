from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    DATE_WORDS, GEO_LAT_WORDS, GEO_LON_WORDS, OUTCOME_WORDS,
    has_any, human_label, looks_like_id_name, safe_json, is_bad_axis_name
)


@dataclass
class ColumnProfile:
    name: str
    label: str
    role: str
    dtype: str
    missing_rate: float
    unique: int
    unique_rate: float
    sample_values: list[Any]
    stats: dict[str, Any]
    is_candidate: bool = True
    reason_excluded: str | None = None


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _date_ratio(s: pd.Series, name: str) -> float:
    sample = s.dropna().astype(str).head(300)
    if sample.empty:
        return 0.0
    if not has_any(name, DATE_WORDS):
        mask = sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True, na=False)
        if mask.sum() < max(5, len(sample) * 0.55):
            return 0.0
        sample = sample[mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def _numeric_sequence(num: pd.Series, rows: int) -> bool:
    vals = np.sort(num.dropna().unique())
    if rows <= 80 or len(vals) <= 8 or len(vals) / max(rows, 1) < 0.82:
        return False
    diffs = np.diff(vals)
    return bool(len(diffs) and np.nanmedian(diffs) == 1 and np.nanmean(diffs) < 2.5)


def _positive_value(values: list[Any], name: str) -> Any | None:
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return None
    norm = {str(v).strip().lower(): v for v in vals}
    for p in ["1", "true", "yes", "y", "survived", "converted", "success", "approved", "active", "retained", "paid", "won", "passed"]:
        if p in norm:
            return norm[p]
    if has_any(name, OUTCOME_WORDS):
        try:
            return sorted(vals, key=lambda x: float(x))[-1]
        except Exception:
            return sorted(vals, key=lambda x: str(x))[-1]
    return None


def profile_dataframe(df: pd.DataFrame, max_columns_for_llm: int = 90) -> dict[str, Any]:
    rows = len(df)
    profiles: list[ColumnProfile] = []

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        unique = int(non_null.nunique(dropna=True))
        unique_rate = unique / max(rows, 1)
        missing_rate = float(s.isna().mean())
        sample_values = safe_json(non_null.head(5).tolist())
        dtype = str(s.dtype)
        label = human_label(col)
        stats: dict[str, Any] = {}
        role = "text"
        is_candidate = True
        reason = None

        num = _numeric(s)
        num_non_null = num.dropna().to_numpy(dtype=float)
        num_ratio = float(np.isfinite(num_non_null).sum() / max(len(non_null), 1))
        date_ratio = _date_ratio(s, col)

        if has_any(col, GEO_LAT_WORDS):
            role = "latitude"
        elif has_any(col, GEO_LON_WORDS):
            role = "longitude"
        elif looks_like_id_name(col):
            role = "identifier"; is_candidate = False; reason = "identifier-like column"
        elif date_ratio >= 0.75:
            role = "datetime"
        elif unique <= 2 and unique > 0:
            pos = _positive_value(non_null.unique().tolist(), col)
            role = "outcome" if pos is not None else "categorical"
            stats["positive_value"] = safe_json(pos)
        elif num_ratio >= 0.85:
            if _numeric_sequence(num, rows):
                role = "identifier"; is_candidate = False; reason = "numeric sequence / identifier-like"
            elif unique <= min(20, max(3, rows * 0.04)) and not has_any(col, {"age", "price", "fare", "cost", "salary", "amount", "revenue", "sales", "score", "rating", "height", "weight", "income"}):
                role = "categorical"
                stats["numeric_codes"] = True
            else:
                role = "measure"
                arr = num_non_null[np.isfinite(num_non_null)]
                if len(arr):
                    q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
                    std = float(np.std(arr)) if len(arr) > 1 else 0.0
                    mean = float(np.mean(arr))
                    skew = float(np.mean(((arr - mean) / std) ** 3)) if std > 0 and len(arr) > 2 else 0.0
                    stats.update(min=float(np.min(arr)), max=float(np.max(arr)), mean=mean, median=float(q[2]), std=std, skew=skew, p05=float(q[0]), p25=float(q[1]), p75=float(q[3]), p95=float(q[4]))
        elif unique <= min(80, max(12, rows * 0.25)):
            role = "categorical"
        else:
            avg_len = float(non_null.astype(str).str.len().mean()) if len(non_null) else 0.0
            role = "text"
            is_candidate = False
            reason = "free text or mostly unique" if unique_rate > 0.5 or avg_len > 50 else "not useful for charts"

        if role in {"categorical", "outcome"}:
            top = non_null.astype(str).value_counts().head(15)
            stats["top_values"] = [{"value": idx, "rows": int(val), "share": float(val / max(len(non_null), 1))} for idx, val in top.items()]
        if role == "datetime":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                d = pd.to_datetime(s, errors="coerce")
            if d.notna().any():
                stats["min"] = d.min(); stats["max"] = d.max()
        if missing_rate > 0.85:
            is_candidate = False; reason = "mostly missing"
        if is_bad_axis_name(col) and role not in {"latitude", "longitude"}:
            is_candidate = False
            if role not in {"identifier", "text"}:
                reason = "name/code/text field; not used as chart axis"

        profiles.append(ColumnProfile(str(col), label, role, dtype, round(missing_rate, 4), unique, round(unique_rate, 4), sample_values, safe_json(stats), is_candidate, reason))

    profile_dicts = [asdict(p) for p in profiles]
    compact_cols = [c for c in profile_dicts if c["is_candidate"] or c["role"] in {"outcome", "datetime", "latitude", "longitude"}]
    compact_cols = compact_cols[:max_columns_for_llm]

    return safe_json({
        "rows": rows,
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "column_profiles": profile_dicts,
        "compact_profiles": compact_cols,
        "sample_rows": df.head(2).to_dict("records"),
    })
