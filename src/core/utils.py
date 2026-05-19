from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

ID_WORDS = {
    "id", "ids", "uuid", "guid", "key", "token", "hash", "slug", "url", "uri",
    "email", "phone", "ssn", "passport", "ip", "sku", "code", "zip", "zipcode", "postal",
    "account", "acct", "record", "row", "index"
}
BAD_AXIS_WORDS = ID_WORDS | {"name", "first", "last", "description", "comment", "comments", "notes", "text", "address"}
MONEY_WORDS = {"price", "fare", "cost", "amount", "revenue", "sales", "income", "salary", "wage", "spend", "payment", "profit", "loss", "value", "budget"}
RATING_WORDS = {"rate", "rating", "score", "satisfaction", "nps", "grade", "percent", "percentage", "ratio", "share"}
COUNT_WORDS = {"quantity", "qty", "units", "volume", "orders", "visits", "clicks", "views", "count", "number", "sessions"}
DATE_WORDS = {"date", "time", "year", "month", "day", "created", "updated", "timestamp"}
OUTCOME_WORDS = {"survived", "churn", "converted", "conversion", "retained", "cancelled", "canceled", "default", "fraud", "approved", "success", "status", "outcome", "target", "label", "purchased", "bought", "active", "paid", "won", "lost", "passed", "failed"}
GEO_LAT_WORDS = {"lat", "latitude"}
GEO_LON_WORDS = {"lon", "lng", "long", "longitude"}


def tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def has_any(name: str, words: set[str]) -> bool:
    low = str(name).lower()
    ts = tokens(low)
    return bool(ts & words) or any(w in low for w in words if len(w) > 4)


def looks_like_id_name(name: str) -> bool:
    low = str(name).lower().strip()
    ts = tokens(low)
    if ts & ID_WORDS:
        return True
    return low.endswith("_id") or low.endswith("id") or low.startswith("id_")


def is_bad_axis_name(name: str) -> bool:
    return looks_like_id_name(name) or has_any(name, BAD_AXIS_WORDS)


def human_label(name: str) -> str:
    text = str(name).strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    mapping = {
        "pclass": "Passenger Class", "sibsp": "Siblings/Spouses", "parch": "Parents/Children",
        "lat": "Latitude", "lng": "Longitude", "lon": "Longitude", "id": "ID", "url": "URL",
        "nps": "NPS", "roi": "ROI", "sku": "SKU",
    }
    low = text.lower().replace(" ", "")
    if low in mapping:
        return mapping[low]
    words = []
    for w in text.split():
        if w.lower() in {"id", "url", "api", "ip", "sku", "nps", "roi"}:
            words.append(w.upper())
        elif len(w) <= 2 and w.isupper():
            words.append(w)
        else:
            words.append(w.capitalize())
    return " ".join(words)[:64]


def safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [safe_json(v) for v in list(value)]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def format_number(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return ""
    if abs(v) >= 1_000_000_000:
        return f"{v/1_000_000_000:.1f}B"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:,.1f}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


def compact_text(v: Any, max_len: int = 42) -> str:
    s = str(v)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def choose_metric_agg(name: str) -> tuple[str, str]:
    low = str(name).lower()
    if has_any(low, {"revenue", "sales", "amount", "income", "profit", "loss", "spend", "payment", "quantity", "qty", "units", "orders", "clicks", "views", "volume"}):
        return "sum", "Total"
    if has_any(low, {"rating", "score", "rate", "percent", "percentage", "ratio", "age", "nps"}):
        return "mean", "Average"
    if has_any(low, {"price", "fare", "cost", "salary", "wage", "value"}):
        return "median", "Typical"
    return "mean", "Average"
