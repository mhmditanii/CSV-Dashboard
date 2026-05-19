from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_uploaded_csv(contents: str, filename: str) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if not contents or "," not in contents:
        raise ValueError("No uploaded file content was received.")
    if not filename.lower().endswith(".csv"):
        raise ValueError("Please upload a CSV file.")

    _, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    digest = hashlib.md5(raw).hexdigest()[:16]
    csv_path = CACHE_DIR / f"{digest}_{Path(filename).name}"
    csv_path.write_bytes(raw)

    read_errors = []
    df = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
            break
        except Exception as e:
            read_errors.append(f"{encoding}: {e}")
    if df is None:
        raise ValueError("Could not read CSV. " + " | ".join(read_errors[:2]))

    if df.empty:
        raise ValueError("The CSV was read successfully, but it has no rows.")

    df.columns = [str(c).strip() or f"Column {i+1}" for i, c in enumerate(df.columns)]
    parquet_path = CACHE_DIR / f"{digest}_{Path(filename).stem}.parquet"
    df.to_parquet(parquet_path, index=False)
    meta = {"filename": filename, "rows": len(df), "columns": len(df.columns), "cache": str(parquet_path)}
    return df, str(parquet_path), meta


def read_cached(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
