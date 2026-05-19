from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from .utils import choose_metric_agg, compact_text, format_number, pct, safe_json

MIN_GROUP_SIZE = 15
MAX_CANDIDATES_FOR_LLM = 45
MAX_SELECTED_CARDS = 12


def _pmap(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in profile["column_profiles"]}


def _segments(profile: dict[str, Any]) -> list[str]:
    out = []
    rows = profile.get("rows", 0)
    for c in profile["column_profiles"]:
        if c.get("is_candidate", True) and c["role"] == "categorical" and 2 <= c["unique"] <= min(60, max(8, rows // 10)):
            out.append(c["name"])
    return out


def _measures(profile: dict[str, Any]) -> list[str]:
    return [c["name"] for c in profile["column_profiles"] if c.get("is_candidate", True) and c["role"] == "measure"]


def _outcomes(profile: dict[str, Any]) -> list[str]:
    return [c["name"] for c in profile["column_profiles"] if c.get("is_candidate", True) and c["role"] == "outcome" and c.get("stats", {}).get("positive_value") is not None]


def _dates(profile: dict[str, Any]) -> list[str]:
    return [c["name"] for c in profile["column_profiles"] if c.get("is_candidate", True) and c["role"] == "datetime"]


def _geo(profile: dict[str, Any]) -> tuple[str | None, str | None]:
    lat = next((c["name"] for c in profile["column_profiles"] if c["role"] == "latitude"), None)
    lon = next((c["name"] for c in profile["column_profiles"] if c["role"] == "longitude"), None)
    return lat, lon


def _np_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _valid_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def _min_group_size(rows: int) -> int:
    return max(MIN_GROUP_SIZE, min(80, int(rows * 0.01)))


def _auto_takeaway(kind: str, family: str, data: list[dict[str, Any]], columns: list[str] | None) -> str:
    """Create a result-focused takeaway when Ollama does not provide one.

    "Why" explains why the view exists. "Takeaway" should say what to notice.
    This fallback keeps those two fields from becoming identical.
    """
    cols = columns or (list(data[0].keys()) if data else [])
    if family == "overview":
        vals = {str(r.get("Metric")): r.get("Value") for r in data if isinstance(r, dict)}
        rows, col_count = vals.get("Rows"), vals.get("Columns")
        return f"This file has {rows or 'multiple'} rows and {col_count or 'multiple'} columns to explore."
    if family == "quality":
        return f"There are {len(data)} data-quality note(s) to review before trusting every result."
    if kind in {"bar_table", "line_table"} and data and len(cols) >= 2:
        first = data[0]
        label_col = cols[0]
        metric_col = cols[-1]
        label = first.get(label_col, "the leading group")
        metric = first.get(metric_col, "the highest value")
        return f"{label} is the first item to inspect, with {metric_col.lower()} of {metric}."
    if kind == "line" and data and len(cols) >= 2:
        metric_col = cols[-1]
        values = [r.get(metric_col) for r in data if isinstance(r.get(metric_col), (int, float))]
        if values:
            return f"{metric_col} ranges from {format_number(min(values))} to {format_number(max(values))} over the period shown."
        return "Look for periods where the line rises, falls, or clusters unusually."
    if kind == "histogram":
        return "Most records are concentrated in the tallest bars; long tails usually deserve a closer look."
    if kind == "map":
        return "The densest areas on the map show where records are most concentrated."
    return "Start with the highest and lowest values; they usually explain the main pattern."


def _candidate_base(cid: str, kind: str, family: str, score: float, question: str, title: str, why: str, data: list[dict[str, Any]], columns: list[str] | None = None, **extra) -> dict[str, Any]:
    cols = columns or (list(data[0].keys()) if data else [])
    takeaway = extra.pop("takeaway", None) or _auto_takeaway(kind, family, data, cols)
    card = {
        "id": cid,
        "kind": kind,
        "family": family,
        "score": round(float(score), 3),
        "question": question,
        "title": title,
        "why": why,
        "takeaway": takeaway,
        "data": data,
        "columns": cols,
    }
    card.update(extra)
    return safe_json(card)


def dataset_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    roles: dict[str, int] = {}
    for c in profile["column_profiles"]:
        roles[c["role"]] = roles.get(c["role"], 0) + 1
    rows = [
        {"Metric": "Rows", "Value": f"{profile['rows']:,}"},
        {"Metric": "Columns", "Value": f"{profile['columns']:,}"},
        {"Metric": "Measures", "Value": roles.get("measure", 0)},
        {"Metric": "Segments", "Value": roles.get("categorical", 0)},
        {"Metric": "Outcomes", "Value": roles.get("outcome", 0)},
        {"Metric": "Date Fields", "Value": roles.get("datetime", 0)},
        {"Metric": "Missing Cells", "Value": f"{profile['missing_cells']:,}"},
        {"Metric": "Duplicate Rows", "Value": f"{profile['duplicate_rows']:,}"},
    ]
    return _candidate_base(
        "snapshot", "table", "overview", 100,
        "What is in this file?", "Dataset Overview",
        "This gives a quick baseline before interpreting the rest of the dashboard.",
        rows, ["Metric", "Value"],
        takeaway=f"The dataset has {profile['rows']:,} rows, {profile['columns']:,} columns, {profile['missing_cells']:,} missing cells, and {profile['duplicate_rows']:,} duplicate rows."
    )


def data_quality_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for c in profile["column_profiles"]:
        if c["missing_rate"] >= 0.1:
            rows.append({"Column": c["label"], "Issue": f"{c['missing_rate']:.0%} missing", "Impact": "Use with caution"})
    if profile.get("duplicate_rows", 0) > 0:
        rows.append({"Column": "All rows", "Issue": f"{profile['duplicate_rows']:,} duplicate rows", "Impact": "May inflate totals"})
    if not rows:
        return []
    return [_candidate_base("quality", "table", "quality", 96, "Any data issues to know?", "Data Quality Notes", "These issues can affect how much confidence to place in the results.", rows[:10], ["Column", "Issue", "Impact"])]


def top_category_candidates(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    p = _pmap(profile); rows_total = len(df); cards = []
    for seg in _segments(profile):
        vc = df[seg].dropna().astype(str).value_counts().head(12)
        if len(vc) < 2:
            continue
        shares = vc / max(df[seg].notna().sum(), 1)
        concentration = float(shares.iloc[:3].sum())
        if concentration < 0.55 and shares.iloc[0] < 0.32:
            # Not interesting as a standalone composition chart.
            continue
        label = p[seg]["label"]
        data = [{label: compact_text(k), "Rows": int(v), "Share": pct(v / rows_total)} for k, v in vc.items()]
        why = f"The largest {label.lower()} group is {data[0][label]}, representing {data[0]['Share']} of rows."
        cards.append(_candidate_base(f"top_category__{seg}", "bar_table", "composition", 54 + concentration * 30, f"Which {label.lower()} groups dominate?", f"Largest {label} Groups", why, data, [label, "Rows", "Share"], x_label=label, y_label="Rows", plot_data=[{"Group": r[label], "Value": r["Rows"]} for r in data]))
    return cards


def outcome_by_segment_candidates(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    p = _pmap(profile); cards = []; min_n = _min_group_size(len(df))
    for outcome in _outcomes(profile):
        positive = p[outcome]["stats"].get("positive_value")
        out_label = p[outcome]["label"]
        for seg in _segments(profile):
            if seg == outcome:
                continue
            tmp = df[[outcome, seg]].dropna().copy()
            if len(tmp) < min_n * 2:
                continue
            pos = tmp[outcome].astype(str).str.lower().to_numpy() == str(positive).lower()
            groups = tmp[seg].astype(str).to_numpy()
            rows = []
            for g in pd.Series(groups).value_counts().index[:30]:
                m = groups == g
                if int(m.sum()) >= min_n:
                    rows.append((str(g), int(m.sum()), float(pos[m].mean())))
            if len(rows) < 2:
                continue
            rows = sorted(rows, key=lambda x: x[2], reverse=True)
            spread = rows[0][2] - rows[-1][2]
            if spread < 0.08:
                continue
            seg_label = p[seg]["label"]
            data = [{seg_label: compact_text(g), "Rows": n, f"{out_label} Rate": pct(rate)} for g, n, rate in rows[:12]]
            why = f"{data[0][seg_label]} has the highest {out_label.lower()} rate at {data[0][f'{out_label} Rate']} among sufficiently large groups."
            cards.append(_candidate_base(f"outcome_segment__{outcome}__{seg}", "bar_table", "outcome", 84 + min(spread * 60, 16), f"Which {seg_label.lower()} groups differ most?", f"{out_label} Rate by {seg_label}", why, data, [seg_label, "Rows", f"{out_label} Rate"], x_label=seg_label, y_label=f"{out_label} Rate", plot_data=[{"Group": r[seg_label], "Value": float(str(r[f'{out_label} Rate']).rstrip('%'))} for r in data], evidence={"outcome": outcome, "segment": seg, "spread": round(spread, 3)}))
    return cards


def outcome_by_measure_candidates(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    p = _pmap(profile); cards = []; min_n = _min_group_size(len(df))
    for outcome in _outcomes(profile):
        positive = p[outcome]["stats"].get("positive_value")
        out_label = p[outcome]["label"]
        for measure in _measures(profile):
            arr = _np_numeric(df, measure)
            valid = np.isfinite(arr) & df[outcome].notna().to_numpy()
            if valid.sum() < min_n * 4 or len(np.unique(arr[valid])) < 8:
                continue
            pos = df.loc[valid, outcome].astype(str).str.lower().to_numpy() == str(positive).lower()
            vals = arr[valid]
            try:
                bins = pd.qcut(vals, q=5, duplicates="drop")
            except Exception:
                continue
            tmp = pd.DataFrame({"bin": bins, "value": vals, "positive": pos})
            grouped = tmp.groupby("bin", observed=True).agg(rows=("positive", "size"), rate=("positive", "mean"), mid=("value", "median")).reset_index()
            grouped = grouped[grouped["rows"] >= min_n]
            if len(grouped) < 3:
                continue
            spread = float(grouped["rate"].max() - grouped["rate"].min())
            if spread < 0.08:
                continue
            meas_label = p[measure]["label"]
            data = [{f"{meas_label} Range": str(r["bin"]), "Rows": int(r["rows"]), f"{out_label} Rate": pct(r["rate"])} for _, r in grouped.iterrows()]
            best = max(data, key=lambda r: float(str(r[f"{out_label} Rate"]).rstrip("%")))
            why = f"The highest {out_label.lower()} rate appears in {best[f'{meas_label} Range']}."
            cards.append(_candidate_base(f"outcome_measure__{outcome}__{measure}", "line_table", "outcome", 80 + min(spread * 60, 14), f"Does {meas_label.lower()} change {out_label.lower()}?", f"{out_label} Rate by {meas_label} Range", why, data, [f"{meas_label} Range", "Rows", f"{out_label} Rate"], x_label=meas_label, y_label=f"{out_label} Rate", plot_data=[{"X": float(r["mid"]), "Y": float(r["rate"] * 100)} for _, r in grouped.iterrows()], evidence={"outcome": outcome, "measure": measure, "spread": round(spread, 3)}))
    return cards


def metric_by_segment_candidates(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    p = _pmap(profile); cards = []; min_n = _min_group_size(len(df))
    for measure in _measures(profile):
        agg, agg_label = choose_metric_agg(measure)
        vals = _np_numeric(df, measure)
        for seg in _segments(profile):
            tmp = pd.DataFrame({"group": df[seg].astype(str), "value": vals}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(tmp) < min_n * 2:
                continue
            if agg == "sum":
                grouped = tmp.groupby("group").agg(rows=("value", "size"), value=("value", "sum")).reset_index()
            elif agg == "median":
                grouped = tmp.groupby("group").agg(rows=("value", "size"), value=("value", "median")).reset_index()
            else:
                grouped = tmp.groupby("group").agg(rows=("value", "size"), value=("value", "mean")).reset_index()
            grouped = grouped[grouped["rows"] >= min_n].sort_values("value", ascending=False)
            if len(grouped) < 2:
                continue
            top, bottom = float(grouped.iloc[0]["value"]), float(grouped.iloc[-1]["value"])
            baseline = abs(float(np.nanmedian(grouped["value"]))) or 1.0
            rel_spread = (top - bottom) / baseline
            if rel_spread < 0.22:
                continue
            seg_label, meas_label = p[seg]["label"], p[measure]["label"]
            value_label = f"{agg_label} {meas_label}"
            data = [{seg_label: compact_text(r["group"]), "Rows": int(r["rows"]), value_label: format_number(r["value"])} for _, r in grouped.head(12).iterrows()]
            why = f"{data[0][seg_label]} stands out with the highest {value_label.lower()} ({data[0][value_label]})."
            cards.append(_candidate_base(f"metric_segment__{measure}__{seg}", "bar_table", "segment", 68 + min(rel_spread * 18, 18), f"Which {seg_label.lower()} groups stand out?", f"{value_label} by {seg_label}", why, data, [seg_label, "Rows", value_label], x_label=seg_label, y_label=value_label, plot_data=[{"Group": r[seg_label], "Value": float(str(r[value_label]).replace('K','e3').replace('M','e6').replace(',','')) if False else i} for i, r in enumerate(data)], raw_plot_data=[{"Group": compact_text(r["group"]), "Value": float(r["value"])} for _, r in grouped.head(12).iterrows()], evidence={"measure": measure, "segment": seg, "aggregation": agg, "relative_spread": round(rel_spread, 2)}))
    return cards


def distribution_candidates(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    p = _pmap(profile); cards = []
    for measure in _measures(profile):
        arr = _np_numeric(df, measure); arr = arr[np.isfinite(arr)]
        if len(arr) < 60:
            continue
        q = np.quantile(arr, [0.25, 0.5, 0.75, 0.95, 0.99])
        iqr = q[2] - q[0]
        if iqr <= 0:
            continue
        outlier_share = float((arr > q[2] + 1.5 * iqr).mean())
        skew = abs(float(np.mean(((arr - arr.mean()) / arr.std()) ** 3))) if arr.std() > 0 else 0.0
        if outlier_share < 0.03 and skew < 1.1:
            continue
        label = p[measure]["label"]
        capped = arr[arr <= q[4]] if len(arr) > 100 else arr
        rng = np.random.default_rng(42)
        sample = rng.choice(capped, size=min(5000, len(capped)), replace=False) if len(capped) else capped
        why = f"Most {label.lower()} values are between {format_number(q[0])} and {format_number(q[2])}, with some high outliers."
        cards.append(_candidate_base(f"distribution__{measure}", "histogram", "distribution", 57 + min(skew * 5 + outlier_share * 80, 20), f"What is a normal range for {label.lower()}?", f"Spread of {label}", why, [{"value": float(v)} for v in sample], [label], x_label=label, y_label="Rows", evidence={"measure": measure, "skew": round(skew, 2), "outlier_share": round(outlier_share, 3)}))
    return cards


def time_candidates(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    p = _pmap(profile); cards = []
    for date_col in _dates(profile)[:3]:
        d = pd.to_datetime(df[date_col], errors="coerce")
        valid_dates = d.dropna()
        if len(valid_dates) < 50:
            continue
        span_days = max(1, int((valid_dates.max() - valid_dates.min()).days))
        freq = "M" if span_days > 120 else "D"
        label = p[date_col]["label"]
        trend = pd.DataFrame({"date": d}).dropna().set_index("date").assign(Rows=1).resample(freq)["Rows"].sum().reset_index()
        if len(trend) >= 3:
            data = [{"Date": x.date().isoformat() if hasattr(x, "date") else str(x), "Rows": int(y)} for x, y in zip(trend["date"], trend["Rows"])]
            why = "This shows whether activity is rising, falling, or concentrated in specific periods."
            cards.append(_candidate_base(f"time_count__{date_col}", "line", "time", 64, f"How does row volume change over {label.lower()}?", f"Rows Over {label}", why, data, ["Date", "Rows"], x_label=label, y_label="Rows"))
        for measure in _measures(profile)[:5]:
            vals = _np_numeric(df, measure)
            tmp = pd.DataFrame({"date": d, "value": vals}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(tmp) < 60:
                continue
            agg, agg_label = choose_metric_agg(measure)
            trend = tmp.set_index("date").resample(freq)["value"].agg(agg).dropna().reset_index()
            if len(trend) < 3:
                continue
            meas_label = p[measure]["label"]
            data = [{"Date": x.date().isoformat() if hasattr(x, "date") else str(x), f"{agg_label} {meas_label}": float(y)} for x, y in zip(trend["date"], trend["value"])]
            why = f"This tracks how {meas_label.lower()} changes over time."
            cards.append(_candidate_base(f"time_metric__{date_col}__{measure}", "line", "time", 61, f"How does {meas_label.lower()} change over time?", f"{agg_label} {meas_label} Over {label}", why, data, ["Date", f"{agg_label} {meas_label}"], x_label=label, y_label=f"{agg_label} {meas_label}"))
    return cards


def relationship_candidates(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    p = _pmap(profile); cards = []; measures = _measures(profile)[:14]
    for a, b in itertools.combinations(measures, 2):
        x = _np_numeric(df, a); y = _np_numeric(df, b); m = _valid(x, y)
        if m.sum() < 100:
            continue
        xv, yv = x[m], y[m]
        if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
            continue
        corr = float(np.corrcoef(xv, yv)[0, 1])
        if not np.isfinite(corr) or abs(corr) < 0.55:
            continue
        try:
            bins = pd.qcut(xv, q=6, duplicates="drop")
        except Exception:
            continue
        tmp = pd.DataFrame({"bin": bins, "x": xv, "y": yv})
        grouped = tmp.groupby("bin", observed=True).agg(rows=("y", "size"), x_mid=("x", "median"), y_mid=("y", "median")).reset_index()
        if len(grouped) < 3:
            continue
        a_label, b_label = p[a]["label"], p[b]["label"]
        data = [{f"{a_label} Range": str(r["bin"]), "Rows": int(r["rows"]), f"Typical {b_label}": format_number(r["y_mid"])} for _, r in grouped.iterrows()]
        direction = "tend to rise together" if corr > 0 else "tend to move in opposite directions"
        why = f"As {a_label.lower()} changes, {b_label.lower()} values {direction}. Treat this as a lead for investigation, not proof of cause."
        cards.append(_candidate_base(f"relationship__{a}__{b}", "line_table", "relationship", 59 + min(abs(corr) * 22, 18), f"Do {a_label.lower()} and {b_label.lower()} move together?", f"{b_label} by {a_label} Range", why, data, [f"{a_label} Range", "Rows", f"Typical {b_label}"], x_label=a_label, y_label=f"Typical {b_label}", plot_data=[{"X": float(r["x_mid"]), "Y": float(r["y_mid"])} for _, r in grouped.iterrows()], evidence={"x": a, "y": b, "relationship_strength": round(abs(corr), 3), "direction": "positive" if corr > 0 else "negative"}))
    return cards


def _valid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


def map_candidates(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    lat, lon = _geo(profile)
    if not lat or not lon:
        return []
    la = _np_numeric(df, lat); lo = _np_numeric(df, lon); m = _valid(la, lo)
    if m.sum() < 10:
        return []
    idx = np.where(m)[0]
    rng = np.random.default_rng(42)
    idx = rng.choice(idx, size=min(5000, len(idx)), replace=False)
    data = [{"lat": float(la[i]), "lon": float(lo[i])} for i in idx]
    return [_candidate_base("map", "map", "geo", 78, "Where are records located?", "Geographic Distribution", "The map shows where records are concentrated. Points are sampled for speed on large files.", data, ["lat", "lon"], x_label="Longitude", y_label="Latitude")]


def build_candidate_catalog(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [dataset_snapshot(profile)]
    candidates += data_quality_candidates(profile)
    candidates += map_candidates(df, profile)
    candidates += outcome_by_segment_candidates(df, profile)
    candidates += outcome_by_measure_candidates(df, profile)
    candidates += metric_by_segment_candidates(df, profile)
    candidates += time_candidates(df, profile)
    candidates += relationship_candidates(df, profile)
    candidates += distribution_candidates(df, profile)
    candidates += top_category_candidates(df, profile)
    # Deduplicate by id and sort.
    unique = {c["id"]: c for c in candidates}
    return sorted(unique.values(), key=lambda c: c.get("score", 0), reverse=True)


def compact_catalog_for_llm(candidates: list[dict[str, Any]], limit: int = MAX_CANDIDATES_FOR_LLM) -> list[dict[str, Any]]:
    compact = []
    for c in candidates[:limit]:
        compact.append({
            "id": c["id"],
            "kind": c["kind"],
            "family": c.get("family"),
            "score": c.get("score"),
            "question": c.get("question"),
            "title": c.get("title"),
            "why": c.get("why"),
            "x_label": c.get("x_label"),
            "y_label": c.get("y_label"),
            "columns": c.get("columns"),
            "preview_rows": c.get("data", [])[:3],
            "evidence": c.get("evidence", {}),
        })
    return safe_json(compact)


def deterministic_select(candidates: list[dict[str, Any]], max_cards: int = MAX_SELECTED_CARDS) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    limits = {"overview": 1, "quality": 1, "geo": 1, "outcome": 5, "segment": 4, "time": 3, "relationship": 2, "distribution": 2, "composition": 1}
    for c in sorted(candidates, key=lambda x: x.get("score", 0), reverse=True):
        fam = c.get("family", "other")
        if family_counts.get(fam, 0) >= limits.get(fam, 2):
            continue
        selected.append(c)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        if len(selected) >= max_cards:
            break
    return selected


def build_dashboard_plan(df: pd.DataFrame, profile: dict[str, Any], selected_ids: list[str] | None = None) -> dict[str, Any]:
    candidates = build_candidate_catalog(df, profile)
    by_id = {c["id"]: c for c in candidates}
    if selected_ids:
        cards = [by_id[i] for i in selected_ids if i in by_id]
        if len(cards) < 4:
            # Fill with deterministic picks if the model selected too few valid cards.
            for c in deterministic_select(candidates):
                if c["id"] not in {x["id"] for x in cards}:
                    cards.append(c)
                if len(cards) >= min(MAX_SELECTED_CARDS, max(8, len(selected_ids))):
                    break
    else:
        cards = deterministic_select(candidates)
    non_overview = [c for c in cards if c.get("family") not in {"overview", "quality"}]
    summary = "Insights are generated from the strongest patterns found in the file. Review the first few cards for the clearest opportunities or risks."
    if non_overview:
        summary = f"Found {len(non_overview)} useful insight views. Start with {non_overview[0]['title']} because it has the strongest evidence in this file."
    return safe_json({"name": "Insights", "summary": summary, "cards": cards, "profile": profile, "candidate_count": len(candidates)})
