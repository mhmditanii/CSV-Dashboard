from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .planner import compact_catalog_for_llm, deterministic_select, build_dashboard_plan
from .utils import safe_json

log = logging.getLogger(__name__)
try:
    import ollama
except Exception:  # pragma: no cover
    ollama = None

SYSTEM = """You are a senior business analyst building a dashboard for a non-technical user.
Python has already profiled the CSV and generated a catalog of SAFE, valid chart/table candidates.
You MUST choose only from the given candidate IDs. Do not invent columns, IDs, metrics, or numbers.

Your job:
1. Select 8-12 useful dashboard cards if available.
2. Prefer cards that answer business questions, not filler.
3. Prefer a mix: overview/quality if useful, outcome comparisons, segment comparisons, trends, distributions, maps, relationships.
4. Improve titles, axis labels, feature names, and takeaways in plain English.
5. For each selected card, explain WHY those features belong together.

Return ONLY valid JSON:
{
  "summary": "3-4 sentences. Name the most useful patterns. No jargon.",
  "selected_ids": ["candidate id", "candidate id"],
  "cards": [
    {
      "id": "same candidate id",
      "question": "short question a business user understands",
      "title": "clear title, max 10 words",
      "x_label": "human readable x-axis or table first column label",
      "y_label": "human readable y-axis or metric label",
      "why": "why these features are useful together, 1 sentence; do not state the result here",
      "takeaway": "what to notice from the numbers, 1 sentence; must be different from why"
    }
  ]
}

Rules:
- Never use the words correlation, skew, dataframe, variable, feature, percentile unless there is no simple alternative.
- Do not claim causation.
- WHY and TAKEAWAY must never be the same sentence.
- WHY = why this chart/table is useful. TAKEAWAY = the actual result to notice.
- Do not over-select duplicate cards that answer the same question.
- For IDs in selected_ids, copy them exactly.
- If a candidate is weak or sounds like filler, do not select it.
"""


def _json_from_text(text: str) -> dict[str, Any] | None:
    raw = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{"); end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            return None
    return None


def _candidate_context(candidates: list[dict[str, Any]], profile: dict[str, Any], filename: str) -> dict[str, Any]:
    cols = []
    for c in profile.get("compact_profiles", []):
        cols.append({
            "name": c.get("name"), "label": c.get("label"), "role": c.get("role"),
            "missing_rate": c.get("missing_rate"), "unique": c.get("unique"),
            "sample_values": c.get("sample_values", [])[:3],
            "stats": {k: v for k, v in (c.get("stats") or {}).items() if k in {"min", "max", "mean", "median", "top_values", "positive_value"}},
        })
    return safe_json({
        "file_name": filename,
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
        "sample_rows": profile.get("sample_rows", [])[:2],
        "column_profiles": cols,
        "candidate_cards": compact_catalog_for_llm(candidates),
    })


def _sameish(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    clean = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
    ca, cb = clean(a), clean(b)
    if ca == cb:
        return True
    aw, bw = set(ca.split()), set(cb.split())
    if not aw or not bw:
        return False
    return len(aw & bw) / max(len(aw | bw), 1) > 0.82


def _fallback_takeaway(card: dict[str, Any]) -> str:
    data = card.get("data") or []
    cols = card.get("columns") or (list(data[0].keys()) if data else [])
    family = card.get("family")
    kind = card.get("kind")
    if family == "overview":
        vals = {str(r.get("Metric")): r.get("Value") for r in data if isinstance(r, dict)}
        return f"This file has {vals.get('Rows', 'multiple')} rows and {vals.get('Columns', 'multiple')} columns."
    if family == "quality":
        return f"Review {len(data)} data-quality note(s) before relying on every result."
    if kind in {"bar_table", "line_table"} and data and len(cols) >= 2:
        first = data[0]
        left, right = cols[0], cols[-1]
        return f"{first.get(left, 'The top row')} stands out with {right.lower()} of {first.get(right, 'the highest value')}."
    if kind == "line" and data and len(cols) >= 2:
        metric = cols[-1]
        vals = [r.get(metric) for r in data if isinstance(r.get(metric), (int, float))]
        if vals:
            return f"{metric} ranges from {min(vals):,.0f} to {max(vals):,.0f} across the period shown."
    if kind == "histogram":
        return "The tallest bars show where most records are concentrated."
    if kind == "map":
        return "The densest areas show where records are most concentrated."
    return "Focus first on the highest and lowest values in this view."


def select_and_polish(df, profile: dict[str, Any], candidates: list[dict[str, Any]], filename: str) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    if ollama is None:
        issues.append("Ollama package is not installed; used deterministic selection.")
        return build_dashboard_plan(df, profile, [c["id"] for c in deterministic_select(candidates)]), issues

    model = os.getenv("OLLAMA_MODEL", "llama3")
    payload = _candidate_context(candidates, profile, filename)
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
            options={"temperature": 0.25, "num_predict": 2400},
        )
        result = _json_from_text(resp["message"]["content"])
        if not result:
            issues.append("Ollama returned invalid JSON; used deterministic selection.")
            return build_dashboard_plan(df, profile, [c["id"] for c in deterministic_select(candidates)]), issues

        valid_ids = {c["id"] for c in candidates}
        selected_ids = [x for x in result.get("selected_ids", []) if x in valid_ids]
        if not selected_ids:
            selected_ids = [c["id"] for c in deterministic_select(candidates)]
            issues.append("Ollama selected no valid cards; used deterministic selection.")

        plan = build_dashboard_plan(df, profile, selected_ids)
        by_id = {c["id"]: c for c in plan.get("cards", [])}
        for pc in result.get("cards", []):
            cid = pc.get("id")
            if cid not in by_id:
                continue
            card = by_id[cid]
            original_takeaway = card.get("takeaway", "")
            for field, max_len in {"question": 140, "title": 90, "x_label": 50, "y_label": 50, "why": 220, "takeaway": 240}.items():
                val = pc.get(field)
                if isinstance(val, str) and 2 <= len(val.strip()) <= max_len:
                    card[field] = val.strip()
            if _sameish(card.get("why"), card.get("takeaway")):
                if isinstance(original_takeaway, str) and original_takeaway and not _sameish(card.get("why"), original_takeaway):
                    card["takeaway"] = original_takeaway
                else:
                    card["takeaway"] = _fallback_takeaway(card)
        if isinstance(result.get("summary"), str) and 20 <= len(result["summary"]) <= 700:
            plan["summary"] = result["summary"].strip()
        return safe_json(plan), issues
    except Exception as e:
        log.warning("Ollama selection failed: %s", e)
        issues.append(f"Ollama failed ({type(e).__name__}); used deterministic selection.")
        return build_dashboard_plan(df, profile, [c["id"] for c in deterministic_select(candidates)]), issues
