"""
insight_engine.py
-----------------
Hybrid insight generation:
  1. Deterministic — extract key facts from DatasetStats (always runs, free, fast)
  2. Ollama narrative — pass those facts to a local LLM to write plain-English insights

If Ollama is unavailable or fails, the deterministic insights are returned alone.
The UI should always have something to show.

Usage:
    from insight_engine import generate_insights
    insights = generate_insights(profile, stats, use_ollama=True)
    # insights: {"bullets": [...], "narrative": "...", "source": "llm" | "rules"}
"""

import json
import logging
import re
from typing import Optional

from schema_detector import DatasetProfile
from stats_engine import DatasetStats

log = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    log.warning("ollama package not installed — narrative insights disabled")

DEFAULT_MODEL = "llama3"


# ===========================================================================
# Step 1 — Deterministic insight extraction
# ===========================================================================

def _extract_deterministic_insights(
    profile: DatasetProfile,
    stats: DatasetStats,
) -> list[str]:
    """
    Build a list of plain-English insight bullets from computed stats.
    These are always generated — no LLM required.
    """
    insights = []

    # --- Dataset shape ---
    insights.append(
        f"Your dataset contains {profile.total_rows:,} rows and "
        f"{profile.total_columns} columns, of which "
        f"{len(profile.chart_eligible_columns())} are suitable for charting."
    )

    # --- Missing data ---
    missing_cols = [c for c in profile.columns if 0.05 < c.null_rate <= 0.5]
    if missing_cols:
        for col in missing_cols[:3]:
            insights.append(
                f"'{col.column_name}' has {col.null_rate:.0%} missing values — "
                "consider whether this affects your analysis."
            )

    excluded = [c for c in profile.columns if c.null_rate > 0.5]
    if excluded:
        names = ", ".join(f"'{c.column_name}'" for c in excluded)
        insights.append(
            f"{names} {'was' if len(excluded) == 1 else 'were'} excluded from charts "
            "due to more than 50% missing values."
        )

    # --- Numeric distribution highlights ---
    for ns in stats.numeric:
        if ns.skewness > 1.5:
            insights.append(
                f"'{ns.column}' is heavily skewed to the right — most values sit around "
                f"{ns.median:,.2f}, but some extreme values reach up to {ns.max:,.2f}."
            )
        elif ns.skewness < -1.5:
            insights.append(
                f"'{ns.column}' is skewed to the left — the bulk of values are near "
                f"{ns.median:,.2f} with a lower tail down to {ns.min:,.2f}."
            )

        if ns.outlier_rate > 0.03:
            insights.append(
                f"'{ns.column}' contains {ns.outlier_count} outliers "
                f"({ns.outlier_rate:.1%} of rows) — these may be worth investigating."
            )

    # --- Categorical highlights ---
    for cs in stats.categorical:
        if cs.top_values:
            top_label, top_count, top_pct = cs.top_values[0]
            if top_pct > 60:
                insights.append(
                    f"'{cs.column}' is dominated by '{top_label}', "
                    f"which accounts for {top_pct}% of all rows."
                )
            elif len(cs.top_values) >= 2:
                second_label, _, second_pct = cs.top_values[1]
                insights.append(
                    f"The most common values in '{cs.column}' are "
                    f"'{top_label}' ({top_pct}%) and '{second_label}' ({second_pct}%)."
                )

    # --- Datetime highlights ---
    for dt in stats.datetime:
        insights.append(
            f"'{dt.column}' spans {dt.span_days:,} days "
            f"(from {dt.min} to {dt.max}), "
            f"with data recorded approximately {dt.most_common_period}."
        )

    # --- Correlation highlights ---
    for pair in stats.correlations[:3]:
        direction_word = "tend to increase together" if pair.direction == "positive" \
            else "move in opposite directions"
        insights.append(
            f"'{pair.col_a}' and '{pair.col_b}' {direction_word} "
            f"({pair.strength} correlation, r={pair.pearson_r})."
        )

    return insights


# ===========================================================================
# Step 2 — Build a compact context string for the LLM
# ===========================================================================

def _build_llm_context(
    profile: DatasetProfile,
    stats: DatasetStats,
    deterministic_bullets: list[str],
) -> str:
    """
    Summarise the dataset into a compact text block to send to Ollama.
    We deliberately keep this small to stay within context limits and
    reduce latency for local models.
    """
    lines = [
        f"File: {profile.file_name}",
        f"Rows: {profile.total_rows:,}  |  Columns: {profile.total_columns}",
        "",
        "Column types:",
    ]

    for col in profile.columns:
        if col.chart_eligible:
            lines.append(f"  - {col.column_name}: {col.semantic_type}"
                         + (f" ({col.sub_type})" if col.sub_type else ""))

    lines.append("")
    lines.append("Key statistical facts:")
    for bullet in deterministic_bullets:
        lines.append(f"  • {bullet}")

    # Top correlations
    if stats.correlations:
        lines.append("")
        lines.append("Strongest correlations:")
        for pair in stats.correlations[:5]:
            lines.append(
                f"  • {pair.col_a} ↔ {pair.col_b}: "
                f"r={pair.pearson_r} ({pair.strength}, {pair.direction})"
            )

    # Numeric summaries
    lines.append("")
    lines.append("Numeric column summaries:")
    for ns in stats.numeric[:6]:
        lines.append(
            f"  • {ns.column}: mean={ns.mean}, median={ns.median}, "
            f"min={ns.min}, max={ns.max}, skew={ns.skewness}"
        )

    # Categorical top values
    lines.append("")
    lines.append("Categorical top values:")
    for cs in stats.categorical[:4]:
        top = ", ".join(f"{v}({p}%)" for v, _, p in cs.top_values[:3])
        lines.append(f"  • {cs.column}: {top}")

    return "\n".join(lines)


# ===========================================================================
# Step 3 — Ollama narrative generation
# ===========================================================================

_SYSTEM_PROMPT = """You are a data analyst writing insights for a non-technical business user.
Your audience has no statistics background — avoid jargon.
Be concise, specific, and actionable.
Write in plain English as if explaining to a curious colleague."""

def _call_ollama(context: str, model: str) -> Optional[str]:
    """
    Send the dataset context to Ollama and return a narrative string.
    Returns None on any failure so the caller can fall back gracefully.
    """
    prompt = f"""Here is a summary of a dataset:

{context}

Write 4 to 6 clear, plain-English insights about this data.
Focus on what is surprising, useful, or worth investigating.
Do not repeat the raw numbers mechanically — interpret what they mean.
Format your response as a numbered list.
"""
    try:
        response = ollama.chat(
            model    = model,
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


# ===========================================================================
# Public API
# ===========================================================================

def generate_insights(
    profile:     DatasetProfile,
    stats:       DatasetStats,
    use_ollama:  bool = True,
    model:       str  = DEFAULT_MODEL,
) -> dict:
    """
    Generate insights for the dashboard.

    Returns
    -------
    {
        "bullets"  : list[str],   # always present — deterministic bullets
        "narrative": str | None,  # LLM narrative, or None if Ollama unavailable
        "source"   : "llm" | "rules",
        "model"    : str | None,
    }
    """
    bullets = _extract_deterministic_insights(profile, stats)

    narrative = None
    source    = "rules"

    if use_ollama and OLLAMA_AVAILABLE:
        log.info(f"Calling Ollama ({model}) for narrative insights...")
        context   = _build_llm_context(profile, stats, bullets)
        narrative = _call_ollama(context, model)
        if narrative:
            source = "llm"
            log.info("Ollama narrative generated successfully")
        else:
            log.warning("Ollama failed — showing rule-based insights only")

    return {
        "bullets"  : bullets,
        "narrative": narrative,
        "source"   : source,
        "model"    : model if source == "llm" else None,
    }
