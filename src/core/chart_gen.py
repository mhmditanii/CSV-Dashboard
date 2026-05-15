"""
chart_generator.py
------------------
Automatically selects and builds Plotly figures based on column semantic types.

The schema_detector assigns chart_roles to each column.
This module queries those roles to decide WHAT charts to build and HOW.

Design principle:
  - Chart selection is driven entirely by the DatasetProfile — no hardcoded column names.
  - Every chart function returns a go.Figure so Dash can render it directly.
  - Non-technical users see clean, labelled charts with no jargon.

Usage:
    from chart_generator import generate_charts
    charts = generate_charts(df, profile, stats)
    # charts is a list of {"title": str, "figure": go.Figure, "description": str}
"""

import logging
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from schema_detector import DatasetProfile, ColumnProfile
from stats_engine import DatasetStats

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual theme — consistent across all charts
# ---------------------------------------------------------------------------

THEME = dict(
    template="plotly_white",
    font_family="Georgia, serif",
    title_font=dict(size=16, color="#1a1a2e"),
    axis_color="#555",
    grid_color="#e8e8e8",
    # Colour palette — accessible, non-garish
    palette=px.colors.qualitative.Set2,
)

MAX_CATEGORY_BARS = 20  # truncate long-tail categoricals in bar charts
MAX_CHARTS = 12  # cap total charts to avoid overwhelming the UI
HISTOGRAM_BINS = 30


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        template=THEME["template"],
        font=dict(family=THEME["font_family"], color=THEME["axis_color"]),
        title=dict(text=title, font=THEME["title_font"], x=0.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=THEME["palette"],
    )
    fig.update_xaxes(gridcolor=THEME["grid_color"], linecolor=THEME["grid_color"])
    fig.update_yaxes(gridcolor=THEME["grid_color"], linecolor=THEME["grid_color"])
    return fig


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------


def _histogram(df: pd.DataFrame, col: ColumnProfile) -> Optional[go.Figure]:
    """Distribution of a numeric column."""
    series = pd.to_numeric(df[col.column_name], errors="coerce").dropna()
    if series.empty:
        return None

    fig = px.histogram(
        series,
        nbins=HISTOGRAM_BINS,
        labels={"value": col.column_name, "count": "Count"},
        color_discrete_sequence=["#4C72B0"],
    )
    fig.update_traces(marker_line_color="white", marker_line_width=0.5)
    sub = f"Mean: {series.mean():.2f}  |  Median: {series.median():.2f}  |  Std: {series.std():.2f}"
    return _apply_theme(fig, f"Distribution of {col.column_name}")


def _bar_chart(df: pd.DataFrame, col: ColumnProfile) -> Optional[go.Figure]:
    """Value counts for a categorical column."""
    counts = (
        df[col.column_name]
        .dropna()
        .value_counts()
        .head(MAX_CATEGORY_BARS)
        .reset_index()
    )
    counts.columns = [col.column_name, "Count"]

    fig = px.bar(
        counts,
        x=col.column_name,
        y="Count",
        color=col.column_name,
        color_discrete_sequence=THEME["palette"],
        labels={"Count": "Number of rows"},
    )
    fig.update_layout(showlegend=False)
    return _apply_theme(fig, f"{col.column_name} — Value Counts")


def _pie_chart(df: pd.DataFrame, col: ColumnProfile) -> Optional[go.Figure]:
    """Pie chart for low-cardinality categoricals (≤8 unique values)."""
    if col.unique_count > 8:
        return None
    counts = df[col.column_name].dropna().value_counts().reset_index()
    counts.columns = [col.column_name, "Count"]

    fig = px.pie(
        counts,
        names=col.column_name,
        values="Count",
        color_discrete_sequence=THEME["palette"],
        hole=0.35,  # donut style — easier to read for non-technical users
    )
    fig.update_traces(textinfo="label+percent", textposition="outside")
    return _apply_theme(fig, f"{col.column_name} — Breakdown")


def _box_plot(
    df: pd.DataFrame, num_col: ColumnProfile, cat_col: Optional[ColumnProfile] = None
) -> Optional[go.Figure]:
    """
    Box plot of a numeric column, optionally split by a categorical.
    Great for showing spread and outliers to non-technical users.
    """
    y_series = pd.to_numeric(df[num_col.column_name], errors="coerce")

    if cat_col and cat_col.unique_count <= 8:
        fig = px.box(
            df,
            x=cat_col.column_name,
            y=num_col.column_name,
            color=cat_col.column_name,
            color_discrete_sequence=THEME["palette"],
            points="outliers",
        )
        title = f"{num_col.column_name} by {cat_col.column_name}"
    else:
        fig = px.box(
            df,
            y=num_col.column_name,
            points="outliers",
            color_discrete_sequence=["#4C72B0"],
        )
        title = f"{num_col.column_name} — Spread & Outliers"

    return _apply_theme(fig, title)


def _scatter(
    df: pd.DataFrame,
    col_x: ColumnProfile,
    col_y: ColumnProfile,
    color_col: Optional[ColumnProfile] = None,
) -> Optional[go.Figure]:
    """Scatter plot between two numeric columns."""
    kwargs = dict(
        x=col_x.column_name,
        y=col_y.column_name,
        opacity=0.6,
        labels={
            col_x.column_name: col_x.column_name,
            col_y.column_name: col_y.column_name,
        },
        color_discrete_sequence=THEME["palette"],
    )
    if color_col and color_col.unique_count <= 8:
        kwargs["color"] = color_col.column_name

    # Sample large datasets for scatter to avoid sluggish rendering
    plot_df = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df

    fig = px.scatter(plot_df, **kwargs)
    title = f"{col_x.column_name} vs {col_y.column_name}"
    if color_col:
        title += f" (by {color_col.column_name})"
    return _apply_theme(fig, title)


def _time_series(
    df: pd.DataFrame, dt_col: ColumnProfile, num_col: ColumnProfile
) -> Optional[go.Figure]:
    """Line chart of a numeric column over time."""
    try:
        temp = df[[dt_col.column_name, num_col.column_name]].copy()
        temp[dt_col.column_name] = pd.to_datetime(
            temp[dt_col.column_name], infer_datetime_format=True, errors="coerce"
        )
        temp = temp.dropna().sort_values(dt_col.column_name)

        # Aggregate by date to avoid over-plotting
        temp = (
            temp.groupby(pd.Grouper(key=dt_col.column_name, freq="D"))[
                num_col.column_name
            ]
            .mean()
            .reset_index()
        )

        fig = px.line(
            temp,
            x=dt_col.column_name,
            y=num_col.column_name,
            labels={num_col.column_name: f"Avg {num_col.column_name}"},
            color_discrete_sequence=["#4C72B0"],
        )
        fig.update_traces(line_width=1.5)
        return _apply_theme(fig, f"{num_col.column_name} Over Time")
    except Exception as e:
        log.warning(f"Time series chart failed: {e}")
        return None


def _map_scatter(
    df: pd.DataFrame,
    lat_col: ColumnProfile,
    lon_col: ColumnProfile,
    color_col: Optional[ColumnProfile] = None,
) -> Optional[go.Figure]:
    """Scatter map for datasets with lat/lon columns."""
    try:
        plot_df = (
            df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df
        )
        kwargs = dict(lat=lat_col.column_name, lon=lon_col.column_name)
        if color_col:
            kwargs["color"] = color_col.column_name

        fig = px.scatter_mapbox(
            plot_df,
            **kwargs,
            zoom=9,
            mapbox_style="carto-positron",
            opacity=0.5,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        return _apply_theme(fig, "Geographic Distribution")
    except Exception as e:
        log.warning(f"Map chart failed: {e}")
        return None


def _correlation_heatmap(
    df: pd.DataFrame, numeric_profiles: list[ColumnProfile]
) -> Optional[go.Figure]:
    """Correlation heatmap across all numeric columns."""
    if len(numeric_profiles) < 3:
        return None
    cols = [p.column_name for p in numeric_profiles]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_traces(textfont_size=11)
    return _apply_theme(fig, "Correlation Between Numeric Columns")


# ---------------------------------------------------------------------------
# Chart selection logic
# ---------------------------------------------------------------------------


def _pick_color_col(profiles: list[ColumnProfile]) -> Optional[ColumnProfile]:
    """Pick the best low-cardinality column to use as a color dimension."""
    candidates = [
        p for p in profiles if "color" in p.chart_roles and p.unique_count <= 6
    ]
    return candidates[0] if candidates else None


def generate_charts(
    df: pd.DataFrame, profile: DatasetProfile, stats: DatasetStats
) -> list[dict]:
    """
    Auto-generate charts based on the dataset profile.

    Returns
    -------
    List of dicts: [{"title": str, "figure": go.Figure, "description": str}, ...]
    """
    charts = []

    eligible = profile.chart_eligible_columns()
    numeric = profile.by_type("numeric")
    cats = profile.by_type("categorical")
    booleans = profile.by_type("boolean")
    datetimes = profile.by_type("datetime")
    geo_lats = profile.with_role("map_lat")
    geo_lons = profile.with_role("map_lon")
    color_col = _pick_color_col(eligible)

    # 1. Histograms — one per numeric column (up to 4)
    for col in numeric[:4]:
        fig = _histogram(df, col)
        if fig:
            charts.append(
                {
                    "title": f"Distribution of {col.column_name}",
                    "figure": fig,
                    "description": (
                        f"Shows how {col.column_name} values are distributed. "
                        f"Range: {col.numeric_min} – {col.numeric_max}."
                    ),
                }
            )

    # 2. Bar charts — one per categorical column (up to 3)
    for col in (cats + booleans)[:3]:
        # Prefer pie for very low cardinality
        fig = _pie_chart(df, col) if col.unique_count <= 5 else _bar_chart(df, col)
        if fig:
            charts.append(
                {
                    "title": col.column_name,
                    "figure": fig,
                    "description": f"Breakdown of values in {col.column_name}.",
                }
            )

    # 3. Box plots — numeric split by best categorical
    best_cat = cats[0] if cats else (booleans[0] if booleans else None)
    for col in numeric[:3]:
        fig = _box_plot(df, col, best_cat)
        if fig:
            label = f" by {best_cat.column_name}" if best_cat else ""
            charts.append(
                {
                    "title": f"{col.column_name}{label}",
                    "figure": fig,
                    "description": (
                        f"Shows the spread and outliers in {col.column_name}"
                        + (
                            f", broken down by {best_cat.column_name}."
                            if best_cat
                            else "."
                        )
                    ),
                }
            )

    # 4. Scatter — top correlated pair
    if stats.correlations and len(numeric) >= 2:
        top = stats.correlations[0]
        col_a = next((p for p in numeric if p.column_name == top.col_a), None)
        col_b = next((p for p in numeric if p.column_name == top.col_b), None)
        if col_a and col_b:
            fig = _scatter(df, col_a, col_b, color_col)
            if fig:
                charts.append(
                    {
                        "title": f"{col_a.column_name} vs {col_b.column_name}",
                        "figure": fig,
                        "description": (
                            f"These two columns have a {top.strength} {top.direction} "
                            f"correlation (r={top.pearson_r})."
                        ),
                    }
                )

    # 5. Time series — datetime × numeric
    if datetimes and numeric:
        fig = _time_series(df, datetimes[0], numeric[0])
        if fig:
            charts.append(
                {
                    "title": f"{numeric[0].column_name} Over Time",
                    "figure": fig,
                    "description": f"Average {numeric[0].column_name} per day over time.",
                }
            )

    # 6. Map — if lat/lon detected
    if geo_lats and geo_lons:
        num_for_color = numeric[0] if numeric else None
        fig = _map_scatter(df, geo_lats[0], geo_lons[0], num_for_color)
        if fig:
            charts.append(
                {
                    "title": "Geographic Distribution",
                    "figure": fig,
                    "description": "Points plotted by their latitude and longitude coordinates.",
                }
            )

    # 7. Correlation heatmap — if enough numeric columns
    if len(numeric) >= 3:
        fig = _correlation_heatmap(df, numeric)
        if fig:
            charts.append(
                {
                    "title": "Correlation Heatmap",
                    "figure": fig,
                    "description": (
                        "Shows how strongly each pair of numeric columns move together. "
                        "Values near 1 or -1 indicate strong relationships."
                    ),
                }
            )

    log.info(f"Generated {len(charts)} charts")
    return charts[:MAX_CHARTS]
