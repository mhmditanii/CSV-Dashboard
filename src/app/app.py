"""
app.py
------
Main Dash application.

Layout:
  ┌─────────────────────────────────────┐
  │  Header + Upload zone               │
  ├─────────────────────────────────────┤
  │  Dataset summary bar                │
  ├──────────────┬──────────────────────┤
  │  Insights    │  Charts grid         │
  │  panel       │                      │
  ├──────────────┴──────────────────────┤
  │  Data preview table                 │
  └─────────────────────────────────────┘

Run:
    python app.py
    open http://localhost:8050
"""

import base64
import io
import logging
import os
import tempfile
import time
from pathlib import Path
import sys

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

# NEW
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))

from data_loader import load_csv, get_dataframe, preview
from schema_detector import detect_schema
from stats_engine import compute_stats
from chart_gen import generate_charts
from insight_engine import generate_insights

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    assets_folder=os.path.join(os.path.dirname(__file__), "css"),
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+3:wght@300;400;600&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="CSV Insights",
)
server = app.server  # for production deployment (gunicorn)


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------


def _stat_card(label: str, value: str, icon: str = "") -> dbc.Col:
    return dbc.Col(
        html.Div(
            [
                html.Span(icon + " ", className="stat-icon"),
                html.Div(value, className="stat-value"),
                html.Div(label, className="stat-label"),
            ],
            className="stat-card",
        ),
        xs=6,
        sm=4,
        md=2,
    )


def _insight_card(text: str, index: int) -> html.Div:
    icons = ["🔍", "📊", "⚠️", "🔗", "📅", "💡", "📌", "🧩"]
    icon = icons[index % len(icons)]
    return html.Div(
        [
            html.Span(icon, className="insight-icon"),
            html.P(text, className="insight-text"),
        ],
        className="insight-card",
    )


def _chart_card(chart: dict) -> dbc.Col:
    return dbc.Col(
        html.Div(
            [
                dcc.Graph(
                    figure=chart["figure"],
                    config={"displayModeBar": False, "responsive": True},
                    className="chart-figure",
                ),
                html.P(chart.get("description", ""), className="chart-description"),
            ],
            className="chart-card",
        ),
        xs=12,
        md=6,
        xl=6,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    [
        # ── Hidden state stores ──────────────────────────────────────────────
        dcc.Store(id="store-parquet-path"),  # path to cached parquet file
        dcc.Store(id="store-profile-json"),  # serialised DatasetProfile
        dcc.Loading(
            id="loading-overlay",
            fullscreen=True,
            type="dot",
            color="#4C72B0",
            style={"backgroundColor": "rgba(255,255,255,0.7)"},
        ),
        # ── Header ───────────────────────────────────────────────────────────
        html.Header(
            [
                html.Div(
                    [
                        html.H1("CSV Insights", className="header-title"),
                        html.P(
                            "Upload any CSV. Get instant charts and insights.",
                            className="header-sub",
                        ),
                    ],
                    className="header-inner",
                ),
            ],
            className="app-header",
        ),
        # ── Main content ─────────────────────────────────────────────────────
        html.Main(
            [
                # Upload zone
                html.Div(
                    [
                        dcc.Upload(
                            id="upload-csv",
                            children=html.Div(
                                [
                                    html.Div("📂", className="upload-icon"),
                                    html.Div(
                                        [
                                            html.Span("Drag & drop a CSV file here"),
                                            html.Br(),
                                            html.Span("or "),
                                            html.Span(
                                                "browse to upload",
                                                className="upload-link",
                                            ),
                                        ],
                                        className="upload-text",
                                    ),
                                    html.Div(
                                        "Supports files up to ~100 MB · Any schema",
                                        className="upload-hint",
                                    ),
                                ]
                            ),
                            className="upload-zone",
                            accept=".csv",
                            max_size=100 * 1024 * 1024,
                        ),
                        html.Div(id="upload-error", className="upload-error"),
                    ],
                    className="upload-section",
                ),
                # Dashboard output (hidden until file uploaded)
                html.Div(id="dashboard-output", className="dashboard-hidden"),
            ],
            className="main-content",
        ),
    ],
    className="app-root",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("dashboard-output", "children"),
    Output("dashboard-output", "className"),
    Output("upload-error", "children"),
    Output("store-parquet-path", "data"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def process_upload(contents: str, filename: str):
    """
    Main pipeline callback — fires when a CSV is uploaded.
    Runs: load → schema detect → stats → charts → insights → render
    """
    if contents is None:
        return dash.no_update, dash.no_update, "", dash.no_update

    # ── 1. Decode upload ─────────────────────────────────────────────────
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
    except Exception as e:
        return "", "dashboard-hidden", f"Failed to read file: {e}", dash.no_update

    # ── 2. Save to temp file so DuckDB can read it ────────────────────────
    try:
        suffix = Path(filename).suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
    except Exception as e:
        return "", "dashboard-hidden", f"Failed to save file: {e}", dash.no_update

    # ── 3. Full pipeline ─────────────────────────────────────────────────
    try:
        t0 = time.time()

        # Load
        df, parquet_path = load_csv(tmp_path)

        # Schema detection
        profile = detect_schema(
            tmp_path, use_ollama=False
        )  # Ollama called separately below

        # Stats
        stats = compute_stats(df, profile)

        # Charts
        charts = generate_charts(df, profile, stats)

        # Insights
        insights = generate_insights(profile, stats, use_ollama=True)

        elapsed = time.time() - t0
        log.info(f"Pipeline completed in {elapsed:.1f}s")

    except Exception as e:
        log.exception("Pipeline failed")
        return "", "dashboard-hidden", f"Analysis failed: {e}", dash.no_update
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # ── 4. Build dashboard layout ─────────────────────────────────────────
    dashboard = _build_dashboard(
        df, profile, stats, charts, insights, filename, elapsed
    )
    return dashboard, "dashboard-visible", "", parquet_path


def _build_dashboard(df, profile, stats, charts, insights, filename, elapsed):
    """Assemble all dashboard sections into a Dash layout tree."""

    # ── Summary bar ──────────────────────────────────────────────────────
    chart_eligible = len(profile.chart_eligible_columns())
    n_numeric = len(profile.by_type("numeric"))
    n_categorical = len(profile.by_type("categorical"))

    summary_bar = html.Div(
        [
            html.Div(
                [
                    html.Span("📄 ", className="file-icon"),
                    html.Span(filename, className="file-name"),
                    html.Span(f" — analysed in {elapsed:.1f}s", className="file-time"),
                ],
                className="file-badge",
            ),
            dbc.Row(
                [
                    _stat_card("Total rows", f"{profile.total_rows:,}", "🗂"),
                    _stat_card("Columns", str(profile.total_columns), "📐"),
                    _stat_card("Numeric", str(n_numeric), "🔢"),
                    _stat_card("Categorical", str(n_categorical), "🏷"),
                    _stat_card("Chart-ready", str(chart_eligible), "📈"),
                    _stat_card("Insights", str(len(insights["bullets"])), "💡"),
                ],
                className="g-2",
            ),
            # Dataset-level warnings
            *[
                html.Div([html.Span("⚠️ "), w], className="dataset-warning")
                for w in profile.dataset_warnings
            ],
        ],
        className="summary-bar",
    )

    # ── Insights panel ────────────────────────────────────────────────────
    insight_cards = [_insight_card(b, i) for i, b in enumerate(insights["bullets"])]

    narrative_section = html.Div()
    if insights.get("narrative"):
        # Parse numbered list from Ollama response
        lines = [
            l.strip()
            for l in insights["narrative"].split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]
        narrative_section = html.Div(
            [
                html.Div(
                    [
                        html.Span("🤖", className="ai-badge-icon"),
                        html.Span(
                            f"AI Narrative · {insights.get('model', 'ollama')}",
                            className="ai-badge-label",
                        ),
                    ],
                    className="ai-badge",
                ),
                *[html.P(line, className="narrative-line") for line in lines if line],
            ],
            className="narrative-section",
        )

    insights_panel = html.Div(
        [
            html.H2("Insights", className="section-title"),
            html.Div(insight_cards, className="insights-grid"),
            narrative_section,
        ],
        className="insights-panel",
    )

    # ── Charts grid ───────────────────────────────────────────────────────
    chart_cols = [_chart_card(c) for c in charts]
    charts_section = html.Div(
        [
            html.H2("Charts", className="section-title"),
            dbc.Row(chart_cols, className="g-3 charts-grid"),
        ],
        className="charts-section",
    )

    # ── Data preview ──────────────────────────────────────────────────────
    preview_df = df.head(100)
    table = dash_table.DataTable(
        data=preview_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in preview_df.columns],
        page_size=15,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1a1a2e",
            "color": "white",
            "fontWeight": "600",
            "fontFamily": "Source Sans 3, sans-serif",
            "fontSize": "13px",
            "padding": "10px 14px",
            "borderBottom": "2px solid #4C72B0",
        },
        style_cell={
            "fontFamily": "Source Sans 3, sans-serif",
            "fontSize": "13px",
            "padding": "8px 14px",
            "color": "#333",
            "border": "1px solid #eee",
            "maxWidth": "200px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9fb"},
        ],
        tooltip_delay=0,
        tooltip_duration=None,
    )

    preview_section = html.Div(
        [
            html.H2("Data Preview", className="section-title"),
            html.P(
                f"Showing first 100 rows of {profile.total_rows:,}.",
                className="section-sub",
            ),
            table,
        ],
        className="preview-section",
    )

    # ── Assemble ──────────────────────────────────────────────────────────
    return html.Div(
        [
            summary_bar,
            html.Div(
                [
                    html.Div([insights_panel], className="left-panel"),
                    html.Div([charts_section], className="right-panel"),
                ],
                className="main-panels",
            ),
            preview_section,
        ],
        className="dashboard-inner",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
