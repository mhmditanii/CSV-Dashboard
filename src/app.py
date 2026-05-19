from __future__ import annotations

import json
import logging

import dash
from dash import Input, Output, State, dcc, html, dash_table, no_update
import dash_bootstrap_components as dbc

from core.figures import figure_for_card
from core.io import load_uploaded_csv, read_cached
from core.ollama_client import select_and_polish
from core.planner import build_candidate_catalog
from core.profiler import profile_dataframe
from core.utils import safe_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CARD = {
    "background": "white",
    "border": "1px solid #e5e7eb",
    "borderRadius": "18px",
    "boxShadow": "0 8px 24px rgba(15,23,42,0.06)",
    "overflow": "hidden",
    "marginBottom": "22px",
}

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Serif+Display&display=swap",
    ],
    suppress_callback_exceptions=True,
)
server = app.server


def header():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "◇",
                        style={
                            "fontSize": "32px",
                            "color": "#4f46e5",
                            "lineHeight": "1",
                        },
                    ),
                    html.Div(
                        [
                            html.H1(
                                "Insights",
                                style={
                                    "margin": 0,
                                    "fontSize": "30px",
                                    "fontFamily": "DM Serif Display, serif",
                                    "color": "#0f172a",
                                },
                            ),
                            html.P(
                                "Upload any CSV. Ollama helps choose useful views; calculations stay local.",
                                style={
                                    "margin": 0,
                                    "color": "#64748b",
                                    "fontSize": "14px",
                                },
                            ),
                        ]
                    ),
                ],
                style={"display": "flex", "gap": "14px", "alignItems": "center"},
            ),
        ],
        style={
            "background": "white",
            "borderBottom": "1px solid #e5e7eb",
            "padding": "20px 34px",
            "position": "sticky",
            "top": 0,
            "zIndex": 50,
        },
    )


def upload_box():
    return html.Div(
        [
            dcc.Upload(
                id="upload",
                children=html.Div(
                    [
                        html.Div(
                            "⇪",
                            style={
                                "fontSize": "42px",
                                "color": "#4f46e5",
                                "marginBottom": "8px",
                            },
                        ),
                        html.Div(
                            [
                                html.Span("Drop a CSV here, or "),
                                html.Span(
                                    "browse",
                                    style={
                                        "fontWeight": 800,
                                        "color": "#4f46e5",
                                        "textDecoration": "underline",
                                    },
                                ),
                            ],
                            style={"fontSize": "18px", "color": "#334155"},
                        ),
                        html.P(
                            "The full CSV stays local. Ollama receives column profiles, candidate chart ideas, and two sample rows only.",
                            style={
                                "fontSize": "13px",
                                "color": "#94a3b8",
                                "marginTop": "8px",
                            },
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
                style={
                    "border": "2px dashed #c7d2fe",
                    "borderRadius": "20px",
                    "padding": "58px 24px",
                    "background": "#fafafe",
                    "cursor": "pointer",
                },
                max_size=250 * 1024 * 1024,
            ),
            html.Div(
                id="upload-error",
                style={"color": "#dc2626", "textAlign": "center", "marginTop": "12px"},
            ),
        ],
        id="upload-section",
        style={"maxWidth": "780px", "margin": "48px auto"},
    )


def stat_pill(label: str, value: str):
    return html.Div(
        [
            html.Div(
                label,
                style={
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "color": "#64748b",
                    "fontWeight": 800,
                },
            ),
            html.Div(
                value, style={"fontSize": "22px", "fontWeight": 800, "color": "#111827"}
            ),
        ],
        style={
            "background": "#f8fafc",
            "border": "1px solid #e5e7eb",
            "borderRadius": "14px",
            "padding": "14px 16px",
        },
    )


def sample_table(path: str, n: int = 10):
    df = read_cached(path).head(int(n or 10)).copy()
    rows = safe_json(df.to_dict("records"))
    cols = [{"name": str(c), "id": str(c)} for c in df.columns]
    return dash_table.DataTable(
        data=rows,
        columns=cols,
        page_action="none",
        fixed_rows={"headers": True},
        style_table={"overflowX": "auto", "maxHeight": "460px", "overflowY": "auto"},
        style_cell={
            "fontSize": "12px",
            "padding": "8px",
            "textAlign": "left",
            "minWidth": "110px",
            "maxWidth": "240px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "whiteSpace": "nowrap",
        },
        style_header={
            "fontWeight": "800",
            "background": "#f8fafc",
            "borderBottom": "2px solid #e5e7eb",
        },
        style_data={"border": "none", "borderBottom": "1px solid #f1f5f9"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fbfdff"}
        ],
    )


app.layout = html.Div(
    [
        dcc.Store(id="store-cache"),
        dcc.Store(id="store-plan"),
        header(),
        html.Main(
            [
                upload_box(),
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=html.Div(id="dashboard", style={"display": "none"}),
                ),
            ],
            style={"maxWidth": "1320px", "margin": "0 auto", "padding": "0 24px 80px"},
        ),
    ],
    style={
        "minHeight": "100vh",
        "background": "#f8fafc",
        "fontFamily": "Inter, system-ui, sans-serif",
    },
)


@app.callback(
    Output("store-cache", "data"),
    Output("store-plan", "data"),
    Output("upload-section", "style"),
    Output("dashboard", "style"),
    Output("upload-error", "children"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def analyze_upload(contents, filename):
    if not contents or not filename:
        return no_update, no_update, no_update, no_update, "No file received."
    try:
        df, cache_path, meta = load_uploaded_csv(contents, filename)
        log.info(
            "Loaded %s rows × %s columns from %s", len(df), len(df.columns), filename
        )
        profile = profile_dataframe(df)
        candidates = build_candidate_catalog(df, profile)
        plan, llm_issues = select_and_polish(df, profile, candidates, filename)
        plan["llm_issues"] = llm_issues
        plan["meta"] = meta
        return (
            safe_json({"path": cache_path, "filename": filename}),
            json.dumps(safe_json(plan)),
            {"display": "none"},
            {"display": "block"},
            "",
        )
    except Exception as e:
        log.exception("Upload analysis failed")
        return no_update, no_update, no_update, {"display": "none"}, f"Error: {e}"


@app.callback(
    Output("dashboard", "children"),
    Input("store-plan", "data"),
    State("store-cache", "data"),
    prevent_initial_call=True,
)
def render_dashboard(plan_json, cache):
    if not plan_json or not cache:
        return no_update
    plan = json.loads(plan_json)
    meta = plan.get("meta", {})
    profile = plan.get("profile", {})
    cards = plan.get("cards", [])
    issues = plan.get("llm_issues", [])

    top = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "Insights",
                        style={
                            "fontSize": "12px",
                            "fontWeight": 800,
                            "letterSpacing": "0.08em",
                            "textTransform": "uppercase",
                            "color": "#4f46e5",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Div(
                        plan.get("summary", ""),
                        style={
                            "fontFamily": "DM Serif Display, serif",
                            "fontSize": "25px",
                            "lineHeight": "1.35",
                            "color": "#111827",
                        },
                    ),
                    html.Div(
                        f"File: {meta.get('filename', '')}",
                        style={
                            "fontSize": "13px",
                            "color": "#64748b",
                            "marginTop": "10px",
                        },
                    ),
                ],
                style={"padding": "26px 30px"},
            )
        ],
        style={
            **CARD,
            "marginTop": "28px",
            "background": "linear-gradient(135deg,#eef2ff,#ffffff)",
        },
    )

    stats = dbc.Row(
        [
            dbc.Col(stat_pill("Rows", f"{profile.get('rows', 0):,}"), md=3),
            dbc.Col(stat_pill("Columns", f"{profile.get('columns', 0):,}"), md=3),
            dbc.Col(
                stat_pill(
                    "Candidate Views", f"{plan.get('candidate_count', len(cards)):,}"
                ),
                md=3,
            ),
            dbc.Col(stat_pill("Shown", f"{len(cards):,}"), md=3),
        ],
        className="g-3",
        style={"marginBottom": "22px"},
    )

    notes = None
    if issues:
        notes = html.Details(
            [
                html.Summary(
                    "Analysis notes",
                    style={"cursor": "pointer", "color": "#64748b", "fontSize": "13px"},
                ),
                html.Ul(
                    [html.Li(i) for i in issues],
                    style={"fontSize": "13px", "color": "#64748b", "marginTop": "8px"},
                ),
            ],
            style={"marginBottom": "18px"},
        )

    card_components = []
    for card in cards:
        fig, height = figure_for_card(card)
        component = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            card.get("question", ""),
                            style={
                                "fontSize": "12px",
                                "fontWeight": 800,
                                "letterSpacing": "0.08em",
                                "textTransform": "uppercase",
                                "color": "#6366f1",
                                "marginBottom": "8px",
                            },
                        ),
                        html.H3(
                            card.get("title", ""),
                            style={
                                "fontSize": "19px",
                                "fontFamily": "DM Serif Display, serif",
                                "margin": 0,
                                "color": "#111827",
                            },
                        ),
                    ],
                    style={"padding": "18px 20px 0"},
                ),
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": f"{height}px"},
                ),
                html.Div(
                    [html.Strong("Why this view: "), html.Span(card.get("why", ""))],
                    style={
                        "borderTop": "1px solid #eef2f7",
                        "background": "#fbfdff",
                        "padding": "12px 18px 4px",
                        "fontSize": "13px",
                        "color": "#64748b",
                        "lineHeight": "1.5",
                    },
                ),
                html.Div(
                    [html.Strong("Takeaway: "), html.Span(card.get("takeaway", ""))],
                    style={
                        "background": "#fbfdff",
                        "padding": "4px 18px 14px",
                        "fontSize": "14px",
                        "color": "#475569",
                        "lineHeight": "1.5",
                    },
                ),
            ],
            style=CARD,
        )
        width = 12 if card.get("kind") in {"table", "map"} else 6
        card_components.append(dbc.Col(component, md=width))

    sample = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Sample Data",
                                style={
                                    "fontSize": "19px",
                                    "fontFamily": "DM Serif Display, serif",
                                    "margin": 0,
                                },
                            ),
                            html.Div(
                                "A quick look at the uploaded rows.",
                                style={"fontSize": "13px", "color": "#64748b"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Show ", style={"fontSize": "13px", "color": "#64748b"}
                            ),
                            dcc.Dropdown(
                                id="sample-n",
                                options=[
                                    {"label": str(n), "value": n}
                                    for n in [10, 25, 50, 100]
                                ],
                                value=10,
                                clearable=False,
                                style={"width": "92px", "fontSize": "13px"},
                            ),
                            html.Span(
                                " rows", style={"fontSize": "13px", "color": "#64748b"}
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "alignItems": "center"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "padding": "18px 20px",
                    "borderBottom": "1px solid #eef2f7",
                },
            ),
            html.Div(
                id="sample-table",
                children=sample_table(cache["path"], 10),
                style={"padding": "14px"},
            ),
        ],
        style=CARD,
    )

    return [top, stats, notes, dbc.Row(card_components, className="g-3"), sample]


@app.callback(
    Output("sample-table", "children"),
    Input("sample-n", "value"),
    State("store-cache", "data"),
    prevent_initial_call=True,
)
def render_sample(n, cache):
    if not cache:
        return no_update
    return sample_table(cache["path"], int(n or 10))


if __name__ == "__main__":
    app.run(debug=True, port=8050)
