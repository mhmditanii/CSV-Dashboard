from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = px.colors.qualitative.Set2


def make_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> go.Figure:
    if not rows:
        rows = [{"Message": "No rows to show"}]
    if columns is None:
        columns = list(rows[0].keys())
    df = pd.DataFrame(rows)
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    df = df[columns]
    fill = [["#f8fafc" if i % 2 else "#ffffff" for i in range(len(df))] for _ in columns]
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in columns], fill_color="#4f46e5", font=dict(color="white", size=12), align="left", height=34),
        cells=dict(values=[df[c].astype(str).tolist() for c in columns], fill_color=fill, font=dict(color="#334155", size=12), align="left", height=31, line_color="#e2e8f0"),
    ))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=min(470, 88 + len(df) * 32))
    return fig


def theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title or "", x=0.02, font=dict(size=15, color="#111827", family="Georgia, serif")),
        font=dict(family="Inter, system-ui, sans-serif", color="#334155", size=12),
        margin=dict(l=60, r=24, t=55 if title else 20, b=70),
        colorway=PALETTE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickangle=-25, automargin=True, title_standoff=14)
    fig.update_yaxes(automargin=True, title_standoff=14)
    return fig


def _bar_from_plot_data(card: dict[str, Any]) -> tuple[go.Figure, int]:
    plot = card.get("raw_plot_data") or card.get("plot_data") or []
    df = pd.DataFrame(plot)
    if df.empty:
        return make_table(card.get("data", []), card.get("columns")), 330
    if "Group" not in df.columns:
        df = pd.DataFrame(card.get("data", []))
        x_col = card.get("columns", [df.columns[0]])[0]
        y_col = card.get("columns", list(df.columns))[-1]
        fig = px.bar(df, x=x_col, y=y_col, labels={x_col: card.get("x_label", x_col), y_col: card.get("y_label", y_col)})
    else:
        ycol = "Value" if "Value" in df.columns else df.columns[-1]
        fig = px.bar(df, x="Group", y=ycol, labels={"Group": card.get("x_label", "Group"), ycol: card.get("y_label", ycol)})
    fig.update_layout(showlegend=False)
    return theme(fig, card.get("title", "")), 390


def figure_for_card(card: dict[str, Any]) -> tuple[go.Figure, int]:
    kind = card.get("kind")
    title = card.get("title", "")

    if kind == "table":
        rows = card.get("data", [])
        return make_table(rows, card.get("columns")), min(480, 120 + max(2, len(rows)) * 34)

    if kind in {"bar_table", "table_bar"}:
        return _bar_from_plot_data(card)

    if kind == "histogram":
        df = pd.DataFrame(card.get("data", []))
        if df.empty or "value" not in df.columns:
            return make_table([], None), 260
        fig = px.histogram(df, x="value", nbins=38, labels={"value": card.get("x_label", card.get("x", "Value")), "count": "Rows"})
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title=card.get("y_label", "Rows"))
        return theme(fig, title), 390

    if kind == "line":
        df = pd.DataFrame(card.get("data", []))
        if df.empty:
            return make_table([], None), 260
        x = "Date" if "Date" in df.columns else df.columns[0]
        y = [c for c in df.columns if c != x][-1]
        fig = px.line(df, x=x, y=y, markers=True, labels={x: card.get("x_label", x), y: card.get("y_label", y)})
        fig.update_traces(line_width=2.5)
        return theme(fig, title), 390

    if kind == "line_table":
        df = pd.DataFrame(card.get("plot_data", []))
        if df.empty:
            return make_table(card.get("data", []), card.get("columns")), 330
        fig = px.line(df, x="X", y="Y", markers=True, labels={"X": card.get("x_label", "Range"), "Y": card.get("y_label", "Value")})
        fig.update_traces(line_width=2.5)
        return theme(fig, title), 390

    if kind == "map":
        df = pd.DataFrame(card.get("data", []))
        if df.empty:
            return make_table([], None), 260
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", zoom=9, mapbox_style="carto-positron", opacity=0.55)
        fig.update_layout(margin=dict(l=0, r=0, t=42, b=0), title=dict(text=title, x=0.02, font=dict(size=15, color="#111827", family="Georgia, serif")))
        return fig, 460

    return make_table(card.get("data", []), card.get("columns")), 300
