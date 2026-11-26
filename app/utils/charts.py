"""Reusable Plotly layout helpers for consistent portfolio visual aesthetics."""

from __future__ import annotations

import plotly.graph_objects as go

from app.utils.styles import PALETTE

DAY_NAMES_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def style_plotly(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PALETTE["text"]), x=0.01) if title else None,
        template="plotly_dark",
        paper_bgcolor="rgba(19, 26, 42, 0.85)",
        plot_bgcolor="rgba(11, 15, 25, 0.70)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=PALETTE["text"], size=12),
        height=height,
        margin=dict(l=45, r=25, t=52 if title else 28, b=45),
        legend=dict(
            bgcolor="rgba(19, 26, 42, 0.6)",
            bordercolor=PALETTE["border"],
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="#334155", font_color="#F8FAFC"),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.20)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.20)")
    return fig
