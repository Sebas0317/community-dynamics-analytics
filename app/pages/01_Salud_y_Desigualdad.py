"""Page 01: Macro Ecosystem Health, Participation Inequality (Gini/Lorenz) & Activation (P01-P05)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.charts import style_plotly
from app.utils.data_loader import load_output_json, load_output_parquet
from app.utils.styles import (
    CATEGORICAL_COLORS,
    PALETTE,
    render_hero,
    render_insight,
    render_question_header,
    render_sidebar_footer,
    setup_page,
)

setup_page("Salud y Desigualdad", icon="📈")
exec_sum = load_output_json("executive_summary.json")
s1 = load_output_json("p01_p09_summary.json")
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Pilar 1 · Preguntas P01 – P05",
    title="Crecimiento, Escala y Desigualdad de Participación",
    subtitle="Medimos la vitalidad diaria (DAU/WAU/MAU), la concentración de voces (Curva de Lorenz y Gini), la salud neta de cada grupo, la fricción por moderación y el embudo real de activación.",
)

# =============================================================================
# P01: Scale, DAU / WAU / MAU & Stickiness
# =============================================================================
render_question_header(
    "P01",
    "¿Cómo evolucionó la escala y vitalidad de la comunidad (DAU / WAU / MAU y Stickiness)?",
    "Agregaciones temporales diarias y ventanas móviles de 7d/30d en DuckDB sobre mensajes de usuario.",
)
df_p01 = load_output_parquet("p01_daily_metrics.parquet")

k1, k2, k3, k4 = st.columns(4)
k1.metric("DAU Promedio", f"{s1.get('p01_avg_dau', 0):.1f}", f"Pico: {s1.get('p01_peak_dau', 0)} usuarios/día")
k2.metric("Stickiness (DAU/MAU)", f"{s1.get('p01_avg_stickiness', 0) * 100:.1f}%", "Estándar social top >25%")
k3.metric("Pico Diario Mensajes", f"{int(df_p01['user_messages'].max()):,}", df_p01.loc[df_p01["user_messages"].idxmax(), "date_str"])
k4.metric("Miembros Acumulados", f"{int(df_p01['cum_unique_members'].max()):,}", "Emisores únicos históricos")

col1, col2 = st.columns([3, 2])
with col1:
    fig_dau = go.Figure()
    fig_dau.add_trace(go.Scatter(x=df_p01["date_str"], y=df_p01["dau"], name="DAU (Diario)", line=dict(color=PALETTE["primary"], width=2)))
    fig_dau.add_trace(go.Scatter(x=df_p01["date_str"], y=df_p01["wau"], name="WAU (7 días)", line=dict(color=PALETTE["secondary"], width=2)))
    fig_dau.add_trace(go.Scatter(x=df_p01["date_str"], y=df_p01["mau"], name="MAU (30 días)", line=dict(color=PALETTE["purple"], width=2.5)))
    style_plotly(fig_dau, "Evolución de Usuarios Activos: DAU vs. WAU vs. MAU", height=370)
    st.plotly_chart(fig_dau, use_container_width=True)

with col2:
    fig_stick = go.Figure()
    fig_stick.add_trace(
        go.Scatter(
            x=df_p01["date_str"],
            y=df_p01["stickiness_dau_mau"] * 100,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.14)",
            line=dict(color=PALETTE["primary"], width=2.5),
            name="Stickiness (%)",
        )
    )
    fig_stick.add_hline(y=25, line_dash="dash", line_color=PALETTE["accent"], annotation_text="Benchmark SaaS/Social (25%)")
    style_plotly(fig_stick, "Stickiness Diario (DAU / MAU %)", height=370)
    fig_stick.update_yaxes(title="%")
    st.plotly_chart(fig_stick, use_container_width=True)

# =============================================================================
# P02: 1-9-90 Rule, Lorenz Curve & Gini
# =============================================================================
render_question_header(
    "P02",
    "¿Se cumple la Regla del 1%–9%–90% en una comunidad de mensajería real?",
    "Curva de Lorenz, Índice de Gini global y desagregado por grupo + umbral de Pareto (80%).",
)
df_lor = load_output_parquet("p02_lorenz_global.parquet")
df_gini = load_output_parquet("p02_gini_by_group.parquet").sort_values("gini", ascending=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Gini Global", f"{s1.get('p02_global_gini', 0):.4f}", "Alta concentración")
m2.metric("Top 1% Emisores", f"{s1.get('p02_share_top_1pct', 0):.1f}%", "de todos los mensajes")
m3.metric("Siguiente 9% (Top 2–10%)", f"{s1.get('p02_share_next_9pct', 0):.1f}%", f"Top 10% = {s1.get('p02_share_top_1pct', 0) + s1.get('p02_share_next_9pct', 0):.1f}%")
m4.metric("Umbral Pareto 80%", f"{s1.get('p02_pareto_80_members', 0)} miembros", f"Top 20% = {s1.get('p02_share_top_20pct', 0):.1f}%")

c_l, c_r = st.columns([1, 1.35])
with c_l:
    fig_lor = go.Figure()
    fig_lor.add_trace(
        go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines", name="Igualdad Perfecta",
            line=dict(color=PALETTE["slate"], dash="dash", width=1.5),
        )
    )
    fig_lor.add_trace(
        go.Scatter(
            x=df_lor["pop_pct"],
            y=df_lor["msg_pct"],
            mode="lines",
            name=f"Curva de Lorenz (Gini={s1.get('p02_global_gini', 0):.3f})",
            fill="tonexty",
            fillcolor="rgba(244, 63, 94, 0.18)",
            line=dict(color=PALETTE["danger"], width=3),
        )
    )
    style_plotly(fig_lor, "Curva de Lorenz Global de Participación", height=420)
    fig_lor.update_xaxes(title="% Acumulado de Miembros (orden ascendente)")
    fig_lor.update_yaxes(title="% Acumulado de Mensajes")
    st.plotly_chart(fig_lor, use_container_width=True)

with c_r:
    fig_gini = px.bar(
        df_gini,
        x="gini",
        y="group_label",
        color="macro_category",
        orientation="h",
        color_discrete_sequence=CATEGORICAL_COLORS,
        hover_data=["active_members", "top_10pct_share", "median_msgs_per_member"],
    )
    style_plotly(fig_gini, "Índice de Gini por Grupo (menor = más democrático)", height=420)
    fig_gini.update_xaxes(title="Índice de Gini (0 a 1)", range=[0, 1])
    fig_gini.update_yaxes(title="")
    st.plotly_chart(fig_gini, use_container_width=True)

render_insight(
    "Lectura de Producto (P02)",
    f"La desigualdad global (Gini <strong>{s1.get('p02_global_gini', 0):.3f}</strong>) confirma una variante aguda de la regla 1–9–90: "
    f"el <strong>Top 10% produce el {s1.get('p02_share_top_1pct', 0) + s1.get('p02_share_next_9pct', 0):.1f}%</strong> de la conversación. "
    f"Sin embargo, los subgrupos pequeños y temáticos presentan menor Gini, actuando como 'salas seguras' donde participan voces menos dominantes.",
)

# =============================================================================
# P03: Group Health & Lifecycle Quadrant
# =============================================================================
render_question_header(
    "P03",
    "¿Cuál es la salud neta de crecimiento por grupo y en qué fase del ciclo de vida está cada uno?",
    "Crecimiento neto (Joins - Leaves - Kicks), tasa de churn, velocidad diaria y tendencia 14 días vs. 14 días previos.",
)
df_p03 = load_output_parquet("p03_group_health.parquet")

fig_quad = px.scatter(
    df_p03,
    x="net_growth",
    y="msgs_per_active_day",
    size="unique_members",
    color="lifecycle_stage",
    hover_name="group_label",
    hover_data=["macro_category", "joins", "leaves", "kicks", "churn_rate_pct", "trend_14d_pct"],
    color_discrete_sequence=CATEGORICAL_COLORS,
    size_max=48,
)
style_plotly(fig_quad, "Mapa de Salud de Grupos: Crecimiento Neto vs. Velocidad Diaria", height=420)
fig_quad.update_xaxes(title="Crecimiento Neto (Joins - Leaves - Kicks)")
fig_quad.update_yaxes(title="Mensajes / Día Activo", type="log")
st.plotly_chart(fig_quad, use_container_width=True)

with st.expander("📋 Tabla completa de salud y ciclo de vida de los 20 grupos"):
    st.dataframe(
        df_p03[["group_label", "macro_category", "lifecycle_stage", "unique_members", "msgs_per_active_day", "joins", "leaves", "kicks", "net_growth", "churn_rate_pct", "trend_14d_pct"]]
        .sort_values("msgs_per_active_day", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# P04 & P05 Side by Side
# =============================================================================
col_p4, col_p5 = st.columns([1.25, 1])

with col_p4:
    render_question_header(
        "P04",
        "¿Los picos de volumen aumentan la fricción y moderación?",
        "Correlación de Spearman entre volumen diario, mensajes eliminados y eventos de fricción.",
    )
    df_p04 = load_output_parquet("p04_moderation_daily.parquet")
    fig_mod = go.Figure()
    fig_mod.add_trace(go.Bar(x=df_p04["date_str"], y=df_p04["total_msgs"], name="Mensajes totales", marker_color="rgba(56, 189, 248, 0.28)"))
    fig_mod.add_trace(go.Scatter(x=df_p04["date_str"], y=df_p04["friction_events"], name="Fricción (deleted+kicks+leaves)", yaxis="y2", line=dict(color=PALETTE["danger"], width=2.2)))
    style_plotly(fig_mod, f"Volumen vs. Fricción Diaria (Spearman r = {s1.get('p04_spearman_vol_vs_friction', 0):.3f})", height=370)
    fig_mod.update_layout(yaxis=dict(title="Mensajes"), yaxis2=dict(title="Eventos de fricción", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig_mod, use_container_width=True)

with col_p5:
    render_question_header(
        "P05",
        "¿Cuál es la Tasa de Activación real (Lurkers vs. Activos)?",
        "Embudo progresivo desde ingresos al grupo matriz hasta Power Users (>=100 msgs).",
    )
    df_p05 = load_output_parquet("p05_activation_funnel.parquet")
    fig_fun = go.Figure(
        go.Funnel(
            y=df_p05["stage"],
            x=df_p05["count"],
            textinfo="value+percent initial",
            marker=dict(color=["#38BDF8", "#00D4AA", "#10B981", "#22C55E", "#A78BFA", "#F59E0B", "#F43F5E"][: len(df_p05)]),
        )
    )
    style_plotly(fig_fun, "Embudo de Activación Comunitaria", height=370)
    st.plotly_chart(fig_fun, use_container_width=True)
