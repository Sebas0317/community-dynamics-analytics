"""Page 02: Chronobiology, Urban Rhythms, First Response Time & Bursts (P06-P09)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.charts import DAY_NAMES_ES, style_plotly
from app.utils.data_loader import load_lakehouse_parquet, load_output_json, load_output_parquet
from app.utils.styles import (
    CATEGORICAL_COLORS,
    PALETTE,
    render_hero,
    render_insight,
    render_question_header,
    render_sidebar_footer,
    setup_page,
)

setup_page("Cronobiología y Ritmos", icon="⏰")
exec_sum = load_output_json("executive_summary.json")
s1 = load_output_json("p01_p09_summary.json")
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Pilar 2 · Preguntas P06 – P09",
    title="Cronobiología, Ritmos Urbanos y Velocidad Conversacional",
    subtitle="Analizamos cuándo respira la comunidad en Bogotá, cómo difieren los ritmos entre grupos nocturnos, deportivos y zonales, qué tan rápido se responde un mensaje (FRT) y si los picos generan 'resaca conversacional'.",
)

# =============================================================================
# P06: 7x24 Chronobiology Heatmap (Filterable by Group)
# =============================================================================
render_question_header(
    "P06",
    "¿Cuál es el 'reloj biológico' colectivo de la comunidad?",
    "Matriz de densidad Hora (0–23) × Día de la semana (Lun–Dom) filtrable por grupo o métrica.",
)
df_hm = load_output_parquet("p06_heatmap_7x24.parquet")
df_groups = load_lakehouse_parquet("groups_anon.parquet")
label_map = dict(zip(df_groups["group_id"], df_groups["group_label"]))

f1, f2 = st.columns([2, 1])
with f1:
    group_options = ["🌐 Ecosistema Completo (20 grupos)"] + sorted(label_map.values())
    selected_grp_lbl = st.selectbox("Filtrar Heatmap por grupo:", group_options)
with f2:
    metric_choice = st.radio("Métrica:", ["Mensajes", "Miembros únicos"], horizontal=True)

if selected_grp_lbl.startswith("🌐"):
    hm_sub = df_hm.groupby(["day_of_week", "hour"], as_index=False).agg(
        messages=("messages", "sum"), unique_members=("unique_members", "sum")
    )
else:
    inv_map = {v: k for k, v in label_map.items()}
    gid = inv_map[selected_grp_lbl]
    hm_sub = df_hm[df_hm["group_id"] == gid]

val_col = "messages" if metric_choice == "Mensajes" else "unique_members"
piv = hm_sub.pivot_table(index="day_of_week", columns="hour", values=val_col, fill_value=0).reindex(index=range(7), columns=range(24), fill_value=0)
piv.index = DAY_NAMES_ES

fig_hm = px.imshow(
    piv,
    labels=dict(x="Hora del día (Bogotá UTC-5)", y="Día", color=metric_choice),
    color_continuous_scale="Tealgrn",
    aspect="auto",
)
style_plotly(fig_hm, f"Densidad 7×24 — {selected_grp_lbl}", height=360)
fig_hm.update_xaxes(dtick=1)
st.plotly_chart(fig_hm, use_container_width=True)

# =============================================================================
# P07: Circadian Profiles by Group Archetype
# =============================================================================
render_question_header(
    "P07",
    "¿Cómo cambian las firmas horarias según el propósito del grupo?",
    "Perfiles circadianos normalizados (% de actividad diaria por hora) y distribución por franjas horarias.",
)
df_circ = load_output_parquet("p07_group_circadian_profiles.parquet")
df_csum = load_output_parquet("p07_circadian_summary.parquet")

col_c1, col_c2 = st.columns([1.3, 1])
with col_c1:
    macro_circ = df_circ.groupby(["macro_category", "hour"], as_index=False)["messages"].sum()
    macro_circ["share_pct"] = macro_circ["messages"] / macro_circ.groupby("macro_category")["messages"].transform("sum") * 100
    fig_circ = px.line(
        macro_circ,
        x="hour",
        y="share_pct",
        color="macro_category",
        markers=True,
        color_discrete_sequence=CATEGORICAL_COLORS,
    )
    style_plotly(fig_circ, "Firma Circadiana Normalizada por Macro-Categoría", height=390)
    fig_circ.update_xaxes(title="Hora (0–23h)", dtick=2)
    fig_circ.update_yaxes(title="% de la actividad del día")
    st.plotly_chart(fig_circ, use_container_width=True)

with col_c2:
    fig_night = px.bar(
        df_csum.sort_values("late_night_share_pct", ascending=True),
        x=["morning_share_pct", "afternoon_share_pct", "prime_evening_share_pct", "late_night_share_pct"],
        y="group_label",
        orientation="h",
        color_discrete_sequence=["#38BDF8", "#00D4AA", "#F59E0B", "#A78BFA"],
    )
    style_plotly(fig_night, "Distribución por Franja Horaria (Mañana / Tarde / Prime / Madrugada)", height=390)
    fig_night.update_xaxes(title="% del volumen")
    fig_night.update_yaxes(title="")
    st.plotly_chart(fig_night, use_container_width=True)

# =============================================================================
# P08: First Response Time (FRT)
# =============================================================================
render_question_header(
    "P08",
    "¿Qué tan rápido responde la comunidad y cuáles son las 'horas muertas'?",
    "Función de ventana LAG() en DuckDB midiendo delta temporal entre emisores consecutivos distintos (<=2h).",
)
df_frt_hr = load_output_parquet("p08_frt_by_hour.parquet")
df_frt_grp = load_output_parquet("p08_frt_by_group.parquet")

c8a, c8b = st.columns(2)
with c8a:
    fig_fhr = go.Figure()
    fig_fhr.add_trace(go.Scatter(x=df_frt_hr["hour"], y=df_frt_hr["p75_frt_sec"] / 60, name="P75 (min)", line=dict(color=PALETTE["accent"], dash="dot")))
    fig_fhr.add_trace(go.Scatter(x=df_frt_hr["hour"], y=df_frt_hr["mean_frt_sec"] / 60, name="Promedio (min)", line=dict(color=PALETTE["secondary"], width=2.5)))
    fig_fhr.add_trace(go.Scatter(x=df_frt_hr["hour"], y=df_frt_hr["median_frt_sec"] / 60, name="Mediana (min)", line=dict(color=PALETTE["primary"], width=3)))
    style_plotly(fig_fhr, "Tiempo de Primera Respuesta (FRT) según Hora del Día", height=370)
    fig_fhr.update_xaxes(title="Hora (0–23h)", dtick=2)
    fig_fhr.update_yaxes(title="Minutos hasta la respuesta")
    st.plotly_chart(fig_fhr, use_container_width=True)

with c8b:
    fig_fgrp = px.bar(
        df_frt_grp.sort_values("pct_under_5min", ascending=True),
        x="pct_under_5min",
        y="group_label",
        color="macro_category",
        orientation="h",
        color_discrete_sequence=CATEGORICAL_COLORS,
        hover_data=["reply_pairs", "median_frt_sec", "mean_frt_sec"],
    )
    style_plotly(fig_fgrp, "% de Mensajes Respondidos en < 5 Minutos por Grupo", height=370)
    fig_fgrp.update_xaxes(title="% respondidos en <= 5 min", range=[0, 100])
    fig_fgrp.update_yaxes(title="")
    st.plotly_chart(fig_fgrp, use_container_width=True)

# =============================================================================
# P09: Activity Bursts, Hangover Effect & ACF
# =============================================================================
render_question_header(
    "P09",
    "¿Las rachas de hiperactividad generan 'resaca conversacional' al día siguiente?",
    "Detección de días anómalos (Z-score >= 1.5), trayectoria de decaimiento post-pico y Autocorrelación (ACF).",
)
df_burst = load_output_parquet("p09_bursts_daily.parquet")
df_hang = load_output_parquet("p09_burst_hangover.parquet")
df_acf = load_output_parquet("p09_acf.parquet")

c9a, c9b = st.columns(2)
with c9a:
    fig_hang = go.Figure()
    fig_hang.add_trace(
        go.Bar(
            x=[f"t{d:+d}d" if d != 0 else "Día Pico (t0)" for d in df_hang["day_offset"]],
            y=df_hang["avg_pct_of_burst"],
            marker_color=[PALETTE["secondary"] if d < 0 else (PALETTE["danger"] if d == 0 else PALETTE["primary"]) for d in df_hang["day_offset"]],
            text=[f"{v:.1f}%" for v in df_hang["avg_pct_of_burst"]],
            textposition="outside",
        )
    )
    style_plotly(fig_hang, "Trayectoria Promedio Antes y Después de un Día Pico (t0 = 100%)", height=360)
    fig_hang.update_yaxes(title="% del volumen del día pico", range=[0, 118])
    st.plotly_chart(fig_hang, use_container_width=True)

with c9b:
    fig_acf = px.bar(df_acf, x="lag_days", y="acf", color_discrete_sequence=[PALETTE["purple"]])
    fig_acf.add_hline(y=0, line_color=PALETTE["slate"])
    style_plotly(fig_acf, f"Autocorrelación Temporal (ACF Lag 1 = {s1.get('p09_acf_lag1', 0):.3f})", height=360)
    fig_acf.update_xaxes(title="Rezago (días)", dtick=1)
    fig_acf.update_yaxes(title="Coeficiente ACF")
    st.plotly_chart(fig_acf, use_container_width=True)

render_insight(
    "Diagnóstico de Inercia y Resaca (P09)",
    f"Con un <strong>ACF Lag-1 = {s1.get('p09_acf_lag1', 0):.3f}</strong>, la energía de la comunidad tiene inercia positiva de un día al siguiente, "
    f"pero tras un día pico extremo (Z >= 1.5, detectados <strong>{s1.get('p09_burst_days_count', 0)} días</strong>), el volumen al día siguiente cae en promedio al "
    f"<strong>{s1.get('p09_next_day_pct_of_burst', 0):.1f}%</strong> del pico — una corrección natural de 'resaca conversacional' sin colapsar la actividad base.",
)
