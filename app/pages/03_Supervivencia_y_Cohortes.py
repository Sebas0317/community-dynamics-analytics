"""Page 03: Survival Analysis (Kaplan-Meier), Cohorts, Onboarding Hypotheses & Purge Impact (P10-P14)."""

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

from app.utils.charts import style_plotly
from app.utils.data_loader import load_output_json, load_output_parquet
from app.utils.styles import (
    PALETTE,
    render_hero,
    render_insight,
    render_question_header,
    render_sidebar_footer,
    setup_page,
)

setup_page("Supervivencia y Cohortes", icon="🧬")
exec_sum = load_output_json("executive_summary.json")
s2 = load_output_json("p10_p14_summary.json")
onb_stats = load_output_json("p12_onboarding_stats.json")
purge_stats = load_output_json("p14_purge_stats.json")
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Pilar 3 · Preguntas P10 – P14",
    title="Supervivencia, Retención por Cohortes y Pruebas de Hipótesis",
    subtitle="Aplicamos bioestadística de producto (Kaplan-Meier, Log-Rank Test y Mann-Whitney U) para medir cuánto dura un miembro, si el onboarding realmente salva retención y qué efecto tuvo la depuración de Halloween.",
)

# =============================================================================
# P10: Global Kaplan-Meier Survival Curve
# =============================================================================
render_question_header(
    "P10",
    "¿Cuánto 'vive' un miembro promedio en la comunidad y cuándo ocurre el abandono?",
    "Estimador de Kaplan-Meier (lifelines) con censura a la derecha (inactividad >=10d o salida explícita) e IC 95%.",
)
df_km = load_output_parquet("p10_kaplan_meier_global.parquet")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Supervivencia Mediana", f"{s2.get('p10_median_survival_days', 0):.1f} días", "50% sigue activo")
k2.metric("Supervivencia Día 7", f"{s2.get('p10_survival_d7_pct', 0):.1f}%")
k3.metric("Supervivencia Día 14", f"{s2.get('p10_survival_d14_pct', 0):.1f}%")
k4.metric("Supervivencia Día 30", f"{s2.get('p10_survival_d30_pct', 0):.1f}%")

fig_km = go.Figure()
fig_km.add_trace(
    go.Scatter(
        x=pd.concat([df_km["timeline_days"], df_km["timeline_days"][::-1]]),
        y=pd.concat([df_km["ci_upper"] * 100, (df_km["ci_lower"] * 100)[::-1]]),
        fill="toself",
        fillcolor="rgba(0, 212, 170, 0.14)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 95%",
        hoverinfo="skip",
    )
)
fig_km.add_trace(
    go.Scatter(
        x=df_km["timeline_days"],
        y=df_km["survival_prob"] * 100,
        mode="lines",
        line=dict(color=PALETTE["primary"], width=3, shape="hv"),
        name="Supervivencia Global (%)",
    )
)
fig_km.add_hline(y=50, line_dash="dash", line_color=PALETTE["accent"], annotation_text="Mediana (50%)")
style_plotly(fig_km, "Curva de Supervivencia Global de Kaplan-Meier", height=380)
fig_km.update_xaxes(title="Días desde el primer mensaje")
fig_km.update_yaxes(title="Probabilidad de seguir activo (%)", range=[0, 105])
st.plotly_chart(fig_km, use_container_width=True)

# =============================================================================
# P11: Weekly Cohort Retention Matrix
# =============================================================================
render_question_header(
    "P11",
    "¿Cómo evoluciona la retención por cohortes semanales de ingreso?",
    "Matriz de cohortes por semana ISO de primer mensaje (Semanas 0..N) + hitos D1, D7, D14, D30, D60.",
)
df_coh = load_output_parquet("p11_weekly_cohorts.parquet")
df_mil = load_output_parquet("p11_milestone_retention.parquet")

c11a, c11b = st.columns([2.2, 1])
with c11a:
    piv_coh = df_coh.pivot_table(index="cohort_week", columns="week_index", values="retention_pct")
    piv_coh.columns = [f"Sem {c}" for c in piv_coh.columns]
    fig_coh = px.imshow(
        piv_coh,
        text_auto=".0f",
        color_continuous_scale="Teal",
        aspect="auto",
        labels=dict(x="Semanas desde ingreso", y="Cohorte semanal", color="Retención (%)"),
    )
    style_plotly(fig_coh, "Heatmap de Retención por Cohortes Semanales (%)", height=390)
    st.plotly_chart(fig_coh, use_container_width=True)

with c11b:
    fig_mil = px.bar(
        df_mil,
        x=[f"D{d}" for d in df_mil["milestone_day"]],
        y="retention_pct",
        text=[f"{v:.1f}%" for v in df_mil["retention_pct"]],
        color_discrete_sequence=[PALETTE["secondary"]],
        hover_data=["eligible_members", "retained_members"],
    )
    style_plotly(fig_mil, "Retención por Hitos Elegibles (D1–D60)", height=390)
    fig_mil.update_yaxes(title="% Retenido", range=[0, 105])
    fig_mil.update_xaxes(title="Hito")
    st.plotly_chart(fig_mil, use_container_width=True)

# =============================================================================
# P12: Onboarding Hypothesis Test (Presentate!!!)
# =============================================================================
render_question_header(
    "P12",
    "Hipótesis de Onboarding: ¿Presentarse en 'Presentate!!!' predice retención a largo plazo?",
    "Curvas Kaplan-Meier estratificadas + Log-Rank Test + Pruebas Mann-Whitney U con tamaño de efecto rank-biserial.",
)
df_onb_km = load_output_parquet("p12_onboarding_km.parquet")

o1, o2, o3, o4 = st.columns(4)
o1.metric("Mediana CON Onboarding", f"{onb_stats.get('median_survival_with_onb', 0):.1f} días", f"n = {onb_stats.get('n_with_onboarding', 0)}")
o2.metric("Mediana SIN Onboarding", f"{onb_stats.get('median_survival_without_onb', 0):.1f} días", f"n = {onb_stats.get('n_without_onboarding', 0)}")
o3.metric("Log-Rank p-valor", f"{onb_stats.get('logrank_p_value', 1):.2e}", "Altamente significativo" if onb_stats.get("logrank_p_value", 1) < 0.001 else "No sig.")
mw_msgs = onb_stats.get("metrics_mann_whitney", {}).get("total_messages", {})
o4.metric("Mediana Mensajes (A vs B)", f"{mw_msgs.get('median_with_onboarding', 0):.0f} vs {mw_msgs.get('median_without_onboarding', 0):.0f}", f"p = {mw_msgs.get('p_value', 1):.2e}")

col_o1, col_o2 = st.columns([1.4, 1])
with col_o1:
    fig_okm = go.Figure()
    for grp_name, clr in [("Con Onboarding", PALETTE["primary"]), ("Sin Onboarding", PALETTE["danger"])]:
        sub = df_onb_km[df_onb_km["group"] == grp_name]
        fig_okm.add_trace(
            go.Scatter(
                x=sub["timeline_days"],
                y=sub["survival_prob"] * 100,
                mode="lines",
                line=dict(color=clr, width=3, shape="hv"),
                name=grp_name,
            )
        )
    style_plotly(fig_okm, "Supervivencia Kaplan-Meier: Con Onboarding vs. Sin Onboarding", height=370)
    fig_okm.update_xaxes(title="Días de vida")
    fig_okm.update_yaxes(title="Supervivencia (%)", range=[0, 105])
    st.plotly_chart(fig_okm, use_container_width=True)

with col_o2:
    mw_rows = []
    for m_name, vals in onb_stats.get("metrics_mann_whitney", {}).items():
        mw_rows.append(
            {
                "Métrica": m_name,
                "Mediana CON": vals["median_with_onboarding"],
                "Mediana SIN": vals["median_without_onboarding"],
                "p-valor": f"{vals['p_value']:.2e}",
                "Efecto |r|": abs(vals["effect_size_rb"]),
            }
        )
    st.markdown("##### Pruebas Mann-Whitney U (Diferencia de Distribuciones)")
    st.dataframe(pd.DataFrame(mw_rows), use_container_width=True, hide_index=True)
    render_insight(
        "Veredicto Estadístico (P12)",
        "Se rechaza contundentemente la hipótesis nula ($p < 10^{-23}$). Pasar por el canal de presentación es el mayor predictor temprano de retención, profundidad multigrupo y volumen de interacción.",
    )

# =============================================================================
# P13 & P14 Side by Side
# =============================================================================
c13, c14 = st.columns(2)

with c13:
    render_question_header(
        "P13",
        "Efecto 'Primera Impresión': ¿Recibir respuesta rápida (<10m) al llegar mejora la retención?",
        "Tiempo hasta la 1ra respuesta en el primer mensaje histórico del miembro vs. retención D7/D14.",
    )
    df_p13 = load_output_parquet("p13_first_impression.parquet")
    fig_p13 = go.Figure()
    fig_p13.add_trace(go.Bar(x=df_p13["reply_bucket"], y=df_p13["retention_d7_pct"], name="Retención D7 (%)", marker_color=PALETTE["secondary"]))
    fig_p13.add_trace(go.Bar(x=df_p13["reply_bucket"], y=df_p13["retention_d14_pct"], name="Retención D14 (%)", marker_color=PALETTE["primary"]))
    style_plotly(fig_p13, f"Retención según Velocidad de 1ra Respuesta (p = {s2.get('p13_lifespan_p_value', 1):.3f})", height=360)
    fig_p13.update_layout(barmode="group")
    fig_p13.update_yaxes(title="% Retenido", range=[0, 100])
    st.plotly_chart(fig_p13, use_container_width=True)
    st.caption(f"Muestra: {s2.get('p13_fast_reply_n', 0)} miembros recibieron respuesta en <=10 min (94%) vs. solo {s2.get('p13_slow_reply_n', 0)} tardíos — evidencia de altísima capacidad de acogida.")

with c14:
    render_question_header(
        "P14",
        "Impacto Before/After: ¿Qué provocó la depuración de inactivos de Halloween?",
        "Comparativa de ventanas [-14d, +14d] alrededor del 2026-10-31 en ecosistema global y grupo matriz.",
    )
    df_p14 = load_output_parquet("p14_purge_impact_daily.parquet")
    fig_p14 = go.Figure()
    fig_p14.add_trace(go.Scatter(x=df_p14["date_str"], y=df_p14["total_msgs_global"], name="Global (msgs/día)", line=dict(color=PALETTE["primary"], width=2.5)))
    fig_p14.add_trace(go.Scatter(x=df_p14["date_str"], y=df_p14["core_msgs"], name="Grupo Matriz", line=dict(color=PALETTE["secondary"], width=2)))
    fig_p14.add_vline(x=purge_stats.get("purge_date", "2026-10-31"), line_dash="dash", line_color=PALETTE["danger"])
    style_plotly(fig_p14, f"Ventana ±14d Depuración (Global {s2.get('p14_global_msgs_delta_pct', 0):+.1f}% | Matriz {s2.get('p14_core_msgs_delta_pct', 0):+.1f}%)", height=360)
    st.plotly_chart(fig_p14, use_container_width=True)
    st.caption(f"DAU global pasó de {purge_stats.get('dau_global', {}).get('before_mean', 0)} a {purge_stats.get('dau_global', {}).get('after_mean', 0)} (+{purge_stats.get('dau_global', {}).get('delta_pct', 0)}%, p = {purge_stats.get('dau_global', {}).get('p_value', 1):.3f}).")
