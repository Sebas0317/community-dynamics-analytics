"""Page 06: Plans Economy (Digital -> Real World) & Live ML Plan Success Simulator (P25-P29)."""

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
from app.utils.data_loader import load_output_json, load_output_parquet, load_plan_ml_model
from app.utils.styles import (
    CATEGORICAL_COLORS,
    PALETTE,
    render_hero,
    render_insight,
    render_question_header,
    render_sidebar_footer,
    setup_page,
)

setup_page("Planes y Mundo Real", icon="🎯")
exec_sum = load_output_json("executive_summary.json")
s4 = load_output_json("p20_p29_summary.json")
pin_stats = load_output_json("p28_pin_impact.json")
ml_metrics = load_output_json("p29_plan_ml_metrics.json")
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Pilar 6 · Preguntas P25 – P29 · Machine Learning Aplicado",
    title="Del Chat al Mundo Real: Economía de los Planes Presenciales",
    subtitle="Evaluamos las 1,104 convocatorias a planes midiendo tracción real a 2h y 24h, validamos si pesa más el capital social o fijar el mensaje, y desplegamos un modelo Random Forest interactivo para predecir el éxito de un plan.",
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Convocatorias Modeladas", f"{ml_metrics.get('total_plans_modeled', 0):,}", f"Alta tracción: {ml_metrics.get('positive_class_rate_pct', 0):.1f}%")
k2.metric("Efecto Mensaje Fijado (Pin)", f"{pin_stats.get('pinned_median_responders_2h', 0):.0f} vs {pin_stats.get('unpinned_median_responders_2h', 0):.0f}", f"p = {pin_stats.get('mann_whitney_p_value', 1):.1e}")
k3.metric("Efecto Capital Social (r)", f"{s4.get('p27_spearman_betweenness_vs_traction', 0):.3f}", "Comunidad meritocrática")
k4.metric(f"Modelo ML ({ml_metrics.get('selected_model', 'RF')})", f"ROC-AUC {ml_metrics.get('selected_cv_roc_auc', 0):.3f}", f"Precision {ml_metrics.get('selected_cv_precision', 0) * 100:.1f}%")

# =============================================================================
# P25 & P26: Plans by Category + Best Day x Time Block
# =============================================================================
c25, c26 = st.columns([1, 1.25])

with c25:
    render_question_header(
        "P25",
        "¿Qué categorías de planes generan mayor tracción real?",
        "Mediana de participantes únicos respondiendo en 2h y % de convocatorias de alta tracción.",
    )
    df_p25 = load_output_parquet("p25_plans_by_category.parquet").sort_values("high_traction_rate_pct", ascending=True)
    fig_p25 = px.bar(
        df_p25,
        x="high_traction_rate_pct",
        y="plan_category",
        orientation="h",
        color="median_responders_2h",
        color_continuous_scale="Tealgrn",
        text=[f"{v:.1f}% (n={n})" for v, n in zip(df_p25["high_traction_rate_pct"], df_p25["plans_count"])],
        hover_data=["plans_count", "median_responders_2h", "median_responders_24h"],
    )
    style_plotly(fig_p25, "Tasa de Alta Tracción (%) por Categoría de Plan", height=370)
    fig_p25.update_xaxes(title="% de planes de alta tracción", range=[0, 100])
    fig_p25.update_yaxes(title="")
    st.plotly_chart(fig_p25, use_container_width=True)

with c26:
    render_question_header(
        "P26",
        "¿Cuál es el mejor día y franja horaria para lanzar un plan?",
        "Tasa de éxito (% alta tracción) cruzada por Día de la semana × Franja horaria.",
    )
    df_p26 = load_output_parquet("p26_plans_heatmap_day_hour.parquet")
    block_order = ["Mañana (06–11h)", "Tarde (12–16h)", "Prime (17–20h)", "Noche (21–23h)", "Madrugada (00–05h)"]
    piv_p26 = df_p26.pivot_table(index="day_of_week", columns="time_block", values="high_traction_rate_pct", fill_value=0)
    piv_p26 = piv_p26.reindex(index=range(7), columns=[b for b in block_order if b in piv_p26.columns], fill_value=0)
    piv_p26.index = DAY_NAMES_ES
    fig_p26 = px.imshow(
        piv_p26,
        text_auto=".0f",
        color_continuous_scale="YlGnBu",
        aspect="auto",
        labels=dict(x="Franja Horaria", y="Día", color="Éxito (%)"),
    )
    style_plotly(fig_p26, "Probabilidad Histórica de Alta Tracción (%) Día × Franja", height=370)
    st.plotly_chart(fig_p26, use_container_width=True)

# =============================================================================
# P27 & P28: Social Capital vs Pins
# =============================================================================
c27, c28 = st.columns(2)

with c27:
    render_question_header(
        "P27",
        "¿Importa quién propone el plan (Capital Social) o es meritocrático?",
        "Tracción según cuartiles de Betweenness Centrality del iniciador + Spearman r.",
    )
    df_p27 = load_output_parquet("p27_initiator_social_capital.parquet")
    fig_p27 = px.bar(
        df_p27,
        x="social_capital_tier",
        y="median_responders_2h",
        color="high_traction_rate_pct",
        color_continuous_scale="Teal",
        text=[f"{v:.0f} pers." for v in df_p27["median_responders_2h"]],
    )
    style_plotly(fig_p27, f"Tracción 2h según Capital Social del Iniciador (r = {s4.get('p27_spearman_betweenness_vs_traction', 0):.3f})", height=350)
    fig_p27.update_xaxes(title="Cuartil de Centralidad del Organizador")
    fig_p27.update_yaxes(title="Mediana de personas respondiendo en 2h")
    st.plotly_chart(fig_p27, use_container_width=True)

with c28:
    render_question_header(
        "P28",
        "¿Fijar un mensaje (Pin) realmente multiplica el alcance de una convocatoria?",
        "Comparativa emparejada de participantes únicos en 2h (Con Pin ±60m vs. Sin Pin) + Mann-Whitney U.",
    )
    df_plans = load_output_parquet("p25_plans_enriched.parquet")
    df_plans["pin_label"] = df_plans["had_pin_near"].map({1: "Con Pin (±60m)", 0: "Sin Pin"})
    fig_p28 = px.box(
        df_plans,
        x="pin_label",
        y="unique_responders_2h",
        color="pin_label",
        color_discrete_map={"Con Pin (±60m)": PALETTE["primary"], "Sin Pin": PALETTE["slate"]},
    )
    style_plotly(fig_p28, f"Distribución de Respuestas Únicas en 2h (p = {pin_stats.get('mann_whitney_p_value', 1):.2e})", height=350)
    fig_p28.update_xaxes(title="")
    fig_p28.update_yaxes(title="Participantes únicos en 2h")
    st.plotly_chart(fig_p28, use_container_width=True)

# =============================================================================
# P29: Feature Importance + LIVE Interactive ML Simulator
# =============================================================================
render_question_header(
    "P29",
    "Modelo Predictivo ML: ¿Qué variables predicen que un plan prenda? + Simulador en Vivo",
    f"Validación cruzada 5-fold (LogReg AUC={ml_metrics.get('logistic_regression_cv_roc_auc', 0):.3f} vs. Random Forest AUC={ml_metrics.get('random_forest_cv_roc_auc', 0):.3f}) + Permutation Importance.",
)

df_fi = load_output_parquet("p29_feature_importance.parquet").sort_values("importance_mean", ascending=True)
col_fi, col_sim = st.columns([1.05, 1.25])

with col_fi:
    fig_fi = px.bar(
        df_fi,
        x="importance_mean",
        y="feature",
        error_x="importance_std",
        orientation="h",
        color_discrete_sequence=[PALETTE["primary"]],
    )
    style_plotly(fig_fi, "Importancia de Variables (Permutation Importance sobre ROC-AUC)", height=410)
    fig_fi.update_xaxes(title="Caída de ROC-AUC al permutar variable")
    fig_fi.update_yaxes(title="")
    st.plotly_chart(fig_fi, use_container_width=True)

with col_sim:
    st.markdown("#### 🔮 Simulador ML: *¿Prenderá mi Plan?*")
    model = load_plan_ml_model()
    if model is not None:
        sc1, sc2 = st.columns(2)
        with sc1:
            sim_cat = st.selectbox("Categoría del plan:", sorted(df_plans["plan_category"].dropna().unique().tolist()))
            sim_macro = st.selectbox("Tipo de grupo donde publicas:", sorted(df_plans["macro_category"].dropna().unique().tolist()))
            sim_day = st.selectbox("Día de la semana:", list(range(7)), format_func=lambda i: DAY_NAMES_ES[i], index=4)
            sim_hour = st.slider("Hora de publicación:", 0, 23, 19)
        with sc2:
            sim_pin = st.toggle("¿El plan será fijado (Pin)?", value=True)
            sim_fast = st.toggle("¿Recibe 1ra respuesta en < 5 min?", value=True)
            sim_tenure = st.slider("Antigüedad del promotor (días):", 0, 65, 25)
            sim_groups = st.slider("Grupos donde participa el promotor:", 1, 15, 4)

        X_sim = pd.DataFrame(
            [
                {
                    "hour": sim_hour,
                    "day_of_week": sim_day,
                    "initiator_betweenness": float(df_plans["initiator_betweenness"].median()),
                    "initiator_total_msgs": float(df_plans["initiator_total_msgs"].median()),
                    "initiator_groups_count": sim_groups,
                    "initiator_tenure_days": sim_tenure,
                    "had_pin_near": int(sim_pin),
                    "fast_first_reply_5m": int(sim_fast),
                    "plan_category": sim_cat,
                    "macro_category": sim_macro,
                }
            ]
        )
        prob = float(model.predict_proba(X_sim)[0, 1]) * 100.0

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Probabilidad de Alta Tracción (2h)", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PALETTE["primary"] if prob >= 55 else (PALETTE["accent"] if prob >= 35 else PALETTE["danger"])},
                    "steps": [
                        {"range": [0, 35], "color": "rgba(244, 63, 94, 0.18)"},
                        {"range": [35, 60], "color": "rgba(245, 158, 11, 0.18)"},
                        {"range": [60, 100], "color": "rgba(0, 212, 170, 0.18)"},
                    ],
                },
            )
        )
        style_plotly(fig_gauge, "", height=230)
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.warning("Modelo `p29_plan_model.joblib` no encontrado. Ejecuta el pipeline Fase 2.")
