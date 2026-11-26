"""Home Page: Executive Portfolio Story & 36-Question Index."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.charts import style_plotly
from app.utils.data_loader import load_lakehouse_parquet, load_output_json, load_output_parquet
from app.utils.styles import PALETTE, render_hero, render_insight, render_sidebar_footer, setup_page

setup_page("Resumen Ejecutivo", icon="🧠")

summary = load_output_json("executive_summary.json")
vols = summary.get("volumes", {})
p1 = summary.get("p01_p09", {})
p2 = summary.get("p10_p14", {})
p3 = summary.get("p15_p19", {})
p4 = summary.get("p20_p29", {})
p5 = summary.get("p30_p36", {})

render_sidebar_footer(summary.get("date_range"))

render_hero(
    tag="Portafolio Data Analytics · Caso Real End-to-End · 36 Preguntas",
    title="Anatomía de una Comunidad Social: Datos, Grafos y Comportamiento Humano",
    subtitle=(
        "Estudio exhaustivo construido sobre una comunidad urbana real en Bogotá nacida para conectar desconocidos "
        "y transformar interacciones digitales en planes presenciales. Pipeline 100% reproducible en Python + DuckDB "
        "+ Parquet con privacidad irreversible Zero-PII (Ley 1581 de 2012)."
    ),
)

# === EXECUTIVE KPI STRIP ===
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Mensajes Analizados", f"{vols.get('messages_meta', 0):,}", f"{vols.get('groups_anon', 0)} grupos")
c2.metric("Miembros Únicos", f"{vols.get('members_anon', 0):,}", f"DAU prom: {p1.get('p01_avg_dau', 0)}")
c3.metric("Piezas Multimedia", f"{vols.get('media_meta', 0):,}", f"{p5.get('p34_unique_stickers', 0):,} stickers únicos")
c4.metric("Eventos Ciclo de Vida", f"{vols.get('events_anon', 0):,}", "Joins / Leaves / Kicks / Pins")
c5.metric("Planes Presenciales", f"{vols.get('plans_anon', 0):,}", f"ML ROC-AUC: {p4.get('p29_cv_roc_auc', 0)}")
c6.metric("Índice de Gini", f"{p1.get('p02_global_gini', 0):.3f}", f"Top 10% = {p1.get('p02_share_top_1pct', 0) + p1.get('p02_share_next_9pct', 0):.1f}%")

st.markdown("### 🔥 Los 6 Hallazgos Insignia del Estudio")
col_a, col_b, col_c = st.columns(3)

with col_a:
    render_insight(
        "1. El Onboarding multiplica ×12 la supervivencia (P12)",
        f"Quienes se presentan en el canal de bienvenida sobreviven una mediana de <strong>59.3 días</strong> vs. "
        f"<strong>4.9 días</strong> en quienes no lo hacen (Log-Rank p = <code>{p2.get('p12_onboarding_logrank_p', 0):.2e}</code>), "
        f"y escriben <strong>{p2.get('p12_median_msgs_with_onb', 0):.0f}</strong> msgs vs. <strong>{p2.get('p12_median_msgs_without_onb', 0):.0f}</strong>.",
    )
    render_insight(
        "4. Expansión vs. Canibalización (P20)",
        "Los subgrupos geográficos (ej. <strong>Zona Occidente</strong>) expandieron el ecosistema total (+250%), "
        "mientras que un canal genérico (<strong>Chat Libre Social</strong>) sí canibalizó el <strong>75%</strong> del volumen del grupo matriz.",
    )

with col_b:
    render_insight(
        "2. Comunidad genuinamente bidireccional (P17–P19)",
        f"Reciprocidad global del <strong>{p3.get('p19_global_reciprocity', 0) * 100:.1f}%</strong> y asortatividad "
        f"negativa (<strong>{p3.get('p17_degree_assortativity', 0):.3f}</strong>): los más activos responden a los nuevos en vez de aislarse en una élite.",
    )
    render_insight(
        "5. Meritocracia en Planes + ML (P27–P29)",
        f"El capital social del organizador casi no afecta la tracción (Spearman r = <strong>{p4.get('p27_spearman_betweenness_vs_traction', 0):.3f}</strong>), "
        f"pero fijar el mensaje (<code>pin</code>) dispara la respuesta (p = <code>{p4.get('p28_pin_impact_p', 0):.1e}</code>). Nuestro <strong>Random Forest</strong> predice éxito con ROC-AUC <strong>{p4.get('p29_cv_roc_auc', 0):.3f}</strong>.",
    )

with col_c:
    render_insight(
        "3. Regla 1-9-90 y Puentes Silenciosos (P02 & P16)",
        f"Apenas <strong>{p1.get('p02_pareto_80_members', 0)} personas</strong> producen el 80% de los mensajes (Gini {p1.get('p02_global_gini', 0):.3f}), "
        f"pero el grafo reveló <strong>{p3.get('p16_silent_bridges_count', 0)} Puentes Silenciosos</strong> que conectan subgrupos sin ser hiperactivos.",
    )
    render_insight(
        "6. Cultura Visual y Audios Ultracortos (P34–P35)",
        f"De <strong>11,103 stickers</strong>, el <strong>{p5.get('p34_sticker_reuse_rate_pct', 0):.1f}%</strong> son reutilizados y "
        f"<strong>{p5.get('p34_cross_group_viral_stickers', 0):,}</strong> viajaron entre múltiples grupos. Las notas de voz duran una mediana de solo <strong>{p5.get('p35_median_voice_note_sec', 0)}s</strong>.",
    )

# === MACRO TIMELINE OVERVIEW ===
st.markdown("### 📈 Pulso Histórico de la Comunidad")
df_daily = load_output_parquet("p01_daily_metrics.parquet")
if not df_daily.empty:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_daily["date_str"],
            y=df_daily["user_messages"],
            name="Mensajes diarios",
            marker_color="rgba(56, 189, 248, 0.35)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_daily["date_str"],
            y=df_daily["user_msgs_ma7"],
            name="Media móvil 7 días",
            line=dict(color=PALETTE["primary"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_daily["date_str"],
            y=df_daily["dau"],
            name="DAU (Miembros activos/día)",
            yaxis="y2",
            line=dict(color=PALETTE["accent"], width=2, dash="dot"),
        )
    )
    style_plotly(fig, "Volumen Conversacional Diario vs. Usuarios Activos Diarios (DAU)", height=390)
    fig.update_layout(
        yaxis=dict(title="Mensajes de usuario"),
        yaxis2=dict(title="DAU", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

# === 36-QUESTION CATALOG NAVIGATOR ===
st.markdown("### 🗺️ Mapa Completo de las 36 Preguntas Respondidas")
catalog_rows = [
    ("01 · Salud y Desigualdad", "P01–P05", "DAU/WAU/MAU, Stickiness, Curva de Lorenz / Gini, Salud Neta por grupo, Fricción/Moderación y Embudo de Activación."),
    ("02 · Cronobiología y Ritmos", "P06–P09", "Heatmap 7×24, Firmas horarias por arquetipo de grupo, Tiempo de Primera Respuesta (FRT) y Ráfagas / Resaca (ACF)."),
    ("03 · Supervivencia y Cohortes", "P10–P14", "Curvas Kaplan-Meier, Matriz de Cohortes Semanales, Hipótesis de Onboarding (Log-Rank), Primera Impresión (<10m) y Depuración Halloween."),
    ("04 · Grafos Sociales (SNA)", "P15–P19", "Comunidades Louvain (Grafo PyVis navegable), Puentes Silenciosos, Asortatividad, Simulador de Percolación (Bus Factor) y Reciprocidad."),
    ("05 · Ecosistema y Geografía", "P20–P24", "Canibalización vs. Expansión, User Journey (Markov / Sankey), Similitud Jaccard, Zonas de Bogotá y Multi-group Sweet Spot."),
    ("06 · Planes y Mundo Real", "P25–P29", "Tracción 2h/24h por categoría, Mejor Día×Hora, Capital Social vs. Meritocracia, Impacto de Pins y Simulador ML '¿Prenderá mi Plan?'."),
    ("07 · NLP y Cultura Multimedia", "P30–P35", "Sentimiento local, Estilo y Ráfagas, Temas Latentes (NMF), ADN Multimedia, Viralidad de Stickers (SHA-256) y Notas de Voz (.opus)."),
    ("08 · Arquetipos de Miembros", "P36", "Clustering ML (K-Means + PCA 2D) segmentando los 6 perfiles reales de comportamiento humano con radar comparativo."),
    ("09 · Metodología y Privacidad", "Gobernanza", "Arquitectura DuckDB/Parquet, Manifiesto de Auditoría Zero-PII (Ley 1581/2012) y descarga de tablas públicas agregadas."),
]
df_cat = pd.DataFrame(catalog_rows, columns=["Página en Sidebar", "Preguntas", "Alcance Analítico y Visualizaciones"])
st.dataframe(df_cat, use_container_width=True, hide_index=True)

df_groups = load_lakehouse_parquet("groups_anon.parquet")
with st.expander("📋 Ver Inventario Anonimizado de los 20 Grupos Analizados"):
    st.dataframe(
        df_groups[["group_id", "group_label", "macro_category", "bogota_zone", "user_messages", "unique_members", "active_days", "span_days"]]
        .sort_values("user_messages", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
