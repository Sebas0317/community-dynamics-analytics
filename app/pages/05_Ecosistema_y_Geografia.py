"""Page 05: Multigroup Ecosystem, Cannibalization, Markov Sankey, Jaccard & Bogota Geography (P20-P24)."""

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
    CATEGORICAL_COLORS,
    PALETTE,
    render_hero,
    render_insight,
    render_question_header,
    render_sidebar_footer,
    setup_page,
)

setup_page("Ecosistema y Geografía", icon="🗺️")
exec_sum = load_output_json("executive_summary.json")
s4 = load_output_json("p20_p29_summary.json")
cannib = load_output_json("p20_cannibalization_events.json")
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Pilar 5 · Preguntas P20 – P24",
    title="Dinámica Multigrupo, Geografía Urbana y Movilidad Interna",
    subtitle="Evaluamos si los subgrupos expanden o canibalizan al grupo matriz, trazamos el recorrido real (User Journey) con cadenas de Markov, medimos afinidad Jaccard y comparamos las zonas de Bogotá.",
)

# =============================================================================
# P20: Cannibalization vs Expansion
# =============================================================================
render_question_header(
    "P20",
    "¿Los subgrupos especializados canibalizan al grupo matriz o hacen crecer el ecosistema?",
    "Cuota diaria de participación (Share of Voice) + estudio de eventos (±7 días) en lanzamientos explosivos.",
)
df_sov = load_output_parquet("p20_share_of_voice_daily.parquet")

col_20a, col_20b = st.columns([1.8, 1])
with col_20a:
    fig_sov = go.Figure()
    fig_sov.add_trace(
        go.Scatter(
            x=df_sov["date_str"], y=df_sov["core_msgs"], stackgroup="one",
            name="Grupo Matriz (Core)", line=dict(color=PALETTE["secondary"]),
        )
    )
    fig_sov.add_trace(
        go.Scatter(
            x=df_sov["date_str"], y=df_sov["subgroups_msgs"], stackgroup="one",
            name="Subgrupos (19 canales)", line=dict(color=PALETTE["primary"]),
        )
    )
    style_plotly(
        fig_sov,
        f"Volumen Apilado: Grupo Matriz vs. Subgrupos (Spearman r = {cannib.get('spearman_core_vs_subgroups', 0):.3f})",
        height=380,
    )
    st.plotly_chart(fig_sov, use_container_width=True)

with col_20b:
    st.markdown("##### Estudios de Lanzamiento (±7d)")
    for ev in cannib.get("launch_studies", []):
        st.metric(
            f"{ev['group_label']} ({ev['launch_date']})",
            ev["verdict"],
            f"Matriz {ev['core_delta_pct']:+.1f}% | Total {ev['total_delta_pct']:+.1f}%",
        )
    render_insight(
        "Regla de Arquitectura Comunitaria",
        "Los subgrupos con propósito diferenciado (como <strong>Zona Occidente</strong>) multiplican la actividad total; los canales redundantes sin foco (como <strong>Chat Libre</strong>) roban conversación directa del canal principal.",
    )

# =============================================================================
# P21: User Journey (Markov Transitions -> Interactive Sankey)
# =============================================================================
render_question_header(
    "P21",
    "¿Cuál es el 'User Journey' típico? ¿En qué orden migran los miembros entre grupos?",
    "Secuencias cronológicas del primer mensaje en cada grupo + Matriz de Transición de Markov (1er orden).",
)
df_tr = load_output_parquet("p21_markov_transitions.parquet")
df_mtr = load_output_parquet("p21_markov_macro_transitions.parquet")

view_sankey = st.radio("Nivel de detalle del flujo:", ["Top 18 Flujos entre Grupos", "Flujos entre Macro-Categorías"], horizontal=True)

if view_sankey.startswith("Top"):
    sub_tr = df_tr.head(18).copy()
    src_col, tgt_col = "source_group", "target_group"
else:
    sub_tr = df_mtr.head(18).copy()
    src_col, tgt_col = "source_category", "target_category"

labels_s = list(dict.fromkeys(sub_tr[src_col].tolist() + sub_tr[tgt_col].tolist()))
idx_map = {lbl: i for i, lbl in enumerate(labels_s)}

fig_sankey = go.Figure(
    go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=20,
            line=dict(color=PALETTE["bg"], width=1),
            label=labels_s,
            color=[CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] for i in range(len(labels_s))],
        ),
        link=dict(
            source=[idx_map[s] for s in sub_tr[src_col]],
            target=[idx_map[t] for t in sub_tr[tgt_col]],
            value=sub_tr["transitions"].tolist(),
            color="rgba(56, 189, 248, 0.28)",
        ),
    )
)
style_plotly(fig_sankey, f"Diagrama Sankey de Migración de Miembros ({view_sankey})", height=440)
st.plotly_chart(fig_sankey, use_container_width=True)

# =============================================================================
# P22: Jaccard Similarity Matrix
# =============================================================================
render_question_header(
    "P22",
    "¿Qué grupos comparten más audiencia y forman constelaciones de afinidad?",
    "Índice de Similitud de Jaccard |A ∩ B| / |A ∪ B| entre los 20 grupos del ecosistema.",
)
df_jacc = load_output_parquet("p22_jaccard_matrix.parquet")
piv_j = df_jacc.pivot_table(index="label_a", columns="label_b", values="jaccard", fill_value=0)
fig_jacc = px.imshow(
    piv_j,
    color_continuous_scale="Viridis",
    aspect="auto",
    labels=dict(x="", y="", color="Jaccard"),
)
style_plotly(fig_jacc, f"Matriz de Solapamiento de Audiencia (Top par: {s4.get('p22_top_jaccard_pair', '')})", height=520)
st.plotly_chart(fig_jacc, use_container_width=True)

# =============================================================================
# P23 & P24 Side by Side
# =============================================================================
c23, c24 = st.columns(2)

with c23:
    render_question_header(
        "P23",
        "Geografía Urbana (Bogotá): ¿Los grupos zonales cohesionan más que los temáticos?",
        "Comparativa de velocidad diaria, Gini y churn entre zonas de Bogotá y macro-categorías.",
    )
    df_geo = load_output_parquet("p23_geography_vs_thematic.parquet")
    geo_only = df_geo[df_geo["macro_category"] == "Geográfico"].sort_values("msgs_per_active_day", ascending=False)
    fig_geo = px.bar(
        geo_only,
        x="group_label",
        y="msgs_per_active_day",
        color="bogota_zone",
        text="msgs_per_active_day",
        hover_data=["unique_members", "gini", "churn_rate_pct", "median_frt_sec"],
        color_discrete_sequence=CATEGORICAL_COLORS,
    )
    style_plotly(fig_geo, "Velocidad Diaria en Grupos Zonales de Bogotá (msgs/día activo)", height=360)
    fig_geo.update_xaxes(title="")
    fig_geo.update_yaxes(title="Mensajes / día activo")
    st.plotly_chart(fig_geo, use_container_width=True)

with c24:
    render_question_header(
        "P24",
        "¿Existe un 'Multi-group Sweet Spot' antes de la fatiga por notificaciones?",
        "Relación entre cantidad de grupos donde participa un miembro y su retención D14 / vida media.",
    )
    df_sweet = load_output_parquet("p24_multigroup_sweetspot.parquet")
    fig_sw = go.Figure()
    fig_sw.add_trace(go.Bar(x=df_sweet["groups_bucket"], y=df_sweet["members"], name="Miembros", marker_color="rgba(56, 189, 248, 0.30)"))
    fig_sw.add_trace(
        go.Scatter(
            x=df_sweet["groups_bucket"],
            y=df_sweet["retention_d14_pct"],
            name="Retención D14 (%)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color=PALETTE["primary"], width=3),
            marker=dict(size=9),
        )
    )
    style_plotly(fig_sw, "Multi-group Sweet Spot: Cantidad de Grupos vs. Retención D14", height=360)
    fig_sw.update_layout(
        yaxis=dict(title="Miembros en bucket"),
        yaxis2=dict(title="Retención D14 (%)", overlaying="y", side="right", range=[0, 105], showgrid=False),
    )
    st.plotly_chart(fig_sw, use_container_width=True)
