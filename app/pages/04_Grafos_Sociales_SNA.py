"""Page 04: Social Network Analysis (SNA), Louvain Graph (PyVis), Silent Bridges & Percolation (P15-P19)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

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

setup_page("Grafos Sociales (SNA)", icon="🕸️")
exec_sum = load_output_json("executive_summary.json")
s3 = load_output_json("p15_p19_summary.json")
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Pilar 4 · Preguntas P15 – P19 · Ciencia de Redes",
    title="Teoría de Grafos, Comunidades Louvain y Resiliencia (Bus Factor)",
    subtitle="Reconstruimos la red social real a partir de secuencias de respuesta (Δt <= 180s) para revelar 'roscas' latentes, Puentes Silenciosos, apertura hacia recién llegados y fragilidad ante pérdida de líderes.",
)

df_nodes = load_output_parquet("p15_sna_nodes.parquet")
df_edges = load_output_parquet("p15_sna_edges.parquet")
df_comm = load_output_parquet("p15_louvain_communities.parquet")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Nodos Activos", f"{s3.get('p15_active_nodes', 0)}", f"{s3.get('p15_directed_edges', 0):,} aristas dirigidas")
k2.metric("Comunidades Louvain", f"{s3.get('p15_louvain_clusters', 0)}", f"Modularidad Q = {s3.get('p15_modularity_q', 0):.3f}")
k3.metric("Puentes Silenciosos", f"{s3.get('p16_silent_bridges_count', 0)}", "Bajo vol + Alto Betweenness")
k4.metric("Asortatividad Grado", f"{s3.get('p17_degree_assortativity', 0):.3f}", "Red abierta (<0)")
k5.metric("Reciprocidad Global", f"{s3.get('p19_global_reciprocity', 0) * 100:.1f}%", f"{s3.get('p19_mutual_dyads_pct', 0):.1f}% díadas mutuas")

# =============================================================================
# P15: Interactive Louvain Community Graph (PyVis)
# =============================================================================
render_question_header(
    "P15",
    "¿Cómo se estructuran realmente las 'roscas' o microcomunidades internas?",
    "Grafo dirigido ponderado (Δt <= 180s) + detección de comunidades Louvain + visualización interactiva PyVis.",
)

ctrl1, ctrl2 = st.columns(2)
with ctrl1:
    top_n_nodes = st.slider("Número de miembros más conectados a visualizar:", min_value=25, max_value=120, value=60, step=5)
with ctrl2:
    min_interactions = st.slider("Mínimo de interacciones por arista:", min_value=2, max_value=30, value=8, step=2)


@st.cache_data(show_spinner=False)
def build_pyvis_html(top_n: int, min_inter: int) -> str:
    sub_nodes = df_nodes.sort_values("pagerank", ascending=False).head(top_n)
    keep_ids = set(sub_nodes["member_id"])
    sub_edges = df_edges[
        (df_edges["source_member_id"].isin(keep_ids))
        & (df_edges["target_member_id"].isin(keep_ids))
        & (df_edges["interaction_count"] >= min_inter)
    ]

    net = Network(height="500px", width="100%", bgcolor="#0B0F19", font_color="#E2E8F0", directed=False)
    net.barnes_hut(gravity=-3500, central_gravity=0.35, spring_length=120, spring_strength=0.04, damping=0.18)

    for r in sub_nodes.itertuples(index=False):
        cid = int(r.louvain_cluster) if r.louvain_cluster >= 0 else 0
        color = CATEGORICAL_COLORS[cid % len(CATEGORICAL_COLORS)]
        size = float(np.clip(8 + np.sqrt(r.total_messages) * 0.45, 8, 42))
        tooltip = (
            f"{r.member_id}\n"
            f"Clúster Louvain: #{cid}\n"
            f"Mensajes: {r.total_messages:,} | Grupos: {r.groups_count}\n"
            f"Betweenness: {r.betweenness:.4f} | PageRank: {r.pagerank:.4f}\n"
            f"Rol SNA: {r.sna_quadrant}"
        )
        net.add_node(r.member_id, label=r.member_id, title=tooltip, color=color, size=size)

    for e in sub_edges.itertuples(index=False):
        width = float(np.clip(np.log1p(e.interaction_count) * 0.7, 0.5, 6.0))
        net.add_edge(e.source_member_id, e.target_member_id, value=width, color="rgba(148, 163, 184, 0.28)")

    return net.generate_html()


g_col, t_col = st.columns([2.2, 1])
with g_col:
    html_graph = build_pyvis_html(top_n_nodes, min_interactions)
    components.html(html_graph, height=520, scrolling=False)
    st.caption("Arrastra los nodos, haz zoom con la rueda del ratón y pasa el cursor sobre cualquier ID anónimo (`MEM_XXX`) para ver su centralidad.")

with t_col:
    st.markdown("##### Comunidades Louvain Detectadas")
    st.dataframe(
        df_comm[["louvain_cluster", "members_count", "total_messages", "internal_density", "dominant_group_label"]],
        use_container_width=True,
        hide_index=True,
    )
    render_insight(
        "Lectura de Modularidad (Q)",
        f"Una modularidad de <strong>Q = {s3.get('p15_modularity_q', 0):.3f}</strong> con {s3.get('p15_louvain_clusters', 0)} clústeres indica una red <strong>altamente integrada y permeable</strong>: existen subgrupos por afinidad, pero NO silos aislados que fracturen la comunidad.",
    )

# =============================================================================
# P16: Volume vs Betweenness Centrality (Silent Bridges)
# =============================================================================
render_question_header(
    "P16",
    "¿Hablar mucho equivale a conectar a la comunidad? (Búsqueda de Puentes Silenciosos)",
    "Correlación Spearman (Volumen vs. Betweenness y PageRank) + segmentación por cuadrantes P75.",
)
active_nodes = df_nodes[df_nodes["degree_undirected"] > 0].copy()

fig_sb = px.scatter(
    active_nodes,
    x="total_messages",
    y="betweenness",
    color="sna_quadrant",
    size="groups_count",
    hover_name="member_id",
    hover_data=["total_messages", "groups_count", "pagerank", "primary_group_label"],
    color_discrete_map={
        "Conector Clave (Alto Vol + Alto Puente)": PALETTE["primary"],
        "Puente Silencioso (Bajo Vol + Alto Puente)": PALETTE["accent"],
        "Megáfono Local (Alto Vol + Bajo Puente)": PALETTE["danger"],
        "Participante Regular / Periferia": "#64748B",
    },
    size_max=28,
)
style_plotly(
    fig_sb,
    f"Volumen de Mensajes vs. Betweenness Centrality (Spearman r = {s3.get('p16_spearman_vol_vs_betweenness', 0):.3f})",
    height=420,
)
fig_sb.update_xaxes(title="Total de mensajes (escala log)", type="log")
fig_sb.update_yaxes(title="Betweenness Centrality (poder conector)")
st.plotly_chart(fig_sb, use_container_width=True)

with st.expander(f"🌉 Ver los {s3.get('p16_silent_bridges_count', 0)} Puentes Silenciosos identificados (IDs anónimos)"):
    sb_df = active_nodes[active_nodes["sna_quadrant"].str.contains("Puente Silencioso")].sort_values("betweenness", ascending=False)
    st.dataframe(
        sb_df[["member_id", "total_messages", "groups_count", "betweenness", "pagerank", "primary_group_label"]],
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# P17 & P19 Side by Side
# =============================================================================
c17, c19 = st.columns(2)

with c17:
    render_question_header(
        "P17",
        "¿Qué tan abierta o endogámica es la comunidad hacia los nuevos?",
        "Asortatividad de grado + Matriz de mezcla entre Fundadores (D0–14), Intermedios (D15–40) y Nuevos (D41+).",
    )
    df_mix = load_output_parquet("p17_assortativity_mixing.parquet")
    piv_mix = df_mix.pivot_table(index="source_tier", columns="target_tier", values="share_of_source_pct", fill_value=0)
    fig_mix = px.imshow(
        piv_mix,
        text_auto=".1f",
        color_continuous_scale="Blues",
        labels=dict(x="A quién responde (Target)", y="Quién inicia (Source)", color="% Interacciones"),
    )
    style_plotly(fig_mix, f"Matriz de Mezcla por Antigüedad (Clustering = {s3.get('p17_avg_clustering', 0):.3f})", height=360)
    st.plotly_chart(fig_mix, use_container_width=True)

with c19:
    render_question_header(
        "P19",
        "¿Predominan relaciones recíprocas o asimétricas?",
        "Reciprocidad global de díadas dirigidas y distribución del índice de simetría min(A->B, B->A) / max.",
    )
    df_dyads = load_output_parquet("p19_reciprocity_dyads.parquet")
    mutual = df_dyads[df_dyads["is_mutual"] == 1]
    fig_rec = px.histogram(
        mutual,
        x="symmetry_score",
        nbins=20,
        color_discrete_sequence=[PALETTE["primary"]],
    )
    style_plotly(fig_rec, f"Distribución de Simetría en Díadas Mutuas (Reciprocidad = {s3.get('p19_global_reciprocity', 0) * 100:.1f}%)", height=360)
    fig_rec.update_xaxes(title="Índice de Simetría (1.0 = conversación 50/50)")
    fig_rec.update_yaxes(title="Cantidad de pares de miembros")
    st.plotly_chart(fig_rec, use_container_width=True)

# =============================================================================
# P18: Interactive Percolation Simulator (Bus Factor)
# =============================================================================
render_question_header(
    "P18",
    "Simulador de Resiliencia (Percolación / Bus Factor): ¿Qué pasa si se van los líderes?",
    "Simulación algorítmica removiendo del 0% al 25% de nodos: Aleatorio (15x Monte Carlo) vs. Top Volumen vs. Top Betweenness.",
)
df_perc = load_output_parquet("p18_percolation_simulation.parquet")

rem_pct_sel = st.select_slider(
    "🎛️ Simula la salida simultánea de este porcentaje de miembros:",
    options=sorted(df_perc["removed_pct"].unique().tolist()),
    value=10,
    format_func=lambda x: f"{x}% de los miembros",
)
snap = df_perc[df_perc["removed_pct"] == rem_pct_sel].set_index("strategy")
p_cols = st.columns(3)
for idx, strat in enumerate(["Remoción Aleatoria (Promedio 15x)", "Top Volumen (Habladores)", "Top Betweenness (Puentes)"]):
    if strat in snap.index:
        row_s = snap.loc[strat]
        p_cols[idx].metric(
            strat,
            f"{row_s['lcc_pct_of_original']:.1f}% red conectada",
            f"{row_s['edges_remaining_pct']:.1f}% aristas vivas ({int(row_s['removed_nodes'])} nodos fuera)",
        )

fig_perc = px.line(
    df_perc,
    x="removed_pct",
    y="lcc_pct_of_original",
    color="strategy",
    markers=True,
    color_discrete_map={
        "Remoción Aleatoria (Promedio 15x)": PALETTE["secondary"],
        "Top Volumen (Habladores)": PALETTE["accent"],
        "Top Betweenness (Puentes)": PALETTE["danger"],
    },
)
fig_perc.add_vline(x=rem_pct_sel, line_dash="dot", line_color=PALETTE["primary"])
style_plotly(fig_perc, "Curva de Colapso del Componente Gigante (Percolación)", height=380)
fig_perc.update_xaxes(title="% de miembros removidos")
fig_perc.update_yaxes(title="% del Componente Gigante conectado", range=[0, 105])
st.plotly_chart(fig_perc, use_container_width=True)
