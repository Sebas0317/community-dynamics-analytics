"""Page 08: Unsupervised Human Archetypes Segmentation (K-Means + PCA 2D) (P36)."""

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

setup_page("Arquetipos de Miembros", icon="🧩")
exec_sum = load_output_json("executive_summary.json")
c_meta = load_output_json("p36_clustering_metrics.json")
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Pilar 8 · Pregunta P36 · Machine Learning No Supervisado",
    title="Segmentación de Arquetipos Humanos en la Comunidad",
    subtitle="Combinamos 14 variables de comportamiento (volumen, longevidad, multigrupo, centralidad SNA, planes convocados, sentimiento y mix multimedia) con K-Means + PCA 2D para descubrir los perfiles reales que sostienen la comunidad.",
)

df_mem = load_output_parquet("p36_member_archetypes.parquet")
df_prof = load_output_parquet("p36_archetype_profiles.parquet")

render_question_header(
    "P36",
    "¿Qué arquetipos o perfiles de personas emergen naturalmente al cruzar todas las dimensiones?",
    f"K-Means (K={c_meta.get('k_selected', 6)}, Silhouette={c_meta.get('selected_silhouette', 0):.3f}) sobre 14 features estandarizadas + proyección PCA 2D.",
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Arquetipos Descubiertos", f"K = {c_meta.get('k_selected', 6)}", f"Silhouette: {c_meta.get('selected_silhouette', 0):.3f}")
pca_v = c_meta.get("pca_explained_variance_pct", [0, 0])
k2.metric("Varianza Explicada PCA 2D", f"{sum(pca_v):.1f}%", f"PC1 {pca_v[0]:.1f}% + PC2 {pca_v[1]:.1f}%")
k3.metric("Miembros Segmentados", f"{len(df_mem):,}", "100% IDs anónimos MEM_XXX")
top_arch = df_prof.sort_values("members", ascending=False).iloc[0]
k4.metric("Arquetipo Más Numeroso", f"{int(top_arch['members'])} pers.", str(top_arch["archetype_name"])[:28])

col_pca, col_rad = st.columns([1.4, 1])

with col_pca:
    fig_pca = px.scatter(
        df_mem,
        x="pca_1",
        y="pca_2",
        color="archetype_name",
        size=np.clip(np.log1p(df_mem["total_messages"]), 1, 10),
        hover_name="member_id",
        hover_data=["total_messages", "groups_count", "lifespan_days", "plans_initiated", "betweenness", "media_ratio"],
        color_discrete_sequence=CATEGORICAL_COLORS,
        size_max=22,
    )
    style_plotly(fig_pca, "Mapa 2D de Comportamiento Humano (Proyección PCA)", height=450)
    fig_pca.update_xaxes(title=f"Componente Principal 1 ({pca_v[0]:.1f}% var)")
    fig_pca.update_yaxes(title=f"Componente Principal 2 ({pca_v[1]:.1f}% var)")
    st.plotly_chart(fig_pca, use_container_width=True)

with col_rad:
    radar_cols = ["avg_msgs", "avg_groups", "avg_lifespan", "avg_plans", "avg_media_ratio", "avg_words"]
    radar_labels = ["Volumen", "Multigrupo", "Longevidad", "Planes", "Multimedia", "Extensión"]
    norm_prof = df_prof.copy()
    for c in radar_cols:
        mx = float(norm_prof[c].max())
        norm_prof[c] = norm_prof[c] / mx if mx > 0 else 0.0

    sel_archs = st.multiselect(
        "Comparar huella de arquetipos en Radar:",
        options=df_prof["archetype_name"].tolist(),
        default=df_prof["archetype_name"].tolist()[:4],
    )
    fig_rad = go.Figure()
    for i, row in norm_prof[norm_prof["archetype_name"].isin(sel_archs)].iterrows():
        vals = [row[c] for c in radar_cols] + [row[radar_cols[0]]]
        fig_rad.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=f"{row['archetype_name']} (n={int(row['members'])})",
                line_color=CATEGORICAL_COLORS[int(i) % len(CATEGORICAL_COLORS)],
            )
        )
    style_plotly(fig_rad, "Huella Normalizada por Arquetipo (0–1)", height=390)
    fig_rad.update_layout(polar=dict(bgcolor="rgba(11,15,25,0.6)", radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig_rad, use_container_width=True)

st.markdown("### 📋 Perfil Cuantitativo de Cada Arquetipo")
st.dataframe(
    df_prof[["archetype_name", "members", "avg_msgs", "median_msgs", "avg_groups", "avg_lifespan", "avg_plans", "avg_media_ratio", "avg_words", "avg_sentiment"]],
    use_container_width=True,
    hide_index=True,
)

# Interactive Member Explorer by Archetype
sel_arch_tbl = st.selectbox("🔎 Explorar miembros anónimos de un arquetipo:", df_prof["archetype_name"].tolist())
sub_m = df_mem[df_mem["archetype_name"] == sel_arch_tbl].sort_values("total_messages", ascending=False)
st.dataframe(
    sub_m[["member_id", "total_messages", "groups_count", "active_days", "lifespan_days", "plans_initiated", "betweenness", "media_ratio", "primary_group_label"]].head(25),
    use_container_width=True,
    hide_index=True,
)

render_insight(
    "Conclusión de Segmentación (P36)",
    "Una comunidad social viva no depende de un solo tipo de miembro: requiere un <strong>Núcleo Motor</strong> que mantenga el pulso diario, una capa media amplia de <strong>Organizadores de Planes</strong> que conviertan chat en vida real, y <strong>Especialistas de Nicho</strong> que den profundidad a los subgrupos temáticos.",
)
