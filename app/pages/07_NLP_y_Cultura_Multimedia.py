"""Page 07: Conversational Psychology, Local NLP, Topics & Multimodal Intelligence (P30-P35)."""

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

setup_page("NLP y Cultura Multimedia", icon="🎭")
exec_sum = load_output_json("executive_summary.json")
s5 = load_output_json("p30_p36_summary.json")
stk_sum = load_output_json("p34_sticker_summary.json")
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Pilares 7 & 8 · Preguntas P30 – P35 · NLP + Multimedia (0% Contenido Crudo)",
    title="Psicología Conversacional, Temas Latentes e Inteligencia Multimedia",
    subtitle="Extraemos sentimiento adaptado a español colombiano, estilo de escritura, tópicos NMF, ADN multimedia, viralidad de stickers (por huella HMAC-SHA256) y dinámica de notas de voz (.opus) y videos (.mp4).",
)

# =============================================================================
# P30: Emotional Climate & Sentiment
# =============================================================================
render_question_header(
    "P30",
    "¿Cómo evoluciona el clima emocional (sentimiento) por grupo, semana y hora?",
    "Scoring léxico español/colombiano + emojis procesado 100% local en zona privada; aquí solo se muestran promedios agregados.",
)
df_sgrp = load_output_parquet("p30_sentiment_by_group.parquet").sort_values("avg_sentiment", ascending=True)
df_swk = load_output_parquet("p30_sentiment_weekly.parquet")

c30a, c30b = st.columns([1.3, 1])
with c30a:
    fig_sgrp = px.bar(
        df_sgrp,
        x="avg_sentiment",
        y="group_label",
        orientation="h",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        hover_data=["text_messages", "pct_positivo", "pct_negativo"],
    )
    fig_sgrp.add_vline(x=0, line_color=PALETTE["slate"])
    style_plotly(fig_sgrp, "Termómetro Emocional por Grupo (Score [-1, +1])", height=430)
    fig_sgrp.update_yaxes(title="")
    fig_sgrp.update_xaxes(title="Sentimiento promedio")
    st.plotly_chart(fig_sgrp, use_container_width=True)

with c30b:
    fig_swk = go.Figure()
    fig_swk.add_trace(go.Scatter(x=df_swk["week_start"], y=df_swk["pct_positivo"], name="% Positivo", line=dict(color=PALETTE["primary"], width=2.5)))
    fig_swk.add_trace(go.Scatter(x=df_swk["week_start"], y=df_swk["pct_negativo"], name="% Negativo", line=dict(color=PALETTE["danger"], width=2.5)))
    style_plotly(fig_swk, "Evolución Semanal de Mensajes Positivos vs. Negativos (%)", height=430)
    fig_swk.update_yaxes(title="% de mensajes de texto")
    st.plotly_chart(fig_swk, use_container_width=True)

# =============================================================================
# P31 & P32: Conversational Style & Latent Topics
# =============================================================================
c31, c32 = st.columns([1.15, 1.15])

with c31:
    render_question_header(
        "P31",
        "¿Cómo cambia el estilo conversacional entre grupos (palabras, risas, ráfagas)?",
        "Longitud media, tasa de preguntas (?), risas, emojis por cada 100 mensajes e índice de ráfaga (<60s).",
    )
    df_style = load_output_parquet("p31_conversational_style.parquet")
    fig_st = px.scatter(
        df_style,
        x="avg_words_per_msg",
        y="laughter_rate_pct",
        size="text_msgs",
        color="macro_category",
        hover_name="group_label",
        hover_data=["burst_continuation_pct", "question_rate_pct", "emojis_per_100_msgs"],
        color_discrete_sequence=CATEGORICAL_COLORS,
        size_max=40,
    )
    style_plotly(fig_st, "Estilo por Grupo: Palabras/Mensaje vs. Tasa de Risas (%)", height=380)
    fig_st.update_xaxes(title="Palabras promedio por mensaje")
    fig_st.update_yaxes(title="% de mensajes con risa")
    st.plotly_chart(fig_st, use_container_width=True)

with c32:
    render_question_header(
        "P32",
        "¿Cuáles son los temas latentes y cómo evolucionan semana a semana?",
        "Modelado de tópicos TF-IDF + NMF local (con lista negra PII) sobre mensajes de texto.",
    )
    df_twk = load_output_parquet("p32_topics_weekly.parquet")
    df_tcat = load_lakehouse_parquet("topic_catalog.parquet")
    fig_top = px.area(
        df_twk,
        x="week_start",
        y="week_share_pct",
        color="topic_label",
        color_discrete_sequence=CATEGORICAL_COLORS,
    )
    style_plotly(fig_top, "Evolución Semanal de Participación de Tópicos Latentes (%)", height=380)
    fig_top.update_yaxes(title="% del volumen temático")
    st.plotly_chart(fig_top, use_container_width=True)

with st.expander("📚 Catálogo de Tópicos Latentes NMF (palabras clave genéricas 100% libres de PII)"):
    st.dataframe(df_tcat, use_container_width=True, hide_index=True)

# =============================================================================
# P33: Multimodal DNA by Group
# =============================================================================
render_question_header(
    "P33",
    "¿Cuál es el ADN multimedia de cada grupo (Texto vs. Stickers vs. Fotos vs. Audios vs. Videos)?",
    "Desglose exacto de formatos por grupo cruzando `messages_meta` con el escáner físico de archivos.",
)
df_mix = load_output_parquet("p33_media_mix_by_group.parquet")
media_only = st.toggle("Ver solo desglose dentro de Multimedia (ocultar texto)", value=False)
if media_only:
    sub_mix = df_mix[df_mix["format_type"] != "text"].copy()
    sub_mix["share_pct"] = sub_mix["msg_count"] / sub_mix.groupby("group_id")["msg_count"].transform("sum") * 100
else:
    sub_mix = df_mix

fig_mix = px.bar(
    sub_mix,
    x="share_pct",
    y="group_label",
    color="format_type",
    orientation="h",
    color_discrete_sequence=CATEGORICAL_COLORS,
    hover_data=["msg_count"],
)
style_plotly(fig_mix, "Mix Comunicacional por Grupo (%)", height=440)
fig_mix.update_xaxes(title="% del total", range=[0, 100])
fig_mix.update_yaxes(title="")
st.plotly_chart(fig_mix, use_container_width=True)

# =============================================================================
# P34 & P35: Sticker Virality & Voice Notes Economy
# =============================================================================
c34, c35 = st.columns(2)

with c34:
    render_question_header(
        "P34",
        "Cultura de Stickers: ¿Cuántos son virales y cruzan múltiples grupos?",
        "Huellas criptográficas HMAC sobre SHA-256 de los .webp físicos (sin exponer imágenes).",
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Stickers Indexados", f"{stk_sum.get('total_sticker_files_indexed', 0):,}", f"{stk_sum.get('animated_stickers_pct', 0):.1f}% animados")
    m2.metric("Tasa de Reutilización", f"{stk_sum.get('reuse_rate_pct', 0):.1f}%", f"{stk_sum.get('unique_sticker_fingerprints', 0):,} únicos")
    m3.metric("Virales Multigrupo", f"{stk_sum.get('cross_group_viral_stickers', 0):,}", f"Top: {stk_sum.get('top_sticker_groups', 0)} grupos")

    df_stk = load_output_parquet("p34_sticker_virality.parquet").head(15)
    fig_stk = px.bar(
        df_stk.sort_values("total_uses", ascending=True),
        x="total_uses",
        y="media_hash",
        orientation="h",
        color="unique_groups",
        color_continuous_scale="Tealgrn",
        hover_data=["unique_senders", "is_animated", "avg_kb"],
    )
    style_plotly(fig_stk, "Top 15 Stickers Más Virales de la Comunidad (por Huella Anónima)", height=360)
    fig_stk.update_xaxes(title="Usos totales registrados")
    fig_stk.update_yaxes(title="Huella anónima")
    st.plotly_chart(fig_stk, use_container_width=True)

with c35:
    render_question_header(
        "P35",
        "Economía de Notas de Voz (.opus) y Videos (.mp4): ¿Los audios largos matan el chat?",
        "Duración exacta extraída localmente con Mutagen + tasa de respuesta inmediata (<5m) por duración.",
    )
    df_av = load_output_parquet("p35_media_technical_profile.parquet")
    df_vimp = load_output_parquet("p35_voice_response_impact.parquet")

    v1, v2, v3 = st.columns(3)
    v1.metric("Mediana Nota de Voz", f"{s5.get('p35_median_voice_note_sec', 0):.1f}s", f"n = {int((df_av['media_subtype'] == 'voice_note').sum()):,}")
    v2.metric("Mediana Video", f"{s5.get('p35_median_video_sec', 0):.1f}s", f"n = {int((df_av['media_subtype'] == 'video').sum()):,}")
    v3.metric("Test Corto vs. Largo", f"p = {s5.get('p35_short_vs_long_voice_reply_p', 1):.3f}", "Sin penalización")

    if not df_vimp.empty:
        fig_v = px.bar(
            df_vimp,
            x="duration_bucket",
            y="reply_within_5m_pct",
            text=[f"{v:.1f}% (n={n})" for v, n in zip(df_vimp["reply_within_5m_pct"], df_vimp["voice_notes"])],
            color="median_duration_sec",
            color_continuous_scale="Blues",
        )
        style_plotly(fig_v, "% de Notas de Voz Respondidas en < 5 min según Duración", height=360)
        fig_v.update_xaxes(title="Duración de la nota de voz")
        fig_v.update_yaxes(title="% respondidas en <= 5 min", range=[0, 105])
        st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.info("No hay suficientes notas de voz vinculadas con duración en este corte.")
