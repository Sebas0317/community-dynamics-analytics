"""Page 09: Methodology, Architecture, Zero-PII Privacy Audit & Public Data Exports."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.utils.data_loader import (
    load_lakehouse_parquet,
    load_output_json,
    load_output_parquet,
    load_privacy_manifest,
)
from app.utils.styles import render_hero, render_insight, render_sidebar_footer, setup_page

setup_page("Metodología y Privacidad", icon="🔒")
exec_sum = load_output_json("executive_summary.json")
manifest = load_privacy_manifest()
render_sidebar_footer(exec_sum.get("date_range"))

render_hero(
    tag="Gobernanza · Ética de Datos · Ley 1581 de 2012 (Colombia)",
    title="Metodología, Arquitectura Reproducible y Auditoría Zero-PII",
    subtitle="Transparencia total sobre el pipeline de anonimización irreversible, controles automatizados con pytest y descarga directa de tablas agregadas libres de información personal.",
)

# === PRIVACY AUDIT BANNER ===
c1, c2, c3, c4 = st.columns(4)
c1.metric("Texto Crudo Publicado", "0 filas", "✅ 100% descartado")
c2.metric("Fotos / Audios Publicados", "0 archivos", "✅ Solo metadatos técnicos")
c3.metric("Método Anonimización", "HMAC-SHA256", "Salt 256-bit efímero")
c4.metric("Patrones Privados Bloqueados", f"{manifest.get('excluded_private_patterns_count', 0)}", "Chats 1-a-1 excluidos")

render_insight(
    "Garantía Verificable con `pytest` (`tests/test_privacy_guard.py`)",
    "Cada ejecución del pipeline audita automáticamente todos los archivos <code>.parquet</code> y <code>.json</code> en <code>data_public/</code> "
    "verificando: (1) ausencia total de columnas de texto crudo o nombres, (2) formato estricto <code>MEM_NNN</code> y <code>GNN_ARQUETIPO</code>, "
    "(3) cero coincidencias regex de teléfonos colombianos/internacionales, correos electrónicos o handles <code>@usuario</code>.",
)

st.markdown("### 🧱 Arquitectura del Lakehouse Anonimizado (`data_public/parquet/`)")
tables_info = manifest.get("tables", {})
if tables_info:
    rows = [
        {
            "Tabla Parquet": tname,
            "Filas": f"{tmeta['rows']:,}",
            "Tamaño (KB)": tmeta["size_kb"],
            "Columnas Exportadas (0% PII)": ", ".join(tmeta["columns"]),
        }
        for tname, tmeta in tables_info.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("### 🛠️ Stack Técnico y Reproducibilidad")
st.markdown(
    """
    * **Ingesta Incremental (`00_ingest_whatsapp_drop.py`):** Soporta tanto línea base SQLite como nuevos dumps `.zip`/`.txt` + escáner multimedia local (`Pillow` para `.webp`/`.jpg`, `Mutagen` para duración de `.opus`/`.mp4`, `SHA-256` para deduplicación de stickers).
    * **Anonimización & NLP Local (`01_anonymize_lakehouse.py`):** Calcula sentimiento léxico español/colombiano, estilo conversacional y tópicos `TF-IDF + NMF` en memoria antes de destruir cualquier rastro de texto crudo y escribir el Lakehouse en **Parquet + DuckDB**.
    * **Motor Analítico (`02`–`06`):** `DuckDB` (funciones de ventana `LAG()`/`LEAD()`), `Lifelines` (Kaplan-Meier & Log-Rank), `SciPy` (Mann-Whitney U, Spearman), `NetworkX` + `python-louvain` (SNA y Percolación), `Scikit-learn` (Random Forest CV + K-Means + PCA).
    * **Capa Visual (`app/`):** `Streamlit` multipágina + `Plotly` + `PyVis`, desplegable con costo **USD 0** en Streamlit Community Cloud.
    """
)

st.markdown("### ⬇️ Descarga de Datasets Públicos Agregados (CSV — 0% PII)")
download_catalog = {
    "Grupos Anonimizados (groups_anon)": ("lakehouse", "groups_anon.parquet"),
    "Resumen Diario DAU/WAU/MAU (P01)": ("output", "p01_daily_metrics.parquet"),
    "Gini y Desigualdad por Grupo (P02)": ("output", "p02_gini_by_group.parquet"),
    "Supervivencia Kaplan-Meier Global (P10)": ("output", "p10_kaplan_meier_global.parquet"),
    "Nodos SNA & Centralidades (P15-P16)": ("output", "p15_sna_nodes.parquet"),
    "Simulación de Percolación (P18)": ("output", "p18_percolation_simulation.parquet"),
    "Planes por Categoría (P25)": ("output", "p25_plans_by_category.parquet"),
    "Perfiles de Arquetipos ML (P36)": ("output", "p36_archetype_profiles.parquet"),
}
sel_dl = st.selectbox("Selecciona una tabla pública agregada para inspeccionar y descargar:", list(download_catalog.keys()))
kind, fname = download_catalog[sel_dl]
df_dl = load_lakehouse_parquet(fname) if kind == "lakehouse" else load_output_parquet(fname)

if not df_dl.empty:
    st.dataframe(df_dl.head(15), use_container_width=True, hide_index=True)
    st.download_button(
        label=f"⬇️ Descargar {fname.replace('.parquet', '.csv')}",
        data=df_dl.to_csv(index=False).encode("utf-8"),
        file_name=fname.replace(".parquet", ".csv"),
        mime="text/csv",
    )
