"""Cached data loaders reading strictly from `data_public/` (Zero-PII Parquet & JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = PROJECT_ROOT / "data_public"
PARQUET_DIR = PUBLIC_DIR / "parquet"
OUTPUTS_DIR = PUBLIC_DIR / "outputs"


@st.cache_data(show_spinner=False)
def load_output_parquet(filename: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_lakehouse_parquet(filename: str) -> pd.DataFrame:
    path = PARQUET_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_output_json(filename: str) -> dict[str, Any]:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_privacy_manifest() -> dict[str, Any]:
    path = PUBLIC_DIR / "privacy_audit_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_plan_ml_model():
    path = OUTPUTS_DIR / "p29_plan_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
