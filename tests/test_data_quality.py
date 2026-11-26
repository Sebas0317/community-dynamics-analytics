"""Data Quality & Referential Integrity Tests for the Public Lakehouse."""

from __future__ import annotations

import duckdb
import pandas as pd

from pipeline.config import DUCKDB_PATH, PARQUET_DIR


def test_duckdb_exists_and_queryable() -> None:
    assert DUCKDB_PATH.exists(), "community_analytics.duckdb missing!"
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    con.close()
    for req in {"groups_anon", "members_anon", "messages_meta", "events_anon", "plans_anon", "media_meta"}:
        assert req in tables, f"Table {req} missing in DuckDB!"


def test_core_row_volumes_and_integrity() -> None:
    df_g = pd.read_parquet(PARQUET_DIR / "groups_anon.parquet")
    df_m = pd.read_parquet(PARQUET_DIR / "members_anon.parquet")
    df_msg = pd.read_parquet(PARQUET_DIR / "messages_meta.parquet")
    df_ev = pd.read_parquet(PARQUET_DIR / "events_anon.parquet")
    df_pl = pd.read_parquet(PARQUET_DIR / "plans_anon.parquet")

    assert len(df_g) >= 19, "Expected at least 19 community groups"
    assert len(df_m) >= 350, "Expected at least 350 anonymized members"
    assert len(df_msg) >= 150_000, "Expected at least 150k messages"
    assert len(df_ev) >= 3_000, "Expected at least 3k lifecycle events"
    assert len(df_pl) >= 1_000, "Expected at least 1k plans"

    # Referential integrity
    valid_gids = set(df_g["group_id"])
    valid_mids = set(df_m["member_id"])

    assert set(df_msg["group_id"].dropna()).issubset(valid_gids)
    assert set(df_msg.loc[df_msg["is_system"] == 0, "member_id"].dropna()).issubset(valid_mids)
    assert set(df_ev["group_id"].dropna()).issubset(valid_gids)
    assert set(df_pl["group_id"].dropna()).issubset(valid_gids)


def test_temporal_and_sentiment_bounds() -> None:
    df_msg = pd.read_parquet(PARQUET_DIR / "messages_meta.parquet")
    assert df_msg["dt_iso"].isna().sum() == 0
    assert df_msg["hour"].between(0, 23).all()
    assert df_msg["day_of_week"].between(0, 6).all()
    assert df_msg["sentiment_score"].between(-1.0, 1.0).all()
