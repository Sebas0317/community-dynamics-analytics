"""Automated Zero-PII Privacy Guard Test Suite.

Scans every Parquet, JSON, and CSV artifact inside `data_public/` to mathematically
guarantee that ZERO personal identifiers, phone numbers, emails, @handles, real chat
names, or raw message text columns can ever leak into the public portfolio repository.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from pipeline.config import EXCLUDED_CHAT_PATTERNS, PARQUET_DIR, PUBLIC_DIR

FORBIDDEN_COLUMNS = {
    "content",
    "message_text",
    "raw_text",
    "sender_raw",
    "sender_norm",
    "sender_masked",
    "display_names",
    "phone",
    "telefono",
    "email",
    "actor",
    "target",
    "description",
    "title",
    "evidence_text",
    "initiator",
    "txt_rel",
    "folder",
    "filename",
    "raw_sha256",
}

PHONE_RE = re.compile(r"(?<![0-9a-fA-F])(?:\+?57[\s-]?)?3\d{2}[\s-]?\d{3}[\s-]?\d{4}(?![0-9a-fA-F])|\+\d{10,15}|\*{3}\s*\d{4}")
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
HANDLE_RE = re.compile(r"(?:^|\s)@[a-zA-Z0-9_.]{3,}")
MEMBER_ID_RE = re.compile(r"^MEM_\d{3,5}$")
GROUP_ID_RE = re.compile(r"^G\d{2}_[A-Z0-9_]+$")
MEDIA_HASH_RE = re.compile(r"^MED_[0-9a-f]{14}$")


def _iter_public_parquets() -> list[Path]:
    return sorted(PUBLIC_DIR.rglob("*.parquet"))


def _iter_public_jsons() -> list[Path]:
    return sorted(PUBLIC_DIR.rglob("*.json"))


def test_public_parquet_files_exist() -> None:
    expected = [
        "groups_anon.parquet",
        "members_anon.parquet",
        "messages_meta.parquet",
        "events_anon.parquet",
        "plans_anon.parquet",
        "media_meta.parquet",
        "topic_catalog.parquet",
    ]
    for fname in expected:
        assert (PARQUET_DIR / fname).exists(), f"Missing required anonymized parquet: {fname}"


def test_no_forbidden_columns_in_any_parquet() -> None:
    for pq in _iter_public_parquets():
        df = pd.read_parquet(pq)
        bad = set(c.lower() for c in df.columns) & FORBIDDEN_COLUMNS
        assert not bad, f"Forbidden PII/raw column(s) {bad} detected in {pq.name}!"


def test_member_and_group_id_formats_strict() -> None:
    df_members = pd.read_parquet(PARQUET_DIR / "members_anon.parquet")
    bad_members = [m for m in df_members["member_id"].dropna() if not MEMBER_ID_RE.match(str(m))]
    assert not bad_members, f"Non-anonymized member_id(s): {bad_members[:5]}"

    df_groups = pd.read_parquet(PARQUET_DIR / "groups_anon.parquet")
    bad_groups = [g for g in df_groups["group_id"].dropna() if not GROUP_ID_RE.match(str(g))]
    assert not bad_groups, f"Non-anonymized group_id(s): {bad_groups[:5]}"


def test_no_phone_email_handle_or_private_chat_in_parquets() -> None:
    for pq in _iter_public_parquets():
        df = pd.read_parquet(pq)
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            unique_vals = df[col].dropna().astype(str).unique()
            if col == "media_hash":
                bad_hashes = [v for v in unique_vals if not MEDIA_HASH_RE.match(v)]
                assert not bad_hashes, f"Malformed media_hash in {pq.name}: {bad_hashes[:5]}"
                continue
            for val in unique_vals:
                assert not PHONE_RE.search(val), f"Phone pattern leaked in {pq.name}.{col}: {val}"
                assert not EMAIL_RE.search(val), f"Email leaked in {pq.name}.{col}: {val}"
                assert not HANDLE_RE.search(val), f"@handle leaked in {pq.name}.{col}: {val}"
                low = val.lower()
                for excl in EXCLUDED_CHAT_PATTERNS:
                    assert excl not in low, f"Excluded private chat token '{excl}' leaked in {pq.name}.{col}: {val}"


def test_no_pii_in_public_jsons() -> None:
    for jp in _iter_public_jsons():
        text = jp.read_text(encoding="utf-8")
        # Ensure valid JSON
        json.loads(text)
        assert not PHONE_RE.search(text), f"Phone pattern leaked in {jp.name}!"
        assert not EMAIL_RE.search(text), f"Email leaked in {jp.name}!"
        assert not HANDLE_RE.search(text), f"@handle leaked in {jp.name}!"
        low = text.lower()
        for excl in EXCLUDED_CHAT_PATTERNS:
            assert excl not in low, f"Excluded private token '{excl}' leaked in {jp.name}!"
