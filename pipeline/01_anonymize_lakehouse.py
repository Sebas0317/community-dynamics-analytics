"""Step 01: Irreversible Zero-PII Anonymization & Public Lakehouse Builder.

Reads `data_private/staging_private.db`, computes local NLP features (sentiment, style,
latent topics via TF-IDF + NMF) BEFORE discarding all raw text forever, applies ephemeral
HMAC-SHA256 irreversible identity anonymization (`MEM_001`..`MEM_NNN`), links physical
media metadata with HMAC-hashed fingerprints, and writes:
  - `data_public/parquet/*.parquet` (7 strictly Zero-PII tables)
  - `data_public/community_analytics.duckdb` (fast OLAP DuckDB database)
  - `data_public/privacy_audit_manifest.json`
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.config import (
    DUCKDB_PATH,
    EXCLUDED_CHAT_PATTERNS,
    MEDIA_EXT_MAP,
    NEGATIVE_EMOJIS,
    NEGATIVE_WORDS,
    PARQUET_DIR,
    POSITIVE_EMOJIS,
    POSITIVE_WORDS,
    PRIVACY_MANIFEST_PATH,
    PUBLIC_DIR,
    SPANISH_STOPWORDS,
    STAGING_DB_PATH,
    resolve_group_archetype,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
LAUGH_RE = re.compile(r"\b(?:j[aei]j[aei][jaei]*|jsjs[js]*|xd+|lmao|lol)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{3,}")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U00002300-\U000023FF]"
)

# Curated human-readable topic labels matched to NMF clusters dynamically
TOPIC_LABEL_HINTS = [
    ("Planes y Convocatorias", {"plan", "parche", "sabado", "sábado", "domingo", "viernes", "caigan", "hora", "encuentro", "salir", "salida"}),
    ("Rumba, Bares y Fiesta", {"pola", "cerveza", "bar", "rumba", "fiesta", "trago", "noche", "tomar", "farra", "shots", "chiva"}),
    ("Gaming y Juegos de Mesa", {"juego", "juegos", "jugar", "partida", "mesa", "cartas", "bolirana", "billar", "discord", "minecraft", "uno"}),
    ("Presentaciones y Bienvenida", {"hola", "años", "vivo", "gusto", "llamó", "llamo", "nombre", "bogotá", "bogota", "norte", "sur", "engativa", "bienvenido"}),
    ("Comida y Gastronomía", {"comer", "comida", "almuerzo", "cena", "pizza", "hamburguesa", "café", "cafe", "restaurante", "rico", "hambre"}),
    ("Deporte, Gym y Rodadas", {"gym", "entrenar", "fútbol", "futbol", "partido", "cancha", "moto", "rodada", "bici", "caminata", "correr"}),
    ("Música, Arte y Cultura", {"música", "musica", "rock", "concierto", "festival", "banda", "arte", "foto", "fotos", "pintura", "cine", "película", "salsa", "baile"}),
    ("Charla Cotidiana y Humor", {"trabajo", "casa", "tiempo", "gente", "vida", "día", "semana", "cosa", "cosas", "verdad", "claro", "mejor"}),
]


def hmac_hex(salt: bytes, value: str, length: int = 16) -> str:
    return hmac.new(salt, value.encode("utf-8"), hashlib.sha256).hexdigest()[:length]


def compute_text_signals(text: str) -> tuple[int, int, int, int, int, int, int, int, float]:
    """Compute local privacy-safe numerical NLP signals from raw message text."""
    if not text:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    char_count = len(text)
    emojis = EMOJI_RE.findall(text)
    emoji_count = len(emojis)
    question_count = text.count("?")
    exclamation_count = text.count("!")
    has_url = 1 if URL_RE.search(text) else 0
    has_laughter = 1 if (LAUGH_RE.search(text) or "😂" in text or "🤣" in text) else 0
    has_mention = 1 if MENTION_RE.search(text) else 0

    alpha_chars = [c for c in text if c.isalpha()]
    is_caps_shout = 1 if (len(alpha_chars) >= 8 and sum(c.isupper() for c in alpha_chars) / len(alpha_chars) > 0.75) else 0

    # Local lexical sentiment score in [-1.0, 1.0]
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS) + sum(1 for e in emojis if e in POSITIVE_EMOJIS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS) + sum(1 for e in emojis if e in NEGATIVE_EMOJIS)
    if pos + neg == 0:
        sentiment = 0.0
    else:
        sentiment = round((pos - neg) / (pos + neg), 4)

    return (
        char_count,
        emoji_count,
        question_count,
        exclamation_count,
        has_url,
        has_laughter,
        has_mention,
        is_caps_shout,
        sentiment,
    )


def infer_media_subtype_from_msg(attachment: str | None, content: str, is_media: int) -> str:
    if is_media == 0:
        return "none"
    if attachment:
        ext = Path(attachment).suffix.lower()
        return MEDIA_EXT_MAP.get(ext, "other")
    c_low = (content or "").lower()
    if "sticker" in c_low or ".webp" in c_low:
        return "sticker"
    if "imagen" in c_low or "image" in c_low or ".jpg" in c_low:
        return "image"
    if "audio" in c_low or ".opus" in c_low:
        return "voice_note"
    if "video" in c_low or ".mp4" in c_low:
        return "video"
    return "media_omitted"


def fit_local_topic_model(
    texts: pd.Series,
    is_text_mask: pd.Series,
    pii_tokens_blacklist: set[str],
    n_topics: int = 8,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit local TF-IDF + NMF topic model strictly on non-PII tokens.

    Returns per-row `topic_id` (-1 for non-text/unassigned) and a sanitized `topic_catalog` DataFrame.
    """
    topic_ids = np.full(len(texts), -1, dtype=np.int32)
    valid_idx = np.where(is_text_mask.to_numpy() & (texts.str.len() >= 18).to_numpy())[0]

    if len(valid_idx) < 200:
        catalog = pd.DataFrame(columns=["topic_id", "topic_label", "top_keywords", "doc_count"])
        return topic_ids, catalog

    stop_all = SPANISH_STOPWORDS | pii_tokens_blacklist

    def clean_doc(s: str) -> str:
        s = URL_RE.sub(" ", s)
        s = MENTION_RE.sub(" ", s)
        toks = [t.lower() for t in TOKEN_RE.findall(s) if t.lower() not in stop_all and len(t) >= 3]
        return " ".join(toks)

    cleaned_docs = [clean_doc(str(texts.iat[i])) for i in valid_idx]
    non_empty_pos = [pos for pos, doc in enumerate(cleaned_docs) if len(doc.split()) >= 2]
    if len(non_empty_pos) < 100:
        catalog = pd.DataFrame(columns=["topic_id", "topic_label", "top_keywords", "doc_count"])
        return topic_ids, catalog

    docs_fit = [cleaned_docs[p] for p in non_empty_pos]
    orig_indices = [valid_idx[p] for p in non_empty_pos]

    vectorizer = TfidfVectorizer(max_features=2500, min_df=15, max_df=0.35, ngram_range=(1, 1))
    tfidf = vectorizer.fit_transform(docs_fit)
    feature_names = vectorizer.get_feature_names_out()

    k = min(n_topics, max(3, tfidf.shape[1] // 20))
    nmf = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=300)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    assigned = W.argmax(axis=1)
    max_weight = W.max(axis=1)
    for idx_in_fit, orig_i in enumerate(orig_indices):
        if max_weight[idx_in_fit] > 0.005:
            topic_ids[orig_i] = int(assigned[idx_in_fit])

    used_labels: set[str] = set()
    catalog_rows = []
    for t_idx in range(k):
        top_idx = H[t_idx].argsort()[::-1][:10]
        top_words = [str(feature_names[j]) for j in top_idx if str(feature_names[j]) not in pii_tokens_blacklist][:8]
        word_set = set(top_words)

        # Match best descriptive label
        best_label = f"Tópico Comunitario {t_idx + 1}"
        best_overlap = 0
        for label, hints in TOPIC_LABEL_HINTS:
            overlap = len(word_set & hints)
            if overlap > best_overlap and label not in used_labels:
                best_overlap = overlap
                best_label = label
        if best_label in used_labels:
            best_label = f"{best_label} ({t_idx + 1})"
        used_labels.add(best_label)

        doc_count = int((topic_ids == t_idx).sum())
        catalog_rows.append(
            {
                "topic_id": t_idx,
                "topic_label": best_label,
                "top_keywords": ", ".join(top_words[:6]),
                "doc_count": doc_count,
            }
        )

    return topic_ids, pd.DataFrame(catalog_rows)


def run_anonymization(staging_db: Path = STAGING_DB_PATH) -> Path:
    if not staging_db.exists():
        raise FileNotFoundError(f"Staging DB not found: {staging_db}. Run 00_ingest_whatsapp_drop.py first.")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(staging_db)

    # Ephemeral cryptographic salt — lives ONLY in RAM during this execution, NEVER written to disk!
    ephemeral_salt = secrets.token_bytes(32)

    # 1. Load raw tables
    df_groups = pd.read_sql_query("SELECT code, name, category, folder FROM groups", conn)
    df_msgs = pd.read_sql_query(
        """
        SELECT id, group_code, msg_index, dt_iso, date_str, hour, day_of_week,
               sender_norm, is_system, is_media, is_deleted, message_type,
               content, word_count, category, attachment_filename
        FROM messages
        ORDER BY group_code, dt_iso, msg_index
        """,
        conn,
    )
    df_events = pd.read_sql_query(
        "SELECT id, group_code, dt_iso, event_type, actor, target, description, message_id FROM events",
        conn,
    )
    df_plans = pd.read_sql_query(
        "SELECT id, group_code, dt_iso, date_str, category, initiator, message_id FROM plans",
        conn,
    )
    df_media = pd.read_sql_query(
        """
        SELECT group_code, filename, ext, media_subtype, file_size_bytes,
               raw_sha256, width, height, aspect_ratio, is_animated, duration_sec
        FROM media_files
        """,
        conn,
    )
    conn.close()

    # 2. Build anonymized group mapping
    code_to_group_id: dict[str, str] = {}
    groups_anon_rows = []
    for _, row in df_groups.iterrows():
        gid, label, macro, sub_arch, zone = resolve_group_archetype(str(row["code"]), str(row["name"] or ""))
        code_to_group_id[str(row["code"])] = gid
        groups_anon_rows.append(
            {
                "group_id": gid,
                "group_label": label,
                "macro_category": macro,
                "sub_archetype": sub_arch,
                "bogota_zone": zone,
            }
        )
    df_groups_anon = pd.DataFrame(groups_anon_rows).drop_duplicates(subset=["group_id"])

    # 3. Build irreversible HMAC-SHA256 member mapping (MEM_001 .. MEM_NNN)
    user_msgs = df_msgs[(df_msgs["is_system"] == 0) & (df_msgs["sender_norm"].notna()) & (df_msgs["sender_norm"] != "")]
    sender_first_dt = user_msgs.groupby("sender_norm")["dt_iso"].min().reset_index()
    sender_first_dt["hmac_sort"] = sender_first_dt["sender_norm"].map(lambda s: hmac_hex(ephemeral_salt, str(s), 24))
    sender_first_dt = sender_first_dt.sort_values(["dt_iso", "hmac_sort"]).reset_index(drop=True)
    sender_first_dt["member_id"] = [f"MEM_{i + 1:03d}" for i in range(len(sender_first_dt))]

    sender_to_member_id: dict[str, str] = dict(zip(sender_first_dt["sender_norm"], sender_first_dt["member_id"]))

    # Collect PII tokens blacklist (first names / handles + excluded patterns) so topic model NEVER outputs a real name
    pii_blacklist: set[str] = set(EXCLUDED_CHAT_PATTERNS) | {"admins", "administrador"}
    for raw_s in sender_to_member_id:
        for tok in TOKEN_RE.findall(str(raw_s).lower()):
            if len(tok) >= 3:
                pii_blacklist.add(tok)

    # Also build lookup from lowercase sender / last-4 phone digits for matching plans & system events
    masked_to_member_id: dict[str, str] = {}
    name_lookup_tokens: list[tuple[str, str]] = []
    for raw_s, mid in sender_to_member_id.items():
        rs = str(raw_s).strip()
        masked_to_member_id[rs.lower()] = mid
        if re.match(r"^\+?\d{8,}$", rs):
            masked_form = (rs[:6] + " *** " + rs[-4:]).lower()
            masked_to_member_id[masked_form] = mid
            # Also match formatted phone tails in system descriptions
            name_lookup_tokens.append((rs[-4:], mid))
        elif len(rs) >= 4:
            name_lookup_tokens.append((rs.lower(), mid))

    # 4. Compute local NLP signals & fit local topic model on raw text BEFORE dropping content
    print("[ANON] Computing local NLP signals (sentiment, style, emojis, bursts)...")
    is_text_mask = (df_msgs["message_type"] == "text") & (df_msgs["is_system"] == 0)
    nlp_tuples = [
        compute_text_signals(txt) if is_t else (0, 0, 0, 0, 0, 0, 0, 0, 0.0)
        for txt, is_t in zip(df_msgs["content"].fillna(""), is_text_mask)
    ]
    nlp_cols = pd.DataFrame(
        nlp_tuples,
        columns=[
            "char_count",
            "emoji_count",
            "question_count",
            "exclamation_count",
            "has_url",
            "has_laughter",
            "has_mention",
            "is_caps_shout",
            "sentiment_score",
        ],
    )

    print("[ANON] Fitting local TF-IDF + NMF latent topic model (PII-filtered)...")
    topic_ids, df_topic_catalog = fit_local_topic_model(
        df_msgs["content"].fillna(""), is_text_mask, pii_blacklist, n_topics=8
    )

    # 5. Construct messages_meta (STRICTLY ZERO RAW CONTENT OR PII)
    df_msgs["group_id"] = df_msgs["group_code"].map(code_to_group_id)
    df_msgs["member_id"] = df_msgs["sender_norm"].map(sender_to_member_id)
    df_msgs["media_subtype"] = [
        infer_media_subtype_from_msg(att, cnt, ism)
        for att, cnt, ism in zip(df_msgs["attachment_filename"], df_msgs["content"], df_msgs["is_media"])
    ]
    df_msgs["sentiment_label"] = np.where(
        nlp_cols["sentiment_score"] > 0.1,
        "positivo",
        np.where(nlp_cols["sentiment_score"] < -0.1, "negativo", "neutral"),
    )

    # Map message_id -> member_id for plans/events alignment
    msg_id_to_member = dict(zip(df_msgs["id"], df_msgs["member_id"]))

    df_messages_meta = pd.DataFrame(
        {
            "message_id": df_msgs["id"].astype("int64"),
            "group_id": df_msgs["group_id"].astype("string"),
            "msg_index": df_msgs["msg_index"].astype("int64"),
            "dt_iso": df_msgs["dt_iso"].astype("string"),
            "date_str": df_msgs["date_str"].astype("string"),
            "hour": df_msgs["hour"].astype("int16"),
            "day_of_week": df_msgs["day_of_week"].astype("int8"),
            "member_id": df_msgs["member_id"].astype("string"),
            "is_system": df_msgs["is_system"].astype("int8"),
            "is_media": df_msgs["is_media"].astype("int8"),
            "is_deleted": df_msgs["is_deleted"].astype("int8"),
            "message_type": df_msgs["message_type"].astype("string"),
            "media_subtype": df_msgs["media_subtype"].astype("string"),
            "word_count": df_msgs["word_count"].astype("int32"),
            "char_count": nlp_cols["char_count"].astype("int32"),
            "emoji_count": nlp_cols["emoji_count"].astype("int16"),
            "question_count": nlp_cols["question_count"].astype("int16"),
            "exclamation_count": nlp_cols["exclamation_count"].astype("int16"),
            "has_question": (nlp_cols["question_count"] > 0).astype("int8"),
            "has_url": nlp_cols["has_url"].astype("int8"),
            "has_laughter": nlp_cols["has_laughter"].astype("int8"),
            "has_mention": nlp_cols["has_mention"].astype("int8"),
            "is_caps_shout": nlp_cols["is_caps_shout"].astype("int8"),
            "sentiment_score": nlp_cols["sentiment_score"].astype("float32"),
            "sentiment_label": df_msgs["sentiment_label"].astype("string"),
            "theme_category": df_msgs["category"].astype("string"),
            "topic_id": topic_ids.astype("int16"),
        }
    )

    # 6. Enrich groups_anon with aggregate stats
    grp_stats = (
        df_messages_meta.groupby("group_id", as_index=False)
        .agg(
            total_messages=("message_id", "count"),
            user_messages=("is_system", lambda s: int((s == 0).sum())),
            system_messages=("is_system", lambda s: int((s == 1).sum())),
            media_messages=("is_media", "sum"),
            deleted_messages=("is_deleted", "sum"),
            unique_members=("member_id", lambda s: s.dropna().nunique()),
            first_dt=("dt_iso", "min"),
            last_dt=("dt_iso", "max"),
            active_days=("date_str", "nunique"),
        )
    )
    grp_stats["span_days"] = (
        (pd.to_datetime(grp_stats["last_dt"], format="mixed") - pd.to_datetime(grp_stats["first_dt"], format="mixed")).dt.days + 1
    ).astype("int32")
    df_groups_anon = df_groups_anon.merge(grp_stats, on="group_id", how="left")

    # 7. Build members_anon (STRICTLY ZERO PII)
    um = df_messages_meta[df_messages_meta["member_id"].notna()].copy()
    members_agg = (
        um.groupby("member_id", as_index=False)
        .agg(
            total_messages=("message_id", "count"),
            text_messages=("message_type", lambda s: int((s == "text").sum())),
            media_messages=("is_media", "sum"),
            deleted_messages=("is_deleted", "sum"),
            total_words=("word_count", "sum"),
            total_emojis=("emoji_count", "sum"),
            avg_sentiment=("sentiment_score", "mean"),
            groups_count=("group_id", "nunique"),
            first_dt=("dt_iso", "min"),
            last_dt=("dt_iso", "max"),
            active_days=("date_str", "nunique"),
        )
    )
    members_agg["lifespan_days"] = (
        (pd.to_datetime(members_agg["last_dt"]) - pd.to_datetime(members_agg["first_dt"])).dt.days
    ).astype("int32")
    members_agg["avg_sentiment"] = members_agg["avg_sentiment"].round(4)

    # Sort member_id numerically
    members_agg["_sort"] = members_agg["member_id"].str.replace("MEM_", "", regex=False).astype(int)
    df_members_anon = members_agg.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    # 8. Build events_anon (resolve member_id from description locally when unambiguous, then DROP raw text)
    def match_event_member(desc: str) -> str | None:
        if not desc:
            return None
        d_low = desc.lower()
        # Check longest display names first to avoid false positives
        for token, mid in sorted_name_tokens:
            if token in d_low:
                return mid
        return None

    sorted_name_tokens = sorted(
        [(t, m) for t, m in name_lookup_tokens if len(t) >= 5 and not t.isdigit()],
        key=lambda x: len(x[0]),
        reverse=True,
    )
    df_events["group_id"] = df_events["group_code"].map(code_to_group_id)
    df_events["member_id"] = df_events["description"].fillna("").map(match_event_member)
    df_events["date_str"] = df_events["dt_iso"].str.slice(0, 10)
    df_events_anon = df_events[["id", "group_id", "dt_iso", "date_str", "event_type", "member_id", "message_id"]].rename(
        columns={"id": "event_id"}
    )

    # 9. Build plans_anon (resolve initiator via message_id -> member_id, fallback to masked lookup; DROP raw text)
    df_plans["group_id"] = df_plans["group_code"].map(code_to_group_id)
    plan_members = []
    for mid_msg, init_raw in zip(df_plans["message_id"], df_plans["initiator"]):
        mem = msg_id_to_member.get(mid_msg)
        if not mem and init_raw:
            mem = masked_to_member_id.get(str(init_raw).strip().lower())
        plan_members.append(mem)
    df_plans["initiator_member_id"] = plan_members
    df_plans_anon = df_plans[
        ["id", "group_id", "dt_iso", "date_str", "category", "initiator_member_id", "message_id"]
    ].rename(columns={"id": "plan_id", "category": "plan_category"})

    # 10. Build media_meta (join physical media scan with anonymized messages via attachment_filename; HMAC file hash)
    if not df_media.empty:
        df_media["group_id"] = df_media["group_code"].map(code_to_group_id)
        df_media["media_hash"] = df_media["raw_sha256"].map(lambda h: "MED_" + hmac_hex(ephemeral_salt, str(h), 14))

        msg_attach = df_msgs[df_msgs["attachment_filename"].notna()][
            ["id", "group_code", "group_id", "member_id", "dt_iso", "date_str", "hour", "attachment_filename"]
        ].drop_duplicates(subset=["group_code", "attachment_filename"])

        merged_media = df_media.merge(
            msg_attach,
            left_on=["group_code", "filename"],
            right_on=["group_code", "attachment_filename"],
            how="left",
            suffixes=("", "_msg"),
        )
        # Fallback match by filename alone for unmatched rows
        unmatched_mask = merged_media["id"].isna()
        if unmatched_mask.any():
            fname_lookup = msg_attach.drop_duplicates(subset=["attachment_filename"]).set_index("attachment_filename")
            for col in ["id", "group_id", "member_id", "dt_iso", "date_str", "hour"]:
                fallback_vals = merged_media.loc[unmatched_mask, "filename"].map(fname_lookup[col])
                target_col = f"{col}_msg" if col == "group_id" else col
                if target_col in merged_media.columns:
                    merged_media.loc[unmatched_mask, target_col] = merged_media.loc[unmatched_mask, target_col].fillna(fallback_vals)

        merged_media["group_id_final"] = merged_media["group_id"].fillna(merged_media.get("group_id_msg"))

        df_media_meta = pd.DataFrame(
            {
                "media_id": range(1, len(merged_media) + 1),
                "message_id": merged_media["id"].astype("Int64"),
                "group_id": merged_media["group_id_final"].astype("string"),
                "member_id": merged_media["member_id"].astype("string"),
                "dt_iso": merged_media["dt_iso"].astype("string"),
                "date_str": merged_media["date_str"].astype("string"),
                "media_subtype": merged_media["media_subtype"].astype("string"),
                "file_size_bytes": merged_media["file_size_bytes"].astype("int64"),
                "media_hash": merged_media["media_hash"].astype("string"),
                "width": merged_media["width"].astype("Int32"),
                "height": merged_media["height"].astype("Int32"),
                "aspect_ratio": merged_media["aspect_ratio"].astype("float32"),
                "is_animated": merged_media["is_animated"].fillna(0).astype("int8"),
                "duration_sec": merged_media["duration_sec"].astype("float32"),
            }
        )
    else:
        df_media_meta = pd.DataFrame(
            columns=[
                "media_id", "message_id", "group_id", "member_id", "dt_iso", "date_str",
                "media_subtype", "file_size_bytes", "media_hash", "width", "height",
                "aspect_ratio", "is_animated", "duration_sec",
            ]
        )

    # 11. Write Parquet Lakehouse
    tables = {
        "groups_anon": df_groups_anon,
        "members_anon": df_members_anon,
        "messages_meta": df_messages_meta,
        "events_anon": df_events_anon,
        "plans_anon": df_plans_anon,
        "media_meta": df_media_meta,
        "topic_catalog": df_topic_catalog,
    }
    manifest_tables = {}
    for name, df in tables.items():
        out_path = PARQUET_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        manifest_tables[name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "size_kb": round(out_path.stat().st_size / 1024, 1),
        }
        print(f"[LAKEHOUSE] Wrote {out_path.name}: {len(df):,} rows ({manifest_tables[name]['size_kb']} KB)")

    # 12. Materialize DuckDB OLAP database over Parquet tables
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()
    con = duckdb.connect(str(DUCKDB_PATH))
    for name in tables:
        p = (PARQUET_DIR / f"{name}.parquet").as_posix()
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{p}')")
    con.close()
    print(f"[LAKEHOUSE] DuckDB built at: {DUCKDB_PATH}")

    # 13. Write Privacy Audit Manifest
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_standard": "Ley 1581 de 2012 (Colombia) — Zero-PII Privacy-by-Design",
        "anonymization_method": "HMAC-SHA256 with ephemeral in-memory 256-bit salt (discarded after run)",
        "raw_text_exported": False,
        "raw_media_exported": False,
        "excluded_private_patterns_count": len(EXCLUDED_CHAT_PATTERNS),
        "date_range": {
            "min_dt": str(df_messages_meta["dt_iso"].min()),
            "max_dt": str(df_messages_meta["dt_iso"].max()),
        },
        "tables": manifest_tables,
    }
    PRIVACY_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[LAKEHOUSE] Privacy manifest saved to: {PRIVACY_MANIFEST_PATH}")
    return DUCKDB_PATH


if __name__ == "__main__":
    run_anonymization()
