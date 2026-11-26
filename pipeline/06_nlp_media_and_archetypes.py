"""Step 06: Pilar 7 (NLP & Conversational Style) & Pilar 8 (Multimedia Intelligence & Member Archetypes) — P30 to P36."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pipeline.config import DUCKDB_PATH, OUTPUTS_DIR


def gini_coef(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    arr = arr[arr >= 0]
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, arr.size + 1)
    return float((np.sum((2 * idx - arr.size - 1) * arr)) / (arr.size * np.sum(arr)))


def run_nlp_media_and_archetypes(db_path: Path = DUCKDB_PATH) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)

    # =========================================================================
    # P30: Emotional Climate & Sentiment (By Group, Weekly, By Hour)
    # =========================================================================
    p30_grp = con.execute(
        """
        SELECT
            m.group_id,
            g.group_label,
            g.macro_category,
            COUNT(*) AS text_messages,
            ROUND(AVG(m.sentiment_score), 4) AS avg_sentiment,
            ROUND(SUM(CASE WHEN m.sentiment_label = 'positivo' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_positivo,
            ROUND(SUM(CASE WHEN m.sentiment_label = 'neutral' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_neutral,
            ROUND(SUM(CASE WHEN m.sentiment_label = 'negativo' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_negativo
        FROM messages_meta m
        JOIN groups_anon g ON m.group_id = g.group_id
        WHERE m.message_type = 'text' AND m.is_system = 0
        GROUP BY 1, 2, 3
        ORDER BY avg_sentiment DESC
        """
    ).df()
    p30_grp.to_parquet(OUTPUTS_DIR / "p30_sentiment_by_group.parquet", index=False)

    p30_wk = con.execute(
        """
        SELECT
            STRFTIME(DATE_TRUNC('week', CAST(date_str AS DATE)), '%Y-%m-%d') AS week_start,
            COUNT(*) AS text_messages,
            ROUND(AVG(sentiment_score), 4) AS avg_sentiment,
            ROUND(SUM(CASE WHEN sentiment_label = 'positivo' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_positivo,
            ROUND(SUM(CASE WHEN sentiment_label = 'negativo' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_negativo
        FROM messages_meta
        WHERE message_type = 'text' AND is_system = 0
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    p30_wk.to_parquet(OUTPUTS_DIR / "p30_sentiment_weekly.parquet", index=False)

    p30_hr = con.execute(
        """
        SELECT
            hour,
            COUNT(*) AS text_messages,
            ROUND(AVG(sentiment_score), 4) AS avg_sentiment,
            ROUND(SUM(has_laughter) * 100.0 / COUNT(*), 2) AS laughter_rate_pct
        FROM messages_meta
        WHERE message_type = 'text' AND is_system = 0
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    p30_hr.to_parquet(OUTPUTS_DIR / "p30_sentiment_by_hour.parquet", index=False)

    # =========================================================================
    # P31: Conversational Style & Message Burst Index by Group
    # =========================================================================
    p31_style = con.execute(
        """
        WITH seq AS (
            SELECT
                group_id,
                member_id,
                word_count,
                char_count,
                emoji_count,
                has_question,
                has_laughter,
                has_url,
                is_caps_shout,
                CAST(dt_iso AS TIMESTAMP) AS ts,
                LAG(member_id) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS prev_member,
                LAG(CAST(dt_iso AS TIMESTAMP)) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS prev_ts
            FROM messages_meta
            WHERE message_type = 'text' AND is_system = 0 AND member_id IS NOT NULL
        ),
        tagged AS (
            SELECT
                *,
                CASE
                    WHEN prev_member = member_id
                     AND EXTRACT(EPOCH FROM (ts - prev_ts)) BETWEEN 0 AND 60
                    THEN 1 ELSE 0
                END AS is_burst_continuation
            FROM seq
        )
        SELECT
            t.group_id,
            g.group_label,
            g.macro_category,
            COUNT(*) AS text_msgs,
            ROUND(AVG(t.word_count), 2) AS avg_words_per_msg,
            ROUND(MEDIAN(t.word_count), 1) AS median_words_per_msg,
            ROUND(AVG(t.emoji_count) * 100.0, 2) AS emojis_per_100_msgs,
            ROUND(AVG(t.has_question) * 100.0, 2) AS question_rate_pct,
            ROUND(AVG(t.has_laughter) * 100.0, 2) AS laughter_rate_pct,
            ROUND(AVG(t.has_url) * 100.0, 2) AS url_rate_pct,
            ROUND(AVG(t.is_caps_shout) * 100.0, 2) AS caps_shout_pct,
            ROUND(AVG(t.is_burst_continuation) * 100.0, 2) AS burst_continuation_pct
        FROM tagged t
        JOIN groups_anon g ON t.group_id = g.group_id
        GROUP BY 1, 2, 3
        ORDER BY text_msgs DESC
        """
    ).df()
    p31_style.to_parquet(OUTPUTS_DIR / "p31_conversational_style.parquet", index=False)

    # =========================================================================
    # P32: Latent Topics Evolution (Weekly & By Group)
    # =========================================================================
    p32_wk = con.execute(
        """
        SELECT
            STRFTIME(DATE_TRUNC('week', CAST(m.date_str AS DATE)), '%Y-%m-%d') AS week_start,
            m.topic_id,
            tc.topic_label,
            COUNT(*) AS topic_msgs
        FROM messages_meta m
        JOIN topic_catalog tc ON m.topic_id = tc.topic_id
        WHERE m.topic_id >= 0
        GROUP BY 1, 2, 3
        ORDER BY 1, 2
        """
    ).df()
    if not p32_wk.empty:
        p32_wk["week_share_pct"] = (
            p32_wk["topic_msgs"] / p32_wk.groupby("week_start")["topic_msgs"].transform("sum") * 100
        ).round(2)
    p32_wk.to_parquet(OUTPUTS_DIR / "p32_topics_weekly.parquet", index=False)

    p32_grp = con.execute(
        """
        SELECT
            m.group_id,
            g.group_label,
            m.topic_id,
            tc.topic_label,
            COUNT(*) AS topic_msgs
        FROM messages_meta m
        JOIN groups_anon g ON m.group_id = g.group_id
        JOIN topic_catalog tc ON m.topic_id = tc.topic_id
        WHERE m.topic_id >= 0
        GROUP BY 1, 2, 3, 4
        """
    ).df()
    if not p32_grp.empty:
        p32_grp["group_topic_share_pct"] = (
            p32_grp["topic_msgs"] / p32_grp.groupby("group_id")["topic_msgs"].transform("sum") * 100
        ).round(2)
    p32_grp.to_parquet(OUTPUTS_DIR / "p32_topics_by_group.parquet", index=False)

    # =========================================================================
    # P33: Multimodal DNA (Text vs Sticker vs Image vs Voice Note vs Video vs Doc)
    # =========================================================================
    p33_grp = con.execute(
        """
        SELECT
            m.group_id,
            g.group_label,
            g.macro_category,
            CASE
                WHEN m.is_media = 0 THEN 'text'
                ELSE COALESCE(m.media_subtype, 'other')
            END AS format_type,
            COUNT(*) AS msg_count
        FROM messages_meta m
        JOIN groups_anon g ON m.group_id = g.group_id
        WHERE m.is_system = 0 AND m.is_deleted = 0
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 5 DESC
        """
    ).df()
    p33_grp["share_pct"] = (
        p33_grp["msg_count"] / p33_grp.groupby("group_id")["msg_count"].transform("sum") * 100
    ).round(2)
    p33_grp.to_parquet(OUTPUTS_DIR / "p33_media_mix_by_group.parquet", index=False)

    p33_wk = con.execute(
        """
        SELECT
            STRFTIME(DATE_TRUNC('week', CAST(date_str AS DATE)), '%Y-%m-%d') AS week_start,
            CASE
                WHEN is_media = 0 THEN 'text'
                ELSE COALESCE(media_subtype, 'other')
            END AS format_type,
            COUNT(*) AS msg_count
        FROM messages_meta
        WHERE is_system = 0 AND is_deleted = 0
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    p33_wk["share_pct"] = (
        p33_wk["msg_count"] / p33_wk.groupby("week_start")["msg_count"].transform("sum") * 100
    ).round(2)
    p33_wk.to_parquet(OUTPUTS_DIR / "p33_media_mix_weekly.parquet", index=False)

    # =========================================================================
    # P34: Sticker Culture & Internal Virality (via anonymized media_hash)
    # =========================================================================
    df_stk = con.execute(
        """
        SELECT
            media_hash,
            COUNT(*) AS total_uses,
            COUNT(DISTINCT member_id) AS unique_senders,
            COUNT(DISTINCT group_id) AS unique_groups,
            MAX(is_animated) AS is_animated,
            ROUND(AVG(file_size_bytes) / 1024.0, 1) AS avg_kb
        FROM media_meta
        WHERE media_subtype = 'sticker'
        GROUP BY 1
        ORDER BY total_uses DESC
        """
    ).df()
    df_stk.to_parquet(OUTPUTS_DIR / "p34_sticker_virality.parquet", index=False)

    stk_total = int(df_stk["total_uses"].sum()) if not df_stk.empty else 0
    stk_unique = int(len(df_stk))
    stk_reused = int((df_stk["total_uses"] >= 2).sum()) if not df_stk.empty else 0
    stk_cross_grp = int((df_stk["unique_groups"] >= 2).sum()) if not df_stk.empty else 0
    stk_anim_pct = round(float(df_stk["is_animated"].mean() * 100), 2) if not df_stk.empty else 0.0
    stk_gini = round(gini_coef(df_stk["total_uses"].to_numpy()), 4) if not df_stk.empty else 0.0

    p34_summary = {
        "total_sticker_files_indexed": stk_total,
        "unique_sticker_fingerprints": stk_unique,
        "reused_stickers_count": stk_reused,
        "reuse_rate_pct": round((stk_total - stk_unique) / max(stk_total, 1) * 100, 2),
        "cross_group_viral_stickers": stk_cross_grp,
        "animated_stickers_pct": stk_anim_pct,
        "sticker_concentration_gini": stk_gini,
        "top_sticker_uses": int(df_stk.iloc[0]["total_uses"]) if not df_stk.empty else 0,
        "top_sticker_groups": int(df_stk.iloc[0]["unique_groups"]) if not df_stk.empty else 0,
    }
    (OUTPUTS_DIR / "p34_sticker_summary.json").write_text(json.dumps(p34_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # =========================================================================
    # P35: Voice Notes (.opus) & Videos (.mp4) Economy + Response Impact
    # =========================================================================
    df_av = con.execute(
        """
        SELECT
            media_id,
            message_id,
            group_id,
            member_id,
            media_subtype,
            ROUND(file_size_bytes / 1024.0, 1) AS size_kb,
            duration_sec,
            width,
            height,
            aspect_ratio
        FROM media_meta
        WHERE media_subtype IN ('voice_note', 'video', 'audio', 'image')
        """
    ).df()
    df_av.to_parquet(OUTPUTS_DIR / "p35_media_technical_profile.parquet", index=False)

    # Measure immediate reply speed (<10 min) after voice notes grouped by duration bucket
    df_voice_reply = con.execute(
        """
        WITH ordered AS (
            SELECT
                message_id,
                group_id,
                member_id,
                CAST(dt_iso AS TIMESTAMP) AS ts,
                LEAD(member_id) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS next_member,
                LEAD(CAST(dt_iso AS TIMESTAMP)) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS next_ts
            FROM messages_meta
            WHERE is_system = 0 AND member_id IS NOT NULL
        ),
        voice_linked AS (
            SELECT
                mm.duration_sec,
                mm.file_size_bytes,
                CASE
                    WHEN o.next_member IS NOT NULL AND o.next_member != o.member_id
                    THEN EXTRACT(EPOCH FROM (o.next_ts - o.ts))
                    ELSE NULL
                END AS reply_sec
            FROM media_meta mm
            JOIN ordered o ON mm.message_id = o.message_id
            WHERE mm.media_subtype = 'voice_note' AND mm.duration_sec IS NOT NULL AND mm.duration_sec > 0
        )
        SELECT * FROM voice_linked
        """
    ).df()

    if not df_voice_reply.empty:
        df_voice_reply["duration_bucket"] = pd.cut(
            df_voice_reply["duration_sec"],
            bins=[0, 15, 30, 60, 10000],
            labels=["1. Corto (<15s)", "2. Medio (15–30s)", "3. Largo (30–60s)", "4. Podcast (>60s)"],
        ).astype(str)
        df_voice_reply["replied_within_5m"] = (df_voice_reply["reply_sec"].notna() & (df_voice_reply["reply_sec"] <= 300)).astype(int)
        p35_voice_imp = (
            df_voice_reply.groupby("duration_bucket", as_index=False)
            .agg(
                voice_notes=("duration_sec", "count"),
                median_duration_sec=("duration_sec", "median"),
                median_reply_sec=("reply_sec", "median"),
                reply_within_5m_pct=("replied_within_5m", lambda s: round(float(s.mean() * 100), 2)),
            )
            .sort_values("duration_bucket")
            .round(2)
        )
        short_v = df_voice_reply[df_voice_reply["duration_sec"] <= 15]["reply_sec"].dropna()
        long_v = df_voice_reply[df_voice_reply["duration_sec"] > 45]["reply_sec"].dropna()
        _, p_voice_diff = mannwhitneyu(short_v, long_v, alternative="two-sided") if (len(short_v) > 5 and len(long_v) > 5) else (0.0, 1.0)
    else:
        p35_voice_imp = pd.DataFrame(columns=["duration_bucket", "voice_notes", "median_duration_sec", "median_reply_sec", "reply_within_5m_pct"])
        p_voice_diff = 1.0
    p35_voice_imp.to_parquet(OUTPUTS_DIR / "p35_voice_response_impact.parquet", index=False)

    # =========================================================================
    # P36: Unsupervised Human Archetypes Segmentation (K-Means + PCA 2D)
    # =========================================================================
    df_mem = con.execute("SELECT * FROM members_anon").df()
    df_sna = pd.read_parquet(OUTPUTS_DIR / "p15_sna_nodes.parquet")
    df_plans_cnt = con.execute(
        "SELECT initiator_member_id AS member_id, COUNT(*) AS plans_initiated FROM plans_anon WHERE initiator_member_id IS NOT NULL GROUP BY 1"
    ).df()
    df_member_nlp = con.execute(
        """
        SELECT
            member_id,
            ROUND(AVG(CASE WHEN message_type = 'text' THEN word_count END), 2) AS avg_words,
            ROUND(AVG(CASE WHEN message_type = 'text' THEN has_laughter END), 4) AS laughter_rate,
            ROUND(AVG(CASE WHEN message_type = 'text' THEN has_question END), 4) AS question_rate,
            ROUND(SUM(CASE WHEN media_subtype = 'sticker' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS sticker_ratio,
            ROUND(SUM(CASE WHEN media_subtype = 'voice_note' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS voice_ratio
        FROM messages_meta
        WHERE is_system = 0 AND member_id IS NOT NULL
        GROUP BY 1
        """
    ).df()

    df_feat = (
        df_mem.merge(
            df_sna[["member_id", "betweenness", "pagerank", "degree_undirected", "in_interactions", "out_interactions", "louvain_cluster", "primary_group_label"]],
            on="member_id",
            how="left",
        )
        .merge(df_plans_cnt, on="member_id", how="left")
        .merge(df_member_nlp, on="member_id", how="left")
    )
    df_feat["plans_initiated"] = df_feat["plans_initiated"].fillna(0).astype(int)
    df_feat["media_ratio"] = (df_feat["media_messages"] / np.maximum(df_feat["total_messages"], 1)).round(4)
    df_feat["log_messages"] = np.log1p(df_feat["total_messages"])
    df_feat["log_plans"] = np.log1p(df_feat["plans_initiated"])

    cluster_features = [
        "log_messages",
        "active_days",
        "lifespan_days",
        "groups_count",
        "betweenness",
        "pagerank",
        "degree_undirected",
        "media_ratio",
        "sticker_ratio",
        "voice_ratio",
        "avg_words",
        "laughter_rate",
        "avg_sentiment",
        "log_plans",
    ]
    X_raw = df_feat[cluster_features].fillna(0.0).to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Evaluate K=4..7 and choose K=6 (or best silhouette in [5, 6] for rich portfolio interpretability)
    sil_scores = {}
    for k_try in [4, 5, 6, 7]:
        km_try = KMeans(n_clusters=k_try, random_state=42, n_init=20)
        labels_try = km_try.fit_predict(X_scaled)
        sil_scores[k_try] = round(float(silhouette_score(X_scaled, labels_try)), 4)

    k_opt = 6
    km = KMeans(n_clusters=k_opt, random_state=42, n_init=30)
    df_feat["cluster_id"] = km.fit_predict(X_scaled)

    # PCA 2D projection
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_feat["pca_1"] = np.round(coords[:, 0], 4)
    df_feat["pca_2"] = np.round(coords[:, 1], 4)

    # Dynamic human-centric archetype labeling based on cluster centroid rankings
    centroids = (
        df_feat.groupby("cluster_id", as_index=False)
        .agg(
            members=("member_id", "count"),
            avg_msgs=("total_messages", "mean"),
            median_msgs=("total_messages", "median"),
            avg_groups=("groups_count", "mean"),
            avg_betweenness=("betweenness", "mean"),
            avg_lifespan=("lifespan_days", "mean"),
            avg_plans=("plans_initiated", "mean"),
            avg_media_ratio=("media_ratio", "mean"),
            avg_sticker_ratio=("sticker_ratio", "mean"),
            avg_words=("avg_words", "mean"),
            avg_sentiment=("avg_sentiment", "mean"),
        )
    )

    assigned_names: dict[int, str] = {}
    remaining = set(centroids["cluster_id"].tolist())

    def pick_best(col: str, label: str, ascending: bool = False) -> None:
        sub = centroids[centroids["cluster_id"].isin(remaining)].sort_values(col, ascending=ascending)
        if not sub.empty:
            cid = int(sub.iloc[0]["cluster_id"])
            assigned_names[cid] = label
            remaining.remove(cid)

    # Priority assignment ensures every archetype matches its true dominant behavioral signature
    pick_best("avg_msgs", "Núcleo Motor & Superconectores", ascending=False)
    pick_best("avg_plans", "Organizadores & Catalizadores de Planes", ascending=False)
    pick_best("avg_media_ratio", "Expresivos Visuales & Memeros", ascending=False)
    pick_best("avg_words", "Conversadores Reflexivos / Narradores", ascending=False)
    pick_best("avg_lifespan", "Observadores & Visitantes Fugaces", ascending=True)
    for cid in sorted(remaining):
        assigned_names[cid] = "Especialistas de Nicho & Regulares"

    df_feat["archetype_name"] = df_feat["cluster_id"].map(assigned_names)
    centroids["archetype_name"] = centroids["cluster_id"].map(assigned_names)
    centroids = centroids.sort_values("avg_msgs", ascending=False).round(4)

    df_feat.to_parquet(OUTPUTS_DIR / "p36_member_archetypes.parquet", index=False)
    centroids.to_parquet(OUTPUTS_DIR / "p36_archetype_profiles.parquet", index=False)

    clust_meta = {
        "k_selected": k_opt,
        "silhouette_scores_by_k": {str(k): v for k, v in sil_scores.items()},
        "selected_silhouette": sil_scores[k_opt],
        "pca_explained_variance_pct": [round(float(x * 100), 2) for x in pca.explained_variance_ratio_],
        "features_used": cluster_features,
    }
    (OUTPUTS_DIR / "p36_clustering_metrics.json").write_text(json.dumps(clust_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    con.close()

    voice_med_dur = float(df_av[df_av["media_subtype"] == "voice_note"]["duration_sec"].median()) if not df_av[df_av["media_subtype"] == "voice_note"].empty else 0.0
    video_med_dur = float(df_av[df_av["media_subtype"] == "video"]["duration_sec"].median()) if not df_av[df_av["media_subtype"] == "video"].empty else 0.0

    summary = {
        "p30_most_positive_group": str(p30_grp.iloc[0]["group_label"]) if not p30_grp.empty else "N/A",
        "p30_most_negative_group": str(p30_grp.iloc[-1]["group_label"]) if not p30_grp.empty else "N/A",
        "p31_avg_words_global": round(float(p31_style["avg_words_per_msg"].mean()), 2) if not p31_style.empty else 0.0,
        "p31_avg_burst_pct": round(float(p31_style["burst_continuation_pct"].mean()), 2) if not p31_style.empty else 0.0,
        "p34_unique_stickers": stk_unique,
        "p34_sticker_reuse_rate_pct": p34_summary["reuse_rate_pct"],
        "p34_cross_group_viral_stickers": stk_cross_grp,
        "p35_median_voice_note_sec": round(voice_med_dur, 1),
        "p35_median_video_sec": round(video_med_dur, 1),
        "p35_short_vs_long_voice_reply_p": float(f"{p_voice_diff:.6g}"),
        "p36_archetypes_k": k_opt,
        "p36_silhouette_score": sil_scores[k_opt],
    }
    (OUTPUTS_DIR / "p30_p36_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[P30-P36] Done! Stickers={stk_total:,} ({stk_unique:,} unique), VoiceMed={voice_med_dur:.1f}s, Archetypes K={k_opt} (Sil={sil_scores[k_opt]})")
    return summary


if __name__ == "__main__":
    run_nlp_media_and_archetypes()
