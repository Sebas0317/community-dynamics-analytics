"""Step 05: Pilar 5 (Multigroup Dynamics & Geography) & Pilar 6 (Plans Economy & ML) — P20 to P29."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.config import (
    CORE_GROUP_ID,
    DUCKDB_PATH,
    OUTPUTS_DIR,
    PLAN_TRACTION_LONG_HOURS,
    PLAN_TRACTION_SHORT_HOURS,
)


def run_multigroup_and_plans_ml(db_path: Path = DUCKDB_PATH) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)

    df_groups = con.execute("SELECT * FROM groups_anon").df()
    df_mem = con.execute("SELECT * FROM members_anon").df()

    # =========================================================================
    # P20: Cannibalization vs Expansion (Core vs Subgroups Share of Voice)
    # =========================================================================
    df_sov = con.execute(
        f"""
        SELECT
            m.date_str,
            SUM(CASE WHEN m.group_id = '{CORE_GROUP_ID}' THEN 1 ELSE 0 END) AS core_msgs,
            SUM(CASE WHEN m.group_id != '{CORE_GROUP_ID}' THEN 1 ELSE 0 END) AS subgroups_msgs,
            COUNT(*) AS total_msgs,
            COUNT(DISTINCT CASE WHEN m.group_id = '{CORE_GROUP_ID}' THEN m.member_id END) AS core_dau,
            COUNT(DISTINCT CASE WHEN m.group_id != '{CORE_GROUP_ID}' THEN m.member_id END) AS subgroups_dau,
            COUNT(DISTINCT m.member_id) AS total_dau
        FROM messages_meta m
        WHERE m.is_system = 0
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    df_sov["core_share_pct"] = np.where(df_sov["total_msgs"] > 0, df_sov["core_msgs"] / df_sov["total_msgs"] * 100, 0.0).round(2)
    df_sov["subgroups_share_pct"] = (100.0 - df_sov["core_share_pct"]).round(2)
    df_sov.to_parquet(OUTPUTS_DIR / "p20_share_of_voice_daily.parquet", index=False)

    # Event study around explosive subgroup launches (G02_SOCIAL_CHAT_LIBRE, G19_GEO_OCCIDENTE)
    df_sov["dt"] = pd.to_datetime(df_sov["date_str"])
    cannibal_events = []
    for target_gid in ["G19_GEO_OCCIDENTE", "G02_SOCIAL_CHAT_LIBRE"]:
        row_g = df_groups[df_groups["group_id"] == target_gid]
        if row_g.empty:
            continue
        launch_dt = pd.to_datetime(row_g["first_dt"].iloc[0]).normalize()
        bef = df_sov[(df_sov["dt"] >= launch_dt - pd.Timedelta(days=7)) & (df_sov["dt"] < launch_dt)]
        aft = df_sov[(df_sov["dt"] >= launch_dt) & (df_sov["dt"] < launch_dt + pd.Timedelta(days=7))]
        if len(bef) > 0 and len(aft) > 0:
            core_b, core_a = float(bef["core_msgs"].mean()), float(aft["core_msgs"].mean())
            tot_b, tot_a = float(bef["total_msgs"].mean()), float(aft["total_msgs"].mean())
            cannibal_events.append(
                {
                    "group_id": target_gid,
                    "group_label": str(row_g["group_label"].iloc[0]),
                    "launch_date": launch_dt.strftime("%Y-%m-%d"),
                    "core_daily_before": round(core_b, 1),
                    "core_daily_after": round(core_a, 1),
                    "core_delta_pct": round((core_a - core_b) / max(core_b, 1e-6) * 100, 2),
                    "total_daily_before": round(tot_b, 1),
                    "total_daily_after": round(tot_a, 1),
                    "total_delta_pct": round((tot_a - tot_b) / max(tot_b, 1e-6) * 100, 2),
                    "verdict": "Expansión Neta" if tot_a > tot_b * 1.05 else ("Canibalización" if core_a < core_b * 0.90 else "Estable"),
                }
            )
    rho_cs, p_cs = spearmanr(df_sov["core_msgs"], df_sov["subgroups_msgs"])
    (OUTPUTS_DIR / "p20_cannibalization_events.json").write_text(
        json.dumps(
            {
                "spearman_core_vs_subgroups": round(float(rho_cs), 4),
                "p_value": float(f"{p_cs:.6g}"),
                "launch_studies": cannibal_events,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # P21: User Journey — First-Order Markov Transition Matrix (Sankey)
    # =========================================================================
    df_seq = con.execute(
        """
        SELECT
            m.member_id,
            m.group_id,
            g.group_label,
            g.macro_category,
            MIN(m.dt_iso) AS first_in_group
        FROM messages_meta m
        JOIN groups_anon g ON m.group_id = g.group_id
        WHERE m.is_system = 0 AND m.member_id IS NOT NULL
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 5
        """
    ).df()

    trans_counts: dict[tuple[str, str], int] = {}
    macro_trans: dict[tuple[str, str], int] = {}
    for _, grp in df_seq.groupby("member_id"):
        g_list = grp["group_label"].tolist()
        m_list = grp["macro_category"].tolist()
        for i in range(len(g_list) - 1):
            if g_list[i] != g_list[i + 1]:
                pair = (g_list[i], g_list[i + 1])
                trans_counts[pair] = trans_counts.get(pair, 0) + 1
            if m_list[i] != m_list[i + 1]:
                mpair = (m_list[i], m_list[i + 1])
                macro_trans[mpair] = macro_trans.get(mpair, 0) + 1

    df_trans = pd.DataFrame(
        [{"source_group": k[0], "target_group": k[1], "transitions": v} for k, v in trans_counts.items()]
    )
    if not df_trans.empty:
        df_trans["prob_from_source"] = (
            df_trans["transitions"] / df_trans.groupby("source_group")["transitions"].transform("sum")
        ).round(4)
        df_trans = df_trans.sort_values("transitions", ascending=False).reset_index(drop=True)
    df_trans.to_parquet(OUTPUTS_DIR / "p21_markov_transitions.parquet", index=False)

    df_mtrans = pd.DataFrame(
        [{"source_category": k[0], "target_category": k[1], "transitions": v} for k, v in macro_trans.items()]
    ).sort_values("transitions", ascending=False)
    df_mtrans.to_parquet(OUTPUTS_DIR / "p21_markov_macro_transitions.parquet", index=False)

    # =========================================================================
    # P22: Group Audience Overlap & Jaccard Similarity Matrix
    # =========================================================================
    grp_members = df_seq.groupby("group_id")["member_id"].apply(set).to_dict()
    lbl_map = dict(zip(df_groups["group_id"], df_groups["group_label"]))
    gids = sorted(grp_members.keys())
    jacc_rows = []
    for g1 in gids:
        s1 = grp_members[g1]
        for g2 in gids:
            s2 = grp_members[g2]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            jacc = round(inter / union, 4) if union > 0 else 0.0
            jacc_rows.append(
                {
                    "group_a": g1,
                    "label_a": lbl_map.get(g1, g1),
                    "group_b": g2,
                    "label_b": lbl_map.get(g2, g2),
                    "shared_members": inter,
                    "union_members": union,
                    "jaccard": jacc,
                }
            )
    df_jacc = pd.DataFrame(jacc_rows)
    df_jacc.to_parquet(OUTPUTS_DIR / "p22_jaccard_matrix.parquet", index=False)

    # =========================================================================
    # P23: Urban Geography (Bogota Zones vs Thematic / Social / Core)
    # =========================================================================
    df_gini_grp = pd.read_parquet(OUTPUTS_DIR / "p02_gini_by_group.parquet")
    df_frt_grp = pd.read_parquet(OUTPUTS_DIR / "p08_frt_by_group.parquet")
    df_health = pd.read_parquet(OUTPUTS_DIR / "p03_group_health.parquet")

    df_geo = df_health[
        ["group_id", "group_label", "macro_category", "bogota_zone", "user_messages", "unique_members", "active_days", "msgs_per_active_day", "net_growth", "churn_rate_pct"]
    ].merge(df_gini_grp[["group_id", "gini", "median_msgs_per_member"]], on="group_id", how="left").merge(
        df_frt_grp[["group_id", "median_frt_sec", "pct_under_5min"]], on="group_id", how="left"
    )
    df_geo.to_parquet(OUTPUTS_DIR / "p23_geography_vs_thematic.parquet", index=False)

    df_macro_comp = (
        df_geo.groupby("macro_category", as_index=False)
        .agg(
            groups_count=("group_id", "count"),
            total_user_msgs=("user_messages", "sum"),
            avg_msgs_per_active_day=("msgs_per_active_day", "mean"),
            avg_gini=("gini", "mean"),
            avg_median_frt_sec=("median_frt_sec", "mean"),
            avg_churn_rate_pct=("churn_rate_pct", "mean"),
        )
        .round(2)
    )
    df_macro_comp.to_parquet(OUTPUTS_DIR / "p23_macro_category_comparison.parquet", index=False)

    # =========================================================================
    # P24: Multi-group Sweet Spot (Notification Fatigue Curve)
    # =========================================================================
    def bucket_groups(n: int) -> str:
        if n <= 1:
            return "1 grupo"
        if n == 2:
            return "2 grupos"
        if n == 3:
            return "3 grupos"
        if n == 4:
            return "4 grupos"
        if n == 5:
            return "5 grupos"
        if n <= 7:
            return "6–7 grupos"
        return "8+ grupos"

    df_mem["groups_bucket"] = df_mem["groups_count"].map(bucket_groups)
    df_mem["retained_d14"] = (df_mem["lifespan_days"] >= 14).astype(int)
    df_mem["msgs_per_group"] = df_mem["total_messages"] / np.maximum(df_mem["groups_count"], 1)
    order_map = {"1 grupo": 1, "2 grupos": 2, "3 grupos": 3, "4 grupos": 4, "5 grupos": 5, "6–7 grupos": 6, "8+ grupos": 7}

    df_sweet = (
        df_mem.groupby("groups_bucket", as_index=False)
        .agg(
            members=("member_id", "count"),
            median_lifespan_days=("lifespan_days", "median"),
            mean_lifespan_days=("lifespan_days", "mean"),
            retention_d14_pct=("retained_d14", lambda s: round(float(s.mean() * 100), 2)),
            median_total_msgs=("total_messages", "median"),
            median_msgs_per_group=("msgs_per_group", "median"),
            mean_active_days=("active_days", "mean"),
        )
    )
    df_sweet["bucket_order"] = df_sweet["groups_bucket"].map(order_map)
    df_sweet = df_sweet.sort_values("bucket_order").round(2)
    df_sweet.to_parquet(OUTPUTS_DIR / "p24_multigroup_sweetspot.parquet", index=False)

    # =========================================================================
    # P25–P29: Plans Economy (Digital -> Real-World Traction + ML Predictor)
    # =========================================================================
    short_sec = PLAN_TRACTION_SHORT_HOURS * 3600
    long_sec = PLAN_TRACTION_LONG_HOURS * 3600

    df_plans_tr = con.execute(
        f"""
        WITH plan_base AS (
            SELECT
                p.plan_id,
                p.group_id,
                g.group_label,
                g.macro_category,
                p.dt_iso,
                p.date_str,
                EXTRACT(HOUR FROM CAST(p.dt_iso AS TIMESTAMP)) AS hour,
                EXTRACT(ISODOW FROM CAST(p.dt_iso AS TIMESTAMP)) - 1 AS day_of_week,
                p.plan_category,
                p.initiator_member_id,
                p.message_id
            FROM plans_anon p
            JOIN groups_anon g ON p.group_id = g.group_id
        ),
        traction AS (
            SELECT
                pb.plan_id,
                COUNT(CASE WHEN EXTRACT(EPOCH FROM (CAST(m.dt_iso AS TIMESTAMP) - CAST(pb.dt_iso AS TIMESTAMP))) BETWEEN 1 AND {short_sec} THEN 1 END) AS replies_2h,
                COUNT(DISTINCT CASE WHEN EXTRACT(EPOCH FROM (CAST(m.dt_iso AS TIMESTAMP) - CAST(pb.dt_iso AS TIMESTAMP))) BETWEEN 1 AND {short_sec}
                                     AND m.member_id != COALESCE(pb.initiator_member_id, '') THEN m.member_id END) AS unique_responders_2h,
                COUNT(CASE WHEN EXTRACT(EPOCH FROM (CAST(m.dt_iso AS TIMESTAMP) - CAST(pb.dt_iso AS TIMESTAMP))) BETWEEN 1 AND {long_sec} THEN 1 END) AS replies_24h,
                COUNT(DISTINCT CASE WHEN EXTRACT(EPOCH FROM (CAST(m.dt_iso AS TIMESTAMP) - CAST(pb.dt_iso AS TIMESTAMP))) BETWEEN 1 AND {long_sec}
                                     AND m.member_id != COALESCE(pb.initiator_member_id, '') THEN m.member_id END) AS unique_responders_24h,
                MIN(CASE WHEN EXTRACT(EPOCH FROM (CAST(m.dt_iso AS TIMESTAMP) - CAST(pb.dt_iso AS TIMESTAMP))) BETWEEN 1 AND {short_sec}
                          AND m.member_id != COALESCE(pb.initiator_member_id, '')
                         THEN EXTRACT(EPOCH FROM (CAST(m.dt_iso AS TIMESTAMP) - CAST(pb.dt_iso AS TIMESTAMP))) END) AS first_reply_sec
            FROM plan_base pb
            LEFT JOIN messages_meta m
              ON m.group_id = pb.group_id
             AND m.is_system = 0
             AND m.dt_iso > pb.dt_iso
             AND CAST(m.dt_iso AS TIMESTAMP) <= CAST(pb.dt_iso AS TIMESTAMP) + INTERVAL '{PLAN_TRACTION_LONG_HOURS} hours'
            GROUP BY 1
        ),
        pins_near AS (
            SELECT
                pb.plan_id,
                MAX(CASE WHEN ABS(EXTRACT(EPOCH FROM (CAST(e.dt_iso AS TIMESTAMP) - CAST(pb.dt_iso AS TIMESTAMP)))) <= 3600 THEN 1 ELSE 0 END) AS had_pin_near
            FROM plan_base pb
            LEFT JOIN events_anon e
              ON e.group_id = pb.group_id AND e.event_type = 'pin'
            GROUP BY 1
        )
        SELECT
            pb.*,
            COALESCE(t.replies_2h, 0) AS replies_2h,
            COALESCE(t.unique_responders_2h, 0) AS unique_responders_2h,
            COALESCE(t.replies_24h, 0) AS replies_24h,
            COALESCE(t.unique_responders_24h, 0) AS unique_responders_24h,
            t.first_reply_sec,
            COALESCE(pn.had_pin_near, 0) AS had_pin_near
        FROM plan_base pb
        LEFT JOIN traction t ON pb.plan_id = t.plan_id
        LEFT JOIN pins_near pn ON pb.plan_id = pn.plan_id
        """
    ).df()

    # Merge initiator SNA & tenure features
    sna_nodes = pd.read_parquet(OUTPUTS_DIR / "p15_sna_nodes.parquet")
    init_feats = sna_nodes[["member_id", "total_messages", "betweenness", "pagerank", "groups_count", "first_dt"]].rename(
        columns={
            "member_id": "initiator_member_id",
            "total_messages": "initiator_total_msgs",
            "betweenness": "initiator_betweenness",
            "pagerank": "initiator_pagerank",
            "groups_count": "initiator_groups_count",
            "first_dt": "initiator_first_dt",
        }
    )
    df_plans_tr = df_plans_tr.merge(init_feats, on="initiator_member_id", how="left")
    df_plans_tr["initiator_tenure_days"] = (
        (pd.to_datetime(df_plans_tr["dt_iso"]) - pd.to_datetime(df_plans_tr["initiator_first_dt"])).dt.days.fillna(0).clip(lower=0)
    )

    # Define high-traction binary target (top ~35% traction: unique_responders_2h >= threshold)
    thresh_resp = max(5, int(np.percentile(df_plans_tr["unique_responders_2h"], 65)))
    df_plans_tr["plan_high_traction"] = (df_plans_tr["unique_responders_2h"] >= thresh_resp).astype(int)
    df_plans_tr["fast_first_reply_5m"] = (
        df_plans_tr["first_reply_sec"].notna() & (df_plans_tr["first_reply_sec"] <= 300)
    ).astype(int)

    df_plans_tr.drop(columns=["initiator_first_dt"]).to_parquet(OUTPUTS_DIR / "p25_plans_enriched.parquet", index=False)

    # P25: Traction by Plan Category
    p25_cat = (
        df_plans_tr.groupby("plan_category", as_index=False)
        .agg(
            plans_count=("plan_id", "count"),
            median_responders_2h=("unique_responders_2h", "median"),
            mean_responders_2h=("unique_responders_2h", "mean"),
            median_replies_2h=("replies_2h", "median"),
            median_responders_24h=("unique_responders_24h", "median"),
            high_traction_rate_pct=("plan_high_traction", lambda s: round(float(s.mean() * 100), 2)),
        )
        .sort_values("high_traction_rate_pct", ascending=False)
        .round(2)
    )
    p25_cat.to_parquet(OUTPUTS_DIR / "p25_plans_by_category.parquet", index=False)

    # P26: Optimal Day x Time Block Heatmap for Plans
    def time_block(h: int) -> str:
        if 6 <= h < 12:
            return "Mañana (06–11h)"
        if 12 <= h < 17:
            return "Tarde (12–16h)"
        if 17 <= h < 21:
            return "Prime (17–20h)"
        if 21 <= h <= 23:
            return "Noche (21–23h)"
        return "Madrugada (00–05h)"

    df_plans_tr["time_block"] = df_plans_tr["hour"].astype(int).map(time_block)
    p26_hm = (
        df_plans_tr.groupby(["day_of_week", "time_block"], as_index=False)
        .agg(
            plans_count=("plan_id", "count"),
            median_responders_2h=("unique_responders_2h", "median"),
            high_traction_rate_pct=("plan_high_traction", lambda s: round(float(s.mean() * 100), 2)),
        )
        .round(2)
    )
    p26_hm.to_parquet(OUTPUTS_DIR / "p26_plans_heatmap_day_hour.parquet", index=False)

    # P27: Initiator Social Capital vs Plan Traction
    valid_init = df_plans_tr.dropna(subset=["initiator_betweenness", "initiator_total_msgs"]).copy()
    rho_init_bet, p_init_bet = spearmanr(valid_init["initiator_betweenness"], valid_init["unique_responders_2h"]) if len(valid_init) > 10 else (0.0, 1.0)
    rho_init_vol, p_init_vol = spearmanr(valid_init["initiator_total_msgs"], valid_init["unique_responders_2h"]) if len(valid_init) > 10 else (0.0, 1.0)
    valid_init["social_capital_tier"] = pd.qcut(
        valid_init["initiator_betweenness"].rank(method="first"),
        q=4,
        labels=["Q1 Bajo Capital", "Q2 Medio-Bajo", "Q3 Medio-Alto", "Q4 Alto Capital Social"],
    ).astype(str)
    p27_sc = (
        valid_init.groupby("social_capital_tier", as_index=False)
        .agg(
            plans_count=("plan_id", "count"),
            median_responders_2h=("unique_responders_2h", "median"),
            mean_responders_2h=("unique_responders_2h", "mean"),
            high_traction_rate_pct=("plan_high_traction", lambda s: round(float(s.mean() * 100), 2)),
        )
        .round(2)
    )
    p27_sc.to_parquet(OUTPUTS_DIR / "p27_initiator_social_capital.parquet", index=False)

    # P28: Impact of Pinned Messages on Plan Traction
    pinned_p = df_plans_tr[df_plans_tr["had_pin_near"] == 1]["unique_responders_2h"]
    unpinned_p = df_plans_tr[df_plans_tr["had_pin_near"] == 0]["unique_responders_2h"]
    u_pin, p_pin = mannwhitneyu(pinned_p, unpinned_p, alternative="two-sided") if (len(pinned_p) > 2 and len(unpinned_p) > 2) else (0.0, 1.0)
    p28_stats = {
        "pinned_plans_count": int(len(pinned_p)),
        "unpinned_plans_count": int(len(unpinned_p)),
        "pinned_median_responders_2h": round(float(pinned_p.median()), 2) if len(pinned_p) else 0.0,
        "unpinned_median_responders_2h": round(float(unpinned_p.median()), 2) if len(unpinned_p) else 0.0,
        "pinned_mean_responders_2h": round(float(pinned_p.mean()), 2) if len(pinned_p) else 0.0,
        "unpinned_mean_responders_2h": round(float(unpinned_p.mean()), 2) if len(unpinned_p) else 0.0,
        "mann_whitney_p_value": float(f"{p_pin:.6g}"),
    }
    (OUTPUTS_DIR / "p28_pin_impact.json").write_text(json.dumps(p28_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # P29: Predictive ML Model for Plan Traction (Logistic Regression vs Random Forest)
    ml_df = df_plans_tr.copy()
    num_cols = [
        "hour",
        "day_of_week",
        "initiator_betweenness",
        "initiator_total_msgs",
        "initiator_groups_count",
        "initiator_tenure_days",
        "had_pin_near",
        "fast_first_reply_5m",
    ]
    cat_cols = ["plan_category", "macro_category"]
    for c in num_cols:
        ml_df[c] = ml_df[c].fillna(ml_df[c].median() if ml_df[c].notna().any() else 0.0)
    for c in cat_cols:
        ml_df[c] = ml_df[c].fillna("Otro").astype(str)

    X = ml_df[num_cols + cat_cols]
    y = ml_df["plan_high_traction"].to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=500, random_state=42))])
    rf_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=250, max_depth=7, min_samples_leaf=8, random_state=42)),
        ]
    )

    lr_probs = cross_val_predict(lr_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    rf_probs = cross_val_predict(rf_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    lr_auc = round(float(roc_auc_score(y, lr_probs)), 4)
    rf_auc = round(float(roc_auc_score(y, rf_probs)), 4)

    best_pipe = rf_pipe if rf_auc >= lr_auc else lr_pipe
    best_name = "Random Forest" if rf_auc >= lr_auc else "Logistic Regression"
    best_probs = rf_probs if rf_auc >= lr_auc else lr_probs
    best_preds = (best_probs >= 0.5).astype(int)

    best_pipe.fit(X, y)
    perm = permutation_importance(best_pipe, X, y, n_repeats=15, random_state=42, scoring="roc_auc")
    feat_imp = (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": np.round(perm.importances_mean, 4),
                "importance_std": np.round(perm.importances_std, 4),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    feat_imp.to_parquet(OUTPUTS_DIR / "p29_feature_importance.parquet", index=False)

    # Save trained lightweight pipeline for Streamlit interactive Plan Simulator!
    joblib.dump(best_pipe, OUTPUTS_DIR / "p29_plan_model.joblib")

    ml_metrics = {
        "target_definition": f"unique_responders_2h >= {thresh_resp}",
        "total_plans_modeled": int(len(y)),
        "positive_class_rate_pct": round(float(y.mean() * 100), 2),
        "logistic_regression_cv_roc_auc": lr_auc,
        "random_forest_cv_roc_auc": rf_auc,
        "selected_model": best_name,
        "selected_cv_roc_auc": max(lr_auc, rf_auc),
        "selected_cv_precision": round(float(precision_score(y, best_preds, zero_division=0)), 4),
        "selected_cv_recall": round(float(recall_score(y, best_preds, zero_division=0)), 4),
        "selected_cv_f1": round(float(f1_score(y, best_preds, zero_division=0)), 4),
        "top_feature": str(feat_imp.iloc[0]["feature"]) if not feat_imp.empty else "",
    }
    (OUTPUTS_DIR / "p29_plan_ml_metrics.json").write_text(json.dumps(ml_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    con.close()

    top_jacc = df_jacc[df_jacc["group_a"] < df_jacc["group_b"]].sort_values("jaccard", ascending=False)
    summary = {
        "p20_spearman_core_vs_subgroups": round(float(rho_cs), 4),
        "p21_top_transition": f"{df_trans.iloc[0]['source_group']} -> {df_trans.iloc[0]['target_group']} ({int(df_trans.iloc[0]['transitions'])})" if not df_trans.empty else "N/A",
        "p22_top_jaccard_pair": f"{top_jacc.iloc[0]['label_a']} <-> {top_jacc.iloc[0]['label_b']} (J={top_jacc.iloc[0]['jaccard']})" if not top_jacc.empty else "N/A",
        "p25_total_plans": int(len(df_plans_tr)),
        "p25_high_traction_threshold": int(thresh_resp),
        "p27_spearman_betweenness_vs_traction": round(float(rho_init_bet), 4),
        "p27_spearman_volume_vs_traction": round(float(rho_init_vol), 4),
        "p28_pin_impact_p": p28_stats["mann_whitney_p_value"],
        "p29_best_model": best_name,
        "p29_cv_roc_auc": ml_metrics["selected_cv_roc_auc"],
    }
    (OUTPUTS_DIR / "p20_p29_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[P20-P29] Done! Plans={len(df_plans_tr)}, BestML={best_name} (ROC-AUC={ml_metrics['selected_cv_roc_auc']}), TopPair={summary['p22_top_jaccard_pair']}")
    return summary


if __name__ == "__main__":
    run_multigroup_and_plans_ml()
