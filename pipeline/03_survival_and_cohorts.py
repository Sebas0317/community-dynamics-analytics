"""Step 03: Pilar 3 (Survival Analysis, Cohorts, Onboarding Hypotheses & Purge Impact) — P10 to P14."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu

from pipeline.config import CORE_GROUP_ID, DUCKDB_PATH, ONBOARDING_GROUP_ID, OUTPUTS_DIR, PURGE_DATE_ISO


def rank_biserial_effect(u_stat: float, n1: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 0.0
    return round(float(1.0 - (2.0 * u_stat) / (n1 * n2)), 4)


def run_survival_and_cohorts(db_path: Path = DUCKDB_PATH) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)

    df_mem = con.execute("SELECT * FROM members_anon").df()
    df_mem["first_ts"] = pd.to_datetime(df_mem["first_dt"])
    df_mem["last_ts"] = pd.to_datetime(df_mem["last_dt"])
    max_ts = pd.to_datetime(con.execute("SELECT MAX(dt_iso) FROM messages_meta").fetchone()[0])

    # Identify members who explicitly left or were kicked
    exited_members = set(
        con.execute(
            "SELECT DISTINCT member_id FROM events_anon WHERE event_type IN ('leave', 'kick') AND member_id IS NOT NULL"
        ).df()["member_id"]
    )

    # Event observed (churned = 1) if inactive >= 10 days before snapshot cutoff OR explicit exit
    inactivity_days = (max_ts - df_mem["last_ts"]).dt.total_seconds() / 86400.0
    df_mem["duration_days"] = np.maximum(0.5, (df_mem["last_ts"] - df_mem["first_ts"]).dt.total_seconds() / 86400.0)
    df_mem["event_churned"] = ((inactivity_days >= 10.0) | (df_mem["member_id"].isin(exited_members))).astype(int)

    # =========================================================================
    # P10: Global Kaplan-Meier Survival Curve & Weekly Hazard
    # =========================================================================
    kmf = KaplanMeierFitter()
    kmf.fit(df_mem["duration_days"], event_observed=df_mem["event_churned"], label="Global")
    km_df = pd.DataFrame(
        {
            "timeline_days": kmf.survival_function_.index.astype(float),
            "survival_prob": kmf.survival_function_["Global"].values.astype(float),
            "ci_lower": kmf.confidence_interval_survival_function_.iloc[:, 0].values.astype(float),
            "ci_upper": kmf.confidence_interval_survival_function_.iloc[:, 1].values.astype(float),
        }
    )
    km_df.to_parquet(OUTPUTS_DIR / "p10_kaplan_meier_global.parquet", index=False)

    median_survival = float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else float(df_mem["duration_days"].median())
    surv_d7 = float(kmf.predict(7))
    surv_d14 = float(kmf.predict(14))
    surv_d30 = float(kmf.predict(30))

    # =========================================================================
    # P11: Weekly Cohort Retention Matrix
    # =========================================================================
    df_act = con.execute(
        """
        SELECT DISTINCT member_id, CAST(date_str AS DATE) AS act_date
        FROM messages_meta
        WHERE is_system = 0 AND member_id IS NOT NULL
        """
    ).df()
    df_act["act_date"] = pd.to_datetime(df_act["act_date"])

    mem_first = df_mem[["member_id", "first_ts"]].copy()
    mem_first["cohort_week"] = mem_first["first_ts"].dt.to_period("W").apply(lambda p: p.start_time.strftime("%Y-%m-%d"))
    mem_first["first_date"] = mem_first["first_ts"].dt.normalize()

    df_act = df_act.merge(mem_first, on="member_id", how="inner")
    df_act["days_since_first"] = (df_act["act_date"] - df_act["first_date"]).dt.days
    df_act["week_index"] = (df_act["days_since_first"] // 7).astype(int)

    cohort_sizes = mem_first.groupby("cohort_week")["member_id"].nunique().rename("cohort_size")
    coh_counts = df_act.groupby(["cohort_week", "week_index"])["member_id"].nunique().reset_index(name="active_members")
    coh_counts = coh_counts.merge(cohort_sizes.reset_index(), on="cohort_week", how="left")
    coh_counts["retention_pct"] = (coh_counts["active_members"] / coh_counts["cohort_size"] * 100).round(2)
    coh_counts.to_parquet(OUTPUTS_DIR / "p11_weekly_cohorts.parquet", index=False)

    # Milestone retention rates (D1, D7, D14, D30, D60)
    milestone_rows = []
    for d in [1, 7, 14, 30, 60]:
        eligible = mem_first[(max_ts.normalize() - mem_first["first_date"]).dt.days >= d]
        if len(eligible) == 0:
            continue
        retained_ids = set(df_act[df_act["days_since_first"] >= d]["member_id"])
        ret_n = int(eligible["member_id"].isin(retained_ids).sum())
        milestone_rows.append(
            {
                "milestone_day": d,
                "eligible_members": int(len(eligible)),
                "retained_members": ret_n,
                "retention_pct": round(ret_n / len(eligible) * 100, 2),
            }
        )
    pd.DataFrame(milestone_rows).to_parquet(OUTPUTS_DIR / "p11_milestone_retention.parquet", index=False)

    # =========================================================================
    # P12: Onboarding Hypothesis Test (Posted in G03_ONBOARDING_CENSO vs Not)
    # =========================================================================
    onb_members = set(
        con.execute(
            f"SELECT DISTINCT member_id FROM messages_meta WHERE group_id = '{ONBOARDING_GROUP_ID}' AND is_system = 0 AND member_id IS NOT NULL"
        ).df()["member_id"]
    )
    df_mem["did_onboarding"] = df_mem["member_id"].isin(onb_members).astype(int)
    grp_a = df_mem[df_mem["did_onboarding"] == 1]
    grp_b = df_mem[df_mem["did_onboarding"] == 0]

    km_a = KaplanMeierFitter().fit(grp_a["duration_days"], event_observed=grp_a["event_churned"], label="Con Onboarding")
    km_b = KaplanMeierFitter().fit(grp_b["duration_days"], event_observed=grp_b["event_churned"], label="Sin Onboarding")

    km_onb = pd.concat(
        [
            pd.DataFrame(
                {
                    "group": "Con Onboarding",
                    "timeline_days": km_a.survival_function_.index.astype(float),
                    "survival_prob": km_a.survival_function_.iloc[:, 0].values.astype(float),
                    "ci_lower": km_a.confidence_interval_survival_function_.iloc[:, 0].values.astype(float),
                    "ci_upper": km_a.confidence_interval_survival_function_.iloc[:, 1].values.astype(float),
                }
            ),
            pd.DataFrame(
                {
                    "group": "Sin Onboarding",
                    "timeline_days": km_b.survival_function_.index.astype(float),
                    "survival_prob": km_b.survival_function_.iloc[:, 0].values.astype(float),
                    "ci_lower": km_b.confidence_interval_survival_function_.iloc[:, 0].values.astype(float),
                    "ci_upper": km_b.confidence_interval_survival_function_.iloc[:, 1].values.astype(float),
                }
            ),
        ],
        ignore_index=True,
    )
    km_onb.to_parquet(OUTPUTS_DIR / "p12_onboarding_km.parquet", index=False)

    lr = logrank_test(
        grp_a["duration_days"], grp_b["duration_days"],
        event_observed_A=grp_a["event_churned"], event_observed_B=grp_b["event_churned"],
    )
    onb_tests = {}
    for metric in ["total_messages", "active_days", "lifespan_days", "groups_count"]:
        u_stat, p_val = mannwhitneyu(grp_a[metric], grp_b[metric], alternative="two-sided")
        onb_tests[metric] = {
            "median_with_onboarding": round(float(grp_a[metric].median()), 2),
            "median_without_onboarding": round(float(grp_b[metric].median()), 2),
            "mean_with_onboarding": round(float(grp_a[metric].mean()), 2),
            "mean_without_onboarding": round(float(grp_b[metric].mean()), 2),
            "u_stat": round(float(u_stat), 1),
            "p_value": float(f"{p_val:.6g}"),
            "effect_size_rb": rank_biserial_effect(float(u_stat), len(grp_a), len(grp_b)),
        }
    onb_stats = {
        "n_with_onboarding": int(len(grp_a)),
        "n_without_onboarding": int(len(grp_b)),
        "logrank_stat": round(float(lr.test_statistic), 4),
        "logrank_p_value": float(f"{lr.p_value:.6g}"),
        "median_survival_with_onb": round(float(km_a.median_survival_time_), 2) if np.isfinite(km_a.median_survival_time_) else None,
        "median_survival_without_onb": round(float(km_b.median_survival_time_), 2) if np.isfinite(km_b.median_survival_time_) else None,
        "metrics_mann_whitney": onb_tests,
    }
    (OUTPUTS_DIR / "p12_onboarding_stats.json").write_text(json.dumps(onb_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # =========================================================================
    # P13: First Impression Effect (Reply speed to a member's very first message)
    # =========================================================================
    df_first_imp = con.execute(
        """
        WITH first_msg AS (
            SELECT member_id, group_id, dt_iso AS first_dt, msg_index AS first_idx
            FROM (
                SELECT
                    member_id, group_id, dt_iso, msg_index,
                    ROW_NUMBER() OVER (PARTITION BY member_id ORDER BY dt_iso, msg_index) AS rn
                FROM messages_meta
                WHERE is_system = 0 AND member_id IS NOT NULL
            )
            WHERE rn = 1
        ),
        next_reply AS (
            SELECT
                f.member_id,
                MIN(EXTRACT(EPOCH FROM (CAST(m.dt_iso AS TIMESTAMP) - CAST(f.first_dt AS TIMESTAMP)))) AS first_reply_sec
            FROM first_msg f
            LEFT JOIN messages_meta m
              ON m.group_id = f.group_id
             AND m.is_system = 0
             AND m.member_id IS NOT NULL
             AND m.member_id != f.member_id
             AND m.dt_iso >= f.first_dt
             AND EXTRACT(EPOCH FROM (CAST(m.dt_iso AS TIMESTAMP) - CAST(f.first_dt AS TIMESTAMP))) BETWEEN 0 AND 86400
            GROUP BY 1
        )
        SELECT
            mem.member_id,
            mem.total_messages,
            mem.active_days,
            mem.lifespan_days,
            mem.groups_count,
            nr.first_reply_sec
        FROM members_anon mem
        LEFT JOIN next_reply nr ON mem.member_id = nr.member_id
        """
    ).df()

    def bucket_reply(sec: float | None) -> str:
        if pd.isna(sec) or sec > 3600:
            return "4. >60 min / Sin respuesta"
        if sec <= 120:
            return "1. <2 min (Inmediata)"
        if sec <= 600:
            return "2. 2–10 min (Rápida)"
        return "3. 10–60 min (Moderada)"

    df_first_imp["reply_bucket"] = df_first_imp["first_reply_sec"].map(bucket_reply)
    df_first_imp["retained_d7"] = (df_first_imp["lifespan_days"] >= 7).astype(int)
    df_first_imp["retained_d14"] = (df_first_imp["lifespan_days"] >= 14).astype(int)

    p13_agg = (
        df_first_imp.groupby("reply_bucket", as_index=False)
        .agg(
            members=("member_id", "count"),
            retention_d7_pct=("retained_d7", lambda s: round(float(s.mean() * 100), 2)),
            retention_d14_pct=("retained_d14", lambda s: round(float(s.mean() * 100), 2)),
            median_lifespan_days=("lifespan_days", "median"),
            median_total_msgs=("total_messages", "median"),
            mean_groups=("groups_count", lambda s: round(float(s.mean()), 2)),
        )
        .sort_values("reply_bucket")
    )
    p13_agg.to_parquet(OUTPUTS_DIR / "p13_first_impression.parquet", index=False)

    fast_g = df_first_imp[df_first_imp["first_reply_sec"].notna() & (df_first_imp["first_reply_sec"] <= 600)]
    slow_g = df_first_imp[df_first_imp["first_reply_sec"].isna() | (df_first_imp["first_reply_sec"] > 600)]
    u_fi, p_fi = mannwhitneyu(fast_g["lifespan_days"], slow_g["lifespan_days"], alternative="two-sided") if (len(fast_g) > 0 and len(slow_g) > 0) else (0.0, 1.0)

    # =========================================================================
    # P14: Before/After Structural Impact of Halloween Inactive Purge (2026-10-31)
    # =========================================================================
    df_purge = con.execute(
        f"""
        SELECT
            CAST(date_str AS DATE) AS date,
            date_str,
            COUNT(*) AS total_msgs_global,
            SUM(CASE WHEN group_id = '{CORE_GROUP_ID}' THEN 1 ELSE 0 END) AS core_msgs,
            COUNT(DISTINCT member_id) AS dau_global,
            COUNT(DISTINCT CASE WHEN group_id = '{CORE_GROUP_ID}' THEN member_id END) AS dau_core,
            ROUND(AVG(sentiment_score), 4) AS avg_sentiment
        FROM messages_meta
        WHERE is_system = 0
          AND CAST(date_str AS DATE) BETWEEN DATE '{PURGE_DATE_ISO}' - INTERVAL 14 DAY
                                         AND DATE '{PURGE_DATE_ISO}' + INTERVAL 14 DAY
        GROUP BY 1, 2
        ORDER BY 1
        """
    ).df()
    purge_dt = pd.Timestamp(PURGE_DATE_ISO)
    df_purge["period"] = np.where(pd.to_datetime(df_purge["date"]) < purge_dt, "Antes (-14d)", np.where(pd.to_datetime(df_purge["date"]) == purge_dt, "Día Depuración", "Después (+14d)"))
    df_purge.drop(columns=["date"]).to_parquet(OUTPUTS_DIR / "p14_purge_impact_daily.parquet", index=False)

    bef = df_purge[df_purge["period"] == "Antes (-14d)"]
    aft = df_purge[df_purge["period"] == "Después (+14d)"]
    purge_stats = {
        "purge_date": PURGE_DATE_ISO,
        "before_days": int(len(bef)),
        "after_days": int(len(aft)),
    }
    for col in ["total_msgs_global", "core_msgs", "dau_global", "dau_core"]:
        b_mean = float(bef[col].mean()) if len(bef) else 0.0
        a_mean = float(aft[col].mean()) if len(aft) else 0.0
        u_p, p_p = mannwhitneyu(bef[col], aft[col], alternative="two-sided") if (len(bef) > 1 and len(aft) > 1) else (0.0, 1.0)
        purge_stats[col] = {
            "before_mean": round(b_mean, 2),
            "after_mean": round(a_mean, 2),
            "delta_pct": round((a_mean - b_mean) / max(b_mean, 1e-6) * 100, 2),
            "p_value": float(f"{p_p:.6g}"),
        }
    (OUTPUTS_DIR / "p14_purge_stats.json").write_text(json.dumps(purge_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    con.close()

    summary = {
        "p10_median_survival_days": round(median_survival, 2),
        "p10_survival_d7_pct": round(surv_d7 * 100, 2),
        "p10_survival_d14_pct": round(surv_d14 * 100, 2),
        "p10_survival_d30_pct": round(surv_d30 * 100, 2),
        "p12_onboarding_logrank_p": onb_stats["logrank_p_value"],
        "p12_median_msgs_with_onb": onb_tests["total_messages"]["median_with_onboarding"],
        "p12_median_msgs_without_onb": onb_tests["total_messages"]["median_without_onboarding"],
        "p13_fast_reply_n": int(len(fast_g)),
        "p13_slow_reply_n": int(len(slow_g)),
        "p13_lifespan_p_value": float(f"{p_fi:.6g}"),
        "p14_global_msgs_delta_pct": purge_stats["total_msgs_global"]["delta_pct"],
        "p14_core_msgs_delta_pct": purge_stats["core_msgs"]["delta_pct"],
    }
    (OUTPUTS_DIR / "p10_p14_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[P10-P14] Done! MedianSurvival={summary['p10_median_survival_days']}d, Onb LogRank p={summary['p12_onboarding_logrank_p']}, Purge Global Delta={summary['p14_global_msgs_delta_pct']}%")
    return summary


if __name__ == "__main__":
    run_survival_and_cohorts()
