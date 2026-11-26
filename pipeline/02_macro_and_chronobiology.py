"""Step 02: Pilar 1 (Macro Health & Inequality) & Pilar 2 (Chronobiology) — P01 to P09."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.config import DUCKDB_PATH, OUTPUTS_DIR


def gini_coefficient(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[arr >= 0]
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * arr)) / (n * np.sum(arr)))


def compute_lorenz_curve(values: np.ndarray, points: int = 100) -> pd.DataFrame:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    arr = arr[arr >= 0]
    if arr.size == 0 or arr.sum() == 0:
        return pd.DataFrame({"pop_pct": [0.0, 100.0], "msg_pct": [0.0, 100.0]})
    cum_pop = np.linspace(0, 100, arr.size + 1)
    cum_val = np.concatenate([[0.0], np.cumsum(arr) / arr.sum() * 100.0])
    grid = np.linspace(0, 100, points + 1)
    interp_val = np.interp(grid, cum_pop, cum_val)
    return pd.DataFrame({"pop_pct": np.round(grid, 2), "msg_pct": np.round(interp_val, 3)})


def run_macro_and_chronobiology(db_path: Path = DUCKDB_PATH) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)

    # =========================================================================
    # P01: Daily scale, DAU / WAU / MAU & Stickiness
    # =========================================================================
    df_daily = con.execute(
        """
        SELECT
            CAST(date_str AS DATE) AS date,
            COUNT(*) AS total_messages,
            SUM(CASE WHEN is_system = 0 THEN 1 ELSE 0 END) AS user_messages,
            SUM(is_media) AS media_messages,
            SUM(is_deleted) AS deleted_messages,
            COUNT(DISTINCT CASE WHEN is_system = 0 THEN member_id END) AS dau
        FROM messages_meta
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    # Compute exact rolling WAU (7d) and MAU (30d) from (date, member_id) pairs
    df_dm = con.execute(
        """
        SELECT DISTINCT CAST(date_str AS DATE) AS date, member_id
        FROM messages_meta
        WHERE is_system = 0 AND member_id IS NOT NULL
        """
    ).df()
    df_dm["date"] = pd.to_datetime(df_dm["date"])

    all_dates = pd.date_range(df_daily["date"].min(), df_daily["date"].max(), freq="D")
    df_daily = pd.DataFrame({"date": all_dates}).merge(df_daily, on="date", how="left").fillna(0)
    for c in ["total_messages", "user_messages", "media_messages", "deleted_messages", "dau"]:
        df_daily[c] = df_daily[c].astype(int)

    date_to_members: dict[pd.Timestamp, set[str]] = (
        df_dm.groupby("date")["member_id"].apply(set).to_dict()
    )
    wau_list, mau_list, cum_list = [], [], []
    seen_cum: set[str] = set()
    for d in df_daily["date"]:
        seen_cum |= date_to_members.get(d, set())
        cum_list.append(len(seen_cum))
        w_set: set[str] = set()
        for k in range(7):
            w_set |= date_to_members.get(d - pd.Timedelta(days=k), set())
        wau_list.append(len(w_set))
        m_set: set[str] = set()
        for k in range(30):
            m_set |= date_to_members.get(d - pd.Timedelta(days=k), set())
        mau_list.append(len(m_set))

    df_daily["wau"] = wau_list
    df_daily["mau"] = mau_list
    df_daily["cum_unique_members"] = cum_list
    df_daily["stickiness_dau_mau"] = np.where(df_daily["mau"] > 0, df_daily["dau"] / df_daily["mau"], 0.0).round(4)
    df_daily["user_msgs_ma7"] = df_daily["user_messages"].rolling(7, min_periods=1).mean().round(2)
    df_daily["dau_ma7"] = df_daily["dau"].rolling(7, min_periods=1).mean().round(2)
    df_daily["date_str"] = df_daily["date"].dt.strftime("%Y-%m-%d")
    df_daily.drop(columns=["date"]).to_parquet(OUTPUTS_DIR / "p01_daily_metrics.parquet", index=False)

    # =========================================================================
    # P02: Participation Inequality (1%-9%-90%, Gini, Lorenz)
    # =========================================================================
    df_mem = con.execute("SELECT member_id, total_messages FROM members_anon ORDER BY total_messages DESC").df()
    vals = df_mem["total_messages"].to_numpy(dtype=np.float64)
    global_gini = round(gini_coefficient(vals), 4)
    n_mem = len(vals)
    top_1_n = max(1, int(np.ceil(n_mem * 0.01)))
    top_10_n = max(1, int(np.ceil(n_mem * 0.10)))
    top_20_n = max(1, int(np.ceil(n_mem * 0.20)))
    share_top1 = round(vals[:top_1_n].sum() / vals.sum() * 100, 2)
    share_next9 = round(vals[top_1_n:top_10_n].sum() / vals.sum() * 100, 2)
    share_bottom90 = round(vals[top_10_n:].sum() / vals.sum() * 100, 2)
    share_top20 = round(vals[:top_20_n].sum() / vals.sum() * 100, 2)
    cum_shares = np.cumsum(vals) / vals.sum()
    pareto_80_members = int(np.searchsorted(cum_shares, 0.80) + 1)

    lorenz_global = compute_lorenz_curve(vals)
    lorenz_global.to_parquet(OUTPUTS_DIR / "p02_lorenz_global.parquet", index=False)

    df_gm = con.execute(
        """
        SELECT group_id, member_id, COUNT(*) AS msgs
        FROM messages_meta
        WHERE is_system = 0 AND member_id IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    df_groups = con.execute("SELECT * FROM groups_anon").df()
    gini_rows = []
    for gid, grp in df_gm.groupby("group_id"):
        v = np.sort(grp["msgs"].to_numpy(dtype=np.float64))[::-1]
        k10 = max(1, int(np.ceil(len(v) * 0.10)))
        k20 = max(1, int(np.ceil(len(v) * 0.20)))
        gini_rows.append(
            {
                "group_id": gid,
                "active_members": int(len(v)),
                "user_messages": int(v.sum()),
                "gini": round(gini_coefficient(v), 4),
                "top_10pct_share": round(v[:k10].sum() / v.sum() * 100, 2),
                "top_20pct_share": round(v[:k20].sum() / v.sum() * 100, 2),
                "median_msgs_per_member": float(np.median(v)),
                "mean_msgs_per_member": round(float(np.mean(v)), 2),
            }
        )
    df_gini_grp = df_groups[["group_id", "group_label", "macro_category"]].merge(
        pd.DataFrame(gini_rows), on="group_id", how="left"
    )
    df_gini_grp.to_parquet(OUTPUTS_DIR / "p02_gini_by_group.parquet", index=False)

    # =========================================================================
    # P03: Net Growth & Group Lifecycle Classification
    # =========================================================================
    df_ev_grp = con.execute(
        """
        SELECT
            group_id,
            SUM(CASE WHEN event_type = 'join' THEN 1 ELSE 0 END) AS joins,
            SUM(CASE WHEN event_type = 'leave' THEN 1 ELSE 0 END) AS leaves,
            SUM(CASE WHEN event_type = 'kick' THEN 1 ELSE 0 END) AS kicks,
            SUM(CASE WHEN event_type = 'pin' THEN 1 ELSE 0 END) AS pins
        FROM events_anon
        GROUP BY 1
        """
    ).df()

    max_date = con.execute("SELECT MAX(CAST(date_str AS DATE)) FROM messages_meta").fetchone()[0]
    df_recent = con.execute(
        f"""
        SELECT
            group_id,
            SUM(CASE WHEN CAST(date_str AS DATE) > DATE '{max_date}' - INTERVAL 14 DAY THEN 1 ELSE 0 END) AS msgs_last_14d,
            SUM(CASE WHEN CAST(date_str AS DATE) <= DATE '{max_date}' - INTERVAL 14 DAY
                      AND CAST(date_str AS DATE) > DATE '{max_date}' - INTERVAL 28 DAY THEN 1 ELSE 0 END) AS msgs_prev_14d
        FROM messages_meta
        WHERE is_system = 0
        GROUP BY 1
        """
    ).df()

    df_health = df_groups.merge(df_ev_grp, on="group_id", how="left").merge(df_recent, on="group_id", how="left")
    for c in ["joins", "leaves", "kicks", "pins", "msgs_last_14d", "msgs_prev_14d"]:
        df_health[c] = df_health[c].fillna(0).astype(int)
    df_health["net_growth"] = df_health["joins"] - df_health["leaves"] - df_health["kicks"]
    df_health["churn_rate_pct"] = np.where(
        df_health["joins"] > 0, (df_health["leaves"] + df_health["kicks"]) / df_health["joins"] * 100, 0.0
    ).round(2)
    df_health["msgs_per_active_day"] = np.where(
        df_health["active_days"] > 0, df_health["user_messages"] / df_health["active_days"], 0.0
    ).round(2)
    df_health["trend_14d_pct"] = np.where(
        df_health["msgs_prev_14d"] > 0,
        (df_health["msgs_last_14d"] - df_health["msgs_prev_14d"]) / df_health["msgs_prev_14d"] * 100,
        np.where(df_health["msgs_last_14d"] > 0, 100.0, 0.0),
    ).round(2)

    def classify_lifecycle(row: pd.Series) -> str:
        if row["macro_category"] == "Efímero" or row["span_days"] <= 7:
            return "Efímero / Explosivo"
        if row["trend_14d_pct"] >= 20:
            return "Expansión"
        if row["trend_14d_pct"] >= -20 and row["msgs_last_14d"] >= 100:
            return "Consolidado / Meseta"
        return "Declive / Baja Tracción"

    df_health["lifecycle_stage"] = df_health.apply(classify_lifecycle, axis=1)
    df_health.to_parquet(OUTPUTS_DIR / "p03_group_health.parquet", index=False)

    # =========================================================================
    # P04: Moderation & Friction vs Volume
    # =========================================================================
    df_mod = con.execute(
        """
        SELECT
            m.date_str,
            COUNT(*) AS total_msgs,
            SUM(m.is_deleted) AS deleted_msgs,
            COALESCE(e.kicks, 0) AS kicks,
            COALESCE(e.pins, 0) AS pins,
            COALESCE(e.leaves, 0) AS leaves
        FROM messages_meta m
        LEFT JOIN (
            SELECT
                date_str,
                SUM(CASE WHEN event_type = 'kick' THEN 1 ELSE 0 END) AS kicks,
                SUM(CASE WHEN event_type = 'pin' THEN 1 ELSE 0 END) AS pins,
                SUM(CASE WHEN event_type = 'leave' THEN 1 ELSE 0 END) AS leaves
            FROM events_anon
            GROUP BY 1
        ) e ON m.date_str = e.date_str
        GROUP BY 1, 4, 5, 6
        ORDER BY 1
        """
    ).df()
    df_mod["deleted_per_1k"] = np.where(df_mod["total_msgs"] > 0, df_mod["deleted_msgs"] / df_mod["total_msgs"] * 1000, 0.0).round(2)
    df_mod["friction_events"] = df_mod["deleted_msgs"] + df_mod["kicks"] + df_mod["leaves"]
    df_mod.to_parquet(OUTPUTS_DIR / "p04_moderation_daily.parquet", index=False)

    rho_del, p_del = spearmanr(df_mod["total_msgs"], df_mod["deleted_msgs"])
    rho_fric, p_fric = spearmanr(df_mod["total_msgs"], df_mod["friction_events"])

    # =========================================================================
    # P05: Activation Funnel (Lurkers vs Active Members)
    # =========================================================================
    total_join_events = int(con.execute("SELECT COUNT(*) FROM events_anon WHERE event_type = 'join'").fetchone()[0])
    core_joins = int(
        con.execute("SELECT COUNT(*) FROM events_anon WHERE event_type = 'join' AND group_id = 'G01_CORE_GENERAL'").fetchone()[0]
    )
    df_mfull = con.execute("SELECT * FROM members_anon").df()
    funnel_rows = [
        {"stage_order": 1, "stage": "Ingresos Grupo Matriz (Joins)", "count": core_joins, "pct_of_joins": 100.0},
        {"stage_order": 2, "stage": "Emisores Activos (>=1 msg)", "count": int(len(df_mfull)), "pct_of_joins": round(len(df_mfull) / max(core_joins, 1) * 100, 2)},
        {"stage_order": 3, "stage": "Activados (>=5 msgs)", "count": int((df_mfull["total_messages"] >= 5).sum()), "pct_of_joins": round((df_mfull["total_messages"] >= 5).sum() / max(core_joins, 1) * 100, 2)},
        {"stage_order": 4, "stage": "Recurrentes (>=20 msgs)", "count": int((df_mfull["total_messages"] >= 20).sum()), "pct_of_joins": round((df_mfull["total_messages"] >= 20).sum() / max(core_joins, 1) * 100, 2)},
        {"stage_order": 5, "stage": "Retenidos (>7 días vida)", "count": int((df_mfull["lifespan_days"] > 7).sum()), "pct_of_joins": round((df_mfull["lifespan_days"] > 7).sum() / max(core_joins, 1) * 100, 2)},
        {"stage_order": 6, "stage": "Multigrupo (>=2 grupos)", "count": int((df_mfull["groups_count"] >= 2).sum()), "pct_of_joins": round((df_mfull["groups_count"] >= 2).sum() / max(core_joins, 1) * 100, 2)},
        {"stage_order": 7, "stage": "Power Users (>=100 msgs)", "count": int((df_mfull["total_messages"] >= 100).sum()), "pct_of_joins": round((df_mfull["total_messages"] >= 100).sum() / max(core_joins, 1) * 100, 2)},
    ]
    pd.DataFrame(funnel_rows).to_parquet(OUTPUTS_DIR / "p05_activation_funnel.parquet", index=False)

    # =========================================================================
    # P06: Global & Per-Group 7x24 Chronobiology Heatmap
    # =========================================================================
    df_hm = con.execute(
        """
        SELECT
            group_id,
            day_of_week,
            hour,
            COUNT(*) AS messages,
            COUNT(DISTINCT member_id) AS unique_members,
            SUM(is_media) AS media_messages,
            ROUND(AVG(sentiment_score), 4) AS avg_sentiment
        FROM messages_meta
        WHERE is_system = 0
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).df()
    df_hm.to_parquet(OUTPUTS_DIR / "p06_heatmap_7x24.parquet", index=False)

    # =========================================================================
    # P07: Circadian Profiles by Group & Macro Category
    # =========================================================================
    df_circ = con.execute(
        """
        SELECT
            m.group_id,
            g.group_label,
            g.macro_category,
            m.hour,
            COUNT(*) AS messages
        FROM messages_meta m
        JOIN groups_anon g ON m.group_id = g.group_id
        WHERE m.is_system = 0
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 4
        """
    ).df()
    grp_totals = df_circ.groupby("group_id")["messages"].transform("sum")
    df_circ["hour_share_pct"] = (df_circ["messages"] / grp_totals * 100).round(3)
    df_circ.to_parquet(OUTPUTS_DIR / "p07_group_circadian_profiles.parquet", index=False)

    # Summary circadian indicators per group
    circ_summary = []
    for (gid, glabel, macro), gdf in df_circ.groupby(["group_id", "group_label", "macro_category"]):
        peak_h = int(gdf.loc[gdf["messages"].idxmax(), "hour"])
        night = float(gdf[gdf["hour"].isin([22, 23, 0, 1, 2, 3, 4])]["hour_share_pct"].sum())
        morning = float(gdf[gdf["hour"].isin([5, 6, 7, 8, 9, 10, 11])]["hour_share_pct"].sum())
        afternoon = float(gdf[gdf["hour"].isin([12, 13, 14, 15, 16, 17])]["hour_share_pct"].sum())
        prime = float(gdf[gdf["hour"].isin([18, 19, 20, 21])]["hour_share_pct"].sum())
        circ_summary.append(
            {
                "group_id": gid,
                "group_label": glabel,
                "macro_category": macro,
                "peak_hour": peak_h,
                "morning_share_pct": round(morning, 2),
                "afternoon_share_pct": round(afternoon, 2),
                "prime_evening_share_pct": round(prime, 2),
                "late_night_share_pct": round(night, 2),
            }
        )
    pd.DataFrame(circ_summary).to_parquet(OUTPUTS_DIR / "p07_circadian_summary.parquet", index=False)

    # =========================================================================
    # P08: First Response Time (FRT) via DuckDB LAG()
    # =========================================================================
    df_frt_hour = con.execute(
        """
        WITH ordered AS (
            SELECT
                group_id,
                hour,
                member_id,
                CAST(dt_iso AS TIMESTAMP) AS ts,
                LAG(member_id) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS prev_member,
                LAG(CAST(dt_iso AS TIMESTAMP)) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS prev_ts
            FROM messages_meta
            WHERE is_system = 0 AND member_id IS NOT NULL
        ),
        replies AS (
            SELECT
                group_id,
                hour,
                EXTRACT(EPOCH FROM (ts - prev_ts)) AS delta_sec
            FROM ordered
            WHERE prev_member IS NOT NULL
              AND member_id != prev_member
              AND EXTRACT(EPOCH FROM (ts - prev_ts)) >= 0
              AND EXTRACT(EPOCH FROM (ts - prev_ts)) <= 7200
        )
        SELECT
            hour,
            COUNT(*) AS reply_pairs,
            ROUND(MEDIAN(delta_sec), 1) AS median_frt_sec,
            ROUND(AVG(delta_sec), 1) AS mean_frt_sec,
            ROUND(QUANTILE_CONT(delta_sec, 0.25), 1) AS p25_frt_sec,
            ROUND(QUANTILE_CONT(delta_sec, 0.75), 1) AS p75_frt_sec,
            ROUND(QUANTILE_CONT(delta_sec, 0.90), 1) AS p90_frt_sec
        FROM replies
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    df_frt_hour.to_parquet(OUTPUTS_DIR / "p08_frt_by_hour.parquet", index=False)

    df_frt_grp = con.execute(
        """
        WITH ordered AS (
            SELECT
                group_id,
                member_id,
                CAST(dt_iso AS TIMESTAMP) AS ts,
                LAG(member_id) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS prev_member,
                LAG(CAST(dt_iso AS TIMESTAMP)) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS prev_ts
            FROM messages_meta
            WHERE is_system = 0 AND member_id IS NOT NULL
        ),
        replies AS (
            SELECT
                group_id,
                EXTRACT(EPOCH FROM (ts - prev_ts)) AS delta_sec
            FROM ordered
            WHERE prev_member IS NOT NULL
              AND member_id != prev_member
              AND EXTRACT(EPOCH FROM (ts - prev_ts)) >= 0
              AND EXTRACT(EPOCH FROM (ts - prev_ts)) <= 7200
        )
        SELECT
            r.group_id,
            g.group_label,
            g.macro_category,
            COUNT(*) AS reply_pairs,
            ROUND(MEDIAN(r.delta_sec), 1) AS median_frt_sec,
            ROUND(AVG(r.delta_sec), 1) AS mean_frt_sec,
            ROUND(QUANTILE_CONT(r.delta_sec, 0.75), 1) AS p75_frt_sec,
            ROUND(SUM(CASE WHEN r.delta_sec <= 60 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_under_1min,
            ROUND(SUM(CASE WHEN r.delta_sec <= 300 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_under_5min
        FROM replies r
        JOIN groups_anon g ON r.group_id = g.group_id
        GROUP BY 1, 2, 3
        ORDER BY median_frt_sec ASC
        """
    ).df()
    df_frt_grp.to_parquet(OUTPUTS_DIR / "p08_frt_by_group.parquet", index=False)

    # =========================================================================
    # P09: Activity Bursts, Hangover Effect & Autocorrelation (ACF)
    # =========================================================================
    y = df_daily["user_messages"].to_numpy(dtype=np.float64)
    y_mean, y_std = float(y.mean()), float(y.std(ddof=1)) if len(y) > 1 else 1.0
    df_bursts = df_daily[["date_str", "user_messages", "dau"]].copy()
    df_bursts["z_score"] = np.round((y - y_mean) / max(y_std, 1e-6), 3)
    df_bursts["is_burst_day"] = (df_bursts["z_score"] >= 1.5).astype(int)
    df_bursts.to_parquet(OUTPUTS_DIR / "p09_bursts_daily.parquet", index=False)

    # Compute ACF lags 1..14
    acf_rows = []
    y_centered = y - y_mean
    denom = float(np.dot(y_centered, y_centered))
    for lag in range(1, min(15, len(y) // 2)):
        val = float(np.dot(y_centered[:-lag], y_centered[lag:]) / denom) if denom > 0 else 0.0
        acf_rows.append({"lag_days": lag, "acf": round(val, 4)})
    pd.DataFrame(acf_rows).to_parquet(OUTPUTS_DIR / "p09_acf.parquet", index=False)

    # Post-burst trajectory (relative to burst day = 100%)
    burst_indices = np.where(df_bursts["is_burst_day"].to_numpy() == 1)[0]
    traj_rows = []
    for offset in [-2, -1, 0, 1, 2, 3]:
        ratios = []
        for idx in burst_indices:
            pos = idx + offset
            if 0 <= pos < len(y) and y[idx] > 0:
                ratios.append(y[pos] / y[idx] * 100.0)
        traj_rows.append(
            {
                "day_offset": offset,
                "avg_pct_of_burst": round(float(np.mean(ratios)), 2) if ratios else 0.0,
                "median_pct_of_burst": round(float(np.median(ratios)), 2) if ratios else 0.0,
                "sample_bursts": len(ratios),
            }
        )
    pd.DataFrame(traj_rows).to_parquet(OUTPUTS_DIR / "p09_burst_hangover.parquet", index=False)

    con.close()

    summary = {
        "p01_avg_dau": round(float(df_daily["dau"].mean()), 1),
        "p01_peak_dau": int(df_daily["dau"].max()),
        "p01_avg_stickiness": round(float(df_daily[df_daily["mau"] > 0]["stickiness_dau_mau"].mean()), 4),
        "p02_global_gini": global_gini,
        "p02_share_top_1pct": share_top1,
        "p02_share_next_9pct": share_next9,
        "p02_share_bottom_90pct": share_bottom90,
        "p02_share_top_20pct": share_top20,
        "p02_pareto_80_members": pareto_80_members,
        "p04_spearman_vol_vs_deleted": round(float(rho_del), 4),
        "p04_p_vol_vs_deleted": round(float(p_del), 6),
        "p04_spearman_vol_vs_friction": round(float(rho_fric), 4),
        "p05_total_join_events": total_join_events,
        "p05_core_joins": core_joins,
        "p05_activation_pct_vs_core_joins": funnel_rows[1]["pct_of_joins"],
        "p06_peak_hour_global": int(df_hm.groupby("hour")["messages"].sum().idxmax()),
        "p08_median_frt_global_sec": float(df_frt_hour["median_frt_sec"].median()),
        "p09_burst_days_count": int(len(burst_indices)),
        "p09_acf_lag1": acf_rows[0]["acf"] if acf_rows else 0.0,
        "p09_next_day_pct_of_burst": next((r["avg_pct_of_burst"] for r in traj_rows if r["day_offset"] == 1), 0.0),
    }
    (OUTPUTS_DIR / "p01_p09_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[P01-P09] Done! Gini={global_gini}, Top10%={share_top1 + share_next9:.1f}%, PeakHour={summary['p06_peak_hour_global']}h, ACF1={summary['p09_acf_lag1']}")
    return summary


if __name__ == "__main__":
    run_macro_and_chronobiology()
