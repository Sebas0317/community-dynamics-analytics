"""Step 04: Pilar 4 (Social Network Analysis, Louvain, Silent Bridges, Percolation & Reciprocity) — P15 to P19."""

from __future__ import annotations

import json
import random
from pathlib import Path

import community as community_louvain
import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.config import DUCKDB_PATH, OUTPUTS_DIR, REPLY_WINDOW_SECONDS


def run_social_graph_sna(db_path: Path = DUCKDB_PATH) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)

    # =========================================================================
    # Build weighted directed interaction graph (delta_t <= 180s in same group)
    # =========================================================================
    df_edges = con.execute(
        f"""
        WITH ordered AS (
            SELECT
                group_id,
                member_id AS target,
                CAST(dt_iso AS TIMESTAMP) AS ts,
                LAG(member_id) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS source,
                LAG(CAST(dt_iso AS TIMESTAMP)) OVER (PARTITION BY group_id ORDER BY dt_iso, msg_index) AS prev_ts
            FROM messages_meta
            WHERE is_system = 0 AND member_id IS NOT NULL
        ),
        pairs AS (
            SELECT
                source,
                target,
                group_id,
                EXTRACT(EPOCH FROM (ts - prev_ts)) AS delta_sec
            FROM ordered
            WHERE source IS NOT NULL
              AND source != target
              AND EXTRACT(EPOCH FROM (ts - prev_ts)) BETWEEN 0 AND {REPLY_WINDOW_SECONDS}
        )
        SELECT
            source,
            target,
            COUNT(*) AS interaction_count,
            ROUND(SUM(1.0 / (1.0 + delta_sec / 60.0)), 4) AS weight,
            COUNT(DISTINCT group_id) AS shared_groups_interacted
        FROM pairs
        GROUP BY 1, 2
        """
    ).df()

    df_mem = con.execute("SELECT * FROM members_anon").df()
    # Dominant group per member
    df_dom_grp = con.execute(
        """
        SELECT member_id, group_id AS primary_group_id
        FROM (
            SELECT
                member_id, group_id, COUNT(*) AS cnt,
                ROW_NUMBER() OVER (PARTITION BY member_id ORDER BY COUNT(*) DESC) AS rn
            FROM messages_meta
            WHERE is_system = 0 AND member_id IS NOT NULL
            GROUP BY 1, 2
        )
        WHERE rn = 1
        """
    ).df()
    df_groups = con.execute("SELECT group_id, group_label, macro_category FROM groups_anon").df()
    df_mem = df_mem.merge(df_dom_grp, on="member_id", how="left").merge(
        df_groups.rename(columns={"group_id": "primary_group_id", "group_label": "primary_group_label"}),
        on="primary_group_id",
        how="left",
    )

    G_dir = nx.DiGraph()
    for mid in df_mem["member_id"]:
        G_dir.add_node(mid)
    for r in df_edges.itertuples(index=False):
        G_dir.add_edge(r.source, r.target, weight=float(r.weight), count=int(r.interaction_count))

    # Undirected weighted projection for Louvain & percolation (edges with >= 2 interactions for clean backbone, fallback >=1)
    G_und = nx.Graph()
    for mid in df_mem["member_id"]:
        G_und.add_node(mid)
    for u, v, d in G_dir.edges(data=True):
        w = d["weight"]
        c = d["count"]
        if G_und.has_edge(u, v):
            G_und[u][v]["weight"] += w
            G_und[u][v]["count"] += c
        else:
            G_und.add_edge(u, v, weight=w, count=c)

    # =========================================================================
    # P15: Louvain Community Detection & Modularity Q
    # =========================================================================
    G_active = G_und.subgraph([n for n, deg in G_und.degree() if deg > 0]).copy()
    partition = community_louvain.best_partition(G_active, weight="weight", random_state=42)
    modularity_q = round(float(community_louvain.modularity(partition, G_active, weight="weight")), 4)

    # Relabel clusters by descending size (0 = largest)
    cluster_counts = pd.Series(partition).value_counts()
    cluster_remap = {old_c: new_i for new_i, old_c in enumerate(cluster_counts.index)}
    partition = {n: cluster_remap[c] for n, c in partition.items()}

    # =========================================================================
    # P16: Centralities & Silent Bridges Detection
    # =========================================================================
    # Use inverse weight distance for weighted betweenness on active component
    for u, v, d in G_active.edges(data=True):
        d["distance"] = 1.0 / max(d["weight"], 1e-6)

    betweenness = nx.betweenness_centrality(G_active, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(G_active, distance="distance")
    pagerank = nx.pagerank(G_dir, weight="weight")
    clustering_node = nx.clustering(G_active, weight="weight")

    in_deg = dict(G_dir.in_degree())
    out_deg = dict(G_dir.out_degree())
    in_w = dict(G_dir.in_degree(weight="count"))
    out_w = dict(G_dir.out_degree(weight="count"))

    df_nodes = df_mem[
        ["member_id", "total_messages", "groups_count", "lifespan_days", "active_days", "first_dt", "primary_group_id", "primary_group_label", "macro_category"]
    ].copy()
    df_nodes["louvain_cluster"] = df_nodes["member_id"].map(partition).fillna(-1).astype(int)
    df_nodes["degree_undirected"] = df_nodes["member_id"].map(dict(G_und.degree())).fillna(0).astype(int)
    df_nodes["in_degree"] = df_nodes["member_id"].map(in_deg).fillna(0).astype(int)
    df_nodes["out_degree"] = df_nodes["member_id"].map(out_deg).fillna(0).astype(int)
    df_nodes["in_interactions"] = df_nodes["member_id"].map(in_w).fillna(0).astype(int)
    df_nodes["out_interactions"] = df_nodes["member_id"].map(out_w).fillna(0).astype(int)
    df_nodes["betweenness"] = df_nodes["member_id"].map(betweenness).fillna(0.0).round(6)
    df_nodes["closeness"] = df_nodes["member_id"].map(closeness).fillna(0.0).round(6)
    df_nodes["pagerank"] = df_nodes["member_id"].map(pagerank).fillna(0.0).round(6)
    df_nodes["local_clustering"] = df_nodes["member_id"].map(clustering_node).fillna(0.0).round(4)

    # Quadrant classification: Volume vs Betweenness (using 75th percentile thresholds on active nodes)
    active_mask = df_nodes["degree_undirected"] > 0
    vol_p75 = float(df_nodes.loc[active_mask, "total_messages"].quantile(0.75))
    bet_p75 = float(df_nodes.loc[active_mask, "betweenness"].quantile(0.75))

    def tag_role(row: pd.Series) -> str:
        hi_v = row["total_messages"] >= vol_p75
        hi_b = row["betweenness"] >= bet_p75 and row["betweenness"] > 0
        if hi_v and hi_b:
            return "Conector Clave (Alto Vol + Alto Puente)"
        if (not hi_v) and hi_b:
            return "Puente Silencioso (Bajo Vol + Alto Puente)"
        if hi_v and (not hi_b):
            return "Megáfono Local (Alto Vol + Bajo Puente)"
        return "Participante Regular / Periferia"

    df_nodes["sna_quadrant"] = df_nodes.apply(tag_role, axis=1)
    df_nodes.to_parquet(OUTPUTS_DIR / "p15_sna_nodes.parquet", index=False)

    # Save top edges for fast PyVis / Plotly interactive rendering
    df_edges_out = (
        df_edges.rename(columns={"source": "source_member_id", "target": "target_member_id"})
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    df_edges_out.to_parquet(OUTPUTS_DIR / "p15_sna_edges.parquet", index=False)

    # Community summary table
    comm_rows = []
    for cid, cdf in df_nodes[df_nodes["louvain_cluster"] >= 0].groupby("louvain_cluster"):
        sub_nodes = list(cdf["member_id"])
        sub_g = G_active.subgraph(sub_nodes)
        dom_grp = cdf["primary_group_label"].mode().iloc[0] if not cdf["primary_group_label"].dropna().empty else "Mixto"
        comm_rows.append(
            {
                "louvain_cluster": int(cid),
                "members_count": int(len(cdf)),
                "total_messages": int(cdf["total_messages"].sum()),
                "avg_groups_per_member": round(float(cdf["groups_count"].mean()), 2),
                "internal_density": round(float(nx.density(sub_g)), 4),
                "dominant_group_label": str(dom_grp),
            }
        )
    pd.DataFrame(comm_rows).sort_values("members_count", ascending=False).to_parquet(
        OUTPUTS_DIR / "p15_louvain_communities.parquet", index=False
    )

    rho_vb, p_vb = spearmanr(df_nodes.loc[active_mask, "total_messages"], df_nodes.loc[active_mask, "betweenness"])
    rho_vp, p_vp = spearmanr(df_nodes.loc[active_mask, "total_messages"], df_nodes.loc[active_mask, "pagerank"])

    # =========================================================================
    # P17: Openness vs Cliquishness (Clustering, Assortativity, Tenure Mixing)
    # =========================================================================
    avg_clustering = round(float(nx.average_clustering(G_active)), 4)
    try:
        deg_assort = round(float(nx.degree_assortativity_coefficient(G_active)), 4)
    except Exception:
        deg_assort = 0.0

    # Assign tenure cohort: Fundadores (first 14d), Intermedios (d15-d40), Nuevos (d41+)
    min_dt = pd.to_datetime(df_nodes["first_dt"]).min()
    df_nodes["join_day_offset"] = (pd.to_datetime(df_nodes["first_dt"]) - min_dt).dt.days
    df_nodes["tenure_tier"] = pd.cut(
        df_nodes["join_day_offset"],
        bins=[-1, 14, 40, 10000],
        labels=["1. Fundadores (D0–14)", "2. Intermedios (D15–40)", "3. Nuevos (D41+)"],
    ).astype(str)
    tier_map = dict(zip(df_nodes["member_id"], df_nodes["tenure_tier"]))

    df_mix = df_edges.copy()
    df_mix["source_tier"] = df_mix["source"].map(tier_map)
    df_mix["target_tier"] = df_mix["target"].map(tier_map)
    df_mix_agg = (
        df_mix.dropna(subset=["source_tier", "target_tier"])
        .groupby(["source_tier", "target_tier"], as_index=False)
        .agg(interactions=("interaction_count", "sum"), edge_pairs=("source", "count"))
    )
    src_tot = df_mix_agg.groupby("source_tier")["interactions"].transform("sum")
    df_mix_agg["share_of_source_pct"] = (df_mix_agg["interactions"] / src_tot * 100).round(2)
    df_mix_agg.to_parquet(OUTPUTS_DIR / "p17_assortativity_mixing.parquet", index=False)

    # =========================================================================
    # P18: Percolation / Resilience Simulation (Bus Factor)
    # =========================================================================
    # Use core active graph (edges with >= 2 interactions so structural backbone is meaningful)
    G_perc = nx.Graph()
    for u, v, d in G_und.edges(data=True):
        if d["count"] >= 2:
            G_perc.add_edge(u, v, weight=d["weight"], count=d["count"])
    if G_perc.number_of_nodes() < 30:
        G_perc = G_active.copy()

    base_n = G_perc.number_of_nodes()
    base_e = G_perc.number_of_edges()
    base_lcc = len(max(nx.connected_components(G_perc), key=len))

    # Pre-rank nodes by volume and by betweenness
    vol_map = dict(zip(df_nodes["member_id"], df_nodes["total_messages"]))
    bet_map = dict(zip(df_nodes["member_id"], df_nodes["betweenness"]))
    nodes_by_vol = sorted(G_perc.nodes(), key=lambda n: vol_map.get(n, 0), reverse=True)
    nodes_by_bet = sorted(G_perc.nodes(), key=lambda n: bet_map.get(n, 0.0), reverse=True)
    all_perc_nodes = list(G_perc.nodes())

    removal_pcts = [0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25]
    perc_rows = []
    rng = random.Random(42)

    for pct in removal_pcts:
        k_rem = int(round(base_n * pct / 100.0))

        # Deterministic strategies
        for strat_name, ordered_nodes in [("Top Betweenness (Puentes)", nodes_by_bet), ("Top Volumen (Habladores)", nodes_by_vol)]:
            rem_set = set(ordered_nodes[:k_rem])
            H = G_perc.subgraph([n for n in all_perc_nodes if n not in rem_set])
            lcc_sz = len(max(nx.connected_components(H), key=len)) if H.number_of_nodes() > 0 else 0
            n_comp = nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0
            perc_rows.append(
                {
                    "removed_pct": pct,
                    "removed_nodes": k_rem,
                    "strategy": strat_name,
                    "lcc_pct_of_original": round(lcc_sz / base_lcc * 100, 2),
                    "edges_remaining_pct": round(H.number_of_edges() / max(base_e, 1) * 100, 2),
                    "num_components": int(n_comp),
                }
            )

        # Random baseline averaged over 15 Monte Carlo trials
        lccs, edge_pcts, comps = [], [], []
        for _ in range(15):
            rem_set = set(rng.sample(all_perc_nodes, k_rem)) if k_rem > 0 else set()
            H = G_perc.subgraph([n for n in all_perc_nodes if n not in rem_set])
            lccs.append((len(max(nx.connected_components(H), key=len)) if H.number_of_nodes() > 0 else 0) / base_lcc * 100)
            edge_pcts.append(H.number_of_edges() / max(base_e, 1) * 100)
            comps.append(nx.number_connected_components(H) if H.number_of_nodes() > 0 else 0)
        perc_rows.append(
            {
                "removed_pct": pct,
                "removed_nodes": k_rem,
                "strategy": "Remoción Aleatoria (Promedio 15x)",
                "lcc_pct_of_original": round(float(np.mean(lccs)), 2),
                "edges_remaining_pct": round(float(np.mean(edge_pcts)), 2),
                "num_components": round(float(np.mean(comps)), 1),
            }
        )

    df_perc = pd.DataFrame(perc_rows)
    df_perc.to_parquet(OUTPUTS_DIR / "p18_percolation_simulation.parquet", index=False)

    # =========================================================================
    # P19: Reciprocity & Dyadic Symmetry
    # =========================================================================
    global_reciprocity = round(float(nx.overall_reciprocity(G_dir)), 4)
    dyad_rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for u, v, d in G_dir.edges(data=True):
        pair = (min(u, v), max(u, v))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        w_uv = int(G_dir[u][v]["count"]) if G_dir.has_edge(u, v) else 0
        w_vu = int(G_dir[v][u]["count"]) if G_dir.has_edge(v, u) else 0
        tot = w_uv + w_vu
        sym = round(min(w_uv, w_vu) / max(w_uv, w_vu), 4) if max(w_uv, w_vu) > 0 else 0.0
        dyad_rows.append(
            {
                "member_a": pair[0],
                "member_b": pair[1],
                "a_to_b": w_uv,
                "b_to_a": w_vu,
                "total_interactions": tot,
                "is_mutual": int(w_uv > 0 and w_vu > 0),
                "symmetry_score": sym,
            }
        )
    df_dyads = pd.DataFrame(dyad_rows).sort_values("total_interactions", ascending=False).reset_index(drop=True)
    df_dyads.to_parquet(OUTPUTS_DIR / "p19_reciprocity_dyads.parquet", index=False)

    con.close()

    silent_bridges_n = int((df_nodes["sna_quadrant"] == "Puente Silencioso (Bajo Vol + Alto Puente)").sum())
    lcc_drop_bet10 = float(
        df_perc[(df_perc["removed_pct"] == 10) & (df_perc["strategy"].str.contains("Betweenness"))]["lcc_pct_of_original"].iloc[0]
    )
    lcc_drop_rand10 = float(
        df_perc[(df_perc["removed_pct"] == 10) & (df_perc["strategy"].str.contains("Aleatoria"))]["lcc_pct_of_original"].iloc[0]
    )

    summary = {
        "p15_active_nodes": int(G_active.number_of_nodes()),
        "p15_directed_edges": int(G_dir.number_of_edges()),
        "p15_louvain_clusters": int(len(comm_rows)),
        "p15_modularity_q": modularity_q,
        "p16_spearman_vol_vs_betweenness": round(float(rho_vb), 4),
        "p16_p_vol_vs_betweenness": float(f"{p_vb:.6g}"),
        "p16_spearman_vol_vs_pagerank": round(float(rho_vp), 4),
        "p16_silent_bridges_count": silent_bridges_n,
        "p17_avg_clustering": avg_clustering,
        "p17_degree_assortativity": deg_assort,
        "p18_lcc_after_10pct_betweenness_removal": lcc_drop_bet10,
        "p18_lcc_after_10pct_random_removal": lcc_drop_rand10,
        "p19_global_reciprocity": global_reciprocity,
        "p19_mutual_dyads_pct": round(float(df_dyads["is_mutual"].mean() * 100), 2) if len(df_dyads) else 0.0,
    }
    (OUTPUTS_DIR / "p15_p19_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[P15-P19] Done! Louvain Q={modularity_q} ({len(comm_rows)} clusters), SilentBridges={silent_bridges_n}, Reciprocity={global_reciprocity}")
    return summary


if __name__ == "__main__":
    run_social_graph_sna()
