"""Verifies that all analytical outputs covering P01–P36 were produced cleanly."""

from __future__ import annotations

import json
import pandas as pd

from pipeline.config import OUTPUTS_DIR

REQUIRED_PARQUETS = [
    "p01_daily_metrics.parquet",
    "p02_lorenz_global.parquet",
    "p02_gini_by_group.parquet",
    "p03_group_health.parquet",
    "p04_moderation_daily.parquet",
    "p05_activation_funnel.parquet",
    "p06_heatmap_7x24.parquet",
    "p07_group_circadian_profiles.parquet",
    "p07_circadian_summary.parquet",
    "p08_frt_by_hour.parquet",
    "p08_frt_by_group.parquet",
    "p09_bursts_daily.parquet",
    "p09_acf.parquet",
    "p09_burst_hangover.parquet",
    "p10_kaplan_meier_global.parquet",
    "p11_weekly_cohorts.parquet",
    "p11_milestone_retention.parquet",
    "p12_onboarding_km.parquet",
    "p13_first_impression.parquet",
    "p14_purge_impact_daily.parquet",
    "p15_sna_nodes.parquet",
    "p15_sna_edges.parquet",
    "p15_louvain_communities.parquet",
    "p17_assortativity_mixing.parquet",
    "p18_percolation_simulation.parquet",
    "p19_reciprocity_dyads.parquet",
    "p20_share_of_voice_daily.parquet",
    "p21_markov_transitions.parquet",
    "p22_jaccard_matrix.parquet",
    "p23_geography_vs_thematic.parquet",
    "p24_multigroup_sweetspot.parquet",
    "p25_plans_enriched.parquet",
    "p25_plans_by_category.parquet",
    "p26_plans_heatmap_day_hour.parquet",
    "p27_initiator_social_capital.parquet",
    "p29_feature_importance.parquet",
    "p30_sentiment_by_group.parquet",
    "p30_sentiment_weekly.parquet",
    "p31_conversational_style.parquet",
    "p32_topics_weekly.parquet",
    "p33_media_mix_by_group.parquet",
    "p34_sticker_virality.parquet",
    "p35_media_technical_profile.parquet",
    "p35_voice_response_impact.parquet",
    "p36_member_archetypes.parquet",
    "p36_archetype_profiles.parquet",
]

REQUIRED_JSONS = [
    "p01_p09_summary.json",
    "p10_p14_summary.json",
    "p12_onboarding_stats.json",
    "p14_purge_stats.json",
    "p15_p19_summary.json",
    "p20_cannibalization_events.json",
    "p20_p29_summary.json",
    "p28_pin_impact.json",
    "p29_plan_ml_metrics.json",
    "p30_p36_summary.json",
    "p34_sticker_summary.json",
    "p36_clustering_metrics.json",
    "executive_summary.json",
]


def test_all_analytical_outputs_exist_and_nonempty() -> None:
    for fname in REQUIRED_PARQUETS:
        p = OUTPUTS_DIR / fname
        assert p.exists(), f"Missing output parquet: {fname}"
        df = pd.read_parquet(p)
        assert len(df) > 0, f"Output parquet is empty: {fname}"

    for fname in REQUIRED_JSONS:
        p = OUTPUTS_DIR / fname
        assert p.exists(), f"Missing output json: {fname}"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and len(data) > 0

    assert (OUTPUTS_DIR / "p29_plan_model.joblib").exists(), "Missing serialized P29 ML model!"
