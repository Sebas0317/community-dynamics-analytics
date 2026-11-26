"""End-to-End Pipeline Orchestrator (Steps 00 -> 06 + Executive Summary + Privacy Guard Tests).

Usage:
  Baseline run (current analytics.db + exportaciones_raw):
      uv run python -m pipeline.run_all --run-tests

  Friday fresh drop run (merges new WhatsApp exports + full media):
      uv run python -m pipeline.run_all --drop-dir "C:/path/to/friday_drop" --run-tests
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import DEFAULT_BASELINE_DB, DEFAULT_RAW_DIR, OUTPUTS_DIR, PARQUET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Community Dynamics Analytics pipeline (P01-P36).")
    parser.add_argument("--baseline-db", type=Path, default=DEFAULT_BASELINE_DB)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--drop-dir", type=Path, default=None, help="Optional fresh WhatsApp export directory (Friday drop)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip steps 00 and 01 and recompute analytics only")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest Zero-PII privacy & quality suite after pipeline")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("=" * 72)
    print("  COMMUNITY DYNAMICS ANALYTICS — END-TO-END PIPELINE (P01–P36)")
    print("=" * 72)

    if not args.skip_ingest:
        m00 = importlib.import_module("pipeline.00_ingest_whatsapp_drop")
        m00.run_ingestion(baseline_db=args.baseline_db, raw_dir=args.raw_dir, drop_dir=args.drop_dir)

        m01 = importlib.import_module("pipeline.01_anonymize_lakehouse")
        m01.run_anonymization()

    m02 = importlib.import_module("pipeline.02_macro_and_chronobiology")
    s1 = m02.run_macro_and_chronobiology()

    m03 = importlib.import_module("pipeline.03_survival_and_cohorts")
    s2 = m03.run_survival_and_cohorts()

    m04 = importlib.import_module("pipeline.04_social_graph_sna")
    s3 = m04.run_social_graph_sna()

    m05 = importlib.import_module("pipeline.05_multigroup_and_plans_ml")
    s4 = m05.run_multigroup_and_plans_ml()

    m06 = importlib.import_module("pipeline.06_nlp_media_and_archetypes")
    s5 = m06.run_nlp_media_and_archetypes()

    # Combine unified executive summary
    manifest = json.loads((PARQUET_DIR.parent / "privacy_audit_manifest.json").read_text(encoding="utf-8"))
    exec_summary = {
        "pipeline_completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "date_range": manifest["date_range"],
        "volumes": {k: v["rows"] for k, v in manifest["tables"].items()},
        "p01_p09": s1,
        "p10_p14": s2,
        "p15_p19": s3,
        "p20_p29": s4,
        "p30_p36": s5,
    }
    (OUTPUTS_DIR / "executive_summary.json").write_text(json.dumps(exec_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 72)
    print(f"[PIPELINE] All 36 analytical questions computed in {exec_summary['elapsed_seconds']}s!")
    print(f"[PIPELINE] Executive summary -> {OUTPUTS_DIR / 'executive_summary.json'}")

    if args.run_tests:
        print("-" * 72)
        print("[TESTS] Running Zero-PII Privacy Guard & Data Quality Suite...")
        res = subprocess.run([sys.executable, "-m", "pytest", "-v"], check=False)
        if res.returncode != 0:
            print("[ERROR] Privacy or quality tests FAILED!")
            sys.exit(res.returncode)
        print("[TESTS] 100% PASSED — Lakehouse & Outputs are strictly Zero-PII!")


if __name__ == "__main__":
    main()
