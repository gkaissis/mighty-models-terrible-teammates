"""Run Phase 1 v2.1 Meta-Analysis with disclosure integration enabled by default ($0 API spend).

Usage:
    uv run python scripts/run_phase1_meta_analysis.py
    uv run python scripts/run_phase1_meta_analysis.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from common_ground.config import DATA_DIR
from common_ground.meta_analysis import (
    render_meta_v2_1_markdown,
    run_all_phase1_meta_analyses,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic Phase 1 v2.1 meta-analysis directly from raw data/v2 files."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DATA_DIR / "v2",
        help="Path to raw v2 run directory (default: data/v2).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DATA_DIR / "v2" / "analysis" / "meta_v2_1.json",
        help="Output JSON path (default: data/v2/analysis/meta_v2_1.json).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DATA_DIR / "v2" / "analysis" / "meta_v2_1.md",
        help="Output Markdown summary path (default: data/v2/analysis/meta_v2_1.md).",
    )
    parser.add_argument(
        "--skip-disclosure",
        action="store_true",
        help="Skip disclosure meta-analysis (default: enabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all analyses and validate JSON/Markdown serialization without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("error", category=SingularMatrixWarning)
    args = parse_args(argv)
    report = run_all_phase1_meta_analyses(
        args.run_dir,
        include_disclosure=not args.skip_disclosure,
    )
    # Parity guard: on the full v2 corpus, multi-ACTION glitch telemetry must be present
    # (via raw ledger.jsonl or exported turns.parquet `n_action_blocks`) before overwriting.
    if report["data_inventory"]["n_episodes_main"] >= 600:
        neon_multi = report["triage_paralysis_and_upsets"]["neon_s303_s304_anomaly"][
            "neon_multi_action_total_turns"
        ]
        if neon_multi <= 0:
            raise RuntimeError(
                "Refusing to overwrite meta_v2_1.json: neon_multi_action_total_turns == 0 "
                "on full v2 corpus (missing response_text / n_action_blocks telemetry)."
            )
    json_blob = json.dumps(report, indent=2, sort_keys=False)
    md_blob = render_meta_v2_1_markdown(report)

    if args.dry_run:
        print(
            f"[dry-run] meta_v2_1 ok: {len(json_blob):,} bytes JSON, {len(md_blob):,} bytes MD; "
            f"main_eps={report['data_inventory']['n_episodes_main']}, "
            f"non_aa_choices={report['choice_logits']['scope']['all_non_aa_n']}, "
            f"upsets={report['triage_paralysis_and_upsets']['score_upsets']['n_total_upsets']}, "
            f"disclosure_wired={report['disclosure_meta_analysis'].get('wired')}"
        )
        return 0

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json_blob + "\n", encoding="utf-8")
    args.out_md.write_text(md_blob + "\n", encoding="utf-8")
    print(
        f"Wrote {args.out_json} ({len(json_blob):,} bytes) and "
        f"{args.out_md} ({len(md_blob):,} bytes)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
