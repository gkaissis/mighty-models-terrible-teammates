"""CLI runner for Experiment 1 ("The Partner Drops Out") — Gate & Full 320x8 Handoff Matrix.

Usage examples:
  # Offline zero-cost mock calibration gate
  uv run python scripts/run_exp1_dropout.py --mock --gate-only --out-dir data/exp1_mock

  # Offline zero-cost mock full run (small slice)
  uv run python scripts/run_exp1_dropout.py --mock --full --limit-cases 16 --out-dir data/exp1_mock

  # Live calibration gate (defaults to $15.00 cap; writes locked_receiver_roster.json)
  uv run python scripts/run_exp1_dropout.py --gate-only --budget-cap 15.0

  # Live full 320 x 8 handoff run (reuses locked_receiver_roster.json)
  uv run python scripts/run_exp1_dropout.py --full --budget-cap 80.0
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from common_ground.dropout import (
    DEFAULT_EPISODES_DIR,
    DEFAULT_EXP1_OUT_DIR,
    DEFAULT_FULL_BUDGET_CAP_USD,
    DEFAULT_GATE_BUDGET_CAP_USD,
    DEFAULT_GATE_CEILING_THRESHOLD,
    DEFAULT_MAX_STEP_UPS,
    DEFAULT_PROBE_MANIFEST_V2_1_PATH,
    run_dropout_experiment,
    run_dropout_gate,
)

console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Experiment 1 ("The Partner Drops Out") — Gate & Full Handoff Harness.'
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gate-only",
        action="store_true",
        help="Run ONLY the 16-case Floor-vs.-Ceiling calibration gate + worker auto step-up.",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the full 320-case x 8-receiver handoff experiment. "
            "Automatically runs the gate first if locked_receiver_roster.json is missing."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline with deterministic zero-cost mock client ($0 API spend).",
    )
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=None,
        help=(
            f"Hard USD spend cap on the shared ledger (defaults to ${DEFAULT_GATE_BUDGET_CAP_USD:.2f} "
            f"for --gate-only, ${DEFAULT_FULL_BUDGET_CAP_USD:.2f} for --full)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_EXP1_OUT_DIR,
        help=f"Output directory for ledger, gate reports, roster, results, and summary (default: {DEFAULT_EXP1_OUT_DIR}).",
    )
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=DEFAULT_PROBE_MANIFEST_V2_1_PATH,
        help=f"Path to v2.1 receiver probe manifest (default: {DEFAULT_PROBE_MANIFEST_V2_1_PATH}).",
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=DEFAULT_EPISODES_DIR,
        help=f"Directory containing frozen v2 Goalpost episode JSONs (default: {DEFAULT_EPISODES_DIR}).",
    )
    parser.add_argument(
        "--ceiling-threshold",
        type=float,
        default=DEFAULT_GATE_CEILING_THRESHOLD,
        help=f"Minimum worker ceiling strict pass rate before stepping up (default: {DEFAULT_GATE_CEILING_THRESHOLD:.2f}).",
    )
    parser.add_argument(
        "--max-step-ups",
        type=int,
        default=DEFAULT_MAX_STEP_UPS,
        help=f"Maximum worker reasoning-tier step-ups during calibration gate (default: {DEFAULT_MAX_STEP_UPS}).",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Optional cap on number of handoff cases for --full (useful for smoke/mock tests).",
    )
    parser.add_argument(
        "--force-gate",
        action="store_true",
        help="Re-run the calibration gate before --full even if locked_receiver_roster.json exists.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Optional per-model concurrency ceiling for --full (useful for gentle 429-free resumes).",
    )
    return parser.parse_args(argv)


def _print_gate_table(report: dict[str, Any]) -> None:
    mode_tag = "[MOCK]" if report.get("mock") else "[LIVE]"
    table = Table(title=f"Exp 1 Floor-vs.-Ceiling Calibration Gate {mode_tag}")
    table.add_column("Receiver")
    table.add_column("Tier")
    table.add_column("Effort")
    table.add_column("Step-Ups", justify="right")
    table.add_column("Floor Strict", justify="right")
    table.add_column("Ceil Strict", justify="right")
    table.add_column("Ceil Own+R3", justify="right")
    table.add_column("Status")
    table.add_column("Cost ($)", justify="right")

    for key, rec in report["receivers"].items():
        lk = rec["locked"]
        eff = (
            f"{lk['initial_effort']} -> {lk['locked_effort']}"
            if lk["step_ups_performed"] > 0
            else lk["locked_effort"]
        )
        status = (
            "LOCKED_FRONTIER"
            if lk["tier"] == "frontier"
            else ("PASS" if lk["gate_passed"] else "BELOW_THRESH")
        )
        table.add_row(
            key,
            lk["tier"],
            eff,
            str(lk["step_ups_performed"]),
            f"{lk['gate_floor_strict_pass_rate'] * 100:.1f}%",
            f"{lk['gate_ceiling_strict_pass_rate'] * 100:.1f}%",
            f"{lk['gate_ceiling_own_plus_r3_pass_rate'] * 100:.1f}%",
            status,
            f"{rec['total_cost_usd']:.4f}",
        )
    console.print(table)
    console.print(
        f"[bold green]Gate cost: ${report['total_gate_cost_usd']:.4f} "
        f"(ledger cumulative: ${report['ledger_total_cost_usd']:.4f})[/bold green]"
    )


def _print_summary_table(summary: dict[str, Any]) -> None:
    mode_tag = "[MOCK]" if summary.get("mock") else "[LIVE]"
    table = Table(title=f"Exp 1 Handoff Survival by Sender (Primary Role A, Cross-Vendor) {mode_tag}")
    table.add_column("Sender")
    table.add_column("N", justify="right")
    table.add_column("Strict 15/15", justify="right")
    table.add_column("Own+R3 8/8", justify="right")
    table.add_column("Partner 7/7", justify="right")
    table.add_column("Partner | Own", justify="right")
    table.add_column("Mean R1-2 Disc", justify="right")

    for sender, stats_map in summary["by_sender"].items():
        m = stats_map["primary_role_a_cross_vendor"]
        p_given = (
            f"{m['partner_pass_given_own_pass'] * 100:.1f}%"
            if m["partner_pass_given_own_pass"] is not None
            else "—"
        )
        mean_disc = (
            f"{m['mean_sender_r1_r2_n_formula']:.2f}/7"
            if m["mean_sender_r1_r2_n_formula"] is not None
            else "—"
        )
        table.add_row(
            sender,
            str(m["n_cases"]),
            f"{m['strict_all_pass_rate'] * 100:.1f}%",
            f"{m['own_plus_r3_pass_rate'] * 100:.1f}%",
            f"{m['partner_rules_pass_rate'] * 100:.1f}%",
            p_given,
            mean_disc,
        )
    console.print(table)
    ov = summary["overview"]
    console.print(
        f"[bold green]Exp 1 Complete: {ov['n_ok']}/{ov['n_cases']} ok "
        f"(strict={ov['strict_all_pass_rate'] * 100:.1f}%, "
        f"handoff_cost=${ov['cost_usd']:.4f}, "
        f"ledger_total=${summary['ledger_total_cost_usd']:.4f})[/bold green]"
    )


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    run_full = bool(args.full)
    run_gate_only = bool(args.gate_only) or not run_full

    budget_cap = (
        float(args.budget_cap)
        if args.budget_cap is not None
        else (DEFAULT_FULL_BUDGET_CAP_USD if run_full else DEFAULT_GATE_BUDGET_CAP_USD)
    )
    out_dir: Path = args.out_dir
    roster_path = out_dir / "locked_receiver_roster.json"

    out: dict[str, Any] = {}
    if run_gate_only or args.force_gate or not roster_path.exists():
        console.print(
            f"[bold cyan]Running Exp 1 Calibration Gate "
            f"(mock={args.mock}, budget_cap=${budget_cap:.2f}, out_dir={out_dir})...[/bold cyan]"
        )
        gate_report = await run_dropout_gate(
            episodes_dir=args.episodes_dir,
            out_dir=out_dir,
            probe_manifest_path=args.probe_manifest,
            budget_cap_usd=budget_cap,
            ceiling_threshold=args.ceiling_threshold,
            max_step_ups=args.max_step_ups,
            mock=args.mock,
        )
        _print_gate_table(gate_report)
        out["gate_report"] = gate_report

    if run_full:
        console.print(
            f"[bold cyan]Running Exp 1 Full Handoff Matrix "
            f"(mock={args.mock}, budget_cap=${budget_cap:.2f}, limit_cases={args.limit_cases})...[/bold cyan]"
        )
        summary = await run_dropout_experiment(
            episodes_dir=args.episodes_dir,
            out_dir=out_dir,
            roster_path=roster_path,
            limit_cases=args.limit_cases,
            budget_cap_usd=budget_cap,
            max_concurrency_per_model=args.max_concurrency,
            mock=args.mock,
        )
        _print_summary_table(summary)
        out["summary"] = summary

    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(async_main(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
