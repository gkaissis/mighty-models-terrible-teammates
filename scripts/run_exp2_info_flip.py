"""CLI runner for Experiment 2 ("The Information Flip") — Causal Mirror of Capstone Tone-Flip.

Usage examples:
  # Zero-cost dry-run: builds & audits all 30 pairs (24 main + 6 placebo), prints pair & parity table
  uv run python scripts/run_exp2_info_flip.py --dry-run

  # Zero-cost offline mock end-to-end run (384 main + 96 placebo = 480 choices)
  uv run python scripts/run_exp2_info_flip.py --mock --out-dir data/exp2_mock

  # Live execution (default $30.00 hard cap, concurrency=16 per reviewer, crash-resumable)
  uv run python scripts/run_exp2_info_flip.py --budget-cap 30.0 --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from common_ground.info_flip import (
    DEFAULT_CAPSTONE_DIR,
    DEFAULT_EPISODES_DIR,
    DEFAULT_EXP2_BUDGET_CAP_USD,
    DEFAULT_EXP2_CONCURRENCY,
    DEFAULT_EXP2_OUT_DIR,
    DEFAULT_SELECTION_SEED,
    EXP2_REVIEWERS,
    run_exp2_info_flip,
)

console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Experiment 2 ("The Information Flip") — Symmetric Within-Pair Crossover CLI.'
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline with deterministic zero-cost mock client ($0 API spend).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and audit all 30 pairs (word-count parity, zero leakage, vendor leaks) at $0 spend without making calls.",
    )
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=DEFAULT_EXP2_BUDGET_CAP_USD,
        help=f"Hard USD spend cap on ledger.jsonl (default: ${DEFAULT_EXP2_BUDGET_CAP_USD:.2f}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_EXP2_OUT_DIR,
        help=f"Output directory for pairs.json, ledger.jsonl, choices.jsonl, and summaries (default: {DEFAULT_EXP2_OUT_DIR}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_EXP2_CONCURRENCY,
        help=f"Per-reviewer concurrency limit (default: {DEFAULT_EXP2_CONCURRENCY}, recommended 16-20).",
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=DEFAULT_EPISODES_DIR,
        help=f"Directory containing frozen v2 episode JSONs (default: {DEFAULT_EPISODES_DIR}).",
    )
    parser.add_argument(
        "--capstone-dir",
        type=Path,
        default=DEFAULT_CAPSTONE_DIR,
        help=f"Directory containing Capstone Tone-Flip JSONL/JSON for side-by-side comparison (default: {DEFAULT_CAPSTONE_DIR}).",
    )
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=None,
        help="Optional probe manifest JSON override (defaults to data/v2/probe_manifest_v2.json).",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=DEFAULT_SELECTION_SEED,
        help=f"Deterministic seed for pair selection (default: {DEFAULT_SELECTION_SEED}).",
    )
    return parser.parse_args(argv)


def _print_dry_run_report(report: dict[str, Any]) -> None:
    pairs = report["pairs"]
    audit = report["audit"]

    table = Table(title="Exp 2 Information Flip — 30-Pair Manifest & ±5% Parity Audit (--dry-run)")
    table.add_column("Pair ID")
    table.add_column("Arm")
    table.add_column("Task/Role")
    table.add_column("Seed")
    table.add_column("P1 vs P2")
    table.add_column("Base Disc", justify="center")
    table.add_column("P1 Priv/Pub (Δ%)", justify="right")
    table.add_column("P2 Priv/Pub (Δ%)", justify="right")
    table.add_column("ZeroLeak/Full", justify="center")

    for p in pairs:
        a1, a2 = p["arms_1"], p["arms_2"]
        seed_str = str(p["seed_1"]) if p["same_seed"] else f"{p['seed_1']}/{p['seed_2']}"
        ok_flag = (
            "OK"
            if (
                a1["public_echo_zero_leak"]
                and a2["public_echo_zero_leak"]
                and a1["private_info_complete"]
                and a2["private_info_complete"]
                and a1["word_parity_ok"]
                and a2["word_parity_ok"]
            )
            else "FAIL"
        )
        table.add_row(
            p["pair_id"],
            p["arm"],
            f"{p['task']}/{p['target_role']}",
            seed_str,
            f"{p['partner_1']} vs {p['partner_2']}",
            f"{a1['baseline_disclosure_count']} vs {a2['baseline_disclosure_count']}",
            f"{a1['added_private_words']}/{a1['added_public_words']} ({a1['added_word_rel_diff'] * 100:.1f}%)",
            f"{a2['added_private_words']}/{a2['added_public_words']} ({a2['added_word_rel_diff'] * 100:.1f}%)",
            ok_flag,
        )
    console.print(table)

    console.print(
        f"[bold green]Dry-Run Integrity Audit PASSED ($0.00 spent):[/bold green]\n"
        f"  Pairs: {audit['n_pairs_main']} main ({audit['n_pairs_triage_main']} triage + "
        f"{audit['n_pairs_goalpost_main']} goalpost) + {audit['n_pairs_placebo']} placebo "
        f"({audit['n_pairs_triage_placebo']} triage + {audit['n_pairs_goalpost_placebo']} goalpost) "
        f"= {audit['n_pairs_total']} pairs ({report['n_planned_choices']} planned choices)\n"
        f"  Unique Episodes: {audit['n_unique_episodes']}/{audit['n_candidate_episodes']} "
        f"(disjoint={audit['strict_disjoint_ok']}, all_score_1={audit['all_score_1_ok']}, "
        f"all_same_seed={audit['all_same_seed_ok']}, all_distinct_partners={audit['all_distinct_partners_ok']})\n"
        f"  Triage Zero-Contradiction: {audit['triage_zero_contradiction_count']}/{audit['triage_zero_contradiction_total']}\n"
        f"  Word Parity (±5%): {audit['word_parity_pass_count']}/{audit['word_parity_total']} "
        f"(max added Δ={audit['max_added_word_rel_diff'] * 100:.2f}%, "
        f"mean added Δ={audit['mean_added_word_rel_diff'] * 100:.2f}%, "
        f"max total R1 Δ={audit['max_total_word_rel_diff'] * 100:.2f}%)\n"
        f"  Public-Echo Zero Leakage: {audit['public_echo_zero_leak_count']}/{audit['public_echo_zero_leak_total']} | "
        f"Private-Info Complete: {audit['private_info_complete_count']}/{audit['private_info_complete_total']}\n"
        f"  Prompt Vendor Leaks: {audit['prompt_vendor_leak_count']}/{audit['n_prompts_checked']} prompts scanned\n"
        f"  Main Seeds: {audit['by_task_and_seed']} | Main Roles: {audit['by_task_and_role_main']}\n"
        f"  Main Partners by Task: {audit['by_task_and_partner_main']}"
    )


def _print_run_summary(summary: dict[str, Any], out_dir: Path) -> None:
    mode_tag = "[MOCK]" if summary.get("mock") else "[LIVE]"
    table = Table(title=f"Exp 2 Information Flip vs. Capstone Tone-Flip {mode_tag}")
    table.add_column("Experiment / Slice")
    table.add_column("Scope")
    table.add_column("Cases (Pairs)", justify="right")
    table.add_column("Mean Δ", justify="right")
    table.add_column("Pair-Boot 95% CI", justify="center")
    table.add_column("Sign-Flip p", justify="right")
    table.add_column("Unanimous Toward", justify="right")
    table.add_column("Away", justify="right")

    def _add(label: str, scope: str, m: dict[str, Any], dk: str, twk: str, awk: str) -> None:
        if not m or m.get("n_cases", 0) == 0:
            return
        ci = m.get(f"{dk}_pair_boot_ci95") or [0.0, 0.0]
        table.add_row(
            label,
            scope,
            f"{m['n_cases']} ({m['n_pairs']})",
            f"{m[f'mean_{dk}']:+.4f}",
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
            f"{m['pair_sign_flip_perm_p']:.4f}",
            f"{m[f'n_{twk}']}/{m['n_cases']} ({m[f'{twk}_rate'] * 100:.1f}%)",
            f"{m[f'n_{awk}']}/{m['n_cases']} ({m[f'{awk}_rate'] * 100:.1f}%)",
        )

    cap = summary.get("capstone_comparison")
    if cap:
        _add(
            "Capstone Tone+Readability",
            "Cross-Vendor",
            cap["itt_cross_vendor"],
            "delta_warm",
            "switched_toward_warm",
            "switched_away_from_warm",
        )
        _add(
            "Capstone Tone+Readability",
            "Pooled",
            cap["itt_all_reviewers"],
            "delta_warm",
            "switched_toward_warm",
            "switched_away_from_warm",
        )

    m_cv = summary["main_arm"]["itt_cross_vendor"]
    m_all = summary["main_arm"]["itt_all_reviewers"]
    p_cv = summary["placebo_arm"]["itt_cross_vendor"]
    p_all = summary["placebo_arm"]["itt_all_reviewers"]

    _add("Exp 2 Info Flip (Main)", "Cross-Vendor", m_cv["overall"], "delta_info", "switched_toward_info", "switched_away_from_info")
    _add("Exp 2 Info Flip (Main)", "Pooled", m_all["overall"], "delta_info", "switched_toward_info", "switched_away_from_info")
    if "triage" in m_all["by_task"]:
        _add("  Triage Full-Transcript", "Pooled", m_all["by_task"]["triage"], "delta_info", "switched_toward_info", "switched_away_from_info")
    if "goalpost" in m_all["by_task"]:
        _add("  Goalpost R1-Handoff", "Pooled", m_all["by_task"]["goalpost"], "delta_info", "switched_toward_info", "switched_away_from_info")
    _add("Exp 2 Public-Echo Placebo", "Cross-Vendor", p_cv["overall"], "delta_echo", "switched_toward_echo", "switched_away_from_echo")
    _add("Exp 2 Public-Echo Placebo", "Pooled", p_all["overall"], "delta_echo", "switched_toward_echo", "switched_away_from_echo")

    console.print(table)

    rv_table = Table(title=f"Exp 2 Main Arm — Per-Reviewer Breakdown {mode_tag}")
    rv_table.add_column("Reviewer")
    rv_table.add_column("Cases", justify="right")
    rv_table.add_column("Mean ΔInfo", justify="right")
    rv_table.add_column("Unanimous Toward", justify="right")
    rv_table.add_column("Held", justify="right")
    rv_table.add_column("Split", justify="right")
    for rv in EXP2_REVIEWERS:
        rm = m_all["by_reviewer"].get(rv)
        if rm:
            rv_table.add_row(
                rv,
                str(rm["n_cases"]),
                f"{rm['mean_delta_info']:+.4f}",
                f"{rm['n_switched_toward_info']}/{rm['n_cases']} ({rm['switched_toward_info_rate'] * 100:.1f}%)",
                f"{rm['held_consistent_rate'] * 100:.1f}%",
                f"{rm['order_split_rate'] * 100:.1f}%",
            )
    console.print(rv_table)

    cnt = summary["counts"]
    console.print(
        f"[bold green]Exp 2 Complete:[/bold green] {cnt['n_choices_main_parsed']}/{cnt['n_choices_main']} main + "
        f"{cnt['n_choices_placebo_parsed']}/{cnt['n_choices_placebo']} placebo parsed "
        f"(excluded={summary.get('n_choices_excluded', 0)}, budget_stopped={summary.get('budget_stopped', False)}, "
        f"ledger_cost=${summary.get('ledger_total_cost_usd', 0.0):.4f}) -> {out_dir / 'exp2_summary.json'}"
    )


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    result = await run_exp2_info_flip(
        episodes_dir=args.episodes_dir,
        out_dir=args.out_dir,
        capstone_dir=args.capstone_dir,
        probe_manifest=args.probe_manifest,
        budget_cap_usd=args.budget_cap,
        concurrency=args.concurrency,
        mock=args.mock,
        dry_run=args.dry_run,
        selection_seed=args.selection_seed,
    )
    if args.dry_run:
        _print_dry_run_report(result)
    else:
        _print_run_summary(result, args.out_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(async_main(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
