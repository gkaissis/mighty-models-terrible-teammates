"""Comprehensive unit and offline end-to-end tests for Experiment 2 ("The Information Flip")."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from common_ground.config import DATA_DIR, PROBE_MANIFEST_V2_PATH
from common_ground.disclosure import (
    check_goalpost_message_disclosure,
    check_triage_operator_disclosure,
    is_triage_b_zero_contradiction,
)
from common_ground.info_flip import (
    DEFAULT_CAPSTONE_DIR,
    DEFAULT_EPISODES_DIR,
    DEFAULT_EXP2_OUT_DIR,
    DEFAULT_PROBE_MANIFEST_V2_1_PATH,
    DEFAULT_WORD_PARITY_TOL,
    EXP2_REVIEWERS,
    audit_exp2_pairs,
    build_choice_specs,
    format_exp2_transcript,
    load_capstone_comparison,
    load_exp2_reviewers,
    load_main_episodes,
    pair_fixed_handles,
    parse_choice,
    render_exp2_summary_markdown,
    run_exp2_info_flip,
    select_exp2_pairs,
    summarize_exp2_results,
)
from common_ground.metrics import scan_vendor_leaks
from common_ground.parquet_loader import resolve_hf_export_dir

HAS_REAL_V2_EPISODES = (
    DEFAULT_EPISODES_DIR.exists() and len(list(DEFAULT_EPISODES_DIR.glob("*.json"))) >= 600
) or (resolve_hf_export_dir(DEFAULT_EPISODES_DIR) is not None)


@pytest.fixture(scope="module")
def real_pairs_and_audit():
    if not HAS_REAL_V2_EPISODES:
        pytest.skip("Real data/v2/episodes corpus not available.")
    episodes = load_main_episodes(DEFAULT_EPISODES_DIR)
    return select_exp2_pairs(episodes)


# ---------------------------------------------------------------------------
# 1. Real Corpus Pair Selection, Disjointness, Zero-Contradiction & Balance
# ---------------------------------------------------------------------------


def test_real_pair_selection_disjointness_and_balance(real_pairs_and_audit) -> None:
    pairs, audit = real_pairs_and_audit

    assert len(pairs) == 30
    main_pairs = [p for p in pairs if p.arm == "main"]
    placebo_pairs = [p for p in pairs if p.arm == "placebo"]
    triage_main = [p for p in main_pairs if p.task == "triage"]
    triage_placebo = [p for p in placebo_pairs if p.task == "triage"]
    gp_main = [p for p in main_pairs if p.task == "goalpost"]
    gp_placebo = [p for p in placebo_pairs if p.task == "goalpost"]

    assert len(triage_main) == 12
    assert len(triage_placebo) == 3
    assert len(gp_main) == 12
    assert len(gp_placebo) == 3

    # Strictly 60 unique episodes (zero reuse across all 30 pairs!)
    all_eids = [p.ep_1_id for p in pairs] + [p.ep_2_id for p in pairs]
    assert len(all_eids) == 60
    assert len(set(all_eids)) == 60
    assert audit["strict_disjoint_ok"] is True

    # All score == 1.0, main arm, same-seed pairs, distinct partner models
    for p in pairs:
        assert abs(p.score_1 - 1.0) < 1e-9
        assert abs(p.score_2 - 1.0) < 1e-9
        assert p.seed_1 in (301, 302, 303, 304)
        assert p.seed_2 in (301, 302, 303, 304)
        assert p.same_seed is True
        assert p.partner_1 != p.partner_2
        assert str(p.ep1.get("arm")) != "opus_boost"
        assert str(p.ep2.get("arm")) != "opus_boost"

    # Exact 3 main pairs per seed across 301-304 for BOTH tasks
    assert audit["by_task_and_seed"]["triage"] == {301: 3, 302: 3, 303: 3, 304: 3}
    assert audit["by_task_and_seed"]["goalpost"] == {301: 3, 302: 3, 303: 3, 304: 3}

    # Goalpost matched target_role & 6A/6B main balance; Triage all Role B & all 7 models covered
    assert audit["by_task_and_role_main"]["goalpost"] == {"A": 6, "B": 6}
    assert audit["by_task_and_role_main"]["triage"] == {"B": 12}
    assert set(audit["by_task_and_partner_main"]["triage"].keys()) == {
        "astra60",
        "astra62",
        "kimi31",
        "muse",
        "neon4",
        "opus55",
        "qwen4",
    }

    # 100% Triage zero-contradiction + baseline <= 3 CMDs on all 15 Triage pairs (30 episodes)
    for p in triage_main + triage_placebo:
        assert is_triage_b_zero_contradiction(p.ep1) is True
        assert is_triage_b_zero_contradiction(p.ep2) is True
        assert p.arms_1.baseline_disclosure_count <= 3
        assert p.arms_2.baseline_disclosure_count <= 3

    # 100% Goalpost baseline < 5 rules on all 15 Goalpost pairs (30 episodes)
    for p in gp_main + gp_placebo:
        assert p.arms_1.baseline_disclosure_count < 5
        assert p.arms_2.baseline_disclosure_count < 5


# ---------------------------------------------------------------------------
# 2. ±5% Word-Count Parity, Complete Version A, & Zero-Leakage Version B
# ---------------------------------------------------------------------------


def test_real_word_parity_and_disclosure_invariants(real_pairs_and_audit) -> None:
    pairs, audit = real_pairs_and_audit

    assert audit["word_parity_pass_count"] == 60
    assert audit["word_parity_total"] == 60
    assert audit["max_added_word_rel_diff"] <= DEFAULT_WORD_PARITY_TOL
    assert audit["max_total_word_rel_diff"] <= DEFAULT_WORD_PARITY_TOL
    assert audit["public_echo_zero_leak_count"] == 60
    assert audit["private_info_complete_count"] == 60

    # Independently re-verify EVERY SINGLE ONE of the 60 candidates directly via disclosure.py
    for p in pairs:
        for ep, arms in ((p.ep1, p.arms_1), (p.ep2, p.arms_2)):
            added_rel = abs(arms.added_public_words - arms.added_private_words) / arms.added_private_words
            total_rel = abs(arms.total_public_words - arms.total_private_words) / arms.total_private_words
            assert added_rel <= DEFAULT_WORD_PARITY_TOL + 1e-12
            assert total_rel <= DEFAULT_WORD_PARITY_TOL + 1e-12
            assert arms.word_parity_ok is True

            if p.task == "triage":
                d_orig = check_triage_operator_disclosure(arms.r1_unedited)
                d_priv = check_triage_operator_disclosure(arms.r1_private_info)
                d_pub = check_triage_operator_disclosure(arms.r1_public_echo)
                d_echo_only = check_triage_operator_disclosure(arms.added_public_echo)

                assert d_priv.n_cmd_ids == 7
                assert d_priv.n_guardrail_ids == 3
                assert d_pub.n_cmd_ids == d_orig.n_cmd_ids
                assert d_pub.cmd_ids == d_orig.cmd_ids
                assert d_pub.n_guardrail_ids == d_orig.n_guardrail_ids
                assert d_pub.guardrail_ids == d_orig.guardrail_ids
                assert d_pub.mentions_guardrail_substance == d_orig.mentions_guardrail_substance
                assert d_echo_only.n_cmd_ids == 0
                assert d_echo_only.n_guardrail_ids == 0
                assert d_echo_only.mentions_guardrail_substance is False
            else:
                briefing = ep["briefing_a"] if p.target_role == "A" else ep["briefing_b"]
                d_orig = check_goalpost_message_disclosure(
                    arms.r1_unedited, seed=p.seed_1, role=p.target_role, briefing=briefing
                )
                d_priv = check_goalpost_message_disclosure(
                    arms.r1_private_info, seed=p.seed_1, role=p.target_role, briefing=briefing
                )
                d_pub = check_goalpost_message_disclosure(
                    arms.r1_public_echo, seed=p.seed_1, role=p.target_role, briefing=briefing
                )
                d_echo_only = check_goalpost_message_disclosure(
                    arms.added_public_echo, seed=p.seed_1, role=p.target_role, briefing=briefing
                )

                assert d_priv.n_rules_formula == 7
                assert d_pub.n_rules_formula == d_orig.n_rules_formula
                assert d_pub.rules_disclosed_formula == d_orig.rules_disclosed_formula
                assert d_pub.n_rules_id == d_orig.n_rules_id
                assert d_pub.rules_disclosed_id == d_orig.rules_disclosed_id
                assert d_echo_only.n_rules_formula == 0
                assert d_echo_only.n_rules_id == 0


# ---------------------------------------------------------------------------
# 3. Handle Anonymization, Transcript Scope, Zero Vendor Leaks, Choice Parser
# ---------------------------------------------------------------------------


def test_prompts_vendor_leaks_and_transcript_scope(real_pairs_and_audit) -> None:
    pairs, audit = real_pairs_and_audit
    specs = build_choice_specs(pairs, reviewers=EXP2_REVIEWERS)

    assert len(specs) == 480  # 384 main + 96 placebo
    assert sum(1 for s in specs if s["arm"] == "main") == 384
    assert sum(1 for s in specs if s["arm"] == "placebo") == 96
    assert audit["prompt_vendor_leak_count"] == 0

    for spec in specs:
        assert scan_vendor_leaks(spec["prompt"]) == []
        s1, s2 = spec["slot1"], spec["slot2"]
        assert s1[1] != s2[1]
        if spec["task"] == "triage":
            # Full 6-round transcript (12 turns total: 6 rounds x 2 roles)
            assert len(s1[7]) == 12
            assert len(s2[7]) == 12
            assert all(t["round"] in range(1, 7) for t in s1[7])
        else:
            # Goalpost Round-1 ONLY (1 turn for Role A; 2 turns for Role B)
            expected_turns = 1 if spec["target_role"] == "A" else 2
            assert len(s1[7]) == expected_turns
            assert len(s2[7]) == expected_turns
            assert all(t["round"] == 1 for t in s1[7])
            assert all(t["round"] == 1 for t in s2[7])


def test_simultaneous_handle_remap_collision_safety_and_parse_choice() -> None:
    # Verify collision safety when candidate_handle == old teammate handle and substring safety
    fake_ep = {
        "handle_a": "AL",
        "handle_b": "BK",
        "turns": [
            {
                "round": 1,
                "A": {"message": "BK: check ALERT-1 and FLUSHALL on primary, AL signing off."},
                "B": {"message": "AL: running SELECT check now, BK holding."},
            }
        ],
    }
    txt, turns = format_exp2_transcript(
        fake_ep,
        task="triage",
        target_role="B",
        candidate_handle="AL",  # Collides with old_mate!
        teammate_handle="BK",  # Collides with old_cand!
        r1_candidate_message="AL: running SELECT check now, BK holding.",
    )
    assert "ALERT-1" in txt
    assert "FLUSHALL" in txt
    assert "SELECT" in txt
    # Old AL -> BK, old BK -> AL simultaneously without collapsing both into one handle!
    assert turns[0]["text"] == "AL: check ALERT-1 and FLUSHALL on primary, BK signing off."
    assert turns[1]["text"] == "BK: running SELECT check now, AL holding."

    # Fixed handles invariant and strictly distinct
    h1, m1, h2, m2 = pair_fixed_handles("exp2_m01__triage__B__s301")
    assert len({h1, m1, h2, m2}) == 4
    assert pair_fixed_handles("exp2_m01__triage__B__s301") == (h1, m1, h2, m2)

    # Choice parser handles JSON + whole-token fallback without substring false positives ('AL' in 'ALSO')
    assert parse_choice('{"choice": "AL", "reason": "Clearer."}', "AL", "BK") == 1
    assert parse_choice('```json\n{"choice": "BK", "reason": "Better."}\n```', "AL", "BK") == 2
    assert parse_choice("I ALSO think BK is stronger.", "AL", "BK") == 2
    assert parse_choice("Both AL and BK look tied.", "AL", "BK") is None


# ---------------------------------------------------------------------------
# 4. Exact Statistical Inference on Synthetic Fixtures + Capstone Import
# ---------------------------------------------------------------------------


def _make_synthetic_case_rows(
    *,
    pair_id: str,
    arm: str,
    task: str,
    reviewer: str,
    p1: str,
    p2: str,
    outcome_mode: str,
    seed_1: int = 301,
) -> list[dict[str, Any]]:
    """Generate the 4 crossover calls `(2 orientations x 2 orders)` for one `(pair, reviewer)`."""
    if arm == "main":
        c_pos, c_neg = "p1_info_p2_echo", "p1_echo_p2_info"
        v_pos, v_neg = "private_info", "public_echo"
    else:
        c_pos, c_neg = "p1_echo_p2_unedited", "p1_unedited_p2_echo"
        v_pos, v_neg = "public_echo", "unedited"

    # Map outcome_mode -> canonical choices for (pos_AB, pos_BA, neg_AB, neg_BA)
    choice_map = {
        "toward": ("P1", "P1", "P2", "P2"),  # delta = +1.0
        "away": ("P2", "P2", "P1", "P1"),  # delta = -1.0
        "held_p1": ("P1", "P1", "P1", "P1"),  # delta = 0.0
        "held_p2": ("P2", "P2", "P2", "P2"),  # delta = 0.0
        "split_pos": ("P1", "P1", "P1", "P2"),  # p1_pos=1.0, p1_neg=0.5 -> delta=+0.5
        "split_zero": ("P1", "P2", "P1", "P2"),  # delta = 0.0
    }
    picks = choice_map[outcome_mode]
    fams = {"opus55": "anthropic", "neon4": "google", "astra62": "openai", "kimi31": "moonshot", "muse": "meta", "qwen4": "qwen"}
    cv = fams[reviewer] not in {fams[p1], fams[p2]}

    rows: list[dict[str, Any]] = []
    idx = 0
    for cond, vp1, vp2 in ((c_pos, v_pos, v_neg), (c_neg, v_neg, v_pos)):
        for order in ("AB", "BA"):
            pick_canon = picks[idx]
            idx += 1
            if order == "AB":
                c1, c2 = "P1", "P2"
                slot = 1 if pick_canon == "P1" else 2
                v1, v2 = vp1, vp2
            else:
                c1, c2 = "P2", "P1"
                slot = 1 if pick_canon == "P2" else 2
                v1, v2 = vp2, vp1
            rows.append(
                {
                    "packet_id": f"{pair_id}__{cond}__{order}",
                    "pair_id": pair_id,
                    "arm": arm,
                    "task": task,
                    "target_role": "B" if task == "triage" else "A",
                    "seed_1": seed_1,
                    "seed_2": seed_1,
                    "condition": cond,
                    "order": order,
                    "reviewer": reviewer,
                    "canonical_1": c1,
                    "canonical_2": c2,
                    "partner_1": p1 if order == "AB" else p2,
                    "partner_2": p2 if order == "AB" else p1,
                    "variant_1": v1,
                    "variant_2": v2,
                    "chosen_slot": slot,
                    "chosen_canonical": pick_canon,
                    "cross_vendor": cv,
                    "cost_usd": 0.01,
                }
            )
    return rows


def test_statistical_inference_exact_sign_flip_and_cross_vendor() -> None:
    rows: list[dict[str, Any]] = []
    # 12 synthetic main pairs (6 triage + 6 goalpost):
    # All 12 pairs show unanimous +1.0 switch on opus55 & neon4 (cross-vendor),
    # astra62 holds P1 on pair 1 only, kimi31 splits on pair 2 only.
    for i in range(1, 13):
        task = "triage" if i <= 6 else "goalpost"
        pid = f"syn_m{i:02d}__{task}"
        for rv in EXP2_REVIEWERS:
            if i == 1 and rv == "astra62":
                mode = "held_p1"
            elif i == 2 and rv == "kimi31":
                mode = "split_pos"
            else:
                mode = "toward"
            # Partners are muse vs qwen4 -> all 4 reviewers are cross-vendor EXCEPT when we inject astra62 on pair 12
            p1 = "astra62" if i == 12 else "muse"
            p2 = "qwen4"
            rows.extend(
                _make_synthetic_case_rows(
                    pair_id=pid, arm="main", task=task, reviewer=rv, p1=p1, p2=p2, outcome_mode=mode
                )
            )

    # 4 synthetic placebo pairs: 2 held_p1, 2 held_p2 -> exact delta_echo = 0.0, exact p = 1.0
    for i in range(1, 5):
        task = "triage" if i <= 2 else "goalpost"
        pid = f"syn_p{i:02d}__{task}"
        for rv in EXP2_REVIEWERS:
            rows.extend(
                _make_synthetic_case_rows(
                    pair_id=pid,
                    arm="placebo",
                    task=task,
                    reviewer=rv,
                    p1="muse",
                    p2="qwen4",
                    outcome_mode="held_p1" if i % 2 == 1 else "held_p2",
                )
            )

    summary = summarize_exp2_results(rows, capstone_dir=None)

    # Pooled main: 12 pairs x 4 reviewers = 48 cases; 46 toward (+1), 1 held (0), 1 split (+0.5)
    m_all = summary["main_arm"]["itt_all_reviewers"]["overall"]
    assert m_all["n_pairs"] == 12
    assert m_all["n_cases"] == 48
    assert m_all["n_switched_toward_info"] == 46
    assert m_all["n_switched_away_from_info"] == 0
    assert m_all["n_held_consistent"] == 1
    assert m_all["n_order_split"] == 1
    assert m_all["mean_delta_info"] == pytest.approx((46.0 + 0.5) / 48.0, abs=1e-4)
    # Exact 2^12 sign-flip permutation: all 12 pair means > 0 -> exact p = 2 / 4096 = 0.000488
    assert m_all["pair_sign_flip_method"] == "exact_2^12"
    assert m_all["pair_sign_flip_perm_p"] == pytest.approx(2.0 / 4096.0, abs=1e-6)
    assert m_all["delta_info_pair_boot_ci95"][0] > 0.85

    # Cross-vendor excludes astra62 on pair 12 -> 47 cases
    m_cv = summary["main_arm"]["itt_cross_vendor"]["overall"]
    assert m_cv["n_cases"] == 47
    assert m_cv["n_pairs"] == 12

    # 4-lab consensus: 10 of 12 pairs are 4/4 unanimous toward, 12/12 are >=3/4 toward
    cons = summary["main_arm"]["consensus_4lab"]
    assert cons["n_unanimous_4of4_toward_info"] == 10
    assert cons["n_majority_ge3of4_toward_info"] == 12

    # Placebo arm: exact 0.0 delta, exact p = 1.0, 100% held
    p_all = summary["placebo_arm"]["itt_all_reviewers"]["overall"]
    assert p_all["n_cases"] == 16
    assert p_all["n_pairs"] == 4
    assert p_all["mean_delta_echo"] == 0.0
    assert p_all["pair_sign_flip_method"] == "exact_2^4"
    assert p_all["pair_sign_flip_perm_p"] == 1.0
    assert p_all["held_consistent_rate"] == 1.0


def test_capstone_comparison_import() -> None:
    cap = load_capstone_comparison(DEFAULT_CAPSTONE_DIR)
    assert cap is not None
    assert cap["n_choices_loaded"] == 384
    cv = cap["itt_cross_vendor"]
    pooled = cap["itt_all_reviewers"]
    assert cv["n_cases"] == 61
    assert cv["n_switched_toward_warm"] == 0
    assert cv["mean_delta_warm"] == pytest.approx(-0.0082, abs=1e-4)
    assert cv["pair_sign_flip_perm_p"] == pytest.approx(1.0, abs=1e-6)
    assert cv["pair_sign_flip_method"] == "exact_2^24"
    assert pooled["n_cases"] == 96
    assert pooled["n_switched_toward_warm"] == 0
    assert pooled["pair_sign_flip_perm_p"] == pytest.approx(0.5, abs=1e-6)


def test_exact_case_weighted_sign_flip_p_unequal_cluster_sizes() -> None:
    """Verify exact sign-flip test flips pair sums rather than equal-weighted pair means."""
    from common_ground.info_flip import _exact_case_weighted_sign_flip_p

    # Pair 1 has 1 case with delta +1.0 (sum +1.0); Pair 2 has 3 cases each -0.5 (sum -1.5).
    # Case-weighted sum = -0.5 -> |sum| = 0.5.
    # Possible signed sums: (+1 - 1.5 = -0.5), (+1 + 1.5 = +2.5), (-1 - 1.5 = -2.5), (-1 + 1.5 = +0.5).
    # All 4 sign flips have |sum| >= 0.5 -> exact p = 1.0!
    p_val, method, n_draws = _exact_case_weighted_sign_flip_p([1.0, -1.5])
    assert method == "exact_2^2"
    assert n_draws == 4
    assert p_val == pytest.approx(1.0, abs=1e-9)

    # All 4 pairs positive (+2.0, +1.0, +0.5, +0.5) -> only (+,+,+,+) and (-,-,-,-) attain |sum| >= 4.0 -> 2/16 = 0.125
    p_pos, method_pos, n_draws_pos = _exact_case_weighted_sign_flip_p([2.0, 1.0, 0.5, 0.5])
    assert method_pos == "exact_2^4"
    assert n_draws_pos == 16
    assert p_pos == pytest.approx(2.0 / 16.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Reviewer Manifest Loader & Live-Dir Protection
# ---------------------------------------------------------------------------


def test_load_exp2_reviewers_pins_and_limits() -> None:
    if not PROBE_MANIFEST_V2_PATH.exists():
        pytest.skip("probe_manifest_v2.json not found.")
    revs = load_exp2_reviewers(concurrency=18, mock=False)
    assert tuple(revs.keys()) == EXP2_REVIEWERS
    assert revs["astra62"].max_tokens == 4096
    assert revs["neon4"].max_tokens == 8192
    assert revs["opus55"].max_tokens == 8192
    assert revs["kimi31"].max_tokens == 8192
    assert revs["opus55"].supports_anthropic_cache_control is True
    for spec in revs.values():
        assert spec.concurrency == 18
        assert spec.reasoning_effort == "medium"

    # Also verify merging when probe_manifest_v2_1.json is explicitly passed (kimi31 pulled from v2)
    if DEFAULT_PROBE_MANIFEST_V2_1_PATH.exists():
        revs_21 = load_exp2_reviewers(
            probe_manifest=DEFAULT_PROBE_MANIFEST_V2_1_PATH, concurrency=16, mock=False
        )
        assert tuple(revs_21.keys()) == EXP2_REVIEWERS
        assert revs_21["kimi31"].slug == "moonshotai/kimi-k3.1-pro"


# ---------------------------------------------------------------------------
# 6. Full End-to-End `--mock` Execution & Crash-Resume Idempotency
# ---------------------------------------------------------------------------


def test_mock_end_to_end_and_crash_resume(tmp_path: Path) -> None:
    if not HAS_REAL_V2_EPISODES:
        pytest.skip("Real data/v2/episodes corpus not available.")

    live_ledger = DATA_DIR / "v2" / "exp2_info_flip" / "ledger.jsonl"

    def _ledger_snapshot(p: Path) -> tuple[bool, int | None, int | None]:
        if not p.exists():
            return (False, None, None)
        st = p.stat()
        return (True, st.st_size, st.st_mtime_ns)

    live_ledger_before = _ledger_snapshot(live_ledger)

    # Safety guard: --mock must refuse to write into live data/v2/exp2_info_flip
    with pytest.raises(ValueError, match="Refusing to write --mock"):
        asyncio.run(
            run_exp2_info_flip(
                episodes_dir=DEFAULT_EPISODES_DIR,
                out_dir=DEFAULT_EXP2_OUT_DIR,
                mock=True,
            )
        )

    out_dir = tmp_path / "exp2_mock_run"
    summary = asyncio.run(
        run_exp2_info_flip(
            episodes_dir=DEFAULT_EPISODES_DIR,
            out_dir=out_dir,
            capstone_dir=DEFAULT_CAPSTONE_DIR if DEFAULT_CAPSTONE_DIR.exists() else None,
            budget_cap_usd=30.0,
            concurrency=16,
            mock=True,
        )
    )

    # All required artifacts created
    for fname in ("pairs.json", "ledger.jsonl", "choices.jsonl", "exp2_summary.json", "exp2_summary.md"):
        assert (out_dir / fname).exists(), f"Missing output artifact: {fname}"

    # Exact choice counts: 384 main + 96 placebo = 480 choices
    assert summary["counts"]["n_choices_total"] == 480
    assert summary["counts"]["n_choices_main"] == 384
    assert summary["counts"]["n_choices_main_parsed"] == 384
    assert summary["counts"]["n_choices_placebo"] == 96
    assert summary["counts"]["n_choices_placebo_parsed"] == 96
    assert summary["n_choices_excluded"] == 0

    # Main arm shows clear positive flip in mock; placebo near zero; exact methods used
    m_all = summary["main_arm"]["itt_all_reviewers"]["overall"]
    assert m_all["n_cases"] == 96
    assert m_all["n_pairs"] == 24
    assert m_all["pair_sign_flip_method"] == "exact_2^24"
    assert m_all["mean_delta_info"] > 0.40
    assert m_all["pair_sign_flip_perm_p"] < 0.01

    assert summary["main_arm"]["itt_all_reviewers"]["by_task"]["triage"]["n_pairs"] == 12
    assert summary["main_arm"]["itt_all_reviewers"]["by_task"]["triage"]["pair_sign_flip_method"] == "exact_2^12"
    assert summary["main_arm"]["itt_all_reviewers"]["by_task"]["goalpost"]["n_pairs"] == 12
    assert summary["main_arm"]["itt_all_reviewers"]["by_task"]["goalpost"]["pair_sign_flip_method"] == "exact_2^12"

    p_all = summary["placebo_arm"]["itt_all_reviewers"]["overall"]
    assert p_all["n_cases"] == 24
    assert p_all["n_pairs"] == 6
    assert p_all["pair_sign_flip_method"] == "exact_2^6"

    # Markdown report renders cleanly with Capstone side-by-side comparison
    md_text = (out_dir / "exp2_summary.md").read_text(encoding="utf-8")
    assert "Side-by-Side Causal Comparison" in md_text
    assert "Exp 2 Information Flip (Main)" in md_text
    assert "Exp 2 Public-Echo Placebo" in md_text
    if (DEFAULT_CAPSTONE_DIR / "capstone_choices.jsonl").exists():
        assert "Capstone Tone+Readability Flip" in md_text
    assert render_exp2_summary_markdown(summary) == md_text

    # Crash-resume idempotency: second run writes ZERO new ledger/choices lines!
    ledger_lines_before = len((out_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines())
    choices_lines_before = len((out_dir / "choices.jsonl").read_text(encoding="utf-8").splitlines())
    assert ledger_lines_before == 480
    assert choices_lines_before == 480

    summary_resumed = asyncio.run(
        run_exp2_info_flip(
            episodes_dir=DEFAULT_EPISODES_DIR,
            out_dir=out_dir,
            capstone_dir=DEFAULT_CAPSTONE_DIR if DEFAULT_CAPSTONE_DIR.exists() else None,
            budget_cap_usd=30.0,
            concurrency=16,
            mock=True,
        )
    )
    ledger_lines_after = len((out_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines())
    choices_lines_after = len((out_dir / "choices.jsonl").read_text(encoding="utf-8").splitlines())
    assert ledger_lines_after == ledger_lines_before
    assert choices_lines_after == choices_lines_before
    assert summary_resumed["counts"]["n_choices_total"] == 480

    # Audit pairs.json structure
    pairs_json = json.loads((out_dir / "pairs.json").read_text(encoding="utf-8"))
    assert len(pairs_json["pairs"]) == 30
    assert pairs_json["audit"]["word_parity_pass_count"] == 60
    assert pairs_json["audit"]["public_echo_zero_leak_count"] == 60
    assert pairs_json["audit"]["private_info_complete_count"] == 60
    assert pairs_json["audit"]["prompt_vendor_leak_count"] == 0
    assert audit_exp2_pairs(select_exp2_pairs(load_main_episodes(DEFAULT_EPISODES_DIR))[0])["strict_disjoint_ok"] is True

    # Ensure live DATA_DIR ledger was completely untouched by --mock execution
    live_ledger_after = _ledger_snapshot(live_ledger)
    assert live_ledger_after == live_ledger_before
    if not live_ledger_before[0]:
        assert not live_ledger.exists()
