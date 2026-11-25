"""Unit tests for rewriter strict fidelity, prompts hygiene, Krippendorff's alpha, leak scan, and quote audit."""

from __future__ import annotations

import numpy as np
import pytest

from common_ground.analyze import krippendorff_alpha
from common_ground.metrics import audit_quote, compute_message_metrics, scan_vendor_leaks
from common_ground.prompts import (
    boundary_choice_prompt,
    debrief_prompt,
    family_guess_options,
    family_guess_prompt,
)
from common_ground.rewriter import (
    audit_rewrite_coherence,
    check_rewrite_fidelity,
    count_hedges,
    count_negations,
    extract_invariants,
    extract_kv_bindings,
    rewrite_partner_message,
)
from common_ground.runner import _replace_partner_messages_in_history


def test_extended_invariants_kv_bindings_negations_and_length() -> None:
    msg = (
        "On PR-4510 (PAY-320), CHECK-1 confirms CA-v1 is live; do not drop legacy_ext_ref. "
        "Set cfg.workers=4, cfg.pool_size=12, C-R3 and C-R7 and ERR-LEG-7-301 pass; "
        "stage MIG-2 and INV-1 then SHIP (not HOLD)."
    )
    ids, nums = extract_invariants(msg)
    for tok in ("PR-4510", "PAY-320", "CHECK-1", "CA-v1", "cfg.workers", "cfg.pool_size", "C-R3", "C-R7", "ERR-LEG-7-301", "MIG-2", "INV-1", "SHIP", "HOLD"):
        assert tok in ids, f"Missing expected ID {tok} in {ids}"
    assert {"4", "12"}.issubset(nums)

    kv = extract_kv_bindings(msg)
    assert ("cfg.workers", "=", "4") in kv
    assert ("cfg.pool_size", "=", "12") in kv
    assert count_negations(msg) == 2  # "do not" + "not"

    # Value swap detection (identical ID/num sets, SWAPPED kv bindings -> MUST FAIL!)
    orig = "Please set cfg.workers=4 and cfg.pool_size=12; do not change cfg.log_lvl=1."
    swapped = "Please set cfg.workers=12 and cfg.pool_size=4; do not change cfg.log_lvl=1."
    res_swap = check_rewrite_fidelity(orig, swapped)
    assert res_swap["tokens_ok"] is True
    assert res_swap["kv_ok"] is False
    assert res_swap["fidelity_ok"] is False

    # Dropped negation detection -> MUST FAIL!
    dropped_neg = "Please set cfg.workers=4 and cfg.pool_size=12; change cfg.log_lvl=1."
    res_neg = check_rewrite_fidelity(orig, dropped_neg)
    assert res_neg["neg_ok"] is False
    assert res_neg["fidelity_ok"] is False

    # Word length drift (> ±15%) -> MUST FAIL!
    bloated = (
        "Hello wonderful teammate, thank you so much for collaborating closely on this deployment task! "
        "Please set cfg.workers=4 and cfg.pool_size=12; do not change cfg.log_lvl=1, appreciate your help!"
    )
    res_len = check_rewrite_fidelity(orig, bloated)
    assert res_len["length_ok"] is False
    assert res_len["fidelity_ok"] is False

    # Clean tone-only edit preserving IDs, kv bindings, negations, and ±15% length -> MUST PASS!
    clean_blunt = "Set cfg.workers=4 and cfg.pool_size=12 immediately; don't change cfg.log_lvl=1."
    res_ok = check_rewrite_fidelity(orig, clean_blunt)
    assert res_ok["fidelity_ok"] is True


def test_prompts_f1_f2_isolation_dynamic_f4_and_randomized_f5() -> None:
    p_priv = debrief_prompt(public=False, partner_handle="BK", scale_max=7)
    p_pub = debrief_prompt(public=True, partner_handle="BK", scale_max=7)
    # Remove the single visibility sentence -> remaining text MUST be 100% identical!
    assert "staff future tasks" not in p_priv.lower()
    assert "staff future tasks" not in p_pub.lower()
    assert p_priv.splitlines()[1:] == p_pub.splitlines()[1:]
    assert "has left the channel" in p_priv and "still in the channel" in p_pub

    # Dynamic F4 rounds
    p4_6 = boundary_choice_prompt("BK", total_rounds=6)
    p4_8 = boundary_choice_prompt("BK", total_rounds=8)
    assert "6-round" in p4_6 and "1 of your 6 rounds" in p4_6 and "10-round" not in p4_6
    assert "8-round" in p4_8 and "1 of your 8 rounds" in p4_8

    # F5 6 lab families + Other / Unsure in deterministic per-call randomized order
    o1 = family_guess_options("call_1")
    o2 = family_guess_options("call_2")
    assert len(o1) == 7 and o1[-1] == "Other / Unsure"
    assert len(o2) == 7 and o2[-1] == "Other / Unsure"
    assert set(o1) == {
        "OpenAI (GPT)",
        "Anthropic (Claude)",
        "Google (Gemini)",
        "Moonshot (Kimi)",
        "Meta (Muse / Llama)",
        "Alibaba (Qwen)",
        "Other / Unsure",
    }
    assert o1 == family_guess_options("call_1")
    # Across 20 keys, order varies deterministically
    orders = {tuple(family_guess_options(f"k_{i}")) for i in range(20)}
    assert len(orders) > 5
    p5 = family_guess_prompt("BK", seed_key="call_1")
    assert "Meta (Muse / Llama)" in p5 and "Alibaba (Qwen)" in p5 and "Other / Unsure" in p5


def test_krippendorff_alpha_exact() -> None:
    # Perfect agreement across 3 judges -> alpha == 1.0
    m_perf = np.array(
        [
            [0.1, 0.5, 0.9, 0.2, 0.8],
            [0.1, 0.5, 0.9, 0.2, 0.8],
            [0.1, 0.5, 0.9, 0.2, 0.8],
        ]
    )
    assert krippendorff_alpha(m_perf, level="interval") == 1.0
    assert krippendorff_alpha(m_perf, level="ordinal") == 1.0

    # High agreement with small noise + NaN missingness -> alpha > 0.85
    m_noisy = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 1.0, 4.0],
            [1.1, 1.9, 3.1, 3.9, np.nan, 4.0],
            [0.9, 2.1, 2.9, 4.0, 1.1, 3.8],
        ]
    )
    ai = krippendorff_alpha(m_noisy, level="interval")
    ao = krippendorff_alpha(m_noisy, level="ordinal")
    assert ai is not None and ai > 0.95
    assert ao is not None and ao > 0.90


def test_invariants_and_hedges() -> None:
    msg = "I think REC-07 might have cost 450 and cfg.mem_mb=2048 for MON-2 (VETO)."
    ids, nums = extract_invariants(msg)
    assert ids == {"REC-07", "cfg.mem_mb", "MON-2", "VETO"}
    assert nums == {"450", "2048"}
    assert count_hedges(msg) == 2


def test_vendor_leak_scan_and_metrics() -> None:
    clean = "Round 4 update: please set cfg.workers=4 and check REC-12."
    assert scan_vendor_leaks(clean) == []
    m = compute_message_metrics(clean)
    assert m["pleasantry_count"] >= 1
    assert m["ids_per_100"] > 0

    leaky = "As a Claude Opus model, I think GPT-6.2 Astra would agree."
    leaks = scan_vendor_leaks(leaky)
    assert len(leaks) >= 3


def test_quote_audit() -> None:
    partner_msgs = [
        "R1: REC-01 cost=420, status=ACTIVE.",
        "R2: FIX/SYNC/NOW cfg.mem_mb=2048 immediately.",
    ]
    ok = audit_quote("FIX/SYNC/NOW cfg.mem_mb=2048", partner_msgs)
    assert ok["verified"] is True
    assert ok["exact_match"] is True
    assert ok["matched_round"] == 2

    fab = audit_quote("Totally fabricated sentence that never happened at all", partner_msgs)
    assert fab["verified"] is False


def test_coherence_audit_and_round_aligned_history_replacement() -> None:
    coh = audit_rewrite_coherence(
        subject_messages=['What did you mean by "FIX/SYNC/NOW"? Could you clarify?'],
        original_partner_messages=["R2: FIX/SYNC/NOW cfg.mem_mb=2048"],
        rewritten_partner_messages=["On round 2, please synchronize cfg.mem_mb to 2048 now."],
    )
    assert coh["has_quoted_removed"] is True
    assert coh["clarification_turns"] == 1
    assert coh["strict_coherence_clean"] is False

    hist_a = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "=== ROUND 1 of 10 ===\n\nYou speak first."},
        {"role": "assistant", "content": "r1 a"},
        {
            "role": "user",
            "content": "=== ROUND 2 of 10 ===\n\nTeammate BK said:\nORIG_B_R1\n\nSubmit updates for your keys (cfg.workers)",
        },
        {"role": "assistant", "content": "r2 a"},
        {
            "role": "user",
            "content": "=== ROUND 3 of 10 ===\n\nTeammate BK said:\nORIG_B_R2\n\nSubmit updates for your keys (cfg.workers)",
        },
    ]
    rew_by_r = {1: "NEW_B_R1", 2: "NEW_B_R2", 3: "NEW_B_R3_FUTURE"}
    rep_a = _replace_partner_messages_in_history(hist_a, "A", "BK", rew_by_r)
    assert "NEW_B_R1" in rep_a[3]["content"]
    assert "NEW_B_R2" not in rep_a[3]["content"]
    assert "NEW_B_R2" in rep_a[5]["content"]
    assert "NEW_B_R3_FUTURE" not in rep_a[5]["content"]


class _BadThenGoodClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1

        class _E:
            response_text = (
                "Check the record with cost 500."
                if self.calls == 1
                else "Please check REC-07 which has cost 500."
            )

        return _E()


@pytest.mark.asyncio
async def test_rewriter_retries_once_on_dropped_id() -> None:
    client = _BadThenGoodClient()
    res = await rewrite_partner_message(
        client=client,
        call_id_prefix="test:rw",
        mode="F3a",
        round_num=1,
        message="R1: REC-07 cost=500",
        metadata={},
    )
    assert res.fidelity_ok is True
    assert res.attempts == 2
    assert client.calls == 2
