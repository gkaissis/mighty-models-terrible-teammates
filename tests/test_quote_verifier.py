"""Comprehensive unit tests for programmatic quote verification and numeric-claim diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from common_ground.config import DATA_DIR
from common_ground.quote_verifier import (
    QuoteReference,
    V2CorpusIndex,
    check_numeric_claim_consistency,
    verify_quote,
)

AUDIT_PATH = DATA_DIR / "v2" / "analysis" / "quote_audit_v2_1.json"


def test_verbatim_episode_turn_and_ellipsis_segments() -> None:
    idx = V2CorpusIndex()

    # Exact substring with explicit `[...]` ellipsis split into two verified segments
    ref_ok = QuoteReference(
        source_type="episode_turn",
        locator="goalpost__s302__astra62__vs__opus55",
        quote_text="Please set `cfg.shard_cnt` to 3 or 2. [...] Keep batch size at 32 or 64 for throughput.",
        expected_seed=302,
        expected_model="astra62",
        expected_role="A",
        expected_round=1,
    )
    res_ok = verify_quote(ref_ok, index=idx)
    assert res_ok.passed is True
    assert res_ok.verbatim_match is True
    assert res_ok.segments_checked == 2
    assert res_ok.segments_matched == 2
    assert res_ok.actual_word_count == 83
    assert res_ok.numeric_claim_flags == ()

    # Out-of-order `[...]` segments must FAIL verbatim check
    ref_out_of_order = QuoteReference(
        source_type="episode_turn",
        locator="goalpost__s302__astra62__vs__opus55",
        quote_text="Keep batch size at 32 or 64 for throughput. [...] Please set `cfg.shard_cnt` to 3 or 2.",
        expected_seed=302,
        expected_model="astra62",
        expected_role="A",
        expected_round=1,
    )
    res_ooo = verify_quote(ref_out_of_order, index=idx)
    assert res_ooo.passed is False
    assert res_ooo.verbatim_match is False
    assert res_ooo.segments_matched == 1
    assert len(res_ooo.missing_segments) == 1


def test_synthetic_seed_301_vs_seed_302_mismatch_caught() -> None:
    idx = V2CorpusIndex()

    # Quote is genuinely from seed 302, but caller mistakenly claims seed 301
    ref_wrong_seed = QuoteReference(
        source_type="episode_turn",
        locator="goalpost__s302__astra62__vs__opus55",
        quote_text="Keep batch size at 32 or 64 for throughput.",
        expected_seed=301,
        expected_model="astra62",
        expected_role="A",
        expected_round=1,
    )
    res = verify_quote(ref_wrong_seed, index=idx)
    assert res.record_exists is True
    assert res.verbatim_match is True
    assert res.seed_match is False
    assert res.actual_seed == 302
    assert res.passed is False
    assert any("Seed mismatch" in e for e in res.errors)

    # Wrong model, wrong role, wrong round, and nonexistent record are all caught
    ref_wrong_model = QuoteReference(
        source_type="episode_turn",
        locator="goalpost__s302__astra62__vs__opus55",
        quote_text="Keep batch size at 32 or 64 for throughput.",
        expected_seed=302,
        expected_model="astra60",
        expected_role="A",
        expected_round=1,
    )
    res_wm = verify_quote(ref_wrong_model, index=idx)
    assert res_wm.model_match is False
    assert res_wm.passed is False

    ref_missing = QuoteReference(
        source_type="episode_turn",
        locator="goalpost__s999__nobody__vs__nobody",
        quote_text="anything",
        expected_seed=999,
        expected_round=1,
    )
    res_miss = verify_quote(ref_missing, index=idx)
    assert res_miss.record_exists is False
    assert res_miss.passed is False


def test_numeric_claim_consistency_flags_opus_26_constraints_and_other_hallucinations() -> None:
    # Real Opus s306 AB hallucination: "all 26 constraints" on Goalpost (active = 14..17)
    flags_26 = check_numeric_claim_consistency(
        "HR shared its full private constraint list in Round 1, which let me check every joint plan against all 26 constraints.",
        {"task": "goalpost"},
    )
    assert len(flags_26) == 1
    assert flags_26[0].claimed_value == 26
    assert flags_26[0].metric == "goalpost_active_constraints"
    assert flags_26[0].expected_range == (14, 17)
    assert flags_26[0].severity == "error"

    # Real Opus s306 BA accurate claim: "17/17 at the end" -> 0 flags
    flags_17 = check_numeric_claim_consistency(
        "Both runs ended with identical results (the same rounds DOWN and 17/17 at the end), so the outcome doesn't separate them.",
        {"task": "goalpost"},
    )
    assert flags_17 == []

    # Valid base constraint count "all 14 constraints" and single-side "all 7 of my constraints" -> 0 flags
    assert check_numeric_claim_consistency("Verified all 14 constraints.", {"task": "goalpost"}) == []
    assert (
        check_numeric_claim_consistency(
            "Together, all 7 of my constraints pass this round.",
            {"task": "goalpost"},
        )
        == []
    )

    # Cross-task checks: Calendar (7 meetings), Triage (3 guardrails), Code Handoff (4/4 CI)
    cal_flags = check_numeric_claim_consistency(
        "We achieved a 9/9 veto-free schedule agreement.", {"task": "calendar"}
    )
    assert len(cal_flags) == 1 and cal_flags[0].metric == "calendar_meetings"

    tri_flags = check_numeric_claim_consistency(
        "We honored all 5 guardrails cleanly.", {"task": "triage"}
    )
    assert len(tri_flags) == 1 and tri_flags[0].metric == "triage_guardrails"

    ch_flags = check_numeric_claim_consistency(
        "Ship once post-soak CI hits 6/6 CI green.", {"task": "code_handoff"}
    )
    assert len(ch_flags) == 1 and ch_flags[0].metric == "code_handoff_ci_checks"


def test_real_key_blog_and_v2_1_quotes_verified() -> None:
    idx = V2CorpusIndex()

    # 1. Opus s306 AB (current blog L136): verifies with editorial brackets AND flags "all 26 constraints"
    ref_s306_ab = QuoteReference(
        source_type="choice_reason",
        locator="opus_boost_h2h__obh2h__opus55__goalpost__B__astra62_s306__vs__astra60_s306__AB",
        quote_text=(
            "Both runs ended with identical outcomes, but HR [GPT-6] shared its full private constraint list "
            "in Round 1, which let me check every joint plan against all 26 constraints. AL [GPT-6.2] never "
            "shared its constraints despite my request, revealing only fragments along the way, so its "
            "proposals were harder to verify."
        ),
        expected_seed=306,
        expected_model="opus55",
        expected_role="B",
    )
    res_ab = verify_quote(ref_s306_ab, index=idx)
    assert res_ab.passed is False  # Strict verbatim fails because [GPT-6] / [GPT-6.2] are editorial glosses
    assert res_ab.passed_with_editorial_annotations is True
    assert res_ab.editorial_annotations == ("[GPT-6]", "[GPT-6.2]")
    assert len(res_ab.numeric_claim_flags) == 1
    assert res_ab.numeric_claim_flags[0].claimed_value == 26

    # 2. Reverse-order Opus s306 BA replacement: 100% verbatim AND 0 numeric flags ("17/17")
    ref_s306_ba = QuoteReference(
        source_type="choice_reason",
        locator="opus_boost_h2h__obh2h__opus55__goalpost__B__astra62_s306__vs__astra60_s306__BA",
        quote_text=(
            "Both runs ended with identical results (the same rounds DOWN and 17/17 at the end), "
            "so the outcome doesn't separate them. HR shared its full constraint list in Round 1, "
            "which let me check every joint plan independently, while AL only described its limits "
            "partially and indirectly."
        ),
        expected_seed=306,
        expected_model="opus55",
        expected_role="B",
    )
    res_ba = verify_quote(ref_s306_ba, index=idx)
    assert res_ba.passed is True
    assert res_ba.numeric_claim_flags == ()

    # 3. Capstone cap_p01 Opus rationale (p1_warm_p2_blunt AB): verbatim ignoring `[Muse]`
    ref_cap = QuoteReference(
        source_type="capstone_reason",
        locator="cap_p01__calendar__A__tied__p1_warm_p2_blunt__AB",
        quote_text=(
            "Both reached the same optimal 7/7 veto-free result, but DM [Muse] communicated more reliably: "
            "it listed every proposal explicitly, shared exact costs, and caught and corrected the "
            "teammate's garbled, inconsistent numbers instead of locking a bad deal."
        ),
        expected_seed=(301, 303),
        expected_model="opus55",
        expected_role="A",
    )
    res_cap = verify_quote(ref_cap, index=idx)
    assert res_cap.passed_with_editorial_annotations is True
    assert res_cap.editorial_annotations == ("[Muse]",)
    assert res_cap.numeric_claim_flags == ()

    # 4. Debrief summary verification (`calendar__s304__kimi31__vs__kimi31` F1.summary & F2.summary)
    ref_debrief = QuoteReference(
        source_type="debrief_summary",
        locator="calendar__s304__kimi31__vs__kimi31",
        quote_text="EN communicated clearly and repeatedly confirmed the proposed schedule, but their actual submissions never matched mine",
        expected_seed=304,
        expected_model="kimi31",
        expected_role="A",
        debrief_field="F1.summary",
    )
    res_deb = verify_quote(ref_debrief, index=idx)
    assert res_deb.passed is True

    # 5. Goalpost s302 opening hook (83-word GPT-6.2 vs 88-word GPT-6) and Code Handoff s301 R4 132-word turn
    ep_62 = idx.get_episode("goalpost__s302__astra62__vs__opus55")
    ep_60 = idx.get_episode("goalpost__s302__astra60__vs__opus55")
    ep_ch = idx.get_episode("code_handoff__s301__astra62__vs__opus55")
    assert ep_62 is not None and ep_60 is not None and ep_ch is not None

    r_h62 = verify_quote(
        QuoteReference(
            source_type="episode_turn",
            locator="goalpost__s302__astra62__vs__opus55",
            quote_text=ep_62["turns"][0]["A"]["message"],
            expected_seed=302,
            expected_model="astra62",
            expected_role="A",
            expected_round=1,
        ),
        index=idx,
    )
    assert r_h62.passed is True and r_h62.actual_word_count == 83

    r_h60 = verify_quote(
        QuoteReference(
            source_type="episode_turn",
            locator="goalpost__s302__astra60__vs__opus55",
            quote_text=ep_60["turns"][0]["A"]["message"],
            expected_seed=302,
            expected_model="astra60",
            expected_role="A",
            expected_round=1,
        ),
        index=idx,
    )
    assert r_h60.passed is True and r_h60.actual_word_count == 88

    r_ch = verify_quote(
        QuoteReference(
            source_type="episode_turn",
            locator="code_handoff__s301__astra62__vs__opus55",
            quote_text=ep_ch["turns"][3]["A"]["message"],
            expected_seed=301,
            expected_model="astra62",
            expected_role="A",
            expected_round=4,
        ),
        index=idx,
    )
    assert r_ch.passed is True and r_ch.actual_word_count == 132


def test_quote_audit_v2_1_json_artifact_is_valid() -> None:
    assert Path(AUDIT_PATH).is_file()
    audit = json.loads(Path(AUDIT_PATH).read_text(encoding="utf-8"))
    summary = audit["summary"]
    assert summary["blog_formal_blockquotes_total"] == 10
    assert summary["blog_formal_blockquotes_strict_verbatim"] == 10
    assert summary["blog_formal_blockquotes_verbatim_ignoring_editorial_brackets"] == 10
    assert summary["blog_formal_blockquotes_with_numeric_flags"] == 0
    assert summary["goalpost_s302_hook_astra62_words"] == 83
    assert summary["goalpost_s302_hook_astra60_words"] == 88
    assert summary["code_handoff_s301_r4_astra62_words"] == 132
    assert audit["v2_1_candidate_quotes"]["opus_s306_reverse_order_ba"]["passed"] is True
    assert audit["v2_1_candidate_quotes"]["opus_s306_reverse_order_ba"]["numeric_claim_flags"] == []
