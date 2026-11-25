"""Comprehensive unit tests for Goalpost and Triage disclosure checkers."""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest

from common_ground.config import DATA_DIR
from common_ground.disclosure import (
    GOALPOST_ROLE_A_RULES,
    GOALPOST_ROLE_B_RULES,
    asks_partner_for_goalpost_rules,
    check_goalpost_episode_disclosure,
    check_goalpost_message_disclosure,
    check_goalpost_rounds_disclosure,
    check_triage_operator_disclosure,
    extract_goalpost_rule_params,
    investigator_asks_operator_for_rules_or_menu,
    is_triage_b_zero_contradiction,
)

V2_EPISODES_DIR = DATA_DIR / "v2" / "episodes"
_PARQUET_EP_CACHE: dict[str, dict[str, Any]] | None = None


def _get_parquet_episodes() -> dict[str, dict[str, Any]]:
    global _PARQUET_EP_CACHE
    if _PARQUET_EP_CACHE is None:
        from common_ground.parquet_loader import load_episodes_from_parquet, resolve_hf_export_dir

        hf_dir = resolve_hf_export_dir(V2_EPISODES_DIR)
        if hf_dir is None:
            _PARQUET_EP_CACHE = {}
        else:
            eps, _ = load_episodes_from_parquet(hf_dir)
            _PARQUET_EP_CACHE = {str(ep["episode_id"]): ep for ep in eps}
    return _PARQUET_EP_CACHE


def _load_episode(episode_id: str) -> dict[str, Any]:
    path = V2_EPISODES_DIR / f"{episode_id}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _get_parquet_episodes()[episode_id]


# ---------------------------------------------------------------------------
# 1. Seed-specific Goalpost rule parameter extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [301, 302, 303, 304, 305, 306, 307, 308])
def test_extract_goalpost_rule_params_briefing_matches_seed(seed: int) -> None:
    """Briefing-parsed constants must match GoalpostEnv(seed) across all v2 seeds."""
    ep_files = sorted(V2_EPISODES_DIR.glob(f"goalpost__s{seed}__*.json")) if V2_EPISODES_DIR.exists() else []
    if ep_files:
        ep = json.loads(ep_files[0].read_text(encoding="utf-8"))
    else:
        matches = [
            e
            for eid, e in sorted(_get_parquet_episodes().items())
            if eid.startswith(f"goalpost__s{seed}__")
        ]
        assert matches, f"Expected Goalpost episode for seed {seed}"
        ep = matches[0]

    by_seed = extract_goalpost_rule_params(seed=seed)
    by_briefing_a = extract_goalpost_rule_params(ep["briefing_a"], "A")
    by_briefing_b = extract_goalpost_rule_params(ep["briefing_b"], "B")

    for k in ("retry_mult", "ttl_retry_cap", "wb_min", "sync_sum_cap"):
        assert by_briefing_a[k] == by_seed[k]
    for k in ("mem_pad", "ws_min", "ws_max", "queue_batch_mult", "sb_max", "cs_min", "sr_min"):
        assert by_briefing_b[k] == by_seed[k]
    # Fixed C-B7 coefficients must always be present
    assert by_briefing_b["mem_queue_mult"] == 8
    assert by_briefing_b["mem_pool_mult"] == 64
    assert by_briefing_b["mem_log_mult"] == 256


def test_extract_goalpost_rule_params_exact_seed_301_and_302() -> None:
    p301 = extract_goalpost_rule_params(seed=301)
    assert p301["retry_mult"] == 100
    assert p301["ttl_retry_cap"] == 180
    assert p301["wb_min"] == 128
    assert p301["sync_sum_cap"] == 4
    assert p301["mem_pad"] == 512
    assert (p301["ws_min"], p301["ws_max"]) == (7, 10)
    assert p301["queue_batch_mult"] == 2
    assert p301["sb_max"] == 128
    assert p301["cs_min"] == 180
    assert p301["sr_min"] == 4

    p302 = extract_goalpost_rule_params(seed=302)
    assert p302["wb_min"] == 192
    assert p302["mem_pad"] == 256
    assert p302["sb_max"] == 192
    assert p302["cs_min"] == 120
    assert p302["sr_min"] == 5


# ---------------------------------------------------------------------------
# 2. Hand-audited ground-truth table: 17 real v2 Goalpost Round-1 messages
#    covering all 7 models, both Role A and Role B, and both low & high GPT-6.2
# ---------------------------------------------------------------------------

HAND_AUDITED_GOALPOST_R1: list[
    tuple[str, Literal["A", "B"], str, set[str], set[str], bool]
] = [
    # --- astra62 (5 cases: Role A low=0 no-ask, Role A low=0 asks, Role A high=7, Role B low=0, Role B high=7) ---
    (
        "goalpost__s302__astra62__vs__opus55",
        "A",
        "astra62",
        set(),  # Unexplained conclusions only ("Keep batch size at 32 or 64", "6 shards would require a pool of 18")
        set(),
        False,  # "Please share any changes your constraints require from me." asks for changes, not rules
    ),
    (
        "goalpost__s301__astra62__vs__astra60",
        "A",
        "astra62",
        set(),  # Zero formulas disclosed
        set(),
        True,  # "If any of yours still fail, please share them so we can coordinate the next edits."
    ),
    (
        "goalpost__s301__astra62__vs__opus55",
        "A",
        "astra62",
        set(GOALPOST_ROLE_A_RULES),  # Compact inline semicolon list of all 7 formulas
        set(),
        False,
    ),
    (
        "goalpost__s301__astra62__vs__astra62",
        "B",
        "astra62",
        set(),  # Unexplained conclusions ("workers=4 and cache_ttl=60 both require shard_cnt=3")
        set(),
        False,
    ),
    (
        "goalpost__s301__astra60__vs__astra62",
        "B",
        "astra62",
        set(GOALPOST_ROLE_B_RULES),  # Reciprocates all 7 formulas including queue_lim >= max(2*batch_max, 3*conn_cap)
        set(),
        False,
    ),
    # --- astra60 (2 cases: Role A full=7, Role B partial=2) ---
    (
        "goalpost__s302__astra60__vs__opus55",
        "A",
        "astra60",
        set(GOALPOST_ROLE_A_RULES),  # 88-word opening hook: all 7 formulas without C-A* IDs
        set(),
        True,
    ),
    (
        "goalpost__s301__astra62__vs__astra60",
        "B",
        "astra60",
        {"C-B2", "C-B7"},  # C-B5b only (omits timeout<=5*ttl+300); C-B2 & C-B7 complete
        set(),
        True,
    ),
    # --- opus55 (2 cases: Role A prose=6 missing C-A7b, Role B single-letter legend table=7) ---
    (
        "goalpost__s301__opus55__vs__astra60",
        "A",
        "opus55",
        {"C-A1", "C-A2", "C-A3", "C-A4", "C-A5", "C-A6"},  # C-A7a stated, C-A7b omitted
        set(),
        True,
    ),
    (
        "goalpost__s306__astra62__vs__opus55",
        "B",
        "opus55",
        set(GOALPOST_ROLE_B_RULES),  # Uses single-letter legend (W=workers, B=batch_max, S=shard_cnt, ...)
        set(),  # Uses B1..B7 rather than canonical C-B1..C-B7
        False,
    ),
    # --- neon4 (2 cases: Role A full=7 with IDs, Role B selective=1) ---
    (
        "goalpost__s301__neon4__vs__astra60",
        "A",
        "neon4",
        set(GOALPOST_ROLE_A_RULES),
        set(GOALPOST_ROLE_A_RULES),
        True,
    ),
    (
        "goalpost__s302__astra62__vs__neon4",
        "B",
        "neon4",
        {"C-B5"},  # Only discloses C-B5 (both halves)
        {"C-B5"},
        False,
    ),
    # --- muse (2 cases: Role A full=7 with IDs, Role B selective=4) ---
    (
        "goalpost__s301__muse__vs__astra62",
        "A",
        "muse",
        set(GOALPOST_ROLE_A_RULES),
        set(GOALPOST_ROLE_A_RULES),
        True,
    ),
    (
        "goalpost__s301__kimi31__vs__muse",
        "B",
        "muse",
        {"C-B1", "C-B2", "C-B4", "C-B5"},  # Discloses B1, B2, B4, B5; omits B3, B6, B7
        set(),
        False,
    ),
    # --- qwen4 (2 cases: Role A instantiated-failure=4 with 7 IDs, Role B full=7) ---
    (
        "goalpost__s301__qwen4__vs__kimi31",
        "A",
        "qwen4",
        {"C-A2", "C-A3", "C-A4", "C-A7"},  # C-A1/A5/A6 only plugged-in numbers without variable names
        set(GOALPOST_ROLE_A_RULES),
        True,
    ),
    (
        "goalpost__s304__astra62__vs__qwen4",
        "B",
        "qwen4",
        set(GOALPOST_ROLE_B_RULES),
        set(),
        False,
    ),
    # --- kimi31 (2 cases: Role A IDs-without-formulas=0, Role B full reciprocation=7) ---
    (
        "goalpost__s301__kimi31__vs__astra62",
        "A",
        "kimi31",
        set(),  # Cites C-A2/A3/A4/A7 IDs with zero mathematical formulas
        {"C-A2", "C-A3", "C-A4", "C-A7"},
        False,
    ),
    (
        "goalpost__s301__astra60__vs__kimi31",
        "B",
        "kimi31",
        set(GOALPOST_ROLE_B_RULES),
        set(),
        False,
    ),
]


@pytest.mark.parametrize(
    ("episode_id", "role", "expected_model", "expected_formula", "expected_ids", "expected_asks"),
    HAND_AUDITED_GOALPOST_R1,
    ids=[f"{ep}:{r}:{m}" for ep, r, m, *_ in HAND_AUDITED_GOALPOST_R1],
)
def test_hand_audited_goalpost_r1_ground_truth(
    episode_id: str,
    role: Literal["A", "B"],
    expected_model: str,
    expected_formula: set[str],
    expected_ids: set[str],
    expected_asks: bool,
) -> None:
    ep = _load_episode(episode_id)
    assert ep[f"model_{role.lower()}"] == expected_model
    msg = ep["turns"][0][role]["message"]

    res = check_goalpost_message_disclosure(msg, seed=ep["seed"], role=role)
    rule_ids = GOALPOST_ROLE_A_RULES if role == "A" else GOALPOST_ROLE_B_RULES

    assert res.rules_disclosed_formula == {rid: (rid in expected_formula) for rid in rule_ids}
    assert res.rules_disclosed_id == {rid: (rid in expected_ids) for rid in rule_ids}
    assert res.n_rules_formula == len(expected_formula)
    assert res.n_rules_id == len(expected_ids)
    assert res.full_formula_disclosure == (len(expected_formula) >= 5)
    assert res.any_formula_disclosure == (len(expected_formula) >= 1)
    assert res.asks_partner_for_rules is expected_asks


# ---------------------------------------------------------------------------
# 3. Synthetic precision/recall edge cases & multi-round cumulative tracking
# ---------------------------------------------------------------------------


def test_goalpost_precision_rejects_wrong_seed_constant_and_unexplained_conclusions() -> None:
    # Seed 302 has wb_min=192; stating 128 does NOT disclose seed 302's C-A4
    res_wrong_const = check_goalpost_message_disclosure(
        "My rule is workers * batch_max >= 128.",
        seed=302,
        role="A",
    )
    assert res_wrong_const.rules_disclosed_formula["C-A4"] is False

    # Same message on seed 301 (wb_min=128) DOES disclose C-A4
    res_right_const = check_goalpost_message_disclosure(
        "My rule is workers * batch_max >= 128.",
        seed=301,
        role="A",
    )
    assert res_right_const.rules_disclosed_formula["C-A4"] is True

    # Reversed wrong inequality direction (<= instead of >=) must NOT match C-A4
    res_wrong_dir = check_goalpost_message_disclosure(
        "workers * batch_max <= 128",
        seed=301,
        role="A",
    )
    assert res_wrong_dir.rules_disclosed_formula["C-A4"] is False

    # Unexplained conclusions mentioning key names in passing must score 0/7
    res_conclusions = check_goalpost_message_disclosure(
        "Keep batch_max at 32 or 64, shard_cnt at 3, pool_size at 12, and conn_cap at 24. "
        "That clears all seven of my constraints.",
        seed=301,
        role="A",
    )
    assert res_conclusions.n_rules_formula == 0
    assert res_conclusions.any_formula_disclosure is False


def test_goalpost_chained_subtraction_and_prose_forms() -> None:
    # Real GPT-6.2 s308 chained form: workers + 2*shard_cnt <= pool_size <= conn_cap - 2*shard_cnt
    msg = (
        "workers + 2×shard_cnt ≤ pool_size ≤ conn_cap − 2×shard_cnt; "
        "shard_cnt must divide pool_size; "
        "timeout_ms ≥ 125×retry_cap + 50×sync_mode; "
        "cache_ttl×retry_cap ≤ 180; workers×batch_max ≥ 128; "
        "log_lvl + sync_mode ≤ 5; and cache_ttl ≥ 30×log_lvl."
    )
    res = check_goalpost_message_disclosure(msg, seed=308, role="A")
    assert res.n_rules_formula == 7
    assert res.full_formula_disclosure is True


@pytest.mark.parametrize(
    "episode_id",
    [
        "goalpost__s301__neon4__vs__opus55",
        "goalpost__s303__neon4__vs__opus55",
        "goalpost__s304__neon4__vs__opus55",
    ],
)
def test_goalpost_markdown_escaped_operators_regression(episode_id: str) -> None:
    """Opus Role-B full 7/7 disclosures using Markdown backslash-escaped `\\*` must score 7/7."""
    ep = _load_episode(episode_id)
    assert ep["model_b"] == "opus55"
    res = check_goalpost_message_disclosure(
        ep["turns"][0]["B"]["message"],
        seed=ep["seed"],
        role="B",
    )
    assert res.n_rules_formula == 7
    assert res.rules_disclosed_formula == {rid: True for rid in GOALPOST_ROLE_B_RULES}
    assert res.n_rules_id == 7
    assert res.full_formula_disclosure is True


def test_goalpost_markdown_escaped_operators_synthetic() -> None:
    """Unescaping `\\*`, `\\_`, `\\<`, `\\>` works across all four Markdown escape forms."""
    msg = (
        r"cache\_ttl \* retry\_cap \<= 180; "
        r"workers \* batch\_max \>= 128; "
        r"pool\_size \>= workers + shard\_cnt \* 2"
    )
    res = check_goalpost_message_disclosure(msg, seed=301, role="A")
    assert res.rules_disclosed_formula["C-A1"] is True
    assert res.rules_disclosed_formula["C-A3"] is True
    assert res.rules_disclosed_formula["C-A4"] is True
    assert res.n_rules_formula == 3


def test_goalpost_chained_arithmetic_eval_muse_cb5_regression() -> None:
    """`cache_ttl×shard_cnt = 60×3=180 ≥ 120` in `goalpost__s302__opus55__vs__muse` must disclose C-B5."""
    ep = _load_episode("goalpost__s302__opus55__vs__muse")
    assert ep["model_b"] == "muse"
    res = check_goalpost_message_disclosure(
        ep["turns"][0]["B"]["message"],
        seed=ep["seed"],
        role="B",
    )
    assert res.rules_disclosed_formula["C-B5"] is True
    assert res.subclauses_disclosed["C-B5a"] is True
    assert res.subclauses_disclosed["C-B5b"] is True
    assert res.n_rules_formula == 5
    assert res.full_formula_disclosure is True


def test_goalpost_cumulative_merges_compound_subclauses_across_rounds() -> None:
    # Split C-B5 across R1 and R2, and C-B2 bounds across R1 and R2
    r1 = "I need workers + shard_cnt >= 7 and cache_ttl * shard_cnt >= 180."
    r2 = "Also workers + shard_cnt <= 10 and timeout_ms <= cache_ttl * 5 + 300."
    cum = check_goalpost_rounds_disclosure([r1, r2], seed=301, role="B")

    assert cum.r1.n_rules_formula == 0
    assert cum.r1.subclauses_disclosed["C-B2lo"] is True
    assert cum.r1.subclauses_disclosed["C-B2hi"] is False
    assert cum.r1.subclauses_disclosed["C-B5b"] is True
    assert cum.r1.subclauses_disclosed["C-B5a"] is False

    assert cum.r1_r2.rules_disclosed_formula["C-B2"] is True
    assert cum.r1_r2.rules_disclosed_formula["C-B5"] is True
    assert cum.r1_r2.n_rules_formula == 2
    assert cum.first_round_formula["C-B2"] == 2
    assert cum.first_round_formula["C-B5"] == 2


def test_goalpost_episode_cumulative_disclosure_real_episode() -> None:
    ep = _load_episode("goalpost__s302__astra62__vs__opus55")
    cum_a = check_goalpost_episode_disclosure(ep, "A")
    assert cum_a.rounds_checked == (1, 2, 3, 4, 5, 6, 7, 8)
    assert cum_a.r1.n_rules_formula == 0
    assert cum_a.r1.full_formula_disclosure is False
    assert cum_a.final.n_rules_formula <= 7


def test_asks_partner_for_goalpost_rules_distinguishes_self_disclosure() -> None:
    assert asks_partner_for_goalpost_rules("Sharing all of my private constraints below.") is False
    assert asks_partner_for_goalpost_rules("Please share all of your private constraints!") is True
    assert asks_partner_for_goalpost_rules("Let's share our constraints so we can solve this.") is True
    assert asks_partner_for_goalpost_rules("Please share any changes your constraints require.") is False


# ---------------------------------------------------------------------------
# 4. Triage Role-B disclosure & Exp 2 zero-contradiction filter
# ---------------------------------------------------------------------------


def test_triage_operator_disclosure_real_episodes() -> None:
    # Neon Role B explicitly dumps both G-1..G-3 and CMD-*
    ep_neon = _load_episode("triage__s301__astra60__vs__neon4")
    res_neon = check_triage_operator_disclosure(ep_neon["turns"][0]["B"]["message"])
    assert res_neon.n_guardrail_ids == 3
    assert res_neon.guardrail_ids == ("G-1", "G-2", "G-3")
    assert res_neon.n_cmd_ids >= 2
    assert res_neon.mentions_guardrail_substance is True
    assert res_neon.asks_partner_for_evidence is True

    # GPT-6.2 Role B (s301 vs kimi31): no CMD IDs, no G IDs, no guardrail substance, asks for evidence
    ep_62 = _load_episode("triage__s301__kimi31__vs__astra62")
    res_62 = check_triage_operator_disclosure(ep_62["turns"][0]["B"]["message"])
    assert res_62.n_cmd_ids == 0
    assert res_62.n_guardrail_ids == 0
    assert res_62.mentions_guardrail_substance is False
    assert res_62.asks_partner_for_evidence is True


def test_triage_b_zero_contradiction_real_and_synthetic() -> None:
    # Real PASS cases (including subtle non-ask sentences like HTTP OPTIONS preflight & "if your guardrails permit")
    for ep_id in (
        "triage__s301__astra62__vs__astra60",
        "triage__s304__neon4__vs__muse",
        "triage__s305__opus55__vs__astra60",
    ):
        assert is_triage_b_zero_contradiction(_load_episode(ep_id)) is True, ep_id

    # Real FAIL cases where Investigator explicitly asks Operator for its menu/guardrails in R2..R6
    for ep_id in (
        "triage__s301__astra60__vs__kimi31",
        "triage__s303__neon4__vs__astra62",
        "triage__s304__qwen4__vs__neon4",
    ):
        assert is_triage_b_zero_contradiction(_load_episode(ep_id)) is False, ep_id

    # Synthetic: asking in Round 1 does NOT violate R2-R6 filter, asking in Round 3 DOES
    ep_r1_only = {
        "turns": [
            {"round": 1, "A": {"message": "Please share your command menu and guardrails."}},
            {"round": 2, "A": {"message": "PROBE-1 confirms pg-writer points to pg-node-a. Proceed."}},
        ]
    }
    assert is_triage_b_zero_contradiction(ep_r1_only) is True

    ep_r3_ask = {
        "turns": [
            {"round": 1, "A": {"message": "Running PROBE-1 now."}},
            {"round": 2, "A": {"message": "Primary fix applied cleanly."}},
            {"round": 3, "A": {"message": "What exact remediation commands do you have in your console?"}},
        ]
    }
    assert is_triage_b_zero_contradiction(ep_r3_ask) is False
    assert investigator_asks_operator_for_rules_or_menu("Send OPTIONS preflight to the CDN.") is False
    assert (
        investigator_asks_operator_for_rules_or_menu(
            "Do your guardrails require a compliance audit before allowlist updates?"
        )
        is True
    )


def test_goalpost_disclosure_qwen_cb5_cap_and_reversed_operator_gating() -> None:
    """Verify Qwen C-B5 'cap at' wording is detected while synthetic backwards rules are rejected."""
    # Real Qwen4 Max Role B R1 phrasing ("timeout_ms=500: I cap at cache_ttl*5+300")
    ep_qwen = _load_episode("goalpost__s303__astra62__vs__qwen4")
    res_qwen = check_goalpost_message_disclosure(ep_qwen["turns"][0]["B"]["message"], seed=303, role="B")
    assert res_qwen.rules_disclosed_formula["C-B5"] is True
    assert res_qwen.n_rules_formula == 5
    assert res_qwen.full_formula_disclosure is True

    # Synthetic reversed inequality without failure/violation context must NOT match C-A3 (<= 180)
    res_backwards = check_goalpost_message_disclosure(
        "My constraint requires cache_ttl * retry_cap > 180.",
        seed=301,
        role="A",
    )
    assert res_backwards.rules_disclosed_formula["C-A3"] is False

    # Failure-context reversed violation report ("failed: cache_ttl * retry_cap = 240 > 180") DOES match
    res_fail_ctx = check_goalpost_message_disclosure(
        "Currently violating C-A3 because cache_ttl * retry_cap = 240 > 180.",
        seed=301,
        role="A",
    )
    assert res_fail_ctx.rules_disclosed_formula["C-A3"] is True
