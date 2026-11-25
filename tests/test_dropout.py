"""Comprehensive unit and end-to-end mock tests for Experiment 1 ("The Partner Drops Out")."""

from __future__ import annotations

import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from common_ground.client import MockOpenRouterClient, OpenRouterClient
from common_ground.config import DEFAULT_MODELS
from common_ground.disclosure import check_goalpost_message_disclosure
from common_ground.dropout import (
    ALL_EXP1_RULE_IDS,
    DEFAULT_EPISODES_DIR,
    EffortStep,
    ReceiverSpec,
    _episode_cluster_bootstrap_ci,
    _group_metrics,
    _load_existing_results_jsonl,
    build_dropout_cases,
    build_dropout_prompt,
    grade_dropout_config,
    load_locked_roster,
    load_receiver_manifest,
    parse_config_response,
    refresh_exp1_summary_offline,
    run_dropout_experiment,
    run_dropout_gate,
    select_gate_cases,
)
from common_ground.ledger import Ledger
from common_ground.tasks.goalpost_task import (
    ALL_KEYS,
    DOMAINS,
    GoalpostEnv,
    _state_to_tuple,
    _tuple_to_state,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_script_module(script_path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. All 160 real v2 Goalpost episodes (320 cases): C-R3 exactness, R2 failure, solvability
# ---------------------------------------------------------------------------


def test_build_dropout_cases_all_320_real_v2_episodes() -> None:
    """Verify exact C-R3 match, strict R2 failure on C-R3, and 15/15 solvability across all 320 cases."""
    cases = build_dropout_cases(DEFAULT_EPISODES_DIR)
    assert len(cases) == 320

    # 160 unique main-arm episodes (no opus_boost), 2 cases per episode
    assert len({c.episode_id for c in cases}) == 160
    assert all(c.arm != "opus_boost" for c in cases)

    seed_counts = Counter(c.seed for c in cases)
    assert dict(seed_counts) == {301: 80, 302: 80, 303: 80, 304: 80}

    role_counts = Counter(c.dropped_role for c in cases)
    assert dict(role_counts) == {"A": 160, "B": 160}
    assert sum(1 for c in cases if c.is_primary_unconfounded) == 160

    env_cache: dict[int, GoalpostEnv] = {}
    for c in cases:
        env = env_cache.setdefault(c.seed, GoalpostEnv(c.seed))
        # Exact C-R3 recomputation matches stored rule description 100%
        entry = env._pick_r3_entry(_state_to_tuple(c.r2_state), env.r3_entries)
        assert entry.rule.description == c.r3_rule_description
        assert c.r3_rule().description == entry.rule.description

        # Round-2 state strictly FAILS C-R3 (blind R2 copy gets 0% strict pass)
        assert c.r2_fails_r3 is True
        assert entry.rule.check(c.r2_state) is False
        r2_grade = grade_dropout_config(c, f"CONFIG: {json.dumps(c.r2_state)}")
        assert r2_grade.copied_r2_unchanged is True
        assert r2_grade.r3_passed is False
        assert r2_grade.own_plus_r3_pass is False
        assert r2_grade.strict_all_pass is False

        # At least one domain-valid 15/15 solution exists and grades 15/15 strict pass
        assert c.n_valid_solutions >= 1
        sol = _tuple_to_state(c.valid_solutions()[0])
        sol_grade = grade_dropout_config(c, f"CONFIG: {json.dumps(sol)}")
        assert sol_grade.parse_ok is True
        assert sol_grade.domain_ok is True
        assert sol_grade.own_rules_passed == 7
        assert sol_grade.r3_passed is True
        assert sol_grade.own_plus_r3_pass is True
        assert sol_grade.partner_rules_passed == 7
        assert sol_grade.partner_rules_pass is True
        assert sol_grade.strict_all_pass is True
        assert sol_grade.copied_r2_unchanged is False
        assert all(sol_grade.rule_results[rid] for rid in ALL_EXP1_RULE_IDS)


def test_select_gate_cases_stratification_and_determinism() -> None:
    """Verify deterministic 16-case gate slice: 4/seed, 8 A + 8 B, 16 distinct episodes, all 7 senders."""
    cases = build_dropout_cases(DEFAULT_EPISODES_DIR)
    gate1 = select_gate_cases(cases, n_cases=16)
    gate2 = select_gate_cases(list(reversed(cases)), n_cases=16)

    assert [c.case_id for c in gate1] == [c.case_id for c in gate2]
    assert len(gate1) == 16
    assert len({c.episode_id for c in gate1}) == 16
    assert dict(Counter(c.seed for c in gate1)) == {301: 4, 302: 4, 303: 4, 304: 4}
    assert dict(Counter(c.dropped_role for c in gate1)) == {"A": 8, "B": 8}
    assert all(
        Counter((c.seed, c.dropped_role) for c in gate1)[(s, r)] == 2
        for s in (301, 302, 303, 304)
        for r in ("A", "B")
    )
    sender_counts = Counter(c.sender_model for c in gate1)
    assert len(sender_counts) == 7
    assert min(sender_counts.values()) >= 2


# ---------------------------------------------------------------------------
# 2. Zero hidden-rule leakage on floor, exact 7-rule inclusion on ceiling, handoff framing
# ---------------------------------------------------------------------------


def test_prompt_builder_zero_leakage_and_verbatim_ceiling() -> None:
    """Verify zero partner rule leakage in floor_no_partner and exact 7/7 verbatim rules in ceiling."""
    cases = build_dropout_cases(DEFAULT_EPISODES_DIR)
    gate = select_gate_cases(cases, n_cases=16)

    # Full zero-leakage & ceiling check across ALL 320 cases (fast pure-Python)
    for c in cases:
        fl_msgs = build_dropout_prompt(c, "floor_no_partner")
        cl_msgs = build_dropout_prompt(c, "ceiling_verbatim_rules")
        assert len(fl_msgs) == 2 and len(cl_msgs) == 2
        # Identical system prompt across floor and ceiling
        assert fl_msgs[0]["content"] == cl_msgs[0]["content"]

        fl_text = "\n".join(m["content"] for m in fl_msgs)
        cl_text = "\n".join(m["content"] for m in cl_msgs)

        # Floor must NOT contain any partner rule IDs, formulas, or R1/R2 messages
        fl_disc = check_goalpost_message_disclosure(fl_text, seed=c.seed, role=c.dropped_role)
        assert fl_disc.n_rules_formula == 0, f"Floor leaked formula in {c.case_id}: {fl_disc}"
        assert fl_disc.n_rules_id == 0, f"Floor leaked rule ID in {c.case_id}: {fl_disc}"
        for rid, desc in c.partner_rules:
            assert rid not in fl_text
            assert desc not in fl_text

        # Ceiling MUST include all 7 partner rules verbatim
        cl_disc = check_goalpost_message_disclosure(cl_text, seed=c.seed, role=c.dropped_role)
        assert cl_disc.n_rules_formula == 7
        assert cl_disc.n_rules_id == 7
        for rid, desc in c.partner_rules:
            assert f"- [{rid}] {desc}" in cl_text

        # Both include own 7 rules and exact C-R3
        for rid, desc in c.own_rules:
            assert f"- [{rid}] {desc}" in fl_text
            assert f"- [{rid}] {desc}" in cl_text
        assert c.r3_rule_description in fl_text
        assert c.r3_rule_description in cl_text

    # Handoff prompt check on gate slice + system prompt prefix stability for Anthropic caching
    sys_by_seed_role: dict[tuple[int, str], set[str]] = {}
    for c in gate:
        ho_msgs = build_dropout_prompt(c, "handoff")
        sys_by_seed_role.setdefault((c.seed, c.receiver_role), set()).add(ho_msgs[0]["content"])
        ho_user = ho_msgs[1]["content"]
        assert c.round1_msg_a in ho_user
        assert c.round1_msg_b in ho_user
        assert c.round2_msg_a in ho_user
        assert c.round2_msg_b in ho_user
        assert c.r3_rule_description in ho_user
        assert "unreachable and has dropped out" in ho_user
        assert "emergency authority over all 12 config keys" in ho_user
    assert all(len(v) == 1 for v in sys_by_seed_role.values())


# ---------------------------------------------------------------------------
# 3. 3-way Parser & Grader edge cases
# ---------------------------------------------------------------------------


def test_parser_and_three_way_grader_edge_cases() -> None:
    """Verify valid config, fenced JSON, short keys normalization, out-of-domain, own-pass-partner-fail, R2 copy."""
    case = build_dropout_cases(DEFAULT_EPISODES_DIR)[0]
    sol = _tuple_to_state(case.valid_solutions()[0])

    # A. Standard CONFIG: {...}
    g_valid = grade_dropout_config(case, f"Checked all 15 rules.\nCONFIG: {json.dumps(sol)}")
    assert g_valid.parse_ok is True
    assert g_valid.domain_ok is True
    assert g_valid.normalized_keys is False
    assert g_valid.strict_all_pass is True
    assert g_valid.own_plus_r3_pass is True
    assert g_valid.own_plus_r3_frac == 1.0
    assert g_valid.partner_rules_pass is True
    assert g_valid.partner_rules_frac == 1.0
    assert g_valid.copied_r2_unchanged is False

    # B. Fenced ```json block + trailing comma + earlier scratch dict (must pick last block)
    fenced_resp = (
        f"Old state was {json.dumps(case.r2_state)}\n"
        f"Final answer:\n```json\n{json.dumps(sol)[:-1]},}}\n```"
    )
    g_fenced = grade_dropout_config(case, fenced_resp)
    assert g_fenced.parse_ok is True
    assert g_fenced.strict_all_pass is True

    # C. ACTION: {"updates": {...}} wrapper
    g_action = grade_dropout_config(case, f'ACTION: {{"updates": {json.dumps(sol)}}}')
    assert g_action.parse_ok is True
    assert g_action.strict_all_pass is True

    # D. Short key names ("workers" -> "cfg.workers") with normalized_keys=True
    short_sol = {k.removeprefix("cfg."): v for k, v in sol.items()}
    g_short = grade_dropout_config(case, f"CONFIG: {json.dumps(short_sol)}")
    assert g_short.parse_ok is True
    assert g_short.domain_ok is True
    assert g_short.normalized_keys is True
    assert g_short.strict_all_pass is True
    assert g_short.parsed_config == sol

    # E. Ambiguous duplicate short + canonical key rejected
    dup_sol = dict(sol)
    dup_sol["workers"] = 6
    p_dup = parse_config_response(f"CONFIG: {json.dumps(dup_sol)}")
    assert p_dup.parse_ok is False
    assert p_dup.parse_error is not None and "Ambiguous" in p_dup.parse_error

    # F. Non-integer (bool / float) rejected
    bad_bool = dict(sol)
    bad_bool["cfg.log_lvl"] = True  # type: ignore[assignment]
    assert parse_config_response(f"CONFIG: {json.dumps(bad_bool)}").parse_ok is False

    bad_float = dict(sol)
    bad_float["cfg.workers"] = 4.0  # type: ignore[assignment]
    assert parse_config_response(f"CONFIG: {json.dumps(bad_float)}").parse_ok is False

    # G. Out-of-domain integer key -> parse_ok=True, domain_ok=False, strict/own/partner False
    ood_cfg = dict(sol)
    ood_cfg["cfg.workers"] = 8  # allowed: (2, 4, 6)
    g_ood = grade_dropout_config(case, f"CONFIG: {json.dumps(ood_cfg)}")
    assert g_ood.parse_ok is True
    assert g_ood.domain_ok is False
    assert g_ood.out_of_domain_keys == {"cfg.workers": 8}
    assert g_ood.own_plus_r3_pass is False
    assert g_ood.partner_rules_pass is False
    assert g_ood.strict_all_pass is False

    # H. Own-pass (8/8) + Partner-fail (<7/7)
    env = GoalpostEnv(case.seed)
    own_rules = [r for r in env.base_rules if r.visible_to == case.receiver_role]
    partner_rules = [r for r in env.base_rules if r.visible_to == case.dropped_role]
    r3_rule = case.r3_rule()
    own_only_cfg: dict[str, int] | None = None
    for k in ALL_KEYS:
        for v in DOMAINS[k]:
            if v == sol[k]:
                continue
            cand = dict(sol)
            cand[k] = v
            if (
                r3_rule.check(cand)
                and all(r.check(cand) for r in own_rules)
                and not all(r.check(cand) for r in partner_rules)
            ):
                own_only_cfg = cand
                break
        if own_only_cfg is not None:
            break
    assert own_only_cfg is not None
    g_own = grade_dropout_config(case, f"CONFIG: {json.dumps(own_only_cfg)}")
    assert g_own.parse_ok is True
    assert g_own.domain_ok is True
    assert g_own.own_rules_passed == 7
    assert g_own.r3_passed is True
    assert g_own.own_plus_r3_pass is True
    assert g_own.partner_rules_pass is False
    assert g_own.partner_rules_passed < 7
    assert g_own.strict_all_pass is False

    # I. Blind R2 copy -> copied_r2_unchanged=True, strict_all_pass=False
    g_r2 = grade_dropout_config(case, f"CONFIG: {json.dumps(case.r2_state)}")
    assert g_r2.parse_ok is True
    assert g_r2.domain_ok is True
    assert g_r2.copied_r2_unchanged is True
    assert g_r2.hamming_from_r2 == 0
    assert g_r2.r3_passed is False
    assert g_r2.strict_all_pass is False


# ---------------------------------------------------------------------------
# 4. Automatic <40% worker step-up logic + frontier locking + crash-resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_step_up_gate_worker_only_and_crash_resume(tmp_path: Path) -> None:
    """Verify <40% worker step-up, max 2 step-ups cap, passing worker untouched, and frontier locked."""
    cases = build_dropout_cases(DEFAULT_EPISODES_DIR)
    gate_cases = select_gate_cases(cases, n_cases=16)

    receivers = {
        "frontier_locked": ReceiverSpec(
            key="frontier_locked",
            slug="openai/gpt-6.2-astra",
            family="openai",
            display_name="Frontier Locked",
            tier="frontier",
            provider_only=("OpenAI",),
            concurrency=16,
            default_effort="medium",
            effort_ladder=(
                EffortStep("medium", {"effort": "medium"}, 62.0),
                EffortStep("high", {"effort": "high"}, 65.0),
            ),
            aa_index_v5=62.0,
        ),
        "worker_passes_immediately": ReceiverSpec(
            key="worker_passes_immediately",
            slug="google/gemini-4-hydrogen",
            family="google",
            display_name="Strong Worker",
            tier="worker",
            provider_only=("Google AI Studio",),
            concurrency=24,
            default_effort="low",
            effort_ladder=(
                EffortStep("low", {"effort": "low"}, 27.0),
                EffortStep("medium", {"effort": "medium"}, 41.0),
                EffortStep("high", {"effort": "high"}, 51.0),
            ),
            aa_index_v5=27.0,
        ),
        "worker_steps_up_twice": ReceiverSpec(
            key="worker_steps_up_twice",
            slug="deepseek/deepseek-v4.1-flash",
            family="deepseek",
            display_name="Stepping Worker",
            tier="worker",
            provider_only=("DeepSeek",),
            concurrency=32,
            default_effort="disabled",
            effort_ladder=(
                EffortStep("disabled", {"enabled": False}, 19.0),
                EffortStep("low", {"effort": "low"}, 25.0),
                EffortStep("medium", {"effort": "medium"}, 30.0),
                EffortStep("high", {"effort": "high"}, 33.0),
            ),
            aa_index_v5=19.0,
        ),
        "worker_exhausts_cap": ReceiverSpec(
            key="worker_exhausts_cap",
            slug="openai/gpt-6-luna",
            family="openai",
            display_name="Weak Worker",
            tier="worker",
            provider_only=("OpenAI",),
            concurrency=24,
            default_effort="disabled",
            effort_ladder=(
                EffortStep("disabled", {"enabled": False}, 12.0),
                EffortStep("low", {"effort": "low"}, 17.0),
                EffortStep("medium", {"effort": "medium"}, 18.0),
                EffortStep("high", {"effort": "high"}, 28.0),
            ),
            aa_index_v5=12.0,
        ),
    }

    ceiling_rates = {
        # Frontier stays locked at medium even with 0% ceiling
        ("frontier_locked", "medium"): 0.0,
        # Strong worker passes immediately at low (100% >= 40%)
        ("worker_passes_immediately", "low"): 1.0,
        # Stepping worker fails disabled (0%) and low (0%), then passes at medium (100%)
        ("worker_steps_up_twice", "disabled"): 0.0,
        ("worker_steps_up_twice", "low"): 0.0,
        ("worker_steps_up_twice", "medium"): 1.0,
        # Weak worker fails every step (0%), must stop after max_step_ups=2 (evaluates 3 steps max)
        ("worker_exhausts_cap", "disabled"): 0.0,
        ("worker_exhausts_cap", "low"): 0.0,
        ("worker_exhausts_cap", "medium"): 0.0,
        ("worker_exhausts_cap", "high"): 1.0,
    }
    floor_rates = {k: 0.0 for k in ceiling_rates}

    out_dir = tmp_path / "gate_stepup"
    report = await run_dropout_gate(
        out_dir=out_dir,
        receivers=receivers,
        gate_cases=gate_cases,
        ceiling_threshold=0.40,
        max_step_ups=2,
        mock=True,
        mock_ceiling_pass_rates=ceiling_rates,
        mock_floor_pass_rates=floor_rates,
    )

    # 1. Frontier stays strictly locked at medium (0 step-ups, 1 history entry)
    fr = report["receivers"]["frontier_locked"]
    assert fr["locked"]["locked_effort"] == "medium"
    assert fr["locked"]["step_ups_performed"] == 0
    assert len(fr["step_up_history"]) == 1
    assert fr["step_up_history"][0]["decision"] == "locked_frontier"

    # 2. Passing worker stays untouched at low
    sw = report["receivers"]["worker_passes_immediately"]
    assert sw["locked"]["locked_effort"] == "low"
    assert sw["locked"]["step_ups_performed"] == 0
    assert sw["locked"]["gate_passed"] is True
    assert sw["locked"]["gate_ceiling_strict_pass_rate"] == 1.0
    assert len(sw["step_up_history"]) == 1

    # 3. Stepping worker steps up twice: disabled -> low -> medium and passes
    stw = report["receivers"]["worker_steps_up_twice"]
    assert stw["locked"]["initial_effort"] == "disabled"
    assert stw["locked"]["locked_effort"] == "medium"
    assert stw["locked"]["locked_aa_index_v5"] == 30.0
    assert stw["locked"]["step_ups_performed"] == 2
    assert stw["locked"]["gate_passed"] is True
    assert [h["effort"] for h in stw["step_up_history"]] == ["disabled", "low", "medium"]

    # 4. Always-failing worker respects max_step_ups=2 (never reaches 4th rung "high")
    ww = report["receivers"]["worker_exhausts_cap"]
    assert ww["locked"]["step_ups_performed"] == 2
    assert ww["locked"]["gate_passed"] is False
    assert [h["effort"] for h in ww["step_up_history"]] == ["disabled", "low", "medium"]

    # Artifacts written and parseable
    assert (out_dir / "gate_report.json").exists()
    assert (out_dir / "gate_report.md").exists()
    assert (out_dir / "locked_receiver_roster.json").exists()
    roster = load_locked_roster(out_dir / "locked_receiver_roster.json")
    assert set(roster.keys()) == set(receivers.keys())
    assert roster["worker_steps_up_twice"].locked_effort == "medium"

    # Crash-resume: re-running hits 100% ledger cache (0 new ledger lines)
    ledger_lines_before = len((out_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines())
    await run_dropout_gate(
        out_dir=out_dir,
        receivers=receivers,
        gate_cases=gate_cases,
        ceiling_threshold=0.40,
        max_step_ups=2,
        mock=True,
        mock_ceiling_pass_rates=ceiling_rates,
        mock_floor_pass_rates=floor_rates,
    )
    ledger_lines_after = len((out_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines())
    assert ledger_lines_after == ledger_lines_before


# ---------------------------------------------------------------------------
# 5. Receiver Manifest Loader (primary `receivers` & alternate `verified_pins` schema)
# ---------------------------------------------------------------------------


def test_load_receiver_manifest_both_schemas(tmp_path: Path) -> None:
    """Verify both `receivers` schema and alternate `verified_pins` schema load cleanly."""
    # Schema A: `receivers`
    schema_a = {
        "receivers": {
            "opus55": {
                "slug": "anthropic/claude-opus-5.5",
                "family": "anthropic",
                "display_name": "Claude Opus 5.5",
                "tier": "frontier",
                "provider_only": ["Anthropic"],
                "concurrency": 16,
                "supports_anthropic_cache_control": True,
                "input_per_m": 4.0,
                "output_per_m": 20.0,
                "max_tokens": 8192,
                "default_effort": "medium",
                "effort_ladder": [
                    {"effort": "medium", "reasoning_payload": {"effort": "medium"}, "aa_index_v5": 59.0}
                ],
            },
            "deepseek_flash": {
                "slug": "deepseek/deepseek-v4.1-flash",
                "family": "deepseek",
                "display_name": "DeepSeek V4.1 Flash",
                "tier": "worker",
                "provider_only": ["DeepSeek"],
                "concurrency": 32,
                "supports_anthropic_cache_control": False,
                "input_per_m": 0.14,
                "output_per_m": 0.28,
                "max_tokens": 8192,
                "default_effort": "disabled",
                "effort_ladder": [
                    {"effort": "disabled", "reasoning_payload": {"enabled": False}, "aa_index_v5": 19.0},
                    {"effort": "low", "reasoning_payload": {"effort": "low"}, "aa_index_v5": 25.0},
                ],
            },
        }
    }
    p_a = tmp_path / "manifest_a.json"
    p_a.write_text(json.dumps(schema_a), encoding="utf-8")
    rec_a = load_receiver_manifest(p_a)
    assert list(rec_a.keys()) == ["opus55", "deepseek_flash"]
    assert rec_a["opus55"].tier == "frontier"
    assert rec_a["opus55"].supports_anthropic_cache_control is True
    assert rec_a["deepseek_flash"].effort_ladder[0].reasoning_payload == {"enabled": False}

    # Schema B: `verified_pins` (matching `scripts/probe_workers_v2_1.py` output, including `None` -> omit `{}`)
    schema_b = {
        "schema_version": "v2.1",
        "receiver_order": ["astra62", "deepseek_flash"],
        "verified_pins": {
            "deepseek_flash": {
                "key": "deepseek_flash",
                "display_name": "DeepSeek V4.1 Flash",
                "family": "deepseek",
                "receiver_class": "worker",
                "slug": "deepseek/deepseek-v4.1-flash",
                "provider_only": ["DeepSeek"],
                "max_tokens": 8192,
                "supports_anthropic_cache_control": False,
                "concurrency": 32,
                "input_per_m": 0.14,
                "output_per_m": 0.28,
                "starting_tier": "disabled",
                "effort_ladder": ["disabled", "low", "medium"],
                "tier_reasoning_payloads": {
                    "disabled": None,  # explicit omit!
                    "low": {"effort": "low"},
                    "medium": {"effort": "medium"},
                },
                "aa_index_v5_by_tier": {"disabled": 19.0, "low": 24.0, "medium": 30.0},
                "reasoning_effort": None,
                "reasoning_payload": None,
                "aa_index_v5": 19.0,
            },
            "astra62": {
                "key": "astra62",
                "display_name": "GPT-6.2 Astra",
                "family": "openai",
                "receiver_class": "frontier",
                "slug": "openai/gpt-6.2-astra",
                "provider_only": ["OpenAI"],
                "max_tokens": 8192,
                "supports_anthropic_cache_control": False,
                "concurrency": 16,
                "input_per_m": 8.0,
                "output_per_m": 40.0,
                "starting_tier": "medium",
                "effort_ladder": ["medium"],
                "tier_reasoning_payloads": {"medium": {"effort": "medium"}},
                "aa_index_v5_by_tier": {"medium": 62.0},
                "reasoning_effort": "medium",
                "reasoning_payload": {"effort": "medium"},
                "aa_index_v5": 62.0,
            },
        },
    }
    p_b = tmp_path / "manifest_b.json"
    p_b.write_text(json.dumps(schema_b), encoding="utf-8")
    rec_b = load_receiver_manifest(p_b)
    assert list(rec_b.keys()) == ["astra62", "deepseek_flash"]
    assert rec_b["astra62"].tier == "frontier"
    assert rec_b["deepseek_flash"].tier == "worker"
    assert rec_b["deepseek_flash"].default_effort == "disabled"
    # Verified explicit `None` in `tier_reasoning_payloads` becomes `{}` so client omits `"reasoning"`
    assert rec_b["deepseek_flash"].effort_ladder[0].reasoning_payload == {}
    assert rec_b["deepseek_flash"].effort_ladder[1].reasoning_payload == {"effort": "low"}
    assert rec_b["deepseek_flash"].effort_ladder[2].aa_index_v5 == 30.0


# ---------------------------------------------------------------------------
# 6. OpenRouterClient backward compatibility + connection limits + reasoning payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_client_limits_and_reasoning_payloads(tmp_path: Path) -> None:
    """Verify httpx.Limits, effort='disabled'/'none'/'omit', and reasoning_payload_override without network."""
    captured_payloads: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "provider": "MockProvider",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "cost": 0.0001},
            },
        )

    ledger = Ledger(tmp_path / "client_test.jsonl", budget_cap_usd=10.0)
    client = OpenRouterClient(
        ledger,
        DEFAULT_MODELS,
        api_key="test-key-offline",
        max_connections=200,
        max_keepalive_connections=96,
    )
    try:
        assert client.max_connections == 200
        assert client.max_keepalive_connections == 96
        # Verify default httpx transport received custom Limits before swapping to MockTransport
        pool = client._http._transport._pool  # type: ignore[attr-defined]
        assert pool._max_connections == 200
        assert pool._max_keepalive_connections == 96

        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            transport=httpx.MockTransport(_handler),
        )

        msgs = [{"role": "user", "content": "hi"}]
        # 1. Standard medium (backward compatible)
        await client.complete(call_id="c1", model_key="astra62", messages=msgs)
        assert captured_payloads[-1]["reasoning"] == {"effort": "medium"}

        # 2. effort="disabled" -> {"enabled": False}
        await client.complete(
            call_id="c2",
            model_key="astra62",
            messages=msgs,
            reasoning_effort_override="disabled",
        )
        assert captured_payloads[-1]["reasoning"] == {"enabled": False}

        # 3. effort="none" -> {"effort": "none"}
        await client.complete(
            call_id="c3",
            model_key="astra62",
            messages=msgs,
            reasoning_effort_override="none",
        )
        assert captured_payloads[-1]["reasoning"] == {"effort": "none"}

        # 4. effort="omit" -> reasoning key omitted cleanly
        await client.complete(
            call_id="c4",
            model_key="astra62",
            messages=msgs,
            reasoning_effort_override="omit",
        )
        assert "reasoning" not in captured_payloads[-1]

        # 5. reasoning_payload_override dict -> passed directly
        await client.complete(
            call_id="c5",
            model_key="astra62",
            messages=msgs,
            reasoning_effort_override="low",
            reasoning_payload_override={"max_tokens": 1024},
        )
        assert captured_payloads[-1]["reasoning"] == {"max_tokens": 1024}

        # 6. reasoning_payload_override={} -> omits reasoning key cleanly even if effort="disabled"
        await client.complete(
            call_id="c6",
            model_key="astra62",
            messages=msgs,
            reasoning_effort_override="disabled",
            reasoning_payload_override={},
        )
        assert "reasoning" not in captured_payloads[-1]
    finally:
        await client.aclose()

    # 7. MockOpenRouterClient accepts identical reasoning_payload_override signature cleanly
    mock_ledger = Ledger(tmp_path / "mock_client_test.jsonl", budget_cap_usd=10.0)
    mock_client = MockOpenRouterClient(mock_ledger, DEFAULT_MODELS)
    entry = await mock_client.complete(
        call_id="mc1",
        model_key="astra62",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort_override="disabled",
        reasoning_payload_override={"enabled": False},
    )
    assert entry.status == "ok"


# ---------------------------------------------------------------------------
# 7. End-to-end CLI `--mock --gate-only` & `--mock --full` (small slice)
# ---------------------------------------------------------------------------


def test_cli_mock_gate_only_and_full_end_to_end(tmp_path: Path) -> None:
    """Run scripts/run_exp1_dropout.py with --mock --gate-only and --mock --full end-to-end."""
    cli = _load_script_module(ROOT_DIR / "scripts" / "run_exp1_dropout.py", "run_exp1_dropout_cli")
    out_dir = tmp_path / "exp1_e2e"
    missing_manifest = tmp_path / "nonexistent_probe_manifest_v2_1.json"

    # 1. --mock --gate-only
    rc_gate = cli.main(
        [
            "--mock",
            "--gate-only",
            "--out-dir",
            str(out_dir),
            "--probe-manifest",
            str(missing_manifest),
            "--budget-cap",
            "15.0",
        ]
    )
    assert rc_gate == 0
    assert (out_dir / "ledger.jsonl").exists()
    assert (out_dir / "gate_report.json").exists()
    assert (out_dir / "gate_report.md").exists()
    assert (out_dir / "locked_receiver_roster.json").exists()

    gate_report = json.loads((out_dir / "gate_report.json").read_text(encoding="utf-8"))
    assert gate_report["n_gate_cases"] == 16
    assert len(gate_report["receivers"]) == 8
    # Default mock weak worker (`deepseek_flash`) stepped up automatically from disabled -> low
    df_locked = gate_report["receivers"]["deepseek_flash"]["locked"]
    assert df_locked["initial_effort"] == "disabled"
    assert df_locked["step_ups_performed"] >= 1
    assert df_locked["locked_effort"] != "disabled"
    # All 4 frontier receivers remained locked at medium
    for fk in ("astra62", "neon4", "opus55", "astra60"):
        assert gate_report["receivers"][fk]["locked"]["locked_effort"] == "medium"
        assert gate_report["receivers"][fk]["locked"]["step_ups_performed"] == 0

    # 2. --mock --full (small 16-case slice reusing locked_receiver_roster.json)
    rc_full = cli.main(
        [
            "--mock",
            "--full",
            "--limit-cases",
            "16",
            "--out-dir",
            str(out_dir),
            "--probe-manifest",
            str(missing_manifest),
            "--budget-cap",
            "25.0",
        ]
    )
    assert rc_full == 0
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "exp1_summary.json"
    assert results_path.exists()
    assert summary_path.exists()

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 16 * 8

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["overview"]["n_cases"] == 128
    assert summary["overview"]["n_ok"] == 128
    for section in (
        "by_role",
        "by_tier",
        "by_sender",
        "by_receiver",
        "cross_vendor_vs_same_family",
        "disclosure_gradient",
        "within_astra62_shared_vs_withheld",
        "original_receiver_prodding_check",
        "intelligence_x_sender_bloat",
        "sender_x_receiver_matrix",
    ):
        assert section in summary

    # 3. Crash-resume idempotency: running --full again appends 0 rows and 0 new ledger entries
    ledger_count_before = len((out_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines())
    results_count_before = len(rows)
    rc_resume = cli.main(
        [
            "--mock",
            "--full",
            "--limit-cases",
            "16",
            "--out-dir",
            str(out_dir),
            "--probe-manifest",
            str(missing_manifest),
            "--budget-cap",
            "25.0",
        ]
    )
    assert rc_resume == 0
    assert len((out_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()) == ledger_count_before
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == results_count_before


@pytest.mark.asyncio
async def test_run_dropout_experiment_full_320_slice_summary_richness(tmp_path: Path) -> None:
    """Verify full 320-case mock summary covers all 7 senders, within-astra62 split, and bloat interaction."""
    out_dir = tmp_path / "exp1_320_mock"
    await run_dropout_gate(out_dir=out_dir, mock=True, budget_cap_usd=15.0)
    summary = await run_dropout_experiment(
        out_dir=out_dir,
        limit_cases=80,  # 80 cases spanning both Role A & B across all 7 senders -> 640 calls (<0.5s)
        mock=True,
        budget_cap_usd=25.0,
    )
    assert summary["overview"]["n_cases"] == 640
    assert len(summary["by_sender"]) == 7
    assert len(summary["by_receiver"]) == 8
    # Within-astra62 split has both shared and withheld entries
    w62 = summary["within_astra62_shared_vs_withheld"]["all_r1_r2"]
    assert w62["shared"]["n_cases"] > 0
    assert w62["withheld"]["n_cases"] > 0
    assert w62["episode_perm_p_strict"] is not None
    assert w62["delta_strict_ci95_pp"] is not None
    # Intelligence x bloat interaction has both frontier and worker tiers
    ixb = summary["intelligence_x_sender_bloat"]
    assert "frontier" in ixb["by_tier"] and "worker" in ixb["by_tier"]
    assert "frontier_minus_worker_bloat_interaction_pp" in ixb


# ---------------------------------------------------------------------------
# 8. Dual ITT vs Completed-Only Rates, Latest-Error Precedence, Order-Invariant Bootstrap
# ---------------------------------------------------------------------------


def test_dual_itt_and_completed_only_metrics_and_ok_token_means() -> None:
    """Verify dual ITT (`n_cases`) + completed-only (`n_ok`) rates, truncation counter, and ok-only token/latency means."""
    records = [
        # 3 completed ("ok") rows: 2 strict pass, 1 own-pass/partner-fail
        {
            "episode_id": "ep1",
            "case_id": "ep1__drop_A",
            "status": "ok",
            "parse_ok": True,
            "domain_ok": True,
            "own_plus_r3_pass": True,
            "partner_rules_pass": True,
            "strict_all_pass": True,
            "own_plus_r3_frac": 1.0,
            "partner_rules_frac": 1.0,
            "copied_r2_unchanged": False,
            "cost_usd": 0.10,
            "prompt_tokens": 1000,
            "completion_tokens": 300,
            "reasoning_tokens": 200,
            "latency_ms": 1000,
            "sender_r1_r2_n_formula": 7,
            "sender_r1_r2_full_disclosure": True,
            "sender_r1_r2_word_count": 120,
        },
        {
            "episode_id": "ep2",
            "case_id": "ep2__drop_A",
            "status": "ok",
            "parse_ok": True,
            "domain_ok": True,
            "own_plus_r3_pass": True,
            "partner_rules_pass": True,
            "strict_all_pass": True,
            "own_plus_r3_frac": 1.0,
            "partner_rules_frac": 1.0,
            "copied_r2_unchanged": False,
            "cost_usd": 0.20,
            "prompt_tokens": 2000,
            "completion_tokens": 600,
            "reasoning_tokens": 400,
            "latency_ms": 2000,
            "sender_r1_r2_n_formula": 6,
            "sender_r1_r2_full_disclosure": True,
            "sender_r1_r2_word_count": 150,
        },
        {
            "episode_id": "ep3",
            "case_id": "ep3__drop_A",
            "status": "ok",
            "parse_ok": True,
            "domain_ok": False,
            "own_plus_r3_pass": False,
            "partner_rules_pass": False,
            "strict_all_pass": False,
            "own_plus_r3_frac": 0.75,
            "partner_rules_frac": 0.5,
            "copied_r2_unchanged": False,
            "cost_usd": 0.30,
            "prompt_tokens": 3000,
            "completion_tokens": 900,
            "reasoning_tokens": 600,
            "latency_ms": 3000,
            "sender_r1_r2_n_formula": 0,
            "sender_r1_r2_full_disclosure": False,
            "sender_r1_r2_word_count": 60,
        },
        # 2 non-ok error rows (0-filled tokens & latency): 1 truncation + 1 transient HTTP 429
        {
            "episode_id": "ep4",
            "case_id": "ep4__drop_A",
            "status": "error",
            "error_detail": "persisted truncated: finish_reason=length",
            "finish_reason": None,
            "parse_ok": False,
            "domain_ok": False,
            "own_plus_r3_pass": False,
            "partner_rules_pass": False,
            "strict_all_pass": False,
            "own_plus_r3_frac": 0.0,
            "partner_rules_frac": 0.0,
            "copied_r2_unchanged": False,
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": None,
            "latency_ms": 0,
            "sender_r1_r2_n_formula": 0,
            "sender_r1_r2_full_disclosure": False,
            "sender_r1_r2_word_count": 40,
        },
        {
            "episode_id": "ep5",
            "case_id": "ep5__drop_A",
            "status": "error",
            "error_detail": "exhausted 8 retries: HTTP 429 rate limited",
            "finish_reason": None,
            "parse_ok": False,
            "domain_ok": False,
            "own_plus_r3_pass": False,
            "partner_rules_pass": False,
            "strict_all_pass": False,
            "own_plus_r3_frac": 0.0,
            "partner_rules_frac": 0.0,
            "copied_r2_unchanged": False,
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": None,
            "latency_ms": 0,
            "sender_r1_r2_n_formula": 0,
            "sender_r1_r2_full_disclosure": False,
            "sender_r1_r2_word_count": 50,
        },
    ]

    m = _group_metrics(records)
    assert m["n_cases"] == 5
    assert m["n_ok"] == 3
    assert m["n_errors"] == 2
    assert m["n_truncation_errors"] == 1

    # ITT rates (denominator = n_cases = 5)
    assert m["parse_ok_rate"] == pytest.approx(3 / 5, abs=1e-6)
    assert m["domain_ok_rate"] == pytest.approx(2 / 5, abs=1e-6)
    assert m["own_plus_r3_pass_rate"] == pytest.approx(2 / 5, abs=1e-6)
    assert m["partner_rules_pass_rate"] == pytest.approx(2 / 5, abs=1e-6)
    assert m["strict_all_pass_rate"] == pytest.approx(2 / 5, abs=1e-6)
    assert m["mean_own_plus_r3_frac"] == pytest.approx((1.0 + 1.0 + 0.75) / 5, abs=1e-6)
    assert m["mean_partner_rules_frac"] == pytest.approx((1.0 + 1.0 + 0.5) / 5, abs=1e-6)

    # Completed-only rates (denominator = n_ok = 3)
    assert m["parse_ok_rate_ok"] == pytest.approx(3 / 3, abs=1e-6)
    assert m["domain_ok_rate_ok"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["own_plus_r3_pass_rate_ok"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["partner_rules_pass_rate_ok"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["strict_all_pass_rate_ok"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["mean_own_plus_r3_frac_ok"] == pytest.approx((1.0 + 1.0 + 0.75) / 3, abs=1e-6)
    assert m["mean_partner_rules_frac_ok"] == pytest.approx((1.0 + 1.0 + 0.5) / 3, abs=1e-6)
    assert m["strict_all_pass_ci95"] is not None
    assert m["strict_all_pass_ci95_ok"] is not None
    assert m["strict_all_pass_ci95_ok"][0] <= m["strict_all_pass_rate_ok"] <= m["strict_all_pass_ci95_ok"][1]

    # Token & latency means strictly over ok_recs (3 rows: 1000/2000/3000 -> 2000.0, NOT 1200.0!)
    assert m["mean_prompt_tokens"] == pytest.approx(2000.0, abs=1e-2)
    assert m["mean_completion_tokens"] == pytest.approx(600.0, abs=1e-2)
    assert m["mean_reasoning_tokens"] == pytest.approx(400.0, abs=1e-2)
    assert m["mean_latency_ms"] == pytest.approx(2000.0, abs=1e-1)

    # All-error edge case: completed-only fields and token/latency means cleanly return None
    err_only = _group_metrics(records[3:])
    assert err_only["n_cases"] == 2
    assert err_only["n_ok"] == 0
    assert err_only["n_truncation_errors"] == 1
    assert err_only["strict_all_pass_rate_ok"] is None
    assert err_only["mean_prompt_tokens"] is None
    assert err_only["mean_completion_tokens"] is None
    assert err_only["mean_latency_ms"] is None
    assert err_only["strict_all_pass_ci95_ok"] is None


def test_load_existing_results_jsonl_latest_error_precedence(tmp_path: Path) -> None:
    """Verify newer error rows replace stale error rows while never overwriting an `"ok"` row."""
    p = tmp_path / "results.jsonl"
    lines = [
        # Case 1: stale Run-1 HTTP 429 followed by Run-2 finish_reason=length -> latest error wins
        {"receiver_key": "neon4", "case_id": "c1", "status": "error", "error_detail": "HTTP 429"},
        {
            "receiver_key": "neon4",
            "case_id": "c1",
            "status": "error",
            "error_detail": "persisted truncated: finish_reason=length",
        },
        # Case 2: stale HTTP 429 followed by ok -> ok wins
        {"receiver_key": "neon4", "case_id": "c2", "status": "error", "error_detail": "HTTP 429"},
        {"receiver_key": "neon4", "case_id": "c2", "status": "ok", "strict_all_pass": True},
        # Case 3: ok followed by later error -> ok is strictly preserved
        {"receiver_key": "neon4", "case_id": "c3", "status": "ok", "strict_all_pass": True},
        {
            "receiver_key": "neon4",
            "case_id": "c3",
            "status": "error",
            "error_detail": "late transient error",
        },
        # Case 4: ok v1 followed by ok v2 -> newer ok wins
        {"receiver_key": "neon4", "case_id": "c4", "status": "ok", "strict_all_pass": False},
        {"receiver_key": "neon4", "case_id": "c4", "status": "ok", "strict_all_pass": True},
    ]
    p.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")

    loaded = _load_existing_results_jsonl(p)
    assert len(loaded) == 4
    assert loaded[("neon4", "c1")]["status"] == "error"
    assert "finish_reason=length" in loaded[("neon4", "c1")]["error_detail"]
    assert loaded[("neon4", "c2")]["status"] == "ok"
    assert loaded[("neon4", "c3")]["status"] == "ok"
    assert loaded[("neon4", "c3")]["strict_all_pass"] is True
    assert loaded[("neon4", "c4")]["status"] == "ok"
    assert loaded[("neon4", "c4")]["strict_all_pass"] is True


def test_episode_cluster_bootstrap_ci_order_invariant() -> None:
    """Verify `_episode_cluster_bootstrap_ci` is strictly invariant to record ordering."""
    records = [
        {"episode_id": f"ep_{i:02d}", "strict_all_pass": (i % 3 != 0)}
        for i in range(12)
        for _ in range(2)
    ]
    baseline = _episode_cluster_bootstrap_ci(records, "strict_all_pass")
    assert baseline is not None

    # Reversed order
    assert _episode_cluster_bootstrap_ci(list(reversed(records)), "strict_all_pass") == baseline

    # 10 random permutations
    rng = random.Random(42)
    for _ in range(10):
        shuffled = list(records)
        rng.shuffle(shuffled)
        assert _episode_cluster_bootstrap_ci(shuffled, "strict_all_pass") == baseline


@pytest.mark.asyncio
async def test_refresh_exp1_summary_offline_round_trip(tmp_path: Path) -> None:
    """Verify offline summary regeneration matches live run output with zero API calls."""
    out_dir = tmp_path / "offline_refresh"
    await run_dropout_gate(out_dir=out_dir, mock=True, budget_cap_usd=15.0)
    live_summary = await run_dropout_experiment(
        out_dir=out_dir,
        limit_cases=8,
        mock=True,
        budget_cap_usd=25.0,
    )
    refreshed = refresh_exp1_summary_offline(out_dir)
    assert refreshed["overview"]["n_cases"] == live_summary["overview"]["n_cases"]
    assert refreshed["overview"]["n_ok"] == live_summary["overview"]["n_ok"]
    assert refreshed["overview"]["strict_all_pass_rate"] == live_summary["overview"]["strict_all_pass_rate"]
    assert refreshed["overview"]["strict_all_pass_ci95"] == live_summary["overview"]["strict_all_pass_ci95"]
    assert refreshed["overview"]["strict_all_pass_ci95_ok"] == live_summary["overview"]["strict_all_pass_ci95_ok"]
    assert refreshed["ledger_total_cost_usd"] == pytest.approx(
        live_summary["ledger_total_cost_usd"], abs=1e-6
    )


def test_episode_label_exact_perm_p_and_string_int_coercion_flag() -> None:
    """Verify exact episode-cluster permutation DP and string-int coercion tracking."""
    from common_ground.dropout import (
        _episode_diff_bootstrap_ci_pp,
        _episode_label_exact_perm_p,
    )

    # 3 shared episodes (all 5/5) vs 3 withheld episodes (all 0/5):
    # C(6, 3) = 20 ways; extreme two-sided outcomes = 2/20 = 0.1
    shared = [{"episode_id": f"sh_{i}", "strict_all_pass": True} for i in range(3) for _ in range(5)]
    withheld = [{"episode_id": f"wh_{i}", "strict_all_pass": False} for i in range(3) for _ in range(5)]
    p_val = _episode_label_exact_perm_p(shared, withheld, "strict_all_pass")
    assert p_val == pytest.approx(0.1, abs=1e-9)

    ci = _episode_diff_bootstrap_ci_pp(shared, withheld, "strict_all_pass")
    assert ci == [100.0, 100.0]

    # Native JSON integers vs quoted integer strings tracking
    case = build_dropout_cases(DEFAULT_EPISODES_DIR)[0]
    sol = _tuple_to_state(case.valid_solutions()[0])
    p_native = parse_config_response(f"CONFIG: {json.dumps(sol)}")
    assert p_native.parse_ok is True
    assert p_native.used_string_int_coercion is False

    quoted_sol = {k: str(v) for k, v in sol.items()}
    p_quoted = parse_config_response(f"CONFIG: {json.dumps(quoted_sol)}")
    assert p_quoted.parse_ok is True
    assert p_quoted.used_string_int_coercion is True
    assert p_quoted.config == sol
