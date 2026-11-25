"""Unit tests for Task 4: CodeHandoffEnv (length balance, gate latching, scoring, asymmetry, hygiene)."""

from __future__ import annotations

import re

import pytest

from common_ground.metrics import scan_vendor_leaks
from common_ground.runner import _replace_partner_messages_in_history
from common_ground.tasks import (
    DEFAULT_MAIN_TASKS,
    TASK_REGISTRY,
    CodeHandoffEnv,
    make_task,
)

_STYLE_HINT_RE = re.compile(
    r"\b(?:frustrat(?:ing|ed)?|be concise|be brief|be terse|be polite|be friendly|"
    r"be verbose|sound rude|use shorthand|avoid shorthand|tone of voice)\b",
    re.IGNORECASE,
)


def test_registry_and_default_main_tasks_include_code_handoff() -> None:
    assert "code_handoff" in TASK_REGISTRY
    assert TASK_REGISTRY["code_handoff"] is CodeHandoffEnv
    assert DEFAULT_MAIN_TASKS == ("triage", "calendar", "goalpost", "code_handoff")
    env = make_task("code_handoff", 1)
    assert isinstance(env, CodeHandoffEnv)
    assert env.task_name == "code_handoff"
    assert env.total_rounds == 6


def test_eliminate_longest_option_length_cue() -> None:
    """Hard check: uniquely-longest (or first-longest) option is winner in <= 6/16 slots (and not shortest bias either)."""
    longest_is_winner = 0
    shortest_is_winner = 0
    for seed in (1, 2, 3, 4):
        ar = CodeHandoffEnv(seed=seed).archetype
        for slot in ar.slots:
            by_desc_len = sorted(slot.options, key=lambda o: len(o.description), reverse=True)
            by_asc_len = sorted(slot.options, key=lambda o: len(o.description))
            if by_desc_len[0].kind == "winner":
                longest_is_winner += 1
            if by_asc_len[0].kind == "winner":
                shortest_is_winner += 1
    assert longest_is_winner <= 6, f"Winner is longest in {longest_is_winner}/16 slots (> 6/16 cue!)"
    assert shortest_is_winner <= 6, f"Winner is shortest in {shortest_is_winner}/16 slots (> 6/16 cue!)"


def test_four_distinct_archetypes_per_seed_and_latin_square_winners() -> None:
    ids_1_4 = [CodeHandoffEnv(seed=s).archetype.archetype_id for s in (1, 2, 3, 4)]
    ids_201_204 = [CodeHandoffEnv(seed=s).archetype.archetype_id for s in (201, 202, 203, 204)]
    assert ids_1_4 == [
        "settlement_ledger_pr",
        "auth_oidc_rotation_pr",
        "search_index_sharding_pr",
        "webhook_billing_outbox_pr",
    ]
    assert ids_201_204 == ids_1_4

    for s_low, s_high in zip((1, 2, 3, 4), (201, 202, 203, 204), strict=True):
        a_low = CodeHandoffEnv(seed=s_low).archetype
        a_high = CodeHandoffEnv(seed=s_high).archetype
        assert a_low.pr_id != a_high.pr_id
        assert a_low.ticket_id != a_high.ticket_id
        assert a_low.service_name != a_high.service_name

    win_matrix: list[list[int]] = []
    shock_indices: list[int] = []
    for seed in (1, 2, 3, 4):
        ar = CodeHandoffEnv(seed=seed).archetype
        row: list[int] = []
        for idx, slot in enumerate(ar.slots):
            kinds = sorted(o.kind for o in slot.options)
            assert kinds == ["partial", "trap", "unfixed", "winner"]
            win_opt = slot.option_map()[slot.winner_id()]
            assert win_opt.pass_evidence, f"{ar.archetype_id}/{slot.slot_key} missing pass_evidence"
            win_num = int(slot.winner_id().split("-")[-1])
            row.append(win_num)
            if slot.slot_key == ar.shock_slot:
                shock_indices.append(idx)
                assert win_opt.pre_shock_failure_a and win_opt.pre_shock_failure_b
        assert sorted(row) == [1, 2, 3, 4], f"Archetype {ar.archetype_id} winners not varied: {row}"
        win_matrix.append(row)

    for col_idx in range(4):
        col = sorted(win_matrix[r][col_idx] for r in range(4))
        assert col == [1, 2, 3, 4], f"Slot col {col_idx} winners not balanced: {col}"
    assert sorted(shock_indices) == [0, 1, 2, 3]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 201, 202, 203, 204])
def test_asymmetry_baseline_fixture_tagging_no_leaks_no_style_hints(seed: int) -> None:
    env = CodeHandoffEnv(seed=seed)
    ar = env.archetype
    ba = env.system_briefing("A", "AL", "BK")
    bb = env.system_briefing("B", "BK", "AL")

    assert 2500 <= len(ba) < 6000, f"{ar.archetype_id} Role A briefing len={len(ba)}"
    assert 2500 <= len(bb) < 8000, f"{ar.archetype_id} Role B briefing len={len(bb)}"

    assert "VERIFICATION SCOPE:" in ba
    assert "VERIFICATION SCOPE:" in bb

    # Strict information asymmetry
    assert "CHECK-1" in ba and "CHECK-4" in ba
    assert "INV-1" not in ba and "INV-4" not in ba
    assert "INV-1" in bb and "INV-4" in bb
    assert "CHECK-1" not in bb and "CHECK-4" not in bb
    for slot in ar.slots:
        for opt in slot.options:
            assert opt.option_id not in ba, f"Option {opt.option_id} leaked into Role A briefing"
            assert opt.option_id in bb, f"Option {opt.option_id} missing from Role B briefing"

    # Static CHECK evidence leak fix: all CHECK outputs tagged [REPO BASELINE FIXTURE]
    for cid, (_, pre_out, post_out, _) in ar.checks_a.items():
        assert pre_out.startswith("[REPO BASELINE FIXTURE]"), f"{ar.archetype_id}/{cid} pre missing fixture tag"
        assert post_out.startswith("[REPO BASELINE FIXTURE]"), f"{ar.archetype_id}/{cid} post missing fixture tag"

    all_texts = [ba, bb, ar.shock_alert_a]
    for cname, pre_out, post_out, _ in ar.checks_a.values():
        all_texts.extend([cname, pre_out, post_out])
    for slot in ar.slots:
        for opt in slot.options:
            all_texts.extend(
                [
                    opt.description,
                    opt.failure_feedback_a,
                    opt.failure_feedback_b,
                    opt.pass_evidence,
                    opt.pre_shock_failure_a,
                    opt.pre_shock_failure_b,
                ]
            )
    for r in range(1, env.total_rounds + 1):
        all_texts.append(env.turn_prompt("A", r, "BK", "partner msg", "fb"))
        all_texts.append(env.turn_prompt("B", r, "AL", "partner msg", "fb"))

    for txt in all_texts:
        assert scan_vendor_leaks(txt) == [], f"Vendor leak in {ar.archetype_id}: {scan_vendor_leaks(txt)}"
        assert _STYLE_HINT_RE.search(txt) is None, f"Style hint found in {ar.archetype_id}: {txt[:200]}"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 201, 202, 203, 204])
def test_oracle_gate_latching_prudent_ship_and_no_partial_ship_bonus(seed: int) -> None:
    # 1. Oracle: R1 4/4 pre-soak (+0.20) + R4 1-shot swap & SHIP (+0.20) + 4/4 final (+0.60) = 1.000
    env = CodeHandoffEnv(seed=seed)
    for r in range(1, env.total_rounds + 1):
        p_a = env.turn_prompt("A", r, "BK", None if r == 1 else "ok", None)
        p_b = env.turn_prompt("B", r, "AL", "findings", None)
        if r == 4:
            assert "STAGING SOAK UPDATE" in p_a
            assert "STAGING SOAK UPDATE" not in p_b
        act_a, act_b = env.oracle_actions(r)
        fb = env.step_round(r, act_a, act_b, "root cause details", "staging patches")
        if r == 1:
            assert fb.metrics["passing_count"] == 4
            assert "ALL 4/4 SLOTS PASSED PRE-SOAK CI" in fb.feedback_b
        if r == 4:
            assert fb.metrics["passing_count"] == 4
            assert "RELEASE SHIPPED GREEN" in fb.feedback_b

    res = env.final_score()
    assert res.score == 1.0
    assert res.details["pre_soak_ready_round"] == 1
    assert res.details["ci_ready_round"] == 4
    assert res.details["released"] is True
    assert res.details["released_round"] == 4
    assert res.details["approval_lag"] == 0
    assert res.details["distinct_traps_triggered"] == []
    assert res.details["distinct_wrong_options"] == []

    # 2. Gate latching: once released cleanly in R4, later HOLD / parse glitch / trap CANNOT un-ship it!
    env_latch = CodeHandoffEnv(seed=seed)
    ar = env_latch.archetype
    slot_map = ar.slot_map()
    winners = {s.slot_key: s.winner_id() for s in ar.slots}
    pre_soak = dict(winners)
    pre_soak[ar.shock_slot] = slot_map[ar.shock_slot].partial_ids()[0]
    env_latch.step_round(1, {"check": "CHECK-1"}, {"patches": pre_soak, "gate": "HOLD"}, "a", "b")
    env_latch.step_round(2, {"check": "CHECK-2"}, {"patches": {}, "gate": "HOLD"}, "a", "b")
    env_latch.step_round(3, {"check": "CHECK-3"}, {"patches": {}, "gate": "HOLD"}, "a", "b")
    env_latch.step_round(4, {"check": "CHECK-4"}, {"patches": winners, "gate": "SHIP_IF_GREEN"}, "a", "b")
    assert env_latch.released is True
    # R5 parse failure (None) + R6 explicit HOLD + trap attempt: release MUST remain latched!
    env_latch.step_round(5, {"check": None}, None, "a", "b")
    trap_sk = env_latch.slot_keys[0]
    env_latch.step_round(
        6, {"check": None}, {"patches": {trap_sk: slot_map[trap_sk].trap_ids()[0]}, "gate": "HOLD"}, "a", "b"
    )
    res_l = env_latch.final_score()
    assert res_l.score == 1.0
    assert res_l.details["released"] is True
    assert res_l.details["released_round"] == 4
    assert res_l.details["final_gate"] == "SHIP"

    # 3. Prudent R4 verify (HOLD, ci_ready_round=4) then R5 SHIP (released_round=5, approval_lag=1)
    #    earns FULL +0.20 post-soak velocity! (R1 pre-soak 0.20 + 0.60 + 0.20 = 1.00)
    env_prud = CodeHandoffEnv(seed=seed)
    env_prud.step_round(1, {"check": "CHECK-1"}, {"patches": pre_soak, "gate": "HOLD"}, "a", "b")
    env_prud.step_round(2, {"check": "CHECK-2"}, {"patches": {}, "gate": "HOLD"}, "a", "b")
    env_prud.step_round(3, {"check": "CHECK-3"}, {"patches": {}, "gate": "HOLD"}, "a", "b")
    env_prud.step_round(4, {"check": "CHECK-4"}, {"patches": winners, "gate": "HOLD"}, "a", "b")
    assert env_prud.ci_ready_round == 4 and env_prud.released is False
    env_prud.step_round(5, {"check": None}, {"patches": {}, "gate": "SHIP"}, "a", "b")
    env_prud.step_round(6, {"check": None}, {"patches": {}, "gate": "HOLD"}, "a", "b")
    res_p = env_prud.final_score()
    assert res_p.score == 1.0
    assert res_p.details["ci_ready_round"] == 4
    assert res_p.details["released_round"] == 5
    assert res_p.details["approval_lag"] == 1

    # 4. Staging eventual post-soak winner during R1-R3 is NOT penalized as a wrong_option!
    env_pre_w = CodeHandoffEnv(seed=seed)
    non_shock_3 = {k: winners[k] for k in env_pre_w.slot_keys if k != ar.shock_slot}
    env_pre_w.step_round(1, {"check": None}, {"patches": non_shock_3, "gate": "HOLD"}, "a", "b")
    env_pre_w.step_round(2, {"check": None}, {"patches": {ar.shock_slot: winners[ar.shock_slot]}, "gate": "HOLD"}, "a", "b")
    env_pre_w.step_round(3, {"check": None}, {"patches": {}, "gate": "HOLD"}, "a", "b")
    env_pre_w.step_round(4, {"check": None}, {"patches": {}, "gate": "SHIP_IF_GREEN"}, "a", "b")
    env_pre_w.step_round(5, {"check": None}, {"patches": {}, "gate": "HOLD"}, "a", "b")
    env_pre_w.step_round(6, {"check": None}, {"patches": {}, "gate": "HOLD"}, "a", "b")
    res_pw = env_pre_w.final_score()
    assert res_pw.details["distinct_wrong_options"] == []
    # Missed pre-soak (0.00) + 4/4 final (0.60) + R4 ci_ready/released (0.20) = 0.80
    assert res_pw.score == pytest.approx(0.80, abs=1e-6)

    # 5. Partial-ship bonus (+0.04) REMOVED: shipping with 3/4 earns 0 ship credit (HOLD on 3/4 == SHIP on 3/4 = 0.45)
    env_p3_ship = CodeHandoffEnv(seed=seed)
    env_p3_hold = CodeHandoffEnv(seed=seed)
    for r in range(1, 7):
        env_p3_ship.step_round(r, {"check": None}, {"patches": non_shock_3 if r == 4 else {}, "gate": "SHIP" if r >= 4 else "HOLD"}, "a", "b")
        env_p3_hold.step_round(r, {"check": None}, {"patches": non_shock_3 if r == 4 else {}, "gate": "HOLD"}, "a", "b")
    assert env_p3_ship.final_score().score == pytest.approx(0.45, abs=1e-6)
    assert env_p3_hold.final_score().score == pytest.approx(0.45, abs=1e-6)
    assert env_p3_ship.final_score().details["released"] is False

    # 6. Silent / HOLD baseline -> exact 0.0
    env_s = CodeHandoffEnv(seed=seed)
    for r in range(1, env_s.total_rounds + 1):
        env_s.step_round(r, {"check": None}, {"patches": {}, "gate": "HOLD"}, "", "")
    assert env_s.final_score().score == 0.0


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_aggregate_ci_no_spoonfeeding_anti_deadlock_and_penalties(seed: int) -> None:
    env = CodeHandoffEnv(seed=seed)
    ar = env.archetype
    slot_map = ar.slot_map()
    shock_key = ar.shock_slot
    shock_partial = slot_map[shock_key].partial_ids()[0]
    shock_winner = slot_map[shock_key].winner_id()
    non_shock_keys = [k for k in env.slot_keys if k != shock_key]
    trap_slot_key = non_shock_keys[0]
    trap_opt_id = slot_map[trap_slot_key].trap_ids()[0]
    unfixed_opt_id = next(o.option_id for o in slot_map[non_shock_keys[1]].options if o.kind == "unfixed")

    # Round 1: 2 passing + 1 failing non-trap (unfixed) + 1 unstaged -> aggregate CI ONLY!
    fb1 = env.step_round(
        1,
        {"check": None},
        {
            "patches": {
                shock_key: shock_partial,
                non_shock_keys[2]: slot_map[non_shock_keys[2]].winner_id(),
                non_shock_keys[1]: unfixed_opt_id,
            },
            "gate": "HOLD",
        },
        "a1",
        "b1",
    )
    assert fb1.metrics["passing_count"] == 2
    assert fb1.metrics["non_trap_fail_count"] == 1
    assert "Verified passing staging telemetry" not in fb1.feedback_a
    assert non_shock_keys[1] not in fb1.feedback_a
    assert unfixed_opt_id not in fb1.feedback_a
    fb_b_without_staged = "\n".join(fb1.feedback_b.splitlines()[1:])
    assert non_shock_keys[1] not in fb_b_without_staged
    assert unfixed_opt_id not in fb_b_without_staged

    # Round 2: Trigger trap -> immediate CRITICAL alert, not double-counted in wrong_options
    fb2 = env.step_round(2, {"check": "CHECK-1"}, {"patches": {trap_slot_key: trap_opt_id}, "gate": "HOLD"}, "a2", "b2")
    assert fb2.metrics["active_trap_count"] == 1
    assert "CRITICAL invariant violation(s):" in fb2.feedback_b
    assert env.distinct_wrong_options == {(non_shock_keys[1], unfixed_opt_id)}

    # Round 3: Fix both -> 4/4 pre-soak (+0.08)
    fb3 = env.step_round(
        3,
        {"check": "CHECK-2"},
        {
            "patches": {
                trap_slot_key: slot_map[trap_slot_key].winner_id(),
                non_shock_keys[1]: slot_map[non_shock_keys[1]].winner_id(),
            },
            "gate": "HOLD",
        },
        "a3",
        "b3",
    )
    assert fb3.metrics["passing_count"] == 4
    assert env.pre_soak_ready_round == 3

    # Round 4: Soak flips shock_partial -> 3/4, no penalty for pre-soak carryover
    fb4 = env.step_round(4, {"check": "CHECK-4"}, {"patches": {}, "gate": "HOLD"}, "a4", "b4")
    assert fb4.metrics["passing_count"] == 3
    assert (shock_key, shock_partial) not in env.distinct_wrong_options

    # Round 5: Switch shock_slot -> winner with HOLD -> ci_ready_round=5 (+0.15 once shipped in R6!)
    fb5 = env.step_round(5, {"check": None}, {"patches": {shock_key: shock_winner}, "gate": "HOLD"}, "a5", "b5")
    assert fb5.metrics["passing_count"] == 4
    assert "VERIFICATION COMPLETE" in fb5.feedback_b

    # Round 6: SHIP -> latches release! Score = 0.60 + 0.08 (R3 pre) + 0.15 (R5 ci_ready) - 0.08 (trap) - 0.04 (wrong) = 0.71
    fb6 = env.step_round(6, {"check": None}, {"patches": {}, "gate": "SHIP"}, "a6", "b6")
    assert "RELEASE SHIPPED GREEN" in fb6.feedback_b
    res = env.final_score()
    assert res.score == pytest.approx(0.71, abs=1e-6)
    assert res.details["ci_ready_round"] == 5
    assert res.details["released_round"] == 6
    assert res.details["approval_lag"] == 1


def test_malformed_actions_and_runner_rewrite_history_compat() -> None:
    env = CodeHandoffEnv(seed=201)
    fb = env.step_round(1, {"updates": {"cfg.workers": 4}}, {"updates": {"cfg.batch_max": 32}}, "m1", "m2")
    assert fb.metrics["staged_count"] == 0
    assert fb.metrics["gate"] == "HOLD"

    hist_b = [
        {"role": "system", "content": env.system_briefing("B", "BK", "AL")},
        {"role": "user", "content": env.turn_prompt("B", 1, "AL", "ORIG_R1_MSG", None)},
        {"role": "assistant", "content": 'Holding.\n\nACTION: {"patches": {}, "gate": "HOLD"}'},
        {"role": "user", "content": env.turn_prompt("B", 6, "AL", "ORIG_R6_MSG", "fb")},
    ]
    rewritten = _replace_partner_messages_in_history(
        hist_b, role="B", partner_handle="AL", rewritten_by_round={1: "CLEAN_R1_MSG", 6: "CLEAN_R6_MSG"}
    )
    assert "CLEAN_R1_MSG" in rewritten[1]["content"] and "ORIG_R1_MSG" not in rewritten[1]["content"]
    assert "CLEAN_R6_MSG" in rewritten[3]["content"] and "ORIG_R6_MSG" not in rewritten[3]["content"]
    assert "FINAL ROUND:" in rewritten[3]["content"]
