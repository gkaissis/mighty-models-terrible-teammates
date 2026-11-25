"""Unit tests for Triage (6-round v2), Calendar v2, Goalpost v2, Ledger environments & wire parser."""

from __future__ import annotations

import itertools
import re

import pytest

from common_ground.metrics import scan_vendor_leaks
from common_ground.rewriter import extract_invariants
from common_ground.tasks import (
    BLIND_HANDLES,
    CalendarEnv,
    GoalpostEnv,
    LedgerEnv,
    TriageEnv,
    make_blind_handles,
    parse_turn_response,
)
from common_ground.tasks.goalpost_task import (
    ALL_KEYS,
    DOMAINS,
    KEYS_A,
    KEYS_B,
    MAX_EDITS_PER_ROUND,
    _closest_target,
    _hamming_a,
    _hamming_b,
    _state_to_tuple,
    _tuple_to_state,
)


def test_parse_turn_response_variants_and_safety() -> None:
    msg, act, err = parse_turn_response(
        'Let us check REC-07 and set workers.\n\nACTION: {"query_id": "Q-1", "answer": "REC-07"}'
    )
    assert err is None
    assert msg == "Let us check REC-07 and set workers."
    assert act == {"query_id": "Q-1", "answer": "REC-07"}

    msg2, act2, err2 = parse_turn_response(
        'Notes about ACTION: we should act fast.\n\nACTION:\n```json\n{"schedule": {"MTG-1": "MON-1"}}\n```'
    )
    assert err2 is None
    assert "Notes about ACTION:" in msg2
    assert act2 == {"schedule": {"MTG-1": "MON-1"}}

    msg3, act3, err3 = parse_turn_response("Just prose with no structured action.")
    assert act3 is None
    assert err3 is not None
    assert "Just prose" in msg3

    # Wire safety 1: strip duplicate earlier line-start ACTION: {...} from visible prose!
    msg4, act4, err4 = parse_turn_response(
        'Draft check:\nACTION: {"command": "WAIT"}\nActually let us proceed.\n\nACTION: {"command": "CMD-3"}'
    )
    assert err4 is None
    assert act4 == {"command": "CMD-3"}
    assert '{"command": "WAIT"}' not in msg4
    assert "Actually let us proceed." in msg4

    # Wire safety 2: reject unexpected trailing prose after ACTION JSON (forces clean retry)
    _, act5, err5 = parse_turn_response(
        'Sending update.\n\nACTION: {"command": "WAIT"}\nOops extra text after JSON!'
    )
    assert act5 is None
    assert err5 is not None and "trailing" in err5.lower()


def test_blind_handles_task_disjoint_and_deterministic() -> None:
    h1, h2 = make_blind_handles("v2:goalpost:301")
    assert h1 != h2
    assert h1 in BLIND_HANDLES and h2 in BLIND_HANDLES
    assert (h1, h2) == make_blind_handles("v2:goalpost:301")
    assert BLIND_HANDLES == ("AL", "BK", "CT", "DM", "EN", "FP", "GQ", "HR", "JT", "KV")


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 201, 202, 203, 204, 205, 206])
def test_triage_6round_oracle_trap_and_silent_floor(seed: int) -> None:
    env = TriageEnv(seed=seed)
    sc = env.scenario
    assert env.total_rounds == 6
    assert sc.twist_round == 3

    ba = env.system_briefing("A", "AL", "BK")
    bb = env.system_briefing("B", "BK", "AL")
    assert "PROBE-1" in ba and "CMD-1" not in ba
    assert "CMD-1" in bb and "G-1" in bb and "PROBE-1" not in bb
    assert "6 rounds" in ba and "8 rounds" not in ba and "10 rounds" not in ba
    assert "6 rounds" in bb and "8 rounds" not in bb and "10 rounds" not in bb
    assert "Round 5" in ba and "Round 5" in bb

    for forbidden in ("jargon", "unexplained", "cryptic", "shorthand"):
        assert forbidden not in ba.lower()
        assert forbidden not in bb.lower()
    assert scan_vendor_leaks(ba) == []
    assert scan_vendor_leaks(bb) == []
    for _, pout in sc.probes.values():
        assert scan_vendor_leaks(pout) == []
    assert scan_vendor_leaks(sc.twist_alert_a) == []
    assert scan_vendor_leaks(sc.twist_probe_output) == []

    assert list(sc.probes.keys()) == [f"PROBE-{i}" for i in range(1, 7)]
    assert list(sc.commands_b.keys()) == [f"CMD-{i}" for i in range(1, 8)]
    assert [gid for gid, _ in sc.guardrails_b] == ["G-1", "G-2", "G-3"]
    assert len(sc.trap_cmds) == 3
    assert {gid for gid, _ in sc.trap_cmds.values()} == {"G-1", "G-2", "G-3"}
    assert len(sc.red_herring_cmds) == 2
    all_classified = {sc.step1_cmd, sc.step2_cmd} | set(sc.trap_cmds) | set(sc.red_herring_cmds)
    assert all_classified == set(sc.commands_b)

    for r in range(1, env.total_rounds + 1):
        prompt_a = env.turn_prompt("A", r, "BK", None, None)
        if r == sc.twist_round:
            assert sc.twist_alert_a in prompt_a  # step2 runs AT round 3, not before
        act_a, act_b = env.oracle_actions(r)
        if r >= 5:
            assert act_a == {"probe": None} and act_b == {"command": "WAIT"}
        fb = env.step_round(r, act_a, act_b, "investigating", "applying")
        if r == 3:
            # PROBE-4 reports ONLY what PROBE-4 measures (not stale PROBE-1 text or unrelated probe)
            assert sc.probes["PROBE-4"][1] in fb.feedback_a
        if r == 4:
            assert "LIVE POST-STEP-2 CHECK" in fb.feedback_a

    res = env.final_score()
    assert res.score == 1.0
    assert res.details["step1_done"] is True
    assert res.details["step2_done"] is True
    assert res.details["verified_green"] is True
    assert res.details["first_verified_round"] == 4
    assert res.details["final_verified_round"] == 4
    assert res.details["verified_green_round"] == 4
    assert res.details["mttr_penalty"] == 0.0

    # Silent / WAIT baseline scores 0.0
    env_s = TriageEnv(seed=seed)
    for r in range(1, env_s.total_rounds + 1):
        env_s.step_round(r, {"probe": None}, {"command": "WAIT"}, "", "")
    assert env_s.final_score().score == 0.0


def test_triage_telemetry_consistency_sticky_verify_and_grpc_physics() -> None:
    # 1. Telemetry contradiction fix: after Step 1 succeeds, PROBE-1 must NOT append stale pre-Step-1 text,
    #    and PROBE-3 must ONLY report PROBE-3's own measurement (never leaking twist_probe_output)!
    env1 = TriageEnv(seed=1)  # k8s_mtls_ingress_rotation
    env1.step_round(1, {"probe": "PROBE-1"}, {"command": env1.scenario.step1_cmd}, "a", "b")
    fb_p1 = env1.step_round(2, {"probe": "PROBE-1"}, {"command": "WAIT"}, "a", "b")
    assert "LIVE POST-STEP-1 CHECK" in fb_p1.feedback_a
    assert "currently mounts ONLY `CA-v1`" not in fb_p1.feedback_a
    fb_p3 = env1.step_round(3, {"probe": "PROBE-3"}, {"command": "WAIT"}, "a", "b")
    assert "LIVE SUBSYSTEM CHECK" in fb_p3.feedback_a
    assert "unknown-CA rejection" not in fb_p3.feedback_a
    assert env1.scenario.twist_probe_output not in fb_p3.feedback_a

    # 2. Sticky VERIFY fix: post-verification trap clears verified_green until a fresh VERIFY runs!
    env2 = TriageEnv(seed=2)
    sc2 = env2.scenario
    trap_cmd = next(iter(sc2.trap_cmds))
    env2.step_round(1, {"probe": "PROBE-1"}, {"command": sc2.step1_cmd}, "a", "b")
    env2.step_round(2, {"probe": "PROBE-4"}, {"command": sc2.step2_cmd}, "a", "b")
    env2.step_round(3, {"probe": None}, {"command": "VERIFY"}, "a", "b")
    assert env2.verified_green is True
    assert env2.first_verified_round == 3
    assert env2.final_verified_round == 3

    # Trigger trap on Round 4 -> clears verified_green!
    env2.step_round(4, {"probe": None}, {"command": trap_cmd}, "a", "b")
    assert env2.verified_green is False
    assert env2.final_verified_round is None
    assert env2.first_verified_round == 3  # preserved!

    # Rollback on Round 5 without re-VERIFY -> STILL not verified_green!
    env2.step_round(5, {"probe": None}, {"command": "ROLLBACK"}, "a", "b")
    assert env2.verified_green is False

    # Re-VERIFY cleanly on Round 6 -> verified_green=True, final_verified_round=6, MTTR=0.10, trap=-0.20 -> 0.70!
    env2.step_round(6, {"probe": None}, {"command": "VERIFY"}, "a", "b")
    res2 = env2.final_score()
    assert res2.details["verified_green"] is True
    assert res2.details["first_verified_round"] == 3
    assert res2.details["final_verified_round"] == 6
    assert res2.details["verified_green_round"] == 6
    assert res2.details["verified_green_by_round"] == [False, False, True, False, False, True]
    assert res2.score == pytest.approx(0.70, abs=1e-6)

    # 2b. Strictly subsystem-scoped probe rendering:
    #   - After Step 1 only, PROBE-3 on k8s_mtls must NOT report stale unknown-CA rejection
    #   - After BOTH Step 1 & Step 2, PROBE-5 (compliance audit) must NOT report global service recovery!
    env3 = TriageEnv(seed=1)  # k8s_mtls_ingress_rotation
    sc3 = env3.scenario
    env3.step_round(1, {"probe": "PROBE-1"}, {"command": sc3.step1_cmd}, "a", "b")
    fb_p3 = env3.step_round(2, {"probe": "PROBE-3"}, {"command": sc3.step2_cmd}, "a", "b")
    assert "unknown-ca rejection" not in fb_p3.feedback_a.lower()
    fb_p5 = env3.step_round(3, {"probe": "PROBE-5"}, {"command": "VERIFY"}, "a", "b")
    assert "both primary and secondary fixes are live" not in fb_p5.feedback_a.lower()
    assert "pci transport policy" in fb_p5.feedback_a.lower()

    #   - On cdn_signed_url_s3_cors (seed 203), PROBE-5 after Step 1+2 stays scoped to SOC2/CORS audit!
    env_cdn = TriageEnv(seed=203)
    sc_cdn = env_cdn.scenario
    env_cdn.step_round(1, {"probe": "PROBE-1"}, {"command": sc_cdn.step1_cmd}, "a", "b")
    env_cdn.step_round(2, {"probe": "PROBE-4"}, {"command": sc_cdn.step2_cmd}, "a", "b")
    fb_cdn_p5 = env_cdn.step_round(3, {"probe": "PROBE-5"}, {"command": "VERIFY"}, "a", "b")
    assert "both primary and secondary fixes are live" not in fb_cdn_p5.feedback_a.lower()
    assert "soc2" in fb_cdn_p5.feedback_a.lower()

    # 3. gRPC physics fix (seed 6 = grpc_mesh_breaker_flag)
    sc6 = TriageEnv(seed=6).scenario
    assert sc6.archetype_id == "grpc_mesh_breaker_flag"
    assert "simultaneously" not in sc6.probes["PROBE-1"][1].lower()
    assert "per-endpoint" in sc6.probes["PROBE-1"][1].lower()
    assert "up to 100%" in sc6.probes["PROBE-1"][1].lower()
    assert "28% of all checkout RPCs" in sc6.initial_alerts
    assert "28% of all checkout RPCs" in sc6.probes["PROBE-4"][1]


def test_triage_guardrails_do_not_trivially_negate_trap_command_text() -> None:
    """Static/blind check confirming guardrails state operational invariants instead of copy-pasting trap CLI flags."""
    stop = {
        "the", "a", "an", "to", "on", "of", "in", "for", "and", "or", "with", "is", "are",
        "by", "that", "from", "all", "as", "be", "not", "no", "never", "while", "still", "any",
    }
    for s in range(1, 7):
        sc = TriageEnv(s).scenario
        gmap = dict(sc.guardrails_b)
        for cid, (gid, _) in sc.trap_cmds.items():
            gtxt = gmap[gid]
            ctxt = sc.commands_b[cid]
            bt_g = set(re.findall(r"`([^`]+)`", gtxt))
            bt_c = set(re.findall(r"`([^`]+)`", ctxt))
            assert bt_g.isdisjoint(bt_c), (
                f"{sc.archetype_id} {gid}/{cid} shares backtick CLI token: {bt_g & bt_c}"
            )
            wt_g = {w for w in re.findall(r"[a-z0-9_-]{3,}", gtxt.lower()) if w not in stop}
            wt_c = {w for w in re.findall(r"[a-z0-9_-]{3,}", ctxt.lower()) if w not in stop}
            jacc = len(wt_g & wt_c) / max(1, len(wt_g | wt_c))
            assert jacc < 0.25, f"{sc.archetype_id} {gid}/{cid} lexical overlap too high: {jacc:.3f}"


@pytest.mark.parametrize(
    ("verify_round", "expected_score", "expected_penalty"),
    [
        (3, 1.00, 0.00),
        (4, 1.00, 0.00),
        (5, 1.00, 0.00),
        (6, 0.90, 0.10),
    ],
)
def test_triage_6round_mttr_scoring_curve(
    verify_round: int, expected_score: float, expected_penalty: float
) -> None:
    env = TriageEnv(seed=201)
    sc = env.scenario
    for r in range(1, env.total_rounds + 1):
        if r == 1:
            cmd = sc.step1_cmd
        elif r == 2:
            cmd = sc.step2_cmd
        elif r == verify_round:
            cmd = "VERIFY"
        else:
            cmd = "WAIT"
        env.step_round(r, {"probe": None}, {"command": cmd}, "a", "b")
    res = env.final_score()
    assert res.score == expected_score
    assert res.details["verified_green_round"] == verify_round
    assert res.details["mttr_penalty"] == expected_penalty


def test_triage_six_archetypes_variety_and_guardrail_answerability() -> None:
    envs_1_6 = [TriageEnv(seed=s) for s in range(1, 7)]
    envs_201_206 = [TriageEnv(seed=s) for s in range(201, 207)]

    ids_1_6 = [e.scenario.archetype_id for e in envs_1_6]
    ids_201_206 = [e.scenario.archetype_id for e in envs_201_206]
    assert len(set(ids_1_6)) == 6
    assert set(ids_1_6) == set(ids_201_206)

    step1_cmds = [e.scenario.step1_cmd for e in envs_1_6]
    step2_cmds = [e.scenario.step2_cmd for e in envs_1_6]
    assert len(set(step1_cmds)) == 6
    assert len(set(step2_cmds)) == 6

    expected_evidence_by_archetype: dict[str, dict[str, list[str]]] = {
        "k8s_mtls_ingress_rotation": {
            "G-1": ["STRICT", "PERMISSIVE"],
            "G-2": ["CA-v1", "35%", "7/20", "CA-v2"],
            "G-3": ["wildcard", "spiffe://mesh-v1", "spiffe://mesh-v2"],
        },
        "postgres_readonly_autovacuum": {
            "G-1": ["VACUUM FULL", "ACCESS EXCLUSIVE", "SHARE UPDATE EXCLUSIVE", "0 competing DDL"],
            "G-2": ["wal_lag_bytes", "pg-node-a", "pg-node-b", "pg-node-s"],
            "G-3": ["synchronous_commit = on", "18M rows"],
        },
        "kafka_rebalance_quota_storm": {
            "G-1": ["140,000", "latest"],
            "G-2": ["quota_enforcement = on", "realtime-payments", "80 MB/s"],
            "G-3": ["enable.auto.commit = false", "acks = all"],
        },
        "redis_eviction_ratelimit": {
            "G-1": ["FLUSHALL", "cart:*", "idem:*", "report:tmp:*"],
            "G-2": ["ratelimit_mode = off", "100 req/s", "650 req/s", "325 req/s", "20 req/s", "HTTP 401"],
            "G-3": ["allkeys-lru", "volatile-lru", "lock:payout:*", "TTL = -1"],
        },
        "cdn_signed_url_s3_cors": {
            "G-1": ["public-read", "s3-media-v2", "OAC"],
            "G-2": ["K-2024", "18 hours", "K-2026"],
            "G-3": ["Access-Control-Allow-Origin", "signed-URL", "never `*`"],
        },
        "grpc_mesh_breaker_flag": {
            "G-1": ["outlierDetection", "consecutive5xxErrors: 5", "maxEjectionPercent: 25"],
            "G-2": ["kill-switch", "flag_fraud_shield_v2", "flag_dynamic_tax_v3"],
            "G-3": ["2000ms", "3000ms", "mTLS"],
        },
    }
    for env in envs_1_6:
        sc = env.scenario
        combined_probes = "\n".join(pout for _, pout in sc.probes.values()) + "\n" + sc.twist_probe_output
        reqs = expected_evidence_by_archetype[sc.archetype_id]
        for gid in ("G-1", "G-2", "G-3"):
            for token in reqs[gid]:
                assert token in combined_probes, (
                    f"{sc.archetype_id} missing guardrail {gid} probe evidence token {token!r}"
                )


@pytest.mark.parametrize("seed", [1, 2, 3, 101, 102, 103])
def test_calendar_v2_oracle_r6_shock_no_buggy_disclosure_counter(seed: int) -> None:
    env = CalendarEnv(seed=seed)
    for r in range(1, 11):
        p_a = env.turn_prompt("A", r, "BK", None, None)
        p_b = env.turn_prompt("B", r, "AL", "hi", None)
        if r == 6:
            assert "PRIVATE CALENDAR ALERT" in p_a
            assert "PRIVATE CALENDAR ALERT" in p_b
        act_a, act_b = env.oracle_actions(r)
        fb = env.step_round(r, act_a, act_b, "MTG-1 MON-1 = 2", "MTG-2 TUE-1 = 3")
        assert "disclosures_a" not in fb.metrics
        assert "disclosures_b" not in fb.metrics
    res = env.final_score()
    assert res.score == 1.0
    assert res.details["consensus_reached"] is True
    assert res.details["veto_free"] is True
    assert res.details["regret"] == 0
    assert "disclosure_violations_a" not in res.details
    assert "disclosure_violations_b" not in res.details
    assert res.details["r6_shock_applied"] is not None


@pytest.mark.parametrize("seed", [1, 2, 3, 101, 201, 301])
def test_goalpost_v2_oracle_neutral_err_leg_7_strict_int_and_shock_logging(seed: int) -> None:
    env = GoalpostEnv(seed=seed)
    assert env.total_rounds == 8

    # Strict non-integral rejection: bool and float MUST be rejected BEFORE DOMAINS check!
    orig_log_lvl = env.state["cfg.log_lvl"]
    orig_workers = env.state["cfg.workers"]
    fb_bad = env.step_round(
        1,
        {"updates": {"cfg.log_lvl": True, "cfg.workers": 2.0}},
        {"updates": {"cfg.sync_mode": False, "cfg.retry_cap": 2.9}},
        "a",
        "b",
    )
    assert "Ignored non-integer value" in fb_bad.feedback_a
    assert "Ignored non-integer value" in fb_bad.feedback_b
    assert env.state["cfg.log_lvl"] == orig_log_lvl
    assert env.state["cfg.workers"] == orig_workers

    # Fresh oracle run
    env_o = GoalpostEnv(seed=seed)
    for r in range(1, env_o.total_rounds + 1):
        p_a = env_o.turn_prompt("A", r, "BK", None, None)
        p_b = env_o.turn_prompt("B", r, "AL", "hi", None)
        if r == 5:
            assert f"ERR-LEG-7-{seed}" in p_a
            assert f"ERR-LEG-7-{seed}" in p_b
        act_a, act_b = env_o.oracle_actions(r)
        env_o.step_round(r, act_a, act_b, "updating", "updating")

    res = env_o.final_score()
    assert res.score == 1.0
    assert res.details["r5_rule_id"] == f"ERR-LEG-7-{seed}"
    # Confirm neutral ID does NOT name any Role A key
    for k in KEYS_A:
        assert k.split(".")[-1] not in str(res.details["r5_rule_id"])
    # Realized shocks & pre-shock states & per-round uptime cleanly logged
    shocks = res.details["realized_shocks"]
    assert set(shocks.keys()) == {"r3", "r5", "r7"}
    for sk in ("r3", "r5", "r7"):
        assert shocks[sk]["description"]
        assert isinstance(shocks[sk]["pre_shock_state"], dict)
    assert len(res.details["per_round_all_pass"]) == 8
    assert res.details["strict_uptime_count"] == 6


@pytest.mark.parametrize("seed", [1, 4, 7, 201, 204, 210])
def test_goalpost_two_sided_shocks_and_r7_one_shot_causality(seed: int) -> None:
    env = GoalpostEnv(seed=seed)
    t1 = t2 = t3 = t4 = None
    for r in range(1, env.total_rounds + 1):
        env.turn_prompt("A", r, "BK", None, None)
        env.turn_prompt("B", r, "AL", "sync", None)
        curr = _state_to_tuple(env.state)
        if r == 1:
            act_a: dict[str, dict[str, int]] = {"updates": {}}
            cands = [s for s in env.p1_sols if _hamming_a(curr, s) <= 2 and _hamming_b(curr, s) <= 4]
            assert cands
            t1 = _tuple_to_state(min(cands, key=lambda s: (_hamming_b(curr, s), _hamming_a(curr, s), s)))
            act_b = {"updates": env._step_toward(t1, KEYS_B)}
        elif r == 2:
            assert t1 is not None
            act_a = {"updates": env._step_toward(t1, KEYS_A)}
            act_b = {"updates": env._step_toward(t1, KEYS_B)}
        elif r in (3, 4):
            if r == 3:
                sols = env.active_solutions(3)
                assert all(_hamming_a(curr, t) >= 1 and _hamming_b(curr, t) >= 1 for t in sols)
                t2 = _tuple_to_state(_closest_target(curr, sols))
                assert 1 <= _hamming_a(curr, _state_to_tuple(t2)) <= MAX_EDITS_PER_ROUND
                assert 1 <= _hamming_b(curr, _state_to_tuple(t2)) <= MAX_EDITS_PER_ROUND
            assert t2 is not None
            act_a = {"updates": env._step_toward(t2, KEYS_A)}
            act_b = {"updates": env._step_toward(t2, KEYS_B)}
        elif r in (5, 6):
            if r == 5:
                sols = env.active_solutions(5)
                assert all(_hamming_a(curr, t) >= 1 and _hamming_b(curr, t) >= 1 for t in sols)
                t3 = _tuple_to_state(_closest_target(curr, sols))
                assert 1 <= _hamming_a(curr, _state_to_tuple(t3)) <= MAX_EDITS_PER_ROUND
                assert 1 <= _hamming_b(curr, _state_to_tuple(t3)) <= MAX_EDITS_PER_ROUND
            assert t3 is not None
            act_a = {"updates": env._step_toward(t3, KEYS_A)}
            act_b = {"updates": env._step_toward(t3, KEYS_B)}
        elif r == 7:
            act_a = {"updates": {}}
            sols = env.active_solutions(7)
            assert all(_hamming_a(curr, t) >= 1 and _hamming_b(curr, t) >= 1 for t in sols)
            t4 = _tuple_to_state(_closest_target(curr, sols))
            assert 1 <= _hamming_a(curr, _state_to_tuple(t4)) <= MAX_EDITS_PER_ROUND
            assert 1 <= _hamming_b(curr, _state_to_tuple(t4)) <= MAX_EDITS_PER_ROUND
            act_b = {"updates": {}}
        else:
            assert t4 is not None
            act_a = {"updates": env._step_toward(t4, KEYS_A)}
            act_b = {"updates": env._step_toward(t4, KEYS_B)}
        env.step_round(r, act_a, act_b, "msg", "msg")

    res = env.final_score()
    assert res.score == 1.0
    assert res.details["strict_uptime_count"] == 6


def test_goalpost_exact_csp_matches_bruteforce_and_rule_ids() -> None:
    env = GoalpostEnv(seed=201)
    bf = [
        vals
        for vals in itertools.product(*(DOMAINS[k] for k in ALL_KEYS))
        if all(r.check(dict(zip(ALL_KEYS, vals, strict=True))) for r in env.base_rules)
    ]
    assert set(bf) == set(env.p1_sols)

    for r in range(1, env.total_rounds + 1):
        env.turn_prompt("A", r, "BK", None, None)
        env.turn_prompt("B", r, "AL", "sync", None)
        aa, ab = env.oracle_actions(r)
        env.step_round(r, aa, ab, "a", "b")
    all_rids = {rule.rule_id for rule in env._active_rules(env.total_rounds)}
    assert "C-R3" in all_rids and "C-R7" in all_rids and "ERR-LEG-7-201" in all_rids
    ids_found, _ = extract_invariants(" ".join(sorted(all_rids)))
    assert all_rids.issubset(ids_found)


@pytest.mark.parametrize("seed", [1, 2, 101])
def test_ledger_backward_compat(seed: int) -> None:
    env = LedgerEnv(seed=seed)
    for r in range(1, 11):
        act_a, act_b = env.oracle_actions(r)
        env.step_round(r, act_a, act_b, "ok", "ok")
    assert env.final_score().score == 1.0
