"""End-to-end mock pipeline tests: v2 manifest balance, concurrency budget cap, symmetric splicing, 3-judge panel, analysis, and lossless HF export."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

import common_ground.analyze as analyze_mod
import common_ground.config as cfg_mod
import common_ground.export_hf as export_mod
import common_ground.runner as runner_mod
import common_ground.sim as sim_mod
from common_ground.jev import (
    TURN_DEMEANOR_QUESTIONS,
    MockDecisionPanelClient,
)
from common_ground.ledger import BudgetExceededError, Ledger, LedgerEntry, utc_now_iso


@pytest.mark.asyncio
async def test_concurrency_safe_budget_reservation(tmp_path: Path) -> None:
    """Verify 16 concurrent workers reserving in-flight cost NEVER overshoot the budget cap."""
    ledger = Ledger(tmp_path / "budget.jsonl", budget_cap_usd=1.00)
    ok_calls = 0
    blocked_calls = 0

    async def _worker(i: int) -> None:
        nonlocal ok_calls, blocked_calls
        rid = f"r_{i}"
        try:
            await ledger.reserve_budget(rid, 0.30)
        except BudgetExceededError:
            blocked_calls += 1
            return
        try:
            await asyncio.sleep(0.02)
            assert ledger.total_cost_usd + ledger.reserved_cost_usd <= 1.00 + 1e-9
            await ledger.record(
                LedgerEntry(
                    call_id=f"c_{i}",
                    timestamp_utc=utc_now_iso(),
                    model_key="astra62",
                    requested_slug="openai/gpt-6.2-astra",
                    returned_model="openai/gpt-6.2-astra",
                    provider="OpenAI",
                    reasoning_effort="medium",
                    max_tokens=100,
                    status="ok",
                    finish_reason="stop",
                    attempt=1,
                    prompt_tokens=10,
                    completion_tokens=10,
                    reasoning_tokens=None,
                    cached_tokens=0,
                    cost_usd=0.25,
                    latency_ms=5,
                    request_messages=[],
                    response_text="ok",
                ),
                reservation_id=rid,
            )
            ok_calls += 1
        finally:
            await ledger.release_reservation(rid)

    await asyncio.gather(*(_worker(i) for i in range(16)))
    assert ledger.total_cost_usd <= 1.00 + 1e-9
    assert ledger.reserved_cost_usd == pytest.approx(0.0, abs=1e-9)
    assert ok_calls >= 3 and blocked_calls > 0


@pytest.mark.asyncio
async def test_decision_panel_collision_proof_cache_and_untruncated_state(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "panel.jsonl", budget_cap_usd=10.0)
    panel = MockDecisionPanelClient(ledger)
    long_msg = "Detailed explanation: " + ("x" * 2500)
    res1 = await panel.decide(
        call_id="ep1:turn:r1:A",
        state={"message": long_msg, "round": 1},
        questions=TURN_DEMEANOR_QUESTIONS,
        metadata={"source_model": "astra62"},
    )
    assert set(res1["panel_judges"].keys()) == {"liquid", "perplexity", "typesafe"}
    assert res1["panel_rubric_version"] == "rubric_v2"
    assert len(ledger.all_entries) == 3
    # Full untruncated state stored in ledger!
    stored_req = ledger.all_entries[0].request_messages[0]["content"]
    assert long_msg in stored_req
    # Cache key format: {judge_key}:{rubric_v2}:{call_id}:{state_sha256[:12]}
    for e in ledger.all_entries:
        parts = e.call_id.split(":")
        assert parts[0] in {"liquid", "perplexity", "typesafe"}
        assert parts[1] == "rubric_v2"
        assert len(parts[-1]) == 12

    # Same call_id with DIFFERENT state produces a NEW cache key (collision-proof!)
    res2 = await panel.decide(
        call_id="ep1:turn:r1:A",
        state={"message": long_msg + " MODIFIED", "round": 1},
        questions=TURN_DEMEANOR_QUESTIONS,
        metadata={"source_model": "astra62"},
    )
    assert res2["state_sha256"] != res1["state_sha256"]
    assert len(ledger.all_entries) == 6


def test_symmetric_choice_splicing_order_invariant_and_same_seed_framing() -> None:
    """Verify AB and BA are genuine byte-for-byte order reversals with identical handles,
    zero h1==my2 collisions, balanced handle slots, and truthful same-seed parallel-run framing."""
    ep1 = {
        "episode_id": "triage__s301__opus55__vs__astra62",
        "task": "triage",
        "seed": 301,
        "handle_a": "AL",
        "handle_b": "BK",  # h1 = BK
        "history_a": [
            {"role": "system", "content": "You are AL working with BK on Task A."},
            {"role": "user", "content": "Teammate BK said:\nHello AL from BK. Run ALERT-1."},
            {"role": "assistant", "content": "Thanks BK, I am AL."},
        ],
    }
    ep2 = {
        "episode_id": "triage__s302__opus55__vs__astra60",
        "task": "triage",
        "seed": 302,
        "handle_a": "BK",  # my2 == h1 !!
        "handle_b": "AL",  # h2 == my1 !!
        "history_a": [
            {"role": "system", "content": "You are BK working with AL on Task B."},
            {"role": "user", "content": "Teammate AL said:\nHello BK from AL. Status GREEN."},
            {"role": "assistant", "content": "Thanks AL, I am BK."},
        ],
    }
    ab_spliced, ab_h1, ab_h2 = runner_mod._splice_two_histories(ep1, ep2, "A")
    ba_spliced, ba_h1, ba_h2 = runner_mod._splice_two_histories(ep2, ep1, "A")
    assert len(ab_spliced) == 2 and ab_spliced[0]["role"] == "system" and ab_spliced[1]["role"] == "user"
    # Order invariance: reversing (ep1, ep2) -> (ep2, ep1) swaps (h1, h2) EXACTLY and keeps my_h identical!
    assert (ba_h1, ba_h2) == (ab_h2, ab_h1)
    ab_my = ab_spliced[0]["content"].split("You are teammate ")[1].split(".")[0]
    ba_my = ba_spliced[0]["content"].split("You are teammate ")[1].split(".")[0]
    assert ab_my == ba_my
    assert len({ab_my, ab_h1, ab_h2}) == 3

    # Candidate transcript sections are byte-identical after swapping slot numbers 1 <-> 2!
    ab_txt = ab_spliced[1]["content"]
    ba_txt = ba_spliced[1]["content"]
    assert "ALERT-1" in ab_txt and "GREEN" in ab_txt
    ab_sec1 = ab_txt.split("=== TASK 2")[0].replace("TASK 1", "TASK X")
    ba_sec2 = ("=== TASK 2" + ba_txt.split("=== TASK 2")[1]).split("\n\nYou have completed")[0].replace("TASK 2", "TASK X")
    assert ab_sec1.strip() == ba_sec2.strip()

    # Same-seed framing uses truthful RUN 1 / RUN 2 parallel-run prompt
    ep2_same = {**ep2, "episode_id": "triage__s301__opus55__vs__astra60", "seed": 301}
    ss_spliced, ss_h1, ss_h2 = runner_mod._splice_two_histories(ep1, ep2_same, "A")
    assert "two independent runs of the same collaborative task scenario" in ss_spliced[0]["content"]
    assert f"=== RUN 1 (with teammate {ss_h1}) ===" in ss_spliced[1]["content"]
    assert f"=== RUN 2 (with teammate {ss_h2}) ===" in ss_spliced[1]["content"]


def test_v2_manifest_exact_counts_same_seed_h2h_and_zero_handle_slot_bias() -> None:
    """Verify frozen v2 manifest hits exact 704 episodes, 224 F1_retest, 3,200 choices (same-seed H2H),
    and unbiased handle distribution across slots 1 and 2."""
    m = runner_mod.build_v2_manifest()
    c = m["counts"]
    assert c["episodes_main"] == 640
    assert c["episodes_opus_boost"] == 64
    assert c["episodes_total"] == 704
    assert c["f1_retest_agent_episodes"] == 224
    assert c["choices_total"] == 3200
    assert c["choices_by_type"] == {
        "main_7p": 2688,
        "aa_placebo": 256,
        "extra_h2h": 192,
        "opus_boost_h2h": 64,
    }
    # Verify all extra_h2h and opus_boost_h2h are same-seed (zero scenario confound, zero boost reuse)
    eh = [x for x in m["choices"] if x["choice_type"] == "extra_h2h"]
    ob = [x for x in m["choices"] if x["choice_type"] == "opus_boost_h2h"]
    assert all(x["seed_1"] == x["seed_2"] and x["same_seed"] is True for x in eh)
    assert all(x["seed_1"] == x["seed_2"] and x["same_seed"] is True for x in ob)
    # Zero episode reuse in opus_boost_h2h AB blocks
    ob_ab = [x for x in ob if x["order"] == "AB"]
    assert len(ob_ab) == 32
    ob_eps = [x["ep_1_id"] for x in ob_ab] + [x["ep_2_id"] for x in ob_ab]
    assert len(ob_eps) == len(set(ob_eps)) == 64

    # Verify handle slot symmetry: AL appears equally in slot 1 and slot 2 across all AB/BA blocks!
    al_slot1, al_slot2 = 0, 0
    for ch in m["choices"]:
        if ch["choice_type"] != "main_7p":
            continue
        _, h1, h2 = runner_mod._deterministic_case_handles(ch["ep_1_id"], ch["ep_2_id"], ch["role"])
        al_slot1 += int(h1 == "AL")
        al_slot2 += int(h2 == "AL")
    assert al_slot1 == al_slot2


@pytest.mark.asyncio
async def test_graceful_episode_and_choice_exclusion_on_api_error_and_freeze_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify a single persisted OpenRouterCallError excludes that episode/choice cleanly instead of crashing,
    and verify --mock cannot pollute 'v2' and strict_freeze rejects missing manifests/probes."""
    from common_ground.client import OpenRouterCallError

    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "DATA_DIR", tmp_path)

    # Guard 1: refusing --mock into live 'v2'
    with pytest.raises(ValueError, match="Refusing to write --mock"):
        await runner_mod.run_experiment(phase="v2", run_name="v2", budget_cap_usd=10.0, mock=True)

    # Guard 2: strict_freeze rejects missing probe manifest
    with pytest.raises(FileNotFoundError, match="Required probe manifest missing"):
        await runner_mod.run_experiment(
            phase="v2",
            run_name="live_unfrozen_test",
            budget_cap_usd=10.0,
            mock=False,
            probe_manifest_path=tmp_path / "nonexistent_probe.json",
            strict_freeze=True,
        )

    # Guard 3: single persisted OpenRouterCallError marks 1 episode failed and completes the rest + logs excluded choices
    orig_complete = runner_mod.MockOpenRouterClient.complete

    async def _flaky_complete(self, *, call_id: str, **kwargs):
        if "triage__s1__astra62__vs__opus55:turn:r1:A" in call_id:
            raise OpenRouterCallError(f"Simulated persisted truncation on {call_id}")
        return await orig_complete(self, call_id=call_id, **kwargs)

    monkeypatch.setattr(runner_mod.MockOpenRouterClient, "complete", _flaky_complete)
    res = await runner_mod.run_experiment(
        phase="calibrate",
        run_name="cal_flaky",
        budget_cap_usd=100.0,
        mock=True,
        max_concurrent=4,
    )
    assert res["episodes_failed"] == 1
    assert res["episodes_completed"] == 13
    assert res["budget_stopped"] is False


@pytest.mark.asyncio
async def test_full_mock_pilot_resume_analyze_and_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(analyze_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(export_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sim_mod, "DATA_DIR", tmp_path)

    # 1. Calibration mini-pilot (14 episodes)
    cal = await runner_mod.run_experiment(
        phase="calibrate",
        run_name="cal_test",
        budget_cap_usd=100.0,
        mock=True,
        max_concurrent=14,
        rating_scale_max=7,
    )
    assert cal["episodes_completed"] == 14
    cal_rep = analyze_mod.analyze_run("cal_test")
    assert "decision_panel_reliability" in cal_rep
    assert "clarity" in cal_rep["decision_panel_reliability"]
    assert "triage_role_A_investigator_by_model" in cal_rep["jev_demeanor_and_audits"]

    # 2. Pilot (48 episodes + F1-F5 + F3a/F3b + 3-judge panel + symmetric choices)
    summary1 = await runner_mod.run_experiment(
        phase="pilot",
        run_name="pilot_test",
        budget_cap_usd=500.0,
        mock=True,
        max_concurrent=16,
        rating_scale_max=7,
    )
    assert summary1["episodes_completed"] == 48
    assert summary1["choices_completed"] > 0
    assert summary1["budget_stopped"] is False

    ledger_path = tmp_path / "pilot_test" / "ledger.jsonl"
    lines_after_run1 = len(ledger_path.read_text(encoding="utf-8").splitlines())
    assert lines_after_run1 > 500

    # Resume idempotency: ZERO new API calls!
    summary2 = await runner_mod.run_experiment(
        phase="pilot",
        run_name="pilot_test",
        budget_cap_usd=500.0,
        mock=True,
        max_concurrent=16,
    )
    lines_after_run2 = len(ledger_path.read_text(encoding="utf-8").splitlines())
    assert lines_after_run2 == lines_after_run1
    assert summary2["total_cost_usd"] == summary1["total_cost_usd"]

    # Budget cap enforcement
    summary_cap = await runner_mod.run_experiment(
        phase="pilot",
        run_name="pilot_capped",
        budget_cap_usd=0.05,
        mock=True,
        max_concurrent=4,
    )
    assert summary_cap["budget_stopped"] is True
    assert summary_cap["episodes_completed"] < 48

    # Full analysis + relative figure paths + quote cards + false-blame audit
    report = analyze_mod.analyze_run("pilot_test")
    assert report["n_episodes"] == 48
    assert report["H1_choice_model_adjusted"]["ok"] is True
    assert report["H1b_predecessor_head_to_head"]["ok"] is True
    assert report["AA_placebo_check"]["ok"] is True
    assert len(report["figures"]) >= 4
    for fig_rel in report["figures"]:
        assert not Path(fig_rel).is_absolute(), f"Absolute figure path leaked: {fig_rel}"
        assert (tmp_path / "pilot_test" / "analysis" / fig_rel).exists()
    qc_text = (tmp_path / "pilot_test" / "analysis" / "quote_cards.md").read_text(encoding="utf-8")
    assert "`nan`" not in qc_text.lower()
    assert (tmp_path / "pilot_test" / "analysis" / "false_blame_audit.md").exists()

    # Budget projection
    proj = sim_mod.project_budget_from_pilot("pilot_test")
    assert "main_run_projections" in proj

    # HuggingFace export hygiene: lean zstd Parquet-only, zero human_poll files, zero .jsonl
    hf_dir = export_mod.export_huggingface_dataset("pilot_test")
    for fname in (
        "episodes.parquet",
        "turns.parquet",
        "debriefs.parquet",
        "choices.parquet",
        "README.md",
    ):
        assert (hf_dir / fname).exists(), f"Missing exported file {fname}"

    assert list(hf_dir.rglob("*.jsonl")) == [], "Duplicate .jsonl files found in hf_export/!"
    assert list(hf_dir.rglob("*human_poll*")) == [], "Deprecated human poll file found in hf_export/!"
    assert not (tmp_path / "pilot_test" / "human_poll_answer_key.json").exists()
    readme_txt = (hf_dir / "README.md").read_text(encoding="utf-8")
    assert "configs:" in readme_txt
    assert "—" not in readme_txt
    deb_pilot = pd.read_parquet(hf_dir / "debriefs.parquet")
    assert set(deb_pilot["fork"].unique()) <= {"F1", "F1_retest", "F4"}
    assert "false_blame_json" not in deb_pilot.columns
    turns_pilot = pd.read_parquet(hf_dir / "turns.parquet")
    assert "latency_s" in turns_pilot.columns and "ledger_joined" in turns_pilot.columns
    assert bool(turns_pilot["ledger_joined"].all())


@pytest.mark.asyncio
async def test_v2_end_to_end_mock_704_episodes_3200_choices_capstone_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full v2 offline gate: 704 episodes, 224 F1_retest, 3,200 choices, boost isolation, ICC(2,1),
    symmetric Capstone Tone-Flip crossover, lossless HF export, and strict HF-only replay verification."""
    import importlib.util

    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(analyze_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(export_mod, "DATA_DIR", tmp_path)

    summary = await runner_mod.run_experiment(
        phase="v2",
        run_name="v2_mock",
        budget_cap_usd=5000.0,
        mock=True,
        max_concurrent=32,
        rating_scale_max=7,
    )
    assert summary["episodes_completed"] == 704
    assert summary["episodes_planned"] == 704
    assert summary["episodes_failed"] == 0
    assert summary["choices_completed"] == 3200
    assert summary["choices_planned"] == 3200
    assert summary["budget_stopped"] is False

    rep = analyze_mod.analyze_run("v2_mock")
    assert rep["n_episodes"] == 704
    assert rep["n_episodes_main"] == 640
    assert rep["n_episodes_opus_boost"] == 64
    # Main debrief count is strictly 896 (zero opus_boost leakage!), total is 960 (896 + 64 opus boost)
    assert rep["n_debrief_agent_episodes_main"] == 896
    assert rep["n_debrief_agent_episodes_total"] == 960
    assert rep["n_choices_total"] == 3200
    assert len(rep["partner_league_table"]) == 7
    assert rep["unified_bt_adjusted"]["ok"] is True
    assert rep["unified_bt_adjusted"]["n_choices"] == 2688
    assert rep["unified_bt_adjusted"]["n_boot_achieved"] == 2000
    assert rep["unified_bt_scenario_cluster_sensitivity"]["ok"] is True
    assert rep["aa_intelligence_vs_bt"]["ok"] is True
    assert rep["aa_intelligence_vs_bt"]["n_models"] == 7
    assert rep["F1_retest_reliability"]["ok"] is True
    assert rep["F1_retest_reliability"]["n"] == 224
    assert "icc_2_1" in rep["F1_retest_reliability"]
    assert rep["H1b_predecessor_head_to_head"]["ok"] is True
    assert rep["H1b_predecessor_head_to_head"]["main_shared_seed_cross_vendor"]["n_cases"] == 96
    assert rep["H1b_predecessor_head_to_head"]["opus_boost_same_seed"]["n_cases"] == 32
    assert rep["H1b_predecessor_head_to_head"]["opus_combined_same_seed"]["n_cases"] == 64
    did = rep["H1b_predecessor_head_to_head"]["claude_minus_other_cross_vendor_did"]
    assert did["scope"] == "main_shared_seeds_only_301_304"
    assert did["opus55_n_cases"] == 32 and did["other_cross_vendor_n_cases"] == 64
    assert "leave_one_scenario_out" in rep["leave_one_scenario_out_sensitivity"]
    assert "triage_active_vs_post_green" in rep["supplementary_exploratory"]
    assert rep["supplementary_exploratory"]["demeanor_choice_regression_visibility_masked"]["visibility_masked"] is True
    assert rep["supplementary_exploratory"]["task_group_contrasts"]["code_handoff_only"]["adjusted"]["n_boot_achieved"] == 2000
    assert len(rep["figures"]) == 5

    # Run redesigned Symmetric Capstone Tone-Flip Crossover on the 640 main mock episodes
    cap_script = Path(__file__).resolve().parents[1] / "scripts" / "run_capstone_tone_flip.py"
    cap_spec = importlib.util.spec_from_file_location("run_capstone_tone_flip", cap_script)
    assert cap_spec is not None and cap_spec.loader is not None
    cap_mod = importlib.util.module_from_spec(cap_spec)
    cap_spec.loader.exec_module(cap_mod)
    monkeypatch.setattr(cap_mod, "DATA_DIR", tmp_path)

    cap_res = await cap_mod.run_capstone(
        source_run="v2_mock",
        out_run="v2_mock_capstone",
        n_pairs=12,
        budget_cap_usd=100.0,
        mock=True,
    )
    assert cap_res["schema_version"] == "v2_symmetric_crossover"
    assert cap_res["selection"]["n_pairs_selected"] == 12
    assert cap_res["selection"]["n_unique_episodes"] == 24
    assert len(cap_res["selection"]["by_seed"]) >= 3  # spans multiple seeds, NOT seed 301 only!
    assert cap_res["n_choices_total"] == 12 * 2 * 2 * 4  # 192 choices

    # Verify Capstone handle remapping preserves ALERT-1/GREEN/SELECT/CONFLICT & outcome cards have zero None
    cap_rows = [
        json.loads(ln)
        for ln in (tmp_path / "v2_mock" / "capstone_tone_flip" / "capstone_choices.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(cap_rows) == 192
    for row in cap_rows[:12]:
        assert "None" not in row["outcome_card_1"]
        assert "None" not in row["outcome_card_2"]
        assert len(row["transcript_1_turns"]) > 0
        assert len(row["transcript_2_turns"]) > 0

    hf_dir = export_mod.export_huggingface_dataset("v2_mock")
    assert list(hf_dir.rglob("*.jsonl")) == []
    assert list(hf_dir.rglob("*human_poll*")) == []
    assert not (tmp_path / "v2_mock" / "human_poll_answer_key.json").exists()
    assert (hf_dir / "capstone_tone_flip.parquet").exists()
    assert (hf_dir / "capstone_rewrites.parquet").exists()
    # Strictly retain ONLY F1, F1_retest, F4 (zero F2/F5/F3, zero false_blame_json) + arm column!
    deb_df = pd.read_parquet(hf_dir / "debriefs.parquet")
    assert set(deb_df["fork"].unique()) == {"F1", "F1_retest", "F4"}
    assert "false_blame_json" not in deb_df.columns
    assert "arm" in deb_df.columns
    ch_df = pd.read_parquet(hf_dir / "choices.parquet")
    assert len(ch_df) == 3200
    ep_df = pd.read_parquet(hf_dir / "episodes.parquet")
    assert len(ep_df) == 704
    turns_df = pd.read_parquet(hf_dir / "turns.parquet")
    assert "latency_s" in turns_df.columns and "ledger_joined" in turns_df.columns
    assert bool(turns_df["ledger_joined"].all())

    # Release-level replay verification strictly from hf_export/*.parquet!
    rep_script = Path(__file__).resolve().parents[1] / "scripts" / "verify_hf_replay.py"
    rep_spec = importlib.util.spec_from_file_location("verify_hf_replay", rep_script)
    assert rep_spec is not None and rep_spec.loader is not None
    rep_mod = importlib.util.module_from_spec(rep_spec)
    rep_spec.loader.exec_module(rep_mod)
    monkeypatch.setattr(rep_mod, "DATA_DIR", tmp_path)
    replay_ok = rep_mod.verify_hf_replay("v2_mock", capstone_run="v2_mock_capstone")
    assert replay_ok["ok"] is True
    assert replay_ok["hygiene"]["hygiene_ok"] is True
    assert replay_ok["bt_reproduced"] is True
    assert replay_ok["h1b_reproduced"] is True
    assert replay_ok["f1_retest_reproduced"] is True
    assert replay_ok["capstone_reproduced"] is True


def test_capstone_whole_token_handle_remap_and_outcome_card_schemas() -> None:
    """Verify Capstone handle remapping NEVER corrupts ALERT-1/FLUSHALL/GREEN/SELECT/CONFLICT
    and outcome cards read exact Calendar & Goalpost schema keys."""
    import importlib.util

    cap_script = Path(__file__).resolve().parents[1] / "scripts" / "run_capstone_tone_flip.py"
    cap_spec = importlib.util.spec_from_file_location("run_capstone_tone_flip", cap_script)
    assert cap_spec is not None and cap_spec.loader is not None
    cap_mod = importlib.util.module_from_spec(cap_spec)
    cap_spec.loader.exec_module(cap_mod)

    fake_ep = {
        "episode_id": "ep_test",
        "task": "triage",
        "handle_a": "AL",
        "handle_b": "EN",
        "turns": [
            {
                "round": 1,
                "A": {"message": "EN, check ALERT-1, FLUSHALL, SELECT, CONFLICT, GREEN. —AL"},
                "B": {"message": "AL, confirmed GREEN after ALERT-1. —EN"},
            }
        ],
    }
    # Swap AL -> EN and EN -> KV (would collide & corrupt under sequential .replace()!)
    txt, turns = cap_mod.format_transcript_and_turns(
        fake_ep, target_role="A", candidate_handle="EN", teammate_handle="KV"
    )
    assert "ALERT-1" in txt and "FLUSHALL" in txt and "SELECT" in txt and "CONFLICT" in txt and "GREEN" in txt
    assert "KVERT-1" not in txt and "GREKV" not in txt
    assert "KV, check ALERT-1" in turns[0]["text"]
    assert "—EN" in turns[0]["text"]

    # Calendar card uses real keys (`matched_meetings`, `consensus_reached`, `veto_free`, `regret`)
    cal_card = cap_mod.format_outcome_card(
        {
            "episode_id": "cal_1",
            "task": "calendar",
            "total_rounds": 10,
            "score": 0.92,
            "score_details": {"matched_meetings": 7, "consensus_reached": True, "veto_free": True, "regret": 4},
        },
        "A",
    )
    assert "7/7" in cal_card and "Consensus reached: Yes" in cal_card and "Veto-free: Yes" in cal_card

    # Goalpost card uses real keys (`passed`, `total_active`, `strict_uptime_count`)
    gp_card = cap_mod.format_outcome_card(
        {
            "episode_id": "gp_1",
            "task": "goalpost",
            "total_rounds": 8,
            "score": 0.85,
            "score_details": {"passed": 10, "total_active": 10, "strict_uptime_count": 6, "passable_rounds": [2, 3, 4, 5, 6, 7, 8]},
        },
        "B",
    )
    assert "10/10" in gp_card and "6/7 passable rounds" in gp_card and "None" not in gp_card
