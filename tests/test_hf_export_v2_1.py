"""Unit and integration tests for lean v2.1 HuggingFace Parquet export and strict replay verification."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

import common_ground.export_hf as export_mod
from common_ground.config import DATA_DIR


def _load_replay_module() -> Any:
    rep_script = Path(__file__).resolve().parents[1] / "scripts" / "verify_hf_replay.py"
    spec = importlib.util.spec_from_file_location("verify_hf_replay_v2_1", rep_script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_synthetic_v2_1_run(base_dir: Path, run_name: str = "synth_v2_1") -> Path:
    """Create a minimal synthetic run exercising every v2.1 export table and manifest."""
    run_dir = base_dir / run_name
    (run_dir / "episodes").mkdir(parents=True)
    (run_dir / "exp1_dropout").mkdir(parents=True)
    (run_dir / "exp2_info_flip").mkdir(parents=True)

    # Pre-seed stale v2 fat inside hf_export/ AND an untouched file outside hf_export/
    stale_hf = run_dir / "hf_export"
    stale_hf.mkdir(parents=True)
    (stale_hf / "episodes.jsonl").write_text('{"stale": true}\n', encoding="utf-8")
    (stale_hf / "human_poll_blinded.html").write_text("<html>stale</html>", encoding="utf-8")
    (stale_hf / "stale_table.parquet").write_bytes(b"not a parquet")
    (stale_hf / "stale_subdir").mkdir()
    (stale_hf / "stale_subdir" / "junk.txt").write_text("junk", encoding="utf-8")
    outside_sentinel = run_dir / "human_poll_answer_key.json"
    outside_sentinel.write_text('{"untouched": true}\n', encoding="utf-8")

    # Completed episode with deprecated F2, F5, F3a, R5 false_blame alongside retained F1/F1_retest/F4
    ep_ok = {
        "episode_id": "goalpost__s301__astra62__vs__opus55",
        "run_name": run_name,
        "task": "goalpost",
        "seed": 301,
        "arm": "core",
        "complete": True,
        "total_rounds": 1,
        "model_a": "astra62",
        "model_b": "opus55",
        "handle_a": "AL",
        "handle_b": "BK",
        "briefing_a": "Briefing A",
        "briefing_b": "Briefing B",
        "score": 1.0,
        "score_details": {"passed": 14, "total_active": 14},
        "format_retries": {"A": 0, "B": 0},
        "action_parse_failures": {"A": 0, "B": 0},
        "turns": [
            {
                "round": 1,
                "feedback_a": "ok",
                "feedback_b": "ok",
                "round_metrics": {"passed": 14},
                "A": {
                    "model": "astra62",
                    "handle": "AL",
                    "message": "Hello BK",
                    "action": {"workers": 4},
                    "parse_error": None,
                    "metrics": {"word_count": 2, "ids_per_100": 0.0, "pleasantry_count": 0},
                    "jev_demeanor": {
                        "clarity": 2.5,
                        "warmth": 2.0,
                        "expresses_confusion": 0.01,
                        "panel_consensus": {
                            "clarity": {"mean": 2.5, "n_judges": 3},
                            "warmth": {"mean": 2.0, "n_judges": 3},
                            "expresses_confusion": {"mean": 0.01, "n_judges": 3},
                        },
                        "panel_judges": {"liquid": {"scores": {"clarity": 2.5}}},
                    },
                    "accepted_call_id": "cid_r1_A",
                    "accepted_attempt": 1,
                    "returned_model": "openai/gpt-6.2-astra",
                    "provider": "OpenAI",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "reasoning_tokens": 20,
                    "cached_tokens": 0,
                    "accepted_cost_usd": 0.01,
                    "total_cost_usd": 0.015,
                },
                "B": {
                    "model": "opus55",
                    "handle": "BK",
                    "message": "Hello AL",
                    "action": {"workers": 4},
                    "parse_error": None,
                    "metrics": {"word_count": 2, "ids_per_100": 0.0, "pleasantry_count": 0},
                    "jev_demeanor": {
                        "clarity": 3.0,
                        "warmth": 2.5,
                        "expresses_confusion": 0.0,
                        "panel_consensus": {
                            "clarity": {"mean": 3.0, "n_judges": 3},
                            "warmth": {"mean": 2.5, "n_judges": 3},
                            "expresses_confusion": {"mean": 0.0, "n_judges": 3},
                        },
                        "panel_judges": {"liquid": {"scores": {"clarity": 3.0}}},
                    },
                    "accepted_call_id": "cid_r1_B",
                    "accepted_attempt": 1,
                    "returned_model": "anthropic/claude-opus-5.5",
                    "provider": "Anthropic",
                    "prompt_tokens": 120,
                    "completion_tokens": 60,
                    "reasoning_tokens": 30,
                    "cached_tokens": 10,
                    "accepted_cost_usd": 0.02,
                    "total_cost_usd": 0.02,
                },
            }
        ],
        "debriefs": {
            "A": {
                "subject_model": "astra62",
                "partner_model": "opus55",
                "false_blame_r5": {"blamed_partner": True},
                "F1": {"raw": '{"again": 6}', "parsed": {"again": 6}, "cost_usd": 0.001},
                "F1_retest": {"raw": '{"again": 6}', "parsed": {"again": 6}, "cost_usd": 0.001},
                "F2": {"raw": '{"again": 7}', "parsed": {"again": 7}, "cost_usd": 0.001},
                "F3a": {"raw": '{"again": 5}', "parsed": {"again": 5}, "cost_usd": 0.001},
                "F4": {"raw": '{"choice": "KEEP"}', "choice": "KEEP", "cost_usd": 0.001},
                "F5": {"raw": '{"guess": "anthropic"}', "parsed": {"guess": "anthropic"}, "cost_usd": 0.001},
            }
        },
    }
    ep_failed = {
        "episode_id": "calendar__s302__neon4__vs__opus55",
        "task": "calendar",
        "seed": 302,
        "model_a": "neon4",
        "model_b": "opus55",
        "arm": "core",
        "complete": False,
        "failed": True,
        "error": "persisted truncated: finish_reason=length",
    }
    (run_dir / "episodes" / f"{ep_ok['episode_id']}.json").write_text(
        json.dumps(ep_ok), encoding="utf-8"
    )
    (run_dir / "episodes" / f"{ep_failed['episode_id']}.json").write_text(
        json.dumps(ep_failed), encoding="utf-8"
    )

    # Turn ledger for latency_s join
    ledger_rows = [
        {
            "call_id": "cid_r1_A",
            "status": "ok",
            "latency_ms": 2345,
            "cost_usd": 0.01,
            "metadata": {"stage": "turn", "episode_id": ep_ok["episode_id"], "round": 1, "role": "A"},
        },
        {
            "call_id": "cid_r1_B",
            "status": "ok",
            "latency_ms": 1500,
            "cost_usd": 0.02,
            "metadata": {"stage": "turn", "episode_id": ep_ok["episode_id"], "round": 1, "role": "B"},
        },
    ]
    (run_dir / "ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ledger_rows) + "\n", encoding="utf-8"
    )

    # Choices + excluded choices
    (run_dir / "choices.jsonl").write_text(
        json.dumps(
            {
                "manifest_item_id": "m1",
                "choice_type": "main_7p",
                "order": "AB",
                "task": "goalpost",
                "role": "A",
                "subject_model": "astra62",
                "partner_1": "opus55",
                "partner_2": "neon4",
                "episode_1_id": ep_ok["episode_id"],
                "episode_2_id": ep_ok["episode_id"],
                "score_1": 1.0,
                "score_2": 1.0,
                "chosen_slot": 1,
                "chosen_partner": "opus55",
                "reason": "Clear formulas.",
                "choice_packet": {"system": "sys", "user": "usr"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "choices_excluded.jsonl").write_text(
        json.dumps(
            {
                "manifest_item_id": "m_excl",
                "block_id": "b1",
                "choice_type": "main_7p",
                "reason": "ep_1_missing",
                "ep_1_id": ep_failed["episode_id"],
                "ep_2_id": ep_ok["episode_id"],
                "ep_1_present": False,
                "ep_2_present": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Exp 1 minimal synthetic artifacts
    gate_rec = {
        "call_id": "g_ds_1",
        "condition": "floor_no_partner",
        "case_id": "c1",
        "episode_id": ep_ok["episode_id"],
        "seed": 301,
        "arm": "core",
        "dropped_role": "A",
        "receiver_role": "B",
        "sender_model": "astra62",
        "receiver_key": "deepseek_flash",
        "receiver_slug": "deepseek/deepseek-v4.1-flash",
        "receiver_family": "deepseek",
        "receiver_tier": "worker",
        "receiver_effort": "disabled",
        "receiver_aa_index_v5": 19.0,
        "status": "ok",
        "error_detail": None,
        "returned_model": "deepseek/deepseek-v4.1-flash",
        "provider": "BaseTen",
        "finish_reason": "stop",
        "prompt_tokens": 500,
        "completion_tokens": 80,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 0.001,
        "latency_ms": 900,
        "parse_ok": True,
        "domain_ok": True,
        "own_rules_passed": 7,
        "r3_passed": True,
        "own_plus_r3_pass": True,
        "partner_rules_passed": 7,
        "partner_rules_pass": True,
        "strict_all_pass": True,
        "total_rules_passed": 15,
        "parsed_config": {"workers": 4},
        "rule_results": {"C-A1": True},
    }
    (run_dir / "exp1_dropout" / "gate_report.json").write_text(
        json.dumps(
            {
                "receivers": {
                    "deepseek_flash": {
                        "step_up_history": [
                            {"step_index": 0, "passed_gate": True, "records": [gate_rec]}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    # Write a stale non-ok followed by ok for same (receiver_key, case_id) to verify deduplication
    res_err = {
        "call_id": "h_1",
        "condition": "handoff",
        "case_id": "c1",
        "episode_id": ep_ok["episode_id"],
        "seed": 301,
        "arm": "core",
        "dropped_role": "A",
        "receiver_role": "B",
        "is_primary_unconfounded": True,
        "sender_model": "astra62",
        "sender_family": "openai",
        "original_receiver_model": "opus55",
        "original_receiver_family": "anthropic",
        "receiver_key": "deepseek_flash",
        "receiver_slug": "deepseek/deepseek-v4.1-flash",
        "receiver_family": "deepseek",
        "receiver_display_name": "DeepSeek V4.1 Flash",
        "receiver_tier": "worker",
        "receiver_effort": "disabled",
        "receiver_aa_index_v5": 19.0,
        "is_same_family_as_sender": False,
        "is_cross_vendor": True,
        "is_same_model_as_original_receiver": False,
        "episode_score": 1.0,
        "r2_all_pass": True,
        "sender_r1_n_formula": 7,
        "sender_r1_full_disclosure": True,
        "sender_r1_r2_n_formula": 7,
        "sender_r1_r2_full_disclosure": True,
        "sender_r1_n_id": 7,
        "sender_r1_r2_n_id": 7,
        "sender_r1_word_count": 100,
        "sender_r1_r2_word_count": 150,
        "original_receiver_asked_rules_r1": False,
        "original_receiver_asked_rules": False,
        "status": "error",
        "error_detail": "HTTP 429",
        "returned_model": None,
        "provider": None,
        "finish_reason": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 100,
        "parse_ok": False,
        "domain_ok": False,
        "normalized_keys": False,
        "parsed_config": {},
        "parse_error": "err",
        "missing_keys": [],
        "out_of_domain_keys": {},
        "own_rules_passed": 0,
        "r3_passed": False,
        "own_plus_r3_pass": False,
        "own_plus_r3_frac": 0.0,
        "partner_rules_passed": 0,
        "partner_rules_pass": False,
        "partner_rules_frac": 0.0,
        "strict_all_pass": False,
        "total_rules_passed": 0,
        "copied_r2_unchanged": False,
        "hamming_from_r2": 0,
        "rule_results": {},
        "disclosed_partner_rules_passed": 0,
        "disclosed_partner_rules_total": 7,
        "undisclosed_partner_rules_passed": None,
        "undisclosed_partner_rules_total": 0,
    }
    res_ok = {
        **res_err,
        "status": "ok",
        "error_detail": None,
        "returned_model": "deepseek/deepseek-v4.1-flash",
        "provider": "BaseTen",
        "finish_reason": "stop",
        "prompt_tokens": 600,
        "completion_tokens": 90,
        "cost_usd": 0.002,
        "latency_ms": 1200,
        "parse_ok": True,
        "domain_ok": True,
        "own_rules_passed": 7,
        "r3_passed": True,
        "own_plus_r3_pass": True,
        "own_plus_r3_frac": 1.0,
        "partner_rules_passed": 7,
        "partner_rules_pass": True,
        "partner_rules_frac": 1.0,
        "strict_all_pass": True,
        "total_rules_passed": 15,
        "disclosed_partner_rules_passed": 7,
    }
    (run_dir / "exp1_dropout" / "results.jsonl").write_text(
        json.dumps(res_err) + "\n" + json.dumps(res_ok) + "\n", encoding="utf-8"
    )
    exp1_led = [
        {
            "call_id": "g_ds_1",
            "timestamp_utc": "2026-11-23T00:00:00Z",
            "model_key": "deepseek_flash",
            "requested_slug": "deepseek/deepseek-v4.1-flash",
            "returned_model": "deepseek/deepseek-v4.1-flash",
            "provider": "BaseTen",
            "reasoning_effort": "disabled",
            "max_tokens": 16384,
            "status": "ok",
            "finish_reason": "stop",
            "attempt": 1,
            "prompt_tokens": 500,
            "completion_tokens": 80,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0.001,
            "latency_ms": 900,
            "response_text": '{"workers": 4}',
            "error_detail": None,
            "metadata": {
                "stage": "exp1_dropout",
                "condition": "floor_no_partner",
                "case_id": "c1",
                "episode_id": ep_ok["episode_id"],
                "seed": 301,
                "dropped_role": "A",
                "receiver_role": "B",
                "sender_model": "astra62",
                "receiver_tier": "worker",
                "effort": "disabled",
            },
        },
        {
            "call_id": "h_1",
            "timestamp_utc": "2026-11-23T00:00:01Z",
            "model_key": "deepseek_flash",
            "requested_slug": "deepseek/deepseek-v4.1-flash",
            "returned_model": "deepseek/deepseek-v4.1-flash",
            "provider": "BaseTen",
            "reasoning_effort": "disabled",
            "max_tokens": 16384,
            "status": "ok",
            "finish_reason": "stop",
            "attempt": 1,
            "prompt_tokens": 600,
            "completion_tokens": 90,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0.002,
            "latency_ms": 1200,
            "response_text": '{"workers": 4}',
            "error_detail": None,
            "metadata": {
                "stage": "exp1_dropout",
                "condition": "handoff",
                "case_id": "c1",
                "episode_id": ep_ok["episode_id"],
                "seed": 301,
                "dropped_role": "A",
                "receiver_role": "B",
                "sender_model": "astra62",
                "receiver_tier": "worker",
                "effort": "disabled",
            },
        },
    ]
    (run_dir / "exp1_dropout" / "ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in exp1_led) + "\n", encoding="utf-8"
    )
    (run_dir / "exp1_dropout" / "locked_receiver_roster.json").write_text(
        json.dumps({"schema_version": "exp1_locked_roster_v1", "receivers": {}}),
        encoding="utf-8",
    )

    # Exp 2 minimal synthetic artifacts
    arm_stub = {
        "orig_words": 50,
        "added_private_words": 20,
        "added_public_words": 20,
        "total_private_words": 70,
        "total_public_words": 70,
        "added_word_rel_diff": 0.0,
        "total_word_rel_diff": 0.0,
        "baseline_disclosure_count": 0,
        "after_private_disclosure_count": 7,
        "after_public_disclosure_count": 0,
        "baseline_guardrail_id_count": 0,
        "after_private_guardrail_id_count": 0,
        "after_public_guardrail_id_count": 0,
        "n_omitted_items_added": 7,
        "public_echo_zero_leak": True,
        "private_info_complete": True,
        "word_parity_ok": True,
        "vendor_leaks": [],
        "r1_unedited": "u",
        "r1_private_info": "p",
        "r1_public_echo": "e",
        "added_private_info": "ap",
        "added_public_echo": "ae",
    }
    (run_dir / "exp2_info_flip" / "pairs.json").write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "p1",
                        "arm": "main",
                        "task": "goalpost",
                        "target_role": "A",
                        "seed_1": 301,
                        "seed_2": 301,
                        "same_seed": True,
                        "partner_1": "astra62",
                        "partner_2": "kimi31",
                        "partner_1_family": "openai",
                        "partner_2_family": "moonshot",
                        "teammate_1": "opus55",
                        "teammate_2": "neon4",
                        "ep_1_id": ep_ok["episode_id"],
                        "ep_2_id": ep_ok["episode_id"],
                        "score_1": 1.0,
                        "score_2": 1.0,
                        "handle_p1": "AL",
                        "mate_handle_p1": "BK",
                        "handle_p2": "EN",
                        "mate_handle_p2": "KV",
                        "outcome_card_1": "ok1",
                        "outcome_card_2": "ok2",
                        "arms_1": arm_stub,
                        "arms_2": arm_stub,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "exp2_info_flip" / "choices.jsonl").write_text(
        json.dumps(
            {
                "packet_id": "pkt1",
                "pair_id": "p1",
                "arm": "main",
                "task": "goalpost",
                "target_role": "A",
                "seed_1": 301,
                "seed_2": 301,
                "condition": "p1_info_p2_echo",
                "order": "AB",
                "reviewer": "neon4",
                "handle_1": "AL",
                "handle_2": "EN",
                "canonical_1": "P1",
                "canonical_2": "P2",
                "partner_1": "astra62",
                "partner_2": "kimi31",
                "variant_1": "private_info",
                "variant_2": "public_echo",
                "score_1": 1.0,
                "score_2": 1.0,
                "episode_1_id": ep_ok["episode_id"],
                "episode_2_id": ep_ok["episode_id"],
                "chosen_slot": 1,
                "chosen_canonical": "P1",
                "chosen_partner": "astra62",
                "chosen_variant": "private_info",
                "cross_vendor": True,
                "raw": '{"choice": "AL", "reason": "Shared all formulas."}',
                "prompt_tokens": 200,
                "completion_tokens": 40,
                "reasoning_tokens": 10,
                "cached_tokens": 0,
                "cost_usd": 0.003,
                "latency_ms": 800,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "exp2_info_flip" / "ledger.jsonl").write_text(
        json.dumps(
            {
                "call_id": "exp2_info_flip:pkt1:neon4",
                "status": "ok",
                "provider": "Google AI Studio",
                "returned_model": "google/gemini-4-neon",
                "finish_reason": "stop",
                "metadata": {"packet_id": "pkt1", "reviewer": "neon4"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Frozen manifests (include local absolute path in probe_manifest_v2_1.json to test normalization)
    for mf in ("manifest_v2.json", "probe_manifest_v2.json"):
        (run_dir / mf).write_text(json.dumps({"name": mf}), encoding="utf-8")
    (run_dir / "probe_manifest_v2_1.json").write_text(
        json.dumps(
            {
                "name": "probe_manifest_v2_1.json",
                "source_v2_manifest": f"/Users/localdev/repo/data/{run_name}/probe_manifest_v2.json",
            }
        ),
        encoding="utf-8",
    )

    return run_dir


def test_export_purges_stale_strips_deprecated_and_uses_zstd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export_mod, "DATA_DIR", tmp_path)
    run_dir = _build_synthetic_v2_1_run(tmp_path)

    hf_dir = export_mod.export_huggingface_dataset("synth_v2_1")

    # 1. Stale artifacts purged inside hf_export/, external sentinel untouched
    assert not (hf_dir / "episodes.jsonl").exists()
    assert not (hf_dir / "human_poll_blinded.html").exists()
    assert not (hf_dir / "stale_table.parquet").exists()
    assert not (hf_dir / "stale_subdir").exists()
    assert (run_dir / "human_poll_answer_key.json").exists()
    assert list(hf_dir.rglob("*.jsonl")) == []
    assert list(hf_dir.rglob("*human_poll*")) == []

    # 2. All expected v2.1 tables exported with zstd compression
    expected_tables = {
        "episodes.parquet",
        "failed_episodes.parquet",
        "turns.parquet",
        "debriefs.parquet",
        "choices.parquet",
        "choices_excluded.parquet",
        "exp1_dropout_gate.parquet",
        "exp1_dropout.parquet",
        "exp1_dropout_attempts.parquet",
        "exp2_info_flip_pairs.parquet",
        "exp2_info_flip_choices.parquet",
    }
    actual_parquets = {p.name for p in hf_dir.glob("*.parquet")}
    assert actual_parquets == expected_tables
    for pq_file in hf_dir.glob("*.parquet"):
        pf = pq.ParquetFile(pq_file)
        assert pf.metadata.num_row_groups >= 1
        assert pf.metadata.row_group(0).column(0).compression == "ZSTD"

    # 3. Zero deprecated debrief forks/columns
    deb_df = pd.read_parquet(hf_dir / "debriefs.parquet")
    assert set(deb_df["fork"].unique()) == {"F1", "F1_retest", "F4"}
    assert "false_blame_json" not in deb_df.columns

    # 4. Turns ledger join (latency_s + ledger_joined)
    turns_df = pd.read_parquet(hf_dir / "turns.parquet")
    assert len(turns_df) == 2
    assert bool(turns_df["ledger_joined"].all())
    lat_by_role = dict(zip(turns_df["role"], turns_df["latency_s"]))
    assert lat_by_role["A"] == pytest.approx(2.345)
    assert lat_by_role["B"] == pytest.approx(1.5)

    # 5. Exp 1 deduplication (ok beats error) + raw_response join + Exp 2 ledger join
    e1_df = pd.read_parquet(hf_dir / "exp1_dropout.parquet")
    assert len(e1_df) == 1
    assert bool(e1_df.iloc[0]["ok"]) is True
    assert e1_df.iloc[0]["raw_response"] == '{"workers": 4}'
    assert e1_df.iloc[0]["disclosure_bin"] == "7_rules"

    e2c_df = pd.read_parquet(hf_dir / "exp2_info_flip_choices.parquet")
    assert len(e2c_df) == 1
    assert e2c_df.iloc[0]["provider"] == "Google AI Studio"
    assert e2c_df.iloc[0]["rationale"] == "Shared all formulas."

    # 6. Frozen manifests copied cleanly & local absolute path normalized (without mutating source)
    copied_manifests = sorted(p.name for p in (hf_dir / "manifests").glob("*.json"))
    assert copied_manifests == [
        "locked_receiver_roster.json",
        "manifest_v2.json",
        "probe_manifest_v2.json",
        "probe_manifest_v2_1.json",
    ]
    exported_pm21 = json.loads(
        (hf_dir / "manifests" / "probe_manifest_v2_1.json").read_text(encoding="utf-8")
    )
    assert exported_pm21["source_v2_manifest"] == "data/synth_v2_1/probe_manifest_v2.json"
    source_pm21 = json.loads((run_dir / "probe_manifest_v2_1.json").read_text(encoding="utf-8"))
    assert source_pm21["source_v2_manifest"].startswith("/Users/localdev/")

    # 7. Hygiene verification passes cleanly
    rep_mod = _load_replay_module()
    hyg = rep_mod._verify_hygiene(hf_dir)
    assert hyg["hygiene_ok"] is True
    assert set(hyg["yaml_configs"]) == {p.removesuffix(".parquet") for p in expected_tables}


def test_export_and_replay_reject_secrets_and_deprecated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export_mod, "DATA_DIR", tmp_path)
    run_dir = _build_synthetic_v2_1_run(tmp_path, "secret_test")

    # Secret inside manifest must abort export
    (run_dir / "probe_manifest_v2.json").write_text(
        json.dumps({"oops": "sk-or-v1-deadbeef12345"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Secret token detected"):
        export_mod.export_huggingface_dataset("secret_test")

    # Restore clean manifest, then inject API secret into episode turn message (pandas 3.x StringDtype!)
    (run_dir / "probe_manifest_v2.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    ep_path = run_dir / "episodes" / "goalpost__s301__astra62__vs__opus55.json"
    orig_ep_txt = ep_path.read_text(encoding="utf-8")
    bad_ep = json.loads(orig_ep_txt)
    bad_ep["turns"][0]["A"]["message"] = "Leaked key sk-or-v1-deadbeef1234567890 inside turn"
    ep_path.write_text(json.dumps(bad_ep), encoding="utf-8")
    with pytest.raises(ValueError, match="Secret token detected in table 'turns', column 'message'"):
        export_mod.export_huggingface_dataset("secret_test")
    ep_path.write_text(orig_ep_txt, encoding="utf-8")

    # Clean data -> export succeeds, then inject hygiene violations for replay verifier to catch
    hf_dir = export_mod.export_huggingface_dataset("secret_test")
    rep_mod = _load_replay_module()

    # Violation 0: secret leaked inside a Parquet StringDtype ('str') column
    turns_df = pd.read_parquet(hf_dir / "turns.parquet")
    assert pd.api.types.is_string_dtype(turns_df["message"].dtype)
    bad_turns = turns_df.copy()
    bad_turns.loc[0, "message"] = "Secret sk-or-v1-deadbeef1234567890 in parquet"
    bad_turns.to_parquet(hf_dir / "turns.parquet", index=False, compression="zstd")
    with pytest.raises(AssertionError, match="Secret key leaked in turns.parquet column 'message'"):
        rep_mod._verify_hygiene(hf_dir)
    turns_df.to_parquet(hf_dir / "turns.parquet", index=False, compression="zstd")

    # Violation 0b: non-ZSTD Parquet compression (uncompressed and snappy must both be rejected)
    turns_df.to_parquet(hf_dir / "turns.parquet", index=False, compression=None)
    with pytest.raises(AssertionError, match="instead of 'ZSTD'"):
        rep_mod._verify_hygiene(hf_dir)
    turns_df.to_parquet(hf_dir / "turns.parquet", index=False, compression="snappy")
    with pytest.raises(AssertionError, match="instead of 'ZSTD'"):
        rep_mod._verify_hygiene(hf_dir)
    turns_df.to_parquet(hf_dir / "turns.parquet", index=False, compression="zstd")

    # Violation 1: stale .jsonl file
    (hf_dir / "dup.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Duplicate .jsonl"):
        rep_mod._verify_hygiene(hf_dir)
    (hf_dir / "dup.jsonl").unlink()

    # Violation 2: deprecated human_poll file
    (hf_dir / "human_poll_blinded.json").write_text("[]", encoding="utf-8")
    with pytest.raises(AssertionError, match="Deprecated human poll"):
        rep_mod._verify_hygiene(hf_dir)
    (hf_dir / "human_poll_blinded.json").unlink()

    # Violation 3: em-dash or F2 mention in README.md
    orig_readme = (hf_dir / "README.md").read_text(encoding="utf-8")
    (hf_dir / "README.md").write_text(orig_readme + "\nNote — F2 test\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Forbidden pattern"):
        rep_mod._verify_hygiene(hf_dir)
    (hf_dir / "README.md").write_text(orig_readme, encoding="utf-8")

    # Violation 4: F2 row or false_blame_json column inside debriefs.parquet
    deb_df = pd.read_parquet(hf_dir / "debriefs.parquet")
    bad_deb = deb_df.copy()
    bad_deb["false_blame_json"] = None
    bad_deb.to_parquet(hf_dir / "debriefs.parquet", index=False, compression="zstd")
    with pytest.raises(AssertionError, match="false_blame_json"):
        rep_mod._verify_hygiene(hf_dir)

    bad_deb2 = deb_df.copy()
    bad_deb2.loc[0, "fork"] = "F2"
    bad_deb2.to_parquet(hf_dir / "debriefs.parquet", index=False, compression="zstd")
    with pytest.raises(AssertionError, match="Deprecated debrief forks"):
        rep_mod._verify_hygiene(hf_dir)


@pytest.mark.skipif(
    not (DATA_DIR / "v2" / "hf_export" / "exp1_dropout.parquet").exists(),
    reason="Live data/v2/hf_export v2.1 dataset not present",
)
def test_live_v2_hf_export_full_independent_replay() -> None:
    """Full live integration gate: every headline v2.1 claim reproduces strictly from data/v2/hf_export/."""
    rep_mod = _load_replay_module()
    res = rep_mod.verify_hf_replay("v2", capstone_run="v2_capstone")
    assert res["ok"] is True
    assert res["hygiene"]["hygiene_ok"] is True
    assert res["counts"]["episodes_total"] == 702
    assert res["counts"]["episodes_main"] == 638
    assert res["counts"]["episodes_opus_boost"] == 64
    assert res["counts"]["failed_episodes"] == 2
    assert res["counts"]["turns_main"] == 9568
    assert res["counts"]["turns_total"] == 10528
    assert res["counts"]["turns_ledger_joined"] == 10528
    assert res["counts"]["choices_total"] == 3178
    assert res["counts"]["choices_excluded"] == 22
    assert res["counts"]["capstone_choices"] == 384
    assert res["counts"]["capstone_rewrites_candidates"] == 48
    assert res["counts"]["exp1_dropout_gate"] == 256
    assert res["counts"]["exp1_dropout"] == 2560
    assert res["counts"]["exp2_info_flip_pairs"] == 30
    assert res["counts"]["exp2_info_flip_choices"] == 480
    assert res["bt_reproduced"] is True
    assert res["h1b_reproduced"] is True
    assert res["h1b_opus_goalpost_predecessor_win_rate"] == pytest.approx(0.875, abs=1e-3)
    assert res["f1_retest_reproduced"] is True
    assert res["f1_retest"]["icc_2_1"] == pytest.approx(0.8688, abs=1e-3)
    assert res["meta_v2_1_reproduced"] is True
    assert res["capstone_reproduced"] is True
    assert res["exp1_reproduced"] is True
    assert res["exp1_dropout"]["gate_deepseek_flash_zero_reasoning"] == "32/32"
    assert res["exp1_dropout"]["matrix_deepseek_flash_zero_reasoning"] == "320/320"
    assert res["exp2_reproduced"] is True
    for mf in (DATA_DIR / "v2" / "hf_export" / "manifests").glob("*.json"):
        assert "/Users/" not in mf.read_text(encoding="utf-8")
