"""Export lean, key-scrubbed, zstd-compressed HuggingFace Parquet dataset for v2.1."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_ground.config import AA_INDEX_V5, DATA_DIR, DEFAULT_MODELS
from common_ground.dropout import _load_existing_results_jsonl, build_dropout_cases
from common_ground.meta_analysis import _ACTION_LINE_RE, _load_turn_ledger_index

# Strictly retained counterfactual debrief forks (excludes deprecated F2, F5, F3a, F3b)
EXPORTED_DEBRIEF_FORKS: tuple[str, ...] = ("F1", "F1_retest", "F4")

_SECRET_PATTERN = re.compile(r"sk-or-v1-[A-Za-z0-9_-]*")


def _is_text_like_dtype(dtype: Any) -> bool:
    """Return True for any pandas 2.x/3.x dtype that can hold strings (including StringDtype 'str')."""
    return (
        isinstance(dtype, (pd.StringDtype, pd.CategoricalDtype))
        or pd.api.types.is_string_dtype(dtype)
        or dtype == object
    )


def _slug_for(model_key: str) -> str:
    spec = DEFAULT_MODELS.get(model_key)
    return spec.slug if spec is not None else model_key


def _assert_no_secrets_text(text: str, context: str) -> None:
    if _SECRET_PATTERN.search(text):
        raise ValueError(f"Secret token detected while exporting {context}; aborting export.")


def _extract_rationale(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj.get("reason") is not None:
                return str(obj["reason"]).strip()
        except json.JSONDecodeError:
            pass
    return s


def _disc_bin(n_f: int) -> str:
    if n_f == 0:
        return "0_rules"
    if n_f <= 4:
        return "1_to_4_rules"
    if n_f <= 6:
        return "5_to_6_rules"
    return "7_rules"


def _load_exp1_ledger_index(
    ledger_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load Exp 1 ledger rows + call_id indexes (preferring status=='ok', fallback to latest)."""
    rows: list[dict[str, Any]] = []
    ok_by_cid: dict[str, dict[str, Any]] = {}
    last_by_cid: dict[str, dict[str, Any]] = {}
    if not ledger_path.exists():
        return rows, ok_by_cid, last_by_cid
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            cid = str(row.get("call_id") or "")
            if cid:
                last_by_cid[cid] = row
                if row.get("status") == "ok":
                    ok_by_cid[cid] = row
    return rows, ok_by_cid, last_by_cid


def _build_exp1_tables(exp1_dir: Path) -> dict[str, pd.DataFrame]:
    """Build exp1_dropout_gate, exp1_dropout, and exp1_dropout_attempts DataFrames."""
    out: dict[str, pd.DataFrame] = {}
    ledger_rows, ok_by_cid, last_by_cid = _load_exp1_ledger_index(exp1_dir / "ledger.jsonl")

    # 1. Calibration gate table
    gate_path = exp1_dir / "gate_report.json"
    if gate_path.exists():
        gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_rows: list[dict[str, Any]] = []
        for rk, rdata in (gate_report.get("receivers") or {}).items():
            history = rdata.get("step_up_history") or []
            n_steps = len(history)
            for idx, step in enumerate(history):
                step_index = int(step.get("step_index", idx))
                is_final_step = idx == n_steps - 1
                passed_gate = bool(step.get("passed_gate"))
                for rec in step.get("records") or []:
                    cid = str(rec.get("call_id") or "")
                    lr = ok_by_cid.get(cid) or last_by_cid.get(cid)
                    lat_ms = rec.get("latency_ms")
                    lat_s = float(lat_ms) / 1000.0 if lat_ms is not None else None
                    gate_rows.append(
                        {
                            "call_id": cid,
                            "step_index": step_index,
                            "is_final_step": is_final_step,
                            "step_passed_gate": passed_gate,
                            "case_id": rec.get("case_id"),
                            "episode_id": rec.get("episode_id"),
                            "seed": rec.get("seed"),
                            "arm": rec.get("arm"),
                            "dropped_role": rec.get("dropped_role"),
                            "receiver_role": rec.get("receiver_role"),
                            "sender_model": rec.get("sender_model"),
                            "receiver": str(rec.get("receiver_key") or rk),
                            "receiver_slug": rec.get("receiver_slug"),
                            "receiver_family": rec.get("receiver_family"),
                            "receiver_tier": rec.get("receiver_tier"),
                            "receiver_effort": rec.get("receiver_effort"),
                            "receiver_aa_index_v5": rec.get("receiver_aa_index_v5"),
                            "condition": rec.get("condition"),
                            "ok": bool(rec.get("status") == "ok"),
                            "status": rec.get("status"),
                            "finish_reason": rec.get("finish_reason"),
                            "error_detail": rec.get("error_detail"),
                            "parse_ok": bool(rec.get("parse_ok")),
                            "domain_ok": bool(rec.get("domain_ok")),
                            "own_r3_ok": bool(rec.get("own_plus_r3_pass")),
                            "partner_7_ok": bool(rec.get("partner_rules_pass")),
                            "strict_15_ok": bool(rec.get("strict_all_pass")),
                            "own_rules_passed": rec.get("own_rules_passed"),
                            "r3_passed": bool(rec.get("r3_passed")),
                            "partner_rules_passed": rec.get("partner_rules_passed"),
                            "total_rules_passed": rec.get("total_rules_passed"),
                            "prompt_tokens": rec.get("prompt_tokens"),
                            "completion_tokens": rec.get("completion_tokens"),
                            "reasoning_tokens": rec.get("reasoning_tokens"),
                            "cached_tokens": rec.get("cached_tokens"),
                            "latency_s": lat_s,
                            "cost_usd": rec.get("cost_usd"),
                            "provider": rec.get("provider"),
                            "returned_model": rec.get("returned_model"),
                            "parsed_config_json": json.dumps(
                                rec.get("parsed_config"), ensure_ascii=False
                            ),
                            "rule_results_json": json.dumps(
                                rec.get("rule_results"), ensure_ascii=False
                            ),
                            "raw_response": lr.get("response_text") if lr is not None else None,
                        }
                    )
        if gate_rows:
            out["exp1_dropout_gate"] = pd.DataFrame(gate_rows)

    # 2. Main handoff matrix table (deduplicated identically to dropout._load_existing_results_jsonl)
    results_path = exp1_dir / "results.jsonl"
    if results_path.exists():
        dedup_map = _load_existing_results_jsonl(results_path)
        dedup_recs = [dict(dedup_map[k]) for k in sorted(dedup_map.keys())]
        # Recompute deterministic disclosure metadata from current disclosure parser if full episodes exist
        ep_dir = exp1_dir.parent / "episodes"
        if ep_dir.is_dir():
            try:
                cases_by_id = {c.case_id: c for c in build_dropout_cases(ep_dir)}
            except (FileNotFoundError, ValueError):
                cases_by_id = {}
            for rec in dedup_recs:
                c_obj = cases_by_id.get(str(rec.get("case_id") or ""))
                if c_obj is not None:
                    rec["sender_r1_n_formula"] = c_obj.sender_r1_n_formula
                    rec["sender_r1_full_disclosure"] = c_obj.sender_r1_full_disclosure
                    rec["sender_r1_r2_n_formula"] = c_obj.sender_r1_r2_n_formula
                    rec["sender_r1_r2_full_disclosure"] = c_obj.sender_r1_r2_full_disclosure
                    rec["sender_r1_n_id"] = c_obj.sender_r1_n_id
                    rec["sender_r1_r2_n_id"] = c_obj.sender_r1_r2_n_id
        full_disc_wcs = [
            int(r["sender_r1_r2_word_count"])
            for r in dedup_recs
            if r.get("sender_r1_r2_full_disclosure") and r.get("sender_r1_r2_word_count") is not None
        ]
        median_bloat_wc = float(np.median(full_disc_wcs)) if full_disc_wcs else 0.0

        matrix_rows: list[dict[str, Any]] = []
        for rec in dedup_recs:
            cid = str(rec.get("call_id") or "")
            lr = ok_by_cid.get(cid) or last_by_cid.get(cid)
            lat_ms = rec.get("latency_ms")
            lat_s = float(lat_ms) / 1000.0 if lat_ms is not None else None
            r12_full = bool(rec.get("sender_r1_r2_full_disclosure"))
            r12_wc = int(rec.get("sender_r1_r2_word_count") or 0)
            if not r12_full:
                bloat_bin = "withheld"
            elif r12_wc > median_bloat_wc:
                bloat_bin = "high_bloat"
            else:
                bloat_bin = "low_bloat"

            matrix_rows.append(
                {
                    "call_id": cid,
                    "condition": rec.get("condition"),
                    "case_id": rec.get("case_id"),
                    "episode_id": rec.get("episode_id"),
                    "seed": rec.get("seed"),
                    "arm": rec.get("arm"),
                    "dropped_role": rec.get("dropped_role"),
                    "receiver_role": rec.get("receiver_role"),
                    "is_primary_unconfounded": bool(rec.get("is_primary_unconfounded")),
                    "sender_model": rec.get("sender_model"),
                    "sender_family": rec.get("sender_family"),
                    "original_receiver_model": rec.get("original_receiver_model"),
                    "original_receiver_family": rec.get("original_receiver_family"),
                    "receiver_key": rec.get("receiver_key"),
                    "receiver_slug": rec.get("receiver_slug"),
                    "receiver_family": rec.get("receiver_family"),
                    "receiver_display_name": rec.get("receiver_display_name"),
                    "receiver_tier": rec.get("receiver_tier"),
                    "receiver_effort": rec.get("receiver_effort"),
                    "receiver_aa_index_v5": rec.get("receiver_aa_index_v5"),
                    "is_same_family_as_sender": bool(rec.get("is_same_family_as_sender")),
                    "is_cross_vendor": bool(rec.get("is_cross_vendor")),
                    "is_same_model_as_original_receiver": bool(
                        rec.get("is_same_model_as_original_receiver")
                    ),
                    "episode_score": rec.get("episode_score"),
                    "r2_all_pass": bool(rec.get("r2_all_pass")),
                    "sender_r1_n_formula": rec.get("sender_r1_n_formula"),
                    "sender_r1_full_disclosure": bool(rec.get("sender_r1_full_disclosure")),
                    "sender_r1_r2_n_formula": rec.get("sender_r1_r2_n_formula"),
                    "sender_r1_r2_full_disclosure": r12_full,
                    "disclosure_bin": _disc_bin(int(rec.get("sender_r1_r2_n_formula") or 0)),
                    "bloat_bin": bloat_bin,
                    "sender_r1_n_id": rec.get("sender_r1_n_id"),
                    "sender_r1_r2_n_id": rec.get("sender_r1_r2_n_id"),
                    "sender_r1_word_count": rec.get("sender_r1_word_count"),
                    "sender_r1_r2_word_count": r12_wc,
                    "original_receiver_asked_rules_r1": bool(
                        rec.get("original_receiver_asked_rules_r1")
                    ),
                    "original_receiver_asked_rules": bool(
                        rec.get("original_receiver_asked_rules")
                    ),
                    "ok": bool(rec.get("status") == "ok"),
                    "status": rec.get("status"),
                    "error_detail": rec.get("error_detail"),
                    "returned_model": rec.get("returned_model"),
                    "provider": rec.get("provider"),
                    "finish_reason": rec.get("finish_reason"),
                    "prompt_tokens": rec.get("prompt_tokens"),
                    "completion_tokens": rec.get("completion_tokens"),
                    "reasoning_tokens": rec.get("reasoning_tokens"),
                    "cached_tokens": rec.get("cached_tokens"),
                    "cost_usd": rec.get("cost_usd"),
                    "latency_s": lat_s,
                    "parse_ok": bool(rec.get("parse_ok")),
                    "domain_ok": bool(rec.get("domain_ok")),
                    "normalized_keys": bool(rec.get("normalized_keys")),
                    "parse_error": rec.get("parse_error"),
                    "own_rules_passed": rec.get("own_rules_passed"),
                    "r3_passed": bool(rec.get("r3_passed")),
                    "own_r3_ok": bool(rec.get("own_plus_r3_pass")),
                    "own_plus_r3_frac": rec.get("own_plus_r3_frac"),
                    "partner_rules_passed": rec.get("partner_rules_passed"),
                    "partner_7_ok": bool(rec.get("partner_rules_pass")),
                    "partner_rules_frac": rec.get("partner_rules_frac"),
                    "strict_15_ok": bool(rec.get("strict_all_pass")),
                    "total_rules_passed": rec.get("total_rules_passed"),
                    "copied_r2_unchanged": bool(rec.get("copied_r2_unchanged")),
                    "hamming_from_r2": rec.get("hamming_from_r2"),
                    "disclosed_partner_rules_passed": rec.get("disclosed_partner_rules_passed"),
                    "disclosed_partner_rules_total": rec.get("disclosed_partner_rules_total"),
                    "undisclosed_partner_rules_passed": rec.get("undisclosed_partner_rules_passed"),
                    "undisclosed_partner_rules_total": rec.get("undisclosed_partner_rules_total"),
                    "parsed_config_json": json.dumps(rec.get("parsed_config"), ensure_ascii=False),
                    "missing_keys_json": json.dumps(rec.get("missing_keys"), ensure_ascii=False),
                    "out_of_domain_keys_json": json.dumps(
                        rec.get("out_of_domain_keys"), ensure_ascii=False
                    ),
                    "rule_results_json": json.dumps(rec.get("rule_results"), ensure_ascii=False),
                    "raw_response": lr.get("response_text") if lr is not None else None,
                }
            )
        if matrix_rows:
            out["exp1_dropout"] = pd.DataFrame(matrix_rows)

    # 3. Attempt-level API telemetry table (scrubbed of prompts/raw reasoning)
    if ledger_rows:
        attempt_rows: list[dict[str, Any]] = []
        for lr in ledger_rows:
            meta = lr.get("metadata") or {}
            lat_ms = lr.get("latency_ms")
            lat_s = float(lat_ms) / 1000.0 if lat_ms is not None else None
            attempt_rows.append(
                {
                    "call_id": lr.get("call_id"),
                    "timestamp_utc": lr.get("timestamp_utc"),
                    "stage": meta.get("stage"),
                    "condition": meta.get("condition"),
                    "case_id": meta.get("case_id"),
                    "episode_id": meta.get("episode_id"),
                    "seed": meta.get("seed"),
                    "dropped_role": meta.get("dropped_role"),
                    "receiver_role": meta.get("receiver_role"),
                    "sender_model": meta.get("sender_model"),
                    "receiver": lr.get("model_key"),
                    "receiver_tier": meta.get("receiver_tier"),
                    "receiver_effort": meta.get("effort") or lr.get("reasoning_effort"),
                    "requested_slug": lr.get("requested_slug"),
                    "returned_model": lr.get("returned_model"),
                    "provider": lr.get("provider"),
                    "attempt": lr.get("attempt"),
                    "ok": bool(lr.get("status") == "ok"),
                    "status": lr.get("status"),
                    "finish_reason": lr.get("finish_reason"),
                    "error": lr.get("error_detail"),
                    "max_tokens": lr.get("max_tokens"),
                    "prompt_tokens": lr.get("prompt_tokens"),
                    "completion_tokens": lr.get("completion_tokens"),
                    "reasoning_tokens": lr.get("reasoning_tokens"),
                    "cached_tokens": lr.get("cached_tokens"),
                    "latency_s": lat_s,
                    "cost_usd": lr.get("cost_usd"),
                }
            )
        out["exp1_dropout_attempts"] = pd.DataFrame(attempt_rows)

    return out


def _build_exp2_tables(exp2_dir: Path) -> dict[str, pd.DataFrame]:
    """Build exp2_info_flip_pairs and exp2_info_flip_choices DataFrames."""
    out: dict[str, pd.DataFrame] = {}

    # 1. Pairs table
    pairs_path = exp2_dir / "pairs.json"
    if pairs_path.exists():
        pdata = json.loads(pairs_path.read_text(encoding="utf-8"))
        pairs_list = pdata.get("pairs") if isinstance(pdata, dict) else pdata
        pair_rows: list[dict[str, Any]] = []
        for p in pairs_list or []:
            a1 = p.get("arms_1") or {}
            a2 = p.get("arms_2") or {}
            v1_leaks = list(a1.get("vendor_leaks") or [])
            v2_leaks = list(a2.get("vendor_leaks") or [])
            row: dict[str, Any] = {
                "pair_id": p.get("pair_id"),
                "arm": p.get("arm"),
                "is_placebo": bool(p.get("arm") == "placebo"),
                "task": p.get("task"),
                "role": p.get("target_role"),
                "seed_1": p.get("seed_1"),
                "seed_2": p.get("seed_2"),
                "same_seed": bool(p.get("same_seed")),
                "cand_a_episode_id": p.get("ep_1_id"),
                "cand_a_partner": p.get("partner_1"),
                "cand_a_partner_family": p.get("partner_1_family"),
                "cand_a_teammate": p.get("teammate_1"),
                "cand_a_score": p.get("score_1"),
                "cand_a_handle": p.get("handle_p1"),
                "cand_a_mate_handle": p.get("mate_handle_p1"),
                "cand_a_outcome_card": p.get("outcome_card_1"),
                "cand_b_episode_id": p.get("ep_2_id"),
                "cand_b_partner": p.get("partner_2"),
                "cand_b_partner_family": p.get("partner_2_family"),
                "cand_b_teammate": p.get("teammate_2"),
                "cand_b_score": p.get("score_2"),
                "cand_b_handle": p.get("handle_p2"),
                "cand_b_mate_handle": p.get("mate_handle_p2"),
                "cand_b_outcome_card": p.get("outcome_card_2"),
                "max_added_word_rel_diff": max(
                    float(a1.get("added_word_rel_diff") or 0.0),
                    float(a2.get("added_word_rel_diff") or 0.0),
                ),
                "max_total_word_rel_diff": max(
                    float(a1.get("total_word_rel_diff") or 0.0),
                    float(a2.get("total_word_rel_diff") or 0.0),
                ),
                "all_word_parity_ok": bool(a1.get("word_parity_ok")) and bool(a2.get("word_parity_ok")),
                "all_public_echo_zero_leak": bool(a1.get("public_echo_zero_leak"))
                and bool(a2.get("public_echo_zero_leak")),
                "all_private_info_complete": bool(a1.get("private_info_complete"))
                and bool(a2.get("private_info_complete")),
                "vendor_leak_count": len(v1_leaks) + len(v2_leaks),
            }
            for prefix, arm_obj in (("cand_a", a1), ("cand_b", a2)):
                row[f"{prefix}_orig_words"] = arm_obj.get("orig_words")
                row[f"{prefix}_added_private_words"] = arm_obj.get("added_private_words")
                row[f"{prefix}_added_public_words"] = arm_obj.get("added_public_words")
                row[f"{prefix}_total_private_words"] = arm_obj.get("total_private_words")
                row[f"{prefix}_total_public_words"] = arm_obj.get("total_public_words")
                row[f"{prefix}_added_word_rel_diff"] = arm_obj.get("added_word_rel_diff")
                row[f"{prefix}_total_word_rel_diff"] = arm_obj.get("total_word_rel_diff")
                row[f"{prefix}_baseline_disclosure_count"] = arm_obj.get("baseline_disclosure_count")
                row[f"{prefix}_after_private_disclosure_count"] = arm_obj.get(
                    "after_private_disclosure_count"
                )
                row[f"{prefix}_after_public_disclosure_count"] = arm_obj.get(
                    "after_public_disclosure_count"
                )
                row[f"{prefix}_baseline_guardrail_id_count"] = arm_obj.get(
                    "baseline_guardrail_id_count"
                )
                row[f"{prefix}_after_private_guardrail_id_count"] = arm_obj.get(
                    "after_private_guardrail_id_count"
                )
                row[f"{prefix}_after_public_guardrail_id_count"] = arm_obj.get(
                    "after_public_guardrail_id_count"
                )
                row[f"{prefix}_n_omitted_items_added"] = arm_obj.get("n_omitted_items_added")
                row[f"{prefix}_public_echo_zero_leak"] = bool(arm_obj.get("public_echo_zero_leak"))
                row[f"{prefix}_private_info_complete"] = bool(arm_obj.get("private_info_complete"))
                row[f"{prefix}_word_parity_ok"] = bool(arm_obj.get("word_parity_ok"))
                row[f"{prefix}_vendor_leaks_json"] = json.dumps(
                    arm_obj.get("vendor_leaks") or [], ensure_ascii=False
                )
                row[f"{prefix}_r1_unedited"] = arm_obj.get("r1_unedited")
                row[f"{prefix}_r1_private_info"] = arm_obj.get("r1_private_info")
                row[f"{prefix}_r1_public_echo"] = arm_obj.get("r1_public_echo")
                row[f"{prefix}_added_private_info"] = arm_obj.get("added_private_info")
                row[f"{prefix}_added_public_echo"] = arm_obj.get("added_public_echo")
            pair_rows.append(row)
        if pair_rows:
            out["exp2_info_flip_pairs"] = pd.DataFrame(pair_rows)

    # 2. Choices table (joined with Exp 2 ledger by (packet_id, reviewer) for provider/returned_model)
    choices_path = exp2_dir / "choices.jsonl"
    if choices_path.exists():
        led_ok: dict[tuple[str, str], dict[str, Any]] = {}
        led_path = exp2_dir / "ledger.jsonl"
        if led_path.exists():
            with led_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    lr = json.loads(line)
                    if lr.get("status") != "ok":
                        continue
                    meta = lr.get("metadata") or {}
                    pid = str(meta.get("packet_id") or "")
                    rev = str(meta.get("reviewer") or lr.get("model_key") or "")
                    if pid and rev:
                        led_ok[(pid, rev)] = lr

        dedup_choices: dict[tuple[str, str], dict[str, Any]] = {}
        for line in choices_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            dedup_choices[(str(rec["packet_id"]), str(rec["reviewer"]))] = rec

        choice_rows: list[dict[str, Any]] = []
        for (pid, rev), rec in sorted(dedup_choices.items()):
            lr = led_ok.get((pid, rev))
            lat_ms = rec.get("latency_ms")
            lat_s = float(lat_ms) / 1000.0 if lat_ms is not None else None
            chosen_slot = rec.get("chosen_slot")
            choice_rows.append(
                {
                    "packet_id": pid,
                    "pair_id": rec.get("pair_id"),
                    "arm": rec.get("arm"),
                    "is_placebo": bool(rec.get("arm") == "placebo"),
                    "task": rec.get("task"),
                    "target_role": rec.get("target_role"),
                    "seed_1": rec.get("seed_1"),
                    "seed_2": rec.get("seed_2"),
                    "condition": rec.get("condition"),
                    "order": rec.get("order"),
                    "reviewer": rev,
                    "handle_1": rec.get("handle_1"),
                    "handle_2": rec.get("handle_2"),
                    "canonical_1": rec.get("canonical_1"),
                    "canonical_2": rec.get("canonical_2"),
                    "partner_1": rec.get("partner_1"),
                    "partner_2": rec.get("partner_2"),
                    "variant_1": rec.get("variant_1"),
                    "variant_2": rec.get("variant_2"),
                    "score_1": rec.get("score_1"),
                    "score_2": rec.get("score_2"),
                    "episode_1_id": rec.get("episode_1_id"),
                    "episode_2_id": rec.get("episode_2_id"),
                    "chosen_slot": chosen_slot,
                    "chosen_canonical": rec.get("chosen_canonical"),
                    "chosen_partner": rec.get("chosen_partner"),
                    "chosen_variant": rec.get("chosen_variant"),
                    "cross_vendor": bool(rec.get("cross_vendor")),
                    "ok": bool(chosen_slot in (1, 2)),
                    "rationale": _extract_rationale(rec.get("raw")),
                    "raw": rec.get("raw"),
                    "provider": lr.get("provider") if lr is not None else None,
                    "returned_model": lr.get("returned_model") if lr is not None else None,
                    "finish_reason": lr.get("finish_reason") if lr is not None else None,
                    "prompt_tokens": rec.get("prompt_tokens"),
                    "completion_tokens": rec.get("completion_tokens"),
                    "reasoning_tokens": rec.get("reasoning_tokens"),
                    "cached_tokens": rec.get("cached_tokens"),
                    "latency_s": lat_s,
                    "cost_usd": rec.get("cost_usd"),
                }
            )
        if choice_rows:
            out["exp2_info_flip_choices"] = pd.DataFrame(choice_rows)

    return out


def _copy_frozen_manifests(run_dir: Path, out_dir: Path) -> list[str]:
    """Copy key-scrubbed frozen manifests into hf_export/manifests/."""
    manifest_sources: list[tuple[Path, str]] = [
        (run_dir / "manifest_v2.json", "manifest_v2.json"),
        (run_dir / "probe_manifest_v2.json", "probe_manifest_v2.json"),
        (run_dir / "probe_manifest_v2_1.json", "probe_manifest_v2_1.json"),
        (run_dir / "exp1_dropout" / "locked_receiver_roster.json", "locked_receiver_roster.json"),
    ]
    copied: list[str] = []
    manifests_dir = out_dir / "manifests"
    for src, dst_name in manifest_sources:
        if not src.exists():
            continue
        raw_txt = src.read_text(encoding="utf-8")
        _assert_no_secrets_text(raw_txt, f"manifest {src}")
        # Validate JSON cleanly before writing
        parsed = json.loads(raw_txt)
        modified = False
        # Normalize local absolute path in probe_manifest_v2_1.json without mutating source file
        if (
            dst_name == "probe_manifest_v2_1.json"
            and isinstance(parsed, dict)
            and isinstance(parsed.get("source_v2_manifest"), str)
        ):
            src_val = parsed["source_v2_manifest"]
            rel_suffix = f"data/{run_dir.name}/probe_manifest_v2.json"
            if src_val.endswith(rel_suffix) and src_val != rel_suffix:
                parsed["source_v2_manifest"] = rel_suffix
                modified = True
        manifests_dir.mkdir(parents=True, exist_ok=True)
        if modified:
            clean_txt = json.dumps(parsed, indent=2, ensure_ascii=False)
            _assert_no_secrets_text(clean_txt, f"manifest {dst_name}")
            (manifests_dir / dst_name).write_text(clean_txt + "\n", encoding="utf-8")
        else:
            (manifests_dir / dst_name).write_bytes(src.read_bytes())
        copied.append(f"manifests/{dst_name}")
    return copied


def export_huggingface_dataset(run_name: str) -> Path:
    """Cleanly purge and rebuild `data/<run_name>/hf_export/` with lean zstd Parquet tables."""
    run_dir = DATA_DIR / run_name
    out_dir = run_dir / "hf_export"

    # 1. Clean purge of stale artifacts inside hf_export/ only
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load raw episodes + turn ledger index (exact join logic from meta_analysis.py)
    all_ep_files = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((run_dir / "episodes").glob("*.json"))
    ]
    episodes = [ep for ep in all_ep_files if ep.get("complete") is True]
    failed_episodes = [ep for ep in all_ep_files if ep.get("complete") is not True]

    by_cid, by_key = _load_turn_ledger_index(run_dir / "ledger.jsonl")

    # 3. Load deduplicated choices + excluded choices
    choices_path = run_dir / "choices.jsonl"
    choices_map: dict[str, dict[str, Any]] = {}
    if choices_path.exists():
        for ln in choices_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                row = json.loads(ln)
                cid = (
                    row.get("manifest_item_id")
                    or f"{row.get('choice_type')}__{row.get('item_id')}__{row.get('order')}"
                )
                choices_map[cid] = row
    choices = list(choices_map.values())

    choices_excluded_rows: list[dict[str, Any]] = []
    choices_excl_path = run_dir / "choices_excluded.jsonl"
    if choices_excl_path.exists():
        for ln in choices_excl_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                choices_excluded_rows.append(json.loads(ln))

    # 4. Load optional Capstone Tone-Flip results & rewrites
    capstone_path = run_dir / "capstone_tone_flip" / "capstone_choices.jsonl"
    capstone_rows: list[dict[str, Any]] = []
    if capstone_path.exists():
        for ln in capstone_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                capstone_rows.append(json.loads(ln))

    capstone_rw_path = run_dir / "capstone_tone_flip" / "capstone_rewrites.jsonl"
    capstone_rw_rows: list[dict[str, Any]] = []
    if capstone_rw_path.exists():
        for ln in capstone_rw_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rec = json.loads(ln)
                rec_flat = {
                    k: v for k, v in rec.items() if k not in ("turns", "rewritten_by_round")
                }
                rec_flat["turns_json"] = json.dumps(rec.get("turns") or [], ensure_ascii=False)
                capstone_rw_rows.append(rec_flat)

    # 5. Assemble core tables (episodes, failed_episodes, turns, debriefs, choices)
    ep_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    debrief_rows: list[dict[str, Any]] = []

    for ep in episodes:
        hist_a = ep.get("history_a") or []
        hist_b = ep.get("history_b") or []
        briefing_a = ep.get("briefing_a") or (hist_a[0]["content"] if hist_a else None)
        briefing_b = ep.get("briefing_b") or (hist_b[0]["content"] if hist_b else None)
        apf = ep.get("action_parse_failures") or {"A": 0, "B": 0}
        ep_rows.append(
            {
                "episode_id": ep["episode_id"],
                "run_name": ep["run_name"],
                "task": ep["task"],
                "seed": ep["seed"],
                "arm": ep["arm"],
                "total_rounds": ep.get("total_rounds", len(ep.get("turns") or [])),
                "model_a": ep["model_a"],
                "model_a_slug": _slug_for(ep["model_a"]),
                "model_a_aa_v5": AA_INDEX_V5.get(ep["model_a"]),
                "model_b": ep["model_b"],
                "model_b_slug": _slug_for(ep["model_b"]),
                "model_b_aa_v5": AA_INDEX_V5.get(ep["model_b"]),
                "handle_a": ep["handle_a"],
                "handle_b": ep["handle_b"],
                "briefing_a": briefing_a,
                "briefing_b": briefing_b,
                "score": ep["score"],
                "score_details_json": json.dumps(ep["score_details"], ensure_ascii=False),
                "format_retries_a": ep["format_retries"]["A"],
                "format_retries_b": ep["format_retries"]["B"],
                "action_parse_failures_a": apf.get("A", 0),
                "action_parse_failures_b": apf.get("B", 0),
            }
        )
        ep_id = str(ep["episode_id"])
        for t in ep.get("turns") or []:
            rnd = int(t["round"])
            for role in ("A", "B"):
                s = t[role]
                jd = s.get("jev_demeanor") or {}
                panel_judges = jd.get("panel_judges")
                panel_consensus = jd.get("panel_consensus")
                lr = by_cid.get(str(s.get("accepted_call_id") or "")) or by_key.get(
                    (ep_id, rnd, role)
                )
                lat_ms = lr.get("latency_ms") if lr is not None else None
                latency_s = float(lat_ms) / 1000.0 if lat_ms is not None else None
                n_action_blocks = (
                    len(_ACTION_LINE_RE.findall(str(lr.get("response_text") or "")))
                    if lr is not None and lr.get("response_text") is not None
                    else None
                )
                turn_rows.append(
                    {
                        "episode_id": ep_id,
                        "task": ep["task"],
                        "seed": ep["seed"],
                        "arm": ep["arm"],
                        "round": rnd,
                        "role": role,
                        "model": s["model"],
                        "model_slug": _slug_for(s["model"]),
                        "handle": s["handle"],
                        "message": s["message"],
                        "action_json": json.dumps(s["action"], ensure_ascii=False),
                        "parse_error": s["parse_error"],
                        "feedback_a": t.get("feedback_a"),
                        "feedback_b": t.get("feedback_b"),
                        "round_metrics_json": json.dumps(t.get("round_metrics"), ensure_ascii=False),
                        "metrics_json": json.dumps(s["metrics"], ensure_ascii=False),
                        "panel_consensus_json": json.dumps(
                            panel_consensus if panel_consensus is not None else jd,
                            ensure_ascii=False,
                        ),
                        "panel_judges_raw_json": json.dumps(panel_judges, ensure_ascii=False),
                        "accepted_call_id": s.get("accepted_call_id"),
                        "accepted_attempt": s.get("accepted_attempt"),
                        "returned_model": s.get("returned_model"),
                        "provider": s.get("provider"),
                        "prompt_tokens": s["prompt_tokens"],
                        "completion_tokens": s["completion_tokens"],
                        "reasoning_tokens": s.get("reasoning_tokens"),
                        "cached_tokens": s["cached_tokens"],
                        "accepted_cost_usd": s.get("accepted_cost_usd", s.get("cost_usd")),
                        "total_cost_usd": s.get("total_cost_usd", s.get("cost_usd")),
                        "latency_s": latency_s,
                        "ledger_joined": bool(lr is not None),
                        "n_action_blocks": n_action_blocks,
                    }
                )
        for role, d in (ep.get("debriefs") or {}).items():
            # Strictly retain ONLY F1, F1_retest, and F4; exclude deprecated F2, F5, F3a, F3b
            # and strictly drop false_blame_json.
            for fork in EXPORTED_DEBRIEF_FORKS:
                fobj = d.get(fork)
                if not isinstance(fobj, dict) or fobj.get("raw") is None:
                    continue
                debrief_rows.append(
                    {
                        "episode_id": ep_id,
                        "task": ep["task"],
                        "seed": ep["seed"],
                        "arm": ep["arm"],
                        "role": role,
                        "subject_model": d["subject_model"],
                        "partner_model": d["partner_model"],
                        "fork": fork,
                        "raw_response": fobj.get("raw"),
                        "parsed_json": json.dumps(
                            fobj.get("parsed") if "parsed" in fobj else fobj.get("choice"),
                            ensure_ascii=False,
                        ),
                        "option_order_json": (
                            json.dumps(fobj.get("option_order"), ensure_ascii=False)
                            if "option_order" in fobj
                            else None
                        ),
                        "quote_audit_json": (
                            json.dumps(fobj.get("quote_audit"), ensure_ascii=False)
                            if "quote_audit" in fobj
                            else None
                        ),
                        "fidelity_ok": fobj.get("fidelity_ok"),
                        "coherence_clean": fobj.get("coherence_clean"),
                        "returned_model": fobj.get("returned_model"),
                        "provider": fobj.get("provider"),
                        "reasoning_tokens": fobj.get("reasoning_tokens"),
                        "cost_usd": fobj.get("cost_usd"),
                    }
                )

    failed_rows: list[dict[str, Any]] = []
    for fe in failed_episodes:
        err_str = str(fe.get("error") or "")
        _assert_no_secrets_text(err_str, f"failed episode {fe.get('episode_id')}")
        failed_rows.append(
            {
                "episode_id": fe.get("episode_id"),
                "task": fe.get("task"),
                "seed": fe.get("seed"),
                "model_a": fe.get("model_a"),
                "model_b": fe.get("model_b"),
                "arm": fe.get("arm"),
                "complete": bool(fe.get("complete") is True),
                "failed": bool(fe.get("failed", True)),
                "error": err_str,
            }
        )

    choice_rows: list[dict[str, Any]] = []
    for ch in choices:
        c_copy = dict(ch)
        packet = c_copy.pop("choice_packet", None)
        c_copy["choice_packet_json"] = (
            json.dumps(packet, ensure_ascii=False) if packet is not None else None
        )
        choice_rows.append(c_copy)

    tables: dict[str, pd.DataFrame] = {
        "episodes": pd.DataFrame(ep_rows),
        "turns": pd.DataFrame(turn_rows),
        "debriefs": pd.DataFrame(debrief_rows),
        "choices": pd.DataFrame(choice_rows),
    }
    if failed_rows:
        tables["failed_episodes"] = pd.DataFrame(failed_rows)
    if choices_excluded_rows:
        tables["choices_excluded"] = pd.DataFrame(choices_excluded_rows)
    if capstone_rows:
        cap_export_rows = []
        for cr in capstone_rows:
            c2 = {
                k: v
                for k, v in cr.items()
                if k not in ("transcript_1_turns", "transcript_2_turns")
            }
            c2["transcript_1_turns_json"] = json.dumps(
                cr.get("transcript_1_turns") or [], ensure_ascii=False
            )
            c2["transcript_2_turns_json"] = json.dumps(
                cr.get("transcript_2_turns") or [], ensure_ascii=False
            )
            cap_export_rows.append(c2)
        tables["capstone_tone_flip"] = pd.DataFrame(cap_export_rows)
    if capstone_rw_rows:
        tables["capstone_rewrites"] = pd.DataFrame(capstone_rw_rows)

    # 6. Optional v2.1 Experiment 1 & Experiment 2 tables
    exp1_dir = run_dir / "exp1_dropout"
    if exp1_dir.exists():
        tables.update(_build_exp1_tables(exp1_dir))
    exp2_dir = run_dir / "exp2_info_flip"
    if exp2_dir.exists():
        tables.update(_build_exp2_tables(exp2_dir))

    # 7. Write zstd-compressed Parquet tables ONLY (zero duplicate .jsonl files!)
    exported_configs: list[str] = []
    table_counts: dict[str, int] = {}
    for name, df in tables.items():
        if df.empty:
            continue
        # Scan all text-like columns (pandas 3.x StringDtype 'str', 'string', object, categorical)
        for col in df.columns:
            if _is_text_like_dtype(df[col].dtype):
                ser = df[col].dropna().astype(str)
                if ser.str.contains("sk-or-v1-", regex=False).any():
                    raise ValueError(f"Secret token detected in table '{name}', column '{col}'!")
        pq_path = out_dir / f"{name}.parquet"
        df.to_parquet(pq_path, index=False, compression="zstd")
        exported_configs.append(name)
        table_counts[name] = len(df)

    # 8. Copy frozen key-scrubbed manifests
    copied_manifests = _copy_frozen_manifests(run_dir, out_dir)

    # 9. Write clean HuggingFace README.md dataset card (zero em-dashes, zero deprecated refs)
    configs_yaml_lines = ["configs:"]
    for cfg_name in exported_configs:
        configs_yaml_lines.extend(
            [
                f"  - config_name: {cfg_name}",
                "    data_files:",
                "      - split: train",
                f"        path: {cfg_name}.parquet",
            ]
        )
    configs_yaml = "\n".join(configs_yaml_lines)

    tasks_str = ", ".join(f"`{t}`" for t in sorted({r["task"] for r in ep_rows}))
    n_main_eps = sum(1 for r in ep_rows if r.get("arm") != "opus_boost")
    n_boost_eps = sum(1 for r in ep_rows if r.get("arm") == "opus_boost")
    n_main_turns = sum(1 for r in turn_rows if r.get("arm") != "opus_boost")
    n_joined_turns = sum(1 for r in turn_rows if r.get("ledger_joined"))

    table_bullets: list[str] = [
        f"- `episodes.parquet` (`{table_counts.get('episodes', 0):,}` rows): completed two-agent episodes across {tasks_str} (`{n_main_eps}` main-arm + `{n_boost_eps}` isolated `opus_boost`), including role briefings, realized shocks, and deterministic scoring details.",
    ]
    if "failed_episodes" in table_counts:
        table_bullets.append(
            f"- `failed_episodes.parquet` (`{table_counts['failed_episodes']:,}` rows): incomplete episodes excluded from primary analysis for transparent completion accounting (`{table_counts.get('episodes', 0)}/{table_counts.get('episodes', 0) + table_counts['failed_episodes']}`)."
        )
    table_bullets.append(
        f"- `turns.parquet` (`{table_counts.get('turns', 0):,}` rows; `{n_main_turns:,}` main-arm turns, `{n_joined_turns:,}` ledger-joined): turn-level visible messages, structured actions, verifier feedback, linguistic metrics, 3-judge panel consensus and raw judge outputs, accepted/total token and cost telemetry, and per-turn API `latency_s`."
    )
    table_bullets.append(
        f"- `debriefs.parquet` (`{table_counts.get('debriefs', 0):,}` rows): private post-episode evaluations (`F1` 1-7 Likert rating, `F1_retest` 25% stratified test-retest subset, and `F4` binary `KEEP`/`REROLL` retention)."
    )
    table_bullets.append(
        f"- `choices.parquet` (`{table_counts.get('choices', 0):,}` rows): order-balanced spliced-history forced teammate choices across all 7 partners, including full `choice_packet_json` prompts and free-text rationales."
    )
    if "choices_excluded" in table_counts:
        table_bullets.append(
            f"- `choices_excluded.parquet` (`{table_counts['choices_excluded']:,}` rows): choice items excluded due to incomplete upstream episodes (`{table_counts.get('choices', 0)}/{table_counts.get('choices', 0) + table_counts['choices_excluded']}`)."
        )
    if "capstone_tone_flip" in table_counts:
        table_bullets.append(
            f"- `capstone_tone_flip.parquet` (`{table_counts['capstone_tone_flip']:,}` rows): symmetric Warm vs. Blunt third-person staffing crossover choices with blinded transcript packets and objective outcome cards."
        )
    if "capstone_rewrites" in table_counts:
        n_cap_cands = len({r["episode_id"] for r in capstone_rw_rows})
        table_bullets.append(
            f"- `capstone_rewrites.parquet` (`{table_counts['capstone_rewrites']:,}` rows across `{n_cap_cands}` candidate episodes): DeepSeek V4 Pro tone rewrites with turn-level automated and 3-judge demeanor/fidelity audits."
        )
    if "exp1_dropout_gate" in table_counts:
        table_bullets.append(
            f"- `exp1_dropout_gate.parquet` (`{table_counts['exp1_dropout_gate']:,}` rows): Experiment 1 calibration gate evaluations across 8 receivers x 16 cases x 2 conditions (`floor_no_partner` and `ceiling_verbatim_rules`), with deterministic 15-rule grading and raw model submissions."
        )
    if "exp1_dropout" in table_counts:
        table_bullets.append(
            f"- `exp1_dropout.parquet` (`{table_counts['exp1_dropout']:,}` rows): Experiment 1 Partner-Drops-Out single-turn handoff outcomes across 320 cases x 8 receivers, including deterministic grading (`own_r3_ok`, `partner_7_ok`, `strict_15_ok`), disclosure/bloat bins, token/latency/cost telemetry, and raw submissions."
        )
    if "exp1_dropout_attempts" in table_counts:
        table_bullets.append(
            f"- `exp1_dropout_attempts.parquet` (`{table_counts['exp1_dropout_attempts']:,}` rows): key-scrubbed attempt-level API ledger for Experiment 1 (calibration gate, accepted matrix calls, retries, and Gemini 4 Neon output-limit truncation attempts)."
        )
    if "exp2_info_flip_pairs" in table_counts:
        n_e2_main = int((tables["exp2_info_flip_pairs"]["arm"] == "main").sum())
        n_e2_plac = int((tables["exp2_info_flip_pairs"]["arm"] == "placebo").sum())
        table_bullets.append(
            f"- `exp2_info_flip_pairs.parquet` (`{table_counts['exp2_info_flip_pairs']:,}` rows: `{n_e2_main}` main + `{n_e2_plac}` placebo): exact word-matched Round-1 surgical edit pairs (`private_info` vs. `public_echo` vs. `unedited`) with zero-leak and parity verification fields."
        )
    if "exp2_info_flip_choices" in table_counts:
        n_e2c_main = int((tables["exp2_info_flip_choices"]["arm"] == "main").sum())
        n_e2c_plac = int((tables["exp2_info_flip_choices"]["arm"] == "placebo").sum())
        table_bullets.append(
            f"- `exp2_info_flip_choices.parquet` (`{table_counts['exp2_info_flip_choices']:,}` rows: `{n_e2c_main}` main + `{n_e2c_plac}` placebo): symmetric Information-Flip third-person staffing crossover decisions across 4 frontier reviewers."
        )

    manifest_section = ""
    if copied_manifests:
        m_bullets = "\n".join(f"- `{m}`" for m in copied_manifests)
        manifest_section = f"\n## Frozen Manifests\n\n{m_bullets}\n"

    card = f"""---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - multi-agent
  - llm-evaluation
  - handoff-reliability
  - communication-style
  - cross-vendor
pretty_name: "Mighty Models, Terrible Teammates ({run_name})"
{configs_yaml}
---

# Mighty Models, Terrible Teammates: Dataset (`{run_name}`)

Key-scrubbed, zstd-compressed Parquet release containing all trajectories, counterfactual choices, deterministic handoff evaluations, and surgical intervention artefacts required to independently reproduce the technical report.

- **Technical Report:** https://gkaissis.github.io/mighty-models-terrible-teammates/
- **Source Code and Reproducibility Package:** https://github.com/gkaissis/mighty-models-terrible-teammates

## Tables

{chr(10).join(table_bullets)}
{manifest_section}
## Pinned Endpoints and Evaluation Protocol

- **Main 7-partner collaboration league (OpenRouter first-party pins, `reasoning.effort = "medium"`):**
  - Core choosers and partners: `openai/gpt-6.2-astra` (`OpenAI`), `google/gemini-4-neon` (`Google AI Studio`), `anthropic/claude-opus-5.5` (`Anthropic`), `moonshotai/kimi-k3.1-pro` (`Moonshot AI`).
  - Partner-only controls: `openai/gpt-6-astra` (`OpenAI`), `meta/muse-wonder` (`Meta`), `qwen/qwen4-max` (`Alibaba`).
- **Vendor-independent 3-judge demeanor panel (`/api/alpha/decisions`):** `liquid/d1.2` (`Liquid`), `perplexity/pplx-decider-v1.2` (`Perplexity`), `typesafe/jev-1.14` (`TypeSafe`).
- **Neutral Capstone tone rewriter:** `deepseek/deepseek-v4-pro-0813` (`BaseTen`, `reasoning.effort = "low"`).
- **Experiment 1 worker-tier receivers:** `google/gemini-4-hydrogen` (`Google AI Studio`, `low`), `anthropic/claude-haiku-5.5` (`Anthropic`, `low`), `openai/gpt-6-luna` (`OpenAI`, `low`), and `deepseek/deepseek-v4.1-flash` (`BaseTen`, reasoning explicitly disabled with `reasoning.enabled = false`, verified `0` reasoning tokens across all gate and matrix calls).

## Independent Replay Verification

Every headline table, statistical test, deterministic dropout verdict, and causal intervention result in the report reproduces strictly from these Parquet files without accessing any external raw logs or making any API calls:

```bash
uv run python scripts/verify_hf_replay.py --run {run_name} --capstone-run v2_capstone
```

## Hygiene and Scope

- Format: zstd-compressed Parquet tables, frozen JSON manifests, and this dataset card only (no duplicate uncompressed `.jsonl` tables).
- Security: verified zero API keys or private credentials across all exported files.
"""
    _assert_no_secrets_text(card, "README.md")
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export lean, key-scrubbed, zstd-compressed HuggingFace Parquet dataset (v2.1)."
    )
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()
    out = export_huggingface_dataset(args.run)
    print(f"Exported HuggingFace v2.1 dataset to {out}")


if __name__ == "__main__":
    main()
