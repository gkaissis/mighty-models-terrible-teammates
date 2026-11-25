"""Lossless Parquet reconstruction helpers for offline reproduction from `hf_export/*.parquet`.

Allows all analysis scripts, figure generators, verifiers, and unit tests to run identically
on a fresh GitHub clone where only `data/v2/hf_export/*.parquet` is tracked and raw uncompressed
`episodes/*.json` / `*.jsonl` logs are excluded by `.gitignore`.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_loads_opt(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s == "null":
            return None
        return json.loads(s)
    return val


def _clean_record(rec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def resolve_hf_export_dir(path: Path) -> Path | None:
    """Resolve the `hf_export/` directory from either a run dir, `episodes/` dir, or `hf_export/` dir."""
    p = Path(path)
    candidates = [p, p / "hf_export", p.parent / "hf_export"]
    for cand in candidates:
        if (cand / "episodes.parquet").is_file():
            return cand
    return None


def load_episodes_from_parquet(
    hf_dir: Path,
    *,
    include_failed: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reconstruct complete episode dicts strictly from `episodes/turns/debriefs` Parquet tables."""
    ep_df = pd.read_parquet(hf_dir / "episodes.parquet")
    turns_df = pd.read_parquet(hf_dir / "turns.parquet")
    deb_path = hf_dir / "debriefs.parquet"
    deb_df = pd.read_parquet(deb_path) if deb_path.is_file() else pd.DataFrame()

    turns_by_ep: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    for row in turns_df.to_dict(orient="records"):
        r = _clean_record(row)
        ep_id = str(r["episode_id"])
        rnd = int(r["round"])
        role = str(r["role"])
        t_obj = turns_by_ep[ep_id].setdefault(
            rnd,
            {
                "round": rnd,
                "feedback_a": r.get("feedback_a"),
                "feedback_b": r.get("feedback_b"),
                "round_metrics": _json_loads_opt(r.get("round_metrics_json")),
            },
        )
        pc = _json_loads_opt(r.get("panel_consensus_json")) or {}
        pj = _json_loads_opt(r.get("panel_judges_raw_json"))
        jd: dict[str, Any] = {"panel_consensus": pc, "panel_judges": pj}
        if isinstance(pc, dict):
            for k, v in pc.items():
                if isinstance(v, dict) and "mean" in v:
                    jd[k] = v["mean"]
                elif not isinstance(v, dict):
                    jd[k] = v
        t_obj[role] = {
            "model": r["model"],
            "handle": r["handle"],
            "message": r.get("message") or "",
            "action": _json_loads_opt(r.get("action_json")),
            "parse_error": r.get("parse_error"),
            "metrics": _json_loads_opt(r.get("metrics_json")) or {},
            "jev_demeanor": jd,
            "accepted_call_id": r.get("accepted_call_id"),
            "accepted_attempt": r.get("accepted_attempt"),
            "returned_model": r.get("returned_model"),
            "provider": r.get("provider"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "reasoning_tokens": r.get("reasoning_tokens"),
            "cached_tokens": r.get("cached_tokens"),
            "accepted_cost_usd": r.get("accepted_cost_usd"),
            "total_cost_usd": r.get("total_cost_usd"),
            "latency_s": r.get("latency_s"),
            "ledger_joined": bool(r.get("ledger_joined")),
            "n_action_blocks": r.get("n_action_blocks"),
        }

    deb_by_ep: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    if not deb_df.empty:
        for row in deb_df.to_dict(orient="records"):
            r = _clean_record(row)
            ep_id = str(r["episode_id"])
            role = str(r["role"])
            fork = str(r["fork"])
            d_role = deb_by_ep[ep_id].setdefault(
                role,
                {
                    "subject_model": r["subject_model"],
                    "partner_model": r["partner_model"],
                },
            )
            parsed = _json_loads_opt(r.get("parsed_json"))
            fentry: dict[str, Any] = {"raw": r.get("raw_response")}
            if fork == "F4":
                fentry["choice"] = parsed
            else:
                fentry["parsed"] = parsed
            d_role[fork] = fentry

    episodes: list[dict[str, Any]] = []
    for row in ep_df.to_dict(orient="records"):
        r = _clean_record(row)
        ep_id = str(r["episode_id"])
        rnd_map = turns_by_ep.get(ep_id, {})
        turns_sorted = [rnd_map[k] for k in sorted(rnd_map.keys())]
        episodes.append(
            {
                "episode_id": ep_id,
                "run_name": r.get("run_name"),
                "task": r["task"],
                "seed": int(r["seed"]),
                "arm": r.get("arm"),
                "total_rounds": int(r.get("total_rounds") or len(turns_sorted)),
                "model_a": r["model_a"],
                "model_b": r["model_b"],
                "handle_a": r["handle_a"],
                "handle_b": r["handle_b"],
                "briefing_a": r.get("briefing_a"),
                "briefing_b": r.get("briefing_b"),
                "score": float(r["score"]),
                "score_details": _json_loads_opt(r.get("score_details_json")) or {},
                "complete": True,
                "turns": turns_sorted,
                "debriefs": deb_by_ep.get(ep_id, {}),
            }
        )

    failed_ids: list[str] = []
    failed_path = hf_dir / "failed_episodes.parquet"
    if failed_path.is_file():
        fdf = pd.read_parquet(failed_path)
        if "episode_id" in fdf.columns:
            failed_ids = [str(x) for x in fdf["episode_id"].dropna().tolist()]
        if include_failed:
            for row in fdf.to_dict(orient="records"):
                r = _clean_record(row)
                r["complete"] = False
                episodes.append(r)

    return episodes, failed_ids


def load_choices_from_parquet(hf_dir: Path) -> list[dict[str, Any]]:
    """Reconstruct choice dicts from `choices.parquet`."""
    p = hf_dir / "choices.parquet"
    if not p.is_file():
        return []
    df = pd.read_parquet(p)
    out: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        r = _clean_record(row)
        packet = _json_loads_opt(r.pop("choice_packet_json", None))
        if packet is not None:
            r["choice_packet"] = packet
        out.append(r)
    return out


def load_excluded_choices_from_parquet(hf_dir: Path) -> list[dict[str, Any]]:
    """Reconstruct excluded choice dicts from `choices_excluded.parquet`."""
    p = hf_dir / "choices_excluded.parquet"
    if not p.is_file():
        return []
    df = pd.read_parquet(p)
    return [_clean_record(row) for row in df.to_dict(orient="records")]


def load_capstone_from_parquet(hf_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct `(capstone_rewrites, capstone_choices)` from Parquet tables."""
    rewrites: list[dict[str, Any]] = []
    rw_path = hf_dir / "capstone_rewrites.parquet"
    if rw_path.is_file():
        df = pd.read_parquet(rw_path)
        for row in df.to_dict(orient="records"):
            r = _clean_record(row)
            r["turns"] = _json_loads_opt(r.pop("turns_json", None)) or []
            rewrites.append(r)

    choices: list[dict[str, Any]] = []
    ch_path = hf_dir / "capstone_tone_flip.parquet"
    if ch_path.is_file():
        df = pd.read_parquet(ch_path)
        for row in df.to_dict(orient="records"):
            r = _clean_record(row)
            r["transcript_1_turns"] = _json_loads_opt(r.pop("transcript_1_turns_json", None))
            r["transcript_2_turns"] = _json_loads_opt(r.pop("transcript_2_turns_json", None))
            choices.append(r)
    return rewrites, choices


def load_exp1_results_from_parquet(hf_dir: Path) -> list[dict[str, Any]]:
    """Reconstruct Experiment 1 dropout result records from `exp1_dropout.parquet`."""
    p = hf_dir / "exp1_dropout.parquet"
    if not p.is_file():
        return []
    df = pd.read_parquet(p)
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        r = _clean_record(row)
        r["strict_all_pass"] = bool(r.get("strict_15_ok"))
        r["parsed_config"] = _json_loads_opt(r.pop("parsed_config_json", None))
        r["missing_keys"] = _json_loads_opt(r.pop("missing_keys_json", None))
        r["out_of_domain_keys"] = _json_loads_opt(r.pop("out_of_domain_keys_json", None))
        r["rule_results"] = _json_loads_opt(r.pop("rule_results_json", None))
        rows.append(r)
    return rows
