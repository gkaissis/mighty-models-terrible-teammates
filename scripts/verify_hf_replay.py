"""Release-level v2.1 replay verifier: reproduces every headline claim strictly from `hf_export/`.

Reads ZERO private raw files outside `hf_export/` (no `episodes/*.json`, no `ledger.jsonl`,
no `choices.jsonl`, no `exp1_dropout/results.jsonl`, no `exp2_info_flip/choices.jsonl`).
Frozen summary JSONs (`report.json`, `meta_v2_1.json`, `capstone_results.json`,
`exp1_summary.json`, `exp2_summary.json`) are read ONLY as reference targets to assert
numerical equality against the independent Parquet-only recomputations.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common_ground.analyze import (
    _compute_direct_predecessor_analysis,
    _compute_icc_2_1,
    _fit_unified_bt_model,
)
from common_ground.config import ALL_PARTNER_MODELS, CORE_MODELS, DATA_DIR
from common_ground.dropout import (
    build_dropout_cases,
    grade_dropout_config,
    parse_config_response,
    summarize_exp1_results,
)
from common_ground.info_flip import summarize_exp2_results
from common_ground.meta_analysis import (
    compute_capstone_audit,
    compute_choice_logits,
    compute_construct_validity,
    compute_disclosure_meta_analysis,
    compute_triage_paralysis_and_upsets,
)

ALLOWED_DEBRIEF_FORKS = {"F1", "F1_retest", "F4"}
DEPRECATED_README_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("em_dash", re.compile(r"—")),
    ("F2_fork", re.compile(r"\bF2\b")),
    ("F5_fork", re.compile(r"\bF5\b")),
    ("R5_blame", re.compile(r"\bR5\b")),
    ("false_blame", re.compile(r"false[_ ]blame", re.IGNORECASE)),
    ("two_faced", re.compile(r"two[- ]faced", re.IGNORECASE)),
    ("human_poll", re.compile(r"human[_ ]poll|google[_ ]form", re.IGNORECASE)),
)


def _is_text_like_dtype(dtype: Any) -> bool:
    """Return True for any pandas 2.x/3.x dtype that can hold strings (including StringDtype 'str')."""
    return (
        isinstance(dtype, (pd.StringDtype, pd.CategoricalDtype))
        or pd.api.types.is_string_dtype(dtype)
        or dtype == object
    )


def _assert_close(
    label: str,
    actual: float | None,
    expected: float | None,
    tol: float,
) -> None:
    if expected is None:
        return
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        raise AssertionError(f"{label} mismatch: expected={expected}, actual={actual}")
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(
            f"{label} mismatch: expected={expected}, actual={actual} (diff={abs(float(actual) - float(expected)):.6g} > tol={tol})"
        )


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


def _parse_readme_frontmatter_configs(readme_text: str) -> dict[str, str]:
    """Validate HuggingFace YAML frontmatter and return {config_name: path}."""
    m = re.match(r"^---\n(.*?)\n---\n", readme_text, re.DOTALL)
    if not m:
        raise AssertionError("hf_export/README.md missing valid YAML frontmatter delimited by '---'!")
    fm = m.group(1)
    if not re.search(r"^license:\s*apache-2\.0\s*$", fm, re.MULTILINE):
        raise AssertionError("hf_export/README.md YAML frontmatter missing 'license: apache-2.0'!")
    if not re.search(r"^pretty_name:\s*.+$", fm, re.MULTILINE):
        raise AssertionError("hf_export/README.md YAML frontmatter missing 'pretty_name'!")
    if not re.search(r"^tags:\s*$", fm, re.MULTILINE):
        raise AssertionError("hf_export/README.md YAML frontmatter missing 'tags'!")
    if not re.search(r"^configs:\s*$", fm, re.MULTILINE):
        raise AssertionError("hf_export/README.md YAML frontmatter missing 'configs'!")

    configs: dict[str, str] = {}
    current_cfg: str | None = None
    in_configs = False
    for raw_line in fm.splitlines():
        if re.match(r"^configs:\s*$", raw_line):
            in_configs = True
            continue
        if in_configs and raw_line and not raw_line.startswith(" "):
            break
        if not in_configs:
            continue
        cm = re.match(r"^\s*-\s*config_name:\s*([A-Za-z0-9_.-]+)\s*$", raw_line)
        if cm:
            current_cfg = cm.group(1)
            continue
        pm = re.match(r"^\s*path:\s*([A-Za-z0-9_./-]+)\s*$", raw_line)
        if pm and current_cfg is not None:
            configs[current_cfg] = pm.group(1)
    if not configs:
        raise AssertionError("hf_export/README.md YAML 'configs:' section has zero entries!")
    return configs


def _verify_hygiene(hf_dir: Path) -> dict[str, Any]:
    """Enforce zero deprecated fat, zero JSONL duplicates, zero secrets, valid YAML card."""
    all_files = sorted(p for p in hf_dir.rglob("*") if p.is_file())
    rel_paths = [p.relative_to(hf_dir).as_posix() for p in all_files]

    for p, rel in zip(all_files, rel_paths):
        if "human_poll" in p.name.lower():
            raise AssertionError(f"Deprecated human poll artifact found inside hf_export/: {rel}")
        if p.suffix.lower() == ".jsonl":
            raise AssertionError(f"Duplicate .jsonl table found inside hf_export/: {rel}")
        if p.suffix.lower() in (".md", ".json", ".txt", ".html"):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "sk-or-v1-" in txt:
                raise AssertionError(f"Possible OpenRouter API key leaked in {rel}!")
            if p.suffix.lower() == ".json":
                json.loads(txt)  # Must be valid JSON

    readme_path = hf_dir / "README.md"
    if not readme_path.exists():
        raise AssertionError("Missing hf_export/README.md!")
    readme_text = readme_path.read_text(encoding="utf-8")
    for label, pat in DEPRECATED_README_PATTERNS:
        if pat.search(readme_text):
            raise AssertionError(f"Forbidden pattern '{label}' found in hf_export/README.md!")

    yaml_configs = _parse_readme_frontmatter_configs(readme_text)
    actual_parquets = sorted(p.name for p in hf_dir.glob("*.parquet"))
    expected_parquets = sorted(yaml_configs.values())
    if actual_parquets != expected_parquets:
        raise AssertionError(
            f"README.md YAML configs do not match *.parquet files: yaml={expected_parquets}, disk={actual_parquets}"
        )

    loaded_tables: dict[str, pd.DataFrame] = {}
    for cfg_name, rel_pq in yaml_configs.items():
        pq_path = hf_dir / rel_pq
        if not pq_path.exists():
            raise AssertionError(f"YAML config '{cfg_name}' references missing file: {rel_pq}")
        pf = pq.ParquetFile(pq_path)
        if pf.metadata.num_row_groups < 1:
            raise AssertionError(f"Parquet table '{rel_pq}' has zero row groups!")
        for rg_idx in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                comp = rg.column(col_idx).compression
                if comp != "ZSTD":
                    col_name = rg.column(col_idx).path_in_schema
                    raise AssertionError(
                        f"Parquet table '{rel_pq}' row_group={rg_idx} column '{col_name}' uses '{comp}' instead of 'ZSTD'!"
                    )
        df = pd.read_parquet(pq_path)
        for col in df.columns:
            if _is_text_like_dtype(df[col].dtype):
                ser = df[col].dropna().astype(str)
                if ser.str.contains("sk-or-v1-", regex=False).any():
                    raise AssertionError(f"Secret key leaked in {rel_pq} column '{col}'!")
        loaded_tables[cfg_name] = df

    deb_df = loaded_tables.get("debriefs")
    if deb_df is None:
        raise AssertionError("Missing required debriefs.parquet!")
    if "false_blame_json" in deb_df.columns:
        raise AssertionError("Deprecated column 'false_blame_json' must NOT exist in debriefs.parquet!")
    forks_present = set(deb_df["fork"].dropna().astype(str).unique())
    forbidden_forks = forks_present - ALLOWED_DEBRIEF_FORKS
    if forbidden_forks:
        raise AssertionError(
            f"Deprecated debrief forks found in debriefs.parquet: {sorted(forbidden_forks)}"
        )

    manifests = sorted(
        p.relative_to(hf_dir).as_posix()
        for p in (hf_dir / "manifests").glob("*.json")
        if p.is_file()
    ) if (hf_dir / "manifests").exists() else []

    return {
        "hygiene_ok": True,
        "yaml_configs_count": len(yaml_configs),
        "yaml_configs": sorted(yaml_configs.keys()),
        "manifests": manifests,
        "debrief_forks": sorted(forks_present),
        "tables": loaded_tables,
    }


def _reconstruct_episodes_from_parquet(
    ep_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    deb_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Reconstruct full episode dicts strictly from episodes/turns/debriefs Parquet tables."""
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
        }

    deb_by_ep: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
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
    return episodes


def _compute_bloat_tax_from_parquet(episodes_main: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute Compute Bloat Tax strictly from reconstructed Parquet episodes+turns (ZERO ledger.jsonl)."""
    n_core_all = 0
    n_core_joined = 0
    n_core_exposed = 0
    rows: list[dict[str, Any]] = []

    for ep in episodes_main:
        ep_id = str(ep["episode_id"])
        task = str(ep["task"])
        seed = int(ep["seed"])
        for role in ("A", "B"):
            subj = str(ep["model_a"] if role == "A" else ep["model_b"])
            part = str(ep["model_b"] if role == "A" else ep["model_a"])
            if subj not in CORE_MODELS:
                continue
            rt_sum = ct_sum = pt_sum = lat_ms_sum = cost_sum = wc_sum = 0.0
            conf_vals: list[float] = []
            for t in ep.get("turns") or []:
                rnd = int(t["round"])
                side = t[role]
                n_core_all += 1
                if side.get("ledger_joined"):
                    n_core_joined += 1
                if role == "A" and rnd < 2:
                    continue
                n_core_exposed += 1
                rt_sum += float(side.get("reasoning_tokens") or 0.0)
                ct_sum += float(side.get("completion_tokens") or 0.0)
                pt_sum += float(side.get("prompt_tokens") or 0.0)
                lat_s = side.get("latency_s")
                lat_ms_sum += float(lat_s) * 1000.0 if lat_s is not None else 0.0
                cost_sum += float(side.get("accepted_cost_usd") or 0.0)
                m = side.get("metrics") or {}
                wc_sum += float(m.get("word_count") or 0.0)
                jd = side.get("jev_demeanor") or {}
                if jd.get("expresses_confusion") is not None:
                    conf_vals.append(float(jd["expresses_confusion"]))
            rows.append(
                {
                    "episode_id": ep_id,
                    "subject": subj,
                    "partner": part,
                    "task": task,
                    "role": role,
                    "seed": seed,
                    "block": f"{subj}__{task}__{role}__{seed}",
                    "reasoning_tokens": rt_sum,
                    "completion_tokens": ct_sum,
                    "prompt_tokens": pt_sum,
                    "latency_ms": lat_ms_sum,
                    "cost_usd": cost_sum,
                    "word_count": wc_sum,
                    "expresses_confusion": float(np.mean(conf_vals)) if conf_vals else np.nan,
                }
            )

    bdf = pd.DataFrame(rows)
    pct_metrics = (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "latency_ms",
        "word_count",
        "cost_usd",
    )
    pct_out: dict[str, dict[str, float]] = {}
    for col in pct_metrics:
        bmean = bdf.groupby("block")[col].transform("mean")
        work = bdf[bmean > 1e-12].copy()
        work["pct_delta"] = (work[col] / work.groupby("block")[col].transform("mean") - 1.0) * 100.0
        pct_out[col] = {
            str(p): round(float(g["pct_delta"].mean()), 4) for p, g in work.groupby("partner")
        }

    conf_work = bdf.dropna(subset=["expresses_confusion"]).copy()
    conf_work["abs_delta"] = conf_work["expresses_confusion"] - conf_work.groupby("block")[
        "expresses_confusion"
    ].transform("mean")
    conf_raw = {
        str(p): round(float(g["expresses_confusion"].mean()), 5)
        for p, g in conf_work.groupby("partner")
    }
    conf_delta = {
        str(p): round(float(g["abs_delta"].mean()), 4) for p, g in conf_work.groupby("partner")
    }

    return {
        "n_core_subject_turns_checked": n_core_all,
        "n_ledger_matched": n_core_joined,
        "n_exposed_turns": n_core_exposed,
        "n_blocks": int(bdf["block"].nunique()) if not bdf.empty else 0,
        "pct_delta_by_metric": pct_out,
        "expresses_confusion_raw_mean": conf_raw,
        "expresses_confusion_abs_delta": conf_delta,
    }


def _replay_f1_retest(deb_df: pd.DataFrame) -> dict[str, Any]:
    """Recompute F1 vs F1_retest ICC(2,1) strictly from debriefs.parquet."""
    main_deb = deb_df[deb_df["arm"] != "opus_boost"].copy() if "arm" in deb_df.columns else deb_df.copy()
    if main_deb.empty:
        main_deb = deb_df.copy()

    def _extract_again(val: Any) -> float | None:
        obj = _json_loads_opt(val)
        if isinstance(obj, dict) and obj.get("again") is not None:
            return float(obj["again"])
        return None

    main_deb["again_val"] = main_deb["parsed_json"].map(_extract_again)
    f1 = main_deb[main_deb["fork"] == "F1"].set_index(["episode_id", "role", "subject_model"])["again_val"]
    f1r = main_deb[main_deb["fork"] == "F1_retest"].set_index(["episode_id", "role", "subject_model"])["again_val"]
    paired = pd.concat([f1.rename("f1"), f1r.rename("f1r")], axis=1).dropna()
    if len(paired) < 4:
        return {"ok": False, "n": len(paired)}
    a = paired["f1"].to_numpy(dtype=float)
    b = paired["f1r"].to_numpy(dtype=float)
    icc = _compute_icc_2_1(a, b)
    exact = float(np.mean(np.abs(a - b) == 0))
    return {
        "ok": True,
        "n": len(paired),
        "icc_2_1": round(icc, 4) if icc is not None and not math.isnan(icc) else None,
        "exact_agreement_rate": round(exact, 4),
    }


def _replay_exp1(
    tables: dict[str, pd.DataFrame],
    hf_dir: Path,
    ref_exp1_path: Path,
    tol: float,
) -> dict[str, Any]:
    """Independently replay all Experiment 1 claims strictly from exp1_dropout*.parquet."""
    gate_df = tables["exp1_dropout_gate"]
    mat_df = tables["exp1_dropout"]
    att_df = tables["exp1_dropout_attempts"]

    # 1. Calibration gate replay
    gate_final = gate_df[gate_df["is_final_step"] == True] if "is_final_step" in gate_df.columns else gate_df
    n_gate_total = len(gate_final)
    n_gate_floor = int((gate_final["condition"] == "floor_no_partner").sum())
    n_gate_ceil = int((gate_final["condition"] == "ceiling_verbatim_rules").sum())
    ds_gate = gate_final[gate_final["receiver"] == "deepseek_flash"]
    ds_zero_rt = int((ds_gate["reasoning_tokens"].fillna(-1) == 0).sum())
    if len(ds_gate) > 0 and ds_zero_rt != len(ds_gate):
        raise AssertionError(
            f"DeepSeek V4.1 Flash gate non-zero reasoning tokens: {ds_zero_rt}/{len(ds_gate)}"
        )

    floor_rates = (
        gate_final[gate_final["condition"] == "floor_no_partner"]
        .groupby("receiver")["strict_15_ok"]
        .mean()
        .to_dict()
    )
    ceil_rates = (
        gate_final[gate_final["condition"] == "ceiling_verbatim_rules"]
        .groupby("receiver")["strict_15_ok"]
        .mean()
        .to_dict()
    )

    roster_path = hf_dir / "manifests" / "locked_receiver_roster.json"
    if roster_path.exists():
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        for rk, rinfo in (roster.get("receivers") or {}).items():
            _assert_close(
                f"exp1 gate floor {rk}",
                floor_rates.get(rk),
                rinfo.get("gate_floor_strict_pass_rate"),
                tol,
            )
            _assert_close(
                f"exp1 gate ceiling {rk}",
                ceil_rates.get(rk),
                rinfo.get("gate_ceiling_strict_pass_rate"),
                tol,
            )

    # 1b. Independently regrade ALL raw receiver submissions (gate + matrix) via grade_dropout_config
    cases_by_id = {c.case_id: c for c in build_dropout_cases(hf_dir)}
    regraded_gate_n = 0
    for row in gate_final.to_dict(orient="records"):
        cid = str(row["case_id"])
        case = cases_by_id[cid]
        raw_resp = row.get("raw_response")
        if bool(row["ok"]) and isinstance(raw_resp, str):
            g = grade_dropout_config(case, raw_resp)
            if (
                g.strict_all_pass != bool(row["strict_15_ok"])
                or g.own_plus_r3_pass != bool(row["own_r3_ok"])
                or g.partner_rules_pass != bool(row["partner_7_ok"])
            ):
                raise AssertionError(
                    f"Gate raw-response regrade mismatch on {row.get('call_id')}: "
                    f"regraded=({g.own_plus_r3_pass},{g.partner_rules_pass},{g.strict_all_pass}) "
                    f"vs stored=({row['own_r3_ok']},{row['partner_7_ok']},{row['strict_15_ok']})"
                )
        regraded_gate_n += 1

    # 2. Main matrix replay using raw-response regrading + summarize_exp1_results
    recs: list[dict[str, Any]] = []
    regraded_mat_n = 0
    native_int_strict_pass_n = 0
    str_coerced_strict_pass_n = 0
    for row in mat_df.to_dict(orient="records"):
        r = _clean_record(row)
        case = cases_by_id[str(r["case_id"])]
        raw_resp = r.get("raw_response")
        if bool(r["ok"]) and isinstance(raw_resp, str):
            parsed = parse_config_response(raw_resp)
            g = grade_dropout_config(case, raw_resp)
            if (
                g.strict_all_pass != bool(r["strict_15_ok"])
                or g.own_plus_r3_pass != bool(r["own_r3_ok"])
                or g.partner_rules_pass != bool(r["partner_7_ok"])
            ):
                raise AssertionError(
                    f"Matrix raw-response regrade mismatch on {r.get('call_id')}: "
                    f"regraded=({g.own_plus_r3_pass},{g.partner_rules_pass},{g.strict_all_pass}) "
                    f"vs stored=({r['own_r3_ok']},{r['partner_7_ok']},{r['strict_15_ok']})"
                )
            if g.strict_all_pass:
                if parsed.used_string_int_coercion:
                    str_coerced_strict_pass_n += 1
                else:
                    native_int_strict_pass_n += 1
            r["own_plus_r3_pass"] = g.own_plus_r3_pass
            r["partner_rules_pass"] = g.partner_rules_pass
            r["strict_all_pass"] = g.strict_all_pass
        else:
            r["own_plus_r3_pass"] = bool(r["own_r3_ok"])
            r["partner_rules_pass"] = bool(r["partner_7_ok"])
            r["strict_all_pass"] = bool(r["strict_15_ok"])
        r["latency_ms"] = (float(r["latency_s"]) * 1000.0) if r.get("latency_s") is not None else 0.0
        recs.append(r)
        regraded_mat_n += 1

    s1 = summarize_exp1_results(recs)

    # Truncation audit & DeepSeek non-reasoning audit from matrix
    ds_mat = mat_df[mat_df["receiver_key"] == "deepseek_flash"]
    ds_mat_zero_rt = int((ds_mat["reasoning_tokens"].fillna(-1) == 0).sum())
    if len(ds_mat) > 0 and ds_mat_zero_rt != len(ds_mat):
        raise AssertionError(
            f"DeepSeek V4.1 Flash matrix non-zero reasoning tokens: {ds_mat_zero_rt}/{len(ds_mat)}"
        )

    err_df = mat_df[mat_df["ok"] == False]
    n_err = len(err_df)
    err_by_rec = {str(k): int(v) for k, v in err_df["receiver_key"].value_counts().items()}
    err_by_sender = {str(k): int(v) for k, v in err_df["sender_model"].value_counts().items()}

    # Role-A CV strict 15/15 ITT & completed-only for all 7 senders
    role_a_cv_by_sender: dict[str, dict[str, Any]] = {}
    for s in ALL_PARTNER_MODELS:
        if s in s1["by_sender"]:
            d = s1["by_sender"][s]["primary_role_a_cross_vendor"]
            role_a_cv_by_sender[s] = {
                "n_cases": d["n_cases"],
                "n_ok": d["n_ok"],
                "itt_rate": d["strict_all_pass_rate"],
                "completed_only_rate": d["strict_all_pass_rate_ok"],
            }

    # Within-GPT-6.2 Role-A CV
    w_a62 = s1["within_astra62_shared_vs_withheld"]["primary_role_a_cross_vendor_r1_r2"]

    # Same-family decoding (GPT-6.2 Role-A)
    sf_a62_df = mat_df[
        (mat_df["sender_model"] == "astra62")
        & (mat_df["dropped_role"] == "A")
        & (mat_df["is_same_family_as_sender"] == True)
    ]
    sf_a62_rate = float(sf_a62_df["strict_15_ok"].mean()) if len(sf_a62_df) else None

    # Role-B contagion (Role-B dropped, conditioned on Role-A partner model)
    rb_df = mat_df[mat_df["dropped_role"] == "B"]
    transp_leads = {"neon4", "muse", "astra60", "opus55"}
    withh_leads = {"astra62", "kimi31"}
    rb_tr = rb_df[rb_df["original_receiver_model"].isin(transp_leads)]
    rb_wh = rb_df[rb_df["original_receiver_model"].isin(withh_leads)]
    op_rb_tr = rb_tr[rb_tr["sender_model"] == "opus55"]
    op_rb_a62 = rb_df[
        (rb_df["sender_model"] == "opus55") & (rb_df["original_receiver_model"] == "astra62")
    ]
    role_b_contagion = {
        "pooled_after_transparent_n": len(rb_tr),
        "pooled_after_transparent_pass": int(rb_tr["strict_15_ok"].sum()),
        "pooled_after_transparent_rate": round(float(rb_tr["strict_15_ok"].mean()), 4),
        "pooled_after_withholding_n": len(rb_wh),
        "pooled_after_withholding_pass": int(rb_wh["strict_15_ok"].sum()),
        "pooled_after_withholding_rate": round(float(rb_wh["strict_15_ok"].mean()), 4),
        "opus_after_transparent_n": len(op_rb_tr),
        "opus_after_transparent_pass": int(op_rb_tr["strict_15_ok"].sum()),
        "opus_after_transparent_rate": round(float(op_rb_tr["strict_15_ok"].mean()), 4),
        "opus_after_astra62_n": len(op_rb_a62),
        "opus_after_astra62_pass": int(op_rb_a62["strict_15_ok"].sum()),
        "opus_after_astra62_rate": round(float(op_rb_a62["strict_15_ok"].mean()), 4),
    }

    # Worker tier Role-A survival by sender
    worker_role_a: dict[str, dict[str, Any]] = {}
    wk_a = mat_df[(mat_df["dropped_role"] == "A") & (mat_df["receiver_tier"] == "worker")]
    for s in ALL_PARTNER_MODELS:
        sub = wk_a[wk_a["sender_model"] == s]
        if len(sub):
            worker_role_a[s] = {
                "n": len(sub),
                "pass": int(sub["strict_15_ok"].sum()),
                "rate": round(float(sub["strict_15_ok"].mean()), 4),
            }

    # 3. Attempt-level cost ledger replay
    gate_cost = float(
        att_df[att_df["condition"].isin(["floor_no_partner", "ceiling_verbatim_rules"])]["cost_usd"]
        .fillna(0.0)
        .sum()
    )
    handoff_att = att_df[att_df["condition"] == "handoff"]
    accepted_cost = float(handoff_att[handoff_att["status"] == "ok"]["cost_usd"].fillna(0.0).sum())
    trunc_att = handoff_att[handoff_att["status"] == "truncated"]
    trunc_cost = float(trunc_att["cost_usd"].fillna(0.0).sum())
    ok_cids = set(handoff_att[handoff_att["status"] == "ok"]["call_id"].astype(str))
    trunc_cids = set(trunc_att["call_id"].astype(str))
    perm_trunc_att = trunc_att[~trunc_att["call_id"].astype(str).isin(ok_cids)]
    retry_trunc_att = trunc_att[trunc_att["call_id"].astype(str).isin(ok_cids)]
    perm_trunc_cost = float(perm_trunc_att["cost_usd"].fillna(0.0).sum())
    retry_trunc_cost = float(retry_trunc_att["cost_usd"].fillna(0.0).sum())
    total_ledger_cost = float(att_df["cost_usd"].fillna(0.0).sum())

    # Verify every truncated handoff call_id was a withheld handoff & 3-way cost reconciliation holds
    trunc_mat = mat_df[mat_df["call_id"].astype(str).isin(trunc_cids)]
    n_trunc_withheld = int((trunc_mat["sender_r1_r2_full_disclosure"] == False).sum())
    if n_trunc_withheld != len(trunc_cids):
        raise AssertionError(
            f"Exp1 truncated handoffs not 100% withheld: {n_trunc_withheld}/{len(trunc_cids)}"
        )
    if abs((gate_cost + accepted_cost + trunc_cost) - total_ledger_cost) >= 1e-4:
        raise AssertionError(
            f"Exp1 3-way cost reconciliation failed: gate({gate_cost:.6f}) + accepted({accepted_cost:.6f}) "
            f"+ trunc({trunc_cost:.6f}) != total({total_ledger_cost:.6f})"
        )
    if abs((perm_trunc_cost + retry_trunc_cost) - trunc_cost) >= 1e-4:
        raise AssertionError(
            f"Exp1 truncation split mismatch: perm({perm_trunc_cost:.6f}) + retry({retry_trunc_cost:.6f}) "
            f"!= trunc({trunc_cost:.6f})"
        )

    if ref_exp1_path.exists():
        ref = json.loads(ref_exp1_path.read_text(encoding="utf-8"))
        if s1["overview"]["n_cases"] != ref["overview"]["n_cases"]:
            raise AssertionError(
                f"Exp1 n_cases mismatch: {s1['overview']['n_cases']} vs {ref['overview']['n_cases']}"
            )
        if s1["overview"]["n_ok"] != ref["overview"]["n_ok"]:
            raise AssertionError(
                f"Exp1 n_ok mismatch: {s1['overview']['n_ok']} vs {ref['overview']['n_ok']}"
            )
        if n_err != ref["overview"]["n_errors"]:
            raise AssertionError(
                f"Exp1 matrix_errors mismatch: {n_err} vs {ref['overview']['n_errors']}"
            )
        if ref["overview"].get("n_truncation_errors") == 14:
            if n_err != 14:
                raise AssertionError(f"Expected 14 matrix errors, got {n_err}")
            if err_by_rec != {"neon4": 14}:
                raise AssertionError(f"Expected matrix_errors_by_receiver == {{'neon4': 14}}, got {err_by_rec}")
            if err_by_sender != {"kimi31": 9, "astra62": 5}:
                raise AssertionError(
                    f"Expected matrix_errors_by_sender == {{'kimi31': 9, 'astra62': 5}}, got {err_by_sender}"
                )
            if len(trunc_att) != 58 or len(trunc_cids) != 22 or n_trunc_withheld != 22:
                raise AssertionError(
                    f"Expected 58 Neon truncation attempts across 22 withheld handoffs, got attempts={len(trunc_att)}, cids={len(trunc_cids)}, withheld={n_trunc_withheld}"
                )
            if int(perm_trunc_att["call_id"].nunique()) != 14 or int(retry_trunc_att["call_id"].nunique()) != 8:
                raise AssertionError("Expected 14 permanently truncated + 8 retried-ok Neon handoffs!")
            if len(ds_mat) != 320 or ds_mat_zero_rt != 320:
                raise AssertionError(
                    f"Expected 320/320 DeepSeek V4.1 Flash matrix calls with 0 reasoning tokens, got {ds_mat_zero_rt}/{len(ds_mat)}"
                )
        _assert_close("exp1 overview cost_usd", accepted_cost, ref["overview"]["cost_usd"], tol)
        _assert_close(
            "exp1 ledger_total_cost_usd", total_ledger_cost, ref["ledger_total_cost_usd"], tol
        )
        for s in ALL_PARTNER_MODELS:
            if s in ref["by_sender"]:
                r_s = ref["by_sender"][s]["primary_role_a_cross_vendor"]
                a_s = s1["by_sender"][s]["primary_role_a_cross_vendor"]
                _assert_close(
                    f"exp1 sender {s} Role-A CV ITT",
                    a_s["strict_all_pass_rate"],
                    r_s["strict_all_pass_rate"],
                    tol,
                )
                _assert_close(
                    f"exp1 sender {s} Role-A CV completed-only",
                    a_s["strict_all_pass_rate_ok"],
                    r_s["strict_all_pass_rate_ok"],
                    tol,
                )
        ref_w = ref["within_astra62_shared_vs_withheld"]["primary_role_a_cross_vendor_r1_r2"]
        _assert_close(
            "exp1 within-GPT-6.2 Role-A CV shared",
            w_a62["shared"]["strict_all_pass_rate"],
            ref_w["shared"]["strict_all_pass_rate"],
            tol,
        )
        _assert_close(
            "exp1 within-GPT-6.2 Role-A CV withheld ITT",
            w_a62["withheld"]["strict_all_pass_rate"],
            ref_w["withheld"]["strict_all_pass_rate"],
            tol,
        )
        _assert_close(
            "exp1 within-GPT-6.2 Role-A CV withheld ok",
            w_a62["withheld"]["strict_all_pass_rate_ok"],
            ref_w["withheld"]["strict_all_pass_rate_ok"],
            tol,
        )
        _assert_close(
            "exp1 within-GPT-6.2 Role-A CV R1-R2 episode_perm_p_strict",
            w_a62.get("episode_perm_p_strict"),
            ref_w.get("episode_perm_p_strict"),
            tol,
        )
        w_a62_r1 = s1["within_astra62_shared_vs_withheld"]["primary_role_a_cross_vendor_r1"]
        ref_w_r1 = ref["within_astra62_shared_vs_withheld"].get("primary_role_a_cross_vendor_r1")
        if ref_w_r1 is not None:
            _assert_close(
                "exp1 within-GPT-6.2 Role-A CV R1 episode_perm_p_strict",
                w_a62_r1.get("episode_perm_p_strict"),
                ref_w_r1.get("episode_perm_p_strict"),
                tol,
            )

    return {
        "reproduced": True,
        "regraded_raw_submissions": regraded_gate_n + regraded_mat_n,
        "matrix_native_int_strict_pass": native_int_strict_pass_n,
        "matrix_string_coerced_strict_pass": str_coerced_strict_pass_n,
        "gate_total_rows": n_gate_total,
        "gate_floor_rows": n_gate_floor,
        "gate_ceiling_rows": n_gate_ceil,
        "gate_deepseek_flash_zero_reasoning": f"{ds_zero_rt}/{len(ds_gate)}",
        "gate_floor_min_max": [
            round(min(floor_rates.values()), 4),
            round(max(floor_rates.values()), 4),
        ]
        if floor_rates
        else None,
        "gate_ceiling_min_max": [
            round(min(ceil_rates.values()), 4),
            round(max(ceil_rates.values()), 4),
        ]
        if ceil_rates
        else None,
        "matrix_total": len(mat_df),
        "matrix_ok": int((mat_df["ok"] == True).sum()),
        "matrix_deepseek_flash_zero_reasoning": f"{ds_mat_zero_rt}/{len(ds_mat)}",
        "matrix_errors": n_err,
        "matrix_errors_by_receiver": err_by_rec,
        "matrix_errors_by_sender": err_by_sender,
        "role_a_cv_by_sender": role_a_cv_by_sender,
        "within_astra62_role_a_cv": {
            "shared_n": w_a62["shared"]["n_cases"],
            "shared_rate": w_a62["shared"]["strict_all_pass_rate"],
            "withheld_n": w_a62["withheld"]["n_cases"],
            "withheld_itt_rate": w_a62["withheld"]["strict_all_pass_rate"],
            "withheld_ok_rate": w_a62["withheld"]["strict_all_pass_rate_ok"],
        },
        "astra62_role_a_same_family": {
            "n": len(sf_a62_df),
            "pass": int(sf_a62_df["strict_15_ok"].sum()),
            "rate": round(sf_a62_rate, 4) if sf_a62_rate is not None else None,
        },
        "role_b_contagion": role_b_contagion,
        "worker_tier_role_a_by_sender": worker_role_a,
        "cost_ledger": {
            "attempts_total": len(att_df),
            "gate_cost_usd": round(gate_cost, 6),
            "accepted_matrix_cost_usd": round(accepted_cost, 6),
            "neon_truncation_attempts": len(trunc_att),
            "neon_truncation_unique_handoffs": len(trunc_cids),
            "neon_truncation_withheld_handoffs": n_trunc_withheld,
            "neon_truncation_cost_usd": round(trunc_cost, 6),
            "neon_perm_trunc_handoffs": int(perm_trunc_att["call_id"].nunique()),
            "neon_perm_trunc_cost_usd": round(perm_trunc_cost, 6),
            "neon_retried_trunc_handoffs": int(retry_trunc_att["call_id"].nunique()),
            "neon_retried_trunc_cost_usd": round(retry_trunc_cost, 6),
            "total_ledger_cost_usd": round(total_ledger_cost, 6),
        },
    }


def _replay_exp2(
    tables: dict[str, pd.DataFrame],
    ref_exp2_path: Path,
    tol: float,
) -> dict[str, Any]:
    """Independently replay all Experiment 2 claims strictly from exp2_info_flip*.parquet."""
    pairs_df = tables["exp2_info_flip_pairs"]
    ch_df = tables["exp2_info_flip_choices"]

    # 1. Parity & zero-leak checks strictly from pairs.parquet
    max_added_diff = float(pairs_df["max_added_word_rel_diff"].max())
    max_total_diff = float(pairs_df["max_total_word_rel_diff"].max())
    total_vendor_leaks = int(pairs_df["vendor_leak_count"].sum())
    all_echo_zero_leak = bool(pairs_df["all_public_echo_zero_leak"].all())
    all_parity_ok = bool(pairs_df["all_word_parity_ok"].all())

    if max_added_diff > 1e-9 or max_total_diff > 1e-9:
        raise AssertionError(
            f"Exp2 word parity violated: max_added={max_added_diff}, max_total={max_total_diff}"
        )
    if total_vendor_leaks != 0 or not all_echo_zero_leak or not all_parity_ok:
        raise AssertionError(
            f"Exp2 leak/parity violation: leaks={total_vendor_leaks}, zero_leak={all_echo_zero_leak}, parity={all_parity_ok}"
        )

    # 2. Crossover & reviewer replay strictly from choices.parquet
    choice_rows = [_clean_record(r) for r in ch_df.to_dict(orient="records")]
    s2 = summarize_exp2_results(choice_rows, capstone_dir=None, n_boot=200, n_perm_mc=200)
    total_spend = round(float(ch_df["cost_usd"].fillna(0.0).sum()), 6)

    if ref_exp2_path.exists():
        ref = json.loads(ref_exp2_path.read_text(encoding="utf-8"))
        if s2["counts"]["n_choices_total"] != ref["counts"]["n_choices_total"]:
            raise AssertionError("Exp2 total choice count mismatch!")
        _assert_close("exp2 total_cost_usd", total_spend, ref["ledger_total_cost_usd"], tol)

        # Placebo arm
        for arm_slice in ("itt_cross_vendor", "itt_all_reviewers"):
            a_ov = s2["placebo_arm"][arm_slice]["overall"]
            r_ov = ref["placebo_arm"][arm_slice]["overall"]
            if a_ov["n_cases"] != r_ov["n_cases"]:
                raise AssertionError(f"Exp2 placebo {arm_slice} n_cases mismatch")
            _assert_close(
                f"exp2 placebo {arm_slice} mean_delta_echo",
                a_ov["mean_delta_echo"],
                r_ov["mean_delta_echo"],
                tol,
            )
            if a_ov["n_switched_toward_echo"] != r_ov["n_switched_toward_echo"]:
                raise AssertionError(f"Exp2 placebo {arm_slice} switches mismatch")

        # Main arm overall + by_task + by_reviewer
        for arm_slice in ("itt_cross_vendor", "itt_all_reviewers"):
            a_ov = s2["main_arm"][arm_slice]["overall"]
            r_ov = ref["main_arm"][arm_slice]["overall"]
            if a_ov["n_cases"] != r_ov["n_cases"]:
                raise AssertionError(f"Exp2 main {arm_slice} n_cases mismatch")
            _assert_close(
                f"exp2 main {arm_slice} mean_delta_info",
                a_ov["mean_delta_info"],
                r_ov["mean_delta_info"],
                tol,
            )
            _assert_close(
                f"exp2 main {arm_slice} pair_sign_flip_perm_p",
                a_ov["pair_sign_flip_perm_p"],
                r_ov["pair_sign_flip_perm_p"],
                tol,
            )
            if a_ov["n_switched_toward_info"] != r_ov["n_switched_toward_info"]:
                raise AssertionError(f"Exp2 main {arm_slice} n_switched_toward_info mismatch")
            if a_ov["n_switched_away_from_info"] != r_ov["n_switched_away_from_info"]:
                raise AssertionError(f"Exp2 main {arm_slice} n_switched_away_from_info mismatch")

        for task_k in ("goalpost", "triage"):
            _assert_close(
                f"exp2 main CV {task_k} mean_delta_info",
                s2["main_arm"]["itt_cross_vendor"]["by_task"][task_k]["mean_delta_info"],
                ref["main_arm"]["itt_cross_vendor"]["by_task"][task_k]["mean_delta_info"],
                tol,
            )
        for rv in ("neon4", "astra62", "opus55", "kimi31"):
            _assert_close(
                f"exp2 main pooled reviewer {rv}",
                s2["main_arm"]["itt_all_reviewers"]["by_reviewer"][rv]["mean_delta_info"],
                ref["main_arm"]["itt_all_reviewers"]["by_reviewer"][rv]["mean_delta_info"],
                tol,
            )

    main_cv = s2["main_arm"]["itt_cross_vendor"]
    main_all = s2["main_arm"]["itt_all_reviewers"]
    plac_cv = s2["placebo_arm"]["itt_cross_vendor"]["overall"]
    plac_all = s2["placebo_arm"]["itt_all_reviewers"]["overall"]

    return {
        "reproduced": True,
        "pairs_total": len(pairs_df),
        "pairs_main": int((pairs_df["arm"] == "main").sum()),
        "pairs_placebo": int((pairs_df["arm"] == "placebo").sum()),
        "max_added_word_rel_diff": max_added_diff,
        "max_total_word_rel_diff": max_total_diff,
        "vendor_leaks": total_vendor_leaks,
        "choices_total": len(ch_df),
        "choices_main": int((ch_df["arm"] == "main").sum()),
        "choices_placebo": int((ch_df["arm"] == "placebo").sum()),
        "placebo_pooled": {
            "n_cases": plac_all["n_cases"],
            "mean_delta_echo": plac_all["mean_delta_echo"],
            "switched_toward": plac_all["n_switched_toward_echo"],
        },
        "placebo_cv": {
            "n_cases": plac_cv["n_cases"],
            "mean_delta_echo": plac_cv["mean_delta_echo"],
            "switched_toward": plac_cv["n_switched_toward_echo"],
        },
        "main_cv": {
            "n_cases": main_cv["overall"]["n_cases"],
            "mean_delta_info": main_cv["overall"]["mean_delta_info"],
            "switched_toward": main_cv["overall"]["n_switched_toward_info"],
            "switched_away": main_cv["overall"]["n_switched_away_from_info"],
            "goalpost_delta_info": main_cv["by_task"]["goalpost"]["mean_delta_info"],
            "triage_delta_info": main_cv["by_task"]["triage"]["mean_delta_info"],
        },
        "main_pooled": {
            "n_cases": main_all["overall"]["n_cases"],
            "mean_delta_info": main_all["overall"]["mean_delta_info"],
            "switched_toward": main_all["overall"]["n_switched_toward_info"],
            "switched_away": main_all["overall"]["n_switched_away_from_info"],
            "by_reviewer": {
                rv: main_all["by_reviewer"][rv]["mean_delta_info"]
                for rv in ("neon4", "astra62", "opus55", "kimi31")
                if rv in main_all["by_reviewer"]
            },
        },
        "total_spend_usd": total_spend,
    }


def verify_hf_replay(
    run_name: str,
    *,
    capstone_run: str | None = None,
    tol: float = 1e-3,
) -> dict[str, Any]:
    run_dir = DATA_DIR / run_name
    hf_dir = run_dir / "hf_export"
    report_path = run_dir / "analysis" / "report.json"

    if not hf_dir.exists():
        raise FileNotFoundError(f"Missing hf_export directory: {hf_dir}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing analysis report: {report_path}")

    # 1. Hygiene & Zero-Deprecated-Fat verification (loads all YAML-registered Parquet tables)
    hyg = _verify_hygiene(hf_dir)
    tables: dict[str, pd.DataFrame] = hyg.pop("tables")

    ep_df = tables["episodes"]
    turns_df = tables["turns"]
    deb_df = tables["debriefs"]
    ch_df = tables["choices"]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # 2. Exact count verification
    counts: dict[str, Any] = {
        "episodes_total": len(ep_df),
        "episodes_main": int((ep_df["arm"] != "opus_boost").sum()),
        "episodes_opus_boost": int((ep_df["arm"] == "opus_boost").sum()),
        "failed_episodes": len(tables["failed_episodes"]) if "failed_episodes" in tables else 0,
        "turns_total": len(turns_df),
        "turns_main": int((turns_df["arm"] != "opus_boost").sum()),
        "turns_ledger_joined": int(turns_df["ledger_joined"].sum()) if "ledger_joined" in turns_df.columns else 0,
        "debriefs_total": len(deb_df),
        "debriefs_by_fork": {
            str(k): int(v) for k, v in deb_df["fork"].value_counts().sort_index().items()
        },
        "choices_total": len(ch_df),
        "choices_excluded": len(tables["choices_excluded"]) if "choices_excluded" in tables else 0,
    }
    if "capstone_tone_flip" in tables:
        counts["capstone_choices"] = len(tables["capstone_tone_flip"])
    if "capstone_rewrites" in tables:
        counts["capstone_rewrites_rows"] = len(tables["capstone_rewrites"])
        counts["capstone_rewrites_candidates"] = int(tables["capstone_rewrites"]["episode_id"].nunique())
    if "exp1_dropout_gate" in tables:
        counts["exp1_dropout_gate"] = len(tables["exp1_dropout_gate"])
    if "exp1_dropout" in tables:
        counts["exp1_dropout"] = len(tables["exp1_dropout"])
    if "exp1_dropout_attempts" in tables:
        counts["exp1_dropout_attempts"] = len(tables["exp1_dropout_attempts"])
    if "exp2_info_flip_pairs" in tables:
        counts["exp2_info_flip_pairs"] = len(tables["exp2_info_flip_pairs"])
    if "exp2_info_flip_choices" in tables:
        counts["exp2_info_flip_choices"] = len(tables["exp2_info_flip_choices"])

    if counts["episodes_total"] != report["n_episodes"]:
        raise AssertionError(
            f"Episode count mismatch: parquet={counts['episodes_total']}, report={report['n_episodes']}"
        )
    if counts["choices_total"] != report["n_choices_total"]:
        raise AssertionError(
            f"Choice count mismatch: parquet={counts['choices_total']}, report={report['n_choices_total']}"
        )

    # 3. Main v2 League (Unified 7-Partner BT + raw direct win rates), H1b, and F1_retest ICC(2,1)
    bt_replay = _fit_unified_bt_model(ch_df, adjust_score=True, n_boot=0)
    if not bt_replay.get("ok"):
        raise AssertionError(f"HF replay BT failed: {bt_replay}")
    ref_bt = report["unified_bt_adjusted"]
    for p, val in ref_bt["bt_avg_h2h_win_rate"].items():
        _assert_close(f"BT avg H2H {p}", bt_replay["bt_avg_h2h_win_rate"].get(p), val, tol)
    for p, val in ref_bt["raw_direct_win_rate"].items():
        _assert_close(f"Raw direct win rate {p}", bt_replay["raw_direct_win_rate"].get(p), val, tol)

    h1b_replay = _compute_direct_predecessor_analysis(ch_df)
    ref_h1b = report["H1b_predecessor_head_to_head"]
    for slice_k in ("main_shared_seed_cross_vendor", "opus_boost_same_seed", "opus_combined_same_seed"):
        rv = (ref_h1b.get(slice_k) or {}).get("astra62_win_rate")
        hv = (h1b_replay.get(slice_k) or {}).get("astra62_win_rate")
        _assert_close(f"H1b {slice_k}", hv, rv, tol)
        rp = (ref_h1b.get(slice_k) or {}).get("primary_cluster_perm_p")
        hp = (h1b_replay.get(slice_k) or {}).get("primary_cluster_perm_p")
        if rp is not None or hp is not None:
            _assert_close(f"H1b {slice_k} primary_cluster_perm_p", hp, rp, tol)

    opus_gp = ((h1b_replay.get("opus_combined_same_seed") or {}).get("by_task") or {}).get("goalpost") or {}
    ref_opus_gp = ((ref_h1b.get("opus_combined_same_seed") or {}).get("by_task") or {}).get("goalpost") or {}
    _assert_close(
        "H1b Opus Goalpost astra62_win_rate",
        opus_gp.get("astra62_win_rate"),
        ref_opus_gp.get("astra62_win_rate"),
        tol,
    )

    f1_retest_replay = _replay_f1_retest(deb_df)
    ref_f1r = report.get("F1_retest_reliability") or {}
    f1_retest_ok = False
    if ref_f1r.get("ok"):
        if f1_retest_replay.get("n") != ref_f1r.get("n"):
            raise AssertionError(
                f"F1_retest n mismatch: parquet={f1_retest_replay.get('n')}, report={ref_f1r.get('n')}"
            )
        _assert_close("F1_retest ICC(2,1)", f1_retest_replay.get("icc_2_1"), ref_f1r.get("icc_2_1"), tol)
        f1_retest_ok = True

    # 4. Capstone Tone+Readability Replay (when capstone tables present)
    cap_verified = False
    cap_summary: dict[str, Any] | None = None
    cap_choices_list: list[dict[str, Any]] | None = None
    cap_rewrites_list: list[dict[str, Any]] | None = None

    if "capstone_tone_flip" in tables:
        cap_df = tables["capstone_tone_flip"]
        if "capstone_rewrites" not in tables:
            raise AssertionError("Missing capstone_rewrites.parquet alongside capstone_tone_flip.parquet!")
        cap_rw_df = tables["capstone_rewrites"]
        cap_choices_list = [_clean_record(r) for r in cap_df.to_dict(orient="records")]
        cap_rewrites_list = []
        for row in cap_rw_df.to_dict(orient="records"):
            r = _clean_record(row)
            r["turns"] = _json_loads_opt(r.pop("turns_json", None)) or []
            cap_rewrites_list.append(r)

        cv_cap = cap_df[(cap_df["cross_vendor"] == True) & cap_df["chosen_slot"].isin([1, 2])]
        deltas: list[float] = []
        for (_pid, _rv), g in cv_cap.groupby(["pair_id", "reviewer"]):
            w = g[g["condition"] == "p1_warm_p2_blunt"]
            b = g[g["condition"] == "p1_blunt_p2_warm"]
            if len(w) == 2 and len(b) == 2:
                deltas.append(
                    float((w["chosen_canonical"] == "P1").mean())
                    - float((b["chosen_canonical"] == "P1").mean())
                )
        rep_dw = round(float(np.mean(deltas)), 4) if deltas else None

        cap_res_name = capstone_run or f"{run_name}_capstone"
        cap_res_path = DATA_DIR / cap_res_name / "capstone_results.json"
        if cap_res_path.exists():
            cap_res = json.loads(cap_res_path.read_text(encoding="utf-8"))
            ref_dw = cap_res["crossover_summary"]["itt_cross_vendor"]["overall"].get("mean_delta_warm")
            _assert_close("Capstone CV mean_delta_warm", rep_dw, ref_dw, tol)

        cap_audit = compute_capstone_audit(cap_rewrites_list, cap_choices_list)
        wb_dims = cap_audit["turn_level_deltas"]["warm_minus_blunt"]["dimensions"]
        cap_summary = {
            "cv_mean_delta_warm": rep_dw,
            "unanimous_switches_toward_warm": cap_audit["flip_verification"][
                "unanimous_flips_toward_warm"
            ],
            "unanimous_switches_away_from_warm": cap_audit["flip_verification"][
                "unanimous_flips_away_from_warm"
            ],
            "n_reviewer_pair_cases": cap_audit["flip_verification"]["n_reviewer_pair_cases"],
            "four_lab_unanimity": (
                f"{cap_audit['inter_lab_consensus']['all_pairs'].get('unanimous_4_0')}/"
                f"{cap_audit['inter_lab_consensus']['all_pairs'].get('n_pairs')}"
            ),
            "warmth_delta": wb_dims["warmth"]["delta_warm_minus_blunt"],
            "clarity_delta": wb_dims["clarity"]["delta_warm_minus_blunt"],
        }
        cap_verified = True

    # 5. Phase 1 Meta-Analysis v2.1 Replay strictly from Parquet (when meta_v2_1.json exists)
    meta_path = run_dir / "analysis" / "meta_v2_1.json"
    meta_verified = False
    meta_summary: dict[str, Any] | None = None
    if meta_path.exists():
        ref_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        episodes_all = _reconstruct_episodes_from_parquet(ep_df, turns_df, deb_df)
        episodes_main = [e for e in episodes_all if e.get("arm") != "opus_boost"]
        choices_list = [_clean_record(r) for r in ch_df.to_dict(orient="records")]

        # 5a. Construct validity & length-residualized clarity
        cv_rep = compute_construct_validity(episodes_main)
        clarity_resid_map = cv_rep.pop("clarity_resid_map")
        ref_cv = ref_meta["construct_validity"]
        if cv_rep["n_turns"] != ref_cv["n_turns"]:
            raise AssertionError(f"Construct validity n_turns mismatch: {cv_rep['n_turns']} vs {ref_cv['n_turns']}")
        _assert_close(
            "clarity OLS R^2",
            cv_rep["ols_length_models"]["clarity"]["r_squared"],
            ref_cv["ols_length_models"]["clarity"]["r_squared"],
            tol,
        )
        for m in ALL_PARTNER_MODELS:
            _assert_close(
                f"clarity_raw {m}",
                cv_rep["per_model"][m]["clarity_raw"],
                ref_cv["per_model"][m]["clarity_raw"],
                tol,
            )
            _assert_close(
                f"clarity_resid {m}",
                cv_rep["per_model"][m]["clarity_resid"],
                ref_cv["per_model"][m]["clarity_resid"],
                tol,
            )

        # 5b. Politeness rationale audit & Visibility-masked Partner-FE cluster-robust choice logits
        lg_rep = compute_choice_logits(
            episodes_all,
            choices_list,
            clarity_resid_map,
            capstone_choices=cap_choices_list,
        )
        ref_lg = ref_meta["choice_logits"]
        pol_rep = lg_rep["tone_and_rationale_audit"]["live_non_aa"]["strict_politeness_mentions"]["count"]
        pol_n = lg_rep["tone_and_rationale_audit"]["live_non_aa"]["n_reasons"]
        pol_ref = ref_lg["tone_and_rationale_audit"]["live_non_aa"]["strict_politeness_mentions"]["count"]
        if pol_rep != pol_ref or pol_n != ref_lg["tone_and_rationale_audit"]["live_non_aa"]["n_reasons"]:
            raise AssertionError(f"Politeness audit mismatch: {pol_rep}/{pol_n} vs {pol_ref}")

        for spec_name, preds_to_check in (
            (
                "all_non_aa_with_fe",
                (
                    "d_answers_directly",
                    "d_s_conf",
                    "d_wooden_legalistic",
                    "d_ids_per_100",
                    "d_clarity_resid",
                    "d_skips_why",
                    "d_warmth",
                ),
            ),
            (
                "exact_ties_with_fe",
                (
                    "d_answers_directly",
                    "d_s_conf",
                    "d_wooden_legalistic",
                    "d_ids_per_100",
                    "d_clarity_resid",
                    "d_skips_why",
                    "d_warmth",
                ),
            ),
        ):
            a_m = lg_rep["multivariate"][spec_name]
            r_m = ref_lg["multivariate"][spec_name]
            if a_m["n_choices"] != r_m["n_choices"] or a_m["n_clusters"] != r_m["n_clusters"]:
                raise AssertionError(
                    f"Logit {spec_name} dims mismatch: ({a_m['n_choices']}, {a_m['n_clusters']}) vs ({r_m['n_choices']}, {r_m['n_clusters']})"
                )
            for pred in preds_to_check:
                _assert_close(
                    f"Logit {spec_name} {pred} coef",
                    a_m["coefs"][pred]["coef"],
                    r_m["coefs"][pred]["coef"],
                    tol,
                )

        # 5c. Disclosure meta-analysis (Within-GPT-6.2, Producer-Consumer, Role-B Contagion)
        disc_rep = compute_disclosure_meta_analysis(
            episodes_main, choices_list, episodes_all=episodes_all
        )
        ref_disc = ref_meta["disclosure_meta_analysis"]
        for m in ALL_PARTNER_MODELS:
            _assert_close(
                f"Goalpost Role-A >=5 {m}",
                disc_rep["goalpost_disclosure_table_main"]["by_model"][m][
                    "r1_role_a_full_formula_rate"
                ],
                ref_disc["goalpost_disclosure_table_main"]["by_model"][m][
                    "r1_role_a_full_formula_rate"
                ],
                tol,
            )

        eb_rep = disc_rep["within_gpt62_eval_blindness"]["score_eq_1_0_only"]
        eb_ref = ref_disc["within_gpt62_eval_blindness"]["score_eq_1_0_only"]
        for grp in ("shared", "withheld"):
            if eb_rep[grp]["n_episodes"] != eb_ref[grp]["n_episodes"]:
                raise AssertionError(f"Within-GPT-6.2 {grp} n_episodes mismatch!")
            _assert_close(
                f"Within-GPT-6.2 {grp} F1",
                eb_rep[grp]["cross_vendor_debriefs"]["mean_f1_again"],
                eb_ref[grp]["cross_vendor_debriefs"]["mean_f1_again"],
                tol,
            )
            _assert_close(
                f"Within-GPT-6.2 {grp} F4 KEEP",
                eb_rep[grp]["cross_vendor_debriefs"]["f4_keep_rate"],
                eb_ref[grp]["cross_vendor_debriefs"]["f4_keep_rate"],
                tol,
            )
            _assert_close(
                f"Within-GPT-6.2 {grp} CV choice overall",
                eb_rep[grp]["cross_vendor_main_7p_choice_overall"]["win_rate"],
                eb_ref[grp]["cross_vendor_main_7p_choice_overall"]["win_rate"],
                tol,
            )
            _assert_close(
                f"Within-GPT-6.2 {grp} CV choice ties",
                eb_rep[grp]["cross_vendor_main_7p_choice_exact_ties"]["win_rate"],
                eb_ref[grp]["cross_vendor_main_7p_choice_exact_ties"]["win_rate"],
                tol,
            )

        for subj in CORE_MODELS:
            _assert_close(
                f"Producer-consumer tie hire {subj}",
                disc_rep["producer_consumer_hypocrisy"]["all_non_aa_exact_ties"]["by_chooser"][
                    subj
                ]["win_rate"],
                ref_disc["producer_consumer_hypocrisy"]["all_non_aa_exact_ties"]["by_chooser"][
                    subj
                ]["win_rate"],
                tol,
            )

        for k_field in ("mean_rules_after_role_a_disclosed", "mean_rules_after_role_a_withheld"):
            _assert_close(
                f"Role-B contagion pooled {k_field}",
                disc_rep["contagious_withholding"]["pooled_all_role_b"][k_field],
                ref_disc["contagious_withholding"]["pooled_all_role_b"][k_field],
                tol,
            )
        op_rep = disc_rep["contagious_withholding"]["role_b_opus55_transparent_vs_withholding_partners"]
        op_ref = ref_disc["contagious_withholding"]["role_b_opus55_transparent_vs_withholding_partners"]
        for k_field in (
            "mean_rules_after_transparent",
            "full_disclosure_rate_after_transparent",
            "mean_rules_after_withholding",
            "full_disclosure_rate_after_withholding",
        ):
            _assert_close(f"Role-B Opus contagion {k_field}", op_rep[k_field], op_ref[k_field], tol)

        # 5d. Compute Bloat Tax strictly from Parquet (turns.parquet latency_s + accepted_cost_usd)
        bl_rep = _compute_bloat_tax_from_parquet(episodes_main)
        ref_bl = ref_meta["bloat_tax_and_contagion"]
        if bl_rep["n_core_subject_turns_checked"] != ref_bl["ledger_join"]["n_core_subject_turns_checked"]:
            raise AssertionError("Bloat core subject turns count mismatch!")
        if bl_rep["n_ledger_matched"] != ref_bl["ledger_join"]["n_ledger_matched"]:
            raise AssertionError("Bloat ledger-matched turns count mismatch!")
        if bl_rep["n_blocks"] != ref_bl["exposed_turns_primary"]["overall"]["n_blocks"]:
            raise AssertionError("Bloat n_blocks mismatch!")
        ref_pct = ref_bl["exposed_turns_primary"]["overall"]["pct_delta_by_metric"]
        for metric in ("prompt_tokens", "reasoning_tokens", "word_count", "latency_ms", "cost_usd"):
            for partner in ALL_PARTNER_MODELS:
                _assert_close(
                    f"Bloat pct {metric} {partner}",
                    bl_rep["pct_delta_by_metric"][metric].get(partner),
                    ref_pct[metric]["by_partner"][partner]["mean"],
                    tol=1e-2,  # percentage points (0.01 pp tolerance)
                )
        ref_conf = ref_bl["exposed_turns_primary"]["overall"]["abs_delta_by_metric"][
            "expresses_confusion"
        ]["by_partner"]
        _assert_close(
            "Bloat Qwen4 expresses_confusion raw_mean",
            bl_rep["expresses_confusion_raw_mean"].get("qwen4"),
            ref_conf["qwen4"]["raw_mean"],
            tol,
        )

        # 5e. Incident Triage deadlocks & Opus s304 upsets
        tr_rep = compute_triage_paralysis_and_upsets(episodes_main, choices_list, ledger_path=None)
        ref_tr = ref_meta["triage_paralysis_and_upsets"]
        n_deadlocks = tr_rep["triage_below_099_inventory"]["astra60_astra62_00_deadlocks_count"]
        n_opus_upsets = len(tr_rep["score_upsets"]["opus_triage_s304_035_over_100_receipts"])
        if n_deadlocks != ref_tr["triage_below_099_inventory"]["astra60_astra62_00_deadlocks_count"]:
            raise AssertionError(f"Triage deadlocks mismatch: {n_deadlocks}")
        if n_opus_upsets != len(ref_tr["score_upsets"]["opus_triage_s304_035_over_100_receipts"]):
            raise AssertionError(f"Triage Opus upsets mismatch: {n_opus_upsets}")

        # 5f. Capstone audit cross-check against meta_v2_1.json
        if cap_summary is not None and "capstone_audit" in ref_meta:
            ref_cap = ref_meta["capstone_audit"]
            _assert_close(
                "Capstone warmth delta",
                cap_summary["warmth_delta"],
                ref_cap["turn_level_deltas"]["warm_minus_blunt"]["dimensions"]["warmth"][
                    "delta_warm_minus_blunt"
                ],
                tol,
            )
            _assert_close(
                "Capstone clarity delta",
                cap_summary["clarity_delta"],
                ref_cap["turn_level_deltas"]["warm_minus_blunt"]["dimensions"]["clarity"][
                    "delta_warm_minus_blunt"
                ],
                tol,
            )
            if (
                cap_summary["unanimous_switches_toward_warm"]
                != ref_cap["flip_verification"]["unanimous_flips_toward_warm"]
            ):
                raise AssertionError("Capstone unanimous switch count mismatch!")
            ref_u = f"{ref_cap['inter_lab_consensus']['all_pairs']['unanimous_4_0']}/{ref_cap['inter_lab_consensus']['all_pairs']['n_pairs']}"
            if cap_summary["four_lab_unanimity"] != ref_u:
                raise AssertionError(f"Capstone 4-lab unanimity mismatch: {cap_summary['four_lab_unanimity']} vs {ref_u}")

        meta_verified = True
        meta_summary = {
            "politeness_mentions_live_non_aa": f"{pol_rep}/{pol_n}",
            "clarity_ols_r2": cv_rep["ols_length_models"]["clarity"]["r_squared"],
            "qwen4_clarity_raw_to_resid": [
                cv_rep["per_model"]["qwen4"]["clarity_raw"],
                cv_rep["per_model"]["qwen4"]["clarity_resid"],
            ],
            "logits_all_non_aa_fe": {
                "n_choices": lg_rep["multivariate"]["all_non_aa_with_fe"]["n_choices"],
                "n_clusters": lg_rep["multivariate"]["all_non_aa_with_fe"]["n_clusters"],
                "coefs": {
                    k: lg_rep["multivariate"]["all_non_aa_with_fe"]["coefs"][k]["coef"]
                    for k in (
                        "d_answers_directly",
                        "d_s_conf",
                        "d_wooden_legalistic",
                        "d_ids_per_100",
                        "d_clarity_resid",
                        "d_skips_why",
                        "d_warmth",
                    )
                },
            },
            "logits_exact_ties_fe": {
                "n_choices": lg_rep["multivariate"]["exact_ties_with_fe"]["n_choices"],
                "n_clusters": lg_rep["multivariate"]["exact_ties_with_fe"]["n_clusters"],
                "coefs": {
                    k: lg_rep["multivariate"]["exact_ties_with_fe"]["coefs"][k]["coef"]
                    for k in (
                        "d_answers_directly",
                        "d_s_conf",
                        "d_wooden_legalistic",
                        "d_ids_per_100",
                        "d_clarity_resid",
                        "d_skips_why",
                        "d_warmth",
                    )
                },
            },
            "within_gpt62_score_1_0": {
                "shared_n": eb_rep["shared"]["n_episodes"],
                "withheld_n": eb_rep["withheld"]["n_episodes"],
                "f1_again": [
                    eb_rep["shared"]["cross_vendor_debriefs"]["mean_f1_again"],
                    eb_rep["withheld"]["cross_vendor_debriefs"]["mean_f1_again"],
                ],
                "f4_keep": [
                    eb_rep["shared"]["cross_vendor_debriefs"]["f4_keep_rate"],
                    eb_rep["withheld"]["cross_vendor_debriefs"]["f4_keep_rate"],
                ],
                "cv_choice_overall": [
                    eb_rep["shared"]["cross_vendor_main_7p_choice_overall"]["win_rate"],
                    eb_rep["withheld"]["cross_vendor_main_7p_choice_overall"]["win_rate"],
                ],
                "cv_choice_ties": [
                    f"{eb_rep['shared']['cross_vendor_main_7p_choice_exact_ties']['wins']}/{eb_rep['shared']['cross_vendor_main_7p_choice_exact_ties']['n_choices']} ({eb_rep['shared']['cross_vendor_main_7p_choice_exact_ties']['win_rate']})",
                    f"{eb_rep['withheld']['cross_vendor_main_7p_choice_exact_ties']['wins']}/{eb_rep['withheld']['cross_vendor_main_7p_choice_exact_ties']['n_choices']} ({eb_rep['withheld']['cross_vendor_main_7p_choice_exact_ties']['win_rate']})",
                ],
            },
            "producer_role_a_r1_ge5": {
                m: disc_rep["goalpost_disclosure_table_main"]["by_model"][m][
                    "r1_role_a_full_formula_rate"
                ]
                for m in ALL_PARTNER_MODELS
            },
            "consumer_tie_discloser_hire": {
                s: disc_rep["producer_consumer_hypocrisy"]["all_non_aa_exact_ties"]["by_chooser"][
                    s
                ]["win_rate"]
                for s in CORE_MODELS
            },
            "role_b_contagion": {
                "pooled_mean_rules": [
                    disc_rep["contagious_withholding"]["pooled_all_role_b"][
                        "mean_rules_after_role_a_disclosed"
                    ],
                    disc_rep["contagious_withholding"]["pooled_all_role_b"][
                        "mean_rules_after_role_a_withheld"
                    ],
                ],
                "opus_mean_rules": [
                    op_rep["mean_rules_after_transparent"],
                    op_rep["mean_rules_after_withholding"],
                ],
                "opus_full_rate": [
                    op_rep["full_disclosure_rate_after_transparent"],
                    op_rep["full_disclosure_rate_after_withholding"],
                ],
            },
            "bloat_tax": {
                "matched_turns": bl_rep["n_ledger_matched"],
                "exposed_turns": bl_rep["n_exposed_turns"],
                "n_blocks": bl_rep["n_blocks"],
                "qwen4_pct": {
                    m: bl_rep["pct_delta_by_metric"][m]["qwen4"]
                    for m in ("prompt_tokens", "reasoning_tokens", "word_count", "latency_ms", "cost_usd")
                },
                "qwen4_confusion_raw_mean": bl_rep["expresses_confusion_raw_mean"]["qwen4"],
                "neon4_pct": {
                    m: bl_rep["pct_delta_by_metric"][m]["neon4"]
                    for m in ("reasoning_tokens", "latency_ms", "cost_usd")
                },
            },
            "triage_astra_deadlocks": n_deadlocks,
            "triage_opus_s304_upsets": n_opus_upsets,
        }

    # 6. Experiment 1 Replay (when exp1_dropout tables present)
    exp1_summary: dict[str, Any] | None = None
    if {"exp1_dropout_gate", "exp1_dropout", "exp1_dropout_attempts"} <= set(tables.keys()):
        exp1_summary = _replay_exp1(
            tables, hf_dir, run_dir / "exp1_dropout" / "exp1_summary.json", tol
        )

    # 7. Experiment 2 Replay (when exp2_info_flip tables present)
    exp2_summary: dict[str, Any] | None = None
    if {"exp2_info_flip_pairs", "exp2_info_flip_choices"} <= set(tables.keys()):
        exp2_summary = _replay_exp2(
            tables, run_dir / "exp2_info_flip" / "exp2_summary.json", tol
        )

    return {
        "ok": True,
        "run_name": run_name,
        "hygiene": hyg,
        "counts": counts,
        "bt_reproduced": True,
        "h1b_reproduced": True,
        "h1b_opus_goalpost_predecessor_win_rate": (
            round(1.0 - float(opus_gp["astra62_win_rate"]), 4)
            if opus_gp.get("astra62_win_rate") is not None
            else None
        ),
        "f1_retest_reproduced": f1_retest_ok,
        "f1_retest": f1_retest_replay,
        "capstone_reproduced": cap_verified,
        "capstone": cap_summary,
        "meta_v2_1_reproduced": meta_verified,
        "meta_v2_1": meta_summary,
        "exp1_reproduced": bool(exp1_summary and exp1_summary.get("reproduced")),
        "exp1_dropout": exp1_summary,
        "exp2_reproduced": bool(exp2_summary and exp2_summary.get("reproduced")),
        "exp2_info_flip": exp2_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify every headline v2.1 claim reproduces strictly from hf_export/*.parquet."
    )
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--capstone-run", type=str, default=None)
    args = parser.parse_args()
    res = verify_hf_replay(args.run, capstone_run=args.capstone_run)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
