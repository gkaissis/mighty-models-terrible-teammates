"""One-command v2 statistical analysis: unified 7-partner BT, 3-judge reliability, LOSO, plots, and quote cards."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

logger = logging.getLogger(__name__)
_GLM_ERRORS = (ValueError, np.linalg.LinAlgError, PerfectSeparationError, FloatingPointError, ZeroDivisionError)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_ground.config import (
    AA_INDEX_V5,
    ALL_PARTNER_MODELS,
    CORE_MODELS,
    DATA_DIR,
    DECISION_PANEL_SPECS,
    DEFAULT_MODELS,
    FAMILY_LABELS,
    PREDECESSOR_MODEL,
)
from common_ground.ledger import Ledger

ORIGINAL_3_TASKS = ("triage", "calendar", "goalpost")


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _model_family(model_key: str) -> str:
    spec = DEFAULT_MODELS.get(model_key)
    return spec.family if spec is not None else "other"


def _load_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Ledger]:
    ep_dir = run_dir / "episodes"
    all_ep_files = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(ep_dir.glob("*.json"))
    ]
    episodes = [e for e in all_ep_files if e.get("complete") is True]
    choices_path = run_dir / "choices.jsonl"
    choices_by_id: dict[str, dict[str, Any]] = {}
    if choices_path.exists():
        for line in choices_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cid = (
                    row.get("manifest_item_id")
                    or f"{row.get('choice_type')}__{row.get('item_id')}__{row.get('order')}"
                )
                choices_by_id[cid] = row
    ledger = Ledger(run_dir / "ledger.jsonl", budget_cap_usd=1e9)
    return episodes, list(choices_by_id.values()), ledger


def _cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261122,
    null_val: float = 0.0,
) -> dict[str, Any]:
    """Cluster bootstrap 95% CI + null-shifted sign-flip cluster permutation p-value + Wald p-value."""
    if len(values) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_boot_achieved": 0}
    mean_val = float(np.mean(values))
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return {
            "mean": round(mean_val, 4),
            "ci_lo": None,
            "ci_hi": None,
            "p_perm": None,
            "p_wald": None,
            "n": len(values),
            "n_clusters": len(uniq),
            "n_boot_requested": n_boot,
            "n_boot_achieved": 0,
        }

    rng = np.random.default_rng(seed)
    by_cl = [values[clusters == c] for c in uniq]
    k = len(by_cl)

    # 1. Cluster bootstrap for 95% CI
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        sample = np.concatenate([by_cl[j] for j in idx])
        boots[i] = float(np.mean(sample))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    se_boot = float(np.std(boots, ddof=1))

    # 2. Proper null-shifted cluster sign-flip permutation p-value (exact under H0: E[x_c] = null_val)
    cl_sums = np.array([float(np.sum(arr - null_val)) for arr in by_cl])
    total_n = float(len(values))
    obs_stat = abs(mean_val - null_val)
    n_perm = n_boot
    signs = rng.choice([-1.0, 1.0], size=(n_perm, k))
    perm_stats = np.abs(signs @ cl_sums) / total_n
    p_perm = float((1 + np.sum(perm_stats >= obs_stat - 1e-12)) / (n_perm + 1))

    # 3. Cluster-bootstrap Wald p-value
    if se_boot > 1e-12:
        z_wald = (mean_val - null_val) / se_boot
        p_wald = float(2.0 * (1.0 - stats.norm.cdf(abs(z_wald))))
    else:
        p_wald = 0.0 if obs_stat > 1e-12 else 1.0

    return {
        "mean": round(mean_val, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "se_boot": round(se_boot, 4),
        "p_perm": round(min(1.0, p_perm), 4),
        "p_perm_fmt": _fmt_p(p_perm),
        "p_wald": round(min(1.0, p_wald), 4),
        "p_wald_fmt": _fmt_p(p_wald),
        "null_val": null_val,
        "n": len(values),
        "n_clusters": int(k),
        "n_boot_requested": n_boot,
        "n_boot_achieved": n_boot,
    }


def krippendorff_alpha(matrix: np.ndarray, *, level: str = "interval") -> float | None:
    """Exact Krippendorff's alpha (interval or ordinal) for an (n_judges, n_units) matrix with NaNs."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    # Keep only units rated by >= 2 judges
    valid_cols = np.sum(~np.isnan(arr), axis=0) >= 2
    sub = arr[:, valid_cols]
    if sub.shape[1] < 2:
        return None

    # Extract pairable values; for ordinal level, replace by pooled mid-ranks (exact Krippendorff identity)
    flat_mask = ~np.isnan(sub)
    pooled = sub[flat_mask]
    n_total = len(pooled)
    if n_total < 4:
        return None

    if level == "ordinal":
        ranked = stats.rankdata(pooled, method="average")
        work = np.full_like(sub, np.nan)
        work[flat_mask] = ranked
        pooled_vals = ranked
    else:
        work = sub
        pooled_vals = pooled

    s_all_sq = float(np.var(pooled_vals, ddof=1))
    if s_all_sq <= 1e-12:
        return None

    num = 0.0
    for j in range(work.shape[1]):
        col = work[:, j]
        col = col[~np.isnan(col)]
        m_u = len(col)
        if m_u >= 2:
            num += m_u * float(np.var(col, ddof=1))
    d_o_half = num / n_total
    alpha = 1.0 - (d_o_half / s_all_sq)
    return round(float(alpha), 4)


def _extract_debrief_df(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        for role, d in (ep.get("debriefs") or {}).items():
            subj = d["subject_model"]
            part = d["partner_model"]
            f1 = (d.get("F1") or {}).get("parsed") or {}
            f1_rt = (d.get("F1_retest") or {}).get("parsed") or {}
            f2 = (d.get("F2") or {}).get("parsed") or {}
            f3a_obj = d.get("F3a") or {}
            f3b_obj = d.get("F3b") or {}
            f3a = f3a_obj.get("parsed") or {}
            f3b = f3b_obj.get("parsed") or {}
            f4 = (d.get("F4") or {}).get("choice")
            f5 = (d.get("F5") or {}).get("parsed") or {}
            qa = (d.get("F1") or {}).get("quote_audit") or {}
            fb_obj = d.get("false_blame_r5") or d.get("false_blame_r7") or {}
            fb_parsed = fb_obj.get("parsed") or {}

            part_family_label = FAMILY_LABELS.get(_model_family(part), "Other / Unsure")
            f5_guess = f5.get("guess")
            f5_correct = (f5_guess == part_family_label) if f5_guess else None

            fb_flag = fb_parsed.get("blames_teammate_for_external_incident")
            if fb_flag is None:
                fb_flag = fb_parsed.get("blames_teammate_for_r7")

            rows.append(
                {
                    "episode_id": ep["episode_id"],
                    "task": ep["task"],
                    "seed": ep["seed"],
                    "arm": ep["arm"],
                    "total_rounds": ep.get("total_rounds", len(ep.get("turns") or [])),
                    "role": role,
                    "subject_model": subj,
                    "partner_model": part,
                    "same_family": _model_family(subj) == _model_family(part),
                    "same_model": subj == part,
                    "score": ep["score"],
                    "f1_again": f1.get("again"),
                    "f1_clarity": f1.get("clarity"),
                    "f1_responsiveness": f1.get("responsiveness"),
                    "f1_recovery": f1.get("recovery"),
                    "f1_overhead": f1.get("overhead"),
                    "f1_summary": f1.get("summary"),
                    "f1_helped_quote": f1.get("helped_quote"),
                    "f1_friction_quote": f1.get("friction_quote"),
                    "f1_helped_verified": (qa.get("helped") or {}).get("verified"),
                    "f1_friction_verified": (qa.get("friction") or {}).get("verified"),
                    "f1_retest_again": f1_rt.get("again"),
                    "f1_retest_clarity": f1_rt.get("clarity"),
                    "f2_again": f2.get("again"),
                    "f2_clarity": f2.get("clarity"),
                    "f2_summary": f2.get("summary"),
                    "f3a_again": f3a.get("again"),
                    "f3a_fidelity_ok": f3a_obj.get("fidelity_ok"),
                    "f3a_coherence_clean": f3a_obj.get("coherence_clean"),
                    "f3a_quoted_removed": f3a_obj.get("has_quoted_removed"),
                    "f3b_again": f3b.get("again"),
                    "f3b_fidelity_ok": f3b_obj.get("fidelity_ok"),
                    "f3b_coherence_clean": f3b_obj.get("coherence_clean"),
                    "f3b_quoted_removed": f3b_obj.get("has_quoted_removed"),
                    "f4_choice": f4,
                    "f5_guess": f5_guess,
                    "f5_confidence": f5.get("confidence"),
                    "f5_correct": f5_correct,
                    "false_blame_flag": fb_flag,
                    "false_blame_prob": fb_obj.get("prob"),
                    "false_blame_evidence": fb_parsed.get("evidence"),
                    "block_id": f"{subj}__{ep['task']}__{role}",
                }
            )
    return pd.DataFrame(rows)


def _extract_turns_df(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        total_r = ep.get("total_rounds", len(ep.get("turns") or []))
        sd = ep.get("score_details") or {}
        vgr = sd.get("verified_green_by_round")
        final_ver_r = sd.get("final_verified_round") if sd.get("verified_green") else None
        for t in ep.get("turns", []):
            rnd = int(t["round"])
            triage_phase: str | None = None
            if ep["task"] == "triage":
                if isinstance(vgr, list) and len(vgr) >= rnd:
                    # Round `rnd` is post-green iff service was already verified green at end of round `rnd - 1`
                    triage_phase = "post_green" if (rnd >= 2 and bool(vgr[rnd - 2])) else "active_incident"
                else:
                    triage_phase = (
                        "post_green"
                        if (final_ver_r is not None and rnd > int(final_ver_r))
                        else "active_incident"
                    )
            for role, other_role in (("A", "B"), ("B", "A")):
                side = t[role]
                other = t[other_role]
                m = side.get("metrics") or {}
                jd = side.get("jev_demeanor") or {}
                row: dict[str, Any] = {
                    "episode_id": ep["episode_id"],
                    "task": ep["task"],
                    "seed": ep["seed"],
                    "arm": ep["arm"],
                    "total_rounds": total_r,
                    "round": rnd,
                    "triage_phase": triage_phase,
                    "role": role,
                    "model": side["model"],
                    "partner_model": other["model"],
                    "same_family": _model_family(side["model"]) == _model_family(other["model"]),
                    "parse_error": side.get("parse_error") is not None,
                    "char_len": m.get("char_len", 0),
                    "word_count": m.get("word_count", 0),
                    "space_ratio": m.get("space_ratio", 0.0),
                    "zlib_ratio": m.get("zlib_ratio", 0.0),
                    "uppercase_ratio": m.get("uppercase_ratio", 0.0),
                    "run_together_per_100": m.get("run_together_per_100", 0.0),
                    "ids_per_100": m.get("ids_per_100", 0.0),
                    "question_marks": m.get("question_marks", 0),
                    "pleasantry_count": m.get("pleasantry_count", 0),
                    "hedge_count": m.get("hedge_count", 0),
                    "imperative_count": m.get("imperative_count", 0),
                    "n_vendor_leaks": len(m.get("vendor_leaks") or []),
                    "vendor_leaks": m.get("vendor_leaks") or [],
                    "jev_wooden_legalistic": jd.get("wooden_legalistic"),
                    "jev_skips_why": jd.get("skips_why"),
                    "jev_answers_directly": jd.get("answers_directly"),
                    "jev_condescending_pedantic": jd.get("condescending_pedantic"),
                    "jev_clarity": jd.get("clarity"),
                    "jev_warmth": jd.get("warmth"),
                    "jev_expresses_confusion": jd.get("expresses_confusion"),
                    "reasoning_tokens": side.get("reasoning_tokens"),  # preserve None!
                    "completion_tokens": side.get("completion_tokens", 0),
                    "prompt_tokens": side.get("prompt_tokens", 0),
                    "cached_tokens": side.get("cached_tokens", 0),
                    "cost_usd": side.get("total_cost_usd", side.get("cost_usd", 0.0)),
                }
                # Store per-judge scores when 3-judge panel data is available
                pjudges = jd.get("panel_judges") or {}
                for jk in DECISION_PANEL_SPECS:
                    jscores = (pjudges.get(jk) or {}).get("scores") or {}
                    for qk in (
                        "wooden_legalistic",
                        "skips_why",
                        "answers_directly",
                        "condescending_pedantic",
                        "clarity",
                        "warmth",
                        "expresses_confusion",
                    ):
                        row[f"j_{jk}_{qk}"] = jscores.get(qk)
                rows.append(row)
    return pd.DataFrame(rows)


def _true_bt_avg_h2h(betas: dict[str, float]) -> dict[str, float]:
    """Exact average head-to-head BT win rate: mean_{j != i} expit(beta_i - beta_j)."""
    keys = list(betas.keys())
    k = len(keys)
    out: dict[str, float] = {}
    for pi in keys:
        probs = [1.0 / (1.0 + math.exp(-(betas[pi] - betas[pj]))) for pj in keys if pj != pi]
        out[pi] = round(float(np.mean(probs)), 4) if k > 1 else 0.5
    return out


def _fit_unified_bt_model(
    choices_df: pd.DataFrame,
    *,
    choice_types: tuple[str, ...] = ("main_7p", "core"),
    partners_override: list[str] | None = None,
    adjust_score: bool = True,
    per_task_score_slopes: bool = True,
    n_boot: int = 2000,
    cluster_mode: str = "subject_task_role",
) -> dict[str, Any]:
    """Unified N-partner Bradley-Terry logistic model with per-task competence slopes,
    block-clustered SEs, exact avg H2H win rates, raw direct win rates, and 2,000-draw cluster bootstrap.
    """
    if choices_df.empty:
        return {"ok": False, "reason": "empty choices dataframe"}

    sub = choices_df[
        choices_df["choice_type"].isin(choice_types) & choices_df["chosen_slot"].isin([1, 2])
    ].copy()
    if len(sub) < 12:
        return {"ok": False, "reason": f"insufficient valid choices (n={len(sub)})"}

    present = sorted(set(sub["partner_1"].unique()) | set(sub["partner_2"].unique()))
    if partners_override is not None:
        partners = [p for p in partners_override if p in present]
        sub = sub[sub["partner_1"].isin(partners) & sub["partner_2"].isin(partners)].copy()
    else:
        ordered_all = [p for p in ALL_PARTNER_MODELS if p in present]
        partners = ordered_all + [p for p in present if p not in ordered_all]

    if len(partners) < 2 or len(sub) < 12:
        return {"ok": False, "reason": f"insufficient partners/rows (k={len(partners)}, n={len(sub)})"}

    # Reference partner = kimi31 if present else last partner
    ref_partner = "kimi31" if "kimi31" in partners else partners[-1]
    non_ref = [p for p in partners if p != ref_partner]
    tasks_present = sorted(sub["task"].unique())

    # Build design matrix: [non_ref partner indicators..., gamma_homophily, (delta_score per task or pooled), eta_pos1]
    use_per_task = adjust_score and per_task_score_slopes and len(tasks_present) >= 1
    score_cols_names = (
        [f"delta_score_{t}" for t in tasks_present]
        if use_per_task
        else (["delta_score"] if adjust_score else [])
    )

    raw_col_names = [f"beta_{p}" for p in non_ref] + ["gamma_homophily"] + score_cols_names + ["eta_pos1"]
    n_non_ref = len(non_ref)
    n_score_cols = len(score_cols_names)
    n_cols_full = n_non_ref + 1 + n_score_cols + 1

    X_rows = []
    y_vals = []
    blocks = []
    p_idx_map = {p: i for i, p in enumerate(non_ref)}
    t_idx_map = {t: i for i, t in enumerate(tasks_present)}

    for _, row in sub.iterrows():
        p1, p2, subj, tname = row["partner_1"], row["partner_2"], row["subject_model"], row["task"]
        x = np.zeros(n_cols_full)
        if p1 in p_idx_map:
            x[p_idx_map[p1]] += 1.0
        if p2 in p_idx_map:
            x[p_idx_map[p2]] -= 1.0
        x[n_non_ref] = float(
            (_model_family(p1) == _model_family(subj)) - (_model_family(p2) == _model_family(subj))
        )
        dscore = float((row["score_1"] or 0.0) - (row["score_2"] or 0.0))
        if use_per_task:
            x[n_non_ref + 1 + t_idx_map[tname]] = dscore
        elif adjust_score:
            x[n_non_ref + 1] = dscore
        x[-1] = 1.0
        X_rows.append(x)
        y_vals.append(1.0 if row["chosen_slot"] == 1 else 0.0)
        if cluster_mode == "scenario_pair":
            s1, s2 = int(row.get("seed_1") or 0), int(row.get("seed_2") or 0)
            blocks.append(f"{tname}__{min(s1, s2)}_{max(s1, s2)}")
        else:
            blocks.append(f"{subj}__{tname}__{row['role']}")

    X_full = np.array(X_rows)
    y = np.array(y_vals)
    block_arr = np.array(blocks)

    # Drop zero-variance score/homophily columns cleanly (keep all partner cols and intercept)
    keep_idx: list[int] = []
    dropped_cols: list[str] = []
    for j, nm in enumerate(raw_col_names):
        if j < n_non_ref or j == n_cols_full - 1 or float(np.std(X_full[:, j])) > 1e-9:
            keep_idx.append(j)
        else:
            dropped_cols.append(nm)

    col_names = [raw_col_names[j] for j in keep_idx]
    X = X_full[:, keep_idx]

    try:
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": block_arr}
        )
    except _GLM_ERRORS as exc:
        return {"ok": False, "reason": f"GLM fit failed: {exc}", "n": len(y)}

    params = dict(zip(col_names, [float(v) for v in fit.params], strict=True))
    cov = np.asarray(fit.cov_params())
    se_map = dict(
        zip(col_names, [float(math.sqrt(max(0.0, cov[i, i]))) for i in range(len(col_names))], strict=True)
    )

    raw_betas = {p: (params[f"beta_{p}"] if p in non_ref else 0.0) for p in partners}
    mean_b = float(np.mean(list(raw_betas.values())))
    centered = {k: v - mean_b for k, v in raw_betas.items()}
    bt_avg_h2h = _true_bt_avg_h2h(raw_betas)
    bt_vs_mean = {k: round(1.0 / (1.0 + math.exp(-v)), 4) for k, v in centered.items()}

    # Raw direct observed win rates (excluding self-family or overall)
    raw_direct_win_rate: dict[str, float] = {}
    raw_direct_cross_vendor: dict[str, float] = {}
    for p in partners:
        rel = sub[(sub["partner_1"] == p) | (sub["partner_2"] == p)]
        raw_direct_win_rate[p] = round(float((rel["chosen_partner"] == p).mean()), 4) if len(rel) else 0.5
        rel_cv = rel[rel["subject_model"].map(_model_family) != _model_family(p)]
        raw_direct_cross_vendor[p] = (
            round(float((rel_cv["chosen_partner"] == p).mean()), 4) if len(rel_cv) else 0.5
        )

    # 4-core replication contrast: beta_astra62 - mean(beta_opus55, beta_neon4, beta_kimi31)
    core_contrast_ok = all(m in partners for m in CORE_MODELS)
    h1_est = h1_se = h1_z = h1_p = h1_prob = None
    c_h1 = np.zeros(len(col_names))
    if core_contrast_ok:
        for m, w in (("astra62", 1.0), ("opus55", -1.0 / 3.0), ("neon4", -1.0 / 3.0), ("kimi31", -1.0 / 3.0)):
            if m != ref_partner:
                c_h1[col_names.index(f"beta_{m}")] = w
        h1_est = float(c_h1 @ fit.params)
        h1_se = math.sqrt(max(1e-12, float(c_h1 @ cov @ c_h1)))
        h1_z = h1_est / h1_se
        h1_p = float(2.0 * (1.0 - stats.norm.cdf(abs(h1_z))))
        h1_prob = 1.0 / (1.0 + math.exp(-h1_est))

    # Full n_boot=2,000 block-cluster bootstrap (no artificial 1,000 cap!)
    uniq_blocks = np.unique(block_arr)
    boot_h1_probs: list[float] = []
    boot_gammas: list[float] = []
    boot_bt_h2h: dict[str, list[float]] = {p: [] for p in partners}
    n_achieved = 0

    if len(uniq_blocks) >= 3 and n_boot > 0:
        rng = np.random.default_rng(20261122)
        idx_by_b = [np.where(block_arr == b)[0] for b in uniq_blocks]
        for _ in range(n_boot):
            sel = rng.integers(0, len(idx_by_b), size=len(idx_by_b))
            rows_idx = np.concatenate([idx_by_b[j] for j in sel])
            X_b_all = X[rows_idx]
            y_b = y[rows_idx]
            # Keep all partner columns + intercept, plus any task-score/homophily cols with variance in this draw
            if not np.all(np.std(X_b_all[:, :n_non_ref], axis=0) > 1e-9):
                continue
            b_keep = [
                j
                for j in range(X_b_all.shape[1])
                if j < n_non_ref
                or j == X_b_all.shape[1] - 1
                or float(np.std(X_b_all[:, j])) > 1e-9
            ]
            X_b = X_b_all[:, b_keep]
            b_names = [col_names[j] for j in b_keep]
            if np.linalg.matrix_rank(X_b) < X_b.shape[1]:
                continue
            try:
                bf = sm.GLM(y_b, X_b, family=sm.families.Binomial()).fit()
                if not np.all(np.isfinite(bf.params)):
                    continue
                n_achieved += 1
                b_param_map = dict(zip(b_names, [float(v) for v in bf.params], strict=True))
                if core_contrast_ok:
                    he = float(
                        b_param_map.get("beta_astra62", 0.0)
                        - (
                            b_param_map.get("beta_opus55", 0.0)
                            + b_param_map.get("beta_neon4", 0.0)
                            + b_param_map.get("beta_kimi31", 0.0)
                        )
                        / 3.0
                    )
                    boot_h1_probs.append(1.0 / (1.0 + math.exp(-he)))
                if "gamma_homophily" in b_param_map:
                    boot_gammas.append(b_param_map["gamma_homophily"])
                b_betas = {
                    p: (b_param_map[f"beta_{p}"] if p in non_ref else 0.0)
                    for p in partners
                }
                b_h2h = _true_bt_avg_h2h(b_betas)
                for p in partners:
                    boot_bt_h2h[p].append(b_h2h[p])
            except _GLM_ERRORS as exc:
                logger.debug("Skipping non-convergent bootstrap draw: %s", exc)
                continue

    bt_avg_h2h_ci95 = {
        p: (
            [
                round(float(np.percentile(boot_bt_h2h[p], 2.5)), 4),
                round(float(np.percentile(boot_bt_h2h[p], 97.5)), 4),
            ]
            if len(boot_bt_h2h[p]) >= 50
            else None
        )
        for p in partners
    }

    gamma_val = params.get("gamma_homophily")
    gamma_se = se_map.get("gamma_homophily")
    gamma_p = (
        float(2.0 * (1.0 - stats.norm.cdf(abs(gamma_val / max(1e-9, gamma_se)))))
        if gamma_val is not None and gamma_se is not None
        else None
    )

    return {
        "ok": True,
        "partners": partners,
        "reference_partner": ref_partner,
        "adjust_score": adjust_score,
        "per_task_score_slopes": use_per_task,
        "dropped_zero_var_cols": dropped_cols,
        "n_choices": len(y),
        "n_blocks": len(uniq_blocks),
        "n_boot_requested": int(n_boot),
        "n_boot_achieved": int(n_achieved),
        "params": {k: round(v, 4) for k, v in params.items()},
        "se_block_clustered": {k: round(v, 4) for k, v in se_map.items()},
        "betas_centered": {k: round(v, 4) for k, v in centered.items()},
        "bt_avg_h2h_win_rate": bt_avg_h2h,
        "bt_avg_h2h_boot_ci95": bt_avg_h2h_ci95,
        "raw_direct_win_rate": raw_direct_win_rate,
        "raw_direct_win_rate_cross_vendor": raw_direct_cross_vendor,
        "bt_win_prob_vs_mean": bt_vs_mean,
        "h1_contrast_logit": round(h1_est, 4) if h1_est is not None else None,
        "h1_contrast_se": round(h1_se, 4) if h1_se is not None else None,
        "h1_z": round(h1_z, 3) if h1_z is not None else None,
        "h1_p_two_sided": round(h1_p, 6) if h1_p is not None else None,
        "h1_p_fmt": _fmt_p(h1_p),
        "h1_prob_astra62_vs_avg_other": round(h1_prob, 4) if h1_prob is not None else None,
        "h1_prob_boot_ci95": (
            [
                round(float(np.percentile(boot_h1_probs, 2.5)), 4),
                round(float(np.percentile(boot_h1_probs, 97.5)), 4),
            ]
            if len(boot_h1_probs) >= 50
            else None
        ),
        "gamma_homophily_logit": round(gamma_val, 4) if gamma_val is not None else None,
        "gamma_se": round(gamma_se, 4) if gamma_se is not None else None,
        "gamma_p_two_sided": round(gamma_p, 6) if gamma_p is not None else None,
        "gamma_p_fmt": _fmt_p(gamma_p),
        "gamma_boot_ci95": (
            [
                round(float(np.percentile(boot_gammas, 2.5)), 4),
                round(float(np.percentile(boot_gammas, 97.5)), 4),
            ]
            if len(boot_gammas) >= 50
            else None
        ),
    }


def _collapse_h2h_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse AB/BA choice rows into one order-balanced case per `(subject_model, task, role, item_id)`."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (subj, task, role, item_id, ctype), g in df.groupby(
        ["subject_model", "task", "role", "item_id", "choice_type"]
    ):
        p62 = float((g["chosen_partner"] == "astra62").mean())
        n_orders = len(g)
        order_agree = bool(n_orders == 2 and p62 in (0.0, 1.0))
        rows.append(
            {
                "subject_model": str(subj),
                "task": str(task),
                "role": str(role),
                "item_id": str(item_id),
                "choice_type": str(ctype),
                "p_astra62": p62,
                "n_orders": n_orders,
                "order_agree": order_agree,
                "block_id": f"{subj}__{task}__{role}",
                "tr_block": f"{task}__{role}",
            }
        )
    return pd.DataFrame(rows)


def _summarize_h2h_case_slice(case_df: pd.DataFrame, *, cluster_col: str = "block_id") -> dict[str, Any]:
    """Primary cluster-level summary for collapsed AB/BA H2H cases (no naive independent-call p-values)."""
    if case_df.empty:
        return {"ok": False, "n_cases": 0}
    vals = case_df["p_astra62"].astype(float).values
    blocks = case_df[cluster_col].astype(str).values
    boot = _cluster_bootstrap_mean(vals, blocks, n_boot=2000, null_val=0.5)
    # Exact cluster-level sign-flip permutation p-value around 0.5 (integer DP over cluster sums)
    cl_grouped = case_df.groupby(cluster_col)["p_astra62"]
    cl_sums = (cl_grouped.sum() - 0.5 * cl_grouped.count()).values
    n_cl = len(cl_sums)
    if n_cl >= 2:
        scaled = [round(2.0 * float(s)) for s in cl_sums]
        if all(abs(2.0 * float(s) - w) < 1e-9 for s, w in zip(cl_sums, scaled, strict=True)):
            counts: dict[int, int] = {0: 1}
            for w in scaled:
                if w == 0:
                    counts = {tot: c * 2 for tot, c in counts.items()}
                else:
                    nxt: dict[int, int] = defaultdict(int)
                    for tot, c in counts.items():
                        nxt[tot + w] += c
                        nxt[tot - w] += c
                    counts = nxt
            obs_int = abs(sum(scaled))
            total_ways = 1 << n_cl
            perm_p = sum(c for tot, c in counts.items() if abs(tot) >= obs_int) / total_ways
        elif n_cl <= 14:
            obs = abs(float(np.sum(cl_sums)))
            signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n_cl)))
            perm_vals = np.abs((signs * cl_sums[None, :]).sum(axis=1))
            perm_p = float(np.mean(perm_vals >= obs - 1e-12))
        else:
            rng = np.random.default_rng(20261122)
            obs = abs(float(np.sum(cl_sums)))
            signs = rng.choice([-1.0, 1.0], size=(5000, n_cl))
            perm_vals = np.abs((signs * cl_sums[None, :]).sum(axis=1))
            perm_p = float((np.sum(perm_vals >= obs - 1e-12) + 1) / (5000 + 1))
    else:
        perm_p = None

    by_task: dict[str, Any] = {}
    for tname, tg in case_df.groupby("task"):
        by_task[str(tname)] = {
            "n_cases": len(tg),
            "astra62_win_rate": round(float(tg["p_astra62"].mean()), 4),
            "order_agree_rate": round(float(tg["order_agree"].mean()), 4),
        }

    return {
        "ok": True,
        "n_cases": len(case_df),
        "n_presentations": int(case_df["n_orders"].sum()),
        "n_clusters": n_cl,
        "astra62_win_rate": round(float(np.mean(vals)), 4),
        "astra60_win_rate": round(1.0 - float(np.mean(vals)), 4),
        "order_agree_rate": round(float(case_df["order_agree"].mean()), 4),
        "cluster_bootstrap": boot,
        "primary_cluster_perm_p": round(perm_p, 6) if perm_p is not None else None,
        "primary_cluster_perm_p_fmt": _fmt_p(perm_p),
        "by_task": by_task,
    }


def _compute_direct_predecessor_analysis(choices_df: pd.DataFrame) -> dict[str, Any]:
    """Case- and cluster-level predecessor H2H analysis (astra62 vs astra60) separating main shared seeds from Opus boost.

    Primary tests operate strictly at the AB/BA-collapsed case & block-cluster level:
      - `main_same_seed` (`extra_h2h`, seeds 301-304): exact shared-seed comparison across opus55, neon4, kimi31
        and the primary shared-seed `Claude − {Neon, Kimi}` DiD.
      - `opus_boost_same_seed` (`opus_boost_h2h`, seeds 305-308): Opus-only extra power arm reported separately.
      - `opus_combined_same_seed` (`extra_h2h + opus_boost_h2h`, 64 distinct zero-reuse same-seed cases).
      - `main_7p_distinct_seed`: direct `astra62 vs astra60` rows inside the 7-partner league (separate sensitivity).
    """
    if choices_df.empty:
        return {"ok": False}

    valid = choices_df[choices_df["chosen_slot"].isin([1, 2])].copy()
    is_pair = (
        ((valid["partner_1"] == "astra62") & (valid["partner_2"] == PREDECESSOR_MODEL))
        | ((valid["partner_1"] == PREDECESSOR_MODEL) & (valid["partner_2"] == "astra62"))
    )
    direct = valid[is_pair & valid["chosen_partner"].isin(["astra62", PREDECESSOR_MODEL])].copy()
    if direct.empty:
        return {"ok": False}

    cases_all = _collapse_h2h_cases(direct)
    cross_cases = cases_all[cases_all["subject_model"] != "astra62"].copy()

    # Slice 1: Shared-seed main same-scenario H2H (`extra_h2h`; fallback to non-boost if legacy)
    main_same = cross_cases[cross_cases["choice_type"] == "extra_h2h"].copy()
    if main_same.empty:
        main_same = cross_cases[cross_cases["choice_type"] != "opus_boost_h2h"].copy()
    boost_same = cross_cases[cross_cases["choice_type"] == "opus_boost_h2h"].copy()
    main_7p_direct = cross_cases[cross_cases["choice_type"].isin(["main_7p", "core"])].copy()

    out: dict[str, Any] = {
        "ok": True,
        "n_total_direct_presentations": len(direct),
        "n_total_direct_cases": len(cases_all),
        "main_shared_seed_cross_vendor": _summarize_h2h_case_slice(main_same, cluster_col="block_id"),
        "main_7p_distinct_seed_cross_vendor": _summarize_h2h_case_slice(main_7p_direct, cluster_col="block_id"),
        "opus_boost_same_seed": _summarize_h2h_case_slice(boost_same, cluster_col="tr_block"),
    }

    # Per-subject shared-seed main (301-304) + Opus combined same-seed (301-308)
    by_subj_main: dict[str, Any] = {}
    for subj, sg in main_same.groupby("subject_model"):
        by_subj_main[str(subj)] = _summarize_h2h_case_slice(sg, cluster_col="tr_block")
    out["by_subject_main_shared_seed"] = by_subj_main

    opus_same_all = cross_cases[
        (cross_cases["subject_model"] == "opus55")
        & cross_cases["choice_type"].isin(["extra_h2h", "opus_boost_h2h"])
    ].copy()
    out["opus_combined_same_seed"] = _summarize_h2h_case_slice(opus_same_all, cluster_col="tr_block")

    # Backward-compatible top-level summary fields sourced strictly from shared-seed main cross-vendor!
    if out["main_shared_seed_cross_vendor"].get("ok"):
        ms = out["main_shared_seed_cross_vendor"]
        out["n_cross_vendor_choices"] = ms["n_presentations"]
        out["astra62_win_rate"] = ms["astra62_win_rate"]
        out["astra60_win_rate"] = ms["astra60_win_rate"]
        out["block_bootstrap"] = ms["cluster_bootstrap"]
        out["primary_cluster_perm_p"] = ms["primary_cluster_perm_p"]
        out["primary_cluster_perm_p_fmt"] = ms["primary_cluster_perm_p_fmt"]
        out["by_task_cross_vendor"] = ms["by_task"]

    # Primary shared-seed DiD: Claude (opus55) minus other cross-vendor ({neon4, kimi31}) on seeds 301-304 ONLY
    op_main = main_same[main_same["subject_model"] == "opus55"]
    ot_main = main_same[main_same["subject_model"] != "opus55"]
    if len(op_main) >= 4 and len(ot_main) >= 4:
        p_opus = float(op_main["p_astra62"].mean())
        p_oth = float(ot_main["p_astra62"].mean())
        did = p_opus - p_oth

        rng = np.random.default_rng(20261122)
        op_blocks = [g["p_astra62"].values for _, g in op_main.groupby("tr_block")]
        ot_blocks = [g["p_astra62"].values for _, g in ot_main.groupby("block_id")]
        boot_dids = []
        if len(op_blocks) >= 2 and len(ot_blocks) >= 2:
            for _ in range(2000):
                s_op = np.concatenate(
                    [op_blocks[i] for i in rng.integers(0, len(op_blocks), size=len(op_blocks))]
                )
                s_ot = np.concatenate(
                    [ot_blocks[i] for i in rng.integers(0, len(ot_blocks), size=len(ot_blocks))]
                )
                boot_dids.append(float(np.mean(s_op) - np.mean(s_ot)))

        # Primary cluster-level permutation p-value (paired by shared task×role block!)
        op_by_tr = op_main.groupby("tr_block")["p_astra62"].mean()
        ot_by_tr = ot_main.groupby("tr_block")["p_astra62"].mean()
        shared_tr = sorted(set(op_by_tr.index) & set(ot_by_tr.index))
        paired_diffs = np.array([float(op_by_tr[k] - ot_by_tr[k]) for k in shared_tr])
        if len(paired_diffs) >= 2:
            obs_d = abs(float(np.mean(paired_diffs)))
            signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(paired_diffs))))
            perm_d = np.abs((signs * paired_diffs[None, :]).mean(axis=1))
            did_perm_p = float(np.mean(perm_d >= obs_d - 1e-12))
        else:
            did_perm_p = None

        out["claude_minus_other_cross_vendor_did"] = {
            "scope": "main_shared_seeds_only_301_304",
            "opus55_astra62_win_rate": round(p_opus, 4),
            "opus55_n_cases": len(op_main),
            "other_cross_vendor_astra62_win_rate": round(p_oth, 4),
            "other_cross_vendor_n_cases": len(ot_main),
            "did_opus_minus_others": round(did, 4),
            "boot_ci95": (
                [round(float(np.percentile(boot_dids, 2.5)), 4), round(float(np.percentile(boot_dids, 97.5)), 4)]
                if len(boot_dids) >= 50
                else None
            ),
            "n_boot_achieved": len(boot_dids),
            "primary_paired_cluster_perm_p": round(did_perm_p, 6) if did_perm_p is not None else None,
            "primary_paired_cluster_perm_p_fmt": _fmt_p(did_perm_p),
            "by_task_did": {
                str(t): round(
                    float(op_main[op_main["task"] == t]["p_astra62"].mean())
                    - float(ot_main[ot_main["task"] == t]["p_astra62"].mean()),
                    4,
                )
                for t in sorted(set(op_main["task"]) & set(ot_main["task"]))
            },
        }

    # Retain per-subject combined view for reference (clearly tagged as including boost for opus55)
    by_subj_all: dict[str, Any] = {}
    for subj, sg in cases_all.groupby("subject_model"):
        by_subj_all[str(subj)] = _summarize_h2h_case_slice(sg, cluster_col="tr_block")
    out["by_subject_all_direct"] = by_subj_all
    return out


def _compute_loso_sensitivity(choices_df: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-scenario-out (drop each (task, seed) & each task) sensitivity on unified BT."""
    if choices_df.empty:
        return {}
    main_c = choices_df[
        choices_df["choice_type"].isin(["main_7p", "core"]) & choices_df["chosen_slot"].isin([1, 2])
    ].copy()
    if len(main_c) < 24:
        return {}

    # Leave-one-task-out
    loto: dict[str, Any] = {}
    a62_rates_task: list[float] = []
    for tname in sorted(main_c["task"].unique()):
        sub = main_c[main_c["task"] != tname]
        if len(sub) < 12:
            continue
        fit = _fit_unified_bt_model(sub, adjust_score=True, n_boot=0)
        if fit.get("ok"):
            r62 = (fit.get("bt_avg_h2h_win_rate") or {}).get("astra62")
            if r62 is not None:
                a62_rates_task.append(r62)
            loto[f"drop_{tname}"] = {
                "astra62_bt_avg_h2h": r62,
                "h1_prob_astra62_vs_core3": fit.get("h1_prob_astra62_vs_avg_other"),
                "h1_p_fmt": fit.get("h1_p_fmt"),
            }

    # Leave-one-(task, seed)-out
    loso: dict[str, Any] = {}
    a62_rates_scen: list[float] = []
    if "seed_1" in main_c.columns and "seed_2" in main_c.columns:
        scenarios = sorted(
            {(str(r["task"]), int(r["seed_1"])) for _, r in main_c.iterrows() if pd.notna(r["seed_1"])}
            | {(str(r["task"]), int(r["seed_2"])) for _, r in main_c.iterrows() if pd.notna(r["seed_2"])}
        )
        for tname, sd in scenarios:
            sub = main_c[~((main_c["task"] == tname) & ((main_c["seed_1"] == sd) | (main_c["seed_2"] == sd)))]
            if len(sub) < 12:
                continue
            fit = _fit_unified_bt_model(sub, adjust_score=True, n_boot=0)
            if fit.get("ok"):
                r62 = (fit.get("bt_avg_h2h_win_rate") or {}).get("astra62")
                if r62 is not None:
                    a62_rates_scen.append(r62)
                loso[f"drop_{tname}_s{sd}"] = {
                    "astra62_bt_avg_h2h": r62,
                    "h1_prob_astra62_vs_core3": fit.get("h1_prob_astra62_vs_avg_other"),
                    "h1_p_fmt": fit.get("h1_p_fmt"),
                }

    return {
        "leave_one_task_out": loto,
        "loto_astra62_bt_range": [min(a62_rates_task), max(a62_rates_task)] if a62_rates_task else None,
        "leave_one_scenario_out": loso,
        "loso_astra62_bt_range": [min(a62_rates_scen), max(a62_rates_scen)] if a62_rates_scen else None,
    }


def _compute_decision_panel_reliability(turns_df: pd.DataFrame) -> dict[str, Any]:
    """3-Judge reliability report: per-judge means, pairwise MAD, Krippendorff's alpha (ordinal & interval), pairwise Spearman r."""
    if turns_df.empty:
        return {}
    jkeys = list(DECISION_PANEL_SPECS.keys())
    metrics = [
        "wooden_legalistic",
        "skips_why",
        "answers_directly",
        "condescending_pedantic",
        "clarity",
        "warmth",
        "expresses_confusion",
    ]
    out: dict[str, Any] = {}
    for m in metrics:
        cols = [f"j_{jk}_{m}" for jk in jkeys]
        if not all(c in turns_df.columns for c in cols):
            continue
        sub = turns_df[cols].apply(pd.to_numeric, errors="coerce")
        if sub.notna().sum().sum() < 6:
            continue
        mat = sub.values.T  # shape (3, n_turns)
        per_judge_means = {
            jk: round(float(sub[f"j_{jk}_{m}"].dropna().mean()), 4)
            if sub[f"j_{jk}_{m}"].notna().any()
            else None
            for jk in jkeys
        }
        pairwise_mad: dict[str, float | None] = {}
        pairwise_spearman: dict[str, float | None] = {}
        for i in range(len(jkeys)):
            for j in range(i + 1, len(jkeys)):
                j1, j2 = jkeys[i], jkeys[j]
                pair_df = sub[[f"j_{j1}_{m}", f"j_{j2}_{m}"]].dropna()
                key = f"{j1}_vs_{j2}"
                if len(pair_df) >= 3:
                    v1 = pair_df[f"j_{j1}_{m}"].values
                    v2 = pair_df[f"j_{j2}_{m}"].values
                    pairwise_mad[key] = round(float(np.mean(np.abs(v1 - v2))), 4)
                    if float(np.std(v1)) > 1e-12 and float(np.std(v2)) > 1e-12:
                        rho = float(stats.spearmanr(v1, v2).statistic)
                        pairwise_spearman[key] = round(rho, 4) if not math.isnan(rho) else None
                    else:
                        pairwise_spearman[key] = None
                else:
                    pairwise_mad[key] = None
                    pairwise_spearman[key] = None

        out[m] = {
            "n_units_ge_2_judges": int(np.sum(np.sum(~np.isnan(mat), axis=0) >= 2)),
            "per_judge_mean": per_judge_means,
            "pairwise_mad": pairwise_mad,
            "pairwise_spearman_r": pairwise_spearman,
            "krippendorff_alpha_interval": krippendorff_alpha(mat, level="interval"),
            "krippendorff_alpha_ordinal": krippendorff_alpha(mat, level="ordinal"),
        }
    return out


def _compute_aa_vs_bt_correlation(bt_fit: dict[str, Any]) -> dict[str, Any]:
    """7-point AA Intelligence Index v5 vs Coworker BT Win-Rate correlation + leave-one-model-out sensitivity."""
    bt_rates = (bt_fit or {}).get("bt_avg_h2h_win_rate") or {}
    pts = [
        (m, AA_INDEX_V5[m], bt_rates[m])
        for m in ALL_PARTNER_MODELS
        if m in AA_INDEX_V5 and m in bt_rates
    ]
    if len(pts) < 4:
        return {"ok": False, "n_models": len(pts)}
    xs = np.array([p[1] for p in pts], dtype=float)
    ys = np.array([p[2] for p in pts], dtype=float)
    sp = stats.spearmanr(xs, ys)
    pr = stats.pearsonr(xs, ys)

    lomo: dict[str, Any] = {}
    for drop_idx, (m_drop, _, _) in enumerate(pts):
        sub_x = np.delete(xs, drop_idx)
        sub_y = np.delete(ys, drop_idx)
        r_sp = float(stats.spearmanr(sub_x, sub_y).statistic)
        lomo[f"drop_{m_drop}"] = round(r_sp, 4) if not math.isnan(r_sp) else None

    return {
        "ok": True,
        "n_models": len(pts),
        "points": [{"model": m, "aa_index_v5": a, "bt_avg_h2h_win_rate": w} for m, a, w in pts],
        "spearman_rho": round(float(sp.statistic), 4) if not math.isnan(float(sp.statistic)) else None,
        "spearman_p": round(float(sp.pvalue), 4) if not math.isnan(float(sp.pvalue)) else None,
        "pearson_r": round(float(pr.statistic), 4) if not math.isnan(float(pr.statistic)) else None,
        "leave_one_model_out_spearman": lomo,
    }


def _compute_icc_2_1(a: np.ndarray, b: np.ndarray) -> float | None:
    """Two-way random effects, absolute agreement, single-measures ICC(2,1) for k=2 repeated ratings."""
    n = len(a)
    if n < 3:
        return None
    Y = np.column_stack([a, b]).astype(float)
    k = 2
    grand_mean = float(np.mean(Y))
    row_means = np.mean(Y, axis=1)
    col_means = np.mean(Y, axis=0)
    ssr = k * float(np.sum((row_means - grand_mean) ** 2))
    ssc = n * float(np.sum((col_means - grand_mean) ** 2))
    sst = float(np.sum((Y - grand_mean) ** 2))
    sse = max(0.0, sst - ssr - ssc)
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if abs(denom) < 1e-12:
        return None
    return float((msr - mse) / denom)


def _compute_f1_retest_reliability(debrief_df: pd.DataFrame) -> dict[str, Any]:
    if debrief_df.empty or "f1_retest_again" not in debrief_df.columns:
        return {"ok": False, "n": 0}
    sub = debrief_df.dropna(subset=["f1_again", "f1_retest_again"])
    if len(sub) < 4:
        return {"ok": False, "n": len(sub)}
    a = sub["f1_again"].astype(float).values
    b = sub["f1_retest_again"].astype(float).values
    diff = np.abs(a - b)
    pr = float(stats.pearsonr(a, b).statistic) if float(np.std(a)) > 0 and float(np.std(b)) > 0 else None
    sp = float(stats.spearmanr(a, b).statistic) if float(np.std(a)) > 0 and float(np.std(b)) > 0 else None
    icc21 = _compute_icc_2_1(a, b)
    return {
        "ok": True,
        "n": len(sub),
        "exact_agreement_rate": round(float(np.mean(diff == 0)), 4),
        "within_1_rate": round(float(np.mean(diff <= 1)), 4),
        "mae": round(float(np.mean(diff)), 4),
        "mean_signed_diff_retest_minus_orig": round(float(np.mean(b - a)), 4),
        "icc_2_1": round(icc21, 4) if icc21 is not None and not math.isnan(icc21) else None,
        "pearson_r": round(pr, 4) if pr is not None and not math.isnan(pr) else None,
        "spearman_rho": round(sp, 4) if sp is not None and not math.isnan(sp) else None,
    }


def _compute_manipulation_check(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether legacy F3a/F3b rewrites altered style metrics (only populated if F3 was run)."""
    out: dict[str, Any] = {}
    for mode in ("F3a", "F3b"):
        before_rows = []
        after_rows = []
        hedge_orig = 0
        hedge_new = 0
        total_msgs = 0
        fidelity_ok_msgs = 0
        for ep in episodes:
            for role, rstore in (ep.get("rewrites") or {}).items():
                mdata = rstore.get(mode)
                if not mdata:
                    continue
                partner_model = ep["model_b"] if role == "A" else ep["model_a"]
                for item in mdata.get("messages", []):
                    total_msgs += 1
                    fidelity_ok_msgs += int(bool(item.get("fidelity_ok")))
                    hedge_orig += int(item.get("orig_hedges") or 0)
                    hedge_new += int(item.get("new_hedges") or 0)
                for mb in mdata.get("metrics_before", []):
                    before_rows.append({"model": partner_model, **mb})
                for ma in mdata.get("metrics_after", []):
                    after_rows.append({"model": partner_model, **ma})
        if not before_rows:
            continue
        dfb = pd.DataFrame(before_rows)
        dfa = pd.DataFrame(after_rows)
        checks = {}
        for col in ("space_ratio", "zlib_ratio", "word_count", "pleasantry_count", "hedge_count"):
            groups_b = [g[col].values for _, g in dfb.groupby("model") if len(g) > 1]
            groups_a = [g[col].values for _, g in dfa.groupby("model") if len(g) > 1]
            fb = float(stats.f_oneway(*groups_b).statistic) if len(groups_b) >= 2 else None
            fa = float(stats.f_oneway(*groups_a).statistic) if len(groups_a) >= 2 else None
            checks[col] = {
                "before_by_model": {k: round(float(v), 4) for k, v in dfb.groupby("model")[col].mean().items()},
                "after_by_model": {k: round(float(v), 4) for k, v in dfa.groupby("model")[col].mean().items()},
                "anova_F_before": round(fb, 3) if fb is not None and not math.isnan(fb) else None,
                "anova_F_after": round(fa, 3) if fa is not None and not math.isnan(fa) else None,
            }
        out[mode] = {
            "total_messages": total_msgs,
            "message_fidelity_rate": round(fidelity_ok_msgs / max(1, total_msgs), 4),
            "hedges_before_total": hedge_orig,
            "hedges_after_total": hedge_new,
            "metrics": checks,
        }
    return out


def _compute_style_distances(turns_df: pd.DataFrame) -> dict[str, Any]:
    feats = [
        "char_len",
        "space_ratio",
        "zlib_ratio",
        "uppercase_ratio",
        "run_together_per_100",
        "pleasantry_count",
        "hedge_count",
        "imperative_count",
    ]
    if turns_df.empty:
        return {}
    means = turns_df.groupby("model")[feats].mean()
    std = turns_df[feats].std().replace(0, 1.0)
    z = (means - turns_df[feats].mean()) / std
    dist_map: dict[str, dict[str, float]] = {}
    for m1 in z.index:
        dist_map[m1] = {}
        for m2 in z.index:
            d = float(np.linalg.norm(z.loc[m1].values - z.loc[m2].values))
            dist_map[m1][m2] = round(d, 3)
    return {"standardized_euclidean_distances": dist_map}


def _compute_visibility_masked_demeanor_regression(
    turns_df: pd.DataFrame, choices_df: pd.DataFrame
) -> dict[str, Any]:
    """Demeanor -> forced-choice regression STRICTLY masking out partner B's unseen final turn when chooser is A."""
    if turns_df.empty or choices_df.empty or not turns_df["jev_clarity"].notna().any():
        return {}

    jdf = turns_df.dropna(subset=["jev_clarity"]).copy()
    feats = [
        "jev_clarity",
        "jev_warmth",
        "jev_skips_why",
        "jev_wooden_legalistic",
        "jev_condescending_pedantic",
        "word_count",
    ]
    # Partner demeanor visible to subject:
    # - If chooser is Role A, partner is B, and Role A ONLY saw B's rounds 1..(total_rounds - 1)!
    # - If chooser is Role B, partner is A, and Role B saw ALL rounds 1..total_rounds of A!
    vis_b_for_a = (
        jdf[(jdf["role"] == "B") & (jdf["round"] < jdf["total_rounds"])]
        .groupby("episode_id")[feats]
        .mean()
        .to_dict(orient="index")
    )
    vis_a_for_b = (
        jdf[jdf["role"] == "A"]
        .groupby("episode_id")[feats]
        .mean()
        .to_dict(orient="index")
    )
    s_conf = (
        jdf[jdf["round"] >= 2]
        .groupby(["episode_id", "role"])["jev_expresses_confusion"]
        .mean()
        .to_dict()
    )

    valid_c = choices_df[
        choices_df["choice_type"].isin(["main_7p", "core", "extra_h2h", "predecessor_h2h"])
        & choices_df["chosen_slot"].isin([1, 2])
    ]
    X_rows, y_vals, blocks = [], [], []
    for _, row in valid_c.iterrows():
        srole = str(row["role"])
        lookup = vis_b_for_a if srole == "A" else vis_a_for_b
        e1 = str(row["episode_1_id"])
        e2 = str(row["episode_2_id"])
        if e1 not in lookup or e2 not in lookup:
            continue
        d_feats = [float(lookup[e1][f] - lookup[e2][f]) for f in feats]
        c1 = float(s_conf.get((e1, srole), 0.0))
        c2 = float(s_conf.get((e2, srole), 0.0))
        d_score = float((row["score_1"] or 0.0) - (row["score_2"] or 0.0))
        X_rows.append(d_feats + [c1 - c2, d_score, 1.0])
        y_vals.append(1.0 if row["chosen_slot"] == 1 else 0.0)
        blocks.append(f"{row['subject_model']}__{row['task']}__{srole}")

    if len(X_rows) < 30:
        return {"ok": False, "n_choices": len(X_rows)}

    X = np.array(X_rows)
    y = np.array(y_vals)
    b_arr = np.array(blocks)
    names = [
        "delta_panel_clarity",
        "delta_panel_warmth",
        "delta_panel_skips_why",
        "delta_panel_wooden_legalistic",
        "delta_panel_condescending_pedantic",
        "delta_word_count",
        "delta_subject_confusion",
        "delta_score",
        "eta_pos1",
    ]
    X_std = X.copy()
    keep_idx: list[int] = []
    dropped_cols: list[str] = []
    for j in range(len(names)):
        if j == len(names) - 1:
            keep_idx.append(j)
            continue
        sd = float(np.std(X[:, j]))
        if sd > 1e-9:
            X_std[:, j] = X[:, j] / sd
            # Only keep if linearly independent of previously kept predictors
            cand = X_std[:, keep_idx + [j]]
            if np.linalg.matrix_rank(cand) == cand.shape[1]:
                keep_idx.append(j)
            else:
                dropped_cols.append(names[j])
        else:
            dropped_cols.append(names[j])

    kept_names = [names[j] for j in keep_idx]
    X_fit = X_std[:, keep_idx]
    try:
        fit = sm.GLM(y, X_fit, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": b_arr}
        )
        cov = np.asarray(fit.cov_params())
        reg_out: dict[str, Any] = {
            "ok": True,
            "visibility_masked": True,
            "n_choices": len(y),
            "dropped_collinear_cols": dropped_cols,
            "standardized_coefs": {},
        }
        for j, nm in enumerate(kept_names):
            est = float(fit.params[j])
            se = math.sqrt(max(1e-12, float(cov[j, j])))
            z = est / se
            p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            reg_out["standardized_coefs"][nm] = {
                "coef": round(est, 4),
                "se": round(se, 4),
                "z": round(z, 3),
                "p": round(p, 6),
                "p_fmt": _fmt_p(p),
            }
        return reg_out
    except _GLM_ERRORS as exc:
        return {"ok": False, "reason": str(exc)}


def _compute_supplementary_breakdowns(
    *,
    debrief_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    choices_df: pd.DataFrame,
) -> dict[str, Any]:
    sup: dict[str, Any] = {}

    if not choices_df.empty:
        valid = choices_df[choices_df["chosen_slot"].isin([1, 2])].copy()
        consistency: dict[str, Any] = {}
        for ctype, cg in valid.groupby("choice_type"):
            same_ep = []
            for _, ig in cg.groupby("item_id"):
                if len(ig) == 2:
                    eps = [
                        r["episode_1_id"] if r["chosen_slot"] == 1 else r["episode_2_id"]
                        for _, r in ig.iterrows()
                    ]
                    same_ep.append(float(eps[0] == eps[1]))
            if same_ep:
                consistency[str(ctype)] = {
                    "n_pairs": len(same_ep),
                    "order_flip_consistency_rate": round(float(np.mean(same_ep)), 4),
                }
        sup["choice_order_flip_consistency"] = consistency

        # Task-by-task BT models + Original 3 tasks vs Code Handoff separately
        by_task_bt: dict[str, Any] = {}
        for tname in sorted(valid["task"].unique()):
            tdf = valid[valid["task"] == tname]
            by_task_bt[str(tname)] = {
                "adjusted": _fit_unified_bt_model(tdf, adjust_score=True, n_boot=500),
                "unadjusted": _fit_unified_bt_model(tdf, adjust_score=False, n_boot=500),
            }
        sup["BT_by_task"] = by_task_bt

        orig3_df = valid[valid["task"].isin(ORIGINAL_3_TASKS)]
        ch_df = valid[valid["task"] == "code_handoff"]
        sup["task_group_contrasts"] = {
            "original_3_tasks": {
                "adjusted": _fit_unified_bt_model(orig3_df, adjust_score=True, n_boot=2000),
                "unadjusted": _fit_unified_bt_model(orig3_df, adjust_score=False, n_boot=2000),
            }
            if len(orig3_df) >= 12
            else {"ok": False},
            "code_handoff_only": {
                "adjusted": _fit_unified_bt_model(ch_df, adjust_score=True, n_boot=2000),
                "unadjusted": _fit_unified_bt_model(ch_df, adjust_score=False, n_boot=2000),
            }
            if len(ch_df) >= 12
            else {"ok": False},
        }

        # Subject x Partner direct win rate matrix (all present partners)
        main_c = valid[valid["choice_type"].isin(["main_7p", "core"])]
        if not main_c.empty:
            partners = [
                p
                for p in ALL_PARTNER_MODELS
                if p in set(main_c["partner_1"]) | set(main_c["partner_2"])
            ]
            subj_part_wins: dict[str, dict[str, float | None]] = {}
            for subj in CORE_MODELS:
                sg = main_c[main_c["subject_model"] == subj]
                subj_part_wins[subj] = {}
                for part in partners:
                    rel = sg[(sg["partner_1"] == part) | (sg["partner_2"] == part)]
                    subj_part_wins[subj][part] = (
                        round(float((rel["chosen_partner"] == part).mean()), 4) if len(rel) else None
                    )
            sup["choice_win_rate_by_subject_x_partner"] = subj_part_wins

    if not debrief_df.empty:
        cross_d = debrief_df[~debrief_df["same_family"]].dropna(subset=["f1_again"])
        f1_by_task: dict[str, Any] = {}
        for tname, tg in cross_d.groupby("task"):
            f1_by_task[str(tname)] = {
                "mean_f1_again_by_partner": tg.groupby("partner_model")["f1_again"].mean().round(4).to_dict(),
                "mean_score_by_partner": tg.groupby("partner_model")["score"].mean().round(4).to_dict(),
            }
        sup["F1_cross_vendor_by_task"] = f1_by_task

        triage_d = cross_d[cross_d["task"] == "triage"].copy()
        if not triage_d.empty:
            triage_d["partner_role"] = np.where(triage_d["role"] == "A", "B", "A")
            tr_roles: dict[str, Any] = {}
            for prole, rg in triage_d.groupby("partner_role"):
                tr_roles[f"partner_is_role_{prole}"] = {
                    "mean_f1_again": rg.groupby("partner_model")["f1_again"].mean().round(4).to_dict(),
                    "mean_score": rg.groupby("partner_model")["score"].mean().round(4).to_dict(),
                    "n_by_partner": rg.groupby("partner_model")["f1_again"].count().to_dict(),
                }
            sup["triage_by_partner_role_cross_vendor"] = tr_roles

        f5_sub = debrief_df.dropna(subset=["f5_guess"]).copy()
        if not f5_sub.empty:
            cm = pd.crosstab(f5_sub["partner_model"], f5_sub["f5_guess"])
            non_abs = f5_sub[~f5_sub["f5_guess"].isin(["Cannot tell", "Other / Unsure"])]
            cm_non_abs = (
                pd.crosstab(non_abs["partner_model"], non_abs["f5_guess"])
                if not non_abs.empty
                else pd.DataFrame()
            )
            sup["F5_confusion_matrix_all"] = cm.to_dict(orient="index")
            sup["F5_confusion_matrix_non_abstaining"] = (
                cm_non_abs.to_dict(orient="index") if not cm_non_abs.empty else {}
            )
            sup["F5_conditional_accuracy_by_subject"] = (
                non_abs.groupby("subject_model")["f5_correct"].mean().round(4).to_dict()
                if not non_abs.empty
                else {}
            )

        low = debrief_df[debrief_df["f1_again"] <= 3].sort_values(["f1_again", "score"])
        sup["lowest_rated_debriefs_le_3"] = [
            {
                "episode_id": str(r["episode_id"]),
                "task": str(r["task"]),
                "role": str(r["role"]),
                "subject": str(r["subject_model"]),
                "partner": str(r["partner_model"]),
                "score": round(float(r["score"]), 4),
                "f1_again": int(r["f1_again"]),
                "f2_again": int(r["f2_again"]) if pd.notna(r["f2_again"]) else None,
                "f4_choice": r["f4_choice"],
                "f1_summary": r["f1_summary"],
                "f1_friction_quote": r["f1_friction_quote"],
            }
            for _, r in low.iterrows()
        ]

    # Active-incident vs post-green Triage splits
    if not turns_df.empty and "triage_phase" in turns_df.columns:
        tr_t = turns_df[turns_df["task"] == "triage"]
        if not tr_t.empty:
            cols = [
                "jev_wooden_legalistic",
                "jev_skips_why",
                "jev_answers_directly",
                "jev_condescending_pedantic",
                "jev_clarity",
                "jev_warmth",
                "word_count",
            ]
            triage_splits: dict[str, Any] = {}
            for phase_lbl, pg in tr_t.groupby("triage_phase"):
                triage_splits[str(phase_lbl)] = {
                    "n_turns": len(pg),
                    "by_model": pg.groupby("model")[cols].mean().round(4).to_dict(orient="index"),
                }
            sup["triage_active_vs_post_green"] = triage_splits

    sup["demeanor_choice_regression_visibility_masked"] = _compute_visibility_masked_demeanor_regression(
        turns_df, choices_df
    )
    return sup


def _generate_figures(
    *,
    debrief_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    choices_df: pd.DataFrame,
    report: dict[str, Any],
    fig_dir: Path,
) -> list[str]:
    """Generate publication figures and return RELATIVE paths only (`figures/<name>.png`)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.02)
    saved: list[str] = []

    bt_adj = report.get("unified_bt_adjusted") or {}
    bt_rates = bt_adj.get("bt_avg_h2h_win_rate") or {}
    bt_cis = bt_adj.get("bt_avg_h2h_boot_ci95") or {}

    # --- Fig 1: 7-Partner League Table (Score-Adjusted BT avg H2H win rate + Cross-Vendor F1) ---
    if not debrief_df.empty and debrief_df["f1_again"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
        cross = debrief_df[~debrief_df["same_family"]].dropna(subset=["f1_again"])
        order = [m for m in ALL_PARTNER_MODELS if m in cross["partner_model"].unique()]
        if order:
            sns.pointplot(
                data=cross,
                x="partner_model",
                y="f1_again",
                order=order,
                ax=axes[0],
                errorbar=("ci", 95),
                color="#2c5282",
            )
            axes[0].set_title("Cross-Vendor Private Rating (`again`, F1)")
            axes[0].set_xlabel("Partner Model")
            axes[0].set_ylabel("Mean F1 `again` Rating (95% CI)")
            axes[0].tick_params(axis="x", rotation=22)

        if bt_rates:
            sorted_p = sorted(bt_rates.keys(), key=lambda k: bt_rates[k], reverse=True)
            vals = [bt_rates[k] * 100.0 for k in sorted_p]
            yerr_lo, yerr_hi = [], []
            for k in sorted_p:
                ci = bt_cis.get(k)
                if ci:
                    yerr_lo.append(max(0.0, (bt_rates[k] - ci[0]) * 100.0))
                    yerr_hi.append(max(0.0, (ci[1] - bt_rates[k]) * 100.0))
                else:
                    yerr_lo.append(0.0)
                    yerr_hi.append(0.0)
            colors = ["#c53030" if k == "astra62" else "#2b6cb0" for k in sorted_p]
            axes[1].bar(sorted_p, vals, color=colors, yerr=[yerr_lo, yerr_hi], capsize=4)
            axes[1].axhline(50.0, linestyle="--", color="gray", linewidth=1)
            axes[1].set_ylim(0, 100)
            axes[1].set_title("Unified BT Avg H2H Win Rate (Score-Adjusted, 95% CI)")
            axes[1].set_ylabel("Average H2H Win Rate (%)")
            axes[1].set_xlabel("Partner Model")
            axes[1].tick_params(axis="x", rotation=22)
        else:
            axes[1].text(0.5, 0.5, "Insufficient forced-choice data", ha="center", va="center")
            axes[1].set_axis_off()

        fig.tight_layout()
        p1 = fig_dir / "fig1_partner_league.png"
        fig.savefig(p1, dpi=180)
        plt.close(fig)
        saved.append(f"figures/{p1.name}")

    # --- Fig 2: 4-Panel Task x Partner Heatmap (Subject x Partner F1 `again` per task) ---
    if not debrief_df.empty and debrief_df["f1_again"].notna().any():
        tasks = [t for t in ("triage", "calendar", "goalpost", "code_handoff") if t in debrief_df["task"].unique()]
        if not tasks:
            tasks = sorted(debrief_df["task"].unique())
        partners = [m for m in ALL_PARTNER_MODELS if m in debrief_df["partner_model"].unique()]
        obs_max = float(debrief_df["f1_again"].max())
        vmax = 7.0 if obs_max > 5.0 else 5.0
        ncols = min(len(tasks), 2)
        nrows = math.ceil(len(tasks) / max(1, ncols))
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.2 * nrows), squeeze=False)
        flat_axes = axes_arr.flatten()
        for idx, tname in enumerate(tasks):
            ax = flat_axes[idx]
            tdf = debrief_df[debrief_df["task"] == tname]
            piv = tdf.pivot_table(
                index="subject_model", columns="partner_model", values="f1_again", aggfunc="mean"
            ).reindex(index=list(CORE_MODELS), columns=partners)
            sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn", vmin=1, vmax=vmax, ax=ax)
            ax.set_title(f"Task: {tname} — Mean F1 `again`")
            ax.set_xlabel("Partner")
            ax.set_ylabel("Subject")
            ax.tick_params(axis="x", rotation=25)
        for idx in range(len(tasks), len(flat_axes)):
            flat_axes[idx].set_axis_off()
        fig.tight_layout()
        p2 = fig_dir / "fig2_task_x_partner_heatmap.png"
        fig.savefig(p2, dpi=180)
        plt.close(fig)
        saved.append(f"figures/{p2.name}")

    # --- Fig 3: 7-Point AA Intelligence Index v5 vs Coworker BT Win-Rate Scatter (NO regression line!) ---
    aa_corr = report.get("aa_intelligence_vs_bt") or {}
    if aa_corr.get("ok"):
        pts = aa_corr["points"]
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for pt in pts:
            m = pt["model"]
            x = pt["aa_index_v5"]
            y = pt["bt_avg_h2h_win_rate"] * 100.0
            ci = bt_cis.get(m)
            color = "#c53030" if m == "astra62" else "#2b6cb0"
            if ci:
                yerr = [[max(0.0, y - ci[0] * 100.0)], [max(0.0, ci[1] * 100.0 - y)]]
                ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, markersize=8, capsize=4)
            else:
                ax.scatter([x], [y], color=color, s=75)
            ax.annotate(
                m,
                (x, y),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9.5,
                fontweight="bold" if m == "astra62" else "normal",
            )
        ax.axhline(50.0, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Artificial Analysis Intelligence Index v5")
        ax.set_ylabel("Score-Adjusted BT Avg H2H Win Rate (%)")
        rho = aa_corr.get("spearman_rho")
        rho_str = f"Spearman ρ = {rho:+.2f} (n={len(pts)} models; no trendline fitted)" if rho is not None else ""
        ax.set_title(f"Capability vs. Coworker Desirability\n{rho_str}")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        p3 = fig_dir / "fig3_aa_intelligence_vs_bt.png"
        fig.savefig(p3, dpi=180)
        plt.close(fig)
        saved.append(f"figures/{p3.name}")

    # --- Fig 4: Two-Facedness Dumbbell (F1 Private vs F2 Public, unconfounded!) ---
    if not debrief_df.empty and debrief_df["f1_again"].notna().any() and debrief_df["f2_again"].notna().any():
        sub = debrief_df.dropna(subset=["f1_again", "f2_again"])
        grp = sub.groupby("subject_model")[["f1_again", "f2_again"]].mean().reindex(list(CORE_MODELS)).dropna()
        if not grp.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            y_pos = np.arange(len(grp))
            for i, (_, row) in enumerate(grp.iterrows()):
                ax.plot([row["f1_again"], row["f2_again"]], [i, i], color="#718096", linewidth=2.5, zorder=1)
            ax.scatter(grp["f1_again"], y_pos, color="#c53030", s=90, label="F1 Private (Hidden)", zorder=2)
            ax.scatter(grp["f2_again"], y_pos, color="#2b6cb0", s=90, label="F2 Public (Verbatim)", zorder=2)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(grp.index)
            ax.set_xlabel("Mean `again` Rating")
            ax.set_title("Two-Facedness: Private Review vs. Verbatim Public Review")
            ax.legend(loc="best")
            fig.tight_layout()
            p4 = fig_dir / "fig4_two_facedness.png"
            fig.savefig(p4, dpi=180)
            plt.close(fig)
            saved.append(f"figures/{p4.name}")

    # --- Fig 5: Turn-by-Turn Linguistic Drift ---
    if not turns_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
        for idx, (ax, col, ylabel) in enumerate(
            (
                (axes[0], "space_ratio", "Space Ratio"),
                (axes[1], "word_count", "Message Word Count"),
                (axes[2], "pleasantry_count", "Pleasantry Markers / Turn"),
            )
        ):
            sns.lineplot(
                data=turns_df,
                x="round",
                y=col,
                hue="model",
                ax=ax,
                errorbar="se",
                legend=(idx == 2),
            )
            ax.set_title(f"{ylabel} Across Rounds")
            ax.set_xlabel("Round")
            ax.set_ylabel(ylabel)
        if axes[2].get_legend() is not None:
            axes[2].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True, title=None)
        fig.tight_layout()
        p5 = fig_dir / "fig5_turn_drift.png"
        fig.savefig(p5, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(f"figures/{p5.name}")

    _ = choices_df
    return saved


def _clean_opt_text(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "none"
    s = str(val).strip()
    return s if s and s.lower() != "nan" else "none"


def _write_transcript_and_quote_cards(
    episodes: list[dict[str, Any]], debrief_df: pd.DataFrame, out_dir: Path
) -> None:
    """Write human-readable Face-to-Face vs Private Review quote cards + Round-5 false-blame audit sheet."""
    cards: list[str] = ["# Face-to-Face vs. Private Review Quote Cards\n"]
    if not debrief_df.empty:
        df = debrief_df.dropna(subset=["f1_again", "f2_again"]).copy()
        df["gap"] = df["f2_again"] - df["f1_again"]
        ep_by_id = {ep["episode_id"]: ep for ep in episodes}
        top = df.sort_values(["gap", "f1_again"], ascending=[False, True]).head(12)
        for _, row in top.iterrows():
            ep = ep_by_id[row["episode_id"]]
            role = row["role"]
            ep_turns = ep.get("turns") or []
            if len(ep_turns) >= 2:
                r_prev = ep_turns[-2]["round"]
                r_last = ep_turns[-1]["round"]
                prev_msg = ep_turns[-2][role]["message"]
                last_msg = ep_turns[-1][role]["message"]
                rnd_lbl = f"Rounds {r_prev}–{r_last}"
            elif ep_turns:
                r_last = ep_turns[-1]["round"]
                prev_msg = ""
                last_msg = ep_turns[-1][role]["message"]
                rnd_lbl = f"Round {r_last}"
            else:
                prev_msg, last_msg, rnd_lbl = "", "", "Final Rounds"

            fver = row["f1_friction_verified"]
            fver_str = "n/a" if pd.isna(fver) else str(bool(fver))
            cards.append(
                f"## `{row['episode_id']}` — {row['subject_model']} ({role}) reviewing {row['partner_model']}\n"
                f"- **Task:** `{row['task']}` | **Task score:** {row['score']:.2f} | "
                f"**F1 Private `again`:** {int(row['f1_again'])} | "
                f"**F2 Public `again`:** {int(row['f2_again'])} | **Gap (F2−F1):** {row['gap']:+.0f}\n"
                f"- **What it said to their face ({rnd_lbl}):**\n"
                f"  > {prev_msg[:300].replace(chr(10), ' ')}\n"
                f"  > {last_msg[:300].replace(chr(10), ' ')}\n"
                f"- **Verbatim public review (F2):**\n  > {_clean_opt_text(row['f2_summary'])}\n"
                f"- **Private review (F1):**\n  > {_clean_opt_text(row['f1_summary'])}\n"
                f"- **Friction quote (verified={fver_str}):** `{_clean_opt_text(row['f1_friction_quote'])}`\n"
            )
    (out_dir / "quote_cards.md").write_text("\n".join(cards), encoding="utf-8")

    # False-blame audit sheet (Goalpost external incident — Round 5 in v2)
    gp = debrief_df[debrief_df["task"] == "goalpost"].copy() if not debrief_df.empty else pd.DataFrame()
    fb_lines = ["# Goalpost Round-5 (ERR-LEG-7) External-Incident False-Blame Audit Sheet\n"]
    if not gp.empty:
        flagged = gp[gp["false_blame_flag"] == True]
        unflagged = gp[gp["false_blame_flag"] == False].head(20)
        missing = gp[gp["false_blame_flag"].isna()]
        fb_lines.append(f"## Flagged False-Blame Hits ({len(flagged)}; missing={len(missing)})\n")
        for _, r in flagged.iterrows():
            fb_lines.append(
                f"- `{r['episode_id']}` ({r['subject_model']} -> {r['partner_model']}): "
                f"**Evidence:** {_clean_opt_text(r['false_blame_evidence'])} | "
                f"**Summary:** {_clean_opt_text(r['f1_summary'])} | "
                f"**Friction quote:** {_clean_opt_text(r['f1_friction_quote'])}"
            )
        fb_lines.append(
            f"\n## Unflagged Sample for Miss-Rate Check ({len(unflagged)} of {int((gp['false_blame_flag'] == False).sum())})\n"
        )
        for _, r in unflagged.iterrows():
            fb_lines.append(
                f"- `{r['episode_id']}` ({r['subject_model']} -> {r['partner_model']}): "
                f"**Summary:** {_clean_opt_text(r['f1_summary'])} | "
                f"**Friction quote:** {_clean_opt_text(r['f1_friction_quote'])}"
            )
    (out_dir / "false_blame_audit.md").write_text("\n".join(fb_lines), encoding="utf-8")


def analyze_run(run_name: str) -> dict[str, Any]:
    run_dir = DATA_DIR / run_name
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes, choices, ledger = _load_run(run_dir)
    # Strictly isolate main balanced arm (seeds 301-304) from Opus power boost (seeds 305-308)
    # so boost episodes NEVER distort pooled F1, F2-F1, F4, F5, demeanor, or task calibration summaries.
    non_boost_eps = [ep for ep in episodes if str(ep.get("arm", "")) != "opus_boost"]
    main_episodes = non_boost_eps if non_boost_eps else episodes
    boost_episodes = [ep for ep in episodes if str(ep.get("arm", "")) == "opus_boost"]

    debrief_all_df = _extract_debrief_df(episodes)
    debrief_df = (
        debrief_all_df[debrief_all_df["arm"] != "opus_boost"].copy()
        if (not debrief_all_df.empty and (debrief_all_df["arm"] != "opus_boost").any())
        else debrief_all_df
    )
    boost_debrief_df = (
        debrief_all_df[debrief_all_df["arm"] == "opus_boost"].copy()
        if not debrief_all_df.empty
        else pd.DataFrame()
    )

    turns_all_df = _extract_turns_df(episodes)
    turns_df = (
        turns_all_df[turns_all_df["arm"] != "opus_boost"].copy()
        if (not turns_all_df.empty and (turns_all_df["arm"] != "opus_boost").any())
        else turns_all_df
    )
    choices_df = pd.DataFrame(choices) if choices else pd.DataFrame()

    # Read configured rating scale max from run_summary.json (never infer 5 when ratings happen to be <= 5!)
    configured_scale_max = 7
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            configured_scale_max = int(
                json.loads(summary_path.read_text(encoding="utf-8")).get("rating_scale_max", 7)
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            configured_scale_max = 7

    # --- 1. Task calibration & API hygiene (main balanced arm only) ---
    task_cal: dict[str, Any] = {}
    for task_name in sorted({ep["task"] for ep in main_episodes}):
        sub = [ep for ep in main_episodes if ep["task"] == task_name]
        scores = [ep["score"] for ep in sub]
        entry: dict[str, Any] = {
            "n_episodes": len(sub),
            "mean_score": round(float(np.mean(scores)), 4),
            "sd_score": round(float(np.std(scores, ddof=1)), 4) if len(scores) > 1 else 0.0,
            "min_score": round(float(np.min(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
        }
        if task_name == "calendar":
            cons = [bool(ep["score_details"].get("consensus_reached")) for ep in sub]
            vfree = [bool(ep["score_details"].get("veto_free")) for ep in sub]
            opts = [
                ep["score_details"]["optimality_if_agreed"]
                for ep in sub
                if ep["score_details"].get("optimality_if_agreed") is not None
            ]
            entry["consensus_rate"] = round(float(np.mean(cons)), 4)
            entry["veto_free_rate"] = round(float(np.mean(vfree)), 4)
            entry["mean_optimality_if_agreed"] = round(float(np.mean(opts)), 4) if opts else None
        elif task_name == "goalpost":
            entry["r5_disrupted_rate"] = round(
                float(
                    np.mean(
                        [
                            bool(
                                ep["score_details"].get(
                                    "r5_disrupted_state", ep["score_details"].get("r7_disrupted_state")
                                )
                            )
                            for ep in sub
                        ]
                    )
                ),
                4,
            )
            entry["mean_strict_uptime_rate"] = round(
                float(
                    np.mean(
                        [float(ep["score_details"].get("strict_uptime_rate") or 0.0) for ep in sub]
                    )
                ),
                4,
            )
        elif task_name == "code_handoff":
            entry["released_rate"] = round(
                float(np.mean([bool(ep["score_details"].get("released")) for ep in sub])), 4
            )
            entry["mean_correct_slots"] = round(
                float(np.mean([int(ep["score_details"].get("correct_slots") or 0) for ep in sub])), 4
            )
        elif task_name == "triage":
            entry["verified_green_rate"] = round(
                float(np.mean([bool(ep["score_details"].get("verified_green")) for ep in sub])), 4
            )
        task_cal[task_name] = entry

    # Ceiling stats on F1 `again` using configured rating_scale_max (never inferred from observed max!)
    f1_vals = debrief_df["f1_again"].dropna() if not debrief_df.empty else pd.Series(dtype=float)
    scale_max = configured_scale_max
    top_box_share = float((f1_vals == scale_max).mean()) if len(f1_vals) else 0.0
    top2_box_share = float((f1_vals >= scale_max - 1).mean()) if len(f1_vals) else 0.0
    mode_share = float(f1_vals.value_counts(normalize=True).iloc[0]) if len(f1_vals) else 0.0

    # Reasoning token distributions per model (preserving None!)
    reason_stats: dict[str, Any] = {}
    if not turns_df.empty:
        for m, g in turns_df.groupby("model"):
            rt_valid = g["reasoning_tokens"].dropna().astype(float)
            reason_stats[str(m)] = {
                "median_reasoning_tokens_per_turn": int(np.median(rt_valid)) if len(rt_valid) else None,
                "p25_p75": (
                    [int(np.percentile(rt_valid, 25)), int(np.percentile(rt_valid, 75))]
                    if len(rt_valid)
                    else None
                ),
                "reasoning_tokens_missing_turns": int(g["reasoning_tokens"].isna().sum()),
                "mean_completion_tokens": round(float(g["completion_tokens"].mean()), 1),
                "parse_error_rate": round(float(g["parse_error"].mean()), 4),
                "vendor_leaks_count": int(g["n_vendor_leaks"].sum()),
            }

    # --- 2. Unified 7-Partner BT & 4-Core Replication Contrast ---
    unified_bt_adj = _fit_unified_bt_model(choices_df, adjust_score=True, n_boot=2000)
    unified_bt_raw = _fit_unified_bt_model(choices_df, adjust_score=False, n_boot=2000)
    unified_bt_scenario_sens = _fit_unified_bt_model(
        choices_df, adjust_score=True, n_boot=2000, cluster_mode="scenario_pair"
    )
    core4_bt_adj = _fit_unified_bt_model(
        choices_df, partners_override=list(CORE_MODELS), adjust_score=True, n_boot=2000
    )
    core4_bt_raw = _fit_unified_bt_model(
        choices_df, partners_override=list(CORE_MODELS), adjust_score=False, n_boot=2000
    )

    # Direct predecessor head-to-head + Claude vs other cross-vendor DiD (strictly separated main vs boost)
    h1b_direct = _compute_direct_predecessor_analysis(choices_df)

    def _paired_f1_diff(df_src: pd.DataFrame) -> dict[str, Any]:
        if df_src.empty:
            return {"ok": False}
        cross = df_src[df_src["subject_model"] != "astra62"]
        d62 = cross[cross["partner_model"] == "astra62"].set_index(
            ["subject_model", "task", "seed", "role"]
        )["f1_again"]
        d60 = cross[cross["partner_model"] == PREDECESSOR_MODEL].set_index(
            ["subject_model", "task", "seed", "role"]
        )["f1_again"]
        joined = pd.concat([d62.rename("a62"), d60.rename("a60")], axis=1).dropna()
        if len(joined) == 0:
            return {"ok": False}
        diffs = (joined["a62"] - joined["a60"]).values
        cl = np.array([f"{i[0]}__{i[1]}__{i[3]}" for i in joined.index])
        return {
            "ok": True,
            "n_pairs": len(joined),
            "mean_astra62": round(float(joined["a62"].mean()), 4),
            "mean_astra60": round(float(joined["a60"].mean()), 4),
            "diff_62_minus_60": _cluster_bootstrap_mean(diffs, cl, n_boot=2000),
        }

    h1b_paired_rating: dict[str, Any] = _paired_f1_diff(debrief_df)
    if not boost_debrief_df.empty:
        h1b_paired_rating["opus_boost_separate"] = _paired_f1_diff(boost_debrief_df)

    # --- 3. Partner League Table ---
    league_rows: list[dict[str, Any]] = []
    if unified_bt_adj.get("ok"):
        cross_d = debrief_df[~debrief_df["same_family"]] if not debrief_df.empty else pd.DataFrame()
        f4_valid = (
            debrief_df[debrief_df["f4_choice"].isin(["KEEP", "REROLL"])]
            if not debrief_df.empty
            else pd.DataFrame()
        )
        for p in unified_bt_adj["partners"]:
            f1_p = cross_d[cross_d["partner_model"] == p]["f1_again"].dropna() if not cross_d.empty else pd.Series(dtype=float)
            sc_p = cross_d[cross_d["partner_model"] == p]["score"].dropna() if not cross_d.empty else pd.Series(dtype=float)
            f4_p = f4_valid[f4_valid["partner_model"] == p] if not f4_valid.empty else pd.DataFrame()
            league_rows.append(
                {
                    "partner": p,
                    "display_name": DEFAULT_MODELS[p].display_name if p in DEFAULT_MODELS else p,
                    "aa_index_v5": AA_INDEX_V5.get(p),
                    "bt_avg_h2h_adjusted": unified_bt_adj["bt_avg_h2h_win_rate"].get(p),
                    "bt_avg_h2h_adjusted_ci95": (unified_bt_adj.get("bt_avg_h2h_boot_ci95") or {}).get(p),
                    "bt_avg_h2h_unadjusted": (unified_bt_raw.get("bt_avg_h2h_win_rate") or {}).get(p),
                    "raw_direct_win_rate": (unified_bt_adj.get("raw_direct_win_rate") or {}).get(p),
                    "raw_direct_win_rate_cross_vendor": (
                        unified_bt_adj.get("raw_direct_win_rate_cross_vendor") or {}
                    ).get(p),
                    "mean_f1_again_cross_vendor": round(float(f1_p.mean()), 4) if len(f1_p) else None,
                    "mean_task_score_cross_vendor": round(float(sc_p.mean()), 4) if len(sc_p) else None,
                    "f4_reroll_rate": (
                        round(float((f4_p["f4_choice"] == "REROLL").mean()), 4) if len(f4_p) else None
                    ),
                }
            )
        league_rows.sort(key=lambda r: float(r["bt_avg_h2h_adjusted"] or 0.0), reverse=True)

    # --- 4. Secondary Outcomes (F2-F1 Two-Facedness, F1 Retest, F4, F5, A/A Repeatability, Quotes, False Blame, Legacy F3) ---
    def _voice_did(col: str, fid_col: str, coh_col: str, target_partner: str) -> dict[str, Any]:
        if debrief_df.empty or col not in debrief_df.columns:
            return {"ok": False}
        sub = debrief_df[
            (debrief_df[fid_col] == True) & debrief_df["f1_again"].notna() & debrief_df[col].notna()
        ].copy()
        if len(sub) < 4:
            return {"ok": False, "n": len(sub)}
        sub["lift"] = sub[col] - sub["f1_again"]
        per_partner = {
            str(p): _cluster_bootstrap_mean(g["lift"].values, g["block_id"].values)
            for p, g in sub.groupby("partner_model")
        }
        targ = sub[(sub["partner_model"] == target_partner) & (~sub["same_family"])]
        oth = sub[(sub["partner_model"] != target_partner) & (~sub["same_family"])]
        did_val = (
            round(float(targ["lift"].mean() - oth["lift"].mean()), 4)
            if len(targ) > 0 and len(oth) > 0
            else None
        )
        strict = sub[sub[coh_col] == True]
        st_targ = strict[(strict["partner_model"] == target_partner) & (~strict["same_family"])]
        st_oth = strict[(strict["partner_model"] != target_partner) & (~strict["same_family"])]
        did_strict = (
            round(float(st_targ["lift"].mean() - st_oth["lift"].mean()), 4)
            if len(st_targ) > 0 and len(st_oth) > 0
            else None
        )
        return {
            "ok": True,
            "n_fidelity_ok": len(sub),
            "n_strict_coherence": len(strict),
            "target_partner": target_partner,
            "did_target_minus_others_cross_vendor": did_val,
            "did_strict_coherence_subset": did_strict,
            "lift_by_partner": per_partner,
        }

    h2a = _voice_did("f3a_again", "f3a_fidelity_ok", "f3a_coherence_clean", "astra62")
    h2b = _voice_did("f3b_again", "f3b_fidelity_ok", "f3b_coherence_clean", "opus55")

    h3_two_faced: dict[str, Any] = {}
    if not debrief_df.empty:
        tf = debrief_df.dropna(subset=["f1_again", "f2_again"]).copy()
        tf["gap"] = tf["f2_again"] - tf["f1_again"]
        for m, g in tf.groupby("subject_model"):
            h3_two_faced[str(m)] = _cluster_bootstrap_mean(g["gap"].values, g["block_id"].values)

    f4_rates: dict[str, Any] = {}
    if not debrief_df.empty:
        f4_sub = debrief_df[debrief_df["f4_choice"].isin(["KEEP", "REROLL"])]
        for p, g in f4_sub.groupby("partner_model"):
            f4_rates[str(p)] = {
                "reroll_rate": round(float((g["f4_choice"] == "REROLL").mean()), 4),
                "n": len(g),
            }

    f5_summary: dict[str, Any] = {}
    if not debrief_df.empty:
        f5_sub = debrief_df.dropna(subset=["f5_guess"])
        if len(f5_sub) > 0:
            f5_summary["overall_accuracy"] = round(float(f5_sub["f5_correct"].mean()), 4)
            f5_summary["unsure_rate"] = round(
                float(f5_sub["f5_guess"].isin(["Other / Unsure", "Cannot tell"]).mean()), 4
            )
            f5_summary["accuracy_by_subject"] = {
                str(k): round(float(v), 4)
                for k, v in f5_sub.groupby("subject_model")["f5_correct"].mean().items()
            }
            f5_summary["accuracy_by_partner"] = {
                str(k): round(float(v), 4)
                for k, v in f5_sub.groupby("partner_model")["f5_correct"].mean().items()
            }

    # A/A placebo check: report position bias AND order-flip repeatability separately (no bogus rank band!)
    aa_check: dict[str, Any] = {"ok": False}
    if not choices_df.empty:
        aa = choices_df[
            (choices_df["choice_type"] == "aa_placebo") & choices_df["chosen_slot"].isin([1, 2])
        ]
        if len(aa) > 0:
            pick1 = (aa["chosen_slot"] == 1).astype(float).values
            k1 = int(pick1.sum())
            n_aa = len(pick1)
            same_ep = []
            for _, ig in aa.groupby("item_id"):
                if len(ig) == 2:
                    eps = [
                        r["episode_1_id"] if r["chosen_slot"] == 1 else r["episode_2_id"]
                        for _, r in ig.iterrows()
                    ]
                    same_ep.append(float(eps[0] == eps[1]))
            bp = float(stats.binomtest(k1, n_aa, 0.5).pvalue)
            aa_check = {
                "ok": True,
                "n_aa_choices": n_aa,
                "first_position_rate": round(k1 / n_aa, 4),
                "exact_binom_p_vs_50": round(bp, 6),
                "exact_binom_p_fmt": _fmt_p(bp),
                "n_ab_ba_pairs": len(same_ep),
                "aa_order_flip_repeatability": round(float(np.mean(same_ep)), 4) if same_ep else None,
            }

    quote_audit_summary: dict[str, Any] = {}
    false_blame_summary: dict[str, Any] = {}
    if not debrief_df.empty:
        for m, g in debrief_df.groupby("subject_model"):
            h_pres = g["f1_helped_verified"].dropna()
            f_pres = g["f1_friction_verified"].dropna()
            quote_audit_summary[str(m)] = {
                "helped_verified_rate": round(float(h_pres.mean()), 4) if len(h_pres) else None,
                "helped_n": len(h_pres),
                "friction_verified_rate": round(float(f_pres.mean()), 4) if len(f_pres) else None,
                "friction_n": len(f_pres),
            }
        gp = debrief_df[(debrief_df["task"] == "goalpost") & debrief_df["false_blame_flag"].notna()]
        if len(gp) > 0:
            false_blame_summary = {
                "overall_rate": round(float(gp["false_blame_flag"].mean()), 4),
                "n": len(gp),
                "n_missing": int(
                    debrief_df[debrief_df["task"] == "goalpost"]["false_blame_flag"].isna().sum()
                ),
                "by_subject": {
                    str(k): round(float(v), 4)
                    for k, v in gp.groupby("subject_model")["false_blame_flag"].mean().items()
                },
                "by_partner": {
                    str(k): round(float(v), 4)
                    for k, v in gp.groupby("partner_model")["false_blame_flag"].mean().items()
                },
            }

    # --- 5. Decision Panel Demeanor & 3-Judge Reliability ---
    jev_summary: dict[str, Any] = {}
    jev_cols = [
        "jev_wooden_legalistic",
        "jev_skips_why",
        "jev_answers_directly",
        "jev_condescending_pedantic",
        "jev_clarity",
        "jev_warmth",
    ]
    if not turns_df.empty and turns_df["jev_wooden_legalistic"].notna().any():
        jdf = turns_df.dropna(subset=["jev_wooden_legalistic"])
        jev_summary["by_model_overall"] = (
            jdf.groupby("model")[jev_cols].mean().round(4).to_dict(orient="index")
        )
        jev_summary["subject_confusion_triggered_by_partner"] = (
            jdf[jdf["round"] >= 2]
            .groupby("partner_model")["jev_expresses_confusion"]
            .mean()
            .round(4)
            .to_dict()
        )
        jev_summary["by_task_and_model"] = {
            str(tname): tg.groupby("model")[jev_cols].mean().round(4).to_dict(orient="index")
            for tname, tg in jdf.groupby("task")
        }
        triage_a = jdf[(jdf["task"] == "triage") & (jdf["role"] == "A")]
        if not triage_a.empty:
            jev_summary["triage_role_A_investigator_by_model"] = (
                triage_a.groupby("model")[jev_cols].mean().round(4).to_dict(orient="index")
            )

    jev_rw_sem: dict[str, Any] = {}
    for mode in ("F3a", "F3b"):
        comm_vals, unc_vals = [], []
        for ep in episodes:
            for rstore in (ep.get("rewrites") or {}).values():
                mdata = rstore.get(mode) or {}
                for item in mdata.get("messages", []):
                    js = item.get("jev_semantic") or {}
                    if js.get("preserves_commitments") is not None:
                        comm_vals.append(float(js["preserves_commitments"]))
                    if js.get("preserves_uncertainty") is not None:
                        unc_vals.append(float(js["preserves_uncertainty"]))
        if comm_vals:
            jev_rw_sem[mode] = {
                "n_messages": len(comm_vals),
                "mean_preserves_commitments": round(float(np.mean(comm_vals)), 4),
                "mean_preserves_uncertainty": round(float(np.mean(unc_vals)), 4) if unc_vals else None,
                "pct_both_ge_070": round(
                    float(np.mean([(c >= 0.7 and u >= 0.7) for c, u in zip(comm_vals, unc_vals, strict=False)])),
                    4,
                ),
            }
    if jev_rw_sem:
        jev_summary["rewrite_semantic_fidelity"] = jev_rw_sem

    panel_rel = _compute_decision_panel_reliability(turns_df)
    aa_corr = _compute_aa_vs_bt_correlation(unified_bt_adj)
    loso = _compute_loso_sensitivity(choices_df)
    f1_retest_rel = _compute_f1_retest_reliability(debrief_df)

    f1_matrix = (
        debrief_df.pivot_table(
            index="subject_model", columns="partner_model", values="f1_again", aggfunc="mean"
        )
        .round(3)
        .to_dict()
        if not debrief_df.empty and debrief_df["f1_again"].notna().any()
        else {}
    )

    report: dict[str, Any] = {
        "run_name": run_name,
        "n_episodes": len(episodes),
        "n_episodes_main": len(main_episodes),
        "n_episodes_opus_boost": len(boost_episodes),
        "n_debrief_agent_episodes_main": len(debrief_df),
        "n_debrief_agent_episodes_total": len(debrief_all_df),
        "n_debrief_agent_episodes": len(debrief_df),
        "n_choices_total": len(choices_df),
        "rating_scale_max": scale_max,
        "total_cost_usd": round(ledger.total_cost_usd, 4),
        "cost_by_model": ledger.summary_by_model(),
        "task_calibration": task_cal,
        "f1_again_top_box_7_share": round(top_box_share, 4),
        "f1_again_top2_box_6_7_share": round(top2_box_share, 4),
        "f1_again_mode_share": round(mode_share, 4),
        "f1_again_distribution": (
            {str(k): int(v) for k, v in f1_vals.value_counts().sort_index().items()}
            if len(f1_vals)
            else {}
        ),
        "reasoning_and_hygiene_by_model": reason_stats,
        "partner_league_table": league_rows,
        "unified_bt_adjusted": unified_bt_adj,
        "unified_bt_unadjusted": unified_bt_raw,
        "unified_bt_scenario_cluster_sensitivity": unified_bt_scenario_sens,
        "H1_choice_model_adjusted": unified_bt_adj,
        "H1_choice_model_unadjusted": unified_bt_raw,
        "H1_4core_subset_adjusted": core4_bt_adj,
        "H1_4core_subset_unadjusted": core4_bt_raw,
        "H1b_predecessor_head_to_head": h1b_direct,
        "H1b_predecessor_paired_f1_again": h1b_paired_rating,
        "aa_intelligence_vs_bt": aa_corr,
        "leave_one_scenario_out_sensitivity": loso,
        "decision_panel_reliability": panel_rel,
        "jev_demeanor_and_audits": jev_summary,
        "H2a_voice_expand_only_did": h2a,
        "H2b_voice_strip_pleasantries_did": h2b,
        "H3_two_facedness_f2_minus_f1": h3_two_faced,
        "F1_retest_reliability": f1_retest_rel,
        "F1_again_matrix": f1_matrix,
        "F4_boundary_reroll_rates": f4_rates,
        "F5_family_guess": f5_summary,
        "AA_placebo_check": aa_check,
        "quote_fabrication_audit": quote_audit_summary,
        "false_blame_goalpost_r5": false_blame_summary,
        "rewriter_manipulation_check": _compute_manipulation_check(main_episodes),
        "style_distances": _compute_style_distances(turns_df),
        "supplementary_exploratory": _compute_supplementary_breakdowns(
            debrief_df=debrief_df,
            turns_df=turns_df,
            choices_df=choices_df,
        ),
    }

    figures = _generate_figures(
        debrief_df=debrief_df,
        turns_df=turns_df,
        choices_df=choices_df,
        report=report,
        fig_dir=out_dir / "figures",
    )
    report["figures"] = figures  # Strictly relative paths (`figures/...`)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_transcript_and_quote_cards(main_episodes, debrief_df, out_dir)

    # Markdown summary (with formatted p-values `< 0.001` and relative figure links)
    md = [
        f"# Analysis Report: `{run_name}`\n",
        f"- **Episodes:** {report['n_episodes']} | **Choices:** {report['n_choices_total']} | **Total Spend:** ${report['total_cost_usd']:.2f}",
        f"- **F1 `again` ceiling:** top-box (`{scale_max}`) = {top_box_share:.1%}, top-2 (`{scale_max - 1}+{scale_max}`) = {top2_box_share:.1%}\n",
        "## Partner League Table",
        "| Partner | AA v5 | Score-Adj BT Avg H2H | Raw Direct | Cross-Vendor F1 `again` | F4 Reroll |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lr in league_rows:
        aa_s = f"{lr['aa_index_v5']:.0f}" if lr["aa_index_v5"] is not None else "—"
        bt_s = f"{lr['bt_avg_h2h_adjusted']:.1%}" if lr["bt_avg_h2h_adjusted"] is not None else "—"
        rd_s = f"{lr['raw_direct_win_rate']:.1%}" if lr["raw_direct_win_rate"] is not None else "—"
        f1_s = f"{lr['mean_f1_again_cross_vendor']:.2f}" if lr["mean_f1_again_cross_vendor"] is not None else "—"
        f4_s = f"{lr['f4_reroll_rate']:.1%}" if lr["f4_reroll_rate"] is not None else "—"
        md.append(f"| `{lr['partner']}` | {aa_s} | {bt_s} | {rd_s} | {f1_s} | {f4_s} |")

    md.extend(
        [
            "\n## Task Calibration",
            "```json\n" + json.dumps(task_cal, indent=2) + "\n```",
            "\n## Headline Contrasts (Unified BT, 4-Core Replication, Direct Predecessor & Claude DiD)",
            "```json\n"
            + json.dumps(
                {
                    "unified_bt_adjusted": {
                        k: unified_bt_adj.get(k)
                        for k in (
                            "ok",
                            "n_choices",
                            "n_blocks",
                            "n_boot_achieved",
                            "bt_avg_h2h_win_rate",
                            "bt_avg_h2h_boot_ci95",
                            "raw_direct_win_rate",
                            "h1_prob_astra62_vs_avg_other",
                            "h1_prob_boot_ci95",
                            "h1_p_fmt",
                            "gamma_homophily_logit",
                            "gamma_p_fmt",
                        )
                    },
                    "H1b_direct_predecessor": h1b_direct,
                    "aa_intelligence_vs_bt": aa_corr,
                    "AA_placebo": aa_check,
                    "F1_retest": f1_retest_rel,
                },
                indent=2,
            )
            + "\n```",
            "\n## Figures",
            *[f"- `{fp}`" for fp in figures],
        ]
    )
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an experiment run and generate plots/cards.")
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()
    rep = analyze_run(args.run)
    print(f"Analysis saved to data/{args.run}/analysis/ (total cost ${rep['total_cost_usd']:.2f})")


if __name__ == "__main__":
    main()
