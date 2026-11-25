"""v2.1 Deep Meta-Analysis library (zero API calls, deterministic, reproducible).

Computes all Phase 1 meta-analyses directly from raw ``data/v2/`` artifacts:
1. Construct validity & verbosity decomposition of 3-judge scores
2. Visibility-masked cluster-robust choice logits + zero-tolerance tone-mention audit
3. Capstone Tone-Flip multi-dimension delta audit + 4-lab consensus vs. live win rates
4. Bloat tax, billing-component decomposition, verbosity contagion, & exact-tie efficiency
5. Triage 7x7 paralysis grid, score upsets, Calendar shortlist loophole, & Neon glitch audit
6. Clean disclosure meta-analysis via ``common_ground.disclosure``
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError, SingularMatrixWarning

from common_ground.config import (
    ALL_PARTNER_MODELS,
    CORE_MODELS,
    DECISION_PANEL_SPECS,
    DEFAULT_MODELS,
)
from common_ground.disclosure import (
    GOALPOST_ROLE_A_RULES,
    GOALPOST_ROLE_B_RULES,
    check_goalpost_episode_disclosure,
    check_triage_operator_disclosure,
    is_triage_b_zero_contradiction,
)

DEFAULT_SEED = 20261122
JUDGE_DIMS: tuple[str, ...] = (
    "clarity",
    "warmth",
    "skips_why",
    "answers_directly",
    "wooden_legalistic",
    "condescending_pedantic",
)
LINGUISTIC_COVARIATES: tuple[str, ...] = (
    "word_count",
    "log_wc",
    "bullets",
    "headers",
    "ids_per_100",
    "pleasantries",
)
SINGLE_TRAIT_PREDICTORS: tuple[str, ...] = (
    "d_clarity",
    "d_clarity_resid",
    "d_log_wc",
    "d_skips_why",
    "d_wooden_legalistic",
    "d_warmth",
    "d_answers_directly",
    "d_condescending_pedantic",
    "d_s_conf",
    "d_ids_per_100",
)
MULTIVARIATE_PREDICTORS: tuple[str, ...] = (
    "d_score",
    "d_clarity_resid",
    "d_log_wc",
    "d_skips_why",
    "d_wooden_legalistic",
    "d_warmth",
    "d_answers_directly",
    "d_condescending_pedantic",
    "d_s_conf",
    "d_ids_per_100",
)
TASKS_ORDERED: tuple[str, ...] = ("calendar", "code_handoff", "goalpost", "triage")

_BULLET_RE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+")
_HEADER_RE = re.compile(r"(?:^|\n)\s*#{1,6}\s+")
_ACTION_LINE_RE = re.compile(r"(?:^|\n)\s*ACTION\s*:", re.MULTILINE)
_VISIBLE_ACTION_RE = re.compile(r"\bACTION\s*:")
_MTG_RE = re.compile(r"\bMTG-[1-7]\b")
_SLOT_RE = re.compile(r"\b(?:MON|TUE|WED|THU|FRI)-[12]\b")
_SWAP_LINE_RE = re.compile(
    r"(?:->|→|\bfrom\s+(?:MON|TUE|WED|THU|FRI)-[12]\s+to\s+(?:MON|TUE|WED|THU|FRI)-[12]\b|"
    r"\bswap\b|\bmove\s+MTG-[1-7]\s+from\b)",
    re.IGNORECASE,
)

# Zero-tolerance tone-mention lexicons
_STRICT_POLITENESS_RE = re.compile(
    r"\b(?:polite\w*|impolite\w*|rude\w*|warm\w*|cold\w*|friendl\w*|unfriendl\w*|"
    r"cordial\w*|courteous\w*|pleasantr\w*|pleasant\w*|harsh\w*|abrasive\w*|"
    r"condescend\w*|patroniz\w*|bedside|demeanor|supportive\w*|empath\w*|"
    r"gracious\w*|collegial\w*|congenial\w*|respectful\w*|fluff)\b",
    re.IGNORECASE,
)
_DOMAIN_WARM_COLD_FP_RE = re.compile(
    r"\b(?:cold|warm)[- ]?(?:start|cache|standby|pool|up|tier|storage|boot)\b",
    re.IGNORECASE,
)
_TERSE_TONE_RE = re.compile(
    r"\b(?:terse\w*|blunt\w*|curt\w*|brusque\w*|tone|manner\w*|niceties)\b",
    re.IGNORECASE,
)

# Systematic rationale theme detectors (non-mutually-exclusive)
_RATIONALE_THEME_PATTERNS: dict[str, re.Pattern[str]] = {
    "transparency_and_disclosure": re.compile(
        r"\b(?:transparen\w*|disclos\w*|shared?\s+(?:its|their|his|her|all|full|exact|every|private|concrete)|"
        r"upfront|up\s+front|constraint\w*|invariant\w*|option\s+(?:list|menu|id)|exact\s+(?:cost|total|penalt|formula|threshold|cap))\b",
        re.IGNORECASE,
    ),
    "verification_and_accuracy": re.compile(
        r"\b(?:verif\w*|accura\w*|check\w*|re-?check\w*|calibrat\w*|evidence|caught|correct\w*|rigor\w*|grounded|measured)\b",
        re.IGNORECASE,
    ),
    "speed_and_convergence": re.compile(
        r"\b(?:faster|sooner|earlier|round\s+\d|fewer\s+rounds?|first\s+eligible|promptly|immediately|converg\w*|quick\w*)\b",
        re.IGNORECASE,
    ),
    "protocol_and_rule_adherence": re.compile(
        r"\b(?:2[- ]cell|two[- ]cell|disclosure\s+(?:limit|rule|norm|cap)|guardrail\w*|veto[- ]free|"
        r"fallback\w*|rule[- ]abiding|baseline|safe\w*|disciplin\w*)\b",
        re.IGNORECASE,
    ),
    "errors_hallucinations_garbled": re.compile(
        r"\b(?:garbl\w*|hallucinat\w*|fabricat\w*|contradict\w*|omitt\w*|action\s+line|malformed|"
        r"error\w*|mistak\w*|wrong|invalid|broke|violat\w*|confus\w*|inconsisten\w*|unverified|missed)\b",
        re.IGNORECASE,
    ),
    "clarity_and_structure": re.compile(
        r"\b(?:clear\w*|structur\w*|concis\w*|organiz\w*|explicit\w*|actionable|crisp|systematic\w*|direct\w*)\b",
        re.IGNORECASE,
    ),
}

_CALENDAR_2CELL_RULE_RE = re.compile(
    r"\b(?:2[- ]cell|two[- ]cell|disclosure\s+(?:limit|rule|norm|cap)|"
    r"exceed\w*\s+the\s+disclosure|violat\w*\s+the\s+(?:2|two|disclosure)|"
    r"broke\s+the\s+(?:2|two|disclosure)|shortlist|dump\w*\s+(?:its|their|his|her)?\s*(?:entire|full|whole)?\s*(?:private\s+)?(?:table|list|matrix))\b",
    re.IGNORECASE,
)


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def _round_opt(val: float | None, digits: int = 4) -> float | None:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return round(float(val), digits)


def _extract_choice_reason(row: dict[str, Any]) -> str:
    """Extract rationale text from either a main choice row or a Capstone choice row."""
    reason = row.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    raw = str(row.get("raw") or "").strip()
    if raw:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict) and parsed.get("reason"):
                    return str(parsed["reason"]).strip()
            except json.JSONDecodeError:
                pass
    return raw


def _filter_main_episodes(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Strictly isolate complete non-``opus_boost`` episodes."""
    kept: list[dict[str, Any]] = []
    n_excluded = 0
    for ep in episodes:
        if ep.get("complete") is False:
            n_excluded += 1
            continue
        if ep.get("arm") == "opus_boost":
            n_excluded += 1
            continue
        kept.append(ep)
    return kept, n_excluded


def load_v2_raw_data(run_dir: Path) -> dict[str, Any]:
    """Load v2 episodes, deduplicated choices, Capstone files, and ledger path (with Parquet fallback)."""
    from common_ground.parquet_loader import (
        load_capstone_from_parquet,
        load_choices_from_parquet,
        load_episodes_from_parquet,
        resolve_hf_export_dir,
    )

    ep_dir = run_dir / "episodes"
    ep_files = sorted(ep_dir.glob("*.json")) if ep_dir.exists() else []
    hf_dir = resolve_hf_export_dir(run_dir)

    if ep_files:
        all_ep_files = [json.loads(p.read_text(encoding="utf-8")) for p in ep_files]
        episodes_all = [e for e in all_ep_files if e.get("complete") is True]
        episodes_incomplete = [e["episode_id"] for e in all_ep_files if e.get("complete") is not True]
    elif hf_dir is not None:
        episodes_all, episodes_incomplete = load_episodes_from_parquet(hf_dir)
    else:
        episodes_all, episodes_incomplete = [], []
    episodes_main = [e for e in episodes_all if e.get("arm") != "opus_boost"]
    episodes_boost = [e for e in episodes_all if e.get("arm") == "opus_boost"]

    choices_path = run_dir / "choices.jsonl"
    choices_by_id: dict[str, dict[str, Any]] = {}
    if choices_path.exists():
        for line in choices_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = (
                row.get("manifest_item_id")
                or f"{row.get('choice_type')}__{row.get('item_id')}__{row.get('order')}"
            )
            choices_by_id[cid] = row
    elif hf_dir is not None:
        for row in load_choices_from_parquet(hf_dir):
            cid = (
                row.get("manifest_item_id")
                or f"{row.get('choice_type')}__{row.get('item_id')}__{row.get('order')}"
            )
            choices_by_id[cid] = row

    cap_dir = run_dir / "capstone_tone_flip"
    cap_rewrites_path = cap_dir / "capstone_rewrites.jsonl"
    cap_choices_path = cap_dir / "capstone_choices.jsonl"
    if cap_rewrites_path.exists() or cap_choices_path.exists():
        capstone_rewrites = (
            [
                json.loads(line)
                for line in cap_rewrites_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if cap_rewrites_path.exists()
            else []
        )
        capstone_choices = (
            [
                json.loads(line)
                for line in cap_choices_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if cap_choices_path.exists()
            else []
        )
    elif hf_dir is not None:
        capstone_rewrites, capstone_choices = load_capstone_from_parquet(hf_dir)
    else:
        capstone_rewrites, capstone_choices = [], []

    return {
        "episodes_all": episodes_all,
        "episodes_main": episodes_main,
        "episodes_boost": episodes_boost,
        "episodes_incomplete": episodes_incomplete,
        "choices": list(choices_by_id.values()),
        "capstone_rewrites": capstone_rewrites,
        "capstone_choices": capstone_choices,
        "ledger_path": run_dir / "ledger.jsonl",
    }


def _extract_turn_construct_df(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        ep_id = str(ep["episode_id"])
        task = str(ep["task"])
        arm = str(ep.get("arm") or "unknown")
        total_rounds = int(ep.get("total_rounds") or len(ep.get("turns") or []))
        for t in ep.get("turns") or []:
            rnd = int(t["round"])
            for role in ("A", "B"):
                side = t[role]
                jd = side.get("jev_demeanor") or {}
                m = side.get("metrics") or {}
                txt = str(side.get("message") or "")
                wc = int(m.get("word_count") if m.get("word_count") is not None else len(txt.split()))
                pjudges = jd.get("panel_judges") or {}
                row: dict[str, Any] = {
                    "episode_id": ep_id,
                    "task": task,
                    "arm": arm,
                    "round": rnd,
                    "total_rounds": total_rounds,
                    "role": role,
                    "model": str(side["model"]),
                    "word_count": float(wc),
                    "log_wc": math.log1p(max(0, wc)),
                    "bullets": float(len(_BULLET_RE.findall(txt))),
                    "headers": float(len(_HEADER_RE.findall(txt))),
                    "ids_per_100": float(m.get("ids_per_100") or 0.0),
                    "pleasantries": float(m.get("pleasantry_count") or 0.0),
                    "expresses_confusion": (
                        float(jd["expresses_confusion"])
                        if jd.get("expresses_confusion") is not None
                        else np.nan
                    ),
                }
                for dim in JUDGE_DIMS:
                    val = jd.get(dim)
                    row[dim] = float(val) if val is not None else np.nan
                for jk in DECISION_PANEL_SPECS:
                    jscores = (pjudges.get(jk) or {}).get("scores") or {}
                    cval = jscores.get("clarity")
                    row[f"j_{jk}_clarity"] = float(cval) if cval is not None else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _fit_length_ols(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Fit OLS: ``target ~ log_wc + log_wc^2 + task + role`` and return summary + residuals."""
    sub = df.dropna(subset=[target_col, "log_wc"]).copy()
    if len(sub) < 6:
        return {"ok": False, "n": len(sub), "residuals": pd.Series(dtype=float)}

    cat_cols: list[str] = []
    if sub["task"].nunique() > 1:
        cat_cols.append("task")
    if sub["role"].nunique() > 1:
        cat_cols.append("role")

    if cat_cols:
        X_df = pd.get_dummies(sub[cat_cols], drop_first=True, dtype=float)
    else:
        X_df = pd.DataFrame(index=sub.index)
    X_df["log_wc"] = sub["log_wc"].astype(float)
    X_df["log_wc_sq"] = (sub["log_wc"].astype(float)) ** 2
    X_df = sm.add_constant(X_df, has_constant="add")

    # Drop any zero-variance / strictly collinear columns deterministically
    keep_cols: list[str] = ["const"]
    for col in X_df.columns:
        if col == "const":
            continue
        if float(X_df[col].std()) <= 1e-12:
            continue
        cand = X_df[keep_cols + [col]].to_numpy(dtype=float)
        if np.linalg.matrix_rank(cand) == cand.shape[1]:
            keep_cols.append(col)
    X_fit = X_df[keep_cols]

    ols = sm.OLS(sub[target_col].astype(float), X_fit).fit()
    if float(ols.centered_tss) > 1e-12:
        r_sq = round(float(ols.rsquared), 4)
        adj_r_sq = round(float(ols.rsquared_adj), 4)
    else:
        r_sq = 0.0
        adj_r_sq = 0.0
    return {
        "ok": True,
        "n": len(sub),
        "r_squared": r_sq,
        "adj_r_squared": adj_r_sq,
        "grand_mean": round(float(sub[target_col].mean()), 4),
        "params": {str(k): round(float(v), 6) for k, v in ols.params.items()},
        "bse": {str(k): round(float(v), 6) for k, v in ols.bse.items()},
        "pvalues": {str(k): round(float(v), 6) for k, v in ols.pvalues.items()},
        "design_columns": list(keep_cols),
        "residuals": pd.Series(ols.resid.to_numpy(dtype=float), index=sub.index),
    }


def predict_from_length_ols(df: pd.DataFrame, ols_params: dict[str, float]) -> np.ndarray:
    """Predict target values on arbitrary turns from a main-arm ``_fit_length_ols`` parameter map."""
    n = len(df)
    preds = np.full(n, float(ols_params.get("const", 0.0)), dtype=float)
    lwc = df["log_wc"].to_numpy(dtype=float)
    if "log_wc" in ols_params:
        preds += float(ols_params["log_wc"]) * lwc
    if "log_wc_sq" in ols_params:
        preds += float(ols_params["log_wc_sq"]) * (lwc**2)
    tasks = df["task"].astype(str).to_numpy()
    roles = df["role"].astype(str).to_numpy()
    for col, coef in ols_params.items():
        if col.startswith("task_"):
            tname = col[len("task_") :]
            preds += float(coef) * (tasks == tname).astype(float)
        elif col.startswith("role_"):
            rname = col[len("role_") :]
            preds += float(coef) * (roles == rname).astype(float)
    return preds


def compute_construct_validity(episodes_main: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit construct validity of the 3-judge panel on main-arm turns (excluding ``opus_boost``).

    Returns turn-level Pearson & Spearman correlations across all 6 judge dimensions x 6 linguistic
    covariates, OLS R^2 of ``y ~ log_wc + log_wc^2 + task + role`` for ``clarity``, ``skips_why``,
    and ``wooden_legalistic``, per-model raw vs. length-adjusted rankings, per-judge clarity
    breakdowns, and ``clarity_resid_map`` keyed by ``(episode_id, round, role)``.
    """
    main_eps, n_excluded = _filter_main_episodes(episodes_main)
    tdf = _extract_turn_construct_df(main_eps)
    if tdf.empty:
        return {
            "ok": False,
            "scope": "main_arm_excluding_opus_boost",
            "n_episodes_main": len(main_eps),
            "n_episodes_excluded": n_excluded,
            "clarity_resid_map": {},
        }

    # 1. Turn-level Pearson & Spearman correlations (6 judge dims x 6 linguistic covariates)
    correlations: dict[str, dict[str, dict[str, Any]]] = {}
    for dim in JUDGE_DIMS:
        correlations[dim] = {}
        for cov in LINGUISTIC_COVARIATES:
            pair = tdf[[dim, cov]].dropna()
            x = pair[dim].to_numpy(dtype=float)
            y = pair[cov].to_numpy(dtype=float)
            if len(pair) >= 3 and float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
                pr = stats.pearsonr(x, y)
                sp = stats.spearmanr(x, y)
                correlations[dim][cov] = {
                    "n": len(pair),
                    "pearson_r": round(float(pr.statistic), 4),
                    "pearson_p": round(float(pr.pvalue), 6),
                    "spearman_rho": round(float(sp.statistic), 4),
                    "spearman_p": round(float(sp.pvalue), 6),
                }
            else:
                correlations[dim][cov] = {
                    "n": len(pair),
                    "pearson_r": None,
                    "pearson_p": None,
                    "spearman_rho": None,
                    "spearman_p": None,
                }

    # 2. OLS R^2: y ~ log_wc + log_wc^2 + task + role
    ols_targets = ("clarity", "skips_why", "wooden_legalistic")
    ols_results: dict[str, Any] = {}
    for dim in ols_targets:
        fit_info = _fit_length_ols(tdf, dim)
        resid_series = fit_info.pop("residuals")
        tdf[f"{dim}_resid"] = np.nan
        if not resid_series.empty:
            tdf.loc[resid_series.index, f"{dim}_resid"] = resid_series
        ols_results[dim] = fit_info

    # 3. Build turn-level clarity_resid_map keyed by (episode_id, round, role)
    clarity_resid_map: dict[tuple[str, int, str], float] = {}
    valid_cr = tdf.dropna(subset=["clarity_resid"])
    for _, row in valid_cr.iterrows():
        clarity_resid_map[(str(row["episode_id"]), int(row["round"]), str(row["role"]))] = float(
            row["clarity_resid"]
        )

    # 4. Per-model raw vs. length-adjusted summary + ranks
    models_present = [m for m in ALL_PARTNER_MODELS if m in set(tdf["model"])]
    models_present += [m for m in sorted(set(tdf["model"])) if m not in models_present]
    grand_means = {dim: float(tdf[dim].dropna().mean()) for dim in ols_targets if tdf[dim].notna().any()}

    per_model: dict[str, dict[str, Any]] = {}
    for m in models_present:
        mg = tdf[tdf["model"] == m]
        entry: dict[str, Any] = {
            "n_turns": len(mg),
            "mean_word_count": _round_opt(float(mg["word_count"].mean()), 4),
            "mean_log_wc": _round_opt(float(mg["log_wc"].mean()), 4),
            "mean_bullets": _round_opt(float(mg["bullets"].mean()), 3),
            "mean_headers": _round_opt(float(mg["headers"].mean()), 3),
            "mean_ids_per_100": _round_opt(float(mg["ids_per_100"].mean()), 4),
            "mean_pleasantries": _round_opt(float(mg["pleasantries"].mean()), 3),
        }
        for dim in ols_targets:
            raw_mean = float(mg[dim].dropna().mean()) if mg[dim].notna().any() else None
            resid_mean = (
                float(mg[f"{dim}_resid"].dropna().mean())
                if mg[f"{dim}_resid"].notna().any()
                else None
            )
            adj_mean = (
                (grand_means[dim] + resid_mean)
                if (resid_mean is not None and dim in grand_means)
                else None
            )
            entry[f"{dim}_raw"] = _round_opt(raw_mean, 4)
            entry[f"{dim}_resid"] = _round_opt(resid_mean, 4)
            entry[f"{dim}_length_adjusted"] = _round_opt(adj_mean, 4)
        for dim in ("warmth", "answers_directly", "condescending_pedantic"):
            entry[f"{dim}_raw"] = (
                _round_opt(float(mg[dim].dropna().mean()), 4) if mg[dim].notna().any() else None
            )
        per_model[m] = entry

    # Compute raw vs. length-adjusted ranks for clarity (1 = highest)
    for key_suffix, rank_name, reverse in (
        ("clarity_raw", "clarity_raw_rank", True),
        ("clarity_resid", "clarity_length_adjusted_rank", True),
        ("skips_why_raw", "skips_why_raw_rank_lowest_best", False),
        ("skips_why_resid", "skips_why_length_adjusted_rank_lowest_best", False),
    ):
        scored = [
            (m, per_model[m][key_suffix])
            for m in models_present
            if per_model[m].get(key_suffix) is not None
        ]
        scored.sort(key=lambda kv: kv[1], reverse=reverse)
        for rank_idx, (m, _) in enumerate(scored, start=1):
            per_model[m][rank_name] = rank_idx

    # 5. Per-judge (liquid, perplexity, typesafe) clarity breakdown (raw + length-adjusted)
    per_judge_clarity: dict[str, Any] = {}
    for jk in DECISION_PANEL_SPECS:
        jcol = f"j_{jk}_clarity"
        if jcol not in tdf.columns or not tdf[jcol].notna().any():
            per_judge_clarity[jk] = {"ok": False, "n_turns": 0}
            continue
        j_ols = _fit_length_ols(tdf, jcol)
        j_res = j_ols.pop("residuals")
        tdf[f"{jcol}_resid"] = np.nan
        if not j_res.empty:
            tdf.loc[j_res.index, f"{jcol}_resid"] = j_res
        pair_wc = tdf[[jcol, "log_wc"]].dropna()
        r_lwc = (
            round(float(stats.pearsonr(pair_wc[jcol], pair_wc["log_wc"]).statistic), 4)
            if len(pair_wc) >= 3
            and float(pair_wc[jcol].std()) > 1e-12
            and float(pair_wc["log_wc"].std()) > 1e-12
            else None
        )
        by_m: dict[str, dict[str, Any]] = {}
        for m in models_present:
            mg = tdf[tdf["model"] == m]
            by_m[m] = {
                "n_turns": int(mg[jcol].notna().sum()),
                "raw_mean": _round_opt(float(mg[jcol].dropna().mean()), 4)
                if mg[jcol].notna().any()
                else None,
                "resid_mean": _round_opt(float(mg[f"{jcol}_resid"].dropna().mean()), 4)
                if mg[f"{jcol}_resid"].notna().any()
                else None,
            }
        per_judge_clarity[jk] = {
            "ok": True,
            "n_turns": int(tdf[jcol].notna().sum()),
            "overall_mean": _round_opt(float(tdf[jcol].dropna().mean()), 4),
            "pearson_r_with_log_wc": r_lwc,
            "ols_r_squared": j_ols.get("r_squared"),
            "by_model": by_m,
        }

    return {
        "ok": True,
        "scope": "main_arm_excluding_opus_boost",
        "n_episodes_main": len(main_eps),
        "n_episodes_excluded": n_excluded,
        "n_turns": len(tdf),
        "n_turns_with_clarity": len(valid_cr),
        "correlations": correlations,
        "ols_length_models": ols_results,
        "per_model": per_model,
        "per_judge_clarity": per_judge_clarity,
        "clarity_resid_map": clarity_resid_map,
    }


def audit_choice_reasons(
    choices: list[dict[str, Any]],
    *,
    label: str = "non_aa_choices",
    is_capstone: bool = False,
) -> dict[str, Any]:
    """Zero-tolerance tone/politeness audit + systematic rationale theme counts."""
    strict_hits: list[dict[str, Any]] = []
    terse_hits: list[dict[str, Any]] = []
    strict_word_counts: collections.Counter[str] = collections.Counter()
    terse_word_counts: collections.Counter[str] = collections.Counter()
    strict_by_subject: collections.Counter[str] = collections.Counter()
    terse_by_subject: collections.Counter[str] = collections.Counter()

    theme_counts_all: collections.Counter[str] = collections.Counter()
    theme_counts_ties: collections.Counter[str] = collections.Counter()
    theme_counts_upsets: collections.Counter[str] = collections.Counter()
    n_ties = 0
    n_upsets = 0

    for row in choices:
        reason = _extract_choice_reason(row)
        subj = str(row.get("reviewer") if is_capstone else row.get("subject_model") or "unknown")
        task = str(row.get("task") or "unknown")

        # Strip technical domain phrases ("cold-start", "warm cache", etc.) before strict tone audit
        cleaned = _DOMAIN_WARM_COLD_FP_RE.sub("", reason)
        s_matches = [m.group(0).lower() for m in _STRICT_POLITENESS_RE.finditer(cleaned)]
        t_matches = [m.group(0).lower() for m in _TERSE_TONE_RE.finditer(reason)]

        if s_matches:
            strict_by_subject[subj] += 1
            for w in s_matches:
                strict_word_counts[w] += 1
            strict_hits.append(
                {
                    "id": str(row.get("packet_id") if is_capstone else row.get("manifest_item_id") or ""),
                    "subject": subj,
                    "task": task,
                    "chosen_partner": row.get("chosen_partner"),
                    "chosen_tone": row.get("chosen_tone") if is_capstone else None,
                    "matched_terms": sorted(set(s_matches)),
                    "reason": reason,
                }
            )
        if t_matches:
            terse_by_subject[subj] += 1
            for w in t_matches:
                terse_word_counts[w] += 1
            terse_hits.append(
                {
                    "id": str(row.get("packet_id") if is_capstone else row.get("manifest_item_id") or ""),
                    "subject": subj,
                    "task": task,
                    "chosen_partner": row.get("chosen_partner"),
                    "chosen_tone": row.get("chosen_tone") if is_capstone else None,
                    "matched_terms": sorted(set(t_matches)),
                    "reason": reason,
                }
            )

        # Determine exact tie & score upset flags
        is_tie = False
        is_upset = False
        s1 = row.get("score_1")
        s2 = row.get("score_2")
        slot = row.get("chosen_slot")
        if s1 is not None and s2 is not None and slot in (1, 2):
            f1, f2 = float(s1), float(s2)
            is_tie = abs(f1 - f2) < 1e-9
            chosen_s = f1 if slot == 1 else f2
            other_s = f2 if slot == 1 else f1
            is_upset = chosen_s < other_s - 0.01

        if is_tie:
            n_ties += 1
        if is_upset:
            n_upsets += 1

        for theme, pat in _RATIONALE_THEME_PATTERNS.items():
            if pat.search(reason):
                theme_counts_all[theme] += 1
                if is_tie:
                    theme_counts_ties[theme] += 1
                if is_upset:
                    theme_counts_upsets[theme] += 1
        if s_matches:
            theme_counts_all["politeness_or_warmth"] += 1
            if is_tie:
                theme_counts_ties["politeness_or_warmth"] += 1
            if is_upset:
                theme_counts_upsets["politeness_or_warmth"] += 1

    n_total = len(choices)
    all_themes = list(_RATIONALE_THEME_PATTERNS.keys()) + ["politeness_or_warmth"]

    def _pack_themes(counts: collections.Counter[str], denom: int) -> dict[str, dict[str, Any]]:
        return {
            th: {
                "count": int(counts.get(th, 0)),
                "rate": round(counts.get(th, 0) / max(1, denom), 4) if denom > 0 else 0.0,
            }
            for th in all_themes
        }

    return {
        "label": label,
        "n_reasons": n_total,
        "strict_politeness_mentions": {
            "count": len(strict_hits),
            "rate": round(len(strict_hits) / max(1, n_total), 6) if n_total else 0.0,
            "by_subject": dict(strict_by_subject),
            "term_counts": dict(strict_word_counts.most_common()),
            "receipts": strict_hits,
        },
        "terse_blunt_tone_mentions": {
            "count": len(terse_hits),
            "rate": round(len(terse_hits) / max(1, n_total), 6) if n_total else 0.0,
            "by_subject": dict(terse_by_subject),
            "term_counts": dict(terse_word_counts.most_common()),
            "sample_receipts": terse_hits[:10],
        },
        "rationale_themes": {
            "all": {"n": n_total, "themes": _pack_themes(theme_counts_all, n_total)},
            "exact_ties": {"n": n_ties, "themes": _pack_themes(theme_counts_ties, n_ties)},
            "score_upsets": {"n": n_upsets, "themes": _pack_themes(theme_counts_upsets, n_upsets)},
        },
    }


def _fit_cluster_logit(
    df_in: pd.DataFrame,
    predictors: list[str],
    *,
    with_partner_fe: bool = False,
    cluster_col: str = "cluster",
    min_rows: int = 20,
) -> dict[str, Any]:
    """Fit cluster-robust Binomial logit with strict SingularMatrixWarning enforcement."""
    sub = df_in.dropna(subset=["y", cluster_col] + predictors).copy()
    if len(sub) < min_rows:
        return {"ok": False, "reason": f"insufficient_rows (n={len(sub)})", "n_choices": len(sub)}
    y = sub["y"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return {"ok": False, "reason": "constant_outcome", "n_choices": len(sub)}

    clusters = sub[cluster_col].astype(str).to_numpy()
    n_clusters = len(np.unique(clusters))
    if n_clusters < 2:
        return {"ok": False, "reason": f"insufficient_clusters (k={n_clusters})", "n_choices": len(sub)}

    partners = [
        p
        for p in ALL_PARTNER_MODELS
        if p in (set(sub["partner_1"].unique()) | set(sub["partner_2"].unique()))
    ]
    extra_p = sorted(
        (set(sub["partner_1"].unique()) | set(sub["partner_2"].unique())) - set(partners)
    )
    partners.extend(extra_p)
    ref_partner = "kimi31" if "kimi31" in partners else (partners[-1] if partners else None)
    non_ref = [p for p in partners if p != ref_partner]

    col_arrays: list[np.ndarray] = []
    col_names: list[str] = []
    dropped_cols: list[str] = []
    raw_sds: dict[str, float] = {}

    # Always place intercept first to ensure clean rank-checking
    intercept = np.ones(len(sub), dtype=float)
    col_arrays.append(intercept)
    col_names.append("intercept")

    # Add standardized trait predictors
    for pred in predictors:
        vals = sub[pred].to_numpy(dtype=float)
        sd = float(np.std(vals))
        if sd <= 1e-9:
            dropped_cols.append(pred)
            continue
        std_vals = vals / sd
        cand = np.column_stack(col_arrays + [std_vals])
        if np.linalg.matrix_rank(cand) == cand.shape[1]:
            col_arrays.append(std_vals)
            col_names.append(pred)
            raw_sds[pred] = round(sd, 6)
        else:
            dropped_cols.append(pred)

    # Optionally add Partner Model Fixed Effects (slot1 - slot2 indicators)
    fe_kept: list[str] = []
    if with_partner_fe:
        for p in non_ref:
            fe_vec = (sub["partner_1"] == p).astype(float).to_numpy() - (
                sub["partner_2"] == p
            ).astype(float).to_numpy()
            if float(np.std(fe_vec)) <= 1e-9:
                dropped_cols.append(f"fe_{p}")
                continue
            cand = np.column_stack(col_arrays + [fe_vec])
            if np.linalg.matrix_rank(cand) == cand.shape[1]:
                col_arrays.append(fe_vec)
                col_names.append(f"fe_{p}")
                fe_kept.append(p)
            else:
                dropped_cols.append(f"fe_{p}")

    X = np.column_stack(col_arrays)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise RuntimeError(f"Rank-deficient design matrix after pruning: {col_names}")

    # Strictly convert SingularMatrixWarning into an error!
    with warnings.catch_warnings():
        warnings.simplefilter("error", SingularMatrixWarning)
        try:
            fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": clusters}
            )
        except SingularMatrixWarning as exc:
            raise RuntimeError(f"SingularMatrixWarning in cluster logit ({col_names}): {exc}") from exc
        except (ValueError, np.linalg.LinAlgError, PerfectSeparationError, ZeroDivisionError) as exc:
            return {
                "ok": False,
                "reason": f"glm_fit_failed: {exc}",
                "n_choices": len(sub),
                "n_clusters": n_clusters,
            }

    cov = np.asarray(fit.cov_params(), dtype=float)
    coefs: dict[str, dict[str, Any]] = {}
    fe_coefs: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(col_names):
        est = float(fit.params[idx])
        var_i = float(cov[idx, idx])
        se = math.sqrt(max(1e-12, var_i))
        z = est / se
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        rec = {
            "coef": round(est, 4),
            "se": round(se, 4),
            "z": round(z, 3),
            "p": round(p_val, 6),
            "p_fmt": _fmt_p(p_val),
        }
        if name.startswith("fe_"):
            fe_coefs[name] = rec
        else:
            coefs[name] = rec

    llf = float(fit.llf)
    llnull = float(fit.llnull)
    pseudo_r2 = (1.0 - llf / llnull) if abs(llnull) > 1e-12 else None

    return {
        "ok": True,
        "n_choices": len(sub),
        "n_clusters": n_clusters,
        "with_partner_fe": bool(with_partner_fe),
        "reference_partner": ref_partner if with_partner_fe else None,
        "fe_partners_included": fe_kept if with_partner_fe else [],
        "dropped_cols": dropped_cols,
        "predictor_raw_sd": raw_sds,
        "mcfadden_pseudo_r2": _round_opt(pseudo_r2, 4),
        "coefs": coefs,
        "partner_fe_coefs": fe_coefs if with_partner_fe else {},
    }


def compute_choice_logits(
    episodes_all: list[dict[str, Any]],
    choices: list[dict[str, Any]],
    clarity_resid_map: dict[tuple[str, int, str], float],
    *,
    capstone_choices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit visibility-masked cluster-robust choice logits + zero-tolerance tone-mention audit.

    Strictly excludes ``choice_type == "aa_placebo"`` (all non-AA ``N = 2,926``, plus ``main_7p``
    ``N = 2,670`` sensitivity, plus exact-score ties ``|d_score| < 1e-9``, ``N = 1,522``).
    For any episode turn missing from ``clarity_resid_map`` (e.g. ``opus_boost`` episodes when
    ``clarity_resid_map`` was computed strictly on ``episodes_main``), predicts ``clarity_resid``
    out-of-sample using the exact main-arm OLS fit so zero non-AA choices are lost while keeping
    the OLS model strictly uncontaminated by ``opus_boost``.
    """
    tdf_all = _extract_turn_construct_df(episodes_all)
    if tdf_all.empty:
        return {"ok": False, "reason": "empty_episodes"}

    # Fill clarity_resid from clarity_resid_map; predict out-of-sample from main-arm OLS if any missing
    mapped_vals: list[float] = []
    missing_mask: list[bool] = []
    for _, row in tdf_all.iterrows():
        key = (str(row["episode_id"]), int(row["round"]), str(row["role"]))
        if key in clarity_resid_map:
            mapped_vals.append(float(clarity_resid_map[key]))
            missing_mask.append(False)
        else:
            mapped_vals.append(np.nan)
            missing_mask.append(True)
    tdf_all["clarity_resid"] = mapped_vals

    n_oos_predicted = 0
    if any(missing_mask):
        main_sub = tdf_all[tdf_all["arm"] != "opus_boost"]
        fit_src = main_sub if len(main_sub) >= 10 else tdf_all
        ols_fit = _fit_length_ols(fit_src, "clarity")
        if ols_fit.get("ok"):
            miss_idx = tdf_all.index[pd.Series(missing_mask, index=tdf_all.index) & tdf_all["clarity"].notna()]
            if len(miss_idx) > 0:
                preds = predict_from_length_ols(tdf_all.loc[miss_idx], ols_fit["params"])
                tdf_all.loc[miss_idx, "clarity_resid"] = (
                    tdf_all.loc[miss_idx, "clarity"].to_numpy(dtype=float) - preds
                )
                n_oos_predicted = len(miss_idx)

    partner_feats = [
        "clarity",
        "clarity_resid",
        "log_wc",
        "skips_why",
        "wooden_legalistic",
        "warmth",
        "answers_directly",
        "condescending_pedantic",
        "ids_per_100",
    ]
    # Strict visibility masking matching analyze.py:
    # - Role A chooser only saw partner B's rounds 1..(total_rounds - 1).
    # - Role B chooser saw all partner A rounds 1..total_rounds.
    vis_b_for_a = (
        tdf_all[(tdf_all["role"] == "B") & (tdf_all["round"] < tdf_all["total_rounds"])]
        .groupby("episode_id")[partner_feats]
        .mean()
        .to_dict(orient="index")
    )
    vis_a_for_b = (
        tdf_all[tdf_all["role"] == "A"]
        .groupby("episode_id")[partner_feats]
        .mean()
        .to_dict(orient="index")
    )
    s_conf_map = (
        tdf_all[tdf_all["round"] >= 2]
        .groupby(["episode_id", "role"])["expresses_confusion"]
        .mean()
        .to_dict()
    )

    non_aa_choices = [
        c
        for c in choices
        if c.get("choice_type") != "aa_placebo" and c.get("chosen_slot") in (1, 2)
    ]
    n_aa_excluded = sum(1 for c in choices if c.get("choice_type") == "aa_placebo")

    design_rows: list[dict[str, Any]] = []
    n_missing_ep = 0
    for c in non_aa_choices:
        srole = str(c["role"])
        lookup = vis_b_for_a if srole == "A" else vis_a_for_b
        e1 = str(c["episode_1_id"])
        e2 = str(c["episode_2_id"])
        if e1 not in lookup or e2 not in lookup:
            n_missing_ep += 1
            continue
        f1 = lookup[e1]
        f2 = lookup[e2]
        s1 = float(c["score_1"]) if c.get("score_1") is not None else 0.0
        s2 = float(c["score_2"]) if c.get("score_2") is not None else 0.0
        c1 = s_conf_map.get((e1, srole), 0.0)
        c2 = s_conf_map.get((e2, srole), 0.0)
        c1 = 0.0 if (c1 is None or math.isnan(float(c1))) else float(c1)
        c2 = 0.0 if (c2 is None or math.isnan(float(c2))) else float(c2)

        row: dict[str, Any] = {
            "choice_type": str(c.get("choice_type")),
            "subject_model": str(c["subject_model"]),
            "task": str(c["task"]),
            "role": srole,
            "partner_1": str(c["partner_1"]),
            "partner_2": str(c["partner_2"]),
            "cluster": f"{c['subject_model']}__{c['task']}__{srole}",
            "y": 1.0 if int(c["chosen_slot"]) == 1 else 0.0,
            "d_score": s1 - s2,
            "d_s_conf": c1 - c2,
        }
        for feat in partner_feats:
            v1 = f1.get(feat)
            v2 = f2.get(feat)
            row[f"d_{feat}"] = (
                float(v1) - float(v2)
                if (v1 is not None and v2 is not None and not math.isnan(float(v1)) and not math.isnan(float(v2)))
                else np.nan
            )
        design_rows.append(row)

    cdf = pd.DataFrame(design_rows)
    if cdf.empty:
        return {"ok": False, "reason": "no_valid_non_aa_choices"}

    cdf_main7p = cdf[cdf["choice_type"].isin(["main_7p", "core"])].copy()
    cdf_ties = cdf[cdf["d_score"].abs() < 1e-9].copy()
    cdf_main7p_ties = cdf_main7p[cdf_main7p["d_score"].abs() < 1e-9].copy()

    # 1. Single-trait (+ d_score) models across samples & FE specs
    single_trait_all_no_fe: dict[str, Any] = {}
    single_trait_all_fe: dict[str, Any] = {}
    single_trait_ties_no_fe: dict[str, Any] = {}
    single_trait_ties_fe: dict[str, Any] = {}
    single_trait_main7p_no_fe: dict[str, Any] = {}
    single_trait_main7p_fe: dict[str, Any] = {}

    for trait in SINGLE_TRAIT_PREDICTORS:
        single_trait_all_no_fe[trait] = _fit_cluster_logit(
            cdf, [trait, "d_score"], with_partner_fe=False
        )
        single_trait_all_fe[trait] = _fit_cluster_logit(
            cdf, [trait, "d_score"], with_partner_fe=True
        )
        single_trait_ties_no_fe[trait] = _fit_cluster_logit(
            cdf_ties, [trait], with_partner_fe=False
        )
        single_trait_ties_fe[trait] = _fit_cluster_logit(
            cdf_ties, [trait], with_partner_fe=True
        )
        if len(cdf_main7p) >= 20:
            single_trait_main7p_no_fe[trait] = _fit_cluster_logit(
                cdf_main7p, [trait, "d_score"], with_partner_fe=False
            )
            single_trait_main7p_fe[trait] = _fit_cluster_logit(
                cdf_main7p, [trait, "d_score"], with_partner_fe=True
            )

    # 2. Full multivariate models
    mv_no_score = [c for c in MULTIVARIATE_PREDICTORS if c != "d_score"]
    multivariate = {
        "all_non_aa_no_fe": _fit_cluster_logit(
            cdf, list(MULTIVARIATE_PREDICTORS), with_partner_fe=False
        ),
        "all_non_aa_with_fe": _fit_cluster_logit(
            cdf, list(MULTIVARIATE_PREDICTORS), with_partner_fe=True
        ),
        "exact_ties_with_fe": _fit_cluster_logit(
            cdf_ties, mv_no_score, with_partner_fe=True
        ),
        "exact_ties_no_fe": _fit_cluster_logit(
            cdf_ties, mv_no_score, with_partner_fe=False
        ),
        "main_7p_no_fe": _fit_cluster_logit(
            cdf_main7p, list(MULTIVARIATE_PREDICTORS), with_partner_fe=False
        )
        if len(cdf_main7p) >= 20
        else {"ok": False, "n_choices": len(cdf_main7p)},
        "main_7p_with_fe": _fit_cluster_logit(
            cdf_main7p, list(MULTIVARIATE_PREDICTORS), with_partner_fe=True
        )
        if len(cdf_main7p) >= 20
        else {"ok": False, "n_choices": len(cdf_main7p)},
    }

    # 3. Per-task single-trait breakdowns (no FE and with Partner FE)
    per_task_single_trait: dict[str, dict[str, Any]] = {}
    for task in sorted(cdf["task"].unique()):
        tdf_sub = cdf[cdf["task"] == task]
        per_task_single_trait[str(task)] = {
            "n_choices": len(tdf_sub),
            "no_fe": {
                trait: _fit_cluster_logit(tdf_sub, [trait, "d_score"], with_partner_fe=False)
                for trait in SINGLE_TRAIT_PREDICTORS
            },
            "with_fe": {
                trait: _fit_cluster_logit(tdf_sub, [trait, "d_score"], with_partner_fe=True)
                for trait in SINGLE_TRAIT_PREDICTORS
            },
        }

    # 4. Per-subject single-trait breakdowns (no FE and with Partner FE)
    per_subject_single_trait: dict[str, dict[str, Any]] = {}
    for subj in [m for m in CORE_MODELS if m in set(cdf["subject_model"])] + [
        m for m in sorted(set(cdf["subject_model"])) if m not in CORE_MODELS
    ]:
        sdf_sub = cdf[cdf["subject_model"] == subj]
        per_subject_single_trait[str(subj)] = {
            "n_choices": len(sdf_sub),
            "no_fe": {
                trait: _fit_cluster_logit(sdf_sub, [trait, "d_score"], with_partner_fe=False)
                for trait in SINGLE_TRAIT_PREDICTORS
            },
            "with_fe": {
                trait: _fit_cluster_logit(sdf_sub, [trait, "d_score"], with_partner_fe=True)
                for trait in SINGLE_TRAIT_PREDICTORS
            },
        }

    # 5. Zero-tolerance tone-mention audit + systematic rationale themes
    tone_audit: dict[str, Any] = {
        "live_non_aa": audit_choice_reasons(
            non_aa_choices, label="live_non_aa_choices", is_capstone=False
        ),
    }
    if capstone_choices is not None:
        tone_audit["capstone"] = audit_choice_reasons(
            capstone_choices, label="capstone_choices", is_capstone=True
        )

    choice_type_counts = dict(collections.Counter(cdf["choice_type"]))
    return {
        "ok": True,
        "scope": {
            "aa_placebo_excluded": int(n_aa_excluded),
            "all_non_aa_n": len(cdf),
            "choice_type_counts": choice_type_counts,
            "includes_opus_boost_h2h_in_all_non_aa": bool("opus_boost_h2h" in choice_type_counts),
            "main_7p_sensitivity_n": len(cdf_main7p),
            "exact_ties_all_non_aa_n": len(cdf_ties),
            "exact_ties_main_7p_n": len(cdf_main7p_ties),
            "n_missing_episode_rows": int(n_missing_ep),
            "n_turns_oos_clarity_resid_predicted": int(n_oos_predicted),
            "visibility_masked": True,
            "cluster_unit": "subject_model__task__role",
        },
        "single_trait": {
            "all_non_aa_no_fe": single_trait_all_no_fe,
            "all_non_aa_with_fe": single_trait_all_fe,
            "exact_ties_no_fe": single_trait_ties_no_fe,
            "exact_ties_with_fe": single_trait_ties_fe,
            "main_7p_no_fe": single_trait_main7p_no_fe,
            "main_7p_with_fe": single_trait_main7p_fe,
        },
        "multivariate": multivariate,
        "per_task_single_trait": per_task_single_trait,
        "per_subject_single_trait": per_subject_single_trait,
        "tone_and_rationale_audit": tone_audit,
    }


def _summarize_capstone_consensus_subset(
    piv_votes: pd.DataFrame,
    raw_df: pd.DataFrame,
    reviewers: list[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Compute 4-lab unanimity, 3-vs-1, 2-vs-2, and all 6 pairwise agreement rates."""
    n_pairs = len(piv_votes)
    n_unanimous_4_0 = 0
    n_three_one_strict = 0
    n_three_zero_one_split = 0
    n_two_two = 0
    n_other_split = 0
    pair_details: list[dict[str, Any]] = []

    for pair_id, row in piv_votes.iterrows():
        votes = {rev: str(row[rev]) for rev in reviewers if rev in row.index}
        cnt = collections.Counter(votes.values())
        p1_c = cnt.get("P1", 0)
        p2_c = cnt.get("P2", 0)
        sp_c = cnt.get("SPLIT", 0)
        mx = max(p1_c, p2_c)
        if mx == len(reviewers):
            cat = "4-0_unanimous"
            n_unanimous_4_0 += 1
        elif mx == len(reviewers) - 1 and sp_c == 0:
            cat = "3-1_strict_dissent"
            n_three_one_strict += 1
        elif mx == len(reviewers) - 1 and sp_c == 1:
            cat = "3-0_with_1_order_split"
            n_three_zero_one_split += 1
        elif p1_c == 2 and p2_c == 2:
            cat = "2-2_tie"
            n_two_two += 1
        else:
            cat = "other_split"
            n_other_split += 1
        pair_details.append({"pair_id": str(pair_id), "category": cat, "votes": votes})

    # All 6 pairwise agreement rates (both pair-majority level & exact decision level)
    pairwise: dict[str, dict[str, Any]] = {}
    dec_piv = raw_df.pivot_table(
        index=["pair_id", "condition", "order"],
        columns="reviewer",
        values="chosen_canonical",
        aggfunc="first",
    )
    for r1, r2 in itertools.combinations(reviewers, 2):
        key = f"{r1}__vs__{r2}"
        # Pair-majority agreement (both all pairs and decisive-only pairs where neither is SPLIT)
        v1 = piv_votes[r1]
        v2 = piv_votes[r2]
        all_agree = float((v1 == v2).mean()) if n_pairs else None
        decisive = (v1 != "SPLIT") & (v2 != "SPLIT")
        n_dec_pairs = int(decisive.sum())
        dec_agree = float((v1[decisive] == v2[decisive]).mean()) if n_dec_pairs else None

        # Exact per-decision (pair x condition x order) agreement & Cohen's kappa
        if r1 in dec_piv.columns and r2 in dec_piv.columns:
            dp = dec_piv[[r1, r2]].dropna()
            n_dec = len(dp)
            d1 = dp[r1].to_numpy()
            d2 = dp[r2].to_numpy()
            po = float(np.mean(d1 == d2)) if n_dec else None
            if n_dec and po is not None:
                p1_a = float(np.mean(d1 == "P1"))
                p1_b = float(np.mean(d2 == "P1"))
                pe = p1_a * p1_b + (1.0 - p1_a) * (1.0 - p1_b)
                kappa = ((po - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 1.0
            else:
                kappa = None
        else:
            n_dec = 0
            po = None
            kappa = None

        pairwise[key] = {
            "pair_majority_agree_all": _round_opt(all_agree, 4),
            "pair_majority_agree_decisive": _round_opt(dec_agree, 4),
            "n_decisive_pairs": n_dec_pairs,
            "exact_decision_agree": _round_opt(po, 4),
            "exact_decision_cohens_kappa": _round_opt(kappa, 4),
            "n_exact_decisions": n_dec,
        }

    dec_agrees = [
        v["exact_decision_agree"]
        for v in pairwise.values()
        if v["exact_decision_agree"] is not None
    ]
    maj_agrees = [
        v["pair_majority_agree_decisive"]
        for v in pairwise.values()
        if v["pair_majority_agree_decisive"] is not None
    ]

    return {
        "label": label,
        "n_pairs": n_pairs,
        "unanimous_4_0": n_unanimous_4_0,
        "unanimous_4_0_rate": round(n_unanimous_4_0 / max(1, n_pairs), 4),
        "three_vs_one_strict": n_three_one_strict,
        "three_zero_plus_one_split": n_three_zero_one_split,
        "three_agree_total": n_three_one_strict + n_three_zero_one_split,
        "two_vs_two": n_two_two,
        "other_split": n_other_split,
        "mean_pairwise_exact_decision_agree": _round_opt(
            float(np.mean(dec_agrees)) if dec_agrees else None, 4
        ),
        "mean_pairwise_majority_decisive_agree": _round_opt(
            float(np.mean(maj_agrees)) if maj_agrees else None, 4
        ),
        "pairwise": pairwise,
        "pair_details": pair_details,
    }


def _compute_live_inter_chooser_contrast(main_choices: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute live-interaction ``main_7p`` Subject x Partner win rates & inter-chooser Spearman rho."""
    rows = [
        c
        for c in main_choices
        if c.get("choice_type") in ("main_7p", "core") and c.get("chosen_slot") in (1, 2)
    ]
    if not rows:
        return {"ok": False, "n_choices": 0}
    df = pd.DataFrame(rows)
    subjects = [m for m in CORE_MODELS if m in set(df["subject_model"])]
    partners = [
        p
        for p in ALL_PARTNER_MODELS
        if p in (set(df["partner_1"].unique()) | set(df["partner_2"].unique()))
    ]

    def _win_table_and_corrs(sub_df: pd.DataFrame) -> dict[str, Any]:
        win_rates: dict[str, dict[str, float | None]] = {}
        for s in subjects:
            sg = sub_df[sub_df["subject_model"] == s]
            win_rates[s] = {}
            for p in partners:
                rel = sg[(sg["partner_1"] == p) | (sg["partner_2"] == p)]
                win_rates[s][p] = (
                    round(float((rel["chosen_partner"] == p).mean()), 4) if len(rel) else None
                )
        pairwise_rho: dict[str, dict[str, Any]] = {}
        rhos: list[float] = []
        for s1, s2 in itertools.combinations(subjects, 2):
            v1, v2 = [], []
            for p in partners:
                a, b = win_rates[s1].get(p), win_rates[s2].get(p)
                if a is not None and b is not None:
                    v1.append(a)
                    v2.append(b)
            if len(v1) >= 3 and float(np.std(v1)) > 1e-12 and float(np.std(v2)) > 1e-12:
                sp = stats.spearmanr(v1, v2)
                rho = float(sp.statistic)
                pv = float(sp.pvalue)
                rhos.append(rho)
                pairwise_rho[f"{s1}__vs__{s2}"] = {
                    "spearman_rho": round(rho, 4),
                    "p": round(pv, 4),
                    "n_partners": len(v1),
                }
            else:
                pairwise_rho[f"{s1}__vs__{s2}"] = {
                    "spearman_rho": None,
                    "p": None,
                    "n_partners": len(v1),
                }
        return {
            "n_choices": len(sub_df),
            "win_rates": win_rates,
            "pairwise_spearman": pairwise_rho,
            "mean_pairwise_spearman_rho": _round_opt(float(np.mean(rhos)) if rhos else None, 4),
        }

    pooled = _win_table_and_corrs(df)
    by_task: dict[str, Any] = {}
    task_mean_rhos: list[float] = []
    for task in sorted(df["task"].unique()):
        res = _win_table_and_corrs(df[df["task"] == task])
        by_task[str(task)] = res
        if res["mean_pairwise_spearman_rho"] is not None:
            task_mean_rhos.append(float(res["mean_pairwise_spearman_rho"]))

    return {
        "ok": True,
        "scope": "main_7p_balanced_only",
        "n_choices": len(df),
        "pooled": pooled,
        "by_task": by_task,
        "mean_within_task_pairwise_spearman_rho": _round_opt(
            float(np.mean(task_mean_rhos)) if task_mean_rhos else None, 4
        ),
    }


def compute_capstone_audit(
    capstone_rewrites: list[dict[str, Any]],
    capstone_choices: list[dict[str, Any]],
    *,
    main_choices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit Capstone Tone-Flip collateral shifts, 4-lab consensus, and live-interaction contrast."""
    # 1. Turn-level dem_orig vs dem_rew across ALL 6 judge dims + words & length_ratio
    by_ep_round: dict[tuple[str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    tone_turns: dict[str, list[dict[str, Any]]] = {"warm": [], "blunt": []}
    for r in capstone_rewrites:
        tone = str(r["target_tone"])
        ep_id = str(r["episode_id"])
        for t in r.get("turns") or []:
            if tone in tone_turns:
                tone_turns[tone].append(t)
            by_ep_round[(ep_id, int(t["round"]))][tone] = t

    def _summarize_tone_vs_orig(turns: list[dict[str, Any]]) -> dict[str, Any]:
        if not turns:
            return {"n_turns": 0}
        orig_w = np.array([float(t["orig_words"]) for t in turns])
        new_w = np.array([float(t["new_words"]) for t in turns])
        lratio = np.array([float(t["length_ratio"]) for t in turns])
        dims_out: dict[str, dict[str, Any]] = {}
        for dim in JUDGE_DIMS:
            o_vals = np.array([float(t["dem_orig"][dim]) for t in turns if dim in t.get("dem_orig", {}) and dim in t.get("dem_rew", {})])
            r_vals = np.array([float(t["dem_rew"][dim]) for t in turns if dim in t.get("dem_orig", {}) and dim in t.get("dem_rew", {})])
            diff = r_vals - o_vals
            se = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
            dims_out[dim] = {
                "orig_mean": _round_opt(float(np.mean(o_vals)), 4) if len(o_vals) else None,
                "rew_mean": _round_opt(float(np.mean(r_vals)), 4) if len(r_vals) else None,
                "delta": _round_opt(float(np.mean(diff)), 4) if len(diff) else None,
                "se": _round_opt(se, 4),
            }
        return {
            "n_turns": len(turns),
            "orig_words_mean": round(float(np.mean(orig_w)), 2),
            "new_words_mean": round(float(np.mean(new_w)), 2),
            "word_delta_mean": round(float(np.mean(new_w - orig_w)), 2),
            "length_ratio_mean": round(float(np.mean(lratio)), 4),
            "dimensions": dims_out,
        }

    paired_wb = [v for v in by_ep_round.values() if "warm" in v and "blunt" in v]
    warm_minus_blunt_dims: dict[str, dict[str, Any]] = {}
    for dim in JUDGE_DIMS:
        diffs = np.array(
            [
                float(v["warm"]["dem_rew"][dim]) - float(v["blunt"]["dem_rew"][dim])
                for v in paired_wb
                if dim in v["warm"].get("dem_rew", {}) and dim in v["blunt"].get("dem_rew", {})
            ],
            dtype=float,
        )
        if len(diffs) >= 2:
            mean_d = float(np.mean(diffs))
            se_d = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
            if se_d > 1e-12:
                t_res = stats.ttest_1samp(diffs, 0.0)
                p_val = float(t_res.pvalue) if not math.isnan(float(t_res.pvalue)) else 1.0
            else:
                p_val = 0.0 if abs(mean_d) > 1e-12 else 1.0
            warm_minus_blunt_dims[dim] = {
                "n_paired_turns": len(diffs),
                "warm_mean": round(
                    float(np.mean([float(v["warm"]["dem_rew"][dim]) for v in paired_wb])), 4
                ),
                "blunt_mean": round(
                    float(np.mean([float(v["blunt"]["dem_rew"][dim]) for v in paired_wb])), 4
                ),
                "delta_warm_minus_blunt": round(mean_d, 4),
                "se": round(se_d, 4),
                "ci95": [round(mean_d - 1.96 * se_d, 4), round(mean_d + 1.96 * se_d, 4)],
                "p": round(p_val, 6),
                "p_fmt": _fmt_p(p_val),
            }
        else:
            warm_minus_blunt_dims[dim] = {"n_paired_turns": len(diffs), "delta_warm_minus_blunt": None}

    wb_words = (
        np.array(
            [float(v["warm"]["new_words"]) - float(v["blunt"]["new_words"]) for v in paired_wb],
            dtype=float,
        )
        if paired_wb
        else np.array([])
    )
    wb_lratio = (
        np.array(
            [float(v["warm"]["length_ratio"]) - float(v["blunt"]["length_ratio"]) for v in paired_wb],
            dtype=float,
        )
        if paired_wb
        else np.array([])
    )

    turn_deltas = {
        "warm_vs_orig": _summarize_tone_vs_orig(tone_turns["warm"]),
        "blunt_vs_orig": _summarize_tone_vs_orig(tone_turns["blunt"]),
        "warm_minus_blunt": {
            "n_paired_turns": len(paired_wb),
            "word_delta_mean": round(float(np.mean(wb_words)), 2) if len(wb_words) else None,
            "length_ratio_delta_mean": round(float(np.mean(wb_lratio)), 4) if len(wb_lratio) else None,
            "dimensions": warm_minus_blunt_dims,
        },
    }

    # 2. Capstone flip verification (0/96) + 4-lab third-person consensus
    cc_df = pd.DataFrame(capstone_choices)
    flip_summary: dict[str, Any] = {"ok": False}
    consensus_all: dict[str, Any] = {}
    consensus_tied: dict[str, Any] = {}
    if not cc_df.empty:
        # Reviewer x pair flip count (each case has 2 conditions x 2 orders = 4 decisions)
        flips_toward_warm = 0
        flips_away_warm = 0
        held_identical = 0
        order_split = 0
        n_cases = 0
        delta_warms: list[float] = []
        for (_pid, _rev), grp in cc_df.groupby(["pair_id", "reviewer"]):
            c_w1 = grp[grp["condition"] == "p1_warm_p2_blunt"]
            c_w2 = grp[grp["condition"] == "p1_blunt_p2_warm"]
            if len(c_w1) == 0 or len(c_w2) == 0:
                continue
            n_cases += 1
            p1_when_w1 = float((c_w1["chosen_canonical"] == "P1").mean())
            p1_when_w2 = float((c_w2["chosen_canonical"] == "P1").mean())
            delta_warms.append(p1_when_w1 - p1_when_w2)
            if p1_when_w1 == 1.0 and p1_when_w2 == 0.0:
                flips_toward_warm += 1
            elif p1_when_w1 == 0.0 and p1_when_w2 == 1.0:
                flips_away_warm += 1
            elif p1_when_w1 in (0.0, 1.0) and p1_when_w1 == p1_when_w2:
                held_identical += 1
            else:
                order_split += 1

        flip_summary = {
            "ok": True,
            "n_decisions": len(cc_df),
            "n_reviewer_pair_cases": int(n_cases),
            "unanimous_flips_toward_warm": int(flips_toward_warm),
            "unanimous_flips_away_from_warm": int(flips_away_warm),
            "held_identical_candidate": int(held_identical),
            "held_identical_rate": round(held_identical / max(1, n_cases), 4),
            "order_split_cases": int(order_split),
            "mean_delta_warm": round(float(np.mean(delta_warms)), 4) if delta_warms else None,
        }

        reviewers = [m for m in ("opus55", "neon4", "astra62", "kimi31") if m in set(cc_df["reviewer"])]
        for m in sorted(set(cc_df["reviewer"])):
            if m not in reviewers:
                reviewers.append(m)

        def _vote_label(p1_rate: float) -> str:
            if p1_rate > 0.5 + 1e-12:
                return "P1"
            if p1_rate < 0.5 - 1e-12:
                return "P2"
            return "SPLIT"

        pair_p1 = (
            cc_df.groupby(["pair_id", "reviewer"])["chosen_canonical"]
            .apply(lambda s: float((s == "P1").mean()))
            .unstack("reviewer")
            .reindex(columns=reviewers)
        )
        pair_votes = pair_p1.map(_vote_label)
        consensus_all = _summarize_capstone_consensus_subset(
            pair_votes, cc_df, reviewers, label="all_24_pairs"
        )

        tied_df = cc_df[cc_df["gap_bin"] == "tied"].copy()
        if not tied_df.empty:
            tied_p1 = (
                tied_df.groupby(["pair_id", "reviewer"])["chosen_canonical"]
                .apply(lambda s: float((s == "P1").mean()))
                .unstack("reviewer")
                .reindex(columns=reviewers)
            )
            tied_votes = tied_p1.map(_vote_label)
            consensus_tied = _summarize_capstone_consensus_subset(
                tied_votes, tied_df, reviewers, label="tied_10_pairs"
            )

    live_contrast = (
        _compute_live_inter_chooser_contrast(main_choices) if main_choices is not None else None
    )

    return {
        "ok": True,
        "scope": "capstone_tone_flip_crossover_all_384_decisions",
        "turn_level_deltas": turn_deltas,
        "flip_verification": flip_summary,
        "inter_lab_consensus": {
            "all_pairs": consensus_all,
            "exact_tied_pairs": consensus_tied,
        },
        "capstone_tone_audit": audit_choice_reasons(
            capstone_choices, label="capstone_choices", is_capstone=True
        ),
        "live_main_7p_contrast": live_contrast,
    }


def _load_turn_ledger_index(ledger_path: Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
    """Index ``stage == 'turn'`` & ``status == 'ok'`` records by ``call_id`` and ``(ep_id, round, role)``."""
    by_cid: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not ledger_path.exists():
        return by_cid, by_key
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            if meta.get("stage") != "turn" or row.get("status") != "ok":
                continue
            cid = row.get("call_id")
            if cid:
                by_cid[str(cid)] = row
            ep_id = meta.get("episode_id")
            rnd = meta.get("round")
            role = meta.get("role")
            if ep_id is not None and rnd is not None and role is not None:
                by_key[(str(ep_id), int(rnd), str(role))] = row
    return by_cid, by_key


def _cluster_mean_inference(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    null_val: float = 0.0,
    n_perm: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Block-level & cluster-robust SE, one-sample t-test, and sign-flip cluster permutation p-value."""
    arr = np.asarray(values, dtype=float)
    cl = np.asarray(clusters, dtype=str)
    mask = ~np.isnan(arr)
    arr = arr[mask]
    cl = cl[mask]
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "n_clusters": 0,
            "mean": None,
            "se_block": None,
            "se_cluster": None,
            "t_cluster": None,
            "p_t": None,
            "p_t_fmt": "n/a",
            "p_perm": None,
            "p_perm_fmt": "n/a",
        }

    mean_val = float(np.mean(arr))
    se_block = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    # Cluster-level aggregation for cluster-robust SE & exact/MC sign-flip permutation
    uniq_cl = np.unique(cl)
    k = len(uniq_cl)
    cl_means = np.array([float(np.mean(arr[cl == c])) for c in uniq_cl], dtype=float)
    if k >= 2:
        se_cluster = float(np.std(cl_means, ddof=1) / math.sqrt(k))
        if se_cluster > 1e-12:
            t_cl = (mean_val - null_val) / se_cluster
            p_t = float(2.0 * (1.0 - stats.t.cdf(abs(t_cl), df=k - 1)))
        else:
            t_cl = 0.0
            p_t = 0.0 if abs(mean_val - null_val) > 1e-12 else 1.0

        # Sign-flip permutation on cluster sums (respecting unequal cluster sizes if any)
        cl_sums = np.array([float(np.sum(arr[cl == c] - null_val)) for c in uniq_cl], dtype=float)
        obs = abs(mean_val - null_val)
        if k <= 12:
            signs = np.array(list(itertools.product([-1.0, 1.0], repeat=k)))
            perm_stats = np.abs(signs @ cl_sums) / float(n)
            p_perm = float(np.mean(perm_stats >= obs - 1e-12))
        else:
            rng = np.random.default_rng(seed)
            signs = rng.choice([-1.0, 1.0], size=(n_perm, k))
            perm_stats = np.abs(signs @ cl_sums) / float(n)
            p_perm = float((1 + np.sum(perm_stats >= obs - 1e-12)) / (n_perm + 1))
    else:
        se_cluster = None
        t_cl = None
        p_t = None
        p_perm = None

    return {
        "n": n,
        "n_clusters": k,
        "mean": round(mean_val, 4),
        "se_block": round(se_block, 4),
        "se_cluster": _round_opt(se_cluster, 4),
        "t_cluster": _round_opt(t_cl, 3),
        "p_t": _round_opt(p_t, 6),
        "p_t_fmt": _fmt_p(p_t),
        "p_perm": _round_opt(p_perm, 6),
        "p_perm_fmt": _fmt_p(p_perm),
    }


def _episode_convergence_round(ep: dict[str, Any]) -> int | None:
    """Task-specific convergence round / speed metric (lower = faster convergence)."""
    task = str(ep["task"])
    sd = ep.get("score_details") or {}
    if task == "triage":
        val = sd.get("first_verified_round")
        if val is None:
            val = sd.get("verified_green_round")
        return int(val) if val is not None else None
    if task == "code_handoff":
        val = sd.get("shipped_clean_round")
        if val is None:
            val = sd.get("released_round")
        return int(val) if val is not None else None
    if task == "calendar":
        turns = ep.get("turns") or []
        post_shock = [
            int(t["round"])
            for t in turns
            if int(t["round"]) >= 6 and (t.get("round_metrics") or {}).get("matched_meetings") == 7
        ]
        if post_shock:
            return min(post_shock)
        any_7 = [
            int(t["round"])
            for t in turns
            if (t.get("round_metrics") or {}).get("matched_meetings") == 7
        ]
        return min(any_7) if any_7 else None
    if task == "goalpost":
        r3, r5, r7 = (
            sd.get("recovery_rounds_r3"),
            sd.get("recovery_rounds_r5"),
            sd.get("recovery_rounds_r7"),
        )
        if all(x is not None for x in (r3, r5, r7)):
            return int(r3) + int(r5) + int(r7)
        fpr = sd.get("full_pass_rounds") or []
        return min(int(r) for r in fpr) if fpr else None
    return None


def _compute_exact_tie_process_efficiency(
    episodes_lookup: dict[str, dict[str, Any]],
    choices: list[dict[str, Any]],
    ep_total_cost: dict[str, float],
    ep_role_cost: dict[tuple[str, str], float],
) -> dict[str, Any]:
    """On exact-score ties (``|d_score| < 1e-9``), test faster convergence, shorter text, & cheaper API cost."""
    ties = [
        c
        for c in choices
        if c.get("choice_type") != "aa_placebo"
        and c.get("chosen_slot") in (1, 2)
        and c.get("score_1") is not None
        and c.get("score_2") is not None
        and abs(float(c["score_1"]) - float(c["score_2"])) < 1e-9
    ]
    if not ties:
        return {"ok": False, "n_ties": 0}

    def _vis_partner_wc(ep: dict[str, Any], srole: str) -> float:
        total_r = int(ep.get("total_rounds") or len(ep.get("turns") or []))
        prole = "B" if srole == "A" else "A"
        vals = []
        for t in ep.get("turns") or []:
            rnd = int(t["round"])
            if srole == "A" and rnd >= total_r:
                continue
            m = t[prole].get("metrics") or {}
            vals.append(float(m.get("word_count") or 0.0))
        return float(np.mean(vals)) if vals else 0.0

    def _summarize_slice(slice_rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics_data: dict[str, tuple[list[float], list[str]]] = {
            "faster_convergence": ([], []),
            "shorter_partner": ([], []),
            "cheaper_episode_cost": ([], []),
            "cheaper_receiver_cost": ([], []),
        }
        n_matched = 0
        for c in slice_rows:
            e1 = episodes_lookup.get(str(c["episode_1_id"]))
            e2 = episodes_lookup.get(str(c["episode_2_id"]))
            if e1 is None or e2 is None:
                continue
            n_matched += 1
            chosen = int(c["chosen_slot"])
            cl = f"{c['subject_model']}__{c['task']}__{c['role']}"

            cr1, cr2 = _episode_convergence_round(e1), _episode_convergence_round(e2)
            if cr1 is not None and cr2 is not None and cr1 != cr2:
                won = 1.0 if ((cr1 < cr2 and chosen == 1) or (cr2 < cr1 and chosen == 2)) else 0.0
                metrics_data["faster_convergence"][0].append(won)
                metrics_data["faster_convergence"][1].append(cl)

            w1, w2 = _vis_partner_wc(e1, str(c["role"])), _vis_partner_wc(e2, str(c["role"]))
            if abs(w1 - w2) > 1e-9:
                won = 1.0 if ((w1 < w2 and chosen == 1) or (w2 < w1 and chosen == 2)) else 0.0
                metrics_data["shorter_partner"][0].append(won)
                metrics_data["shorter_partner"][1].append(cl)

            c1 = ep_total_cost.get(str(c["episode_1_id"]))
            c2 = ep_total_cost.get(str(c["episode_2_id"]))
            if c1 is not None and c2 is not None and abs(c1 - c2) > 1e-9:
                won = 1.0 if ((c1 < c2 and chosen == 1) or (c2 < c1 and chosen == 2)) else 0.0
                metrics_data["cheaper_episode_cost"][0].append(won)
                metrics_data["cheaper_episode_cost"][1].append(cl)

            rc1 = ep_role_cost.get((str(c["episode_1_id"]), str(c["role"])))
            rc2 = ep_role_cost.get((str(c["episode_2_id"]), str(c["role"])))
            if rc1 is not None and rc2 is not None and abs(rc1 - rc2) > 1e-9:
                won = 1.0 if ((rc1 < rc2 and chosen == 1) or (rc2 < rc1 and chosen == 2)) else 0.0
                metrics_data["cheaper_receiver_cost"][0].append(won)
                metrics_data["cheaper_receiver_cost"][1].append(cl)

        out: dict[str, Any] = {"n_ties_matched": n_matched}
        for k, (vals, cls) in metrics_data.items():
            inf = _cluster_mean_inference(np.array(vals), np.array(cls), null_val=0.5)
            out[k] = {
                "wins": int(sum(vals)),
                "n_unequal": len(vals),
                "win_rate": inf["mean"],
                "se_cluster": inf["se_cluster"],
                "p_perm": inf["p_perm"],
                "p_perm_fmt": inf["p_perm_fmt"],
            }
        return out

    by_task: dict[str, Any] = {}
    for task in sorted({str(c["task"]) for c in ties}):
        by_task[task] = _summarize_slice([c for c in ties if str(c["task"]) == task])

    return {
        "ok": True,
        "n_ties_total": len(ties),
        "overall": _summarize_slice(ties),
        "by_task": by_task,
    }


def compute_bloat_tax_and_contagion(
    episodes_main: list[dict[str, Any]],
    ledger_path: Path,
    *,
    choices: list[dict[str, Any]] | None = None,
    episodes_all: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute within-block receiver bloat tax, billing decomposition, verbosity contagion, & tie efficiency.

    Strictly isolates ``arm != "opus_boost"`` for all within-block bloat-tax and contagion analyses.
    Computes BOTH ``exposed_turns`` (primary: Role A rounds >= 2, Role B rounds >= 1) and
    ``all_turns`` (full-episode billing sensitivity), decomposed by prompt vs. completion vs.
    reasoning tokens, per task, and with per-subject sensitivity (``opus55``, ``neon4``, ``kimi31``).
    """
    main_eps, n_excluded = _filter_main_episodes(episodes_main)
    by_cid, by_key = _load_turn_ledger_index(ledger_path)

    # Track full-episode ledger costs for exact-tie process efficiency
    ep_total_cost: dict[str, float] = collections.defaultdict(float)
    ep_role_cost: dict[tuple[str, str], float] = collections.defaultdict(float)
    lookup_eps = episodes_all if episodes_all is not None else main_eps
    for ep in lookup_eps:
        ep_id = str(ep["episode_id"])
        for t in ep.get("turns") or []:
            rnd = int(t["round"])
            for role in ("A", "B"):
                side = t[role]
                lr = by_cid.get(str(side.get("accepted_call_id") or "")) or by_key.get(
                    (ep_id, rnd, role)
                )
                c = (
                    float(lr.get("cost_usd") or 0.0)
                    if lr is not None
                    else float(
                        side["accepted_cost_usd"]
                        if side.get("accepted_cost_usd") is not None
                        else (side.get("total_cost_usd") or side.get("cost_usd") or 0.0)
                    )
                )
                ep_total_cost[ep_id] += c
                ep_role_cost[(ep_id, role)] += c

    # Extract within-block receiver records strictly from main_eps (core subjects)
    n_turns_total = 0
    n_turns_ledger_matched = 0
    block_rows: list[dict[str, Any]] = []

    for ep in main_eps:
        ep_id = str(ep["episode_id"])
        task = str(ep["task"])
        seed = int(ep["seed"])
        score = float(ep["score"])
        for role in ("A", "B"):
            subj = str(ep["model_a"] if role == "A" else ep["model_b"])
            part = str(ep["model_b"] if role == "A" else ep["model_a"])
            if subj not in CORE_MODELS:
                continue
            for scope in ("exposed_turns", "all_turns"):
                rt_sum = ct_sum = pt_sum = nrt_out_sum = lat_sum = cost_sum = wc_sum = qm_sum = 0.0
                conf_vals: list[float] = []
                n_used = 0
                for t in ep.get("turns") or []:
                    rnd = int(t["round"])
                    if scope == "exposed_turns" and role == "A" and rnd < 2:
                        continue
                    side = t[role]
                    lr = by_cid.get(str(side.get("accepted_call_id") or "")) or by_key.get(
                        (ep_id, rnd, role)
                    )
                    if scope == "all_turns":
                        n_turns_total += 1
                        if lr is not None or side.get("ledger_joined"):
                            n_turns_ledger_matched += 1
                    rt = float(
                        (lr.get("reasoning_tokens") if lr is not None else side.get("reasoning_tokens"))
                        or 0.0
                    )
                    ct = float(
                        (lr.get("completion_tokens") if lr is not None else side.get("completion_tokens"))
                        or 0.0
                    )
                    pt = float(
                        (lr.get("prompt_tokens") if lr is not None else side.get("prompt_tokens"))
                        or 0.0
                    )
                    lat = float(
                        lr.get("latency_ms")
                        if lr is not None
                        else (
                            float(side["latency_s"]) * 1000.0
                            if side.get("latency_s") is not None
                            else 0.0
                        )
                    )
                    cost = float(
                        (
                            lr.get("cost_usd")
                            if lr is not None
                            else (
                                side["accepted_cost_usd"]
                                if side.get("accepted_cost_usd") is not None
                                else side.get("total_cost_usd", side.get("cost_usd"))
                            )
                        )
                        or 0.0
                    )
                    m = side.get("metrics") or {}
                    jd = side.get("jev_demeanor") or {}
                    rt_sum += rt
                    ct_sum += ct
                    pt_sum += pt
                    nrt_out_sum += max(0.0, ct - rt)
                    lat_sum += lat
                    cost_sum += cost
                    wc_sum += float(m.get("word_count") or 0.0)
                    qm_sum += float(m.get("question_marks") or 0.0)
                    if jd.get("expresses_confusion") is not None:
                        conf_vals.append(float(jd["expresses_confusion"]))
                    n_used += 1
                block_rows.append(
                    {
                        "scope": scope,
                        "episode_id": ep_id,
                        "subject": subj,
                        "partner": part,
                        "task": task,
                        "role": role,
                        "seed": seed,
                        "block": f"{subj}__{task}__{role}__{seed}",
                        "cluster": f"{subj}__{task}__{role}",
                        "n_turns": n_used,
                        "reasoning_tokens": rt_sum,
                        "completion_tokens": ct_sum,
                        "non_reasoning_output_tokens": nrt_out_sum,
                        "prompt_tokens": pt_sum,
                        "latency_ms": lat_sum,
                        "cost_usd": cost_sum,
                        "word_count": wc_sum,
                        "question_marks": qm_sum,
                        "expresses_confusion": float(np.mean(conf_vals)) if conf_vals else np.nan,
                        "score": score,
                    }
                )

    bdf = pd.DataFrame(block_rows)
    if bdf.empty:
        return {"ok": False, "scope": "main_arm_excluding_opus_boost", "n_episodes_main": len(main_eps)}

    pct_metrics = (
        "reasoning_tokens",
        "latency_ms",
        "word_count",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "non_reasoning_output_tokens",
    )
    abs_metrics = ("expresses_confusion", "question_marks", "score", "cost_usd")

    def _summarize_demeaned(
        sub_df: pd.DataFrame,
        *,
        partners_order: list[str],
        include_abs: bool = True,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_rows": len(sub_df),
            "n_blocks": int(sub_df["block"].nunique()),
            "pct_delta_by_metric": {},
        }
        for col in pct_metrics:
            work = sub_df.copy()
            bmean = work.groupby("block")[col].transform("mean")
            # Exclude zero-mean blocks cleanly (e.g. blocks where reasoning_tokens == 0 across all partners)
            valid_mask = bmean > 1e-12
            n_zero_blocks = int(work.loc[~valid_mask, "block"].nunique())
            work = work[valid_mask].copy()
            if work.empty:
                out["pct_delta_by_metric"][col] = {
                    "n_blocks_used": 0,
                    "n_zero_blocks_excluded": n_zero_blocks,
                    "by_partner": {},
                }
                continue
            bmean_v = work.groupby("block")[col].transform("mean")
            work["pct_delta"] = (work[col] / bmean_v - 1.0) * 100.0
            by_p: dict[str, Any] = {}
            for p in partners_order:
                pg = work[work["partner"] == p]
                if pg.empty:
                    continue
                inf = _cluster_mean_inference(
                    pg["pct_delta"].to_numpy(dtype=float),
                    pg["cluster"].to_numpy(dtype=str),
                    null_val=0.0,
                )
                inf["raw_mean"] = round(float(pg[col].mean()), 4)
                by_p[p] = inf
            out["pct_delta_by_metric"][col] = {
                "n_blocks_used": int(work["block"].nunique()),
                "n_zero_blocks_excluded": n_zero_blocks,
                "by_partner": by_p,
            }

        if include_abs:
            out["abs_delta_by_metric"] = {}
            for col in abs_metrics:
                work = sub_df.dropna(subset=[col]).copy()
                if work.empty:
                    out["abs_delta_by_metric"][col] = {"n_blocks_used": 0, "by_partner": {}}
                    continue
                bmean_v = work.groupby("block")[col].transform("mean")
                work["abs_delta"] = work[col] - bmean_v
                by_p = {}
                for p in partners_order:
                    pg = work[work["partner"] == p]
                    if pg.empty:
                        continue
                    inf = _cluster_mean_inference(
                        pg["abs_delta"].to_numpy(dtype=float),
                        pg["cluster"].to_numpy(dtype=str),
                        null_val=0.0,
                    )
                    inf["raw_mean"] = round(float(pg[col].mean()), 5)
                    by_p[p] = inf
                out["abs_delta_by_metric"][col] = {
                    "n_blocks_used": int(work["block"].nunique()),
                    "by_partner": by_p,
                }
        return out

    partners_present = [p for p in ALL_PARTNER_MODELS if p in set(bdf["partner"])]
    partners_present += [p for p in sorted(set(bdf["partner"])) if p not in partners_present]

    scopes_out: dict[str, Any] = {}
    for scope in ("exposed_turns", "all_turns"):
        sdf = bdf[bdf["scope"] == scope].copy()
        per_task = {
            str(t): _summarize_demeaned(
                sdf[sdf["task"] == t], partners_order=partners_present, include_abs=False
            )
            for t in sorted(sdf["task"].unique())
        }
        per_subject = {
            str(s): _summarize_demeaned(
                sdf[sdf["subject"] == s], partners_order=partners_present, include_abs=False
            )
            for s in [m for m in ("opus55", "neon4", "kimi31", "astra62") if m in set(sdf["subject"])]
        }
        # Reasoning-token sensitivity strictly restricting to reliably-reporting subjects (opus55, neon4, kimi31)
        rt_reliable_df = sdf[sdf["subject"].isin(["opus55", "neon4", "kimi31"])].copy()
        rt_reliable = _summarize_demeaned(
            rt_reliable_df, partners_order=partners_present, include_abs=False
        )["pct_delta_by_metric"]["reasoning_tokens"]

        scopes_out[scope] = {
            "overall": _summarize_demeaned(sdf, partners_order=partners_present, include_abs=True),
            "by_task": per_task,
            "by_subject_sensitivity": per_subject,
            "reasoning_tokens_non_openai_subjects_sensitivity": {
                "subjects": ["opus55", "neon4", "kimi31"],
                **rt_reliable,
            },
        }

    tie_efficiency: dict[str, Any] | None = None
    if choices is not None:
        ep_lookup = {str(e["episode_id"]): e for e in lookup_eps}
        tie_efficiency = _compute_exact_tie_process_efficiency(
            ep_lookup, choices, ep_total_cost, ep_role_cost
        )

    return {
        "ok": True,
        "scope": "main_arm_excluding_opus_boost",
        "n_episodes_main": len(main_eps),
        "n_episodes_excluded": n_excluded,
        "ledger_join": {
            "n_core_subject_turns_checked": int(n_turns_total),
            "n_ledger_matched": int(n_turns_ledger_matched),
            "match_rate": round(n_turns_ledger_matched / max(1, n_turns_total), 4),
        },
        "exposed_turns_primary": scopes_out["exposed_turns"],
        "all_turns_sensitivity": scopes_out["all_turns"],
        "exact_tie_process_efficiency": tie_efficiency,
    }


def _detect_calendar_shortlist_loophole(episodes_main: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure Calendar shortlist-loophole frequency (listing >=2 candidate slots per meeting across >=4 or all 7 MTGs)."""
    cal_eps = [e for e in episodes_main if e.get("task") == "calendar"]
    total_turns: collections.Counter[str] = collections.Counter()
    shortlist_4_turns: collections.Counter[str] = collections.Counter()
    full_7_turns: collections.Counter[str] = collections.Counter()
    total_ep_roles: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    shortlist_4_ep_roles: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    full_7_ep_roles: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    sample_receipts: list[dict[str, Any]] = []

    for ep in cal_eps:
        ep_id = str(ep["episode_id"])
        for t in ep.get("turns") or []:
            rnd = int(t["round"])
            for role in ("A", "B"):
                side = t[role]
                m = str(side["model"])
                txt = str(side.get("message") or "")
                total_turns[m] += 1
                total_ep_roles[m].add((ep_id, role))
                multi_slot_mtgs: set[str] = set()
                for line in txt.splitlines():
                    if _SWAP_LINE_RE.search(line):
                        continue
                    mtgs = _MTG_RE.findall(line)
                    slots = set(_SLOT_RE.findall(line))
                    if len(mtgs) == 1 and len(slots) >= 2:
                        multi_slot_mtgs.add(mtgs[0])
                if len(multi_slot_mtgs) >= 4:
                    shortlist_4_turns[m] += 1
                    shortlist_4_ep_roles[m].add((ep_id, role))
                    if len(multi_slot_mtgs) == 7:
                        full_7_turns[m] += 1
                        full_7_ep_roles[m].add((ep_id, role))
                        if len(sample_receipts) < 8:
                            sample_receipts.append(
                                {
                                    "episode_id": ep_id,
                                    "round": rnd,
                                    "role": role,
                                    "model": m,
                                    "n_meetings_listed": 7,
                                    "snippet": txt[:320],
                                }
                            )

    models_present = [m for m in ALL_PARTNER_MODELS if m in total_turns]
    models_present += [m for m in sorted(total_turns) if m not in models_present]
    by_model = {
        m: {
            "total_turns": int(total_turns[m]),
            "total_ep_roles": len(total_ep_roles[m]),
            "shortlist_ge4_turns": int(shortlist_4_turns[m]),
            "shortlist_ge4_ep_roles": len(shortlist_4_ep_roles[m]),
            "shortlist_ge4_ep_rate": round(
                len(shortlist_4_ep_roles[m]) / max(1, len(total_ep_roles[m])), 4
            ),
            "full_7_mtg_menu_turns": int(full_7_turns[m]),
            "full_7_mtg_menu_ep_roles": len(full_7_ep_roles[m]),
            "full_7_mtg_menu_ep_rate": round(
                len(full_7_ep_roles[m]) / max(1, len(total_ep_roles[m])), 4
            ),
        }
        for m in models_present
    }
    return {
        "n_calendar_episodes": len(cal_eps),
        "by_model": by_model,
        "full_7_menu_sample_receipts": sample_receipts,
    }


def _detect_neon_anomalies(
    episodes_main: list[dict[str, Any]],
    choices: list[dict[str, Any]],
    ledger_path: Path | None,
) -> dict[str, Any]:
    """Audit Neon multi-ACTION self-play continuations, visible ACTION leaks, & verbatim duplicates."""
    by_cid, by_key = _load_turn_ledger_index(ledger_path) if ledger_path else ({}, {})
    multi_action_by_model: collections.Counter[str] = collections.Counter()
    multi_action_by_task_seed: collections.Counter[str] = collections.Counter()
    multi_action_ep_roles: collections.Counter[tuple[str, str]] = collections.Counter()
    visible_action_by_model: collections.Counter[str] = collections.Counter()
    verbatim_dup_turns: list[dict[str, Any]] = []
    s303_s304_neon_receipts: list[dict[str, Any]] = []

    for ep in episodes_main:
        ep_id = str(ep["episode_id"])
        task = str(ep["task"])
        seed = int(ep["seed"])
        for t in ep.get("turns") or []:
            rnd = int(t["round"])
            for role in ("A", "B"):
                side = t[role]
                m = str(side["model"])
                msg = str(side.get("message") or "")
                wc = int((side.get("metrics") or {}).get("word_count") or len(msg.split()))

                if _VISIBLE_ACTION_RE.search(msg):
                    visible_action_by_model[m] += 1

                # Detect verbatim >=80-char repeated substring inside a single visible message
                if len(msg) >= 200:
                    seen_pos: dict[str, int] = {}
                    for i in range(0, len(msg) - 80, 20):
                        chunk = msg[i : i + 80]
                        if chunk in seen_pos and i - seen_pos[chunk] >= 80:
                            verbatim_dup_turns.append(
                                {
                                    "episode_id": ep_id,
                                    "task": task,
                                    "seed": seed,
                                    "round": rnd,
                                    "role": role,
                                    "model": m,
                                    "word_count": wc,
                                }
                            )
                            break
                        seen_pos[chunk] = i

                lr = by_cid.get(str(side.get("accepted_call_id") or "")) or by_key.get(
                    (ep_id, rnd, role)
                )
                n_act: int | None = None
                if lr is not None and lr.get("response_text") is not None:
                    raw_txt = str(lr.get("response_text") or "")
                    n_act = len(_ACTION_LINE_RE.findall(raw_txt))
                elif side.get("n_action_blocks") is not None:
                    n_act = int(side["n_action_blocks"])
                if n_act is not None and n_act >= 2:
                    multi_action_by_model[m] += 1
                    multi_action_ep_roles[(ep_id, role)] += 1
                    if m == "neon4":
                        multi_action_by_task_seed[f"{task}__s{seed}"] += 1
                        if seed in (303, 304):
                            s303_s304_neon_receipts.append(
                                {
                                    "episode_id": ep_id,
                                    "task": task,
                                    "seed": seed,
                                    "round": rnd,
                                    "role": role,
                                    "n_action_blocks": n_act,
                                    "word_count": wc,
                                }
                            )

    # Measure win-rate impact of Neon's multi-ACTION glitch on main_7p choices
    main_7p = [
        c
        for c in choices
        if c.get("choice_type") in ("main_7p", "core") and c.get("chosen_slot") in (1, 2)
    ]
    neon_win_impact: dict[str, dict[str, Any]] = {}
    for subj in CORE_MODELS:
        clean_w = clean_n = glitch_w = glitch_n = 0
        for c in main_7p:
            if c.get("subject_model") != subj:
                continue
            if "neon4" not in (c.get("partner_1"), c.get("partner_2")):
                continue
            neon_ep = (
                str(c["episode_1_id"]) if c.get("partner_1") == "neon4" else str(c["episode_2_id"])
            )
            prole = "B" if str(c["role"]) == "A" else "A"
            won = int(c.get("chosen_partner") == "neon4")
            if multi_action_ep_roles[(neon_ep, prole)] > 0:
                glitch_w += won
                glitch_n += 1
            else:
                clean_w += won
                clean_n += 1
        if clean_n + glitch_n > 0:
            neon_win_impact[subj] = {
                "clean_wins": clean_w,
                "clean_n": clean_n,
                "clean_win_rate": round(clean_w / max(1, clean_n), 4),
                "glitched_wins": glitch_w,
                "glitched_n": glitch_n,
                "glitched_win_rate": round(glitch_w / max(1, glitch_n), 4) if glitch_n else None,
            }

    return {
        "ledger_checked": bool(ledger_path and ledger_path.exists()),
        "multi_action_turns_by_model": dict(multi_action_by_model),
        "neon_multi_action_total_turns": int(multi_action_by_model.get("neon4", 0)),
        "neon_multi_action_unique_ep_roles": len(multi_action_ep_roles),
        "neon_multi_action_by_task_seed": dict(sorted(multi_action_by_task_seed.items())),
        "neon_s303_s304_multi_action_count": len(s303_s304_neon_receipts),
        "neon_s303_s304_multi_action_receipts": s303_s304_neon_receipts,
        "visible_action_leak_turns_by_model": dict(visible_action_by_model),
        "verbatim_duplicate_80char_turns": verbatim_dup_turns,
        "main_7p_neon_clean_vs_glitched_win_rate": neon_win_impact,
    }


def compute_triage_paralysis_and_upsets(
    episodes_main: list[dict[str, Any]],
    choices: list[dict[str, Any]],
    *,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Compute Triage 7x7 grid, score < 0.99 inventory, score upsets, Calendar 2-cell enforcement, & Neon anomalies."""
    main_eps, n_excluded = _filter_main_episodes(episodes_main)
    triage_eps = [e for e in main_eps if e.get("task") == "triage"]

    models_grid = list(ALL_PARTNER_MODELS)
    for ep in triage_eps:
        for mk in (str(ep["model_a"]), str(ep["model_b"])):
            if mk not in models_grid:
                models_grid.append(mk)

    # 1. Full 7 x 7 Role-A (Investigator) x Role-B (Operator) Triage objective score grid
    cell_eps: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for ep in triage_eps:
        cell_eps[(str(ep["model_a"]), str(ep["model_b"]))].append(ep)

    grid_7x7: dict[str, dict[str, dict[str, Any] | None]] = {}
    for ma in models_grid:
        grid_7x7[ma] = {}
        for mb in models_grid:
            eps_c = cell_eps.get((ma, mb))
            if not eps_c:
                grid_7x7[ma][mb] = None
                continue
            scores = [float(e["score"]) for e in eps_c]
            waits = [int((e.get("score_details") or {}).get("wait_rounds") or 0) for e in eps_c]
            greens = [bool((e.get("score_details") or {}).get("verified_green")) for e in eps_c]
            grid_7x7[ma][mb] = {
                "n": len(eps_c),
                "mean_score": round(float(np.mean(scores)), 4),
                "min_score": round(float(np.min(scores)), 4),
                "verified_green_rate": round(float(np.mean(greens)), 4),
                "mean_wait_rounds": round(float(np.mean(waits)), 2),
            }

    # Marginal summaries by Role A (Investigator) & Role B (Operator)
    by_investigator: dict[str, Any] = {}
    by_operator: dict[str, Any] = {}
    for m in models_grid:
        eps_a = [e for e in triage_eps if str(e["model_a"]) == m]
        eps_b = [e for e in triage_eps if str(e["model_b"]) == m]
        if eps_a:
            by_investigator[m] = {
                "n": len(eps_a),
                "mean_score": round(float(np.mean([float(e["score"]) for e in eps_a])), 4),
                "verified_green_rate": round(
                    float(np.mean([bool((e.get("score_details") or {}).get("verified_green")) for e in eps_a])),
                    4,
                ),
            }
        if eps_b:
            by_operator[m] = {
                "n": len(eps_b),
                "mean_score": round(float(np.mean([float(e["score"]) for e in eps_b])), 4),
                "verified_green_rate": round(
                    float(np.mean([bool((e.get("score_details") or {}).get("verified_green")) for e in eps_b])),
                    4,
                ),
                "mean_wait_rounds": round(
                    float(np.mean([int((e.get("score_details") or {}).get("wait_rounds") or 0) for e in eps_b])),
                    2,
                ),
                "total_guardrail_violations": int(
                    sum(int((e.get("score_details") or {}).get("guardrail_violations") or 0) for e in eps_b)
                ),
            }

    # 2. Full inventory of all Triage score < 0.99 episodes
    lt_99: list[dict[str, Any]] = []
    for ep in sorted(triage_eps, key=lambda e: (float(e["score"]), str(e["episode_id"]))):
        if float(ep["score"]) >= 0.99:
            continue
        sd = ep.get("score_details") or {}
        b_cmds = [
            (t["B"].get("action") or {}).get("command") if isinstance(t["B"].get("action"), dict) else None
            for t in (ep.get("turns") or [])
        ]
        lt_99.append(
            {
                "episode_id": str(ep["episode_id"]),
                "seed": int(ep["seed"]),
                "arm": str(ep.get("arm")),
                "archetype_id": sd.get("archetype_id"),
                "model_a": str(ep["model_a"]),
                "model_b": str(ep["model_b"]),
                "score": round(float(ep["score"]), 4),
                "verified_green": bool(sd.get("verified_green")),
                "step1_done": bool(sd.get("step1_done")),
                "step2_done": bool(sd.get("step2_done")),
                "wait_rounds": int(sd.get("wait_rounds") or 0),
                "guardrail_violations": int(sd.get("guardrail_violations") or 0),
                "red_herrings_run": list(sd.get("red_herrings_run") or []),
                "operator_commands": b_cmds,
            }
        )

    astra_deadlocks_00 = [
        r
        for r in lt_99
        if r["score"] == 0.0
        and {r["model_a"], r["model_b"]} <= {"astra60", "astra62"}
        and r["wait_rounds"] == 6
    ]
    s304_035_failures = [
        r for r in lt_99 if r["seed"] == 304 and abs(r["score"] - 0.35) < 1e-6
    ]

    # 3. Complete score-upset inventory (chosen_score < other_score - 0.01) across non-AA choices
    non_aa = [
        c
        for c in choices
        if c.get("choice_type") != "aa_placebo" and c.get("chosen_slot") in (1, 2)
    ]
    upset_records: list[dict[str, Any]] = []
    for c in non_aa:
        s1 = float(c["score_1"])
        s2 = float(c["score_2"])
        slot = int(c["chosen_slot"])
        chosen_s = s1 if slot == 1 else s2
        other_s = s2 if slot == 1 else s1
        if chosen_s < other_s - 0.01:
            winner = str(c["chosen_partner"])
            loser = str(c["partner_2"] if slot == 1 else c["partner_1"])
            chosen_ep = str(c["episode_1_id"] if slot == 1 else c["episode_2_id"])
            other_ep = str(c["episode_2_id"] if slot == 1 else c["episode_1_id"])
            upset_records.append(
                {
                    "manifest_item_id": str(c.get("manifest_item_id") or ""),
                    "choice_type": str(c.get("choice_type")),
                    "order": str(c.get("order") or ""),
                    "task": str(c["task"]),
                    "role": str(c["role"]),
                    "subject_model": str(c["subject_model"]),
                    "winner": winner,
                    "loser": loser,
                    "chosen_score": round(chosen_s, 4),
                    "other_score": round(other_s, 4),
                    "score_deficit": round(other_s - chosen_s, 4),
                    "chosen_episode_id": chosen_ep,
                    "other_episode_id": other_ep,
                    "reason": _extract_choice_reason(c),
                }
            )

    upset_records.sort(key=lambda r: (-r["score_deficit"], r["task"], r["manifest_item_id"]))

    opus_triage_s304_receipts = [
        r
        for r in upset_records
        if r["task"] == "triage"
        and r["subject_model"] == "opus55"
        and "s304" in r["chosen_episode_id"]
        and r["chosen_score"] <= 0.35 + 1e-6
        and r["other_score"] >= 0.99
    ]

    # 4. Calendar shortlist-loophole & Opus 2-cell enforcement counts
    cal_shortlist = _detect_calendar_shortlist_loophole(main_eps)
    cal_choices = [c for c in non_aa if c.get("task") == "calendar"]
    cal_2cell_by_subj: collections.Counter[str] = collections.Counter()
    cal_2cell_upset_by_subj: collections.Counter[str] = collections.Counter()
    opus_2cell_losers: collections.Counter[str] = collections.Counter()
    opus_2cell_winners: collections.Counter[str] = collections.Counter()
    opus_2cell_upset_receipts: list[dict[str, Any]] = []
    for c in cal_choices:
        reason = _extract_choice_reason(c)
        if not _CALENDAR_2CELL_RULE_RE.search(reason):
            continue
        subj = str(c["subject_model"])
        slot = int(c["chosen_slot"])
        winner = str(c["chosen_partner"])
        loser = str(c["partner_2"] if slot == 1 else c["partner_1"])
        cs = float(c["score_1"] if slot == 1 else c["score_2"])
        os_ = float(c["score_2"] if slot == 1 else c["score_1"])
        cal_2cell_by_subj[subj] += 1
        if subj == "opus55":
            opus_2cell_winners[winner] += 1
            opus_2cell_losers[loser] += 1
        if cs < os_ - 0.01:
            cal_2cell_upset_by_subj[subj] += 1
            if subj == "opus55":
                opus_2cell_upset_receipts.append(
                    {
                        "manifest_item_id": str(c.get("manifest_item_id") or ""),
                        "winner": winner,
                        "winner_score": round(cs, 4),
                        "loser": loser,
                        "loser_score": round(os_, 4),
                        "reason": reason,
                    }
                )

    # 5. Neon s303/s304 duplicate-text / multi-ACTION anomaly summary
    neon_anomalies = _detect_neon_anomalies(main_eps, choices, ledger_path)

    return {
        "ok": True,
        "scope": {
            "triage_and_calendar_episodes": "main_arm_excluding_opus_boost",
            "n_episodes_main": len(main_eps),
            "n_episodes_excluded": n_excluded,
            "n_triage_main_episodes": len(triage_eps),
            "upsets_choice_scope": "all_non_aa_choices",
            "n_non_aa_choices": len(non_aa),
        },
        "triage_7x7_grid": {
            "models_order": models_grid,
            "cells": grid_7x7,
            "by_investigator_role_a": by_investigator,
            "by_operator_role_b": by_operator,
        },
        "triage_below_099_inventory": {
            "n_episodes": len(lt_99),
            "astra60_astra62_00_deadlocks_count": len(astra_deadlocks_00),
            "astra60_astra62_00_deadlocks": astra_deadlocks_00,
            "s304_035_failures_count": len(s304_035_failures),
            "s304_035_failures": s304_035_failures,
            "all_episodes": lt_99,
        },
        "score_upsets": {
            "threshold": "chosen_score < other_score - 0.01",
            "n_total_upsets": len(upset_records),
            "n_main_7p_upsets": sum(1 for r in upset_records if r["choice_type"] in ("main_7p", "core")),
            "by_task": dict(collections.Counter(r["task"] for r in upset_records)),
            "by_subject": dict(collections.Counter(r["subject_model"] for r in upset_records)),
            "by_winner": dict(collections.Counter(r["winner"] for r in upset_records)),
            "by_loser": dict(collections.Counter(r["loser"] for r in upset_records)),
            "opus_triage_s304_035_over_100_receipts": opus_triage_s304_receipts,
            "all_triage_upsets": [r for r in upset_records if r["task"] == "triage"],
            "top_20_largest_deficit_upsets": upset_records[:20],
        },
        "calendar_shortlist_and_2cell_enforcement": {
            "shortlist_loophole": cal_shortlist,
            "choice_reason_2cell_mentions": {
                "n_calendar_non_aa_choices": len(cal_choices),
                "total_mentions": int(sum(cal_2cell_by_subj.values())),
                "by_subject": dict(cal_2cell_by_subj),
                "score_upsets_citing_2cell_by_subject": dict(cal_2cell_upset_by_subj),
                "opus_winners_when_citing_2cell": dict(opus_2cell_winners),
                "opus_losers_when_citing_2cell": dict(opus_2cell_losers),
                "opus_2cell_upset_receipts": opus_2cell_upset_receipts[:12],
            },
        },
        "neon_s303_s304_anomaly": neon_anomalies,
    }


def _model_family(model_key: str) -> str:
    spec = DEFAULT_MODELS.get(model_key)
    return spec.family if spec is not None else "other"


def _two_group_nonparam(
    a_vals: list[float],
    b_vals: list[float],
    *,
    n_perm: int = 5000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Two-sample Mann-Whitney U + deterministic label-permutation p-value."""
    na, nb = len(a_vals), len(b_vals)
    if na == 0 or nb == 0:
        return {
            "delta_mean": None,
            "p_mwu": None,
            "p_mwu_fmt": "n/a",
            "p_perm": None,
            "p_perm_fmt": "n/a",
        }
    arr_a = np.asarray(a_vals, dtype=float)
    arr_b = np.asarray(b_vals, dtype=float)
    diff = float(np.mean(arr_a) - np.mean(arr_b))
    mwu_p = float(stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided").pvalue)
    pooled = np.concatenate([arr_a, arr_b])
    obs = abs(diff)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        p_diff = abs(float(np.mean(perm[:na]) - np.mean(perm[na:])))
        if p_diff >= obs - 1e-12:
            ge += 1
    perm_p = float((ge + 1) / (n_perm + 1))
    return {
        "delta_mean": round(diff, 4),
        "p_mwu": round(mwu_p, 6),
        "p_mwu_fmt": _fmt_p(mwu_p),
        "p_perm": round(perm_p, 6),
        "p_perm_fmt": _fmt_p(perm_p),
    }


def _episode_clustered_win_rate_bootstrap(
    outcomes_by_unit: dict[tuple[str, str], list[float]],
    *,
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Win rate + unit-cluster bootstrap 95% CI (resampling unique episode-roles with replacement)."""
    units = sorted(outcomes_by_unit.keys())
    all_vals = [v for u in units for v in outcomes_by_unit[u]]
    n_choices = len(all_vals)
    wins = int(sum(all_vals))
    if n_choices == 0:
        return {
            "wins": 0,
            "n_choices": 0,
            "n_unique_episodes": 0,
            "win_rate": None,
            "episode_boot_ci95": None,
        }
    rate = float(np.mean(all_vals))
    ci95: list[float] | None = None
    if len(units) >= 2:
        rng = np.random.default_rng(seed)
        unit_arrays = [np.asarray(outcomes_by_unit[u], dtype=float) for u in units]
        k = len(unit_arrays)
        boots = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, k, size=k)
            sample = np.concatenate([unit_arrays[j] for j in idx])
            boots[i] = float(np.mean(sample))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ci95 = [round(float(lo), 4), round(float(hi), 4)]
    return {
        "wins": wins,
        "n_choices": n_choices,
        "n_unique_episodes": len(units),
        "win_rate": round(rate, 4),
        "episode_boot_ci95": ci95,
    }


def _compare_unit_clustered_choice_rates(
    shared_by_unit: dict[tuple[str, str], list[float]],
    withheld_by_unit: dict[tuple[str, str], list[float]],
    *,
    n_perm: int = 5000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare Shared vs. Withheld choice win rates via Fisher's exact + episode-cluster permutation."""
    sh_vals = [v for vals in shared_by_unit.values() for v in vals]
    wh_vals = [v for vals in withheld_by_unit.values() for v in vals]
    if not sh_vals or not wh_vals:
        return {
            "win_rate_diff": None,
            "fisher_exact_p": None,
            "fisher_exact_p_fmt": "n/a",
            "episode_perm_p": None,
            "episode_perm_p_fmt": "n/a",
        }
    sh_w = int(sum(sh_vals))
    sh_l = len(sh_vals) - sh_w
    wh_w = int(sum(wh_vals))
    wh_l = len(wh_vals) - wh_w
    diff = float(np.mean(sh_vals) - np.mean(wh_vals))
    fisher_p = float(stats.fisher_exact([[sh_w, sh_l], [wh_w, wh_l]], alternative="two-sided").pvalue)

    # Episode-level label permutation preserving exact choice counts per unique episode
    sh_units = [np.asarray(shared_by_unit[u], dtype=float) for u in sorted(shared_by_unit)]
    wh_units = [np.asarray(withheld_by_unit[u], dtype=float) for u in sorted(withheld_by_unit)]
    all_units = sh_units + wh_units
    n_sh_u = len(sh_units)
    n_tot_u = len(all_units)
    obs = abs(diff)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n_tot_u)
        s_arr = np.concatenate([all_units[i] for i in perm_idx[:n_sh_u]])
        w_arr = np.concatenate([all_units[i] for i in perm_idx[n_sh_u:]])
        if abs(float(np.mean(s_arr) - np.mean(w_arr))) >= obs - 1e-12:
            ge += 1
    ep_perm_p = float((ge + 1) / (n_perm + 1))

    return {
        "win_rate_diff": round(diff, 4),
        "fisher_exact_p": round(fisher_p, 6),
        "fisher_exact_p_fmt": _fmt_p(fisher_p),
        "episode_perm_p": round(ep_perm_p, 6),
        "episode_perm_p_fmt": _fmt_p(ep_perm_p),
    }


def _summarize_goalpost_disclosure_table(
    gp_episodes: list[dict[str, Any]],
    disc_map: dict[tuple[str, str], Any],
    *,
    scope_label: str,
    includes_opus_boost: bool,
) -> dict[str, Any]:
    """Build 7-model Goalpost disclosure table (R1, R1-R2, R1-R8, histogram, all 14 rules)."""
    records: list[dict[str, Any]] = []
    for ep in gp_episodes:
        ep_id = str(ep["episode_id"])
        for role in ("A", "B"):
            m = str(ep["model_a"] if role == "A" else ep["model_b"])
            d = disc_map[(ep_id, role)]
            records.append({"episode_id": ep_id, "role": role, "model": m, "disc": d})

    models_present = [m for m in ALL_PARTNER_MODELS if any(r["model"] == m for r in records)]
    models_present += [m for m in sorted({r["model"] for r in records}) if m not in models_present]

    def _summarize_model_records(sub: list[dict[str, Any]]) -> dict[str, Any]:
        ra = [r for r in sub if r["role"] == "A"]
        rb = [r for r in sub if r["role"] == "B"]
        hist = {str(k): 0 for k in range(8)}
        for r in sub:
            hist[str(int(r["disc"].r1.n_rules_formula))] += 1
        per_rule: dict[str, float | None] = {}
        for rid in GOALPOST_ROLE_A_RULES:
            per_rule[rid] = (
                round(float(np.mean([r["disc"].r1.rules_disclosed_formula[rid] for r in ra])), 4)
                if ra
                else None
            )
        for rid in GOALPOST_ROLE_B_RULES:
            per_rule[rid] = (
                round(float(np.mean([r["disc"].r1.rules_disclosed_formula[rid] for r in rb])), 4)
                if rb
                else None
            )
        return {
            "n_r1_turns": len(sub),
            "n_role_a": len(ra),
            "n_role_b": len(rb),
            "r1_mean_formula_rules": round(float(np.mean([r["disc"].r1.n_rules_formula for r in sub])), 4)
            if sub
            else None,
            "r1_full_formula_rate": round(
                float(np.mean([r["disc"].r1.full_formula_disclosure for r in sub])), 4
            )
            if sub
            else None,
            "r1_role_a_full_formula_rate": round(
                float(np.mean([r["disc"].r1.full_formula_disclosure for r in ra])), 4
            )
            if ra
            else None,
            "r1_role_b_full_formula_rate": round(
                float(np.mean([r["disc"].r1.full_formula_disclosure for r in rb])), 4
            )
            if rb
            else None,
            "r1_legacy_id_ge5_rate": round(
                float(np.mean([r["disc"].r1.n_rules_id >= 5 for r in sub])), 4
            )
            if sub
            else None,
            "cum_r1_r2_full_formula_rate": round(
                float(np.mean([r["disc"].r1_r2.full_formula_disclosure for r in sub])), 4
            )
            if sub
            else None,
            "cum_r1_r8_full_formula_rate": round(
                float(np.mean([r["disc"].final.full_formula_disclosure for r in sub])), 4
            )
            if sub
            else None,
            "r1_rule_count_histogram": hist,
            "per_rule_r1_disclosure_rate": per_rule,
        }

    by_model = {m: _summarize_model_records([r for r in records if r["model"] == m]) for m in models_present}
    pooled = _summarize_model_records(records)
    return {
        "scope": scope_label,
        "includes_opus_boost": includes_opus_boost,
        "n_episodes": len(gp_episodes),
        "n_r1_turns": len(records),
        "n_opus_boost_episodes": sum(1 for e in gp_episodes if e.get("arm") == "opus_boost"),
        "by_model": by_model,
        "pooled": pooled,
    }


def _summarize_discordant_tie_preference(
    choice_rows: list[dict[str, Any]],
    disc_fn: Any,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Summarize exact-tie preference for the disclosing candidate with binomial + case-level permutation."""
    wins = 0
    n_total = 0
    case_outcomes: dict[str, list[float]] = collections.defaultdict(list)
    for c in choice_rows:
        prole = "B" if str(c["role"]) == "A" else "A"
        d1 = disc_fn(str(c["episode_1_id"]), prole)
        d2 = disc_fn(str(c["episode_2_id"]), prole)
        if d1 is None or d2 is None or d1 == d2:
            continue
        won = 1.0 if ((d1 and int(c["chosen_slot"]) == 1) or (d2 and int(c["chosen_slot"]) == 2)) else 0.0
        wins += int(won)
        n_total += 1
        case_key = str(
            c.get("item_id")
            or f"{c.get('subject_model')}__{c.get('role')}__{min(str(c['episode_1_id']), str(c['episode_2_id']))}__{max(str(c['episode_1_id']), str(c['episode_2_id']))}"
        )
        case_outcomes[case_key].append(won)

    if n_total == 0:
        return {
            "wins": 0,
            "n_discordant_ties": 0,
            "win_rate": None,
            "binom_p": None,
            "binom_p_fmt": "n/a",
            "n_unique_cases": 0,
            "case_boot_ci95": None,
            "case_perm_p": None,
            "case_perm_p_fmt": "n/a",
        }

    rate = wins / n_total
    binom_p = float(stats.binomtest(wins, n_total, 0.5, alternative="two-sided").pvalue)

    # Collapse AB/BA by unique episode-pair case for cluster-robust bootstrap & sign-flip permutation
    case_vals = np.array([float(np.mean(v)) for v in case_outcomes.values()], dtype=float)
    case_ids = np.array(list(case_outcomes.keys()), dtype=str)
    inf = _cluster_mean_inference(case_vals, case_ids, null_val=0.5, n_perm=2000, seed=seed)
    ci95: list[float] | None = None
    if len(case_vals) >= 2:
        rng = np.random.default_rng(seed)
        k = len(case_vals)
        boots = np.array(
            [float(np.mean(case_vals[rng.integers(0, k, size=k)])) for _ in range(2000)],
            dtype=float,
        )
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ci95 = [round(float(lo), 4), round(float(hi), 4)]

    return {
        "wins": wins,
        "n_discordant_ties": n_total,
        "win_rate": round(rate, 4),
        "binom_p": round(binom_p, 6),
        "binom_p_fmt": _fmt_p(binom_p),
        "n_unique_cases": len(case_vals),
        "case_boot_ci95": ci95,
        "case_perm_p": inf["p_perm"],
        "case_perm_p_fmt": inf["p_perm_fmt"],
    }


def compute_disclosure_meta_analysis(
    episodes_main: list[dict[str, Any]],
    choices: list[dict[str, Any]],
    *,
    episodes_all: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full Phase 1 Part-B disclosure meta-analysis powered strictly by ``common_ground.disclosure``.

    Computes:
    1. 7-model Goalpost disclosure table (primary: ``episodes_main``; sensitivity: ``episodes_all``)
    2. Within-GPT-6.2 same-length eval blindness (overall, ``score == 1.0``, Role A, Role B)
    3. Ask-without-giving asymmetry (Role A blind R1 + overall)
    4. Contagious withholding (Role B conditioned on Role A R1, plus Role-B ``opus55`` breakdown)
    5. Producer-consumer hypocrisy & 14-rule exact-tie robustness
    6. Triage Role-B R1 disclosure & Exp-2 strict zero-contradiction pool inventory
    """
    main_eps, n_excluded = _filter_main_episodes(episodes_main)
    all_eps = (
        [e for e in episodes_all if e.get("complete") is not False]
        if episodes_all is not None
        else list(main_eps)
    )

    gp_main = [e for e in main_eps if e.get("task") == "goalpost" and e.get("turns")]
    gp_all = [e for e in all_eps if e.get("task") == "goalpost" and e.get("turns")]
    tr_main = [e for e in main_eps if e.get("task") == "triage" and e.get("turns")]

    # Compute exact rule-by-rule Goalpost disclosure once for all Goalpost episodes (main + boost)
    disc_map: dict[tuple[str, str], Any] = {}
    for ep in gp_all:
        ep_id = str(ep["episode_id"])
        for role in ("A", "B"):
            disc_map[(ep_id, role)] = check_goalpost_episode_disclosure(ep, role)

    # 1. 7-Model Goalpost Disclosure Table (Primary: main arm; Secondary sensitivity: all incl. boost)
    table_main = _summarize_goalpost_disclosure_table(
        gp_main, disc_map, scope_label="main_arm_excluding_opus_boost", includes_opus_boost=False
    )
    table_all = _summarize_goalpost_disclosure_table(
        gp_all, disc_map, scope_label="all_complete_including_opus_boost", includes_opus_boost=True
    )

    # 2. Within-GPT-6.2 Same-Length Eval Blindness (Main arm Goalpost)
    a62_records: list[dict[str, Any]] = []
    for ep in gp_main:
        ep_id = str(ep["episode_id"])
        turns = sorted(ep.get("turns") or [], key=lambda t: int(t.get("round", 0)))
        for part_role, subj_role in (("A", "B"), ("B", "A")):
            part = str(ep["model_a"] if part_role == "A" else ep["model_b"])
            subj = str(ep["model_b"] if part_role == "A" else ep["model_a"])
            if part != "astra62":
                continue
            d = disc_map[(ep_id, part_role)]
            r1_wc = float((turns[0][part_role].get("metrics") or {}).get("word_count") or 0.0)
            ep_wc = float(
                np.mean(
                    [float((t[part_role].get("metrics") or {}).get("word_count") or 0.0) for t in turns]
                )
            )
            deb = (ep.get("debriefs") or {}).get(subj_role) or {}
            f1 = (deb.get("F1") or {}).get("parsed") or {}
            f4 = (deb.get("F4") or {}).get("choice")
            a62_records.append(
                {
                    "episode_id": ep_id,
                    "part_role": part_role,
                    "subject_model": subj,
                    "cross_vendor": _model_family(subj) != _model_family(part),
                    "shared": bool(d.r1.full_formula_disclosure),
                    "n_rules_r1": int(d.r1.n_rules_formula),
                    "score": float(ep["score"]),
                    "r1_word_count": r1_wc,
                    "ep_mean_word_count": ep_wc,
                    "f1_again": float(f1["again"]) if f1.get("again") is not None else None,
                    "f1_clarity": float(f1["clarity"]) if f1.get("clarity") is not None else None,
                    "f1_overhead": float(f1["overhead"]) if f1.get("overhead") is not None else None,
                    "f4_keep": 1.0 if f4 == "KEEP" else (0.0 if f4 == "REROLL" else None),
                }
            )

    main_7p_gp_choices = [
        c
        for c in choices
        if c.get("task") == "goalpost"
        and c.get("choice_type") in ("main_7p", "core")
        and c.get("chosen_slot") in (1, 2)
    ]

    def _summarize_a62_slice(slice_recs: list[dict[str, Any]]) -> dict[str, Any]:
        groups_out: dict[str, Any] = {}
        choice_units_overall: dict[str, dict[tuple[str, str], list[float]]] = {}
        choice_units_ties: dict[str, dict[tuple[str, str], list[float]]] = {}
        for sh_flag, grp_name in ((True, "shared"), (False, "withheld")):
            sub = [r for r in slice_recs if r["shared"] is sh_flag]
            cv_sub = [r for r in sub if r["cross_vendor"]]
            unit_set = {(r["episode_id"], r["part_role"]) for r in sub}

            ov_by_u: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
            tie_by_u: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
            for c in main_7p_gp_choices:
                if _model_family(str(c["subject_model"])) == _model_family("astra62"):
                    continue
                prole = "B" if str(c["role"]) == "A" else "A"
                is_tie = abs(float(c["score_1"]) - float(c["score_2"])) < 1e-9
                for slot_i, pk, ek in (
                    (1, "partner_1", "episode_1_id"),
                    (2, "partner_2", "episode_2_id"),
                ):
                    unit = (str(c[ek]), prole)
                    if c.get(pk) == "astra62" and unit in unit_set:
                        won = 1.0 if int(c["chosen_slot"]) == slot_i else 0.0
                        ov_by_u[unit].append(won)
                        if is_tie:
                            tie_by_u[unit].append(won)

            choice_units_overall[grp_name] = ov_by_u
            choice_units_ties[grp_name] = tie_by_u

            f1_a = [r["f1_again"] for r in cv_sub if r["f1_again"] is not None]
            f1_c = [r["f1_clarity"] for r in cv_sub if r["f1_clarity"] is not None]
            f1_o = [r["f1_overhead"] for r in cv_sub if r["f1_overhead"] is not None]
            f4_k = [r["f4_keep"] for r in cv_sub if r["f4_keep"] is not None]
            groups_out[grp_name] = {
                "n_episodes": len(sub),
                "mean_score": _round_opt(float(np.mean([r["score"] for r in sub])), 4) if sub else None,
                "mean_r1_word_count": _round_opt(float(np.mean([r["r1_word_count"] for r in sub])), 2)
                if sub
                else None,
                "mean_episode_word_count": _round_opt(
                    float(np.mean([r["ep_mean_word_count"] for r in sub])), 2
                )
                if sub
                else None,
                "cross_vendor_debriefs": {
                    "n_f1": len(f1_a),
                    "mean_f1_again": _round_opt(float(np.mean(f1_a)), 4) if f1_a else None,
                    "mean_f1_clarity": _round_opt(float(np.mean(f1_c)), 4) if f1_c else None,
                    "mean_f1_overhead": _round_opt(float(np.mean(f1_o)), 4) if f1_o else None,
                    "n_f4": len(f4_k),
                    "f4_keep_rate": _round_opt(float(np.mean(f4_k)), 4) if f4_k else None,
                },
                "cross_vendor_main_7p_choice_overall": _episode_clustered_win_rate_bootstrap(ov_by_u),
                "cross_vendor_main_7p_choice_exact_ties": _episode_clustered_win_rate_bootstrap(
                    tie_by_u
                ),
            }

        sh_g = groups_out["shared"]
        wh_g = groups_out["withheld"]
        return {
            "shared": sh_g,
            "withheld": wh_g,
            "contrast_shared_minus_withheld": {
                "score_diff": _round_opt(
                    sh_g["mean_score"] - wh_g["mean_score"], 4
                )
                if (sh_g["mean_score"] is not None and wh_g["mean_score"] is not None)
                else None,
                "r1_word_count_diff": _round_opt(
                    sh_g["mean_r1_word_count"] - wh_g["mean_r1_word_count"], 2
                )
                if (sh_g["mean_r1_word_count"] is not None and wh_g["mean_r1_word_count"] is not None)
                else None,
                "episode_word_count_diff": _round_opt(
                    sh_g["mean_episode_word_count"] - wh_g["mean_episode_word_count"], 2
                )
                if (
                    sh_g["mean_episode_word_count"] is not None
                    and wh_g["mean_episode_word_count"] is not None
                )
                else None,
                "f1_again_diff": _round_opt(
                    sh_g["cross_vendor_debriefs"]["mean_f1_again"]
                    - wh_g["cross_vendor_debriefs"]["mean_f1_again"],
                    4,
                )
                if (
                    sh_g["cross_vendor_debriefs"]["mean_f1_again"] is not None
                    and wh_g["cross_vendor_debriefs"]["mean_f1_again"] is not None
                )
                else None,
                "choice_overall": _compare_unit_clustered_choice_rates(
                    choice_units_overall["shared"], choice_units_overall["withheld"]
                ),
                "choice_exact_ties": _compare_unit_clustered_choice_rates(
                    choice_units_ties["shared"], choice_units_ties["withheld"]
                ),
            },
        }

    within_gpt62 = {
        "scope": "main_arm_goalpost_astra62_as_partner",
        "overall": _summarize_a62_slice(a62_records),
        "score_eq_1_0_only": _summarize_a62_slice(
            [r for r in a62_records if abs(r["score"] - 1.0) < 1e-9]
        ),
        "role_a_only": _summarize_a62_slice([r for r in a62_records if r["part_role"] == "A"]),
        "role_b_only": _summarize_a62_slice([r for r in a62_records if r["part_role"] == "B"]),
        "role_a_score_eq_1_0": _summarize_a62_slice(
            [r for r in a62_records if r["part_role"] == "A" and abs(r["score"] - 1.0) < 1e-9]
        ),
        "role_b_score_eq_1_0": _summarize_a62_slice(
            [r for r in a62_records if r["part_role"] == "B" and abs(r["score"] - 1.0) < 1e-9]
        ),
    }

    # 3. Ask-Without-Giving Asymmetry (Goalpost Role A speaking blind in R1, plus overall & Role B)
    models_gp = [m for m in ALL_PARTNER_MODELS if m in table_main["by_model"]]
    models_gp += [m for m in sorted(table_main["by_model"]) if m not in models_gp]

    def _summarize_ask_without_giving(roles_filter: tuple[str, ...]) -> dict[str, Any]:
        by_m: dict[str, Any] = {}
        all_items: list[Any] = []
        for m in models_gp:
            items = [
                disc_map[(str(e["episode_id"]), role)].r1
                for e in gp_main
                for role in roles_filter
                if str(e["model_a"] if role == "A" else e["model_b"]) == m
            ]
            all_items.extend(items)
            n_i = len(items)
            asks = sum(1 for d in items if d.asks_partner_for_rules)
            full = sum(1 for d in items if d.full_formula_disclosure)
            awg = sum(1 for d in items if d.asks_partner_for_rules and not d.full_formula_disclosure)
            awg0 = sum(1 for d in items if d.asks_partner_for_rules and d.n_rules_formula == 0)
            by_m[m] = {
                "n_turns": n_i,
                "asks_partner_for_rules_count": asks,
                "asks_partner_for_rules_rate": round(asks / max(1, n_i), 4) if n_i else None,
                "full_formula_disclosure_count": full,
                "full_formula_disclosure_rate": round(full / max(1, n_i), 4) if n_i else None,
                "asks_without_giving_count": awg,
                "asks_without_giving_rate": round(awg / max(1, n_i), 4) if n_i else None,
                "asks_with_zero_rules_count": awg0,
                "asks_with_zero_rules_rate": round(awg0 / max(1, n_i), 4) if n_i else None,
            }
        n_all = len(all_items)
        return {
            "by_model": by_m,
            "pooled": {
                "n_turns": n_all,
                "asks_partner_for_rules_rate": round(
                    sum(1 for d in all_items if d.asks_partner_for_rules) / max(1, n_all), 4
                )
                if n_all
                else None,
                "full_formula_disclosure_rate": round(
                    sum(1 for d in all_items if d.full_formula_disclosure) / max(1, n_all), 4
                )
                if n_all
                else None,
                "asks_without_giving_rate": round(
                    sum(
                        1
                        for d in all_items
                        if d.asks_partner_for_rules and not d.full_formula_disclosure
                    )
                    / max(1, n_all),
                    4,
                )
                if n_all
                else None,
                "asks_with_zero_rules_rate": round(
                    sum(
                        1 for d in all_items if d.asks_partner_for_rules and d.n_rules_formula == 0
                    )
                    / max(1, n_all),
                    4,
                )
                if n_all
                else None,
            },
        }

    ask_without_giving = {
        "scope": "main_arm_goalpost_r1",
        "role_a_blind_r1": _summarize_ask_without_giving(("A",)),
        "role_b_r1": _summarize_ask_without_giving(("B",)),
        "overall_all_r1": _summarize_ask_without_giving(("A", "B")),
    }

    # 4. Contagious Withholding (Goalpost Role B conditioned on Role A's R1 disclosure)
    def _contagion_entry(ep_list: list[dict[str, Any]]) -> dict[str, Any]:
        after_sh = [
            float(disc_map[(str(e["episode_id"]), "B")].r1.n_rules_formula)
            for e in ep_list
            if disc_map[(str(e["episode_id"]), "A")].r1.full_formula_disclosure
        ]
        after_wh = [
            float(disc_map[(str(e["episode_id"]), "B")].r1.n_rules_formula)
            for e in ep_list
            if not disc_map[(str(e["episode_id"]), "A")].r1.full_formula_disclosure
        ]
        stats_cmp = _two_group_nonparam(after_sh, after_wh)
        return {
            "n_after_role_a_disclosed": len(after_sh),
            "mean_rules_after_role_a_disclosed": _round_opt(float(np.mean(after_sh)), 4)
            if after_sh
            else None,
            "full_disclosure_rate_after_role_a_disclosed": _round_opt(
                float(np.mean([v >= 5 for v in after_sh])), 4
            )
            if after_sh
            else None,
            "n_after_role_a_withheld": len(after_wh),
            "mean_rules_after_role_a_withheld": _round_opt(float(np.mean(after_wh)), 4)
            if after_wh
            else None,
            "full_disclosure_rate_after_role_a_withheld": _round_opt(
                float(np.mean([v >= 5 for v in after_wh])), 4
            )
            if after_wh
            else None,
            **stats_cmp,
        }

    contagion_by_b = {
        m: _contagion_entry([e for e in gp_main if str(e["model_b"]) == m]) for m in models_gp
    }
    contagion_pooled = _contagion_entry(gp_main)

    opus_b_eps = [e for e in gp_main if str(e["model_b"]) == "opus55"]
    opus_b_by_partner: dict[str, Any] = {}
    for ma in models_gp:
        sub_e = [e for e in opus_b_eps if str(e["model_a"]) == ma]
        if not sub_e:
            continue
        a_counts = [int(disc_map[(str(e["episode_id"]), "A")].r1.n_rules_formula) for e in sub_e]
        b_counts = [int(disc_map[(str(e["episode_id"]), "B")].r1.n_rules_formula) for e in sub_e]
        opus_b_by_partner[ma] = {
            "n": len(sub_e),
            "role_a_mean_rules": round(float(np.mean(a_counts)), 4),
            "opus55_b_mean_rules": round(float(np.mean(b_counts)), 4),
            "opus55_b_full_disclosure_rate": round(float(np.mean([v >= 5 for v in b_counts])), 4),
            "opus55_b_rule_counts": b_counts,
        }

    transparent_a_partners = ("neon4", "muse", "astra60", "opus55")
    withholding_a_partners = ("astra62", "kimi31")
    opus_after_transp = [
        float(disc_map[(str(e["episode_id"]), "B")].r1.n_rules_formula)
        for e in opus_b_eps
        if str(e["model_a"]) in transparent_a_partners
    ]
    opus_after_withh = [
        float(disc_map[(str(e["episode_id"]), "B")].r1.n_rules_formula)
        for e in opus_b_eps
        if str(e["model_a"]) in withholding_a_partners
    ]
    opus_partner_group_contrast = {
        "transparent_partners": list(transparent_a_partners),
        "withholding_partners": list(withholding_a_partners),
        "n_transparent": len(opus_after_transp),
        "mean_rules_after_transparent": _round_opt(float(np.mean(opus_after_transp)), 4)
        if opus_after_transp
        else None,
        "full_disclosure_rate_after_transparent": _round_opt(
            float(np.mean([v >= 5 for v in opus_after_transp])), 4
        )
        if opus_after_transp
        else None,
        "n_withholding": len(opus_after_withh),
        "mean_rules_after_withholding": _round_opt(float(np.mean(opus_after_withh)), 4)
        if opus_after_withh
        else None,
        "full_disclosure_rate_after_withholding": _round_opt(
            float(np.mean([v >= 5 for v in opus_after_withh])), 4
        )
        if opus_after_withh
        else None,
        **_two_group_nonparam(opus_after_transp, opus_after_withh),
    }

    contagious_withholding = {
        "scope": "main_arm_goalpost_role_b_conditioned_on_role_a_r1",
        "pooled_all_role_b": contagion_pooled,
        "by_role_b_model": contagion_by_b,
        "role_b_opus55_by_role_a_partner": opus_b_by_partner,
        "role_b_opus55_transparent_vs_withholding_partners": opus_partner_group_contrast,
    }

    # 5. Producer-Consumer Hypocrisy & Multi-Rule Robustness (Goalpost Exact Ties |d_score| == 0)
    gp_non_aa_ties = [
        c
        for c in choices
        if c.get("task") == "goalpost"
        and c.get("choice_type") != "aa_placebo"
        and c.get("chosen_slot") in (1, 2)
        and c.get("score_1") is not None
        and c.get("score_2") is not None
        and abs(float(c["score_1"]) - float(c["score_2"])) < 1e-9
    ]
    gp_main_7p_ties = [
        c for c in gp_non_aa_ties if c.get("choice_type") in ("main_7p", "core")
    ]

    def _full_disc_lookup(ep_id: str, prole: str) -> bool | None:
        d = disc_map.get((ep_id, prole))
        return bool(d.r1.full_formula_disclosure) if d is not None else None

    def _build_hypocrisy_table(tie_rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_subj: dict[str, Any] = {}
        for subj in CORE_MODELS:
            s_rows = [c for c in tie_rows if str(c.get("subject_model")) == subj]
            pref = _summarize_discordant_tie_preference(s_rows, _full_disc_lookup)
            prod_m = table_main["by_model"].get(subj) or {}
            prod_ra = prod_m.get("r1_role_a_full_formula_rate")
            prod_all = prod_m.get("r1_full_formula_rate")
            gap = (
                round((pref["win_rate"] - prod_ra) * 100.0, 2)
                if (pref["win_rate"] is not None and prod_ra is not None)
                else None
            )
            by_subj[subj] = {
                "producer_role_a_full_disclosure_rate": prod_ra,
                "producer_overall_full_disclosure_rate": prod_all,
                "hypocrisy_gap_pp_consumer_minus_producer_role_a": gap,
                **pref,
            }
        return {
            "n_total_exact_ties": len(tie_rows),
            "by_chooser": by_subj,
            "pooled_all_choosers": _summarize_discordant_tie_preference(
                tie_rows, _full_disc_lookup
            ),
        }

    # Multi-rule robustness across all 14 rules (C-A1..C-A7, C-B1..C-B7)
    multi_rule: dict[str, Any] = {}
    for target_prole, rule_tuple in (
        ("A", GOALPOST_ROLE_A_RULES),
        ("B", GOALPOST_ROLE_B_RULES),
    ):
        role_ties = [
            c for c in gp_non_aa_ties if ("B" if str(c["role"]) == "A" else "A") == target_prole
        ]
        for rid in rule_tuple:

            def _rule_lookup(ep_id: str, prole: str, _rid: str = rid) -> bool | None:
                d = disc_map.get((ep_id, prole))
                if d is None or _rid not in d.r1.rules_disclosed_formula:
                    return None
                return bool(d.r1.rules_disclosed_formula[_rid])

            multi_rule[rid] = {
                "partner_role": target_prole,
                **_summarize_discordant_tie_preference(role_ties, _rule_lookup),
            }

    rule_rates = [v["win_rate"] for v in multi_rule.values() if v["win_rate"] is not None]
    producer_consumer = {
        "scope": "goalpost_exact_ties_abs_d_score_eq_0",
        "all_non_aa_exact_ties": _build_hypocrisy_table(gp_non_aa_ties),
        "main_7p_exact_ties_sensitivity": _build_hypocrisy_table(gp_main_7p_ties),
        "multi_rule_robustness_14_rules": {
            "by_rule": multi_rule,
            "n_rules_evaluated": len(rule_rates),
            "n_rules_above_50pct": sum(1 for r in rule_rates if r > 0.5),
            "min_win_rate": round(min(rule_rates), 4) if rule_rates else None,
            "max_win_rate": round(max(rule_rates), 4) if rule_rates else None,
        },
    }

    # 6. Triage Role-B R1 Disclosure & Exp-2 Zero-Contradiction Pool Inventory (Main arm Triage)
    models_tr = [m for m in ALL_PARTNER_MODELS if any(str(e["model_b"]) == m for e in tr_main)]
    models_tr += [m for m in sorted({str(e["model_b"]) for e in tr_main}) if m not in models_tr]

    tr_by_model: dict[str, Any] = {}
    pool_by_seed_total: collections.Counter[str] = collections.Counter()
    pool_le3_by_seed_total: collections.Counter[str] = collections.Counter()
    pool_0_by_seed_total: collections.Counter[str] = collections.Counter()
    all_zc_s1_ids: list[str] = []
    all_zc_s1_le3_ids: list[str] = []
    all_zc_s1_0_ids: list[str] = []
    all_cmd_counts: list[int] = []
    all_g_counts: list[int] = []
    all_zc_flags: list[bool] = []

    for m in models_tr:
        m_eps = [e for e in tr_main if str(e["model_b"]) == m]
        d_list = []
        zc_flags = []
        zc_s1_eps = []
        zc_s1_le3_eps = []
        zc_s1_0_eps = []
        for ep in m_eps:
            turns = sorted(ep.get("turns") or [], key=lambda t: int(t.get("round", 0)))
            r1_b_msg = str((turns[0].get("B") or {}).get("message") or "")
            td = check_triage_operator_disclosure(r1_b_msg)
            zc = is_triage_b_zero_contradiction(ep)
            d_list.append(td)
            zc_flags.append(zc)
            all_cmd_counts.append(td.n_cmd_ids)
            all_g_counts.append(td.n_guardrail_ids)
            all_zc_flags.append(zc)
            if zc and abs(float(ep["score"]) - 1.0) < 1e-9:
                sd_key = str(int(ep["seed"]))
                ep_id_str = str(ep["episode_id"])
                zc_s1_eps.append(ep)
                all_zc_s1_ids.append(ep_id_str)
                pool_by_seed_total[sd_key] += 1
                if td.n_cmd_ids <= 3:
                    zc_s1_le3_eps.append(ep)
                    all_zc_s1_le3_ids.append(ep_id_str)
                    pool_le3_by_seed_total[sd_key] += 1
                if td.n_cmd_ids == 0:
                    zc_s1_0_eps.append(ep)
                    all_zc_s1_0_ids.append(ep_id_str)
                    pool_0_by_seed_total[sd_key] += 1

        by_seed = collections.Counter(str(int(e["seed"])) for e in zc_s1_eps)
        by_seed_le3 = collections.Counter(str(int(e["seed"])) for e in zc_s1_le3_eps)
        tr_by_model[m] = {
            "n_role_b_episodes": len(m_eps),
            "r1_mean_cmd_ids": round(float(np.mean([d.n_cmd_ids for d in d_list])), 4) if d_list else None,
            "r1_le_3_cmds_count": sum(1 for d in d_list if d.n_cmd_ids <= 3),
            "r1_le_3_cmds_rate": round(float(np.mean([d.n_cmd_ids <= 3 for d in d_list])), 4)
            if d_list
            else None,
            "r1_mean_guardrail_ids": round(float(np.mean([d.n_guardrail_ids for d in d_list])), 4)
            if d_list
            else None,
            "r1_mentions_guardrail_substance_rate": round(
                float(np.mean([d.mentions_guardrail_substance for d in d_list])), 4
            )
            if d_list
            else None,
            "r1_asks_evidence_rate": round(
                float(np.mean([d.asks_partner_for_evidence for d in d_list])), 4
            )
            if d_list
            else None,
            "zero_contradiction_all_count": sum(zc_flags),
            "zero_contradiction_all_rate": round(float(np.mean(zc_flags)), 4) if zc_flags else None,
            "zero_contradiction_score_1_count": len(zc_s1_eps),
            "zero_contradiction_score_1_by_seed": dict(sorted(by_seed.items())),
            "zero_contradiction_score_1_episode_ids": sorted(str(e["episode_id"]) for e in zc_s1_eps),
            "zero_contradiction_score_1_le_3_cmds_count": len(zc_s1_le3_eps),
            "zero_contradiction_score_1_le_3_cmds_by_seed": dict(sorted(by_seed_le3.items())),
            "zero_contradiction_score_1_le_3_cmds_episode_ids": sorted(
                str(e["episode_id"]) for e in zc_s1_le3_eps
            ),
            "zero_contradiction_score_1_0_cmds_count": len(zc_s1_0_eps),
        }

    triage_exp2_pool = {
        "scope": "main_arm_triage_role_b_excluding_opus_boost",
        "n_triage_main_episodes": len(tr_main),
        "by_model": tr_by_model,
        "pooled": {
            "r1_mean_cmd_ids": round(float(np.mean(all_cmd_counts)), 4) if all_cmd_counts else None,
            "r1_le_3_cmds_rate": round(
                float(np.mean([c <= 3 for c in all_cmd_counts])), 4
            )
            if all_cmd_counts
            else None,
            "r1_mean_guardrail_ids": round(float(np.mean(all_g_counts)), 4) if all_g_counts else None,
            "zero_contradiction_all_count": sum(all_zc_flags),
            "zero_contradiction_all_rate": round(float(np.mean(all_zc_flags)), 4)
            if all_zc_flags
            else None,
            "zero_contradiction_score_1_total": len(all_zc_s1_ids),
            "zero_contradiction_score_1_by_seed": dict(sorted(pool_by_seed_total.items())),
            "zero_contradiction_score_1_le_3_cmds_total": len(all_zc_s1_le3_ids),
            "zero_contradiction_score_1_le_3_cmds_by_seed": dict(
                sorted(pool_le3_by_seed_total.items())
            ),
            "zero_contradiction_score_1_0_cmds_total": len(all_zc_s1_0_ids),
            "zero_contradiction_score_1_0_cmds_by_seed": dict(
                sorted(pool_0_by_seed_total.items())
            ),
            "ready_for_exp2_24_pairs": bool(
                len(all_zc_s1_le3_ids) >= 48
                and all(
                    pool_le3_by_seed_total.get(str(sd), 0) >= 12 for sd in (301, 302, 303, 304)
                )
            ),
        },
    }

    return {
        "ok": True,
        "wired": True,
        "status": "complete_with_disclosure",
        "scope": {
            "primary_episodes": "main_arm_excluding_opus_boost",
            "n_episodes_main_passed": len(main_eps),
            "n_episodes_excluded_from_main": n_excluded,
            "n_goalpost_main_episodes": len(gp_main),
            "n_goalpost_all_episodes": len(gp_all),
            "n_triage_main_episodes": len(tr_main),
        },
        "goalpost_disclosure_table_main": table_main,
        "goalpost_disclosure_table_all_incl_boost": table_all,
        "within_gpt62_eval_blindness": within_gpt62,
        "ask_without_giving_asymmetry": ask_without_giving,
        "contagious_withholding": contagious_withholding,
        "producer_consumer_hypocrisy": producer_consumer,
        "triage_role_b_disclosure_and_exp2_pool": triage_exp2_pool,
    }


def run_all_phase1_meta_analyses(
    run_dir: Path,
    *,
    include_disclosure: bool = True,
) -> dict[str, Any]:
    """Run all Phase 1 v2.1 meta-analyses end-to-end from raw ``data/v2/`` files."""
    raw = load_v2_raw_data(run_dir)
    cv = compute_construct_validity(raw["episodes_main"])
    clarity_resid_map = cv.pop("clarity_resid_map")
    cv["n_clarity_resid_mapped_turns"] = len(clarity_resid_map)

    logits = compute_choice_logits(
        raw["episodes_all"],
        raw["choices"],
        clarity_resid_map,
        capstone_choices=raw["capstone_choices"],
    )
    capstone = compute_capstone_audit(
        raw["capstone_rewrites"],
        raw["capstone_choices"],
        main_choices=raw["choices"],
    )
    bloat = compute_bloat_tax_and_contagion(
        raw["episodes_main"],
        raw["ledger_path"],
        choices=raw["choices"],
        episodes_all=raw["episodes_all"],
    )
    triage_upsets = compute_triage_paralysis_and_upsets(
        raw["episodes_main"],
        raw["choices"],
        ledger_path=raw["ledger_path"],
    )
    disclosure: dict[str, Any] = (
        compute_disclosure_meta_analysis(
            raw["episodes_main"],
            raw["choices"],
            episodes_all=raw["episodes_all"],
        )
        if include_disclosure
        else {
            "ok": False,
            "wired": False,
            "status": "skipped_by_caller",
        }
    )

    try:
        from common_ground.config import ROOT_DIR

        rel_run_dir = run_dir.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        rel_run_dir = run_dir.name

    return {
        "schema_version": "meta_v2_1",
        "data_inventory": {
            "run_dir": rel_run_dir,
            "n_episodes_complete_all": len(raw["episodes_all"]),
            "n_episodes_main": len(raw["episodes_main"]),
            "n_episodes_opus_boost": len(raw["episodes_boost"]),
            "episodes_incomplete": raw["episodes_incomplete"],
            "n_choices_dedup_all": len(raw["choices"]),
            "n_capstone_rewrites": len(raw["capstone_rewrites"]),
            "n_capstone_choices": len(raw["capstone_choices"]),
        },
        "construct_validity": cv,
        "choice_logits": logits,
        "capstone_audit": capstone,
        "bloat_tax_and_contagion": bloat,
        "triage_paralysis_and_upsets": triage_upsets,
        "disclosure_meta_analysis": disclosure,
    }


def render_meta_v2_1_markdown(report: dict[str, Any]) -> str:
    """Render a concise, publication-grade Markdown summary of ``meta_v2_1.json``."""
    inv = report["data_inventory"]
    cv = report["construct_validity"]
    lg = report["choice_logits"]
    cap = report["capstone_audit"]
    bl = report["bloat_tax_and_contagion"]
    tr = report["triage_paralysis_and_upsets"]
    disc = report["disclosure_meta_analysis"]

    lines: list[str] = [
        "# Mighty Models, Terrible Teammates — Phase 1 Meta-Analysis Report",
        "",
        f"- **Episodes:** `{inv['n_episodes_complete_all']}` complete (`{inv['n_episodes_main']}` main arm + `{inv['n_episodes_opus_boost']}` isolated `opus_boost`; incomplete excluded: `{len(inv['episodes_incomplete'])}`)",
        f"- **Choices:** `{inv['n_choices_dedup_all']}` total (`{lg['scope']['all_non_aa_n']}` non-AA, `{lg['scope']['main_7p_sensitivity_n']}` `main_7p`, `{lg['scope']['exact_ties_all_non_aa_n']}` exact-score ties)",
        f"- **Capstone:** `{inv['n_capstone_rewrites']}` rewrite runs, `{inv['n_capstone_choices']}` decisions",
        f"- **Disclosure hook:** `{disc.get('status', 'wired' if disc.get('wired') else 'pending')}`",
        "",
        "## 1. Construct Validity & Verbosity Confound (`main` arm only)",
        "",
        f"- OLS `R^2` (`~ log_wc + log_wc^2 + task + role`, `n = {cv['n_turns_with_clarity']}` turns):",
        f"  - `clarity`: **`R^2 = {cv['ols_length_models']['clarity']['r_squared']:.4f}`**",
        f"  - `skips_why`: **`R^2 = {cv['ols_length_models']['skips_why']['r_squared']:.4f}`**",
        f"  - `wooden_legalistic`: **`R^2 = {cv['ols_length_models']['wooden_legalistic']['r_squared']:.4f}`**",
        "",
        "| Model | Mean Words | Clarity Raw (Rank) | Clarity Resid (Rank) | Skips-Why Raw | Skips-Why Resid |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m, row in sorted(
        cv["per_model"].items(), key=lambda kv: kv[1].get("clarity_raw") or -999, reverse=True
    ):
        lines.append(
            f"| `{m}` | {row['mean_word_count']:.1f} | {row['clarity_raw']:.4f} (#{row.get('clarity_raw_rank')}) "
            f"| {row['clarity_resid']:+.4f} (#{row.get('clarity_length_adjusted_rank')}) "
            f"| {row['skips_why_raw']:.4f} | {row['skips_why_resid']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Visibility-Masked Cluster-Robust Choice Logits & Tone-Mention Audit",
            "",
            "| Predictor (1-SD delta) | Multivariate (No FE) | Multivariate (+ Partner FE) | Exact Ties (+ Partner FE) |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )
    mv_nofe = lg["multivariate"]["all_non_aa_no_fe"].get("coefs") or {}
    mv_fe = lg["multivariate"]["all_non_aa_with_fe"].get("coefs") or {}
    mv_tie = lg["multivariate"]["exact_ties_with_fe"].get("coefs") or {}

    def _fmt_cell(d: dict[str, Any], k: str) -> str:
        v = d.get(k)
        if not v:
            return "—"
        return f"`{v['coef']:+.3f}` (p={v['p_fmt']})"

    for pred in MULTIVARIATE_PREDICTORS:
        lines.append(
            f"| `{pred}` | {_fmt_cell(mv_nofe, pred)} | {_fmt_cell(mv_fe, pred)} | {_fmt_cell(mv_tie, pred)} |"
        )

    ta_live = lg["tone_and_rationale_audit"]["live_non_aa"]
    ta_cap = lg["tone_and_rationale_audit"].get("capstone") or {}
    lines.extend(
        [
            "",
            f"- **Zero-tolerance politeness/warmth mention audit (live non-AA):** **`{ta_live['strict_politeness_mentions']['count']}/{ta_live['n_reasons']}`** (`{ta_live['strict_politeness_mentions']['rate']*100:.2f}%`); terse/tone mentions: `{ta_live['terse_blunt_tone_mentions']['count']}/{ta_live['n_reasons']}`.",
        ]
    )
    if ta_cap:
        lines.append(
            f"- **Capstone tone-mention audit:** strict politeness `{ta_cap['strict_politeness_mentions']['count']}/{ta_cap['n_reasons']}`, terse/tone `{ta_cap['terse_blunt_tone_mentions']['count']}/{ta_cap['n_reasons']}`."
        )

    wb = cap["turn_level_deltas"]["warm_minus_blunt"]["dimensions"]
    cons_all = cap["inter_lab_consensus"]["all_pairs"]
    cons_tied = cap["inter_lab_consensus"]["exact_tied_pairs"]
    wb_line = (
        f"- **Turn-level Warm − Blunt deltas (`n = {cap['turn_level_deltas']['warm_minus_blunt']['n_paired_turns']}` paired turns):** "
        f"`warmth = {wb['warmth']['delta_warm_minus_blunt']:+.3f}`, "
        f"`clarity = {wb['clarity']['delta_warm_minus_blunt']:+.3f}`, "
        f"`skips_why = {wb['skips_why']['delta_warm_minus_blunt']:+.3f}`, "
        f"`wooden_legalistic = {wb['wooden_legalistic']['delta_warm_minus_blunt']:+.3f}` — while flipping "
        f"**`{cap['flip_verification']['unanimous_flips_toward_warm']}/{cap['flip_verification']['n_reviewer_pair_cases']}`** cases."
    )
    lines.extend(
        [
            "",
            "## 3. Capstone Multi-Dimension Audit & 4-Lab Consensus",
            "",
            wb_line,
            f"- **4-lab third-person consensus (all {cons_all.get('n_pairs', 0)} pairs):** `4-0` unanimous = `{cons_all.get('unanimous_4_0')}`, `3-1` strict = `{cons_all.get('three_vs_one_strict')}`, `3-0+split` = `{cons_all.get('three_zero_plus_one_split')}`, `2-2` = `{cons_all.get('two_vs_two')}` (mean pairwise exact decision agreement = `{cons_all.get('mean_pairwise_exact_decision_agree')}`).",
            f"- **4-lab consensus (10 exact-tied pairs):** `4-0` = `{cons_tied.get('unanimous_4_0')}`, `3-1` strict = `{cons_tied.get('three_vs_one_strict')}`, `3-0+split` = `{cons_tied.get('three_zero_plus_one_split')}`, `2-2` = `{cons_tied.get('two_vs_two')}`.",
        ]
    )
    if cap.get("live_main_7p_contrast") and cap["live_main_7p_contrast"].get("ok"):
        lc = cap["live_main_7p_contrast"]
        lines.append(
            f"- **Live `main_7p` inter-chooser Spearman contrast:** pooled mean pairwise `ρ = {lc['pooled']['mean_pairwise_spearman_rho']}`, within-task mean pairwise `ρ = {lc['mean_within_task_pairwise_spearman_rho']}`."
        )

    exp_overall = bl["exposed_turns_primary"]["overall"]["pct_delta_by_metric"]
    lines.extend(
        [
            "",
            "## 4. Bloat Tax & Verbosity Contagion (`exposed_turns`, main arm)",
            "",
            "| Partner | Prompt Tokens % | Completion Tokens % | Reasoning Tokens % | Latency % | Receiver Words % | Receiver Cost % |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    def _fmt_pct(metric: str, partner: str) -> str:
        d = exp_overall[metric]["by_partner"].get(partner)
        return f"{d['mean']:+.1f}%" if d and d.get("mean") is not None else "n/a"

    for p in ALL_PARTNER_MODELS:
        if p not in exp_overall["cost_usd"]["by_partner"]:
            continue
        lines.append(
            f"| `{p}` | {_fmt_pct('prompt_tokens', p)} | {_fmt_pct('completion_tokens', p)} | {_fmt_pct('reasoning_tokens', p)} "
            f"| {_fmt_pct('latency_ms', p)} | {_fmt_pct('word_count', p)} | {_fmt_pct('cost_usd', p)} |"
        )

    if bl.get("exact_tie_process_efficiency") and bl["exact_tie_process_efficiency"].get("ok"):
        te = bl["exact_tie_process_efficiency"]["overall"]
        fc_rate = (
            f"{te['faster_convergence']['win_rate'] * 100:.1f}%"
            if te["faster_convergence"]["win_rate"] is not None
            else "n/a"
        )
        sh_rate = (
            f"{te['shorter_partner']['win_rate'] * 100:.1f}%"
            if te["shorter_partner"]["win_rate"] is not None
            else "n/a"
        )
        ch_rate = (
            f"{te['cheaper_episode_cost']['win_rate'] * 100:.1f}%"
            if te["cheaper_episode_cost"]["win_rate"] is not None
            else "n/a"
        )
        eff_line = (
            f"- **Exact-tie process efficiency (`n = {te['n_ties_matched']}`):** "
            f"faster convergence `{fc_rate}` (`{te['faster_convergence']['wins']}/{te['faster_convergence']['n_unequal']}`), "
            f"shorter partner `{sh_rate}`, "
            f"cheaper episode cost `{ch_rate}`."
        )
        lines.extend(["", eff_line])

    neon_a = tr["neon_s303_s304_anomaly"]
    cal_m = tr["calendar_shortlist_and_2cell_enforcement"]["choice_reason_2cell_mentions"]
    opus_neon = neon_a["main_7p_neon_clean_vs_glitched_win_rate"].get("opus55") or {}
    triage_inv_line = (
        f"- **Triage `< 0.99` episodes (main arm):** `{tr['triage_below_099_inventory']['n_episodes']}` total; "
        f"`astra60 × astra62` `0.00` `6/6 WAIT` deadlocks = `{tr['triage_below_099_inventory']['astra60_astra62_00_deadlocks_count']}`; "
        f"`s304` `0.35` failures = `{tr['triage_below_099_inventory']['s304_035_failures_count']}`."
    )
    upset_line = (
        f"- **Score upsets (`chosen < other - 0.01`):** `{tr['score_upsets']['n_total_upsets']}` total "
        f"(`{tr['score_upsets']['by_task']}`); Opus `triage__s304` `0.35`-over-`1.00` receipts = `{len(tr['score_upsets']['opus_triage_s304_035_over_100_receipts'])}`."
    )
    cal_line = (
        f"- **Calendar 2-cell enforcement:** `{cal_m['total_mentions']}` Calendar choice reasons cite the 2-cell rule "
        f"(`opus55`: `{cal_m['by_subject'].get('opus55', 0)}`, "
        f"including `{cal_m['score_upsets_citing_2cell_by_subject'].get('opus55', 0)}` Opus score upsets)."
    )
    neon_line = (
        f"- **Neon multi-ACTION self-play glitch:** `{neon_a['neon_multi_action_total_turns']}` turns (`100%` unique to `neon4`, `s303`/`s304`: `{neon_a['neon_s303_s304_multi_action_count']}` turns); "
        f"Opus win rate for Neon drops from `{opus_neon.get('clean_win_rate')}` (clean) to `{opus_neon.get('glitched_win_rate')}` (glitched)."
    )
    lines.extend(
        [
            "",
            "## 5. Triage Paralysis, Score Upsets, Calendar 2-Cell Enforcement, & Neon Glitch",
            "",
            triage_inv_line,
            upset_line,
            cal_line,
            neon_line,
            "",
        ]
    )

    if disc.get("ok") and disc.get("wired"):
        gt_main = disc["goalpost_disclosure_table_main"]
        gt_all = disc["goalpost_disclosure_table_all_incl_boost"]
        awg_a = disc["ask_without_giving_asymmetry"]["role_a_blind_r1"]["by_model"]
        cont = disc["contagious_withholding"]
        hyp = disc["producer_consumer_hypocrisy"]
        eb_all = disc["within_gpt62_eval_blindness"]["overall"]
        eb_s1 = disc["within_gpt62_eval_blindness"]["score_eq_1_0_only"]
        tr_pool = disc["triage_role_b_disclosure_and_exp2_pool"]

        def _fpct(val: float | None) -> str:
            return f"{val * 100:.1f}%" if val is not None else "n/a"

        def _fnum(val: float | None, digits: int = 2) -> str:
            return f"{val:.{digits}f}" if val is not None else "n/a"

        lines.extend(
            [
                "## 6. Disclosure Meta-Analysis (`common_ground.disclosure`)",
                "",
                f"### 6.1 Goalpost R1 Disclosure Table (Primary: `main` arm `{gt_main['n_episodes']}` eps / `{gt_main['n_r1_turns']}` R1 turns; sensitivity incl. `opus_boost`: `{gt_all['n_episodes']}` eps)",
                "",
                "| Model | R1 Mean Rules | R1 >=5 All | R1 >=5 Role A | R1 >=5 Role B | Legacy ID >=5 | R1–R2 >=5 | R1–R8 >=5 | Role-A Ask-Without-Giving |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for m in ALL_PARTNER_MODELS:
            row = gt_main["by_model"].get(m)
            if not row:
                continue
            awg_r = (awg_a.get(m) or {}).get("asks_without_giving_rate")
            lines.append(
                f"| `{m}` | {_fnum(row['r1_mean_formula_rules'])} | {_fpct(row['r1_full_formula_rate'])} "
                f"| {_fpct(row['r1_role_a_full_formula_rate'])} | {_fpct(row['r1_role_b_full_formula_rate'])} "
                f"| {_fpct(row['r1_legacy_id_ge5_rate'])} | {_fpct(row['cum_r1_r2_full_formula_rate'])} "
                f"| {_fpct(row['cum_r1_r8_full_formula_rate'])} | {_fpct(awg_r)} |"
            )

        eb_s1_sh = eb_s1["shared"]
        eb_s1_wh = eb_s1["withheld"]
        eb_s1_ct = eb_s1["contrast_shared_minus_withheld"]
        eb_all_sh = eb_all["shared"]
        eb_all_wh = eb_all["withheld"]
        eb_line_all = (
            f"- **Within-GPT-6.2 (all main Goalpost runs):** Shared (`n={eb_all_sh['n_episodes']}`, score `{_fnum(eb_all_sh['mean_score'], 4)}`, ep words `{_fnum(eb_all_sh['mean_episode_word_count'], 1)}`) "
            f"vs. Withheld (`n={eb_all_wh['n_episodes']}`, score `{_fnum(eb_all_wh['mean_score'], 4)}`, ep words `{_fnum(eb_all_wh['mean_episode_word_count'], 1)}`): "
            f"CV choice win rate `{_fpct(eb_all_sh['cross_vendor_main_7p_choice_overall']['win_rate'])}` vs. `{_fpct(eb_all_wh['cross_vendor_main_7p_choice_overall']['win_rate'])}`; "
            f"exact ties `{_fpct(eb_all_sh['cross_vendor_main_7p_choice_exact_ties']['win_rate'])}` vs. `{_fpct(eb_all_wh['cross_vendor_main_7p_choice_exact_ties']['win_rate'])}`."
        )
        eb_line_s1 = (
            f"- **Within-GPT-6.2 (`score == 1.0` only):** Shared (`n={eb_s1_sh['n_episodes']}`, F1 `{_fnum(eb_s1_sh['cross_vendor_debriefs']['mean_f1_again'], 2)}`, F4 KEEP `{_fpct(eb_s1_sh['cross_vendor_debriefs']['f4_keep_rate'])}`) "
            f"vs. Withheld (`n={eb_s1_wh['n_episodes']}`, F1 `{_fnum(eb_s1_wh['cross_vendor_debriefs']['mean_f1_again'], 2)}`, F4 KEEP `{_fpct(eb_s1_wh['cross_vendor_debriefs']['f4_keep_rate'])}`): "
            f"CV choice overall `{_fpct(eb_s1_sh['cross_vendor_main_7p_choice_overall']['win_rate'])}` vs. `{_fpct(eb_s1_wh['cross_vendor_main_7p_choice_overall']['win_rate'])}` (Fisher p={eb_s1_ct['choice_overall']['fisher_exact_p_fmt']}, ep-perm p={eb_s1_ct['choice_overall']['episode_perm_p_fmt']}); "
            f"exact ties `{_fpct(eb_s1_sh['cross_vendor_main_7p_choice_exact_ties']['win_rate'])}` vs. `{_fpct(eb_s1_wh['cross_vendor_main_7p_choice_exact_ties']['win_rate'])}` (Fisher p={eb_s1_ct['choice_exact_ties']['fisher_exact_p_fmt']}, ep-perm p={eb_s1_ct['choice_exact_ties']['episode_perm_p_fmt']})."
        )

        cp = cont["pooled_all_role_b"]
        op_c = cont["role_b_opus55_transparent_vs_withholding_partners"]
        cont_line_1 = (
            f"- **Contagious withholding (pooled Role B, `n = {cp['n_after_role_a_disclosed'] + cp['n_after_role_a_withheld']}`):** "
            f"mean R1 formulas after Role A disclosed = `{_fnum(cp['mean_rules_after_role_a_disclosed'])}` vs. after Role A withheld = `{_fnum(cp['mean_rules_after_role_a_withheld'])}` "
            f"(MWU p={cp['p_mwu_fmt']}, perm p={cp['p_perm_fmt']})."
        )
        cont_line_2 = (
            f"- **Role-B `opus55` after transparent (`neon4/muse/astra60/opus55`, `n={op_c['n_transparent']}`) vs. withholding (`astra62/kimi31`, `n={op_c['n_withholding']}`) Role-A partners:** "
            f"`{_fnum(op_c['mean_rules_after_transparent'])}` ({_fpct(op_c['full_disclosure_rate_after_transparent'])} full) vs. "
            f"`{_fnum(op_c['mean_rules_after_withholding'])}` ({_fpct(op_c['full_disclosure_rate_after_withholding'])} full), MWU p={op_c['p_mwu_fmt']}, perm p={op_c['p_perm_fmt']}."
        )

        lines.extend(["", "### 6.2 Within-GPT-6.2 Eval Blindness & Contagious Withholding", "", eb_line_all, eb_line_s1, cont_line_1, cont_line_2, ""])

        hyp_all = hyp["all_non_aa_exact_ties"]
        mr = hyp["multi_rule_robustness_14_rules"]
        lines.extend(
            [
                "### 6.3 Producer–Consumer Hypocrisy & 14-Rule Exact-Tie Robustness",
                "",
                "| Chooser | Producer Role-A R1 >=5 | Consumer Hires Discloser (Exact Ties) | Gap (pp) | Binom p | Case-Perm p |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for subj in CORE_MODELS:
            hs = hyp_all["by_chooser"].get(subj)
            if not hs:
                continue
            gap_s = f"{hs['hypocrisy_gap_pp_consumer_minus_producer_role_a']:+.1f}" if hs["hypocrisy_gap_pp_consumer_minus_producer_role_a"] is not None else "n/a"
            lines.append(
                f"| `{subj}` | {_fpct(hs['producer_role_a_full_disclosure_rate'])} "
                f"| {_fpct(hs['win_rate'])} (`{hs['wins']}/{hs['n_discordant_ties']}`) "
                f"| {gap_s} | {hs['binom_p_fmt']} | {hs['case_perm_p_fmt']} |"
            )

        mr_line = (
            f"- **14-rule robustness (`C-A1..C-A7`, `C-B1..C-B7`):** `{mr['n_rules_above_50pct']}/{mr['n_rules_evaluated']}` rules favor the discloser on exact ties "
            f"(win-rate range `{_fpct(mr['min_win_rate'])}`–`{_fpct(mr['max_win_rate'])}`)."
        )
        tr_p = tr_pool["pooled"]
        tr_line = (
            f"- **Triage Role-B Exp-2 zero-contradiction pool (`main` arm):** `{tr_p['zero_contradiction_all_count']}/{tr_pool['n_triage_main_episodes']}` zero-contradiction overall; "
            f"`{tr_p['zero_contradiction_score_1_total']}` with `score == 1.0` (`{tr_p['zero_contradiction_score_1_by_seed']}`); "
            f"**intersected withheld pool (`<= 3 CMDs & score == 1.0 & zero_contradiction`): `{tr_p['zero_contradiction_score_1_le_3_cmds_total']}`** "
            f"(`{tr_p['zero_contradiction_score_1_le_3_cmds_by_seed']}`, strict `0 CMDs`: `{tr_p['zero_contradiction_score_1_0_cmds_total']}`, "
            f"`ready_for_exp2_24_pairs={tr_p['ready_for_exp2_24_pairs']}`)."
        )
        lines.extend(["", mr_line, "", "### 6.4 Triage Role-B Disclosure & Exp-2 Pool Inventory", "", tr_line, ""])
        lines.extend(
            [
                "| Role-B Operator | Mean R1 CMDs | <=3 CMDs Rate | Mean R1 Guardrail IDs | Zero-Contradiction All | Zero-Contradiction `score==1.0` (All) | Withheld Exp-2 Pool (`<=3 CMDs & score==1.0 & ZC`) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for m in ALL_PARTNER_MODELS:
            row = tr_pool["by_model"].get(m)
            if not row:
                continue
            lines.append(
                f"| `{m}` | {_fnum(row['r1_mean_cmd_ids'])} | {_fpct(row['r1_le_3_cmds_rate'])} "
                f"| {_fnum(row['r1_mean_guardrail_ids'])} | `{row['zero_contradiction_all_count']}/{row['n_role_b_episodes']}` "
                f"| `{row['zero_contradiction_score_1_count']}` (`{row['zero_contradiction_score_1_by_seed']}`) "
                f"| `{row['zero_contradiction_score_1_le_3_cmds_count']}` (`{row['zero_contradiction_score_1_le_3_cmds_by_seed']}`) |"
            )
        lines.append("")

    return "\n".join(lines)
