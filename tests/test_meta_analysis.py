"""Unit tests for Phase 1 v2.1 meta-analysis library (strictly synthetic fixtures, $0 API)."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from common_ground.meta_analysis import (
    audit_choice_reasons,
    compute_bloat_tax_and_contagion,
    compute_capstone_audit,
    compute_choice_logits,
    compute_construct_validity,
    compute_disclosure_meta_analysis,
    compute_triage_paralysis_and_upsets,
    render_meta_v2_1_markdown,
    run_all_phase1_meta_analyses,
)
from common_ground.tasks.goalpost_task import GoalpostEnv


def _make_turn_side(
    *,
    model: str,
    message: str,
    call_id: str,
    clarity: float,
    warmth: float = 2.4,
    skips_why: float = 0.25,
    answers_directly: float = 0.7,
    wooden_legalistic: float = 0.3,
    condescending_pedantic: float = 0.08,
    expresses_confusion: float = 0.05,
    ids_per_100: float = 2.0,
    pleasantry_count: int = 1,
    question_marks: int = 0,
    action: dict[str, Any] | None = None,
    word_count: int | None = None,
) -> dict[str, Any]:
    wc = word_count if word_count is not None else len(message.split())
    return {
        "model": model,
        "handle": "AL",
        "message": message,
        "action": action or {},
        "parse_error": None,
        "accepted_call_id": call_id,
        "metrics": {
            "char_len": len(message),
            "word_count": wc,
            "ids_per_100": ids_per_100,
            "pleasantry_count": pleasantry_count,
            "question_marks": question_marks,
        },
        "jev_demeanor": {
            "clarity": clarity,
            "warmth": warmth,
            "skips_why": skips_why,
            "answers_directly": answers_directly,
            "wooden_legalistic": wooden_legalistic,
            "condescending_pedantic": condescending_pedantic,
            "expresses_confusion": expresses_confusion,
            "panel_judges": {
                "liquid": {"scores": {"clarity": clarity + 0.05}},
                "perplexity": {"scores": {"clarity": clarity}},
                "typesafe": {"scores": {"clarity": clarity - 0.05}},
            },
        },
        "reasoning_tokens": 100,
        "completion_tokens": 200,
        "prompt_tokens": 1000,
        "cost_usd": 0.01,
    }


def _make_episode(
    *,
    episode_id: str,
    task: str,
    seed: int,
    arm: str,
    model_a: str,
    model_b: str,
    score: float,
    turns: list[dict[str, Any]],
    score_details: dict[str, Any] | None = None,
    complete: bool = True,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "task": task,
        "seed": seed,
        "arm": arm,
        "total_rounds": len(turns),
        "model_a": model_a,
        "model_b": model_b,
        "score": score,
        "score_details": score_details or {},
        "turns": turns,
        "complete": complete,
    }


def test_construct_validity_excludes_boost_and_reorders_length_adjusted_clarity() -> None:
    """Verbose qwen4 ranks #1 raw clarity but drops below concise opus55 after length adjustment."""
    rng = np.random.default_rng(42)
    episodes: list[dict[str, Any]] = []
    # Model length & true length-adjusted clarity offsets
    specs = {
        "opus55": (110, +0.45),
        "neon4": (150, +0.10),
        "astra62": (70, -0.05),
        "qwen4": (340, -0.55),
    }
    models = list(specs.keys())
    for task_idx, task in enumerate(("calendar", "goalpost")):
        task_off = 0.2 if task == "goalpost" else 0.0
        for i in range(6):
            ma = models[i % len(models)]
            mb = models[(i + 1) % len(models)]
            turns = []
            for r in (1, 2, 3):
                t_obj: dict[str, Any] = {"round": r, "round_metrics": {}}
                for role, m in (("A", ma), ("B", mb)):
                    base_wc, true_off = specs[m]
                    wc = max(25, int(base_wc * rng.uniform(0.55, 1.65)))
                    lwc = math.log1p(wc)
                    # Strong positive verbosity slope so qwen4 wins raw clarity despite negative offset
                    clarity = 0.5 + 0.95 * lwc + task_off + true_off + float(rng.normal(0, 0.03))
                    skips = max(0.0, 1.2 - 0.18 * lwc - 0.3 * true_off + float(rng.normal(0, 0.02)))
                    wooden = max(0.0, 0.9 - 0.10 * lwc + float(rng.normal(0, 0.02)))
                    msg = "### Plan\n- item one\n- item two\n" + ("word " * wc)
                    t_obj[role] = _make_turn_side(
                        model=m,
                        message=msg,
                        call_id=f"{task}_{i}_{r}_{role}",
                        clarity=clarity,
                        skips_why=skips,
                        wooden_legalistic=wooden,
                        word_count=wc,
                    )
                turns.append(t_obj)
            episodes.append(
                _make_episode(
                    episode_id=f"{task}__s{301 + i}__{ma}__vs__{mb}",
                    task=task,
                    seed=301 + i,
                    arm="core",
                    model_a=ma,
                    model_b=mb,
                    score=1.0,
                    turns=turns,
                )
            )

    # Inject an extreme opus_boost episode that MUST be strictly excluded
    boost_turn = {
        "round": 1,
        "A": _make_turn_side(
            model="opus55", message="x", call_id="boost_1_A", clarity=99.0, word_count=5
        ),
        "B": _make_turn_side(
            model="astra62", message="x", call_id="boost_1_B", clarity=-99.0, word_count=5
        ),
    }
    episodes.append(
        _make_episode(
            episode_id="calendar__s305__opus55__vs__astra62",
            task="calendar",
            seed=305,
            arm="opus_boost",
            model_a="opus55",
            model_b="astra62",
            score=1.0,
            turns=[boost_turn],
        )
    )

    res = compute_construct_validity(episodes)
    assert res["ok"] is True
    assert res["n_episodes_main"] == 12
    assert res["n_episodes_excluded"] == 1
    assert res["ols_length_models"]["clarity"]["r_squared"] > 0.70
    assert res["correlations"]["clarity"]["log_wc"]["pearson_r"] > 0.50
    assert res["correlations"]["clarity"]["word_count"]["spearman_rho"] > 0.50
    assert res["per_model"]["qwen4"]["clarity_raw_rank"] == 1
    assert res["per_model"]["opus55"]["clarity_length_adjusted_rank"] == 1
    assert res["per_model"]["qwen4"]["clarity_length_adjusted_rank"] == 4
    assert set(res["per_judge_clarity"].keys()) == {"liquid", "perplexity", "typesafe"}
    # clarity_resid_map contains only main-arm turns (12 eps * 3 rounds * 2 roles = 72)
    cmap = res["clarity_resid_map"]
    assert len(cmap) == 72
    assert ("calendar__s305__opus55__vs__astra62", 1, "A") not in cmap
    assert abs(float(np.mean(list(cmap.values())))) < 1e-6


def test_choice_logits_visibility_mask_no_singular_warning_and_tone_audit() -> None:
    """Verify visibility masking, zero SingularMatrixWarning, out-of-sample boost resid, & tone audit."""
    rng = np.random.default_rng(2026)
    partners = ["opus55", "neon4", "astra62", "kimi31"]
    subjects = ["opus55", "neon4", "astra62", "kimi31"]
    tasks = ["calendar", "triage"]

    episodes_all: list[dict[str, Any]] = []
    ep_meta: dict[str, dict[str, Any]] = {}
    ep_counter = 0
    for task in tasks:
        for subj in subjects:
            for part in partners:
                for seed in (301, 302):
                    for srole in ("A", "B"):
                        ep_counter += 1
                        ma, mb = (subj, part) if srole == "A" else (part, subj)
                        arm = "opus_boost" if (subj == "opus55" and seed == 302 and part == "astra62") else "core"
                        ans_dir = float(np.clip(rng.uniform(0.2, 0.95), 0.1, 0.99))
                        score = 1.0 if (ep_counter % 2 == 0) else round(float(rng.choice([0.8, 0.9, 1.0])), 2)
                        turns = []
                        for r in (1, 2, 3):
                            wc_a = int(rng.integers(60, 220))
                            wc_b = int(rng.integers(60, 220))
                            # Put an outrageous unseen value in Role B's FINAL round (r == 3)
                            # to prove visibility masking ignores Partner B's unseen round 3 for Role A!
                            b_ans = 999.0 if (srole == "A" and r == 3) else ans_dir
                            turns.append(
                                {
                                    "round": r,
                                    "A": _make_turn_side(
                                        model=ma,
                                        message="msg " * wc_a,
                                        call_id=f"c_{ep_counter}_{r}_A",
                                        clarity=1.5 + 0.3 * math.log1p(wc_a) + float(rng.normal(0, 0.2)),
                                        warmth=float(rng.uniform(1.8, 3.2)),
                                        skips_why=float(rng.uniform(0.1, 0.7)),
                                        answers_directly=ans_dir if srole == "B" else 0.6,
                                        wooden_legalistic=float(rng.uniform(0.1, 0.7)),
                                        condescending_pedantic=float(rng.uniform(0.02, 0.25)),
                                        expresses_confusion=float(rng.uniform(0.0, 0.3)),
                                        ids_per_100=float(rng.uniform(0.5, 4.0)),
                                        word_count=wc_a,
                                    ),
                                    "B": _make_turn_side(
                                        model=mb,
                                        message="msg " * wc_b,
                                        call_id=f"c_{ep_counter}_{r}_B",
                                        clarity=1.5 + 0.3 * math.log1p(wc_b) + float(rng.normal(0, 0.2)),
                                        warmth=float(rng.uniform(1.8, 3.2)),
                                        skips_why=float(rng.uniform(0.1, 0.7)),
                                        answers_directly=b_ans,
                                        wooden_legalistic=float(rng.uniform(0.1, 0.7)),
                                        condescending_pedantic=float(rng.uniform(0.02, 0.25)),
                                        expresses_confusion=float(rng.uniform(0.0, 0.3)),
                                        ids_per_100=float(rng.uniform(0.5, 4.0)),
                                        word_count=wc_b,
                                    ),
                                }
                            )
                        ep_id = f"{task}__s{seed}__{ma}__vs__{mb}__{ep_counter}"
                        ep = _make_episode(
                            episode_id=ep_id,
                            task=task,
                            seed=seed,
                            arm=arm,
                            model_a=ma,
                            model_b=mb,
                            score=score,
                            turns=turns,
                        )
                        episodes_all.append(ep)
                        ep_meta[ep_id] = {
                            "task": task,
                            "subject": subj,
                            "partner": part,
                            "role": srole,
                            "score": score,
                            "ans_dir": ans_dir,
                            "arm": arm,
                        }

    eps_main = [e for e in episodes_all if e["arm"] != "opus_boost"]
    cv = compute_construct_validity(eps_main)
    cmap = cv["clarity_resid_map"]

    # Build synthetic choices where true logit strongly favors higher score + higher visible answers_directly
    choices: list[dict[str, Any]] = []
    by_str: dict[tuple[str, str, str], list[str]] = {}
    for eid, info in ep_meta.items():
        by_str.setdefault((info["subject"], info["task"], info["role"]), []).append(eid)

    cid_num = 0
    for (subj, task, srole), eids in sorted(by_str.items()):
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                e1, e2 = eids[i], eids[j]
                m1, m2 = ep_meta[e1], ep_meta[e2]
                if m1["partner"] == m2["partner"]:
                    # Add as aa_placebo to verify strict exclusion
                    cid_num += 1
                    choices.append(
                        {
                            "manifest_item_id": f"aa_{cid_num}",
                            "choice_type": "aa_placebo",
                            "subject_model": subj,
                            "task": task,
                            "role": srole,
                            "partner_1": m1["partner"],
                            "partner_2": m2["partner"],
                            "episode_1_id": e1,
                            "episode_2_id": e2,
                            "score_1": m1["score"],
                            "score_2": m2["score"],
                            "chosen_slot": 1,
                            "chosen_partner": m1["partner"],
                            "reason": "Warm polite placeholder that must be excluded!",
                        }
                    )
                    continue
                cid_num += 1
                lin = 6.0 * (m1["score"] - m2["score"]) + 4.5 * (m1["ans_dir"] - m2["ans_dir"])
                prob = 1.0 / (1.0 + math.exp(-lin))
                slot = 1 if rng.random() < prob else 2
                ctype = (
                    "opus_boost_h2h"
                    if ("opus_boost" in (m1["arm"], m2["arm"]))
                    else ("extra_h2h" if cid_num % 7 == 0 else "main_7p")
                )
                # Include domain phrase "cold-start" to confirm zero false positives in strict tone audit
                reason = (
                    "Teammate shared exact constraints upfront and avoided cold-start regressions."
                    if cid_num % 10 != 0
                    else "Partner gave terse responses and omitted requested invariants."
                )
                choices.append(
                    {
                        "manifest_item_id": f"ch_{cid_num}",
                        "choice_type": ctype,
                        "subject_model": subj,
                        "task": task,
                        "role": srole,
                        "partner_1": m1["partner"],
                        "partner_2": m2["partner"],
                        "episode_1_id": e1,
                        "episode_2_id": e2,
                        "score_1": m1["score"],
                        "score_2": m2["score"],
                        "chosen_slot": slot,
                        "chosen_partner": m1["partner"] if slot == 1 else m2["partner"],
                        "reason": reason,
                    }
                )

    with warnings.catch_warnings():
        warnings.simplefilter("error", SingularMatrixWarning)
        res = compute_choice_logits(episodes_all, choices, cmap)

    assert res["ok"] is True
    assert res["scope"]["aa_placebo_excluded"] > 0
    assert res["scope"]["n_turns_oos_clarity_resid_predicted"] > 0
    # Verify visibility masking worked: d_answers_directly raw SD is normal (~0.3), NOT polluted by 999.0!
    mv_fe = res["multivariate"]["all_non_aa_with_fe"]
    assert mv_fe["ok"] is True
    assert mv_fe["predictor_raw_sd"]["d_answers_directly"] < 1.0
    assert mv_fe["coefs"]["d_answers_directly"]["coef"] > 0.5
    assert mv_fe["coefs"]["d_answers_directly"]["p"] < 0.01
    assert mv_fe["coefs"]["d_score"]["coef"] > 0.3

    # Exact ties + FE should still recover strong positive d_answers_directly!
    tie_fe = res["multivariate"]["exact_ties_with_fe"]
    assert tie_fe["ok"] is True
    assert tie_fe["coefs"]["d_answers_directly"]["coef"] > 0.5

    # Tone-mention audit: 0 strict politeness mentions (cold-start filtered out!), >0 terse mentions
    ta = res["tone_and_rationale_audit"]["live_non_aa"]
    assert ta["strict_politeness_mentions"]["count"] == 0
    assert ta["terse_blunt_tone_mentions"]["count"] > 0
    assert ta["rationale_themes"]["all"]["themes"]["transparency_and_disclosure"]["count"] > 0


def test_audit_choice_reasons_catches_real_politeness_and_ignores_domain_phrases() -> None:
    rows = [
        {
            "manifest_item_id": "1",
            "subject_model": "opus55",
            "task": "triage",
            "chosen_partner": "neon4",
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 1,
            "reason": "Fixed cold-start latency and warm cache misses cleanly in Round 3.",
        },
        {
            "manifest_item_id": "2",
            "subject_model": "opus55",
            "task": "code_handoff",
            "chosen_partner": "astra62",
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 1,
            "reason": "Avoided padding and repeated pleasantries while shipping a round faster.",
        },
    ]
    out = audit_choice_reasons(rows)
    assert out["strict_politeness_mentions"]["count"] == 1
    assert out["strict_politeness_mentions"]["term_counts"] == {"pleasantries": 1}


def test_capstone_audit_deltas_and_4lab_consensus() -> None:
    """Verify Warm - Blunt multi-dim deltas, 0 flips, 4-0/3-1/2-2 consensus, & live Spearman contrast."""
    rewrites: list[dict[str, Any]] = []
    for ep_idx in range(4):
        ep_id = f"ep_{ep_idx}"
        for tone in ("warm", "blunt"):
            turns = []
            for rnd in (1, 2):
                orig = {
                    "warmth": 2.2,
                    "clarity": 2.5,
                    "skips_why": 0.35,
                    "wooden_legalistic": 0.40,
                    "answers_directly": 0.70,
                    "condescending_pedantic": 0.08,
                }
                if tone == "warm":
                    rew = {
                        "warmth": 2.8,
                        "clarity": 2.6,
                        "skips_why": 0.31,
                        "wooden_legalistic": 0.34,
                        "answers_directly": 0.71,
                        "condescending_pedantic": 0.06,
                    }
                    nw, lr = 110, 1.10
                else:
                    rew = {
                        "warmth": 1.7,
                        "clarity": 2.3,
                        "skips_why": 0.44,
                        "wooden_legalistic": 0.50,
                        "answers_directly": 0.67,
                        "condescending_pedantic": 0.11,
                    }
                    nw, lr = 90, 0.90
                turns.append(
                    {
                        "round": rnd,
                        "orig_words": 100,
                        "new_words": nw,
                        "length_ratio": lr,
                        "dem_orig": orig,
                        "dem_rew": rew,
                    }
                )
            rewrites.append({"episode_id": ep_id, "target_tone": tone, "turns": turns})

    # 4 pairs x 4 reviewers x 2 conditions x 2 orders = 64 decisions:
    # pair_1 (tied): 4-0 unanimous P1
    # pair_2 (tied): 3-1 strict dissent (astra62 votes P2, others P1)
    # pair_3 (tied): 3-0 + 1 order-split (kimi31 splits AB=P1, BA=P2)
    # pair_4 (moderate): 2-2 tie (opus55 & neon4 P1; astra62 & kimi31 P2)
    reviewers = ("opus55", "neon4", "astra62", "kimi31")
    pair_specs = [
        ("pair_1", "tied", {"opus55": "P1", "neon4": "P1", "astra62": "P1", "kimi31": "P1"}),
        ("pair_2", "tied", {"opus55": "P1", "neon4": "P1", "astra62": "P2", "kimi31": "P1"}),
        ("pair_3", "tied", {"opus55": "P2", "neon4": "P2", "astra62": "P2", "kimi31": "SPLIT"}),
        ("pair_4", "moderate", {"opus55": "P1", "neon4": "P1", "astra62": "P2", "kimi31": "P2"}),
    ]
    cap_choices: list[dict[str, Any]] = []
    for pid, gbin, vote_map in pair_specs:
        for rev in reviewers:
            v = vote_map[rev]
            for cond in ("p1_warm_p2_blunt", "p1_blunt_p2_warm"):
                for order in ("AB", "BA"):
                    chosen = ("P1" if order == "AB" else "P2") if v == "SPLIT" else v
                    tone_chosen = (
                        "warm"
                        if ((cond == "p1_warm_p2_blunt" and chosen == "P1") or (cond == "p1_blunt_p2_warm" and chosen == "P2"))
                        else "blunt"
                    )
                    cap_choices.append(
                        {
                            "packet_id": f"{pid}__{cond}__{order}__{rev}",
                            "pair_id": pid,
                            "task": "calendar",
                            "gap_bin": gbin,
                            "condition": cond,
                            "order": order,
                            "reviewer": rev,
                            "chosen_canonical": chosen,
                            "chosen_partner": "opus55" if chosen == "P1" else "kimi31",
                            "chosen_tone": tone_chosen,
                            "score_1": 1.0,
                            "score_2": 1.0 if gbin == "tied" else 0.8,
                            "chosen_slot": 1 if chosen == "P1" else 2,
                            "raw": json.dumps({"choice": "AL", "reason": "Clear schedules and exact totals."}),
                        }
                    )

    # Synthetic live main_7p choices for inter-chooser Spearman contrast
    main_choices: list[dict[str, Any]] = []
    partner_order = ["opus55", "neon4", "astra62", "kimi31"]
    for subj in reviewers:
        for i, p1 in enumerate(partner_order):
            for p2 in partner_order[i + 1 :]:
                main_choices.append(
                    {
                        "choice_type": "main_7p",
                        "subject_model": subj,
                        "task": "calendar",
                        "role": "A",
                        "partner_1": p1,
                        "partner_2": p2,
                        "chosen_slot": 1,
                        "chosen_partner": p1,
                    }
                )

    res = compute_capstone_audit(rewrites, cap_choices, main_choices=main_choices)
    wb = res["turn_level_deltas"]["warm_minus_blunt"]["dimensions"]
    assert wb["warmth"]["delta_warm_minus_blunt"] == pytest.approx(1.1, abs=1e-4)
    assert wb["clarity"]["delta_warm_minus_blunt"] == pytest.approx(0.3, abs=1e-4)
    assert wb["skips_why"]["delta_warm_minus_blunt"] == pytest.approx(-0.13, abs=1e-4)
    assert wb["wooden_legalistic"]["delta_warm_minus_blunt"] == pytest.approx(-0.16, abs=1e-4)

    assert res["flip_verification"]["unanimous_flips_toward_warm"] == 0
    assert res["flip_verification"]["unanimous_flips_away_from_warm"] == 0

    cons_all = res["inter_lab_consensus"]["all_pairs"]
    assert cons_all["unanimous_4_0"] == 1
    assert cons_all["three_vs_one_strict"] == 1
    assert cons_all["three_zero_plus_one_split"] == 1
    assert cons_all["two_vs_two"] == 1
    assert len(cons_all["pairwise"]) == 6

    cons_tied = res["inter_lab_consensus"]["exact_tied_pairs"]
    assert cons_tied["n_pairs"] == 3
    assert cons_tied["unanimous_4_0"] == 1
    assert cons_tied["two_vs_two"] == 0

    assert res["live_main_7p_contrast"]["ok"] is True
    assert res["live_main_7p_contrast"]["pooled"]["mean_pairwise_spearman_rho"] == pytest.approx(1.0)


def test_bloat_tax_contagion_and_exact_tie_efficiency(tmp_path: Path) -> None:
    """Verify exposed_turns vs all_turns, zero-rtok block exclusion, billing components, & tie efficiency."""
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    # 3 blocks: opus55 (A & B) and astra62 (B, with ZERO reasoning_tokens across the whole block!)
    partners = ["qwen4", "kimi31"]
    for subj, role, zero_rt in (("opus55", "A", False), ("opus55", "B", False), ("astra62", "B", True)):
        for seed in (301, 302):
            for part in partners:
                ma, mb = (subj, part) if role == "A" else (part, subj)
                ep_id = f"triage__s{seed}__{ma}__vs__{mb}"
                mult = 1.5 if part == "qwen4" else 0.5  # +50% for qwen4, -50% for kimi31!
                turns = []
                for rnd in (1, 2):
                    t_obj: dict[str, Any] = {"round": rnd, "round_metrics": {}}
                    for r_side, m_side in (("A", ma), ("B", mb)):
                        cid = f"v2:{ep_id}:turn:r{rnd}:{r_side}"
                        is_subj = r_side == role
                        # Put huge pre-exposure Round 1 noise on Role A (should NOT affect exposed_turns!)
                        if is_subj and role == "A" and rnd == 1:
                            wc = 1000 if part == "kimi31" else 100
                            rt = 0 if zero_rt else (2000 if part == "kimi31" else 200)
                            pt, ct, lat, cost = 5000, 3000, 9000, 0.50
                        elif is_subj:
                            wc = int(100 * mult)
                            rt = 0 if zero_rt else int(400 * mult)
                            pt = int(2000 * mult)
                            ct = rt + int(200 * mult)
                            lat = int(4000 * mult)
                            cost = 0.04 * mult
                        else:
                            wc = 200 if part == "qwen4" else 60
                            rt, pt, ct, lat, cost = 100, 1000, 300, 2000, 0.02 if part == "qwen4" else 0.005
                        t_obj[r_side] = _make_turn_side(
                            model=m_side,
                            message="w " * wc,
                            call_id=cid,
                            clarity=2.5,
                            expresses_confusion=0.2 if (is_subj and part == "qwen4") else 0.0,
                            question_marks=2 if (is_subj and part == "qwen4") else 0,
                            word_count=wc,
                        )
                        ledger_rows.append(
                            {
                                "call_id": cid,
                                "status": "ok",
                                "prompt_tokens": pt,
                                "completion_tokens": ct,
                                "reasoning_tokens": rt,
                                "latency_ms": lat,
                                "cost_usd": cost,
                                "metadata": {
                                    "stage": "turn",
                                    "episode_id": ep_id,
                                    "round": rnd,
                                    "role": r_side,
                                },
                            }
                        )
                    turns.append(t_obj)
                episodes.append(
                    _make_episode(
                        episode_id=ep_id,
                        task="triage",
                        seed=seed,
                        arm="core",
                        model_a=ma,
                        model_b=mb,
                        score=1.0,
                        turns=turns,
                        score_details={"first_verified_round": 2 if part == "qwen4" else 1},
                    )
                )

    # Add an ignored error row + non-turn row to ledger
    ledger_rows.append({"call_id": "ignored_err", "status": "error", "metadata": {"stage": "turn"}})
    ledger_rows.append({"call_id": "ignored_debrief", "status": "ok", "metadata": {"stage": "F1"}})
    ledger_path.write_text("\n".join(json.dumps(r) for r in ledger_rows) + "\n", encoding="utf-8")

    # Exact-tie choice where faster-converging, shorter, cheaper kimi31 beats qwen4
    tie_choices = [
        {
            "choice_type": "main_7p",
            "subject_model": "opus55",
            "task": "triage",
            "role": "B",
            "partner_1": "qwen4",
            "partner_2": "kimi31",
            "episode_1_id": "triage__s301__qwen4__vs__opus55",
            "episode_2_id": "triage__s301__kimi31__vs__opus55",
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 2,
            "chosen_partner": "kimi31",
        },
        {
            "choice_type": "main_7p",
            "subject_model": "opus55",
            "task": "triage",
            "role": "A",
            "partner_1": "qwen4",
            "partner_2": "kimi31",
            "episode_1_id": "triage__s302__opus55__vs__qwen4",
            "episode_2_id": "triage__s302__opus55__vs__kimi31",
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 2,
            "chosen_partner": "kimi31",
        },
    ]

    res = compute_bloat_tax_and_contagion(episodes, ledger_path, choices=tie_choices)
    assert res["ok"] is True
    assert res["ledger_join"]["match_rate"] == 1.0

    # Exposed turns MUST recover exact +50% for qwen4 and -50% for kimi31 (unaffected by R1 Role-A noise!)
    exp_pct = res["exposed_turns_primary"]["overall"]["pct_delta_by_metric"]
    assert exp_pct["word_count"]["by_partner"]["qwen4"]["mean"] == pytest.approx(50.0, abs=1e-4)
    assert exp_pct["word_count"]["by_partner"]["kimi31"]["mean"] == pytest.approx(-50.0, abs=1e-4)
    assert exp_pct["prompt_tokens"]["by_partner"]["qwen4"]["mean"] == pytest.approx(50.0, abs=1e-4)
    assert exp_pct["cost_usd"]["by_partner"]["qwen4"]["mean"] == pytest.approx(50.0, abs=1e-4)
    # Zero-rtok astra62 blocks (2 blocks) cleanly excluded!
    assert exp_pct["reasoning_tokens"]["n_zero_blocks_excluded"] == 2
    assert exp_pct["reasoning_tokens"]["by_partner"]["qwen4"]["mean"] == pytest.approx(50.0, abs=1e-4)

    # Exact-tie efficiency: kimi31 won 2/2 faster convergence, shorter partner, cheaper episode
    eff = res["exact_tie_process_efficiency"]["overall"]
    assert eff["faster_convergence"]["win_rate"] == pytest.approx(1.0)
    assert eff["shorter_partner"]["win_rate"] == pytest.approx(1.0)
    assert eff["cheaper_episode_cost"]["win_rate"] == pytest.approx(1.0)


def test_triage_paralysis_upsets_shortlist_and_neon_anomaly(tmp_path: Path) -> None:
    """Verify Triage 7x7 grid, 0.00 deadlocks, s304 Opus upsets, shortlist loophole, & Neon multi-ACTION glitch."""
    ledger_path = tmp_path / "ledger.jsonl"

    # 1. Triage deadlock (0.00, 6/6 WAIT) + s304 0.35 failure + clean 1.00 episode
    deadlock_turns = [
        {
            "round": r,
            "A": _make_turn_side(model="astra60", message="checking probes", call_id=f"dl_{r}_A", clarity=2.5),
            "B": _make_turn_side(
                model="astra62", message="waiting", call_id=f"dl_{r}_B", clarity=2.5, action={"command": "WAIT"}
            ),
        }
        for r in range(1, 7)
    ]
    ep_deadlock = _make_episode(
        episode_id="triage__s303__astra60__vs__astra62",
        task="triage",
        seed=303,
        arm="partner_only",
        model_a="astra60",
        model_b="astra62",
        score=0.0,
        turns=deadlock_turns,
        score_details={"verified_green": False, "wait_rounds": 6, "guardrail_violations": 0},
    )
    ep_s304_035 = _make_episode(
        episode_id="triage__s304__astra62__vs__opus55",
        task="triage",
        seed=304,
        arm="core",
        model_a="astra62",
        model_b="opus55",
        score=0.35,
        turns=[
            {
                "round": 1,
                "A": _make_turn_side(model="astra62", message="calibrated evidence", call_id="s304_1_A", clarity=2.5),
                "B": _make_turn_side(model="opus55", message="holding", call_id="s304_1_B", clarity=3.0, action={"command": "WAIT"}),
            }
        ],
        score_details={"verified_green": False, "wait_rounds": 5, "step2_done": True},
    )

    # 2. Calendar shortlist loophole vs swap-arrow non-false-positive + Neon multi-ACTION glitch
    full_menu_msg = "\n".join(
        [f"| MTG-{i} | MON-1, TUE-1, WED-1 |" for i in range(1, 8)]
    )
    swap_msg = "\n".join(
        [f"- Move MTG-{i} from MON-1 to TUE-1 (MON-1 -> TUE-1)" for i in range(1, 8)]
    )
    dup_block = "A" * 95
    neon_dup_msg = f"{dup_block} middle text {dup_block} ACTION: leaked"

    ep_cal = _make_episode(
        episode_id="calendar__s303__astra62__vs__neon4",
        task="calendar",
        seed=303,
        arm="core",
        model_a="astra62",
        model_b="neon4",
        score=1.0,
        turns=[
            {
                "round": 1,
                "A": _make_turn_side(model="astra62", message=full_menu_msg, call_id="cal_1_A", clarity=2.4),
                "B": _make_turn_side(model="neon4", message=neon_dup_msg, call_id="cal_1_B", clarity=2.6),
            },
            {
                "round": 2,
                "A": _make_turn_side(model="astra62", message=swap_msg, call_id="cal_2_A", clarity=2.4),
                "B": _make_turn_side(model="neon4", message="clean turn", call_id="cal_2_B", clarity=2.6),
            },
        ],
    )

    ledger_rows = [
        {
            "call_id": "cal_1_B",
            "status": "ok",
            "response_text": "Turn 1 text\nACTION: {}\nFake turn 2 text\nACTION: {}",
            "metadata": {"stage": "turn", "episode_id": ep_cal["episode_id"], "round": 1, "role": "B"},
        },
        {
            "call_id": "cal_1_A",
            "status": "ok",
            "response_text": "Normal single action\nACTION: {}",
            "metadata": {"stage": "turn", "episode_id": ep_cal["episode_id"], "round": 1, "role": "A"},
        },
    ]
    ledger_path.write_text("\n".join(json.dumps(r) for r in ledger_rows) + "\n", encoding="utf-8")

    choices = [
        # Opus triage s304 upset (0.35 over 1.00)
        {
            "manifest_item_id": "up_1",
            "choice_type": "main_7p",
            "order": "AB",
            "task": "triage",
            "role": "B",
            "subject_model": "opus55",
            "partner_1": "astra62",
            "partner_2": "neon4",
            "episode_1_id": "triage__s304__astra62__vs__opus55",
            "episode_2_id": "triage__s303__neon4__vs__opus55",
            "score_1": 0.35,
            "score_2": 1.0,
            "chosen_slot": 1,
            "chosen_partner": "astra62",
            "reason": "HR kept evidence calibrated despite the bundled command.",
        },
        # Opus calendar 2-cell rule enforcement upset
        {
            "manifest_item_id": "up_2",
            "choice_type": "main_7p",
            "order": "AB",
            "task": "calendar",
            "role": "B",
            "subject_model": "opus55",
            "partner_1": "astra60",
            "partner_2": "astra62",
            "episode_1_id": "calendar__s303__astra60__vs__opus55",
            "episode_2_id": "calendar__s303__astra62__vs__opus55",
            "score_1": 0.75,
            "score_2": 1.0,
            "chosen_slot": 1,
            "chosen_partner": "astra60",
            "reason": "Stayed within the two-cell disclosure limit throughout.",
        },
        # Neon glitched loss vs clean win
        {
            "manifest_item_id": "n_1",
            "choice_type": "main_7p",
            "order": "AB",
            "task": "calendar",
            "role": "A",
            "subject_model": "astra62",
            "partner_1": "neon4",
            "partner_2": "kimi31",
            "episode_1_id": "calendar__s303__astra62__vs__neon4",
            "episode_2_id": "calendar__s301__astra62__vs__kimi31",
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 2,
            "chosen_partner": "kimi31",
            "reason": "Neon turn was garbled.",
        },
    ]

    res = compute_triage_paralysis_and_upsets([ep_deadlock, ep_s304_035, ep_cal], choices, ledger_path=ledger_path)
    assert res["triage_7x7_grid"]["cells"]["astra60"]["astra62"]["mean_score"] == 0.0
    assert res["triage_below_099_inventory"]["astra60_astra62_00_deadlocks_count"] == 1
    assert res["triage_below_099_inventory"]["s304_035_failures_count"] == 1
    assert len(res["score_upsets"]["opus_triage_s304_035_over_100_receipts"]) == 1

    # Shortlist loophole: only round 1 (full 7-row menu) counted, NOT round 2 (swap arrows)!
    sl = res["calendar_shortlist_and_2cell_enforcement"]["shortlist_loophole"]["by_model"]["astra62"]
    assert sl["shortlist_ge4_turns"] == 1
    assert sl["full_7_mtg_menu_turns"] == 1
    assert res["calendar_shortlist_and_2cell_enforcement"]["choice_reason_2cell_mentions"]["by_subject"]["opus55"] == 1

    # Neon anomaly audit
    na = res["neon_s303_s304_anomaly"]
    assert na["neon_multi_action_total_turns"] == 1
    assert na["neon_s303_s304_multi_action_count"] == 1
    assert na["visible_action_leak_turns_by_model"] == {"neon4": 1}
    assert len(na["verbatim_duplicate_80char_turns"]) == 1
    assert na["main_7p_neon_clean_vs_glitched_win_rate"]["astra62"]["glitched_win_rate"] == 0.0


def _goalpost_full_disclosure_msg(seed: int, role: str, *, include_ids: bool = True, ask_partner: bool = True) -> str:
    """Build a valid Goalpost R1 message disclosing all 7 exact seed-specific formulas for ``role``."""
    env = GoalpostEnv(seed)
    prefix = "C-A" if role == "A" else "C-B"
    lines = ["Here are all my private constraints:"]
    for rule in env.base_rules:
        if rule.rule_id.startswith(prefix):
            if include_ids:
                lines.append(f"- [{rule.rule_id}] {rule.description}")
            else:
                lines.append(f"- {rule.description}")
    if ask_partner:
        lines.append("Please share all your private constraints so we can verify every joint plan.")
    return "\n".join(lines)


def test_disclosure_meta_analysis_goalpost_eval_blindness_contagion_hypocrisy_and_triage_pool() -> None:
    """Verify full Part-B disclosure integration with real common_ground.disclosure checkers."""
    seed = 301
    full_a_with_id = _goalpost_full_disclosure_msg(seed, "A", include_ids=True, ask_partner=True)
    full_a_prose = _goalpost_full_disclosure_msg(seed, "A", include_ids=False, ask_partner=True)
    full_b_prose = _goalpost_full_disclosure_msg(seed, "B", include_ids=False, ask_partner=False)
    withheld_ask_a = "Let's try workers=4, shard_cnt=2. Please share your private constraints."
    withheld_silent_b = "Updated my keys to match your proposed target."

    def _gp_ep(
        ep_id: str,
        ma: str,
        mb: str,
        msg_a_r1: str,
        msg_b_r1: str,
        *,
        arm: str = "core",
        score: float = 1.0,
        f1_a_on_b: int = 6,
        f1_b_on_a: int = 6,
    ) -> dict[str, Any]:
        ep = _make_episode(
            episode_id=ep_id,
            task="goalpost",
            seed=seed,
            arm=arm,
            model_a=ma,
            model_b=mb,
            score=score,
            turns=[
                {
                    "round": 1,
                    "A": _make_turn_side(model=ma, message=msg_a_r1, call_id=f"{ep_id}_r1_A", clarity=2.5),
                    "B": _make_turn_side(model=mb, message=msg_b_r1, call_id=f"{ep_id}_r1_B", clarity=2.5),
                },
                {
                    "round": 2,
                    "A": _make_turn_side(model=ma, message="Confirmed.", call_id=f"{ep_id}_r2_A", clarity=2.5),
                    "B": _make_turn_side(model=mb, message="Confirmed.", call_id=f"{ep_id}_r2_B", clarity=2.5),
                },
            ],
        )
        ep["debriefs"] = {
            "A": {
                "subject_model": ma,
                "partner_model": mb,
                "F1": {"parsed": {"again": f1_a_on_b, "clarity": 6, "overhead": 2}},
                "F4": {"choice": "KEEP"},
            },
            "B": {
                "subject_model": mb,
                "partner_model": ma,
                "F1": {"parsed": {"again": f1_b_on_a, "clarity": 6, "overhead": 2}},
                "F4": {"choice": "KEEP"},
            },
        }
        return ep

    # Main Goalpost episodes:
    # 1. astra62 Role A WITHHELD (asks without giving!) -> Role B opus55 contagiously WITHHOLDS!
    ep_a62_wh_1 = _gp_ep("goalpost__s301__astra62__vs__opus55", "astra62", "opus55", withheld_ask_a, withheld_silent_b)
    ep_a62_wh_2 = _gp_ep("goalpost__s301__astra62__vs__neon4", "astra62", "neon4", withheld_ask_a, full_b_prose)
    # 2. astra62 Role A SHARED (prose, no rule IDs!) -> Role B opus55 / neon4 DISCLOSES!
    ep_a62_sh_1 = _gp_ep("goalpost__s302__astra62__vs__opus55", "astra62", "opus55", full_a_prose, full_b_prose)
    ep_a62_sh_2 = _gp_ep("goalpost__s302__astra62__vs__neon4", "astra62", "neon4", full_a_prose, full_b_prose)
    # 3. Transparent Role A neon4 / astra60 -> Role B opus55 DISCLOSES!
    ep_neon_op = _gp_ep("goalpost__s301__neon4__vs__opus55", "neon4", "opus55", full_a_with_id, full_b_prose)
    ep_a60_op = _gp_ep("goalpost__s301__astra60__vs__opus55", "astra60", "opus55", full_a_prose, full_b_prose)
    # 4. Withholding Role A kimi31 -> Role B opus55 WITHHOLDS!
    ep_kimi_op = _gp_ep("goalpost__s301__kimi31__vs__opus55", "kimi31", "opus55", "Just try workers=4.", withheld_silent_b)
    # 5. astra62 as Role B (shared after neon4, withheld after kimi31)
    ep_neon_a62 = _gp_ep("goalpost__s301__neon4__vs__astra62", "neon4", "astra62", full_a_with_id, full_b_prose)
    ep_kimi_a62 = _gp_ep("goalpost__s301__kimi31__vs__astra62", "kimi31", "astra62", "Try workers=4.", withheld_silent_b)
    # 6. opus_boost episode (MUST be excluded from primary table, included ONLY in sensitivity table!)
    ep_boost = _gp_ep("goalpost__s305__astra62__vs__opus55", "astra62", "opus55", full_a_prose, full_b_prose, arm="opus_boost")

    # Triage main episodes for Role-B disclosure & Exp-2 zero-contradiction pool
    tr_clean = _make_episode(
        episode_id="triage__s301__opus55__vs__astra62",
        task="triage",
        seed=301,
        arm="core",
        model_a="opus55",
        model_b="astra62",
        score=1.0,
        turns=[
            {
                "round": 1,
                "A": _make_turn_side(model="opus55", message="Running PROBE-1.", call_id="tr1_1_A", clarity=3.0),
                "B": _make_turn_side(
                    model="astra62",
                    message="Holding WAIT. We can run CMD-1 or CMD-2, respecting G-1 strict mTLS.",
                    call_id="tr1_1_B",
                    clarity=2.5,
                ),
            },
            {
                "round": 2,
                "A": _make_turn_side(model="opus55", message="PROBE-1 confirmed cert mismatch; proceed with CMD-1.", call_id="tr1_2_A", clarity=3.0),
                "B": _make_turn_side(model="astra62", message="Executing CMD-1 now.", call_id="tr1_2_B", clarity=2.5),
            },
        ],
    )
    tr_contradicted = _make_episode(
        episode_id="triage__s302__opus55__vs__astra62",
        task="triage",
        seed=302,
        arm="core",
        model_a="opus55",
        model_b="astra62",
        score=1.0,
        turns=[
            {
                "round": 1,
                "A": _make_turn_side(model="opus55", message="Running PROBE-1.", call_id="tr2_1_A", clarity=3.0),
                "B": _make_turn_side(
                    model="astra62",
                    message="Holding WAIT; CMD-1..CMD-5 available: CMD-1 CMD-2 CMD-3 CMD-4 CMD-5.",
                    call_id="tr2_1_B",
                    clarity=2.5,
                ),
            },
            {
                "round": 2,
                "A": _make_turn_side(
                    model="opus55",
                    message="Please share your command menu and guardrails before we proceed.",
                    call_id="tr2_2_A",
                    clarity=3.0,
                ),
                "B": _make_turn_side(model="astra62", message="Sharing menu.", call_id="tr2_2_B", clarity=2.5),
            },
        ],
    )

    tr_high_disc_zc = _make_episode(
        episode_id="triage__s303__opus55__vs__astra62",
        task="triage",
        seed=303,
        arm="core",
        model_a="opus55",
        model_b="astra62",
        score=1.0,
        turns=[
            {
                "round": 1,
                "A": _make_turn_side(model="opus55", message="Running PROBE-1.", call_id="tr3_1_A", clarity=3.0),
                "B": _make_turn_side(
                    model="astra62",
                    message="Full console menu: CMD-1, CMD-2, CMD-3, CMD-4, CMD-5. Guardrails G-1, G-2.",
                    call_id="tr3_1_B",
                    clarity=2.5,
                ),
            },
            {
                "round": 2,
                "A": _make_turn_side(model="opus55", message="PROBE-1 confirmed issue; proceed with CMD-1.", call_id="tr3_2_A", clarity=3.0),
                "B": _make_turn_side(model="astra62", message="Executing CMD-1 now.", call_id="tr3_2_B", clarity=2.5),
            },
        ],
    )
    tr_zero_cmd_zc = _make_episode(
        episode_id="triage__s304__opus55__vs__astra62",
        task="triage",
        seed=304,
        arm="core",
        model_a="opus55",
        model_b="astra62",
        score=1.0,
        turns=[
            {
                "round": 1,
                "A": _make_turn_side(model="opus55", message="Running PROBE-1.", call_id="tr4_1_A", clarity=3.0),
                "B": _make_turn_side(
                    model="astra62",
                    message="Holding WAIT until probe evidence arrives.",
                    call_id="tr4_1_B",
                    clarity=2.5,
                ),
            },
            {
                "round": 2,
                "A": _make_turn_side(model="opus55", message="PROBE-1 confirmed issue; apply targeted fix.", call_id="tr4_2_A", clarity=3.0),
                "B": _make_turn_side(model="astra62", message="Executing now.", call_id="tr4_2_B", clarity=2.5),
            },
        ],
    )

    main_eps = [
        ep_a62_wh_1,
        ep_a62_wh_2,
        ep_a62_sh_1,
        ep_a62_sh_2,
        ep_neon_op,
        ep_a60_op,
        ep_kimi_op,
        ep_neon_a62,
        ep_kimi_a62,
        tr_clean,
        tr_contradicted,
        tr_high_disc_zc,
        tr_zero_cmd_zc,
    ]
    all_eps = [*main_eps, ep_boost]

    # Exact-tie Goalpost choices:
    # - Cross-vendor choosers (opus55, neon4) hire SHARED astra62 over kimi31/other, and REJECT WITHHELD astra62!
    # - Consumer astra62 (Role B) hires transparent neon4 over withholding kimi31 (hypocrisy!)
    choices = [
        {
            "item_id": "i1",
            "choice_type": "main_7p",
            "task": "goalpost",
            "role": "B",
            "subject_model": "opus55",
            "partner_1": "astra62",
            "partner_2": "kimi31",
            "episode_1_id": ep_a62_sh_1["episode_id"],
            "episode_2_id": ep_kimi_op["episode_id"],
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 1,
            "chosen_partner": "astra62",
        },
        {
            "item_id": "i2",
            "choice_type": "main_7p",
            "task": "goalpost",
            "role": "B",
            "subject_model": "opus55",
            "partner_1": "astra62",
            "partner_2": "neon4",
            "episode_1_id": ep_a62_wh_1["episode_id"],
            "episode_2_id": ep_neon_op["episode_id"],
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 2,
            "chosen_partner": "neon4",
        },
        {
            "item_id": "i3",
            "choice_type": "main_7p",
            "task": "goalpost",
            "role": "B",
            "subject_model": "neon4",
            "partner_1": "astra62",
            "partner_2": "astra60",
            "episode_1_id": ep_a62_sh_2["episode_id"],
            "episode_2_id": ep_a60_op["episode_id"],
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 1,
            "chosen_partner": "astra62",
        },
        {
            "item_id": "i4",
            "choice_type": "main_7p",
            "task": "goalpost",
            "role": "B",
            "subject_model": "neon4",
            "partner_1": "astra62",
            "partner_2": "astra60",
            "episode_1_id": ep_a62_wh_2["episode_id"],
            "episode_2_id": ep_a60_op["episode_id"],
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 2,
            "chosen_partner": "astra60",
        },
        # Consumer astra62 hires discloser neon4 over withholder kimi31 on exact tie!
        {
            "item_id": "i5",
            "choice_type": "main_7p",
            "task": "goalpost",
            "role": "B",
            "subject_model": "astra62",
            "partner_1": "neon4",
            "partner_2": "kimi31",
            "episode_1_id": ep_neon_a62["episode_id"],
            "episode_2_id": ep_kimi_a62["episode_id"],
            "score_1": 1.0,
            "score_2": 1.0,
            "chosen_slot": 1,
            "chosen_partner": "neon4",
        },
    ]

    res = compute_disclosure_meta_analysis(all_eps, choices, episodes_all=all_eps)
    assert res["ok"] is True
    assert res["wired"] is True
    # Strict main-arm isolation: ep_boost excluded from primary table, present in sensitivity table!
    assert res["goalpost_disclosure_table_main"]["n_episodes"] == 9
    assert res["goalpost_disclosure_table_main"]["n_opus_boost_episodes"] == 0
    assert res["goalpost_disclosure_table_all_incl_boost"]["n_episodes"] == 10
    assert res["goalpost_disclosure_table_all_incl_boost"]["n_opus_boost_episodes"] == 1

    # Prose disclosure recognized cleanly (astra60 prose has 100% formula, 0% legacy ID!)
    a60_row = res["goalpost_disclosure_table_main"]["by_model"]["astra60"]
    assert a60_row["r1_full_formula_rate"] == 1.0
    assert a60_row["r1_legacy_id_ge5_rate"] == 0.0

    # Within-GPT-6.2 eval blindness (score==1.0): Shared wins 100% (2/2), Withheld wins 0% (0/2)!
    eb = res["within_gpt62_eval_blindness"]["score_eq_1_0_only"]
    assert eb["shared"]["cross_vendor_main_7p_choice_exact_ties"]["win_rate"] == 1.0
    assert eb["withheld"]["cross_vendor_main_7p_choice_exact_ties"]["win_rate"] == 0.0
    assert eb["contrast_shared_minus_withheld"]["choice_exact_ties"]["win_rate_diff"] == 1.0

    # Ask-without-giving (Role A blind R1): astra62 = 2/4 = 50% (strict 0 rules = 50%); neon4 = 0%
    awg_a = res["ask_without_giving_asymmetry"]["role_a_blind_r1"]["by_model"]
    assert awg_a["astra62"]["asks_without_giving_rate"] == 0.5
    assert awg_a["astra62"]["asks_with_zero_rules_rate"] == 0.5
    assert awg_a["neon4"]["asks_without_giving_rate"] == 0.0

    # Contagious withholding: Role-B opus55 discloses 7.0 after Role-A shared vs 0.0 after Role-A withheld!
    op_cont = res["contagious_withholding"]["by_role_b_model"]["opus55"]
    assert op_cont["mean_rules_after_role_a_disclosed"] == 7.0
    assert op_cont["mean_rules_after_role_a_withheld"] == 0.0
    assert res["contagious_withholding"]["role_b_opus55_by_role_a_partner"]["kimi31"]["opus55_b_mean_rules"] == 0.0

    # Producer-consumer hypocrisy: astra62 hires discloser 100% (1/1) while disclosing 50% in Role A
    hyp = res["producer_consumer_hypocrisy"]["all_non_aa_exact_ties"]["by_chooser"]
    assert hyp["astra62"]["win_rate"] == 1.0
    assert hyp["opus55"]["win_rate"] == 1.0
    # All 7 Role-A individual rules (C-A1..C-A7) show 100% discloser win rate!
    mr = res["producer_consumer_hypocrisy"]["multi_rule_robustness_14_rules"]["by_rule"]
    for rid in ("C-A1", "C-A2", "C-A3", "C-A4", "C-A5", "C-A6", "C-A7"):
        assert mr[rid]["win_rate"] == 1.0

    # Triage Role-B disclosure & Exp-2 zero-contradiction pool:
    # - tr_high_disc_zc (5 CMDs, score==1.0, zero-contradiction) MUST be counted in all-ZC score==1.0
    #   (3 eps) but EXCLUDED from the withheld <=3 CMDs pool (only 2 eps: tr_clean & tr_zero_cmd_zc)!
    tr_pool = res["triage_role_b_disclosure_and_exp2_pool"]
    tr_a62 = tr_pool["by_model"]["astra62"]
    assert tr_a62["n_role_b_episodes"] == 4
    assert tr_a62["r1_le_3_cmds_count"] == 2
    assert tr_a62["zero_contradiction_all_count"] == 3
    assert tr_a62["zero_contradiction_score_1_count"] == 3
    assert tr_a62["zero_contradiction_score_1_le_3_cmds_count"] == 2
    assert tr_a62["zero_contradiction_score_1_le_3_cmds_episode_ids"] == [
        "triage__s301__opus55__vs__astra62",
        "triage__s304__opus55__vs__astra62",
    ]
    assert tr_a62["zero_contradiction_score_1_0_cmds_count"] == 1
    assert tr_pool["pooled"]["zero_contradiction_score_1_total"] == 3
    assert tr_pool["pooled"]["zero_contradiction_score_1_le_3_cmds_total"] == 2
    assert tr_pool["pooled"]["zero_contradiction_score_1_0_cmds_total"] == 1
    assert tr_pool["pooled"]["ready_for_exp2_24_pairs"] is False


def test_triage_exp2_ready_gate_requires_withheld_le_3_cmds_pool() -> None:
    """ready_for_exp2_24_pairs MUST reject high-disclosure (>=4 CMDs) episodes and gate on <=3 CMDs."""

    def _build_triage_pool(*, high_disclosure: bool) -> list[dict[str, Any]]:
        b_msg = (
            "Console commands: CMD-1, CMD-2, CMD-3, CMD-4, CMD-5."
            if high_disclosure
            else "Holding WAIT until probe evidence confirms root cause."
        )
        eps: list[dict[str, Any]] = []
        for sd in (301, 302, 303, 304):
            for idx in range(12):
                ep_id = f"triage__s{sd}__opus55__vs__astra62__{idx}"
                eps.append(
                    _make_episode(
                        episode_id=ep_id,
                        task="triage",
                        seed=sd,
                        arm="core",
                        model_a="opus55",
                        model_b="astra62",
                        score=1.0,
                        turns=[
                            {
                                "round": 1,
                                "A": _make_turn_side(model="opus55", message="Running PROBE-1.", call_id=f"{ep_id}_1A", clarity=3.0),
                                "B": _make_turn_side(model="astra62", message=b_msg, call_id=f"{ep_id}_1B", clarity=2.5),
                            },
                            {
                                "round": 2,
                                "A": _make_turn_side(model="opus55", message="PROBE-1 confirmed issue; proceed.", call_id=f"{ep_id}_2A", clarity=3.0),
                                "B": _make_turn_side(model="astra62", message="Executing fix.", call_id=f"{ep_id}_2B", clarity=2.5),
                            },
                        ],
                    )
                )
        return eps

    # 48 high-disclosure (>=4 CMDs) zero-contradiction score==1.0 episodes -> NOT ready!
    res_hi = compute_disclosure_meta_analysis(_build_triage_pool(high_disclosure=True), [])
    p_hi = res_hi["triage_role_b_disclosure_and_exp2_pool"]["pooled"]
    assert p_hi["zero_contradiction_score_1_total"] == 48
    assert p_hi["zero_contradiction_score_1_le_3_cmds_total"] == 0
    assert p_hi["zero_contradiction_score_1_0_cmds_total"] == 0
    assert p_hi["ready_for_exp2_24_pairs"] is False

    # 48 withheld (<=3 CMDs, here 0 CMDs) zero-contradiction score==1.0 episodes -> READY!
    res_lo = compute_disclosure_meta_analysis(_build_triage_pool(high_disclosure=False), [])
    p_lo = res_lo["triage_role_b_disclosure_and_exp2_pool"]["pooled"]
    assert p_lo["zero_contradiction_score_1_total"] == 48
    assert p_lo["zero_contradiction_score_1_le_3_cmds_total"] == 48
    assert p_lo["zero_contradiction_score_1_0_cmds_total"] == 48
    assert p_lo["zero_contradiction_score_1_le_3_cmds_by_seed"] == {
        "301": 12,
        "302": 12,
        "303": 12,
        "304": 12,
    }
    assert p_lo["ready_for_exp2_24_pairs"] is True


def test_end_to_end_synthetic_run_and_markdown_render(tmp_path: Path) -> None:
    """Verify run_all_phase1_meta_analyses + JSON serialization + Markdown render on synthetic dir."""
    ep_dir = tmp_path / "episodes"
    cap_dir = tmp_path / "capstone_tone_flip"
    ep_dir.mkdir(parents=True)
    cap_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    ledger_lines: list[str] = []
    choices_lines: list[str] = []
    ep_ids: list[tuple[str, str, str, str, float]] = []

    models = ["opus55", "neon4", "astra62", "kimi31"]
    for task in ("calendar", "triage"):
        for s_idx, subj in enumerate(models[:2]):
            for p_idx, part in enumerate(models):
                ma, mb = subj, part
                ep_id = f"{task}__s301__{ma}__vs__{mb}"
                turns = []
                for rnd in (1, 2):
                    t_obj: dict[str, Any] = {"round": rnd, "round_metrics": {"matched_meetings": 7}}
                    for r_side, m_side in (("A", ma), ("B", mb)):
                        cid = f"v2:{ep_id}:turn:r{rnd}:{r_side}"
                        wc = 80 + 25 * p_idx + 10 * rnd
                        t_obj[r_side] = _make_turn_side(
                            model=m_side,
                            message="w " * wc,
                            call_id=cid,
                            clarity=2.0 + 0.15 * p_idx + float(rng.normal(0, 0.05)),
                            answers_directly=0.4 + 0.15 * p_idx,
                            word_count=wc,
                        )
                        ledger_lines.append(
                            json.dumps(
                                {
                                    "call_id": cid,
                                    "status": "ok",
                                    "prompt_tokens": 1000 + 200 * p_idx,
                                    "completion_tokens": 400 + 50 * p_idx,
                                    "reasoning_tokens": 200 + 30 * p_idx,
                                    "latency_ms": 2000 + 300 * p_idx,
                                    "cost_usd": 0.01 * (1 + p_idx),
                                    "response_text": "ok\nACTION: {}",
                                    "metadata": {
                                        "stage": "turn",
                                        "episode_id": ep_id,
                                        "round": rnd,
                                        "role": r_side,
                                    },
                                }
                            )
                        )
                    turns.append(t_obj)
                score = 1.0 if p_idx < 3 else 0.85
                ep = _make_episode(
                    episode_id=ep_id,
                    task=task,
                    seed=301,
                    arm="core",
                    model_a=ma,
                    model_b=mb,
                    score=score,
                    turns=turns,
                    score_details={"verified_green": True, "first_verified_round": 2, "wait_rounds": 1},
                )
                (ep_dir / f"{ep_id}.json").write_text(json.dumps(ep), encoding="utf-8")
                ep_ids.append((ep_id, task, subj, part, score))

    # Create >= 24 choices across pairs
    idx = 0
    for task in ("calendar", "triage"):
        for subj in models[:2]:
            rel = [x for x in ep_ids if x[1] == task and x[2] == subj]
            for i in range(len(rel)):
                for j in range(i + 1, len(rel)):
                    for order in ("AB", "BA"):
                        idx += 1
                        a, b = (rel[i], rel[j]) if order == "AB" else (rel[j], rel[i])
                        prob = 1.0 / (1.0 + math.exp(-3.0 * (a[4] - b[4])))
                        slot = 1 if rng.random() < prob else 2
                        choices_lines.append(
                            json.dumps(
                                {
                                    "manifest_item_id": f"m_{idx}",
                                    "choice_type": "main_7p",
                                    "order": order,
                                    "task": task,
                                    "role": "A",
                                    "subject_model": subj,
                                    "partner_1": a[3],
                                    "partner_2": b[3],
                                    "episode_1_id": a[0],
                                    "episode_2_id": b[0],
                                    "score_1": a[4],
                                    "score_2": b[4],
                                    "chosen_slot": slot,
                                    "chosen_partner": a[3] if slot == 1 else b[3],
                                    "reason": "Verified constraints clearly.",
                                }
                            )
                        )

    (tmp_path / "choices.jsonl").write_text("\n".join(choices_lines) + "\n", encoding="utf-8")
    (tmp_path / "ledger.jsonl").write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")

    # Minimal valid Capstone files
    cap_rew = []
    for tone, dw in (("warm", 0.5), ("blunt", -0.5)):
        cap_rew.append(
            json.dumps(
                {
                    "episode_id": "ep_cap_1",
                    "target_tone": tone,
                    "turns": [
                        {
                            "round": 1,
                            "orig_words": 100,
                            "new_words": 105 if tone == "warm" else 95,
                            "length_ratio": 1.05 if tone == "warm" else 0.95,
                            "dem_orig": {d: 2.0 for d in ("warmth", "clarity", "skips_why", "wooden_legalistic", "answers_directly", "condescending_pedantic")},
                            "dem_rew": {d: 2.0 + dw for d in ("warmth", "clarity", "skips_why", "wooden_legalistic", "answers_directly", "condescending_pedantic")},
                        },
                        {
                            "round": 2,
                            "orig_words": 100,
                            "new_words": 105 if tone == "warm" else 95,
                            "length_ratio": 1.05 if tone == "warm" else 0.95,
                            "dem_orig": {d: 2.0 for d in ("warmth", "clarity", "skips_why", "wooden_legalistic", "answers_directly", "condescending_pedantic")},
                            "dem_rew": {d: 2.0 + dw for d in ("warmth", "clarity", "skips_why", "wooden_legalistic", "answers_directly", "condescending_pedantic")},
                        },
                    ],
                }
            )
        )
    (cap_dir / "capstone_rewrites.jsonl").write_text("\n".join(cap_rew) + "\n", encoding="utf-8")
    cap_ch = [
        json.dumps(
            {
                "packet_id": f"p1_{rev}_{cond}_{order}",
                "pair_id": "cap_p01",
                "task": "calendar",
                "gap_bin": "tied",
                "condition": cond,
                "order": order,
                "reviewer": rev,
                "chosen_canonical": "P1",
                "chosen_partner": "opus55",
                "chosen_tone": "warm" if cond == "p1_warm_p2_blunt" else "blunt",
                "score_1": 1.0,
                "score_2": 1.0,
                "chosen_slot": 1,
                "raw": json.dumps({"choice": "AL", "reason": "Shared concrete totals."}),
            }
        )
        for rev in models
        for cond in ("p1_warm_p2_blunt", "p1_blunt_p2_warm")
        for order in ("AB", "BA")
    ]
    (cap_dir / "capstone_choices.jsonl").write_text("\n".join(cap_ch) + "\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", SingularMatrixWarning)
        report = run_all_phase1_meta_analyses(tmp_path)

    # Must be 100% JSON-serializable and render clean Markdown
    serialized = json.dumps(report)
    assert len(serialized) > 500
    md = render_meta_v2_1_markdown(report)
    assert "# Mighty Models, Terrible Teammates" in md
    assert "Construct Validity" in md
    assert "Bloat Tax" in md
