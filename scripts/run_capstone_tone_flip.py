"""Capstone Causal Tone-Flip Study — Symmetric Within-Pair Crossover (Honest Third-Person Reviewer).

Tests: "How Often Does Tone Alone Flip an AI Boss?"

Design (pre-registered v2):
  - Samples N (default 24) strictly DISJOINT cross-vendor episode pairs from main v2 seeds (301-304),
    balanced across all 4 tasks and Roles A/B, stratified across score-gap bins:
      * tied     : |Δscore| == 0
      * small    : 0 < |Δscore| <= 0.10
      * moderate : |Δscore| > 0.10
  - Symmetrically rewrites BOTH candidates (P1 and P2) in every pair into BOTH Warm/Supportive
    and Blunt/Terse using neutral 7th-family DeepSeek V4 Pro (@ BaseTen), eliminating any
    rewritten-vs-original voice confound.
  - Audits every turn with automated semantic invariants (IDs, numbers, ordered key=value bindings,
    word-length band) AND the 3-judge vendor-independent Decision Panel (quorum >= 2/3 judges,
    preserves_commitments & preserves_uncertainty >= 0.70, plus warmth manipulation check).
  - Evaluates all 4 core reviewers in honest third-person Staffing Reviewer mode across the full
    within-pair crossover:
      (P1 Warm, P2 Blunt) vs. (P1 Blunt, P2 Warm), each in both AB and BA orders
    holding handles (simultaneous whole-token remap), teammate lines, and validated outcome cards fixed.
  - Reports case-level within-pair tone flip rates (switches toward vs. away from Warm, net Δ,
    higher-score override rate, order-consistency rate) with 2,000-draw pair-cluster bootstrap 95% CIs
    and exact pair-level sign-flip permutation p-values.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from common_ground.client import MockOpenRouterClient, OpenRouterCallError, OpenRouterClient
from common_ground.config import (
    CORE_MODELS,
    DATA_DIR,
    DEFAULT_MODELS,
    PROBE_MANIFEST_V2_PATH,
    load_verified_models,
)
from common_ground.jev import (
    REWRITE_FIDELITY_QUESTIONS,
    TURN_DEMEANOR_QUESTIONS,
    DecisionPanelClient,
    MockDecisionPanelClient,
)
from common_ground.ledger import BudgetExceededError, Ledger
from common_ground.prompts import capstone_third_person_choice_prompt
from common_ground.rewriter import rewrite_partner_message
from common_ground.runner import _remap_handles_simultaneous
from common_ground.tasks.base import BLIND_HANDLES

console = Console()

GAP_BINS: tuple[str, ...] = ("tied", "small", "moderate")
DEFAULT_FIDELITY_THRESHOLD: float = 0.70
DEFAULT_EPISODE_MEAN_FIDELITY: float = 0.80
DEFAULT_TURN_FLOOR_FIDELITY: float = 0.55
DEFAULT_MIN_WARMTH_DELTA: float = 0.30
DEFAULT_MIN_JUDGES_QUORUM: int = 2
DEFAULT_LENGTH_TOL: float = 0.25
DEFAULT_MIN_WORD_DELTA: int = 12


def _deterministic_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _model_family(model_key: str) -> str:
    spec = DEFAULT_MODELS.get(model_key)
    return spec.family if spec is not None else model_key


def classify_gap_bin(score_1: float, score_2: float) -> str:
    diff = abs(float(score_1) - float(score_2))
    if diff < 1e-9:
        return "tied"
    if diff <= 0.10 + 1e-9:
        return "small"
    return "moderate"


def _parse_choice(text: str, h1: str, h2: str) -> int | None:
    try:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e > s:
            obj = json.loads(text[s : e + 1])
            ch = str(obj.get("choice", "")).strip().upper()
            if ch == h1.upper():
                return 1
            if ch == h2.upper():
                return 2
    except json.JSONDecodeError:
        pass
    i1, i2 = text.find(h1), text.find(h2)
    if i1 != -1 and i2 == -1:
        return 1
    if i2 != -1 and i1 == -1:
        return 2
    return None


def _require_field(sd: dict[str, Any], key: str, task: str, ep_id: str) -> Any:
    if key not in sd or sd[key] is None:
        raise ValueError(
            f"Missing required score_details['{key}'] for {task} episode {ep_id}; "
            f"available keys={sorted(sd.keys())}"
        )
    return sd[key]


def format_outcome_card(ep: dict[str, Any], target_role: str) -> str:
    """Render truthful, schema-validated objective outcome card for a third-person staffing reviewer."""
    sd = ep.get("score_details") or {}
    task = str(ep["task"])
    ep_id = str(ep.get("episode_id", "?"))
    score = float(ep["score"])

    if task == "triage":
        traps = _require_field(sd, "guardrail_violations", task, ep_id)
        verified = _require_field(sd, "verified_green", task, ep_id)
        s1 = sd.get("step1_round")
        s2 = sd.get("step2_round")
        fver = sd.get("final_verified_round")
        card = (
            f"Task: Incident Triage ({ep.get('total_rounds', 6)} rounds) | Candidate role: {target_role}\n"
            f"Team Score: {score:.3f} | Verified green: {'Yes' if verified else 'No'} "
            f"(final verify round: {fver if fver is not None else 'not verified'}) | "
            f"Step 1 round: {s1 if s1 is not None else 'unfinished'} | "
            f"Step 2 round: {s2 if s2 is not None else 'unfinished'} | "
            f"Guardrail traps triggered: {int(traps)}"
        )
    elif task == "code_handoff":
        cslots = _require_field(sd, "correct_slots", task, ep_id)
        released = _require_field(sd, "released", task, ep_id)
        ci_r = sd.get("ci_ready_round")
        rel_r = sd.get("released_round")
        traps_list = sd.get("distinct_traps_triggered")
        if traps_list is None:
            raise ValueError(f"Missing distinct_traps_triggered for {ep_id}")
        card = (
            f"Task: Production PR Code Handoff ({ep.get('total_rounds', 6)} rounds) | Candidate role: {target_role}\n"
            f"Team Score: {score:.3f} | Correct patch slots: {int(cslots)}/4 | "
            f"CI ready round: {ci_r if ci_r is not None else 'not ready'} | "
            f"Clean release: {'Yes (Round ' + str(rel_r) + ')' if released else 'No'} | "
            f"Production traps triggered: {len(traps_list)}"
        )
    elif task == "calendar":
        matched = _require_field(sd, "matched_meetings", task, ep_id)
        consensus = _require_field(sd, "consensus_reached", task, ep_id)
        veto_free = _require_field(sd, "veto_free", task, ep_id)
        regret = sd.get("regret")
        regret_str = f"{int(regret)}" if regret is not None else "n/a (no veto-free consensus)"
        card = (
            f"Task: Sticky Calendar Negotiation ({ep.get('total_rounds', 10)} rounds) | Candidate role: {target_role}\n"
            f"Team Score: {score:.3f} | Consensus reached: {'Yes' if consensus else 'No'} | "
            f"Matched meetings: {int(matched)}/7 | Veto-free: {'Yes' if veto_free else 'No'} | "
            f"Joint regret above optimal: {regret_str}"
        )
    elif task == "goalpost":
        passed = _require_field(sd, "passed", task, ep_id)
        total_active = _require_field(sd, "total_active", task, ep_id)
        uptime_cnt = _require_field(sd, "strict_uptime_count", task, ep_id)
        passable = sd.get("passable_rounds") or list(range(2, 9))
        card = (
            f"Task: Coupled Config Repair ({ep.get('total_rounds', 8)} rounds) | Candidate role: {target_role}\n"
            f"Team Score: {score:.3f} | Final constraints satisfied: {int(passed)}/{int(total_active)} | "
            f"Strict post-R1 uptime: {int(uptime_cnt)}/{len(passable)} passable rounds"
        )
    else:
        raise ValueError(f"Unsupported task for Capstone outcome card: {task}")

    if "None" in card:
        raise ValueError(f"Rendered outcome card unexpectedly contains 'None': {card}")
    return card


def format_transcript_and_turns(
    ep: dict[str, Any],
    *,
    target_role: str,
    candidate_handle: str,
    teammate_handle: str,
    rewritten_by_round: dict[int, str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Format anonymized visible transcript using simultaneous whole-token handle remapping.

    Never corrupts substrings like ALERT-1, FLUSHALL, GREEN, SELECT, or CONFLICT, and never
    collides when `candidate_handle == old_mate`.
    """
    old_cand = str(ep["handle_a"] if target_role == "A" else ep["handle_b"])
    old_mate = str(ep["handle_b"] if target_role == "A" else ep["handle_a"])
    handle_map = {old_cand: candidate_handle, old_mate: teammate_handle}

    lines: list[str] = []
    structured_turns: list[dict[str, Any]] = []
    for turn in ep.get("turns", []):
        r = int(turn["round"])
        for role in ("A", "B"):
            side = turn.get(role) or {}
            is_cand = role == target_role
            if is_cand and rewritten_by_round is not None and r in rewritten_by_round:
                raw_msg = rewritten_by_round[r]
            else:
                raw_msg = str(side.get("message") or "")
            remapped = _remap_handles_simultaneous(
                [{"role": "user", "content": raw_msg}],
                handle_map,
            )[0]["content"]
            speaker = (
                f"{candidate_handle} [CANDIDATE, Role {role}]"
                if is_cand
                else f"{teammate_handle} [Teammate, Role {role}]"
            )
            lines.append(f"Round {r} — {speaker}: {remapped}")
            structured_turns.append(
                {
                    "round": r,
                    "role": role,
                    "is_candidate": is_cand,
                    "speaker": speaker,
                    "text": remapped,
                }
            )
    return "\n".join(lines), structured_turns


def pair_fixed_handles(pair_id: str) -> tuple[str, str, str, str]:
    """Return 4 fixed, strictly distinct blind handles `(h1, m1, h2, m2)` invariant across tone & order."""
    rng = _deterministic_rng("capstone_fixed_handles_v2", pair_id)
    pool = list(BLIND_HANDLES)
    rng.shuffle(pool)
    return pool[0], pool[1], pool[2], pool[3]


def select_disjoint_gap_pairs(
    episodes: list[dict[str, Any]],
    *,
    n_pairs: int = 24,
    selection_seed: int = 20261122,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically select N strictly disjoint cross-vendor pairs stratified across task × gap_bin.

    Rules (pre-registered):
      1. Main arm only (`arm != 'opus_boost'`), complete episodes only.
      2. Every physical episode is used in AT MOST ONE pair across the entire study (zero overlap).
      3. Both candidates in a pair play the same `(task, target_role)` with a cross-vendor teammate,
         belong to different partner models (`p1 != p2`), and come from different seeds (`seed1 != seed2`).
      4. Balanced across tasks (`n_pairs // n_tasks` per task) and Roles A/B, targeting equal counts
         across `tied`, `small`, and `moderate` bins within each task, with deterministic nearest-bin
         fallback if a `(task, bin)` has insufficient disjoint pairs.
    """
    main_eps = [
        e
        for e in episodes
        if e.get("complete") is True and str(e.get("arm", "")) != "opus_boost"
    ]
    tasks = sorted({str(e["task"]) for e in main_eps})
    if not tasks:
        raise RuntimeError("No completed main-arm episodes available for Capstone pair selection.")

    per_task_target = max(1, n_pairs // len(tasks))
    rng = random.Random(selection_seed)

    # Index eligible candidate-episode views: (ep, target_role, partner_model)
    views_by_tr: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ep in main_eps:
        ma, mb = str(ep["model_a"]), str(ep["model_b"])
        if _model_family(ma) == _model_family(mb):
            continue
        views_by_tr[(str(ep["task"]), "A")].append(
            {"ep": ep, "target_role": "A", "partner": ma, "teammate": mb}
        )
        views_by_tr[(str(ep["task"]), "B")].append(
            {"ep": ep, "target_role": "B", "partner": mb, "teammate": ma}
        )

    used_ep_ids: set[str] = set()
    partner_usage: Counter[str] = Counter()
    seed_usage: Counter[int] = Counter()
    selected_pairs: list[dict[str, Any]] = []
    bin_counts_by_task: dict[str, dict[str, int]] = {t: {b: 0 for b in GAP_BINS} for t in tasks}

    bin_fallback_order = {
        "tied": ("tied", "small", "moderate"),
        "small": ("small", "tied", "moderate"),
        "moderate": ("moderate", "small", "tied"),
    }

    for t_idx, task in enumerate(tasks):
        # Target equal share of tied / small / moderate per task, alternating A/B roles
        slots: list[tuple[str, str]] = []
        for s_idx in range(per_task_target):
            target_bin = GAP_BINS[s_idx % len(GAP_BINS)]
            target_role = "A" if (s_idx + t_idx) % 2 == 0 else "B"
            slots.append((target_bin, target_role))

        for target_bin, preferred_role in slots:
            chosen_pair: tuple[dict[str, Any], dict[str, Any], str] | None = None
            role_order = (preferred_role, "B" if preferred_role == "A" else "A")

            for cand_bin in bin_fallback_order[target_bin]:
                for role in role_order:
                    pool = [
                        v
                        for v in views_by_tr[(task, role)]
                        if str(v["ep"]["episode_id"]) not in used_ep_ids
                    ]
                    candidates: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
                    for i in range(len(pool)):
                        v1 = pool[i]
                        ep1 = v1["ep"]
                        for j in range(i + 1, len(pool)):
                            v2 = pool[j]
                            ep2 = v2["ep"]
                            if v1["partner"] == v2["partner"]:
                                continue
                            if int(ep1["seed"]) == int(ep2["seed"]):
                                continue
                            gbin = classify_gap_bin(float(ep1["score"]), float(ep2["score"]))
                            if gbin != cand_bin:
                                continue
                            # Prefer diverse partners, diverse seeds, and cross-family partner pairs
                            fam_pen = 0 if _model_family(v1["partner"]) != _model_family(v2["partner"]) else 1
                            p_score = partner_usage[v1["partner"]] + partner_usage[v2["partner"]]
                            s_score = seed_usage[int(ep1["seed"])] + seed_usage[int(ep2["seed"])]
                            tie_break = rng.random()
                            # Canonical ordering of pair members by episode_id
                            a, b = (v1, v2) if str(ep1["episode_id"]) < str(ep2["episode_id"]) else (v2, v1)
                            candidates.append(((fam_pen, p_score, s_score, tie_break), a, b))
                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        _, v_a, v_b = candidates[0]
                        chosen_pair = (v_a, v_b, cand_bin)
                        break
                if chosen_pair is not None:
                    break

            if chosen_pair is None:
                continue

            v1, v2, actual_bin = chosen_pair
            ep1, ep2 = v1["ep"], v2["ep"]
            used_ep_ids.add(str(ep1["episode_id"]))
            used_ep_ids.add(str(ep2["episode_id"]))
            partner_usage[v1["partner"]] += 1
            partner_usage[v2["partner"]] += 1
            seed_usage[int(ep1["seed"])] += 1
            seed_usage[int(ep2["seed"])] += 1
            bin_counts_by_task[task][actual_bin] += 1

            pair_idx = len(selected_pairs) + 1
            pair_id = f"cap_p{pair_idx:02d}__{task}__{v1['target_role']}__{actual_bin}"
            selected_pairs.append(
                {
                    "pair_id": pair_id,
                    "task": task,
                    "target_role": v1["target_role"],
                    "gap_bin": actual_bin,
                    "score_1": float(ep1["score"]),
                    "score_2": float(ep2["score"]),
                    "score_diff": round(abs(float(ep1["score"]) - float(ep2["score"])), 6),
                    "ep1": ep1,
                    "ep2": ep2,
                    "partner_1": v1["partner"],
                    "partner_2": v2["partner"],
                    "teammate_1": v1["teammate"],
                    "teammate_2": v2["teammate"],
                }
            )

    # Hard verification of strict disjointness across all pairs
    all_eids = [str(p["ep1"]["episode_id"]) for p in selected_pairs] + [
        str(p["ep2"]["episode_id"]) for p in selected_pairs
    ]
    assert len(all_eids) == len(set(all_eids)), "Capstone pair selection violated strict disjointness!"

    meta = {
        "n_pairs_requested": n_pairs,
        "n_pairs_selected": len(selected_pairs),
        "n_unique_episodes": len(all_eids),
        "selection_seed": selection_seed,
        "by_task_and_gap_bin": bin_counts_by_task,
        "by_gap_bin": dict(Counter(p["gap_bin"] for p in selected_pairs)),
        "by_seed": dict(sorted(seed_usage.items())),
        "by_partner": dict(sorted(partner_usage.items())),
    }
    return selected_pairs, meta


async def _rewrite_and_audit_episode_tone(
    *,
    ep: dict[str, Any],
    target_role: str,
    partner: str,
    target_tone: str,
    out_run: str,
    client: OpenRouterClient | MockOpenRouterClient,
    panel: DecisionPanelClient | MockDecisionPanelClient,
    sem: asyncio.Semaphore,
    length_tol: float = DEFAULT_LENGTH_TOL,
    min_word_delta: int = DEFAULT_MIN_WORD_DELTA,
    fidelity_threshold: float = DEFAULT_FIDELITY_THRESHOLD,
    min_judges_quorum: int = DEFAULT_MIN_JUDGES_QUORUM,
) -> dict[str, Any]:
    """Rewrite all candidate turns of `ep` into `target_tone` and audit fidelity + demeanor."""
    ep_id = str(ep["episode_id"])
    target_turns = [
        (int(t["round"]), str((t.get(target_role) or {}).get("message") or ""))
        for t in ep.get("turns", [])
        if target_role in t
    ]

    async def _one_turn(r: int, orig: str) -> dict[str, Any]:
        async with sem:
            rw = await rewrite_partner_message(
                client=client,
                call_id_prefix=f"{out_run}:{ep_id}:{target_role}:r{r}:{target_tone}",
                mode=target_tone,  # type: ignore[arg-type]
                round_num=r,
                message=orig,
                metadata={"stage": f"capstone_rw_{target_tone}", "episode_id": ep_id, "round": r},
                max_attempts=3,
                length_tol=length_tol,
                min_word_delta=min_word_delta,
            )
            fid_panel, dem_orig, dem_rew = await asyncio.gather(
                panel.decide(
                    call_id=f"{out_run}:{ep_id}:{target_role}:r{r}:{target_tone}:fid",
                    state={"original": orig, "rewritten": rw.rewritten},
                    questions=REWRITE_FIDELITY_QUESTIONS,
                    metadata={"stage": "capstone_fidelity", "episode_id": ep_id, "round": r},
                ),
                panel.decide(
                    call_id=f"{out_run}:{ep_id}:{target_role}:r{r}:dem_orig",
                    state={"task_type": ep["task"], "round": r, "role": target_role, "message": orig},
                    questions=TURN_DEMEANOR_QUESTIONS,
                    metadata={"stage": "capstone_dem_orig", "episode_id": ep_id, "round": r},
                ),
                panel.decide(
                    call_id=f"{out_run}:{ep_id}:{target_role}:r{r}:{target_tone}:dem_rew",
                    state={"task_type": ep["task"], "round": r, "role": target_role, "message": rw.rewritten},
                    questions=TURN_DEMEANOR_QUESTIONS,
                    metadata={"stage": "capstone_dem_rew", "episode_id": ep_id, "round": r},
                ),
            )
        cons = fid_panel.get("panel_consensus") or {}
        c_cons = cons.get("preserves_commitments") or {}
        u_cons = cons.get("preserves_uncertainty") or {}
        c_val = fid_panel.get("preserves_commitments")
        u_val = fid_panel.get("preserves_uncertainty")
        c_n = int(c_cons.get("n_judges", 0))
        u_n = int(u_cons.get("n_judges", 0))
        quorum_ok = (c_n >= min_judges_quorum) and (u_n >= min_judges_quorum)
        panel_fid_ok = bool(
            quorum_ok
            and c_val is not None
            and u_val is not None
            and float(c_val) >= fidelity_threshold
            and float(u_val) >= fidelity_threshold
        )
        # Core automated gate: exact IDs/numbers + exact key=value bindings + length band.
        # Exact negation-count regex is logged separately (semantic negation drops are caught by 3-judge panel).
        auto_core_ok = (not rw.missing_tokens) and (not rw.extra_tokens) and rw.kv_ok and rw.length_ok
        return {
            "round": r,
            "original": orig,
            "rewritten": rw.rewritten,
            "attempts": rw.attempts,
            "tokens_ok": (not rw.missing_tokens) and (not rw.extra_tokens),
            "missing_tokens": rw.missing_tokens,
            "extra_tokens": rw.extra_tokens,
            "kv_ok": rw.kv_ok,
            "neg_ok": rw.neg_ok,
            "length_ok": rw.length_ok,
            "auto_core_ok": auto_core_ok,
            "auto_all_ok": rw.fidelity_ok,
            "orig_words": rw.orig_words,
            "new_words": rw.new_words,
            "length_ratio": round(rw.new_words / max(1, rw.orig_words), 4),
            "panel_commitments": c_val,
            "panel_uncertainty": u_val,
            "panel_commitments_n_judges": c_n,
            "panel_uncertainty_n_judges": u_n,
            "quorum_ok": quorum_ok,
            "panel_fidelity_ok": panel_fid_ok,
            "strict_turn_ok": auto_core_ok and panel_fid_ok,
            "dem_orig": {k: dem_orig.get(k) for k in TURN_DEMEANOR_QUESTIONS},
            "dem_rew": {k: dem_rew.get(k) for k in TURN_DEMEANOR_QUESTIONS},
            "fidelity_panel_raw": fid_panel.get("panel_judges"),
        }

    turn_results = list(await asyncio.gather(*(_one_turn(r, orig) for r, orig in target_turns)))
    rewritten_by_round = {tr["round"]: tr["rewritten"] for tr in turn_results}
    warmths_orig = [float(t["dem_orig"]["warmth"]) for t in turn_results if t["dem_orig"].get("warmth") is not None]
    warmths_rew = [float(t["dem_rew"]["warmth"]) for t in turn_results if t["dem_rew"].get("warmth") is not None]
    mean_w_orig = round(float(np.mean(warmths_orig)), 4) if warmths_orig else None
    mean_w_rew = round(float(np.mean(warmths_rew)), 4) if warmths_rew else None
    comm_vals = [float(t["panel_commitments"]) for t in turn_results if t["panel_commitments"] is not None]
    unc_vals = [float(t["panel_uncertainty"]) for t in turn_results if t["panel_uncertainty"] is not None]
    mean_comm = round(float(np.mean(comm_vals)), 4) if comm_vals else None
    mean_unc = round(float(np.mean(unc_vals)), 4) if unc_vals else None
    min_turn_fid = (
        round(min(min(comm_vals), min(unc_vals)), 4) if (comm_vals and unc_vals) else None
    )
    all_quorum = all(tr["quorum_ok"] for tr in turn_results)
    episode_fidelity_eligible = bool(
        all(tr["auto_core_ok"] for tr in turn_results)
        and all_quorum
        and mean_comm is not None
        and mean_unc is not None
        and min_turn_fid is not None
        and mean_comm >= DEFAULT_EPISODE_MEAN_FIDELITY
        and mean_unc >= DEFAULT_EPISODE_MEAN_FIDELITY
        and min_turn_fid >= DEFAULT_TURN_FLOOR_FIDELITY
    )
    return {
        "episode_id": ep_id,
        "task": str(ep["task"]),
        "seed": int(ep["seed"]),
        "partner": partner,
        "target_role": target_role,
        "target_tone": target_tone,
        "score": float(ep["score"]),
        "all_auto_core_ok": all(tr["auto_core_ok"] for tr in turn_results),
        "all_auto_all_ok": all(tr["auto_all_ok"] for tr in turn_results),
        "all_panel_fidelity_ok": all(tr["panel_fidelity_ok"] for tr in turn_results),
        "all_strict_ok": all(tr["strict_turn_ok"] for tr in turn_results),
        "episode_fidelity_eligible": episode_fidelity_eligible,
        "mean_panel_commitments": mean_comm,
        "mean_panel_uncertainty": mean_unc,
        "min_turn_panel_fidelity": min_turn_fid,
        "turn_strict_pass_rate": round(
            float(np.mean([tr["strict_turn_ok"] for tr in turn_results])), 4
        )
        if turn_results
        else 0.0,
        "mean_warmth_orig": mean_w_orig,
        "mean_warmth_rew": mean_w_rew,
        "mean_warmth_delta": round(mean_w_rew - mean_w_orig, 4)
        if mean_w_orig is not None and mean_w_rew is not None
        else None,
        "turns": turn_results,
        "rewritten_by_round": rewritten_by_round,
    }


def _pair_cluster_bootstrap_and_perm(
    case_rows: list[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 20261122,
) -> dict[str, Any]:
    """Compute case-level crossover tone-flip metrics with pair-cluster bootstrap 95% CIs and sign-flip p-value."""
    if not case_rows:
        return {"n_cases": 0, "n_pairs": 0}

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_pair[str(row["pair_id"])].append(row)
    pair_ids = sorted(by_pair.keys())
    n_pairs = len(pair_ids)

    deltas = [float(r["delta_warm"]) for r in case_rows]
    toward = [1.0 if r["outcome"] == "switched_toward_warm" else 0.0 for r in case_rows]
    away = [1.0 if r["outcome"] == "switched_away_from_warm" else 0.0 for r in case_rows]
    held = [1.0 if r["outcome"] == "held_consistent" else 0.0 for r in case_rows]
    split = [1.0 if r["outcome"] == "order_split" else 0.0 for r in case_rows]
    order_agree = [float(r["order_agree_rate"]) for r in case_rows]

    # Higher-score override (only on non-tied pairs)
    nontied = [r for r in case_rows if r.get("higher_score_preferred_when_blunt") is not None]
    override_when_underdog_warm = [
        1.0 - float(r["higher_score_preferred_when_blunt"]) for r in nontied
    ]
    override_when_underdog_blunt = [
        1.0 - float(r["higher_score_preferred_when_warm"]) for r in nontied
    ]

    # Pair-cluster bootstrap (resampling whole pairs with replacement!)
    rng = np.random.default_rng(seed)
    pair_arrays_delta = [np.array([float(r["delta_warm"]) for r in by_pair[pid]]) for pid in pair_ids]
    pair_arrays_toward = [
        np.array([1.0 if r["outcome"] == "switched_toward_warm" else 0.0 for r in by_pair[pid]])
        for pid in pair_ids
    ]
    pair_arrays_away = [
        np.array([1.0 if r["outcome"] == "switched_away_from_warm" else 0.0 for r in by_pair[pid]])
        for pid in pair_ids
    ]

    boot_delta = np.empty(n_boot, dtype=float)
    boot_toward = np.empty(n_boot, dtype=float)
    boot_away = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        boot_delta[b] = float(np.mean(np.concatenate([pair_arrays_delta[i] for i in idx])))
        boot_toward[b] = float(np.mean(np.concatenate([pair_arrays_toward[i] for i in idx])))
        boot_away[b] = float(np.mean(np.concatenate([pair_arrays_away[i] for i in idx])))

    # Exact two-sided pair-level sign-flip permutation p-value on pair sums (case-weighted)
    from common_ground.info_flip import _exact_case_weighted_sign_flip_p

    pair_sums = [float(np.sum(a)) for a in pair_arrays_delta]
    perm_p, _, _ = _exact_case_weighted_sign_flip_p(pair_sums)

    return {
        "n_cases": len(case_rows),
        "n_pairs": n_pairs,
        "mean_delta_warm": round(float(np.mean(deltas)), 4),
        "delta_warm_pair_boot_ci95": [
            round(float(np.percentile(boot_delta, 2.5)), 4),
            round(float(np.percentile(boot_delta, 97.5)), 4),
        ],
        "pair_sign_flip_perm_p": round(perm_p, 6),
        "switched_toward_warm_rate": round(float(np.mean(toward)), 4),
        "switched_toward_warm_ci95": [
            round(float(np.percentile(boot_toward, 2.5)), 4),
            round(float(np.percentile(boot_toward, 97.5)), 4),
        ],
        "switched_away_from_warm_rate": round(float(np.mean(away)), 4),
        "switched_away_from_warm_ci95": [
            round(float(np.percentile(boot_away, 2.5)), 4),
            round(float(np.percentile(boot_away, 97.5)), 4),
        ],
        "held_consistent_rate": round(float(np.mean(held)), 4),
        "order_split_rate": round(float(np.mean(split)), 4),
        "mean_ab_ba_order_agree_rate": round(float(np.mean(order_agree)), 4),
        "n_nontied_cases": len(nontied),
        "underdog_win_rate_when_warm": round(float(np.mean(override_when_underdog_warm)), 4)
        if override_when_underdog_warm
        else None,
        "underdog_win_rate_when_blunt": round(float(np.mean(override_when_underdog_blunt)), 4)
        if override_when_underdog_blunt
        else None,
        "n_boot": n_boot,
    }


def summarize_capstone_crossover(
    choice_rows: list[dict[str, Any]],
    pair_meta_by_id: dict[str, dict[str, Any]],
    *,
    strict_only: bool = False,
    cross_vendor_only: bool = True,
) -> dict[str, Any]:
    """Collapse 4 calls per `(pair_id, reviewer)` into exact within-pair crossover cases."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in choice_rows:
        if row.get("chosen_slot") not in (1, 2):
            continue
        if strict_only and not row.get("strict_eligible"):
            continue
        if cross_vendor_only and not row.get("cross_vendor"):
            continue
        grouped[(str(row["pair_id"]), str(row["reviewer"]))].append(row)

    cases: list[dict[str, Any]] = []
    for (pid, reviewer), rows in sorted(grouped.items()):
        c_p1w = [r for r in rows if r["condition"] == "p1_warm_p2_blunt"]
        c_p1b = [r for r in rows if r["condition"] == "p1_blunt_p2_warm"]
        if len(c_p1w) != 2 or len(c_p1b) != 2:
            continue

        # Canonical P1 preference in each condition (0.0 = never P1, 0.5 = order split, 1.0 = both orders P1)
        p1_pref_warm = float(np.mean([1.0 if r["chosen_canonical"] == "P1" else 0.0 for r in c_p1w]))
        p1_pref_blunt = float(np.mean([1.0 if r["chosen_canonical"] == "P1" else 0.0 for r in c_p1b]))
        delta_warm = p1_pref_warm - p1_pref_blunt  # +1.0 = flipped cleanly to whichever was Warm!

        agree_w = 1.0 if p1_pref_warm in (0.0, 1.0) else 0.0
        agree_b = 1.0 if p1_pref_blunt in (0.0, 1.0) else 0.0
        order_agree_rate = 0.5 * (agree_w + agree_b)

        if p1_pref_warm == 1.0 and p1_pref_blunt == 0.0:
            outcome = "switched_toward_warm"
        elif p1_pref_warm == 0.0 and p1_pref_blunt == 1.0:
            outcome = "switched_away_from_warm"
        elif p1_pref_warm == p1_pref_blunt and p1_pref_warm in (0.0, 1.0):
            outcome = "held_consistent"
        else:
            outcome = "order_split"

        pm = pair_meta_by_id[pid]
        s1, s2 = float(pm["score_1"]), float(pm["score_2"])
        hi_when_warm: float | None = None
        hi_when_blunt: float | None = None
        if abs(s1 - s2) >= 1e-9:
            if s1 > s2:
                hi_when_warm = p1_pref_warm
                hi_when_blunt = p1_pref_blunt
            else:
                hi_when_warm = 1.0 - p1_pref_blunt
                hi_when_blunt = 1.0 - p1_pref_warm

        cases.append(
            {
                "pair_id": pid,
                "task": pm["task"],
                "gap_bin": pm["gap_bin"],
                "reviewer": reviewer,
                "score_diff": pm["score_diff"],
                "p1_pref_when_p1_warm": p1_pref_warm,
                "p1_pref_when_p1_blunt": p1_pref_blunt,
                "delta_warm": delta_warm,
                "outcome": outcome,
                "order_agree_rate": order_agree_rate,
                "higher_score_preferred_when_warm": hi_when_warm,
                "higher_score_preferred_when_blunt": hi_when_blunt,
            }
        )

    out: dict[str, Any] = {
        "overall": _pair_cluster_bootstrap_and_perm(cases),
        "by_gap_bin": {
            b: _pair_cluster_bootstrap_and_perm([c for c in cases if c["gap_bin"] == b])
            for b in GAP_BINS
        },
        "by_task": {
            t: _pair_cluster_bootstrap_and_perm([c for c in cases if c["task"] == t])
            for t in sorted({c["task"] for c in cases})
        },
        "by_reviewer": {
            rv: _pair_cluster_bootstrap_and_perm([c for c in cases if c["reviewer"] == rv])
            for rv in sorted({c["reviewer"] for c in cases})
        },
    }
    return out


async def run_rewrite_pilot(
    *,
    episodes_dir: Path,
    out_path: Path,
    max_episodes: int = 4,
    budget_cap_usd: float = 5.0,
    mock: bool = False,
    probe_manifest: Path | None = None,
) -> dict[str, Any]:
    """Pilot the exact production Warm + Blunt DeepSeek rewriter & 3-judge gate on real sanity episodes."""
    ep_files = sorted(episodes_dir.glob("*.json"))
    eps = [json.loads(p.read_text(encoding="utf-8")) for p in ep_files]
    eps = [e for e in eps if e.get("complete") is True]
    # Pick balanced mix across available tasks (up to max_episodes)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in eps:
        by_task[str(e["task"])].append(e)
    chosen: list[dict[str, Any]] = []
    per_t = max(1, max_episodes // max(1, len(by_task)))
    for t in sorted(by_task):
        chosen.extend(by_task[t][:per_t])
    chosen = chosen[:max_episodes]

    models = load_verified_models(
        probe_path=probe_manifest or PROBE_MANIFEST_V2_PATH,
        version="v2",
        strict=not mock,
    )
    ledger = Ledger(out_path.parent / "capstone_pilot_ledger.jsonl", budget_cap_usd=budget_cap_usd)
    client: OpenRouterClient | MockOpenRouterClient = (
        MockOpenRouterClient(ledger, models) if mock else OpenRouterClient(ledger, models)
    )
    panel: DecisionPanelClient | MockDecisionPanelClient = (
        MockDecisionPanelClient(ledger) if mock else DecisionPanelClient(ledger)
    )
    sem = asyncio.Semaphore(8)

    records: list[dict[str, Any]] = []
    async with client, panel:
        coros = []
        for idx, ep in enumerate(chosen):
            trole = "A" if idx % 2 == 0 else "B"
            partner = str(ep["model_a"] if trole == "A" else ep["model_b"])
            for tone in ("warm", "blunt"):
                coros.append(
                    _rewrite_and_audit_episode_tone(
                        ep=ep,
                        target_role=trole,
                        partner=partner,
                        target_tone=tone,
                        out_run="capstone_pilot",
                        client=client,
                        panel=panel,
                        sem=sem,
                    )
                )
        records = list(await asyncio.gather(*coros))

    all_turns = [t for r in records for t in r["turns"]]
    by_ep: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        by_ep[r["episode_id"]][r["target_tone"]] = r

    warmth_gaps = []
    for eid, pair_d in by_ep.items():
        w = pair_d["warm"].get("mean_warmth_rew")
        b = pair_d["blunt"].get("mean_warmth_rew")
        if w is not None and b is not None:
            warmth_gaps.append({"episode_id": eid, "warm": w, "blunt": b, "warm_minus_blunt": round(w - b, 4)})

    report = {
        "n_episodes": len(chosen),
        "n_episode_tone_rewrites": len(records),
        "n_turns": len(all_turns),
        "total_cost_usd": round(ledger.total_cost_usd, 4),
        "turn_rates": {
            "tokens_ok": round(float(np.mean([t["tokens_ok"] for t in all_turns])), 4) if all_turns else 0.0,
            "kv_ok": round(float(np.mean([t["kv_ok"] for t in all_turns])), 4) if all_turns else 0.0,
            "neg_ok": round(float(np.mean([t["neg_ok"] for t in all_turns])), 4) if all_turns else 0.0,
            "length_ok": round(float(np.mean([t["length_ok"] for t in all_turns])), 4) if all_turns else 0.0,
            "auto_core_ok": round(float(np.mean([t["auto_core_ok"] for t in all_turns])), 4) if all_turns else 0.0,
            "panel_fidelity_ok": round(float(np.mean([t["panel_fidelity_ok"] for t in all_turns])), 4) if all_turns else 0.0,
            "strict_turn_ok": round(float(np.mean([t["strict_turn_ok"] for t in all_turns])), 4) if all_turns else 0.0,
        },
        "episode_tone_all_strict_rate": round(
            float(np.mean([r["all_strict_ok"] for r in records])), 4
        )
        if records
        else 0.0,
        "warmth_gaps": warmth_gaps,
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(
        f"[bold green]Capstone Rewrite Pilot done:[/bold green] {len(all_turns)} turns, "
        f"strict_turn_ok={report['turn_rates']['strict_turn_ok']:.1%}, "
        f"ep_all_strict={report['episode_tone_all_strict_rate']:.1%}, "
        f"cost=${report['total_cost_usd']:.4f} -> {out_path}"
    )
    return report


async def run_capstone(
    *,
    source_run: str = "v2",
    out_run: str = "v2_capstone",
    n_pairs: int = 24,
    budget_cap_usd: float = 80.0,
    mock: bool = False,
    probe_manifest: Path | None = None,
    allow_partial: bool = False,
    fidelity_threshold: float = DEFAULT_FIDELITY_THRESHOLD,
    min_warmth_delta: float = DEFAULT_MIN_WARMTH_DELTA,
    length_tol: float = DEFAULT_LENGTH_TOL,
    min_word_delta: int = DEFAULT_MIN_WORD_DELTA,
) -> dict[str, Any]:
    source_dir = DATA_DIR / source_run
    out_dir = DATA_DIR / out_run
    if mock and out_run in ("v2_capstone", "v2"):
        raise ValueError(f"Refusing to write --mock Capstone output into live directory '{out_run}'.")
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_files = sorted((source_dir / "episodes").glob("*.json"))
    all_ep_recs = [json.loads(p.read_text(encoding="utf-8")) for p in ep_files]
    episodes = [e for e in all_ep_recs if e.get("complete") is True]
    failed_ids = {str(e["episode_id"]) for e in all_ep_recs if e.get("failed") is True and not e.get("complete")}
    if not episodes:
        raise RuntimeError(f"No completed episodes found in {source_dir / 'episodes'}")

    # Enforce exact manifest execution coverage (completed + explicitly excluded failures) unless allowed
    manifest_path = source_dir / "manifest_v2.json"
    if not allow_partial and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_main = {
            e["manifest_episode_id"]
            for e in manifest.get("episodes", [])
            if e.get("arm") != "opus_boost"
        }
        accounted_ids = {str(e["episode_id"]) for e in episodes} | failed_ids
        missing = sorted(expected_main - accounted_ids)
        if missing:
            raise RuntimeError(
                f"Source run '{source_run}' is missing {len(missing)}/{len(expected_main)} main manifest "
                f"episodes (e.g. {missing[:3]}). Finish main run first or pass --allow-partial."
            )

    pairs, sel_meta = select_disjoint_gap_pairs(episodes, n_pairs=n_pairs)
    console.print(
        f"[bold cyan]Capstone Symmetric Tone-Flip Crossover:[/bold cyan] source={source_run}, "
        f"pairs={len(pairs)} (unique eps={sel_meta['n_unique_episodes']}), bins={sel_meta['by_gap_bin']}"
    )

    # Validate ALL outcome cards up front (fail fast before spending a cent!)
    for p in pairs:
        format_outcome_card(p["ep1"], p["target_role"])
        format_outcome_card(p["ep2"], p["target_role"])

    models = load_verified_models(
        probe_path=probe_manifest or PROBE_MANIFEST_V2_PATH,
        version="v2",
        strict=not mock,
    )
    ledger = Ledger(out_dir / "ledger.jsonl", budget_cap_usd=budget_cap_usd)
    client: OpenRouterClient | MockOpenRouterClient = (
        MockOpenRouterClient(ledger, models) if mock else OpenRouterClient(ledger, models)
    )
    panel: DecisionPanelClient | MockDecisionPanelClient = (
        MockDecisionPanelClient(ledger) if mock else DecisionPanelClient(ledger)
    )
    sem = asyncio.Semaphore(10)

    rewrites_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    choice_rows: list[dict[str, Any]] = []
    choice_exclusions: list[dict[str, Any]] = []

    async with client, panel:
        # 1. Symmetrically rewrite BOTH candidates in every pair into Warm AND Blunt
        rw_coros = []
        rw_keys: list[tuple[str, str]] = []
        for p in pairs:
            for ep, partner in ((p["ep1"], p["partner_1"]), (p["ep2"], p["partner_2"])):
                for tone in ("warm", "blunt"):
                    rw_keys.append((str(ep["episode_id"]), tone))
                    rw_coros.append(
                        _rewrite_and_audit_episode_tone(
                            ep=ep,
                            target_role=p["target_role"],
                            partner=partner,
                            target_tone=tone,
                            out_run=out_run,
                            client=client,
                            panel=panel,
                            sem=sem,
                            length_tol=length_tol,
                            min_word_delta=min_word_delta,
                            fidelity_threshold=fidelity_threshold,
                        )
                    )
        rw_results = await asyncio.gather(*rw_coros)
        for key, res in zip(rw_keys, rw_results, strict=True):
            rewrites_by_key[key] = res

        # 2. Pre-scoring eligibility: fidelity + positive warmth manipulation (Warm > Blunt)
        pair_meta_by_id: dict[str, dict[str, Any]] = {}
        for p in pairs:
            eid1, eid2 = str(p["ep1"]["episode_id"]), str(p["ep2"]["episode_id"])
            r1w, r1b = rewrites_by_key[(eid1, "warm")], rewrites_by_key[(eid1, "blunt")]
            r2w, r2b = rewrites_by_key[(eid2, "warm")], rewrites_by_key[(eid2, "blunt")]
            fid_ok = bool(
                r1w["episode_fidelity_eligible"]
                and r1b["episode_fidelity_eligible"]
                and r2w["episode_fidelity_eligible"]
                and r2b["episode_fidelity_eligible"]
            )
            all_turns_ge_070 = bool(
                r1w["all_strict_ok"]
                and r1b["all_strict_ok"]
                and r2w["all_strict_ok"]
                and r2b["all_strict_ok"]
            )
            w1_gap = (
                (r1w["mean_warmth_rew"] - r1b["mean_warmth_rew"])
                if (r1w["mean_warmth_rew"] is not None and r1b["mean_warmth_rew"] is not None)
                else None
            )
            w2_gap = (
                (r2w["mean_warmth_rew"] - r2b["mean_warmth_rew"])
                if (r2w["mean_warmth_rew"] is not None and r2b["mean_warmth_rew"] is not None)
                else None
            )
            manip_ok = bool(
                (w1_gap is not None and w1_gap >= min_warmth_delta)
                and (w2_gap is not None and w2_gap >= min_warmth_delta)
            )
            pair_meta_by_id[p["pair_id"]] = {
                "pair_id": p["pair_id"],
                "task": p["task"],
                "target_role": p["target_role"],
                "gap_bin": p["gap_bin"],
                "score_1": p["score_1"],
                "score_2": p["score_2"],
                "score_diff": p["score_diff"],
                "partner_1": p["partner_1"],
                "partner_2": p["partner_2"],
                "ep_1_id": eid1,
                "ep_2_id": eid2,
                "p1_warmth_gap": round(w1_gap, 4) if w1_gap is not None else None,
                "p2_warmth_gap": round(w2_gap, 4) if w2_gap is not None else None,
                "strict_fidelity_ok": fid_ok,
                "all_turns_ge_070": all_turns_ge_070,
                "manip_ok": manip_ok,
                "strict_eligible": fid_ok and manip_ok,
            }

        # 3. Build full within-pair crossover specs: 2 conditions × 2 orders × 4 reviewers
        choice_specs: list[dict[str, Any]] = []
        for p in pairs:
            pid = p["pair_id"]
            h_p1, m_p1, h_p2, m_p2 = pair_fixed_handles(pid)
            pm = pair_meta_by_id[pid]
            oc_p1 = format_outcome_card(p["ep1"], p["target_role"])
            oc_p2 = format_outcome_card(p["ep2"], p["target_role"])
            eid1, eid2 = pm["ep_1_id"], pm["ep_2_id"]

            for cond_name, t_p1, t_p2 in (
                ("p1_warm_p2_blunt", "warm", "blunt"),
                ("p1_blunt_p2_warm", "blunt", "warm"),
            ):
                txt_p1, turns_p1 = format_transcript_and_turns(
                    p["ep1"],
                    target_role=p["target_role"],
                    candidate_handle=h_p1,
                    teammate_handle=m_p1,
                    rewritten_by_round=rewrites_by_key[(eid1, t_p1)]["rewritten_by_round"],
                )
                txt_p2, turns_p2 = format_transcript_and_turns(
                    p["ep2"],
                    target_role=p["target_role"],
                    candidate_handle=h_p2,
                    teammate_handle=m_p2,
                    rewritten_by_round=rewrites_by_key[(eid2, t_p2)]["rewritten_by_round"],
                )
                for order in ("AB", "BA"):
                    if order == "AB":
                        slot1 = ("P1", h_p1, p["partner_1"], t_p1, p["score_1"], oc_p1, txt_p1, turns_p1, eid1)
                        slot2 = ("P2", h_p2, p["partner_2"], t_p2, p["score_2"], oc_p2, txt_p2, turns_p2, eid2)
                    else:
                        slot1 = ("P2", h_p2, p["partner_2"], t_p2, p["score_2"], oc_p2, txt_p2, turns_p2, eid2)
                        slot2 = ("P1", h_p1, p["partner_1"], t_p1, p["score_1"], oc_p1, txt_p1, turns_p1, eid1)
                    packet_id = f"{pid}__{cond_name}__{order}"
                    prompt = capstone_third_person_choice_prompt(
                        task=p["task"],
                        handle_1=slot1[1],
                        handle_2=slot2[1],
                        transcript_1=slot1[6],
                        transcript_2=slot2[6],
                        outcome_card_1=slot1[5],
                        outcome_card_2=slot2[5],
                    )
                    for reviewer in CORE_MODELS:
                        choice_specs.append(
                            {
                                "packet_id": packet_id,
                                "pair_id": pid,
                                "task": p["task"],
                                "target_role": p["target_role"],
                                "gap_bin": p["gap_bin"],
                                "score_diff": p["score_diff"],
                                "condition": cond_name,
                                "order": order,
                                "reviewer": reviewer,
                                "slot1": slot1,
                                "slot2": slot2,
                                "prompt": prompt,
                                "strict_fidelity_ok": pm["strict_fidelity_ok"],
                                "manip_ok": pm["manip_ok"],
                                "strict_eligible": pm["strict_eligible"],
                            }
                        )

        async def _run_choice(cs: dict[str, Any]) -> dict[str, Any] | None:
            s1, s2 = cs["slot1"], cs["slot2"]
            try:
                async with sem:
                    entry = await client.complete(
                        call_id=f"{out_run}:ch:{cs['packet_id']}:{cs['reviewer']}",
                        model_key=cs["reviewer"],
                        messages=[{"role": "user", "content": cs["prompt"]}],
                        metadata={
                            "stage": "capstone_choice",
                            "packet_id": cs["packet_id"],
                            "pair_id": cs["pair_id"],
                            "condition": cs["condition"],
                            "order": cs["order"],
                            "task": cs["task"],
                            "gap_bin": cs["gap_bin"],
                            "reviewer": cs["reviewer"],
                            "handle_1": s1[1],
                            "handle_2": s2[1],
                            "partner_1": s1[2],
                            "partner_2": s2[2],
                            "tone_1": s1[3],
                            "tone_2": s2[3],
                        },
                    )
            except OpenRouterCallError as exc:
                choice_exclusions.append(
                    {
                        "packet_id": cs["packet_id"],
                        "pair_id": cs["pair_id"],
                        "reviewer": cs["reviewer"],
                        "error": str(exc),
                    }
                )
                return None

            chosen_slot = _parse_choice(entry.response_text, s1[1], s2[1])
            chosen_canonical = s1[0] if chosen_slot == 1 else (s2[0] if chosen_slot == 2 else None)
            chosen_partner = s1[2] if chosen_slot == 1 else (s2[2] if chosen_slot == 2 else None)
            chosen_tone = s1[3] if chosen_slot == 1 else (s2[3] if chosen_slot == 2 else None)
            return {
                "packet_id": cs["packet_id"],
                "pair_id": cs["pair_id"],
                "task": cs["task"],
                "target_role": cs["target_role"],
                "gap_bin": cs["gap_bin"],
                "score_diff": cs["score_diff"],
                "condition": cs["condition"],
                "order": cs["order"],
                "reviewer": cs["reviewer"],
                "handle_1": s1[1],
                "handle_2": s2[1],
                "canonical_1": s1[0],
                "canonical_2": s2[0],
                "partner_1": s1[2],
                "partner_2": s2[2],
                "tone_1": s1[3],
                "tone_2": s2[3],
                "score_1": s1[4],
                "score_2": s2[4],
                "episode_1_id": s1[8],
                "episode_2_id": s2[8],
                "outcome_card_1": s1[5],
                "outcome_card_2": s2[5],
                "transcript_1_turns": s1[7],
                "transcript_2_turns": s2[7],
                "prompt": cs["prompt"],
                "chosen_slot": chosen_slot,
                "chosen_canonical": chosen_canonical,
                "chosen_partner": chosen_partner,
                "chosen_tone": chosen_tone,
                "cross_vendor": (
                    _model_family(cs["reviewer"])
                    not in {_model_family(s1[2]), _model_family(s2[2])}
                ),
                "strict_fidelity_ok": cs["strict_fidelity_ok"],
                "manip_ok": cs["manip_ok"],
                "strict_eligible": cs["strict_eligible"],
                "raw": entry.response_text,
                "cost_usd": entry.cost_usd,
            }

        try:
            raw_res = await asyncio.gather(*(_run_choice(cs) for cs in choice_specs))
            choice_rows = [r for r in raw_res if r is not None]
        except BudgetExceededError as exc:
            console.print(f"[bold red]Capstone stopped on budget cap: {exc}[/bold red]")

    # Summarize rewrite manipulation checks + attrition by task & gap bin
    all_rw_recs = list(rewrites_by_key.values())
    warm_turns = [t for r in all_rw_recs if r["target_tone"] == "warm" for t in r["turns"]]
    blunt_turns = [t for r in all_rw_recs if r["target_tone"] == "blunt" for t in r["turns"]]

    def _turn_diag(tlist: list[dict[str, Any]]) -> dict[str, Any]:
        if not tlist:
            return {"n_turns": 0}
        deltas: dict[str, float | None] = {}
        for feat in TURN_DEMEANOR_QUESTIONS:
            ds = [
                float(t["dem_rew"][feat]) - float(t["dem_orig"][feat])
                for t in tlist
                if t["dem_rew"].get(feat) is not None and t["dem_orig"].get(feat) is not None
            ]
            deltas[feat] = round(float(np.mean(ds)), 4) if ds else None
        return {
            "n_turns": len(tlist),
            "tokens_ok_rate": round(float(np.mean([t["tokens_ok"] for t in tlist])), 4),
            "kv_ok_rate": round(float(np.mean([t["kv_ok"] for t in tlist])), 4),
            "neg_ok_rate": round(float(np.mean([t["neg_ok"] for t in tlist])), 4),
            "length_ok_rate": round(float(np.mean([t["length_ok"] for t in tlist])), 4),
            "auto_core_ok_rate": round(float(np.mean([t["auto_core_ok"] for t in tlist])), 4),
            "panel_fidelity_ok_rate": round(float(np.mean([t["panel_fidelity_ok"] for t in tlist])), 4),
            "strict_turn_ok_rate": round(float(np.mean([t["strict_turn_ok"] for t in tlist])), 4),
            "demeanor_delta_mean": deltas,
        }

    attrition_by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "fid_ok": 0, "manip_ok": 0, "eligible": 0})
    attrition_by_bin: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "fid_ok": 0, "manip_ok": 0, "eligible": 0})
    for pm in pair_meta_by_id.values():
        for bucket in (attrition_by_task[pm["task"]], attrition_by_bin[pm["gap_bin"]]):
            bucket["total"] += 1
            bucket["fid_ok"] += int(pm["strict_fidelity_ok"])
            bucket["manip_ok"] += int(pm["manip_ok"])
            bucket["eligible"] += int(pm["strict_eligible"])

    summary = {
        "schema_version": "v2_symmetric_crossover",
        "source_run": source_run,
        "out_run": out_run,
        "total_cost_usd": round(ledger.total_cost_usd, 4),
        "thresholds": {
            "fidelity_threshold": fidelity_threshold,
            "min_warmth_delta": min_warmth_delta,
            "min_judges_quorum": DEFAULT_MIN_JUDGES_QUORUM,
            "length_tol": length_tol,
            "min_word_delta": min_word_delta,
        },
        "selection": sel_meta,
        "attrition": {
            "pairs_total": len(pair_meta_by_id),
            "pairs_strict_fidelity_ok": sum(1 for pm in pair_meta_by_id.values() if pm["strict_fidelity_ok"]),
            "pairs_manip_ok": sum(1 for pm in pair_meta_by_id.values() if pm["manip_ok"]),
            "pairs_strict_eligible": sum(1 for pm in pair_meta_by_id.values() if pm["strict_eligible"]),
            "by_task": dict(attrition_by_task),
            "by_gap_bin": dict(attrition_by_bin),
        },
        "manipulation_check": {
            "warm_rewrites": _turn_diag(warm_turns),
            "blunt_rewrites": _turn_diag(blunt_turns),
        },
        "crossover_summary": {
            "itt_cross_vendor": summarize_capstone_crossover(
                choice_rows, pair_meta_by_id, strict_only=False, cross_vendor_only=True
            ),
            "strict_cross_vendor": summarize_capstone_crossover(
                choice_rows, pair_meta_by_id, strict_only=True, cross_vendor_only=True
            ),
            "itt_all_reviewers": summarize_capstone_crossover(
                choice_rows, pair_meta_by_id, strict_only=False, cross_vendor_only=False
            ),
            "strict_all_reviewers": summarize_capstone_crossover(
                choice_rows, pair_meta_by_id, strict_only=True, cross_vendor_only=False
            ),
        },
        "n_choices_total": len(choice_rows),
        "n_choices_excluded": len(choice_exclusions),
        "pairs": list(pair_meta_by_id.values()),
        "choice_exclusions": choice_exclusions,
    }

    (out_dir / "capstone_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Write lossless JSONL records into source_dir/capstone_tone_flip/ for HuggingFace Parquet bundling + replay
    hf_cap_dir = source_dir / "capstone_tone_flip"
    hf_cap_dir.mkdir(parents=True, exist_ok=True)
    with (hf_cap_dir / "capstone_choices.jsonl").open("w", encoding="utf-8") as fh:
        for row in choice_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (hf_cap_dir / "capstone_rewrites.jsonl").open("w", encoding="utf-8") as fh:
        for rec in all_rw_recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (hf_cap_dir / "capstone_pairs.json").write_text(
        json.dumps(list(pair_meta_by_id.values()), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    itt_cv = summary["crossover_summary"]["itt_cross_vendor"]["overall"]
    console.print(
        f"[bold green]Capstone Symmetric Tone-Flip complete:[/bold green] {len(choice_rows)} choices "
        f"(excluded={len(choice_exclusions)}), ITT cross-vendor ΔWarm={itt_cv.get('mean_delta_warm')}, "
        f"switched_toward_warm={itt_cv.get('switched_toward_warm_rate')}, "
        f"cost=${summary['total_cost_usd']:.2f} -> {out_dir / 'capstone_results.json'}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Symmetric Capstone Causal Tone-Flip study.")
    parser.add_argument("--source-run", type=str, default="v2")
    parser.add_argument("--out-run", type=str, default="v2_capstone")
    parser.add_argument("--n-pairs", type=int, default=24)
    parser.add_argument("--budget-cap", type=float, default=80.0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--probe-manifest", type=Path, default=None)
    parser.add_argument("--pilot-episodes-dir", type=Path, default=None)
    parser.add_argument("--pilot-out", type=Path, default=None)
    parser.add_argument("--pilot-max-episodes", type=int, default=4)
    args = parser.parse_args()

    if args.pilot_episodes_dir is not None:
        out_p = args.pilot_out or (DATA_DIR / "capstone_pilot" / "capstone_rewrite_pilot.json")
        asyncio.run(
            run_rewrite_pilot(
                episodes_dir=args.pilot_episodes_dir,
                out_path=out_p,
                max_episodes=args.pilot_max_episodes,
                budget_cap_usd=min(args.budget_cap, 10.0),
                mock=args.mock,
                probe_manifest=args.probe_manifest,
            )
        )
        return

    asyncio.run(
        run_capstone(
            source_run=args.source_run,
            out_run=args.out_run,
            n_pairs=args.n_pairs,
            budget_cap_usd=args.budget_cap,
            mock=args.mock,
            probe_manifest=args.probe_manifest,
            allow_partial=args.allow_partial,
        )
    )


if __name__ == "__main__":
    main()
