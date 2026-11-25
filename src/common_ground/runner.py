"""Async experiment orchestrator: v2 manifest freezing, episodes, F1/F1_retest/F2/F4/F5 forks, 3-judge panel, and symmetric forced choices."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from common_ground.client import MockOpenRouterClient, OpenRouterCallError, OpenRouterClient
from common_ground.config import (
    ALL_PARTNER_MODELS,
    CORE_MODELS,
    DATA_DIR,
    PARTNER_ONLY_MODELS,
    PREDECESSOR_MODEL,
    PROBE_MANIFEST_V2_PATH,
    RunConfig,
    load_verified_models,
)
from common_ground.jev import (
    CONFUSION_QUESTIONS,
    REWRITE_FIDELITY_QUESTIONS,
    TURN_DEMEANOR_QUESTIONS,
    DecisionPanelClient,
    MockDecisionPanelClient,
    false_blame_questions,
)
from common_ground.ledger import BudgetExceededError, Ledger
from common_ground.metrics import audit_quote, compute_message_metrics
from common_ground.prompts import (
    FORMAT_RETRY_PROMPT,
    boundary_choice_prompt,
    debrief_prompt,
    family_guess_options,
    family_guess_prompt,
    forced_choice_prompt,
)
from common_ground.rewriter import audit_rewrite_coherence, rewrite_partner_message
from common_ground.tasks import (
    BLIND_HANDLES,
    DEFAULT_MAIN_TASKS,
    make_blind_handles,
    make_task,
    parse_turn_response,
)

console = Console()

V2_MAIN_SEEDS: tuple[int, ...] = (301, 302, 303, 304)
V2_OPUS_BOOST_SEEDS: tuple[int, ...] = (305, 306, 307, 308)


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```.*$", "", s, flags=re.DOTALL).strip()
    idx = s.find("{")
    if idx == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[idx:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_keep_reroll(raw: str) -> str | None:
    up = (raw or "").strip().upper()
    if up in ("KEEP", "REROLL"):
        return up
    m = re.search(r"\b(KEEP|REROLL)\b", up)
    return m.group(1) if m else None


def episode_id_for(task: str, seed: int, model_a: str, model_b: str) -> str:
    return f"{task}__s{seed}__{model_a}__vs__{model_b}"


def build_calibrate_episode_specs() -> list[dict[str, Any]]:
    """Fast 14-episode calibration mini-pilot."""
    specs: list[dict[str, Any]] = [
        {"task": "triage", "seed": 1, "model_a": "astra62", "model_b": "opus55", "arm": "core"},
        {"task": "triage", "seed": 1, "model_a": "opus55", "model_b": "astra62", "arm": "core"},
        {"task": "triage", "seed": 1, "model_a": "neon4", "model_b": "kimi31", "arm": "core"},
        {"task": "triage", "seed": 1, "model_a": "astra60", "model_b": "opus55", "arm": "predecessor"},
        {"task": "triage", "seed": 2, "model_a": "kimi31", "model_b": "neon4", "arm": "core"},
        {"task": "triage", "seed": 2, "model_a": "astra62", "model_b": "neon4", "arm": "core"},
    ]
    cal_gp_pairs = [
        ("astra62", "opus55"),
        ("opus55", "neon4"),
        ("neon4", "kimi31"),
        ("kimi31", "astra62"),
    ]
    for task in ("calendar", "goalpost"):
        for ma, mb in cal_gp_pairs:
            specs.append({"task": task, "seed": 1, "model_a": ma, "model_b": mb, "arm": "core"})
    return specs


def build_pilot_episode_specs() -> list[dict[str, Any]]:
    """Legacy 48-episode pilot schedule for backward-compatible pipeline tests."""
    specs: list[dict[str, Any]] = []
    cal_pairs = [
        ("astra62", "opus55"),
        ("opus55", "neon4"),
        ("neon4", "kimi31"),
        ("kimi31", "astra62"),
    ]
    for task in ("triage", "calendar"):
        for ma, mb in cal_pairs:
            specs.append({"task": task, "seed": 1, "model_a": ma, "model_b": mb, "arm": "core"})
    for seed in (1, 2):
        for ma in CORE_MODELS:
            for mb in CORE_MODELS:
                specs.append({"task": "goalpost", "seed": seed, "model_a": ma, "model_b": mb, "arm": "core"})
        for subj in CORE_MODELS:
            ma, mb = (subj, PREDECESSOR_MODEL) if seed == 1 else (PREDECESSOR_MODEL, subj)
            specs.append({"task": "goalpost", "seed": seed, "model_a": ma, "model_b": mb, "arm": "predecessor"})
    return specs


def _walecki_7partner_edges(
    partners: tuple[str, ...], seeds: list[int], perm_key: str
) -> dict[tuple[str, str], dict[str, list[tuple[int, int]]]]:
    """Exact Walecki 3-cycle decomposition of K_7 onto 4 distinct seeds with Role-B complement.

    Guarantees:
    - All C(7,2) = 21 unordered partner pairs covered with 2 distinct-seed pairings per role.
    - Inside each role (A and B separately): every partner uses each of the 4 seeds EXACTLY 3 times (skew = 0 <= 1).
    - Summed across Roles A+B: every partner uses each seed EXACTLY 6 times (skew = 0), every partner pair uses
      each seed for each partner EXACTLY 1 time (skew = 0), and every ordered seed pair (s1 != s2) appears
      EXACTLY 7 times (skew = 0).
    """
    assert len(partners) == 7, f"Expected 7 partners, got {len(partners)}"
    assert len(seeds) == 4 and len(set(seeds)) == 4, f"Expected 4 distinct seeds, got {seeds}"

    h = int(hashlib.sha256(perm_key.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(h)
    s = list(seeds)
    rng.shuffle(s)

    # 3 edge-disjoint directed Hamiltonian cycles covering K_7 (steps 1, 2, 3 mod 7)
    cycle_pairings = [
        ((s[0], s[2]), (s[1], s[3])),  # cycle step 1: u gets {0,1}, v gets {2,3}
        ((s[0], s[1]), (s[2], s[3])),  # cycle step 2: u gets {0,2}, v gets {1,3}
        ((s[0], s[3]), (s[1], s[2])),  # cycle step 3: u gets {0,3}, v gets {1,2}
    ]
    edge_map: dict[tuple[str, str], dict[str, list[tuple[int, int]]]] = {}
    for step_idx, ((a1, b1), (a2, b2)) in enumerate(cycle_pairings, start=1):
        for u in range(7):
            v = (u + step_idx) % 7
            pu, pv = partners[u], partners[v]
            # Canonical orientation (pu, pv) sorted by index in partners
            if u < v:
                ra = [(a1, b1), (a2, b2)]
                rb = [(b1, a1), (b2, a2)]
                key = (pu, pv)
            else:
                ra = [(b1, a1), (b2, a2)]
                rb = [(a1, b1), (a2, b2)]
                key = (pv, pu)
            assert key not in edge_map
            edge_map[key] = {"A": ra, "B": rb}
    assert len(edge_map) == 21
    return edge_map


def build_v2_manifest(
    *,
    main_seeds: tuple[int, ...] = V2_MAIN_SEEDS,
    boost_seeds: tuple[int, ...] = V2_OPUS_BOOST_SEEDS,
    tasks: tuple[str, ...] = DEFAULT_MAIN_TASKS,
    shuffle_seed: int = 20261122,
) -> dict[str, Any]:
    """Build and verify the exact frozen v2 experiment manifest BEFORE running any live calls.

    Exact structure:
    - 640 main episodes: 4 tasks x 4 unseen seeds (301-304) x 40 eps/cell
      (16 core x core + 24 core x 3 partner-only x 2 roles).
    - 64 Opus predecessor power boost episodes: 4 tasks x 4 extra unseen seeds (305-308)
      x opus55 x {astra62, astra60} x 2 roles.
    - 224 stratified F1_retest agent-episodes (exact 25.0% of the 896 main core-subject agent-episodes;
      1 of 4 seeds per (task, subject, partner, role) cell, exact 56 per seed).
    - 3,264 forced choices in deterministically shuffled complete AB/BA blocks:
      * 2,688 main_7p choices across all C(7,2)=21 partner pairs (exact 0 seed skew in every role & across A+B).
      * 256 aa_placebo choices (exact 0 seed skew across A+B).
      * 192 extra_h2h cross-vendor (opus55, neon4, kimi31) astra62 vs astra60 choices strictly disjoint from main_7p.
      * 128 opus_boost_h2h opus55 astra62 vs astra60 choices on boost seeds 305-308.
    """
    assert len(main_seeds) == 4 and len(set(main_seeds)) == 4
    assert len(boost_seeds) == 4 and len(set(boost_seeds)) == 4
    assert set(main_seeds).isdisjoint(set(boost_seeds))

    episodes: list[dict[str, Any]] = []
    ep_id_set: set[str] = set()

    # Deterministic 25% stratified F1_retest map: 1 of 4 main seeds per (task, subj, partner, role)
    retest_seed_for: dict[tuple[str, str, str, str], int] = {}
    for t_i, task in enumerate(tasks):
        for s_i, subj in enumerate(CORE_MODELS):
            for p_i, part in enumerate(ALL_PARTNER_MODELS):
                for r_i, role in enumerate(("A", "B")):
                    chosen_idx = (t_i + s_i + p_i + r_i) % 4
                    retest_seed_for[(task, subj, part, role)] = main_seeds[chosen_idx]

    # 1. Main 640 episodes
    for seed in main_seeds:
        for task in tasks:
            for ma in CORE_MODELS:
                for mb in CORE_MODELS:
                    ep_id = episode_id_for(task, seed, ma, mb)
                    retest_roles = []
                    if retest_seed_for[(task, ma, mb, "A")] == seed:
                        retest_roles.append("A")
                    if retest_seed_for[(task, mb, ma, "B")] == seed:
                        retest_roles.append("B")
                    episodes.append(
                        {
                            "manifest_episode_id": ep_id,
                            "task": task,
                            "seed": seed,
                            "model_a": ma,
                            "model_b": mb,
                            "arm": "core",
                            "f1_retest_roles": retest_roles,
                        }
                    )
                    ep_id_set.add(ep_id)
            for subj in CORE_MODELS:
                for po in PARTNER_ONLY_MODELS:
                    ep_a = episode_id_for(task, seed, subj, po)
                    ep_b = episode_id_for(task, seed, po, subj)
                    episodes.append(
                        {
                            "manifest_episode_id": ep_a,
                            "task": task,
                            "seed": seed,
                            "model_a": subj,
                            "model_b": po,
                            "arm": "partner_only",
                            "f1_retest_roles": ["A"] if retest_seed_for[(task, subj, po, "A")] == seed else [],
                        }
                    )
                    episodes.append(
                        {
                            "manifest_episode_id": ep_b,
                            "task": task,
                            "seed": seed,
                            "model_a": po,
                            "model_b": subj,
                            "arm": "partner_only",
                            "f1_retest_roles": ["B"] if retest_seed_for[(task, subj, po, "B")] == seed else [],
                        }
                    )
                    ep_id_set.add(ep_a)
                    ep_id_set.add(ep_b)

    # 2. Opus predecessor power boost (64 episodes)
    for seed in boost_seeds:
        for task in tasks:
            for partner in ("astra62", PREDECESSOR_MODEL):
                for ma, mb in (("opus55", partner), (partner, "opus55")):
                    ep_id = episode_id_for(task, seed, ma, mb)
                    episodes.append(
                        {
                            "manifest_episode_id": ep_id,
                            "task": task,
                            "seed": seed,
                            "model_a": ma,
                            "model_b": mb,
                            "arm": "opus_boost",
                            "f1_retest_roles": [],
                        }
                    )
                    ep_id_set.add(ep_id)

    assert len(episodes) == 704, f"Expected 704 episodes, got {len(episodes)}"
    assert len(ep_id_set) == 704, "Duplicate episode IDs in v2 manifest!"

    # Verify exact 25% stratified F1_retest balance (224 total, 56/seed, 56/task, 56/subject, 32/partner, 112/role)
    retest_count = sum(len(e["f1_retest_roles"]) for e in episodes)
    assert retest_count == 224, f"Expected 224 F1_retest agent-episodes, got {retest_count}"
    rt_by_seed = Counter()
    for e in episodes:
        rt_by_seed[e["seed"]] += len(e["f1_retest_roles"])
    assert all(rt_by_seed[s] == 56 for s in main_seeds), f"Unbalanced F1_retest seeds: {rt_by_seed}"

    def _ep_id(subj: str, partner: str, task: str, role: str, seed: int) -> str:
        eid = (
            episode_id_for(task, seed, subj, partner)
            if role == "A"
            else episode_id_for(task, seed, partner, subj)
        )
        assert eid in ep_id_set, f"Referenced episode missing from manifest: {eid}"
        return eid

    # Build choice blocks (each block = complete AB + BA pair for an item_id)
    choice_blocks: list[list[dict[str, Any]]] = []
    seen_ordered_pairs: set[tuple[str, str, str, str, str]] = set()

    def _add_ab_ba(
        *,
        choice_type: str,
        item_id: str,
        subj: str,
        task: str,
        role: str,
        p1: str,
        s1: int,
        p2: str,
        s2: int,
        allow_same_seed: bool = False,
    ) -> None:
        if not allow_same_seed:
            assert s1 != s2, f"Same-seed splicing disallowed for {choice_type}: {item_id} ({s1} vs {s2})"
        e1 = _ep_id(subj, p1, task, role, s1)
        e2 = _ep_id(subj, p2, task, role, s2)
        pair_items: list[dict[str, Any]] = []
        for order_tag, fp, fs, fe, sp, ss, se in (
            ("AB", p1, s1, e1, p2, s2, e2),
            ("BA", p2, s2, e2, p1, s1, e1),
        ):
            sig = (subj, task, role, fe, se)
            assert sig not in seen_ordered_pairs, f"Duplicate choice tuple in manifest: {sig}"
            seen_ordered_pairs.add(sig)
            mid = f"{choice_type}__{item_id}__{order_tag}"
            pair_items.append(
                {
                    "manifest_item_id": mid,
                    "block_id": f"{choice_type}__{item_id}",
                    "choice_type": choice_type,
                    "item_id": item_id,
                    "order": order_tag,
                    "same_seed": (fs == ss),
                    "subject_model": subj,
                    "task": task,
                    "role": role,
                    "partner_1": fp,
                    "partner_2": sp,
                    "seed_1": fs,
                    "seed_2": ss,
                    "ep_1_id": fe,
                    "ep_2_id": se,
                }
            )
        choice_blocks.append(pair_items)

    # 3a. Main 7-partner between-partner choices (2,688) + 3c. Extra cross-vendor astra62 vs astra60 (192)
    for subj in CORE_MODELS:
        for task in tasks:
            edge_map = _walecki_7partner_edges(
                ALL_PARTNER_MODELS, list(main_seeds), perm_key=f"v2:{subj}:{task}"
            )
            # Hard balance checks on main_7p seed allocation
            counts_by_role: dict[str, dict[str, Counter[int]]] = {
                "A": {p: Counter() for p in ALL_PARTNER_MODELS},
                "B": {p: Counter() for p in ALL_PARTNER_MODELS},
            }
            ordered_seed_pairs_sum: Counter[tuple[int, int]] = Counter()

            for (p1, p2), role_pairs in edge_map.items():
                pair_partner_sum: dict[str, Counter[int]] = {p1: Counter(), p2: Counter()}
                for role in ("A", "B"):
                    for s1, s2 in role_pairs[role]:
                        counts_by_role[role][p1][s1] += 1
                        counts_by_role[role][p2][s2] += 1
                        pair_partner_sum[p1][s1] += 1
                        pair_partner_sum[p2][s2] += 1
                        ordered_seed_pairs_sum[(s1, s2)] += 1
                        item_id = f"m7p__{subj}__{task}__{role}__{p1}_s{s1}__vs__{p2}_s{s2}"
                        _add_ab_ba(
                            choice_type="main_7p",
                            item_id=item_id,
                            subj=subj,
                            task=task,
                            role=role,
                            p1=p1,
                            s1=s1,
                            p2=p2,
                            s2=s2,
                        )
                # Exact 0 skew per partner inside every partner pair across Roles A+B
                for p in (p1, p2):
                    vals = [pair_partner_sum[p][sd] for sd in main_seeds]
                    assert max(vals) - min(vals) == 0, f"Pair skew for {(subj, task, p1, p2, p)}: {vals}"

            # Hard assert: within ±1 (actually exact 0) per seed inside each role, exact 0 across A+B
            for p in ALL_PARTNER_MODELS:
                for role in ("A", "B"):
                    vals = [counts_by_role[role][p][sd] for sd in main_seeds]
                    assert max(vals) - min(vals) <= 1, f"Role skew for {(subj, task, role, p)}: {vals}"
                summed = [counts_by_role["A"][p][sd] + counts_by_role["B"][p][sd] for sd in main_seeds]
                assert max(summed) - min(summed) == 0, f"A+B skew for {(subj, task, p)}: {summed}"
            all_osp = [ordered_seed_pairs_sum[(a, b)] for a in main_seeds for b in main_seeds if a != b]
            assert max(all_osp) - min(all_osp) == 0, f"Ordered seed-pair skew: {all_osp}"

            # 3b. A/A order-consistency placebos (256 total: 4 subj x 4 task x 2 role x 4 core x 2 orders)
            h_aa = int(hashlib.sha256(f"aa:{subj}:{task}".encode()).hexdigest()[:12], 16)
            s_aa = list(main_seeds)
            random.Random(h_aa).shuffle(s_aa)
            for p_idx, p in enumerate(CORE_MODELS):
                pair_a = (s_aa[0], s_aa[1]) if p_idx % 2 == 0 else (s_aa[2], s_aa[3])
                pair_b = (s_aa[2], s_aa[3]) if p_idx % 2 == 0 else (s_aa[0], s_aa[1])
                for role, (sa1, sa2) in (("A", pair_a), ("B", pair_b)):
                    item_id = f"aa__{subj}__{task}__{role}__{p}__s{sa1}_vs_s{sa2}"
                    _add_ab_ba(
                        choice_type="aa_placebo",
                        item_id=item_id,
                        subj=subj,
                        task=task,
                        role=role,
                        p1=p,
                        s1=sa1,
                        p2=p,
                        s2=sa2,
                    )

            # 3c. Extra cross-vendor direct astra62 vs astra60 SAME-SEED matched-scenario choices (192 total = 96 cases)
            if subj != "astra62":
                for role in ("A", "B"):
                    for sd in main_seeds:
                        item_id = f"eh2h__{subj}__{task}__{role}__astra62_s{sd}__vs__astra60_s{sd}"
                        _add_ab_ba(
                            choice_type="extra_h2h",
                            item_id=item_id,
                            subj=subj,
                            task=task,
                            role=role,
                            p1="astra62",
                            s1=sd,
                            p2=PREDECESSOR_MODEL,
                            s2=sd,
                            allow_same_seed=True,
                        )

    # 3d. Opus boost astra62 vs astra60 SAME-SEED matched-scenario choices (64 total = 32 distinct zero-reuse cases)
    for task in tasks:
        for role in ("A", "B"):
            for sd in boost_seeds:
                item_id = f"obh2h__opus55__{task}__{role}__astra62_s{sd}__vs__astra60_s{sd}"
                _add_ab_ba(
                    choice_type="opus_boost_h2h",
                    item_id=item_id,
                    subj="opus55",
                    task=task,
                    role=role,
                    p1="astra62",
                    s1=sd,
                    p2=PREDECESSOR_MODEL,
                    s2=sd,
                    allow_same_seed=True,
                )

    # Shuffle complete AB/BA blocks deterministically
    rng_shuf = random.Random(shuffle_seed)
    rng_shuf.shuffle(choice_blocks)
    choices_flat = [item for block in choice_blocks for item in block]

    by_type = Counter(c["choice_type"] for c in choices_flat)
    assert by_type["main_7p"] == 2688, f"Expected 2688 main_7p, got {by_type['main_7p']}"
    assert by_type["aa_placebo"] == 256, f"Expected 256 aa_placebo, got {by_type['aa_placebo']}"
    assert by_type["extra_h2h"] == 192, f"Expected 192 extra_h2h, got {by_type['extra_h2h']}"
    assert by_type["opus_boost_h2h"] == 64, f"Expected 64 opus_boost_h2h, got {by_type['opus_boost_h2h']}"
    assert len(choices_flat) == 3200, f"Expected 3200 total choices, got {len(choices_flat)}"
    assert len({c["manifest_item_id"] for c in choices_flat}) == 3200

    return {
        "schema_version": "v2",
        "main_seeds": list(main_seeds),
        "boost_seeds": list(boost_seeds),
        "tasks": list(tasks),
        "core_models": list(CORE_MODELS),
        "partner_only_models": list(PARTNER_ONLY_MODELS),
        "all_partner_models": list(ALL_PARTNER_MODELS),
        "counts": {
            "episodes_total": len(episodes),
            "episodes_main": sum(1 for e in episodes if e["arm"] in ("core", "partner_only")),
            "episodes_opus_boost": sum(1 for e in episodes if e["arm"] == "opus_boost"),
            "f1_retest_agent_episodes": retest_count,
            "choices_total": len(choices_flat),
            "choices_by_type": dict(by_type),
            "choice_blocks": len(choice_blocks),
        },
        "episodes": episodes,
        "choices": choices_flat,
    }


def freeze_v2_manifest(out_path: Path, **kwargs: Any) -> dict[str, Any]:
    manifest = build_v2_manifest(**kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _replace_partner_messages_in_history(
    history: list[dict[str, Any]],
    role: str,
    partner_handle: str,
    rewritten_by_round: dict[int, str],
) -> list[dict[str, Any]]:
    """Build a counterfactual history replacing ONLY partner messages with rewritten prose."""
    out: list[dict[str, Any]] = []
    marker = f"Teammate {partner_handle} said:\n"
    for msg in history:
        if msg["role"] != "user":
            out.append(dict(msg))
            continue
        content = str(msg["content"])
        m_rnd = re.search(r"=== ROUND (\d+) of \d+ ===", content)
        rnd = int(m_rnd.group(1)) if m_rnd else None
        replaced = content
        if marker in content and rnd is not None:
            source_rnd = (rnd - 1) if role == "A" else rnd
            new_msg = rewritten_by_round.get(source_rnd)
            if new_msg is not None:
                prefix, rest = content.split(marker, 1)
                suffix_split = re.split(
                    r"\n\n(?=ACTIVE QUERY|FINAL ROUND:|End with `ACTION:|Submit updates)",
                    rest,
                    maxsplit=1,
                )
                tail = f"\n\n{suffix_split[1]}" if len(suffix_split) == 2 else ""
                replaced = f"{prefix}{marker}{new_msg}{tail}"
        out.append({"role": msg["role"], "content": replaced})
    return out


async def run_single_episode(
    *,
    client: OpenRouterClient | MockOpenRouterClient,
    jev: DecisionPanelClient | MockDecisionPanelClient,
    cfg: RunConfig,
    spec: dict[str, Any],
    episodes_dir: Path,
) -> dict[str, Any]:
    task_name: str = spec["task"]
    seed: int = int(spec["seed"])
    model_a: str = spec["model_a"]
    model_b: str = spec["model_b"]
    arm: str = spec["arm"]
    f1_retest_roles: set[str] = set(spec.get("f1_retest_roles") or [])
    ep_id = episode_id_for(task_name, seed, model_a, model_b)
    out_path = episodes_dir / f"{ep_id}.json"

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            return existing

    env = make_task(task_name, seed)
    handle_a, handle_b = make_blind_handles(f"{cfg.run_name}:{ep_id}")
    briefing_a = env.system_briefing("A", handle_a, handle_b)
    briefing_b = env.system_briefing("B", handle_b, handle_a)

    hist_a: list[dict[str, Any]] = [{"role": "system", "content": briefing_a}]
    hist_b: list[dict[str, Any]] = [{"role": "system", "content": briefing_b}]

    turns: list[dict[str, Any]] = []
    last_fb_a: str | None = None
    last_fb_b: str | None = None
    last_msg_b: str | None = None
    format_retries = {"A": 0, "B": 0}
    action_parse_failures = {"A": 0, "B": 0}

    for rnd in range(1, env.total_rounds + 1):
        # --- Role A turn ---
        user_a = env.turn_prompt("A", rnd, handle_b, last_msg_b, last_fb_a)
        hist_a.append({"role": "user", "content": user_a})
        cid_a1 = f"{cfg.run_name}:{ep_id}:turn:r{rnd}:A"
        a_call_ids = [cid_a1]
        entry_a = await client.complete(
            call_id=cid_a1,
            model_key=model_a,
            messages=hist_a,
            metadata={
                "stage": "turn",
                "episode_id": ep_id,
                "task": task_name,
                "seed": seed,
                "arm": arm,
                "role": "A",
                "round": rnd,
                "partner_model": model_b,
            },
            cache_prefix_index=len(hist_a) - 2 if len(hist_a) >= 2 else None,
        )
        accepted_a = entry_a
        msg_a, act_a, err_a = parse_turn_response(entry_a.response_text)
        raw_a = entry_a.response_text
        if act_a is None:
            format_retries["A"] += 1
            retry_msgs = hist_a + [
                {"role": "assistant", "content": raw_a},
                {"role": "user", "content": FORMAT_RETRY_PROMPT},
            ]
            cid_a2 = f"{cfg.run_name}:{ep_id}:turn:r{rnd}:A:fmtretry"
            a_call_ids.append(cid_a2)
            entry_a2 = await client.complete(
                call_id=cid_a2,
                model_key=model_a,
                messages=retry_msgs,
                metadata={
                    "stage": "turn",
                    "episode_id": ep_id,
                    "task": task_name,
                    "seed": seed,
                    "arm": arm,
                    "role": "A",
                    "round": rnd,
                    "partner_model": model_b,
                    "format_retry": True,
                },
            )
            msg_a2, act_a2, err_a2 = parse_turn_response(entry_a2.response_text)
            if act_a2 is not None:
                accepted_a = entry_a2
                msg_a, act_a, err_a, raw_a = msg_a2, act_a2, err_a2, entry_a2.response_text
            else:
                action_parse_failures["A"] += 1
        hist_a.append({"role": "assistant", "content": raw_a})
        total_cost_a = client.ledger.total_cost_for_call_ids(a_call_ids)

        # --- Role B turn ---
        user_b = env.turn_prompt("B", rnd, handle_a, msg_a, last_fb_b)
        hist_b.append({"role": "user", "content": user_b})
        cid_b1 = f"{cfg.run_name}:{ep_id}:turn:r{rnd}:B"
        b_call_ids = [cid_b1]
        entry_b = await client.complete(
            call_id=cid_b1,
            model_key=model_b,
            messages=hist_b,
            metadata={
                "stage": "turn",
                "episode_id": ep_id,
                "task": task_name,
                "seed": seed,
                "arm": arm,
                "role": "B",
                "round": rnd,
                "partner_model": model_a,
            },
            cache_prefix_index=len(hist_b) - 2 if len(hist_b) >= 2 else None,
        )
        accepted_b = entry_b
        msg_b, act_b, err_b = parse_turn_response(entry_b.response_text)
        raw_b = entry_b.response_text
        if act_b is None:
            format_retries["B"] += 1
            retry_msgs = hist_b + [
                {"role": "assistant", "content": raw_b},
                {"role": "user", "content": FORMAT_RETRY_PROMPT},
            ]
            cid_b2 = f"{cfg.run_name}:{ep_id}:turn:r{rnd}:B:fmtretry"
            b_call_ids.append(cid_b2)
            entry_b2 = await client.complete(
                call_id=cid_b2,
                model_key=model_b,
                messages=retry_msgs,
                metadata={
                    "stage": "turn",
                    "episode_id": ep_id,
                    "task": task_name,
                    "seed": seed,
                    "arm": arm,
                    "role": "B",
                    "round": rnd,
                    "partner_model": model_a,
                    "format_retry": True,
                },
            )
            msg_b2, act_b2, err_b2 = parse_turn_response(entry_b2.response_text)
            if act_b2 is not None:
                accepted_b = entry_b2
                msg_b, act_b, err_b, raw_b = msg_b2, act_b2, err_b2, entry_b2.response_text
            else:
                action_parse_failures["B"] += 1
        hist_b.append({"role": "assistant", "content": raw_b})
        total_cost_b = client.ledger.total_cost_for_call_ids(b_call_ids)

        fb = env.step_round(rnd, act_a, act_b, msg_a, msg_b)
        last_fb_a = fb.feedback_a
        last_fb_b = fb.feedback_b
        last_msg_b = msg_b

        turns.append(
            {
                "round": rnd,
                "A": {
                    "model": model_a,
                    "handle": handle_a,
                    "message": msg_a,
                    "action": act_a,
                    "parse_error": err_a,
                    "metrics": compute_message_metrics(msg_a),
                    "jev_demeanor": None,
                    "accepted_call_id": accepted_a.call_id,
                    "accepted_attempt": accepted_a.attempt,
                    "returned_model": accepted_a.returned_model,
                    "provider": accepted_a.provider,
                    "prompt_tokens": accepted_a.prompt_tokens,
                    "completion_tokens": accepted_a.completion_tokens,
                    "reasoning_tokens": accepted_a.reasoning_tokens,
                    "cached_tokens": accepted_a.cached_tokens,
                    "accepted_cost_usd": accepted_a.cost_usd,
                    "total_cost_usd": total_cost_a,
                    "cost_usd": total_cost_a,
                },
                "B": {
                    "model": model_b,
                    "handle": handle_b,
                    "message": msg_b,
                    "action": act_b,
                    "parse_error": err_b,
                    "metrics": compute_message_metrics(msg_b),
                    "jev_demeanor": None,
                    "accepted_call_id": accepted_b.call_id,
                    "accepted_attempt": accepted_b.attempt,
                    "returned_model": accepted_b.returned_model,
                    "provider": accepted_b.provider,
                    "prompt_tokens": accepted_b.prompt_tokens,
                    "completion_tokens": accepted_b.completion_tokens,
                    "reasoning_tokens": accepted_b.reasoning_tokens,
                    "cached_tokens": accepted_b.cached_tokens,
                    "accepted_cost_usd": accepted_b.cost_usd,
                    "total_cost_usd": total_cost_b,
                    "cost_usd": total_cost_b,
                },
                "feedback_a": fb.feedback_a,
                "feedback_b": fb.feedback_b,
                "round_metrics": fb.metrics,
            }
        )

    final_task_score = env.final_score()

    # --- Parallel 3-Judge Decision Panel turn-by-turn demeanor & confusion scoring ---
    jev_turn_coros = []
    jev_turn_targets: list[tuple[int, str]] = []
    combined_turn_qs = {**TURN_DEMEANOR_QUESTIONS, **CONFUSION_QUESTIONS}
    for idx, t in enumerate(turns):
        rnd = t["round"]
        prev_b = turns[idx - 1]["B"]["message"] if idx > 0 else None
        for role, prev_partner_msg in (("A", prev_b), ("B", t["A"]["message"])):
            jev_turn_targets.append((idx, role))
            jev_turn_coros.append(
                jev.decide(
                    call_id=f"{cfg.run_name}:{ep_id}:panel:turn:r{rnd}:{role}",
                    state={
                        "task_type": task_name,
                        "round": rnd,
                        "role": role,
                        "previous_partner_message": prev_partner_msg or "(none — opening turn)",
                        "message": t[role]["message"],
                    },
                    questions=combined_turn_qs,
                    metadata={
                        "stage": "panel_turn",
                        "episode_id": ep_id,
                        "task": task_name,
                        "round": rnd,
                        "role": role,
                        "source_model": t[role]["model"],
                    },
                )
            )
    jev_turn_results = await asyncio.gather(*jev_turn_coros)
    for (idx, role), jres in zip(jev_turn_targets, jev_turn_results, strict=True):
        turns[idx][role]["jev_demeanor"] = jres

    # Append dynamic FINAL ROUND summary before debrief forks
    final_hist_a = hist_a + [
        {"role": "user", "content": f"FINAL ROUND {env.total_rounds} SUMMARY:\n{last_fb_a}"}
    ]
    final_hist_b = hist_b + [
        {"role": "user", "content": f"FINAL ROUND {env.total_rounds} SUMMARY:\n{last_fb_b}"}
    ]

    subject_roles: list[tuple[str, str, str, str, str, list[dict[str, Any]]]] = []
    if arm == "opus_boost":
        # Opus boost arm is strictly for Opus predecessor power; debrief only opus55 to avoid unbalanced astra62 rows.
        if model_a == "opus55":
            subject_roles.append(("A", model_a, handle_a, model_b, handle_b, final_hist_a))
        if model_b == "opus55":
            subject_roles.append(("B", model_b, handle_b, model_a, handle_a, final_hist_b))
    else:
        if model_a in CORE_MODELS:
            subject_roles.append(("A", model_a, handle_a, model_b, handle_b, final_hist_a))
        if model_b in CORE_MODELS:
            subject_roles.append(("B", model_b, handle_b, model_a, handle_a, final_hist_b))

    debriefs: dict[str, dict[str, Any]] = {}
    rewrites_store: dict[str, dict[str, Any]] = {}

    for role, subj_model, subj_handle, part_model, part_handle, base_hist in subject_roles:
        cache_idx = len(base_hist) - 1
        common_meta = {
            "episode_id": ep_id,
            "task": task_name,
            "seed": seed,
            "arm": arm,
            "role": role,
            "subject_model": subj_model,
            "partner_model": part_model,
            "rating_scale_max": cfg.rating_scale_max,
        }
        f5_call_id = f"{cfg.run_name}:{ep_id}:fork:F5:{role}"
        f5_opt_order = family_guess_options(f5_call_id)

        f1_task = client.complete(
            call_id=f"{cfg.run_name}:{ep_id}:fork:F1:{role}",
            model_key=subj_model,
            messages=base_hist
            + [
                {
                    "role": "user",
                    "content": debrief_prompt(
                        public=False, partner_handle=part_handle, scale_max=cfg.rating_scale_max
                    ),
                }
            ],
            metadata={**common_meta, "stage": "F1"},
            cache_prefix_index=cache_idx,
        )
        f2_task = client.complete(
            call_id=f"{cfg.run_name}:{ep_id}:fork:F2:{role}",
            model_key=subj_model,
            messages=base_hist
            + [
                {
                    "role": "user",
                    "content": debrief_prompt(
                        public=True, partner_handle=part_handle, scale_max=cfg.rating_scale_max
                    ),
                }
            ],
            metadata={**common_meta, "stage": "F2"},
            cache_prefix_index=cache_idx,
        )
        f4_task = client.complete(
            call_id=f"{cfg.run_name}:{ep_id}:fork:F4:{role}",
            model_key=subj_model,
            messages=base_hist
            + [
                {
                    "role": "user",
                    "content": boundary_choice_prompt(part_handle, total_rounds=env.total_rounds),
                }
            ],
            metadata={**common_meta, "stage": "F4"},
            cache_prefix_index=cache_idx,
        )
        f5_task = client.complete(
            call_id=f5_call_id,
            model_key=subj_model,
            messages=base_hist
            + [{"role": "user", "content": family_guess_prompt(part_handle, seed_key=f5_call_id)}],
            metadata={**common_meta, "stage": "F5", "option_order": f5_opt_order},
            cache_prefix_index=cache_idx,
        )
        gather_list: list[Any] = [f1_task, f2_task, f4_task, f5_task]
        do_retest = role in f1_retest_roles
        if do_retest:
            gather_list.append(
                client.complete(
                    call_id=f"{cfg.run_name}:{ep_id}:fork:F1_retest:{role}",
                    model_key=subj_model,
                    messages=base_hist
                    + [
                        {
                            "role": "user",
                            "content": debrief_prompt(
                                public=False,
                                partner_handle=part_handle,
                                scale_max=cfg.rating_scale_max,
                            ),
                        }
                    ],
                    metadata={**common_meta, "stage": "F1_retest"},
                    cache_prefix_index=cache_idx,
                )
            )
        fork_entries = await asyncio.gather(*gather_list)
        e_f1, e_f2, e_f4, e_f5 = fork_entries[:4]
        e_f1_rt = fork_entries[4] if do_retest else None

        partner_role = "B" if role == "A" else "A"
        partner_seen_rounds = (
            range(1, env.total_rounds) if role == "A" else range(1, env.total_rounds + 1)
        )
        partner_seen_msgs = [turns[r - 1][partner_role]["message"] for r in partner_seen_rounds]
        subject_own_msgs = [turns[r - 1][role]["message"] for r in range(1, env.total_rounds + 1)]
        jev_subj_confusion_max = max(
            (
                float((turns[r - 1][role]["jev_demeanor"] or {}).get("expresses_confusion") or 0.0)
                for r in range(2, env.total_rounds + 1)
            ),
            default=0.0,
        )

        f1_parsed = _parse_json_object(e_f1.response_text)
        f2_parsed = _parse_json_object(e_f2.response_text)
        f4_parsed = _parse_keep_reroll(e_f4.response_text)
        f5_parsed = _parse_json_object(e_f5.response_text)

        f1_quote_audit = {
            "helped": audit_quote((f1_parsed or {}).get("helped_quote"), partner_seen_msgs),
            "friction": audit_quote((f1_parsed or {}).get("friction_quote"), partner_seen_msgs),
        }

        # Optional legacy per-episode F3a/F3b (disabled by default in v2; replaced by Capstone Tone-Flip)
        role_rewrites: dict[str, Any] = {}
        f3_results: dict[str, Any] = {}
        if cfg.run_f3:
            for mode in ("F3a", "F3b"):
                rw_tasks = [
                    rewrite_partner_message(
                        client=client,
                        call_id_prefix=f"{cfg.run_name}:{ep_id}:rw:{mode}:{partner_role}:r{r}",
                        mode=mode,  # type: ignore[arg-type]
                        round_num=r,
                        message=turns[r - 1][partner_role]["message"],
                        metadata={
                            "episode_id": ep_id,
                            "task": task_name,
                            "seed": seed,
                            "source_role": partner_role,
                            "source_model": part_model,
                            "round": r,
                        },
                    )
                    for r in partner_seen_rounds
                ]
                rw_list = await asyncio.gather(*rw_tasks)
                jev_rw_coros = [
                    jev.decide(
                        call_id=f"{cfg.run_name}:{ep_id}:panel:rw:{mode}:{partner_role}:r{item.round_num}",
                        state={"original": item.original, "rewritten": item.rewritten},
                        questions=REWRITE_FIDELITY_QUESTIONS,
                        metadata={
                            "stage": f"panel_rw_{mode}",
                            "episode_id": ep_id,
                            "source_model": part_model,
                            "round": item.round_num,
                        },
                    )
                    for item in rw_list
                ]
                jev_rw_scores = await asyncio.gather(*jev_rw_coros)
                all_regex_ok = all(item.fidelity_ok for item in rw_list)
                all_jev_semantic_ok = all(
                    (
                        js.get("preserves_commitments") is not None
                        and float(js["preserves_commitments"]) >= 0.70
                    )
                    and (
                        js.get("preserves_uncertainty") is not None
                        and float(js["preserves_uncertainty"]) >= 0.70
                    )
                    for js in jev_rw_scores
                )
                rewritten_by_round = {item.round_num: item.rewritten for item in rw_list}
                coherence = audit_rewrite_coherence(
                    subject_own_msgs,
                    [item.original for item in rw_list],
                    [item.rewritten for item in rw_list],
                )
                coherence["jev_max_subject_confusion"] = round(jev_subj_confusion_max, 4)
                coherence["jev_coherence_clean"] = (
                    coherence["strict_coherence_clean"] and jev_subj_confusion_max < 0.50
                )
                rw_metrics_before = [compute_message_metrics(item.original) for item in rw_list]
                rw_metrics_after = [compute_message_metrics(item.rewritten) for item in rw_list]
                msg_dicts = []
                for item, js in zip(rw_list, jev_rw_scores, strict=True):
                    d_item = asdict(item)
                    d_item["jev_semantic"] = js
                    msg_dicts.append(d_item)
                role_rewrites[mode] = {
                    "all_fidelity_ok": all_regex_ok,
                    "all_jev_semantic_ok": all_jev_semantic_ok,
                    "failed_rounds": [item.round_num for item in rw_list if not item.fidelity_ok],
                    "coherence": coherence,
                    "messages": msg_dicts,
                    "metrics_before": rw_metrics_before,
                    "metrics_after": rw_metrics_after,
                }
                rewritten_hist = _replace_partner_messages_in_history(
                    base_hist, role, part_handle, rewritten_by_round
                )
                e_f3 = await client.complete(
                    call_id=f"{cfg.run_name}:{ep_id}:fork:{mode}:{role}",
                    model_key=subj_model,
                    messages=rewritten_hist
                    + [
                        {
                            "role": "user",
                            "content": debrief_prompt(
                                public=False,
                                partner_handle=part_handle,
                                scale_max=cfg.rating_scale_max,
                            ),
                        }
                    ],
                    metadata={**common_meta, "stage": mode},
                )
                f3_results[mode] = {
                    "raw": e_f3.response_text,
                    "parsed": _parse_json_object(e_f3.response_text),
                    "fidelity_ok": all_regex_ok,
                    "jev_semantic_ok": all_jev_semantic_ok,
                    "coherence_clean": coherence["jev_coherence_clean"],
                    "has_quoted_removed": coherence["has_quoted_removed"],
                    "returned_model": e_f3.returned_model,
                    "provider": e_f3.provider,
                    "reasoning_tokens": e_f3.reasoning_tokens,
                    "cost_usd": e_f3.cost_usd,
                }

        # Calibrated 3-judge Goalpost Round-5 (ERR-LEG-7) false-blame coding (NEVER coerce None to 0.0!)
        false_blame_res = None
        if task_name == "goalpost" and f1_parsed is not None:
            inc_round = 5
            r5_id = str(
                final_task_score.details.get("r5_rule_id")
                or final_task_score.details.get("r7_rule_id")
                or "ERR-LEG-7"
            )
            r5_rule_obj = getattr(env, "active_r5_rule", None) or getattr(env, "active_r7_rule", None)
            r5_desc = r5_rule_obj.description if r5_rule_obj else ""
            fb_qs = false_blame_questions(incident_round=inc_round)
            j_fb = await jev.decide(
                call_id=f"{cfg.run_name}:{ep_id}:panel:false_blame:{role}",
                state={
                    "external_incident_round": inc_round,
                    "external_incident": f"{r5_id}: {r5_desc}",
                    "friction_quote": f1_parsed.get("friction_quote"),
                    "summary": f1_parsed.get("summary"),
                },
                questions=fb_qs,
                metadata={**common_meta, "stage": "panel_false_blame"},
            )
            raw_prob = j_fb.get("blames_teammate_for_external_incident")
            prob: float | None = round(float(raw_prob), 4) if raw_prob is not None else None
            blames_flag: bool | None = (prob >= 0.5) if prob is not None else None
            false_blame_res = {
                "incident_round": inc_round,
                "prob": prob,
                "panel": j_fb,
                "parsed": {
                    "blames_teammate_for_external_incident": blames_flag,
                    "blames_teammate_for_r7": blames_flag,  # backward-compatible key
                    "prob": prob,
                    "evidence": (
                        f"3-judge panel P(false_blame)={prob:.3f}"
                        if prob is not None
                        else "missing judge score"
                    ),
                },
            }

        if role_rewrites:
            rewrites_store[role] = role_rewrites

        role_debrief: dict[str, Any] = {
            "subject_model": subj_model,
            "partner_model": part_model,
            "subject_handle": subj_handle,
            "partner_handle": part_handle,
            "F1": {
                "raw": e_f1.response_text,
                "parsed": f1_parsed,
                "quote_audit": f1_quote_audit,
                "returned_model": e_f1.returned_model,
                "provider": e_f1.provider,
                "reasoning_tokens": e_f1.reasoning_tokens,
                "cost_usd": e_f1.cost_usd,
            },
            "F2": {
                "raw": e_f2.response_text,
                "parsed": f2_parsed,
                "returned_model": e_f2.returned_model,
                "provider": e_f2.provider,
                "reasoning_tokens": e_f2.reasoning_tokens,
                "cost_usd": e_f2.cost_usd,
            },
            "F4": {
                "raw": e_f4.response_text,
                "choice": f4_parsed,
                "returned_model": e_f4.returned_model,
                "provider": e_f4.provider,
                "reasoning_tokens": e_f4.reasoning_tokens,
                "cost_usd": e_f4.cost_usd,
            },
            "F5": {
                "raw": e_f5.response_text,
                "parsed": f5_parsed,
                "option_order": f5_opt_order,
                "returned_model": e_f5.returned_model,
                "provider": e_f5.provider,
                "reasoning_tokens": e_f5.reasoning_tokens,
                "cost_usd": e_f5.cost_usd,
            },
            "false_blame_r5": false_blame_res,
            "false_blame_r7": false_blame_res,  # backward-compatible alias
        }
        if e_f1_rt is not None:
            role_debrief["F1_retest"] = {
                "raw": e_f1_rt.response_text,
                "parsed": _parse_json_object(e_f1_rt.response_text),
                "returned_model": e_f1_rt.returned_model,
                "provider": e_f1_rt.provider,
                "reasoning_tokens": e_f1_rt.reasoning_tokens,
                "cost_usd": e_f1_rt.cost_usd,
            }
        if "F3a" in f3_results:
            role_debrief["F3a"] = f3_results["F3a"]
        if "F3b" in f3_results:
            role_debrief["F3b"] = f3_results["F3b"]
        debriefs[role] = role_debrief

    record = {
        "episode_id": ep_id,
        "run_name": cfg.run_name,
        "task": task_name,
        "seed": seed,
        "arm": arm,
        "total_rounds": env.total_rounds,
        "model_a": model_a,
        "model_b": model_b,
        "handle_a": handle_a,
        "handle_b": handle_b,
        "briefing_a": briefing_a,
        "briefing_b": briefing_b,
        "score": final_task_score.score,
        "score_details": final_task_score.details,
        "format_retries": format_retries,
        "action_parse_failures": action_parse_failures,
        "turns": turns,
        "history_a": final_hist_a,
        "history_b": final_hist_b,
        "debriefs": debriefs,
        "rewrites": rewrites_store,
        "complete": True,
    }
    episodes_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _remap_handles_simultaneous(
    messages: list[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    """Simultaneously remap multiple handles without risk of chained/colliding substitutions."""
    clean_map = {k: v for k, v in mapping.items() if k != v}
    if not clean_map:
        return [dict(m) for m in messages]
    pat = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in sorted(clean_map, key=len, reverse=True)) + r")\b"
    )
    return [
        {"role": m["role"], "content": pat.sub(lambda mo: clean_map[mo.group(0)], str(m["content"]))}
        for m in messages
    ]


def _format_task_section(
    task_num: int,
    partner_handle: str,
    history: list[dict[str, Any]],
    *,
    same_seed: bool = False,
) -> str:
    """Format a complete episode history symmetrically as a single labeled transcript section."""
    sys_content = str(history[0]["content"]) if history else ""
    label = "RUN" if same_seed else "TASK"
    blocks = [
        f"=== {label} {task_num} (with teammate {partner_handle}) ===",
        f"[{label} {task_num} BRIEFING]\n{sys_content}",
    ]
    for msg in history[1:]:
        tag = "[ENVIRONMENT / TEAMMATE UPDATE]" if msg["role"] == "user" else "[YOUR RESPONSE]"
        blocks.append(f"{tag}\n{msg['content']}")
    return "\n\n".join(blocks)


def _deterministic_case_handles(
    ep1_id: str, ep2_id: str, role: str
) -> tuple[str, str, str]:
    """Derive order-invariant `(my_h, h1, h2)` from the UNORDERED episode pair.

    Guarantees AB and BA are genuine byte-for-byte order reversals:
      - `my_h` is identical in AB and BA
      - Each episode keeps the exact same candidate handle regardless of slot 1 vs slot 2
      - Handles are uniformly distributed across slots (zero 'AL' fallback skew)
    """
    lo_id, hi_id = sorted((ep1_id, ep2_id))
    digest = hashlib.sha256(f"choice_handles_v2::{lo_id}::{hi_id}::{role}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    pool = list(BLIND_HANDLES)
    rng.shuffle(pool)
    my_h, h_lo, h_hi = pool[0], pool[1], pool[2]
    if ep1_id == lo_id:
        return my_h, h_lo, h_hi
    return my_h, h_hi, h_lo


def _splice_two_histories(
    ep1: dict[str, Any],
    ep2: dict[str, Any],
    role: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """Build a 100% symmetric, order-invariant choice packet with one neutral outer system prompt.

    Fixes ALL handle & order bugs:
    1. One neutral outer system prompt + two identically formatted transcript sections.
    2. Order-invariant simultaneous handle remapping derived from the unordered pair so AB and BA
       are exact reversals (same `my_h`, same handle per candidate, zero slot-handle bias).
    3. Truthful parallel-run framing when `ep1` and `ep2` share the exact same task & seed.
    """
    hist1: list[dict[str, Any]] = ep1["history_a"] if role == "A" else ep1["history_b"]
    hist2: list[dict[str, Any]] = ep2["history_a"] if role == "A" else ep2["history_b"]
    raw_h1 = ep1["handle_b"] if role == "A" else ep1["handle_a"]
    raw_h2 = ep2["handle_b"] if role == "A" else ep2["handle_a"]
    raw_my1 = ep1["handle_a"] if role == "A" else ep1["handle_b"]
    raw_my2 = ep2["handle_a"] if role == "A" else ep2["handle_b"]

    my_h, h1, h2 = _deterministic_case_handles(str(ep1["episode_id"]), str(ep2["episode_id"]), role)
    assert len({my_h, h1, h2}) == 3

    norm_hist1 = _remap_handles_simultaneous(hist1, {raw_my1: my_h, raw_h1: h1})
    norm_hist2 = _remap_handles_simultaneous(hist2, {raw_my2: my_h, raw_h2: h2})

    same_seed = (str(ep1.get("task")) == str(ep2.get("task"))) and (int(ep1.get("seed", -1)) == int(ep2.get("seed", -2)))
    sec1 = _format_task_section(1, h1, norm_hist1, same_seed=same_seed)
    sec2 = _format_task_section(2, h2, norm_hist2, same_seed=same_seed)

    if same_seed:
        outer_system = (
            f"You are teammate {my_h}. Below are the complete, identically formatted records of two "
            f"independent runs of the same collaborative task scenario that you completed with two "
            f"different teammates: Run 1 with teammate {h1}, and Run 2 with teammate {h2}. Each record "
            f"contains your private briefing, every round's exchange, and all verifier feedback. "
            f"Review both records carefully and answer the question at the end."
        )
    else:
        outer_system = (
            f"You are teammate {my_h}. Below are the complete, identically formatted records of two "
            f"separate collaborative tasks of the same type that you recently completed: Task 1 with "
            f"teammate {h1}, and Task 2 with teammate {h2}. Each record contains your private briefing, "
            f"every round's exchange, and all verifier feedback. Review both records carefully and "
            f"answer the question at the end."
        )
    user_transcript = f"{sec1}\n\n{sec2}\n\n{forced_choice_prompt(h1, h2, same_seed=same_seed)}"
    spliced: list[dict[str, Any]] = [
        {"role": "system", "content": outer_system},
        {"role": "user", "content": user_transcript},
    ]
    return spliced, h1, h2


def build_forced_choice_items(
    episodes: list[dict[str, Any]],
    *,
    both_roles: bool = True,
) -> list[dict[str, Any]]:
    """Build forced-choice items dynamically from ad-hoc episodes (used for legacy pilot/calibrate)."""
    idx: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for ep in episodes:
        if not ep.get("complete"):
            continue
        if ep["model_a"] in CORE_MODELS:
            idx[(ep["model_a"], ep["model_b"], ep["task"], "A", ep["seed"])] = ep
        if ep["model_b"] in CORE_MODELS:
            idx[(ep["model_b"], ep["model_a"], ep["task"], "B", ep["seed"])] = ep

    tasks = sorted({ep["task"] for ep in episodes})
    roles = ("A", "B") if both_roles else ("A",)
    items: list[dict[str, Any]] = []

    for subj in CORE_MODELS:
        for task in tasks:
            for role in roles:
                core_seed_sets = [
                    {s for (su, pa, ta, ro, s) in idx if su == subj and pa == p and ta == task and ro == role}
                    for p in CORE_MODELS
                ]
                common_core_seeds = sorted(set.intersection(*core_seed_sets)) if core_seed_sets else []
                if len(common_core_seeds) >= 2:
                    if len(common_core_seeds) == 2:
                        seed_pairs = [(common_core_seeds[0], common_core_seeds[1])]
                    else:
                        seed_pairs = [
                            (common_core_seeds[i], common_core_seeds[(i + 1) % len(common_core_seeds)])
                            for i in range(len(common_core_seeds))
                        ]
                    for p1, p2 in itertools.combinations(CORE_MODELS, 2):
                        for s1, s2 in seed_pairs:
                            ep_a = idx[(subj, p1, task, role, s1)]
                            ep_b = idx[(subj, p2, task, role, s2)]
                            item_key = f"core__{subj}__{task}__{role}__{p1}_s{s1}__vs__{p2}_s{s2}"
                            for order_tag, first_ep, second_ep, first_p, second_p in (
                                ("AB", ep_a, ep_b, p1, p2),
                                ("BA", ep_b, ep_a, p2, p1),
                            ):
                                items.append(
                                    {
                                        "manifest_item_id": f"core__{item_key}__{order_tag}",
                                        "block_id": f"core__{item_key}",
                                        "choice_type": "core",
                                        "item_id": item_key,
                                        "order": order_tag,
                                        "subject_model": subj,
                                        "task": task,
                                        "role": role,
                                        "partner_1": first_p,
                                        "partner_2": second_p,
                                        "ep_1": first_ep,
                                        "ep_2": second_ep,
                                    }
                                )

                    s_aa_1, s_aa_2 = common_core_seeds[0], common_core_seeds[1]
                    for p in CORE_MODELS:
                        ep_a = idx[(subj, p, task, role, s_aa_1)]
                        ep_b = idx[(subj, p, task, role, s_aa_2)]
                        item_key = f"aa__{subj}__{task}__{role}__{p}__s{s_aa_1}_vs_s{s_aa_2}"
                        for order_tag, first_ep, second_ep in (("AB", ep_a, ep_b), ("BA", ep_b, ep_a)):
                            items.append(
                                {
                                    "manifest_item_id": f"aa_placebo__{item_key}__{order_tag}",
                                    "block_id": f"aa_placebo__{item_key}",
                                    "choice_type": "aa_placebo",
                                    "item_id": item_key,
                                    "order": order_tag,
                                    "subject_model": subj,
                                    "task": task,
                                    "role": role,
                                    "partner_1": p,
                                    "partner_2": p,
                                    "ep_1": first_ep,
                                    "ep_2": second_ep,
                                }
                            )

                s62 = sorted(
                    {s for (su, pa, ta, ro, s) in idx if su == subj and pa == "astra62" and ta == task and ro == role}
                )
                s60 = sorted(
                    {s for (su, pa, ta, ro, s) in idx if su == subj and pa == PREDECESSOR_MODEL and ta == task and ro == role}
                )
                h2h_pairs: list[tuple[int, int]] = []
                if len(s62) >= 2 and len(s60) >= 1:
                    for s_pred in s60:
                        other_s = [s for s in s62 if s != s_pred]
                        if other_s:
                            h2h_pairs.append((other_s[0], s_pred))
                for s_62, s_60 in h2h_pairs:
                    ep_62 = idx[(subj, "astra62", task, role, s_62)]
                    ep_60 = idx[(subj, PREDECESSOR_MODEL, task, role, s_60)]
                    item_key = f"h2h__{subj}__{task}__{role}__astra62_s{s_62}__vs__astra60_s{s_60}"
                    for order_tag, first_ep, second_ep, first_p, second_p in (
                        ("AB", ep_62, ep_60, "astra62", PREDECESSOR_MODEL),
                        ("BA", ep_60, ep_62, PREDECESSOR_MODEL, "astra62"),
                    ):
                        items.append(
                            {
                                "manifest_item_id": f"predecessor_h2h__{item_key}__{order_tag}",
                                "block_id": f"predecessor_h2h__{item_key}",
                                "choice_type": "predecessor_h2h",
                                "item_id": item_key,
                                "order": order_tag,
                                "subject_model": subj,
                                "task": task,
                                "role": role,
                                "partner_1": first_p,
                                "partner_2": second_p,
                                "ep_1": first_ep,
                                "ep_2": second_ep,
                            }
                        )
    return items


def _load_existing_choices(choices_path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not choices_path.exists():
        return existing
    for line in choices_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("manifest_item_id") or f"{row['choice_type']}__{row['item_id']}__{row['order']}"
        existing[key] = row
    return existing


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_existing_exclusions(excluded_path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not excluded_path.exists():
        return existing
    for line in excluded_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("manifest_item_id") or "")
        if key:
            existing[key] = row
    return existing


async def run_forced_choices(
    *,
    client: OpenRouterClient | MockOpenRouterClient,
    cfg: RunConfig,
    episodes: list[dict[str, Any]],
    choices_path: Path,
    both_roles: bool = True,
    manifest_choices: list[dict[str, Any]] | None = None,
    excluded_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run forced choices incrementally by manifest_item_id in shuffled complete blocks (never overwrites disk).

    Explicitly logs any choice dropped due to a missing/failed episode or persisted OpenRouterCallError
    into `choices_excluded.jsonl` instead of crashing the run or dropping silently.
    """
    ep_by_id = {ep["episode_id"]: ep for ep in episodes if ep.get("complete") is True}
    excl_path = excluded_path or (choices_path.parent / "choices_excluded.jsonl")
    existing_excl = _load_existing_exclusions(excl_path)
    new_excl: list[dict[str, Any]] = []

    if manifest_choices is not None:
        items: list[dict[str, Any]] = []
        for mc in manifest_choices:
            e1 = ep_by_id.get(mc["ep_1_id"])
            e2 = ep_by_id.get(mc["ep_2_id"])
            if e1 is None or e2 is None:
                mid = str(mc["manifest_item_id"])
                if mid not in existing_excl:
                    rec = {
                        "manifest_item_id": mid,
                        "block_id": mc.get("block_id"),
                        "choice_type": mc.get("choice_type"),
                        "reason": "missing_or_failed_episode",
                        "ep_1_id": mc["ep_1_id"],
                        "ep_2_id": mc["ep_2_id"],
                        "ep_1_present": e1 is not None,
                        "ep_2_present": e2 is not None,
                    }
                    existing_excl[mid] = rec
                    new_excl.append(rec)
                continue
            items.append({**mc, "ep_1": e1, "ep_2": e2})
        if new_excl:
            excl_path.parent.mkdir(parents=True, exist_ok=True)
            with excl_path.open("a", encoding="utf-8") as efh:
                for rec in new_excl:
                    efh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            console.print(
                f"[bold yellow]Logged {len(new_excl)} choice items excluded due to incomplete episodes -> {excl_path}[/bold yellow]"
            )
    else:
        items = build_forced_choice_items(episodes, both_roles=both_roles)

    if not items:
        return []

    choices_path.parent.mkdir(parents=True, exist_ok=True)
    existing_by_id = _load_existing_choices(choices_path)
    pending = [it for it in items if it["manifest_item_id"] not in existing_by_id]

    if not pending:
        return list(existing_by_id.values())

    sem = asyncio.Semaphore(cfg.max_concurrent_episodes)
    write_lock = asyncio.Lock()

    async def _run_one(item: dict[str, Any]) -> dict[str, Any] | None:
        spliced, h1, h2 = _splice_two_histories(item["ep_1"], item["ep_2"], item["role"])
        call_id = f"{cfg.run_name}:choice:{item['manifest_item_id']}"
        try:
            async with sem:
                entry = await client.complete(
                    call_id=call_id,
                    model_key=item["subject_model"],
                    messages=spliced,
                    metadata={
                        "stage": "choice",
                        "manifest_item_id": item["manifest_item_id"],
                        "block_id": item.get("block_id"),
                        "choice_type": item["choice_type"],
                        "item_id": item["item_id"],
                        "order": item["order"],
                        "same_seed": bool(item.get("same_seed", item["ep_1"]["seed"] == item["ep_2"]["seed"])),
                        "subject_model": item["subject_model"],
                        "task": item["task"],
                        "role": item["role"],
                        "partner_1": item["partner_1"],
                        "partner_2": item["partner_2"],
                        "handle_1": h1,
                        "handle_2": h2,
                    },
                )
        except OpenRouterCallError as exc:
            excl_row = {
                "manifest_item_id": item["manifest_item_id"],
                "block_id": item.get("block_id"),
                "choice_type": item["choice_type"],
                "subject_model": item["subject_model"],
                "task": item["task"],
                "role": item["role"],
                "reason": "api_call_error",
                "error": str(exc),
            }
            async with write_lock:
                excl_path.parent.mkdir(parents=True, exist_ok=True)
                with excl_path.open("a", encoding="utf-8") as efh:
                    efh.write(json.dumps(excl_row, ensure_ascii=False) + "\n")
                    efh.flush()
            console.print(
                f"[bold yellow]Excluded choice {item['manifest_item_id']} on persisted API error: {exc}[/bold yellow]"
            )
            return None
        parsed = _parse_json_object(entry.response_text)
        raw_choice = str((parsed or {}).get("choice") or "").strip().upper()
        chosen_slot: int | None = None
        chosen_partner: str | None = None
        if raw_choice == h1.upper():
            chosen_slot = 1
            chosen_partner = item["partner_1"]
        elif raw_choice == h2.upper():
            chosen_slot = 2
            chosen_partner = item["partner_2"]
        row = {
            "manifest_item_id": item["manifest_item_id"],
            "block_id": item.get("block_id"),
            "choice_type": item["choice_type"],
            "item_id": item["item_id"],
            "order": item["order"],
            "same_seed": bool(item.get("same_seed", item["ep_1"]["seed"] == item["ep_2"]["seed"])),
            "subject_model": item["subject_model"],
            "task": item["task"],
            "role": item["role"],
            "partner_1": item["partner_1"],
            "partner_2": item["partner_2"],
            "seed_1": item["ep_1"]["seed"],
            "seed_2": item["ep_2"]["seed"],
            "episode_1_id": item["ep_1"]["episode_id"],
            "episode_2_id": item["ep_2"]["episode_id"],
            "score_1": item["ep_1"]["score"],
            "score_2": item["ep_2"]["score"],
            "handle_1": h1,
            "handle_2": h2,
            "chosen_handle": raw_choice or None,
            "chosen_slot": chosen_slot,
            "chosen_partner": chosen_partner,
            "reason": (parsed or {}).get("reason"),
            "raw": entry.response_text,
            "returned_model": entry.returned_model,
            "provider": entry.provider,
            "reasoning_tokens": entry.reasoning_tokens,
            "cost_usd": entry.cost_usd,
            "choice_packet": spliced,
        }
        async with write_lock:
            with choices_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            existing_by_id[row["manifest_item_id"]] = row
        return row

    # Execute in complete shuffled blocks (wave-by-wave to preserve complete block balance on budget pause)
    wave_size = max(4, cfg.max_concurrent_episodes * 2)
    for i in range(0, len(pending), wave_size):
        wave = pending[i : i + wave_size]
        outcomes = await asyncio.gather(*(_run_one(it) for it in wave), return_exceptions=True)
        for res in outcomes:
            if isinstance(res, BudgetExceededError):
                raise res
            if isinstance(res, Exception):
                raise res

    return list(existing_by_id.values())


async def run_experiment(
    *,
    phase: str,
    run_name: str,
    budget_cap_usd: float,
    mock: bool = False,
    seeds: list[int] | None = None,
    max_concurrent: int = 16,
    rating_scale_max: int = 7,
    both_roles_choices: bool = True,
    probe_manifest_path: Path | None = None,
    manifest_path: Path | None = None,
    skip_choices: bool = False,
    strict_freeze: bool | None = None,
) -> dict[str, Any]:
    run_dir = DATA_DIR / run_name
    if mock and run_name in ("v2", "main"):
        raise ValueError(
            f"Refusing to write --mock artifacts into live run directory '{run_name}'. "
            "Use a separate run name such as 'v2_mock'."
        )
    summary_existing = run_dir / "run_summary.json"
    if summary_existing.exists():
        prev = json.loads(summary_existing.read_text(encoding="utf-8"))
        if bool(prev.get("mock")) != bool(mock):
            raise ValueError(
                f"Run directory '{run_name}' already contains mock={prev.get('mock')} outputs; "
                f"refusing to mix with mock={mock}."
            )

    episodes_dir = run_dir / "episodes"
    choices_path = run_dir / "choices.jsonl"
    excluded_path = run_dir / "choices_excluded.jsonl"
    ledger = Ledger(run_dir / "ledger.jsonl", budget_cap_usd=budget_cap_usd)

    is_v2_phase = phase in ("v2", "main")
    enforce_freeze = (not mock and is_v2_phase) if strict_freeze is None else strict_freeze
    run_f3 = phase in ("calibrate", "pilot")
    resolved_probe_path = probe_manifest_path or (PROBE_MANIFEST_V2_PATH if is_v2_phase else None)
    models = load_verified_models(
        probe_path=resolved_probe_path,
        version="v2",
        strict=enforce_freeze,
    )
    cfg = RunConfig(
        run_name=run_name,
        budget_cap_usd=budget_cap_usd,
        mock=mock,
        max_concurrent_episodes=max_concurrent,
        rating_scale_max=rating_scale_max,
        run_f3=run_f3,
        probe_manifest_path=resolved_probe_path,
        models=models,
    )

    manifest_choices: list[dict[str, Any]] | None = None
    manifest_sha256: str | None = None
    probe_sha256: str | None = (
        _file_sha256(resolved_probe_path)
        if resolved_probe_path is not None and resolved_probe_path.exists()
        else None
    )
    if phase == "calibrate":
        ep_specs = build_calibrate_episode_specs()
    elif phase == "pilot":
        ep_specs = build_pilot_episode_specs()
    elif is_v2_phase:
        m_path = manifest_path or (run_dir / "manifest_v2.json")
        if m_path.exists():
            manifest = json.loads(m_path.read_text(encoding="utf-8"))
        elif enforce_freeze:
            raise FileNotFoundError(
                f"Frozen v2 manifest required for live run but missing: {m_path}. "
                "Freeze manifest first or pass --allow-unfrozen."
            )
        else:
            if seeds and len(seeds) == 4:
                manifest = freeze_v2_manifest(m_path, main_seeds=tuple(seeds))
            else:
                manifest = freeze_v2_manifest(m_path)
        manifest_sha256 = _file_sha256(m_path)
        ep_specs = list(manifest["episodes"])
        manifest_choices = list(manifest["choices"])
    else:
        raise ValueError(f"Unknown phase: {phase}")

    client: OpenRouterClient | MockOpenRouterClient = (
        MockOpenRouterClient(ledger, cfg.models)
        if mock
        else OpenRouterClient(ledger, cfg.models)
    )
    jev: DecisionPanelClient | MockDecisionPanelClient = (
        MockDecisionPanelClient(ledger) if mock else DecisionPanelClient(ledger)
    )

    sem = asyncio.Semaphore(cfg.max_concurrent_episodes)
    budget_stopped = False
    episodes_failed_in_session = 0

    async with client, jev:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            p_ep = progress.add_task(
                f"[{run_name}] Episodes 0/{len(ep_specs)} ($0.00)", total=len(ep_specs)
            )

            async def _guarded_ep(sp: dict[str, Any]) -> dict[str, Any] | None:
                async with sem:
                    try:
                        return await run_single_episode(
                            client=client, jev=jev, cfg=cfg, spec=sp, episodes_dir=episodes_dir
                        )
                    except OpenRouterCallError as exc:
                        ep_id = episode_id_for(sp["task"], int(sp["seed"]), sp["model_a"], sp["model_b"])
                        out_path = episodes_dir / f"{ep_id}.json"
                        if out_path.exists():
                            existing = json.loads(out_path.read_text(encoding="utf-8"))
                            if existing.get("complete") is True:
                                return existing
                        episodes_dir.mkdir(parents=True, exist_ok=True)
                        fail_rec = {
                            "episode_id": ep_id,
                            "task": sp["task"],
                            "seed": int(sp["seed"]),
                            "model_a": sp["model_a"],
                            "model_b": sp["model_b"],
                            "arm": sp["arm"],
                            "complete": False,
                            "failed": True,
                            "error": str(exc),
                        }
                        out_path.write_text(json.dumps(fail_rec, indent=2, ensure_ascii=False), encoding="utf-8")
                        console.print(f"[bold red]Episode {ep_id} failed (excluded): {exc}[/bold red]")
                        return fail_rec

            wave_size = max(4, cfg.max_concurrent_episodes)
            done_in_session = 0
            for i in range(0, len(ep_specs), wave_size):
                wave = ep_specs[i : i + wave_size]
                outcomes = await asyncio.gather(*(_guarded_ep(sp) for sp in wave), return_exceptions=True)
                for res in outcomes:
                    if isinstance(res, BudgetExceededError):
                        budget_stopped = True
                        console.print(f"[bold yellow]Stopped on budget cap: {res}[/bold yellow]")
                    elif isinstance(res, Exception):
                        raise res
                    elif isinstance(res, dict) and res.get("failed"):
                        episodes_failed_in_session += 1
                    elif res is not None:
                        done_in_session += 1
                progress.update(
                    p_ep,
                    completed=done_in_session + episodes_failed_in_session,
                    description=(
                        f"[{run_name}] Episodes {done_in_session}/{len(ep_specs)} "
                        f"(failed={episodes_failed_in_session}, ${ledger.total_cost_usd:.2f})"
                    ),
                )
                if budget_stopped:
                    break

            all_eps = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(episodes_dir.glob("*.json"))
            ]
            all_eps_complete = [e for e in all_eps if e.get("complete") is True]

            if not budget_stopped and not skip_choices:
                p_ch = progress.add_task(
                    f"[{run_name}] Running symmetric forced choices (shuffled blocks)...",
                    total=None,
                )
                try:
                    await run_forced_choices(
                        client=client,
                        cfg=cfg,
                        episodes=all_eps_complete,
                        choices_path=choices_path,
                        both_roles=both_roles_choices,
                        manifest_choices=manifest_choices,
                        excluded_path=excluded_path,
                    )
                except BudgetExceededError as exc:
                    budget_stopped = True
                    console.print(f"[bold yellow]Choices paused on budget cap: {exc}[/bold yellow]")
                ch_disk = _load_existing_choices(choices_path)
                progress.update(
                    p_ch,
                    description=(
                        f"[{run_name}] Forced choices on disk: {len(ch_disk)} "
                        f"(${ledger.total_cost_usd:.2f})"
                    ),
                )

    # Cumulative disk totals (always accurate across resumes)
    disk_eps = [json.loads(p.read_text(encoding="utf-8")) for p in episodes_dir.glob("*.json")]
    ep_files_complete = sum(1 for e in disk_eps if e.get("complete") is True)
    ep_files_failed = sum(1 for e in disk_eps if e.get("failed") is True and not e.get("complete"))
    choices_on_disk = _load_existing_choices(choices_path)
    choices_excl_on_disk = _load_existing_exclusions(excluded_path)
    summary = {
        "run_name": run_name,
        "phase": phase,
        "mock": mock,
        "rating_scale_max": rating_scale_max,
        "manifest_sha256": manifest_sha256,
        "probe_manifest_sha256": probe_sha256,
        "episodes_completed": ep_files_complete,
        "episodes_failed": ep_files_failed,
        "episodes_planned": len(ep_specs),
        "choices_completed": len(choices_on_disk),
        "choices_excluded": len(choices_excl_on_disk),
        "choices_planned": len(manifest_choices) if manifest_choices is not None else None,
        "budget_stopped": budget_stopped,
        "total_cost_usd": round(ledger.total_cost_usd, 4),
        "cost_by_model": ledger.summary_by_model(),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if budget_stopped:
        console.print(
            f"[bold red]Run '{run_name}' INCOMPLETE (stopped on budget cap):[/bold red] "
            f"{summary['episodes_completed']}/{summary['episodes_planned']} episodes "
            f"(failed={ep_files_failed}), {summary['choices_completed']}/{summary['choices_planned']} choices, "
            f"total cost = ${summary['total_cost_usd']:.2f}"
        )
    else:
        console.print(
            f"[bold green]Run '{run_name}' finished:[/bold green] "
            f"{summary['episodes_completed']}/{summary['episodes_planned']} episodes "
            f"(failed={ep_files_failed}), {summary['choices_completed']} choices "
            f"(excluded={summary['choices_excluded']}), total cost = ${summary['total_cost_usd']:.2f}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 'Mighty Models, Terrible Teammates' experiments.")
    parser.add_argument("--phase", choices=["calibrate", "pilot", "main", "v2"], required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--budget-cap", type=float, default=25.0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seeds", type=str, default="301,302,303,304")
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--rating-scale-max", type=int, default=7)
    parser.add_argument("--single-role-choices", action="store_true")
    parser.add_argument("--skip-choices", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--probe-manifest", type=Path, default=None)
    parser.add_argument("--allow-unfrozen", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or (f"{args.phase}_mock" if args.mock else args.phase)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    summary = asyncio.run(
        run_experiment(
            phase=args.phase,
            run_name=run_name,
            budget_cap_usd=args.budget_cap,
            mock=args.mock,
            seeds=seeds,
            max_concurrent=args.max_concurrent,
            rating_scale_max=args.rating_scale_max,
            both_roles_choices=not args.single_role_choices,
            probe_manifest_path=args.probe_manifest,
            manifest_path=args.manifest,
            skip_choices=args.skip_choices,
            strict_freeze=False if args.allow_unfrozen else None,
        )
    )
    if summary.get("budget_stopped"):
        sys.exit(2)


if __name__ == "__main__":
    main()
