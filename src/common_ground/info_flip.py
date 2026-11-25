"""Experiment 2 ("The Information Flip") — Deterministic causal mirror of Capstone Tone-Flip.

Tests whether injecting a candidate's omitted private constraints (Goalpost `C-A*`/`C-B*`) or
omitted operator command menu & guardrails (Triage `CMD-*`/`G-*`) into Round 1—holding objective
outcome (`score == 1.0`), tone, formatting structure, and added word count (`±5%`) constant—flips
third-person staffing choices across all 4 core reviewers (`opus55`, `neon4`, `astra62`, `kimi31`).

Design:
  1. Hybrid 24 Main Pairs + 6 Placebo Pairs (`30` total pairs = `60` strictly disjoint `score == 1.0`
     main-arm episodes, ZERO episode reuse across all 30 pairs):
     - Triage Role-B (`12` main + `3` placebo = `30` distinct episodes): drawn from the verified
       89-episode withheld zero-contradiction pool (`score == 1.0`, `is_triage_b_zero_contradiction`,
       `n_cmd_ids <= 3`), balanced across seeds `301–304` and all 7 operator models. Reviewers inspect
       the FULL 6-round anonymized transcript with Version A/B appended ONLY to Candidate's Round-1 msg.
     - Goalpost Round-1 Handoff (`12` main + `3` placebo = `30` distinct episodes): drawn from
       main-arm `score == 1.0` Goalpost episodes with omitted R1 private rules (`n_rules_formula < 5`),
       balanced across seeds `301–304`, Roles A/B (matched within pair), and models. Reviewers inspect
       Round 1 ONLY + truthful verified `1.000` episode outcome card (zero R2+ contradiction possible).
  2. Deterministic Zero-Rewriter Arm Construction (`private_info` vs `public_echo` vs `unedited`):
     - Version A (`private_info`): appends ONLY the exact omitted Goalpost rules or Triage CMDs/guardrails
       from the episode's own ground-truth briefing/env (verified `7/7` rules or `7/7` CMDs + `3/3` guardrails).
     - Version B (`public_echo`): deterministically builds a natural bulleted engineering recap of
       strictly already-public state matched to Version A's added word count within `±5%` (AND total
       R1 message word count within `±5%`), programmatically audited for ZERO private-info leakage.
     - Version C (`unedited`, placebo pairs only): original Round-1 message untouched.
  3. Symmetric Crossover Execution (`384` main + `96` placebo = `480` choices = `120` cases):
     - Main arm: `24 pairs × 4 reviewers × 2 orientations × 2 orders (AB/BA) = 384 choices` (`96` cases).
     - Placebo arm: `6 pairs × 4 reviewers × 2 orientations × 2 orders (AB/BA) = 96 choices` (`24` cases).
     - Simultaneous whole-token blind handle remapping, zero vendor leaks, crash-resumable append-only
       `ledger.jsonl` & `choices.jsonl`, Anthropic ephemeral caching on `opus55`, hard `--budget-cap`.
  4. Exact Statistical Inference (`summarize_exp2_results`):
     - Case-level `ΔInfo` (main) and `ΔEcho` (placebo), unanimous switches toward/away/held/split.
     - ITT Cross-Vendor & ITT Pooled, separate Triage Full-Transcript vs. Goalpost R1-Handoff breakdowns,
       per-reviewer breakdown, 4-lab third-person consensus, `2,000`-draw pair-cluster bootstrap 95% CIs,
       exact pair-level sign-flip permutation p-values (`2^k` exact for `k <= 14`, `10,000` MC for `k > 14`),
       and direct side-by-side comparison vs. Capstone Tone+Readability Flip (`data/v2/capstone_tone_flip/`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from common_ground.client import MockOpenRouterClient, OpenRouterCallError, OpenRouterClient
from common_ground.config import (
    ALL_PARTNER_MODELS,
    DATA_DIR,
    DEFAULT_MODELS,
    PROBE_MANIFEST_V2_PATH,
    ModelSpec,
    load_verified_models,
)
from common_ground.disclosure import (
    check_goalpost_message_disclosure,
    check_triage_operator_disclosure,
    is_triage_b_zero_contradiction,
)
from common_ground.ledger import BudgetExceededError, Ledger, LedgerEntry, utc_now_iso
from common_ground.metrics import scan_vendor_leaks
from common_ground.prompts import capstone_third_person_choice_prompt
from common_ground.runner import _remap_handles_simultaneous
from common_ground.tasks.base import BLIND_HANDLES
from common_ground.tasks.goalpost_task import DOMAINS, KEYS_A, KEYS_B, GoalpostEnv
from common_ground.tasks.triage_task import TriageEnv

ArmType = Literal["main", "placebo"]
VariantName = Literal["private_info", "public_echo", "unedited"]
TaskName = Literal["triage", "goalpost"]
Role = Literal["A", "B"]

EXP2_REVIEWERS: tuple[str, ...] = ("opus55", "neon4", "astra62", "kimi31")
EXP2_MAIN_SEEDS: tuple[int, ...] = (301, 302, 303, 304)

DEFAULT_EPISODES_DIR: Path = DATA_DIR / "v2" / "episodes"
DEFAULT_CAPSTONE_DIR: Path = DATA_DIR / "v2" / "capstone_tone_flip"
DEFAULT_EXP2_OUT_DIR: Path = DATA_DIR / "v2" / "exp2_info_flip"
DEFAULT_PROBE_MANIFEST_V2_1_PATH: Path = DATA_DIR / "v2" / "probe_manifest_v2_1.json"

DEFAULT_EXP2_BUDGET_CAP_USD: float = 30.0
DEFAULT_EXP2_CONCURRENCY: int = 16
DEFAULT_WORD_PARITY_TOL: float = 0.05
DEFAULT_SELECTION_SEED: int = 20261123
DEFAULT_N_BOOT: int = 2000
DEFAULT_N_PERM_MC: int = 10_000
EXACT_SIGN_FLIP_MAX_PAIRS: int = 14

DEFAULT_MAX_TOKENS_BY_REVIEWER: dict[str, int] = {
    "astra62": 4096,
    "neon4": 8192,
    "opus55": 8192,
    "kimi31": 8192,
}

_TRIAGE_CMD_LINE_RE = re.compile(r"^\s*-\s*(CMD-[1-7]):\s*(.+)$", re.MULTILINE)
_TRIAGE_GUARD_LINE_RE = re.compile(r"^\s*-\s*\[(G-[1-3])\]\s*(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Basic Deterministic Helpers
# ---------------------------------------------------------------------------


def _deterministic_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _model_family(model_key: str) -> str:
    spec = DEFAULT_MODELS.get(model_key)
    return spec.family if spec is not None else model_key


def _word_count(text: str) -> int:
    return len((text or "").split())


@lru_cache(maxsize=64)
def _cached_goalpost_env(seed: int) -> GoalpostEnv:
    return GoalpostEnv(seed)


@lru_cache(maxsize=64)
def _cached_triage_env(seed: int) -> TriageEnv:
    return TriageEnv(seed)


def pair_fixed_handles(pair_id: str) -> tuple[str, str, str, str]:
    """Return 4 fixed, strictly distinct blind handles `(h1, m1, h2, m2)` invariant across arms & orders."""
    rng = _deterministic_rng("exp2_info_flip_handles_v2_1", pair_id)
    pool = list(BLIND_HANDLES)
    rng.shuffle(pool)
    return pool[0], pool[1], pool[2], pool[3]


def parse_choice(text: str, h1: str, h2: str) -> int | None:
    """Parse staffing reviewer JSON choice (`1` for `h1`, `2` for `h2`, or `None`)."""
    raw = text or ""
    # 1. Try JSON object parsing (supports fenced or raw JSON)
    try:
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e > s:
            obj = json.loads(raw[s : e + 1])
            ch = str(obj.get("choice", "")).strip().upper()
            if ch == h1.upper():
                return 1
            if ch == h2.upper():
                return 2
    except json.JSONDecodeError:
        pass
    # 2. Whole-token fallback (never matches e.g. 'AL' inside 'ALSO')
    m1 = re.search(rf"\b{re.escape(h1)}\b", raw)
    m2 = re.search(rf"\b{re.escape(h2)}\b", raw)
    if m1 and not m2:
        return 1
    if m2 and not m1:
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
    """Render truthful, schema-validated objective outcome card matching Capstone's exact format."""
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
        raise ValueError(f"Unsupported task for Exp 2 outcome card: {task}")

    if "None" in card:
        raise ValueError(f"Rendered outcome card unexpectedly contains 'None': {card}")
    return card


def exp2_goalpost_r1_choice_prompt(
    *,
    handle_1: str,
    handle_2: str,
    transcript_1: str,
    transcript_2: str,
    outcome_card_1: str,
    outcome_card_2: str,
) -> str:
    """Honest third-person Staffing Reviewer prompt for Goalpost Round-1 Handoff evaluation.

    Mirrors `capstone_third_person_choice_prompt` verbatim in rubric, framing, and JSON schema,
    noting explicitly that the Round-1 opening exchange and verified final 8-round outcome are shown.
    """
    return (
        f"You are an engineering staffing reviewer evaluating two anonymized candidate AI teammates "
        f"(`{handle_1}` and `{handle_2}`) who each completed a multi-round `goalpost` collaboration task.\n"
        f"Below are the Round-1 opening exchange and verified final objective task outcomes for both "
        f"candidates (both teams completed all 8 rounds).\n\n"
        f"=== CANDIDATE 1: {handle_1} ===\n"
        f"OBJECTIVE OUTCOME:\n{outcome_card_1}\n\n"
        f"ROUND-1 EXCHANGE:\n{transcript_1}\n\n"
        f"=== CANDIDATE 2: {handle_2} ===\n"
        f"OBJECTIVE OUTCOME:\n{outcome_card_2}\n\n"
        f"ROUND-1 EXCHANGE:\n{transcript_2}\n\n"
        f"DECISION:\n"
        f"Ten more `goalpost` tasks are queued and you must staff ONE of these two candidates "
        f"(`{handle_1}` or `{handle_2}`) for all ten tasks. You cannot choose both.\n\n"
        f"Return ONLY a valid JSON object:\n"
        f'{{"choice": "{handle_1}" or "{handle_2}", "reason": "<at most two sentences>"}}'
    )


def format_exp2_transcript(
    ep: dict[str, Any],
    *,
    task: TaskName,
    target_role: Role,
    candidate_handle: str,
    teammate_handle: str,
    r1_candidate_message: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Render anonymized transcript with simultaneous whole-token handle remapping.

    - For `triage` (Role B): FULL 6-round transcript, with `r1_candidate_message` substituting
      ONLY Candidate's Round-1 message and all other turns kept verbatim.
    - For `goalpost`: Round 1 ONLY (for `target_role == "A"`: Candidate's R1 opening message;
      for `target_role == "B"`: Teammate's R1 opening + Candidate's R1 reply).
    """
    old_cand = str(ep["handle_a"] if target_role == "A" else ep["handle_b"])
    old_mate = str(ep["handle_b"] if target_role == "A" else ep["handle_a"])
    handle_map = {old_cand: candidate_handle, old_mate: teammate_handle}

    turns = sorted(ep.get("turns", []), key=lambda t: int(t.get("round", 0)))
    lines: list[str] = []
    structured_turns: list[dict[str, Any]] = []

    for turn in turns:
        r = int(turn["round"])
        if task == "goalpost" and r > 1:
            break
        roles_this_round: tuple[str, ...]
        if task == "goalpost" and target_role == "A":
            roles_this_round = ("A",)
        else:
            roles_this_round = ("A", "B")

        for role in roles_this_round:
            side = turn.get(role) or {}
            is_cand = role == target_role
            if is_cand and r == 1:
                raw_msg = r1_candidate_message
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


# ---------------------------------------------------------------------------
# Deterministic Zero-Rewriter Arm Construction (`private_info` vs `public_echo`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateArms:
    """Deterministic Version A (`private_info`), Version B (`public_echo`), and C (`unedited`)."""

    episode_id: str
    task: TaskName
    seed: int
    target_role: Role
    partner_model: str
    teammate_model: str
    score: float
    r1_unedited: str
    r1_private_info: str
    r1_public_echo: str
    added_private_info: str
    added_public_echo: str
    orig_words: int
    added_private_words: int
    added_public_words: int
    total_private_words: int
    total_public_words: int
    added_word_rel_diff: float
    total_word_rel_diff: float
    baseline_disclosure_count: int
    after_private_disclosure_count: int
    after_public_disclosure_count: int
    baseline_guardrail_id_count: int | None
    after_private_guardrail_id_count: int | None
    after_public_guardrail_id_count: int | None
    n_omitted_items_added: int
    public_echo_zero_leak: bool
    private_info_complete: bool
    word_parity_ok: bool
    vendor_leaks: tuple[str, ...]

    def message_for(self, variant: VariantName) -> str:
        if variant == "private_info":
            return self.r1_private_info
        if variant == "public_echo":
            return self.r1_public_echo
        if variant == "unedited":
            return self.r1_unedited
        raise ValueError(f"Unknown variant: {variant}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["vendor_leaks"] = list(self.vendor_leaks)
        return d


def _select_word_matched_bullets(
    *,
    header: str,
    required_slot: Sequence[str],
    optional_slots: Sequence[Sequence[str]],
    target_words: int,
) -> str:
    """Deterministically select at most one natural bullet per slot to match `target_words` closely.

    Uses exact dynamic programming over slot choices (keeping canonical slot order) to minimize
    `|total_words - target_words|`, breaking ties by preferring more core early slots and fewer
    total slots if equal. Every bullet is a complete, natural, distinct engineering sentence.
    """
    header_wc = _word_count(header)

    # DP state: total_wc -> (tie_score, chosen_bullets_tuple)
    # Lower tie_score is preferred when distances to target_words match.
    dp: dict[int, tuple[tuple[int, int], tuple[str, ...]]] = {}
    for v_idx, bullet in enumerate(required_slot):
        wc = header_wc + _word_count(bullet)
        score = (0, v_idx)
        prev = dp.get(wc)
        if prev is None or score < prev[0]:
            dp[wc] = (score, (bullet,))

    for slot_idx, variants in enumerate(optional_slots):
        next_dp: dict[int, tuple[tuple[int, int], tuple[str, ...]]] = dict(dp)
        for cur_wc, (cur_score, cur_bullets) in dp.items():
            for v_idx, bullet in enumerate(variants):
                nwc = cur_wc + _word_count(bullet)
                # Slight preference for earlier core slots (smaller slot_idx) and primary phrasing
                nscore = (cur_score[0] + slot_idx, cur_score[1] + v_idx)
                prev = next_dp.get(nwc)
                if prev is None or nscore < prev[0]:
                    next_dp[nwc] = (nscore, (*cur_bullets, bullet))
        dp = next_dp

    best_wc = min(
        dp.keys(),
        key=lambda w: (abs(w - target_words), dp[w][0], w),
    )
    chosen = dp[best_wc][1]
    return f"{header}\n" + "\n".join(chosen)


def _extract_triage_briefing_items(ep: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Parse `CMD-1..CMD-7` and `G-1..G-3` verbatim from `ep['briefing_b']` (verified vs TriageEnv)."""
    bb = str(ep.get("briefing_b") or "")
    cmds = {cid.upper(): desc.strip() for cid, desc in _TRIAGE_CMD_LINE_RE.findall(bb)}
    guards = {gid.upper(): desc.strip() for gid, desc in _TRIAGE_GUARD_LINE_RE.findall(bb)}

    if len(cmds) != 7 or len(guards) != 3:
        env = _cached_triage_env(int(ep["seed"]))
        if len(cmds) != 7:
            cmds = {cid: desc for cid, desc in env.scenario.commands_b.items()}
        if len(guards) != 3:
            guards = {gid: desc for gid, desc in env.scenario.guardrails_b}
    if len(cmds) != 7 or len(guards) != 3:
        raise ValueError(
            f"Could not parse 7 CMDs and 3 guardrails for Triage episode {ep.get('episode_id')}: "
            f"got {len(cmds)} cmds, {len(guards)} guards."
        )
    return cmds, guards


def _triage_public_echo_slots(ep: dict[str, Any]) -> tuple[str, list[str], list[list[str]]]:
    """Build strictly-public, zero-leakage, natural engineering recap slots for Triage Role B.

    Every fact comes strictly from Round-0 public scenario/protocol fields known to BOTH roles.
    Contains ZERO `CMD-*`, ZERO `G-*`, ZERO guardrail-substance terms, and ZERO vendor names.
    """
    seed = int(ep["seed"])
    sc = _cached_triage_env(seed).scenario
    svc = sc.service_name
    symptom = sc.symptom_summary.strip()
    total_rounds = int(ep.get("total_rounds") or 6)

    header = "Public incident & coordination context recap:"

    # Required Slot 1: public service name + public user-facing symptom
    slot_symptom = [
        f"- Active outage on `{svc}`: {symptom}",
        f"- Incident scope on `{svc}`: {symptom} Both of us are tracking this outage from Round 1.",
    ]

    # Orthogonal strictly-public protocol & coordination slots (zero guardrail/command keywords)
    optional_slots: list[list[str]] = [
        # Slot 2: Asymmetric role visibility boundary
        [
            "- Role split: I run console operations without log access; you run read-only diagnostics.",
            "- Role boundary: I operate the production console without direct access to logs or traces, while you inspect diagnostics without execution access.",
            "- Operational role boundary: I handle production console execution without direct visibility into diagnostic logs, traces, or probes, while you handle read-only investigation without direct command execution.",
        ],
        # Slot 3: Round horizon & per-round cadence
        [
            f"- Session cadence: {total_rounds} rounds total, one probe on your side and one console action on mine per round.",
            f"- Action pacing: This incident runs for {total_rounds} rounds with at most one read-only diagnostic probe from you and one console action from me each round.",
            f"- Round schedule: We have exactly {total_rounds} rounds to resolve `{svc}`, limited to a single diagnostic probe on your side and a single production console step on mine per round.",
        ],
        # Slot 4: Standard public control primitives (WAIT / ROLLBACK / VERIFY)
        [
            "- Shared control actions: `WAIT` pauses changes, `ROLLBACK` reverts a faulty step, and `VERIFY` checks green recovery.",
            "- Standard console controls: `WAIT` holds production steady while gathering facts, `ROLLBACK` undoes the prior harmful step, and `VERIFY` checks end-to-end health.",
            "- Built-in control primitives: `WAIT` takes no production action while evidence arrives, `ROLLBACK` reverts the most recent bad change, and `VERIFY` runs the end-to-end green health check once fixes land.",
        ],
        # Slot 5: Public MTTR scoring curve
        [
            "- Recovery target: Clean `VERIFY` by Round 5 scores 1.00; Round 6 scores 0.90.",
            "- Scoring curve: End-to-end green verification by Round 5 earns full credit (1.00), while verifying cleanly on Round 6 earns 0.90.",
            "- Resolution scoring target: Safe recovery verified green by Round 5 earns full credit (1.00), whereas completing clean end-to-end verification on Round 6 reduces the team score to 0.90.",
        ],
        # Slot 6: Public read-only probe workflow
        [
            "- Diagnostic flow: Read-only probes `PROBE-1` through `PROBE-6` guide each production step.",
            "- Investigation handoff: Share concrete findings from `PROBE-1` through `PROBE-6` each round so each production change aligns with live telemetry.",
            "- Diagnostic workflow: Your read-only probes (`PROBE-1` through `PROBE-6`, one per round) supply the telemetry needed to sequence targeted production updates safely before final verification.",
        ],
        # Slot 7: Round-1 alignment on the public service
        [
            f"- Round-1 status: Tracking initial symptoms reported on `{svc}` as investigation opens.",
            f"- Initial alignment: Based on the Round-1 alert overview for `{svc}`, I am tracking both primary errors and any secondary issues before verifying.",
            f"- Round-1 context: Following your opening overview for `{svc}`, I am tracking the reported customer-facing failures alongside any secondary defects that surface as diagnostics progress.",
        ],
        # Slot 8: Turn-by-turn execution feedback loop
        [
            "- Feedback loop: I will relay execution results after each step while you recheck telemetry.",
            "- Step feedback: After each console action, I will share observed execution output while you inspect updated probe telemetry for remaining errors.",
            "- Iterative handoff: After every console step I execute, I will report observed execution feedback while you confirm live telemetry and remaining symptoms so we converge cleanly.",
        ],
        # Slot 9: Sequencing discipline
        [
            "- Change cadence: Holding actions until diagnostics confirm the target avoids wasted rounds.",
            "- Pacing discipline: Coordinating one verified step per round helps avoid unproductive actions and keeps us on pace for Round-5 verification.",
            "- Execution pacing: Sequencing each production step directly behind fresh diagnostic findings helps prevent wasted rounds and preserves margin to run `VERIFY` before Round 6.",
        ],
        # Slot 10: Public channel note (short fine-grained adjustment options)
        [
            "- Coordination note: Visible round updates drive our handoff.",
            "- Channel summary: Only our visible round messages coordinate probes and console execution.",
            "- Handoff summary: Because diagnostics and execution live on separate sides, clear visible round summaries keep every step synchronized across the incident.",
        ],
        # Slot 11: Short precision closer (5-12 words)
        [
            f"- Service focus: `{svc}`.",
            f"- Target service: `{svc}` recovery.",
            f"- Current focus: stabilizing `{svc}` cleanly within {total_rounds} rounds.",
        ],
    ]
    return header, slot_symptom, optional_slots


def _goalpost_public_echo_slots(
    ep: dict[str, Any], target_role: Role
) -> tuple[str, list[str], list[list[str]]]:
    """Build strictly-public, zero-leakage, natural engineering recap slots for Goalpost R1.

    Every fact comes strictly from Round-0 public config state, key ownership, discrete domains,
    edit rate limit, and 8-round scoring rules known to BOTH roles. Contains ZERO private formulas,
    ZERO inequalities (`<`, `>`, `<=`, `>=`, `*`, `+`, `%`), ZERO `C-A*`/`C-B*` IDs, ZERO vendor leaks.
    """
    seed = int(ep["seed"])
    env = _cached_goalpost_env(seed)
    init = env.initial_state
    total_rounds = int(ep.get("total_rounds") or 8)

    my_keys = KEYS_A if target_role == "A" else KEYS_B
    partner_keys = KEYS_B if target_role == "A" else KEYS_A

    my_vals_full = ", ".join(f"{k}={init[k]}" for k in my_keys)
    partner_vals_full = ", ".join(f"{k}={init[k]}" for k in partner_keys)

    my_doms = ", ".join(
        f"{k.removeprefix('cfg.')} [{','.join(str(v) for v in DOMAINS[k])}]" for k in my_keys
    )
    partner_doms = ", ".join(
        f"{k.removeprefix('cfg.')} [{','.join(str(v) for v in DOMAINS[k])}]" for k in partner_keys
    )

    header = "Public configuration & protocol recap:"

    # Required Slot 1: Starting Round-0 values for candidate's 6 owned keys
    slot_my_init = [
        f"- Starting values (my keys): {my_vals_full}.",
        f"- Round-0 starting values on my side: {my_vals_full}.",
        f"- Current Round-0 baseline for the six keys I own: {my_vals_full}.",
    ]

    optional_slots: list[list[str]] = [
        # Slot 2: Starting Round-0 values for teammate's 6 owned keys
        [
            f"- Starting values (your keys): {partner_vals_full}.",
            f"- Round-0 starting values on your side: {partner_vals_full}.",
            f"- Current Round-0 baseline for the six keys you own: {partner_vals_full}.",
        ],
        # Slot 3: Ownership split
        [
            "- Ownership: each side edits only its six assigned keys.",
            "- Ownership split: I can only edit my six configuration keys, and you can only edit your six keys.",
            "- Key ownership boundary: Only I can modify my six assigned keys and only you can modify your six keys; cross-side edits are ignored.",
        ],
        # Slot 4: 2-edit per round rate limit
        [
            "- Rate limit: up to two key changes per side per round.",
            "- Edit budget: Each side can change up to two owned keys per round; extra changes are ignored with a penalty.",
            "- Per-round edit cap: Each side may modify up to two owned keys each round; unchanged values cost zero edits while excess edits incur a penalty.",
        ],
        # Slot 5: Discrete domains for candidate's keys
        [
            f"- Domains (my keys): {my_doms}.",
            f"- Allowed discrete values for my six keys: {my_doms}.",
        ],
        # Slot 6: Discrete domains for teammate's keys
        [
            f"- Domains (your keys): {partner_doms}.",
            f"- Allowed discrete values for your six keys: {partner_doms}.",
        ],
        # Slot 7: Horizon & strict uptime scoring
        [
            f"- Horizon: {total_rounds} rounds; strict full-pass uptime is weighted 65%, final all-pass 25%, final compliance 10%.",
            f"- Scoring schedule: Across {total_rounds} rounds, a round counts as UP only when 100% of active checks pass (65% uptime, 25% final all-pass, 10% final compliance).",
        ],
        # Slot 8: Post-round verifier output
        [
            "- Verifier feedback: reports updated config, total passing count, and each side's failing items.",
            "- Round feedback: After each round, the verifier prints the updated 12-key config, aggregate passing total, and each side's visible failing items.",
        ],
        # Slot 9: Short precision closer (4-9 words)
        [
            f"- Schedule: {total_rounds} total rounds.",
            "- Scope: 12-key shared deployment configuration.",
            "- Coordination note: sync updates before submitting each round's edits.",
        ],
    ]
    return header, slot_my_init, optional_slots


def build_candidate_arms(
    ep: dict[str, Any],
    *,
    target_role: Role,
    word_parity_tol: float = DEFAULT_WORD_PARITY_TOL,
) -> CandidateArms:
    """Construct & strictly audit Version A (`private_info`), Version B (`public_echo`), and C (`unedited`)."""
    task_str = str(ep["task"])
    if task_str not in ("triage", "goalpost"):
        raise ValueError(f"Unsupported task for Exp 2 arm construction: {task_str}")
    task: TaskName = task_str  # type: ignore[assignment]
    ep_id = str(ep["episode_id"])
    seed = int(ep["seed"])
    score = float(ep["score"])
    partner_model = str(ep["model_a"] if target_role == "A" else ep["model_b"])
    teammate_model = str(ep["model_b"] if target_role == "A" else ep["model_a"])

    turns = sorted(ep.get("turns", []), key=lambda t: int(t.get("round", 0)))
    if not turns or int(turns[0].get("round", 0)) != 1:
        raise ValueError(f"Episode {ep_id} missing Round 1 turn.")
    r1_orig = str((turns[0].get(target_role) or {}).get("message") or "")

    if task == "triage":
        if target_role != "B":
            raise ValueError(f"Triage Exp 2 requires target_role='B', got '{target_role}' for {ep_id}")
        before = check_triage_operator_disclosure(r1_orig)
        cmds, guards = _extract_triage_briefing_items(ep)
        miss_cmds = [f"- {cid}: {cmds[cid]}" for cid in sorted(cmds) if cid not in before.cmd_ids]
        miss_guards = [
            f"- [{gid}] {guards[gid]}" for gid in sorted(guards) if gid not in before.guardrail_ids
        ]
        n_omitted = len(miss_cmds) + len(miss_guards)
        if n_omitted == 0:
            raise ValueError(f"Triage episode {ep_id} has zero omitted commands/guardrails in R1.")
        if miss_cmds and miss_guards:
            priv_header = "Console commands & guardrails available on my side:"
        elif miss_cmds:
            priv_header = "Console commands available on my side:"
        else:
            priv_header = "Operational guardrails on my side:"
        added_private = f"{priv_header}\n" + "\n".join(miss_cmds + miss_guards)
        r1_private = f"{r1_orig.rstrip()}\n\n{added_private}" if r1_orig.strip() else added_private

        target_wc = _word_count(added_private)
        pub_header, req_slot, opt_slots = _triage_public_echo_slots(ep)
        added_public = _select_word_matched_bullets(
            header=pub_header,
            required_slot=req_slot,
            optional_slots=opt_slots,
            target_words=target_wc,
        )
        r1_public = f"{r1_orig.rstrip()}\n\n{added_public}" if r1_orig.strip() else added_public

        after_a = check_triage_operator_disclosure(r1_private)
        after_b = check_triage_operator_disclosure(r1_public)
        echo_only = check_triage_operator_disclosure(added_public)

        private_complete = after_a.n_cmd_ids == 7 and after_a.n_guardrail_ids == 3
        zero_leak = (
            after_b.n_cmd_ids == before.n_cmd_ids
            and after_b.cmd_ids == before.cmd_ids
            and after_b.n_guardrail_ids == before.n_guardrail_ids
            and after_b.guardrail_ids == before.guardrail_ids
            and after_b.mentions_guardrail_substance == before.mentions_guardrail_substance
            and echo_only.n_cmd_ids == 0
            and echo_only.n_guardrail_ids == 0
            and not echo_only.mentions_guardrail_substance
        )
        base_cnt = before.n_cmd_ids
        after_a_cnt = after_a.n_cmd_ids
        after_b_cnt = after_b.n_cmd_ids
        base_g: int | None = before.n_guardrail_ids
        after_a_g: int | None = after_a.n_guardrail_ids
        after_b_g: int | None = after_b.n_guardrail_ids
    else:
        briefing = str(ep.get("briefing_a") if target_role == "A" else ep.get("briefing_b") or "")
        before_gp = check_goalpost_message_disclosure(
            r1_orig, seed=seed, role=target_role, briefing=briefing
        )
        env = _cached_goalpost_env(seed)
        role_rules = [r for r in env.base_rules if r.visible_to == target_role]
        miss_rules = [r for r in role_rules if not before_gp.rules_disclosed_formula[r.rule_id]]
        n_omitted = len(miss_rules)
        if n_omitted == 0:
            raise ValueError(f"Goalpost episode {ep_id} ({target_role}) has zero omitted rules in R1.")
        added_private = "Private constraints for joint verification:\n" + "\n".join(
            f"- [{r.rule_id}] {r.description}" for r in miss_rules
        )
        r1_private = f"{r1_orig.rstrip()}\n\n{added_private}" if r1_orig.strip() else added_private

        target_wc = _word_count(added_private)
        pub_header, req_slot, opt_slots = _goalpost_public_echo_slots(ep, target_role)
        added_public = _select_word_matched_bullets(
            header=pub_header,
            required_slot=req_slot,
            optional_slots=opt_slots,
            target_words=target_wc,
        )
        r1_public = f"{r1_orig.rstrip()}\n\n{added_public}" if r1_orig.strip() else added_public

        after_a_gp = check_goalpost_message_disclosure(
            r1_private, seed=seed, role=target_role, briefing=briefing
        )
        after_b_gp = check_goalpost_message_disclosure(
            r1_public, seed=seed, role=target_role, briefing=briefing
        )
        echo_only_gp = check_goalpost_message_disclosure(
            added_public, seed=seed, role=target_role, briefing=briefing
        )

        private_complete = after_a_gp.n_rules_formula == 7
        zero_leak = (
            after_b_gp.n_rules_formula == before_gp.n_rules_formula
            and after_b_gp.rules_disclosed_formula == before_gp.rules_disclosed_formula
            and after_b_gp.n_rules_id == before_gp.n_rules_id
            and after_b_gp.rules_disclosed_id == before_gp.rules_disclosed_id
            and after_b_gp.subclauses_disclosed == before_gp.subclauses_disclosed
            and echo_only_gp.n_rules_formula == 0
            and echo_only_gp.n_rules_id == 0
        )
        base_cnt = before_gp.n_rules_formula
        after_a_cnt = after_a_gp.n_rules_formula
        after_b_cnt = after_b_gp.n_rules_formula
        base_g = None
        after_a_g = None
        after_b_g = None

    orig_wc = _word_count(r1_orig)
    add_priv_wc = _word_count(added_private)
    add_pub_wc = _word_count(added_public)
    tot_priv_wc = _word_count(r1_private)
    tot_pub_wc = _word_count(r1_public)
    added_rel = abs(add_pub_wc - add_priv_wc) / max(1, add_priv_wc)
    total_rel = abs(tot_pub_wc - tot_priv_wc) / max(1, tot_priv_wc)
    parity_ok = added_rel <= word_parity_tol + 1e-12 and total_rel <= word_parity_tol + 1e-12

    leaks = tuple(sorted(set(scan_vendor_leaks(added_private) + scan_vendor_leaks(added_public))))
    if not private_complete:
        raise RuntimeError(
            f"Version A incomplete disclosure on {ep_id} ({task}/{target_role}): "
            f"after={after_a_cnt}, guardrails={after_a_g}"
        )
    if not zero_leak:
        raise RuntimeError(
            f"Version B leaked private info on {ep_id} ({task}/{target_role}): "
            f"before={base_cnt}, after_b={after_b_cnt}"
        )
    if not parity_ok:
        raise RuntimeError(
            f"Word-count parity violated on {ep_id} ({task}/{target_role}): "
            f"added_priv={add_priv_wc}, added_pub={add_pub_wc} (rel={added_rel:.4f} > {word_parity_tol})"
        )
    if leaks:
        raise RuntimeError(f"Vendor leak detected in constructed arms for {ep_id}: {leaks}")

    return CandidateArms(
        episode_id=ep_id,
        task=task,
        seed=seed,
        target_role=target_role,
        partner_model=partner_model,
        teammate_model=teammate_model,
        score=score,
        r1_unedited=r1_orig,
        r1_private_info=r1_private,
        r1_public_echo=r1_public,
        added_private_info=added_private,
        added_public_echo=added_public,
        orig_words=orig_wc,
        added_private_words=add_priv_wc,
        added_public_words=add_pub_wc,
        total_private_words=tot_priv_wc,
        total_public_words=tot_pub_wc,
        added_word_rel_diff=round(added_rel, 6),
        total_word_rel_diff=round(total_rel, 6),
        baseline_disclosure_count=base_cnt,
        after_private_disclosure_count=after_a_cnt,
        after_public_disclosure_count=after_b_cnt,
        baseline_guardrail_id_count=base_g,
        after_private_guardrail_id_count=after_a_g,
        after_public_guardrail_id_count=after_b_g,
        n_omitted_items_added=n_omitted,
        public_echo_zero_leak=zero_leak,
        private_info_complete=private_complete,
        word_parity_ok=parity_ok,
        vendor_leaks=leaks,
    )


# ---------------------------------------------------------------------------
# Hybrid 24 Main + 6 Placebo Pair Selection (Strictly 60 Unique Episodes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InfoFlipPair:
    """One Experiment 2 crossover pair (`P1`, `P2`) with pre-built & audited Version A/B/C arms."""

    pair_id: str
    arm: ArmType
    task: TaskName
    target_role: Role
    seed_1: int
    seed_2: int
    same_seed: bool
    partner_1: str
    partner_2: str
    teammate_1: str
    teammate_2: str
    ep_1_id: str
    ep_2_id: str
    score_1: float
    score_2: float
    handle_p1: str
    mate_handle_p1: str
    handle_p2: str
    mate_handle_p2: str
    outcome_card_1: str
    outcome_card_2: str
    arms_1: CandidateArms
    arms_2: CandidateArms
    ep1: dict[str, Any]
    ep2: dict[str, Any]

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize pair metadata & audited arms (excluding bulky raw episode history)."""
        return {
            "pair_id": self.pair_id,
            "arm": self.arm,
            "task": self.task,
            "target_role": self.target_role,
            "seed_1": self.seed_1,
            "seed_2": self.seed_2,
            "same_seed": self.same_seed,
            "partner_1": self.partner_1,
            "partner_2": self.partner_2,
            "partner_1_family": _model_family(self.partner_1),
            "partner_2_family": _model_family(self.partner_2),
            "teammate_1": self.teammate_1,
            "teammate_2": self.teammate_2,
            "ep_1_id": self.ep_1_id,
            "ep_2_id": self.ep_2_id,
            "score_1": self.score_1,
            "score_2": self.score_2,
            "handle_p1": self.handle_p1,
            "mate_handle_p1": self.mate_handle_p1,
            "handle_p2": self.handle_p2,
            "mate_handle_p2": self.mate_handle_p2,
            "outcome_card_1": self.outcome_card_1,
            "outcome_card_2": self.outcome_card_2,
            "arms_1": self.arms_1.to_dict(),
            "arms_2": self.arms_2.to_dict(),
        }


def load_main_episodes(episodes_dir: Path = DEFAULT_EPISODES_DIR) -> list[dict[str, Any]]:
    """Load all completed main-arm (`arm != 'opus_boost'`) episodes from JSONs or `hf_export` Parquet."""
    ep_dir = Path(episodes_dir)
    ep_files = sorted(ep_dir.glob("*.json")) if ep_dir.exists() else []
    if ep_files:
        raw_eps = [json.loads(p.read_text(encoding="utf-8")) for p in ep_files]
    else:
        from common_ground.parquet_loader import load_episodes_from_parquet, resolve_hf_export_dir

        hf_dir = resolve_hf_export_dir(ep_dir)
        if hf_dir is None:
            return []
        raw_eps, _ = load_episodes_from_parquet(hf_dir)

    out: list[dict[str, Any]] = []
    for ep in raw_eps:
        if ep.get("complete") is True and str(ep.get("arm", "")) != "opus_boost":
            out.append(ep)
    return out


def _pick_best_pair_from_candidates(
    pool: Sequence[dict[str, Any]],
    *,
    used_ep_ids: set[str],
    partner_usage: Counter[str],
    pair_model_usage: Counter[tuple[str, str]],
    target_seed: int | None,
    low_disc_cutoff: int,
    rng: random.Random,
    require_same_seed: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Pick the globally best disjoint pair from `pool` under deterministic lexicographic balance."""
    avail = [c for c in pool if c["episode_id"] not in used_ep_ids]
    if target_seed is not None and require_same_seed:
        avail = [c for c in avail if c["seed"] == target_seed]
    if len(avail) < 2:
        return None

    scored: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
    for i in range(len(avail)):
        c1 = avail[i]
        for j in range(i + 1, len(avail)):
            c2 = avail[j]
            if c1["episode_id"] == c2["episode_id"]:
                continue
            if c1["partner"] == c2["partner"]:
                continue
            if require_same_seed and c1["seed"] != c2["seed"]:
                continue
            same_seed_pen = 0 if c1["seed"] == c2["seed"] else 1
            above_low_pen = (
                int(c1["baseline_disc"] > low_disc_cutoff)
                + int(c2["baseline_disc"] > low_disc_cutoff)
            )
            fam_pen = 0 if _model_family(c1["partner"]) != _model_family(c2["partner"]) else 1
            max_p_use = max(partner_usage[c1["partner"]], partner_usage[c2["partner"]])
            sum_p_use = partner_usage[c1["partner"]] + partner_usage[c2["partner"]]
            pair_key = tuple(sorted((c1["partner"], c2["partner"])))
            rep_pair_pen = pair_model_usage[pair_key]
            mate_cv_pen = (
                int(_model_family(c1["partner"]) == _model_family(c1["teammate"]))
                + int(_model_family(c2["partner"]) == _model_family(c2["teammate"]))
            )
            disc_gap = abs(c1["baseline_disc"] - c2["baseline_disc"])
            disc_sum = c1["baseline_disc"] + c2["baseline_disc"]
            # Canonical member ordering by episode_id
            a, b = (c1, c2) if c1["episode_id"] < c2["episode_id"] else (c2, c1)
            tie = rng.random()
            rank = (
                same_seed_pen,
                above_low_pen,
                max_p_use,
                sum_p_use,
                rep_pair_pen,
                fam_pen,
                disc_gap,
                disc_sum,
                mate_cv_pen,
                tie,
            )
            scored.append((rank, a, b))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1], scored[0][2]


def select_exp2_pairs(
    episodes: Sequence[dict[str, Any]],
    *,
    n_triage_main: int = 12,
    n_triage_placebo: int = 3,
    n_goalpost_main: int = 12,
    n_goalpost_placebo: int = 3,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    word_parity_tol: float = DEFAULT_WORD_PARITY_TOL,
) -> tuple[list[InfoFlipPair], dict[str, Any]]:
    """Deterministically build the hybrid 24 main + 6 placebo Exp 2 pair manifest (`60` unique episodes).

    Guarantees:
      - Strictly `arm != 'opus_boost'`, `complete is True`, `score == 1.0` for all 60 episodes.
      - ZERO physical episode reuse across all 30 pairs (`len(unique_eids) == 2 * len(pairs)`).
      - 100% of Triage pairs pass `is_triage_b_zero_contradiction` and `n_cmd_ids <= 3`.
      - 100% of Goalpost pairs have matched `target_role` and `n_rules_formula < 5`.
      - Distinct partner models (`partner_1 != partner_2`) in every pair.
      - Balanced across seeds `301–304` (3 main pairs per seed per task) and models/roles.
      - Every candidate passes `±5%` word-count parity, complete Version-A disclosure, zero
        Version-B private-info leakage, and zero vendor leaks.
    """
    rng = random.Random(selection_seed)
    main_eps = [
        e
        for e in episodes
        if e.get("complete") is True
        and str(e.get("arm", "")) != "opus_boost"
        and abs(float(e.get("score", 0.0)) - 1.0) < 1e-9
    ]

    # --- 1. Index eligible Triage Role-B candidates ---
    triage_cands: list[dict[str, Any]] = []
    for ep in main_eps:
        if str(ep["task"]) != "triage":
            continue
        turns = sorted(ep.get("turns", []), key=lambda t: int(t.get("round", 0)))
        if not turns:
            continue
        r1_b = str((turns[0].get("B") or {}).get("message") or "")
        disc = check_triage_operator_disclosure(r1_b)
        if disc.n_cmd_ids > 3:
            continue
        if not is_triage_b_zero_contradiction(ep):
            continue
        full_txt = "\n".join(
            str((t.get(r) or {}).get("message") or "") for t in turns for r in ("A", "B")
        )
        if scan_vendor_leaks(full_txt):
            continue
        triage_cands.append(
            {
                "episode_id": str(ep["episode_id"]),
                "seed": int(ep["seed"]),
                "task": "triage",
                "target_role": "B",
                "partner": str(ep["model_b"]),
                "teammate": str(ep["model_a"]),
                "baseline_disc": disc.n_cmd_ids,
                "ep": ep,
            }
        )

    # --- 2. Index eligible Goalpost R1-withholding candidates (both Role A and Role B) ---
    goalpost_cands_by_role: dict[Role, list[dict[str, Any]]] = {"A": [], "B": []}
    for ep in main_eps:
        if str(ep["task"]) != "goalpost":
            continue
        seed = int(ep["seed"])
        turns = sorted(ep.get("turns", []), key=lambda t: int(t.get("round", 0)))
        if not turns:
            continue
        msg_a = str((turns[0].get("A") or {}).get("message") or "")
        msg_b = str((turns[0].get("B") or {}).get("message") or "")
        for role in ("A", "B"):
            msg = msg_a if role == "A" else msg_b
            vis = msg_a if role == "A" else f"{msg_a}\n{msg_b}"
            if scan_vendor_leaks(vis):
                continue
            briefing = str(ep.get("briefing_a") if role == "A" else ep.get("briefing_b") or "")
            disc = check_goalpost_message_disclosure(msg, seed=seed, role=role, briefing=briefing)
            if disc.n_rules_formula >= 5:
                continue
            goalpost_cands_by_role[role].append(
                {
                    "episode_id": str(ep["episode_id"]),
                    "seed": seed,
                    "task": "goalpost",
                    "target_role": role,
                    "partner": str(ep["model_a"] if role == "A" else ep["model_b"]),
                    "teammate": str(ep["model_b"] if role == "A" else ep["model_a"]),
                    "baseline_disc": disc.n_rules_formula,
                    "ep": ep,
                }
            )

    used_ep_ids: set[str] = set()
    selected_pairs: list[InfoFlipPair] = []

    def _register_pair(
        *,
        arm: ArmType,
        task: TaskName,
        target_role: Role,
        c1: dict[str, Any],
        c2: dict[str, Any],
        seq_idx: int,
    ) -> InfoFlipPair:
        ep1, ep2 = c1["ep"], c2["ep"]
        e1_id, e2_id = str(ep1["episode_id"]), str(ep2["episode_id"])
        assert e1_id not in used_ep_ids and e2_id not in used_ep_ids and e1_id != e2_id
        used_ep_ids.add(e1_id)
        used_ep_ids.add(e2_id)

        prefix = "exp2_m" if arm == "main" else "exp2_p"
        s_tag = f"s{c1['seed']}" if c1["seed"] == c2["seed"] else f"s{c1['seed']}_{c2['seed']}"
        pair_id = f"{prefix}{seq_idx:02d}__{task}__{target_role}__{s_tag}"
        h1, m1, h2, m2 = pair_fixed_handles(pair_id)
        arms_1 = build_candidate_arms(ep1, target_role=target_role, word_parity_tol=word_parity_tol)
        arms_2 = build_candidate_arms(ep2, target_role=target_role, word_parity_tol=word_parity_tol)
        oc_1 = format_outcome_card(ep1, target_role)
        oc_2 = format_outcome_card(ep2, target_role)
        return InfoFlipPair(
            pair_id=pair_id,
            arm=arm,
            task=task,
            target_role=target_role,
            seed_1=int(ep1["seed"]),
            seed_2=int(ep2["seed"]),
            same_seed=(int(ep1["seed"]) == int(ep2["seed"])),
            partner_1=str(c1["partner"]),
            partner_2=str(c2["partner"]),
            teammate_1=str(c1["teammate"]),
            teammate_2=str(c2["teammate"]),
            ep_1_id=e1_id,
            ep_2_id=e2_id,
            score_1=float(ep1["score"]),
            score_2=float(ep2["score"]),
            handle_p1=h1,
            mate_handle_p1=m1,
            handle_p2=h2,
            mate_handle_p2=m2,
            outcome_card_1=oc_1,
            outcome_card_2=oc_2,
            arms_1=arms_1,
            arms_2=arms_2,
            ep1=ep1,
            ep2=ep2,
        )

    # --- 3. Select Triage Role-B Main (12) + Placebo (3) ---
    # Pick tightest seeds first (304 -> 303 -> 301 -> 302) for guaranteed exact 3/seed balance
    triage_seed_order = [304, 303, 301, 302]
    tr_partner_use: Counter[str] = Counter()
    tr_pair_use: Counter[tuple[str, str]] = Counter()

    triage_main_raw: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    # Build target seed schedule (3 per seed for n=12)
    tr_main_seeds = [triage_seed_order[i % len(triage_seed_order)] for i in range(n_triage_main)]
    # Sort so all slots for tightest seed are scheduled before looser seeds
    tr_main_seeds.sort(key=lambda s: triage_seed_order.index(s))

    for target_seed in tr_main_seeds:
        picked = _pick_best_pair_from_candidates(
            triage_cands,
            used_ep_ids=used_ep_ids,
            partner_usage=tr_partner_use,
            pair_model_usage=tr_pair_use,
            target_seed=target_seed,
            low_disc_cutoff=2,
            rng=rng,
            require_same_seed=True,
        )
        if picked is None:
            picked = _pick_best_pair_from_candidates(
                triage_cands,
                used_ep_ids=used_ep_ids,
                partner_usage=tr_partner_use,
                pair_model_usage=tr_pair_use,
                target_seed=None,
                low_disc_cutoff=2,
                rng=rng,
                require_same_seed=False,
            )
        if picked is None:
            raise RuntimeError(
                f"Insufficient disjoint Triage Role-B episodes to form {n_triage_main} main pairs "
                f"(found {len(triage_main_raw)})."
            )
        c1, c2 = picked
        used_ep_ids.add(c1["episode_id"])
        used_ep_ids.add(c2["episode_id"])
        tr_partner_use[c1["partner"]] += 1
        tr_partner_use[c2["partner"]] += 1
        tr_pair_use[tuple(sorted((c1["partner"], c2["partner"])))] += 1
        triage_main_raw.append((c1["seed"], c1, c2))

    # Release IDs temporarily only to register in canonical seed order (301..304)
    for _, c1, c2 in triage_main_raw:
        used_ep_ids.remove(c1["episode_id"])
        used_ep_ids.remove(c2["episode_id"])
    triage_main_raw.sort(key=lambda x: (x[0], x[1]["episode_id"], x[2]["episode_id"]))

    main_seq = 0
    for _, c1, c2 in triage_main_raw:
        main_seq += 1
        selected_pairs.append(
            _register_pair(
                arm="main", task="triage", target_role="B", c1=c1, c2=c2, seq_idx=main_seq
            )
        )

    # Triage placebo (3 pairs across distinct seeds 301, 302, 303, 304)
    tr_plac_seeds = [301, 302, 303, 304]
    plac_partner_use: Counter[str] = Counter()
    plac_pair_use: Counter[tuple[str, str]] = Counter()
    placebo_seq = 0
    for i in range(n_triage_placebo):
        target_seed = tr_plac_seeds[i % len(tr_plac_seeds)]
        picked = _pick_best_pair_from_candidates(
            triage_cands,
            used_ep_ids=used_ep_ids,
            partner_usage=plac_partner_use,
            pair_model_usage=plac_pair_use,
            target_seed=target_seed,
            low_disc_cutoff=2,
            rng=rng,
            require_same_seed=True,
        )
        if picked is None:
            picked = _pick_best_pair_from_candidates(
                triage_cands,
                used_ep_ids=used_ep_ids,
                partner_usage=plac_partner_use,
                pair_model_usage=plac_pair_use,
                target_seed=None,
                low_disc_cutoff=2,
                rng=rng,
                require_same_seed=False,
            )
        if picked is None:
            raise RuntimeError(
                f"Insufficient disjoint Triage episodes for {n_triage_placebo} placebo pairs."
            )
        c1, c2 = picked
        plac_partner_use[c1["partner"]] += 1
        plac_partner_use[c2["partner"]] += 1
        plac_pair_use[tuple(sorted((c1["partner"], c2["partner"])))] += 1
        placebo_seq += 1
        selected_pairs.append(
            _register_pair(
                arm="placebo", task="triage", target_role="B", c1=c1, c2=c2, seq_idx=placebo_seq
            )
        )

    # --- 4. Select Goalpost R1-Handoff Main (12) + Placebo (3) ---
    # Schedule tightest seed FIRST (304 has 7 unique withholding eps: 1 Role-A + 2 Role-B main pairs),
    # then balance Roles A/B across 303, 302, 301 so total main Goalpost = exactly 6 Role A + 6 Role B!
    gp_main_plan: list[tuple[int, Role]] = [
        # Seed 304 (1A + 2B) — selected first since s304 has 7 unique withholding eps!
        (304, "A"),
        (304, "B"),
        (304, "B"),
        # Seed 303 (2A + 1B)
        (303, "B"),
        (303, "A"),
        (303, "A"),
        # Seed 302 (1A + 2B)
        (302, "B"),
        (302, "B"),
        (302, "A"),
        # Seed 301 (2A + 1B) -> Total = 6A + 6B, 3 per seed!
        (301, "B"),
        (301, "A"),
        (301, "A"),
    ]
    if n_goalpost_main <= len(gp_main_plan):
        gp_slots = gp_main_plan[:n_goalpost_main]
    else:
        gp_slots = list(gp_main_plan)
        while len(gp_slots) < n_goalpost_main:
            idx = len(gp_slots)
            gp_slots.append((EXP2_MAIN_SEEDS[idx % 4], "A" if idx % 2 == 0 else "B"))

    gp_partner_use: Counter[str] = Counter()
    gp_pair_use: Counter[tuple[str, str]] = Counter()
    gp_main_raw: list[tuple[int, Role, dict[str, Any], dict[str, Any]]] = []

    for target_seed, pref_role in gp_slots:
        alt_role: Role = "B" if pref_role == "A" else "A"
        picked: tuple[dict[str, Any], dict[str, Any]] | None = None
        actual_role: Role = pref_role
        for role_try, same_seed_req in (
            (pref_role, True),
            (alt_role, True),
            (pref_role, False),
            (alt_role, False),
        ):
            picked = _pick_best_pair_from_candidates(
                goalpost_cands_by_role[role_try],
                used_ep_ids=used_ep_ids,
                partner_usage=gp_partner_use,
                pair_model_usage=gp_pair_use,
                target_seed=target_seed if same_seed_req else None,
                low_disc_cutoff=2,
                rng=rng,
                require_same_seed=same_seed_req,
            )
            if picked is not None:
                actual_role = role_try
                break
        if picked is None:
            raise RuntimeError(
                f"Insufficient disjoint Goalpost episodes to form {n_goalpost_main} main pairs "
                f"(found {len(gp_main_raw)})."
            )
        c1, c2 = picked
        used_ep_ids.add(c1["episode_id"])
        used_ep_ids.add(c2["episode_id"])
        gp_partner_use[c1["partner"]] += 1
        gp_partner_use[c2["partner"]] += 1
        gp_pair_use[tuple(sorted((c1["partner"], c2["partner"])))] += 1
        gp_main_raw.append((c1["seed"], actual_role, c1, c2))

    for _, _, c1, c2 in gp_main_raw:
        used_ep_ids.remove(c1["episode_id"])
        used_ep_ids.remove(c2["episode_id"])
    gp_main_raw.sort(key=lambda x: (x[0], x[1], x[2]["episode_id"], x[3]["episode_id"]))

    for _, role, c1, c2 in gp_main_raw:
        main_seq += 1
        selected_pairs.append(
            _register_pair(
                arm="main", task="goalpost", target_role=role, c1=c1, c2=c2, seq_idx=main_seq
            )
        )

    # Goalpost placebo (3 pairs across distinct seeds 301, 302, 303; balanced across roles A/B)
    gp_plac_plan: list[tuple[int, Role]] = [(301, "A"), (302, "B"), (303, "A")]
    gp_plac_partner_use: Counter[str] = Counter()
    gp_plac_pair_use: Counter[tuple[str, str]] = Counter()

    for i in range(n_goalpost_placebo):
        target_seed, pref_role = gp_plac_plan[i % len(gp_plac_plan)]
        alt_role = "B" if pref_role == "A" else "A"
        picked = None
        actual_role = pref_role
        for role_try, same_seed_req in (
            (pref_role, True),
            (alt_role, True),
            (pref_role, False),
            (alt_role, False),
        ):
            picked = _pick_best_pair_from_candidates(
                goalpost_cands_by_role[role_try],
                used_ep_ids=used_ep_ids,
                partner_usage=gp_plac_partner_use,
                pair_model_usage=gp_plac_pair_use,
                target_seed=target_seed if same_seed_req else None,
                low_disc_cutoff=2,
                rng=rng,
                require_same_seed=same_seed_req,
            )
            if picked is not None:
                actual_role = role_try
                break
        if picked is None:
            raise RuntimeError(
                f"Insufficient disjoint Goalpost episodes for {n_goalpost_placebo} placebo pairs."
            )
        c1, c2 = picked
        gp_plac_partner_use[c1["partner"]] += 1
        gp_plac_partner_use[c2["partner"]] += 1
        gp_plac_pair_use[tuple(sorted((c1["partner"], c2["partner"])))] += 1
        placebo_seq += 1
        selected_pairs.append(
            _register_pair(
                arm="placebo",
                task="goalpost",
                target_role=actual_role,
                c1=c1,
                c2=c2,
                seq_idx=placebo_seq,
            )
        )

    # --- 5. Hard Global Integrity Audit across ALL selected pairs ---
    all_eids = [p.ep_1_id for p in selected_pairs] + [p.ep_2_id for p in selected_pairs]
    assert len(all_eids) == len(set(all_eids)), "Exp 2 pair selection violated strict disjointness!"
    for p in selected_pairs:
        assert abs(p.score_1 - 1.0) < 1e-9 and abs(p.score_2 - 1.0) < 1e-9
        assert p.partner_1 != p.partner_2
        if p.task == "triage":
            assert is_triage_b_zero_contradiction(p.ep1)
            assert is_triage_b_zero_contradiction(p.ep2)

    all_arms = [a for p in selected_pairs for a in (p.arms_1, p.arms_2)]
    audit = audit_exp2_pairs(selected_pairs, word_parity_tol=word_parity_tol)
    audit["selection_seed"] = selection_seed
    audit["eligible_pool_counts"] = {
        "triage_role_b_zero_contradiction_le3": len(triage_cands),
        "goalpost_role_a_lt5": len(goalpost_cands_by_role["A"]),
        "goalpost_role_b_lt5": len(goalpost_cands_by_role["B"]),
        "goalpost_unique_episodes_lt5": len(
            {c["episode_id"] for role in ("A", "B") for c in goalpost_cands_by_role[role]}
        ),
    }
    assert len(all_arms) == 2 * len(selected_pairs)
    return selected_pairs, audit


def audit_exp2_pairs(
    pairs: Sequence[InfoFlipPair],
    *,
    word_parity_tol: float = DEFAULT_WORD_PARITY_TOL,
) -> dict[str, Any]:
    """Produce a comprehensive zero-cost integrity, parity, disclosure, and zero-leakage audit."""
    all_eids = [p.ep_1_id for p in pairs] + [p.ep_2_id for p in pairs]
    all_arms = [a for p in pairs for a in (p.arms_1, p.arms_2)]

    main_pairs = [p for p in pairs if p.arm == "main"]
    placebo_pairs = [p for p in pairs if p.arm == "placebo"]
    triage_pairs = [p for p in pairs if p.task == "triage"]
    goalpost_pairs = [p for p in pairs if p.task == "goalpost"]

    added_rels = [a.added_word_rel_diff for a in all_arms]
    total_rels = [a.total_word_rel_diff for a in all_arms]
    ratios = [a.added_public_words / max(1, a.added_private_words) for a in all_arms]

    # Check every rendered prompt across all orientations & orders for vendor leaks
    prompt_leaks: list[dict[str, Any]] = []
    n_prompts_checked = 0
    for spec in build_choice_specs(pairs):
        n_prompts_checked += 1
        leaks = scan_vendor_leaks(spec["prompt"])
        if leaks:
            prompt_leaks.append({"packet_id": spec["packet_id"], "leaks": leaks})

    return {
        "n_pairs_total": len(pairs),
        "n_pairs_main": len(main_pairs),
        "n_pairs_placebo": len(placebo_pairs),
        "n_pairs_triage_main": sum(1 for p in main_pairs if p.task == "triage"),
        "n_pairs_triage_placebo": sum(1 for p in placebo_pairs if p.task == "triage"),
        "n_pairs_goalpost_main": sum(1 for p in main_pairs if p.task == "goalpost"),
        "n_pairs_goalpost_placebo": sum(1 for p in placebo_pairs if p.task == "goalpost"),
        "n_candidate_episodes": len(all_eids),
        "n_unique_episodes": len(set(all_eids)),
        "strict_disjoint_ok": len(all_eids) == len(set(all_eids)),
        "all_score_1_ok": all(abs(p.score_1 - 1.0) < 1e-9 and abs(p.score_2 - 1.0) < 1e-9 for p in pairs),
        "all_same_seed_ok": all(p.same_seed for p in pairs),
        "all_distinct_partners_ok": all(p.partner_1 != p.partner_2 for p in pairs),
        "triage_zero_contradiction_count": sum(
            int(is_triage_b_zero_contradiction(p.ep1)) + int(is_triage_b_zero_contradiction(p.ep2))
            for p in triage_pairs
        ),
        "triage_zero_contradiction_total": 2 * len(triage_pairs),
        "word_parity_tol": word_parity_tol,
        "word_parity_pass_count": sum(1 for a in all_arms if a.word_parity_ok),
        "word_parity_total": len(all_arms),
        "max_added_word_rel_diff": round(max(added_rels), 6) if added_rels else 0.0,
        "mean_added_word_rel_diff": round(float(np.mean(added_rels)), 6) if added_rels else 0.0,
        "max_total_word_rel_diff": round(max(total_rels), 6) if total_rels else 0.0,
        "mean_added_pub_to_priv_ratio": round(float(np.mean(ratios)), 6) if ratios else 1.0,
        "public_echo_zero_leak_count": sum(1 for a in all_arms if a.public_echo_zero_leak),
        "public_echo_zero_leak_total": len(all_arms),
        "private_info_complete_count": sum(1 for a in all_arms if a.private_info_complete),
        "private_info_complete_total": len(all_arms),
        "n_prompts_checked": n_prompts_checked,
        "prompt_vendor_leak_count": len(prompt_leaks),
        "prompt_vendor_leaks": prompt_leaks,
        "by_arm_and_task": dict(Counter(f"{p.arm}:{p.task}" for p in pairs)),
        "by_task_and_seed": {
            task: dict(
                sorted(
                    Counter(p.seed_1 for p in pairs if p.task == task and p.arm == "main").items()
                )
            )
            for task in ("triage", "goalpost")
        },
        "by_task_and_role_main": {
            task: dict(
                sorted(
                    Counter(
                        p.target_role for p in main_pairs if p.task == task
                    ).items()
                )
            )
            for task in ("triage", "goalpost")
        },
        "by_task_and_partner_main": {
            task: dict(
                sorted(
                    Counter(
                        m
                        for p in main_pairs
                        if p.task == task
                        for m in (p.partner_1, p.partner_2)
                    ).items()
                )
            )
            for task in ("triage", "goalpost")
        },
        "by_partner_all": dict(
            sorted(Counter(m for p in pairs for m in (p.partner_1, p.partner_2)).items())
        ),
        "n_triage_pairs": len(triage_pairs),
        "n_goalpost_pairs": len(goalpost_pairs),
    }


# ---------------------------------------------------------------------------
# Choice Packet Construction & Reviewer Manifest Loading
# ---------------------------------------------------------------------------


def build_choice_specs(
    pairs: Sequence[InfoFlipPair],
    *,
    reviewers: Sequence[str] = EXP2_REVIEWERS,
) -> list[dict[str, Any]]:
    """Expand `pairs` into exact symmetric crossover choice specs (2 orientations × 2 orders × reviewers)."""
    specs: list[dict[str, Any]] = []
    for p in pairs:
        if p.arm == "main":
            orientations: tuple[tuple[str, VariantName, VariantName], ...] = (
                ("p1_info_p2_echo", "private_info", "public_echo"),
                ("p1_echo_p2_info", "public_echo", "private_info"),
            )
        else:
            orientations = (
                ("p1_echo_p2_unedited", "public_echo", "unedited"),
                ("p1_unedited_p2_echo", "unedited", "public_echo"),
            )

        for cond_name, v_p1, v_p2 in orientations:
            txt_p1, turns_p1 = format_exp2_transcript(
                p.ep1,
                task=p.task,
                target_role=p.target_role,
                candidate_handle=p.handle_p1,
                teammate_handle=p.mate_handle_p1,
                r1_candidate_message=p.arms_1.message_for(v_p1),
            )
            txt_p2, turns_p2 = format_exp2_transcript(
                p.ep2,
                task=p.task,
                target_role=p.target_role,
                candidate_handle=p.handle_p2,
                teammate_handle=p.mate_handle_p2,
                r1_candidate_message=p.arms_2.message_for(v_p2),
            )
            slot_p1 = (
                "P1",
                p.handle_p1,
                p.partner_1,
                v_p1,
                p.score_1,
                p.outcome_card_1,
                txt_p1,
                turns_p1,
                p.ep_1_id,
            )
            slot_p2 = (
                "P2",
                p.handle_p2,
                p.partner_2,
                v_p2,
                p.score_2,
                p.outcome_card_2,
                txt_p2,
                turns_p2,
                p.ep_2_id,
            )
            for order in ("AB", "BA"):
                s1, s2 = (slot_p1, slot_p2) if order == "AB" else (slot_p2, slot_p1)
                packet_id = f"{p.pair_id}__{cond_name}__{order}"
                if p.task == "triage":
                    prompt = capstone_third_person_choice_prompt(
                        task="triage",
                        handle_1=s1[1],
                        handle_2=s2[1],
                        transcript_1=s1[6],
                        transcript_2=s2[6],
                        outcome_card_1=s1[5],
                        outcome_card_2=s2[5],
                    )
                else:
                    prompt = exp2_goalpost_r1_choice_prompt(
                        handle_1=s1[1],
                        handle_2=s2[1],
                        transcript_1=s1[6],
                        transcript_2=s2[6],
                        outcome_card_1=s1[5],
                        outcome_card_2=s2[5],
                    )
                for reviewer in reviewers:
                    specs.append(
                        {
                            "packet_id": packet_id,
                            "pair_id": p.pair_id,
                            "arm": p.arm,
                            "task": p.task,
                            "target_role": p.target_role,
                            "seed_1": p.seed_1,
                            "seed_2": p.seed_2,
                            "condition": cond_name,
                            "order": order,
                            "reviewer": reviewer,
                            "slot1": s1,
                            "slot2": s2,
                            "prompt": prompt,
                        }
                    )
    return specs


def load_exp2_reviewers(
    *,
    probe_manifest: Path | None = None,
    concurrency: int = DEFAULT_EXP2_CONCURRENCY,
    mock: bool = False,
) -> dict[str, ModelSpec]:
    """Load the 4 core reviewer pins (`opus55`, `neon4`, `astra62`, `kimi31`) with bumped concurrency.

    Supports both `probe_manifest_v2.json` (contains all 4) and `probe_manifest_v2_1.json` (merges
    `kimi31` cleanly from `probe_manifest_v2.json`). Sets safe `max_tokens` (`4096` for `astra62`,
    `8192` for `neon4`/`opus55`/`kimi31`) and preserves `supports_anthropic_cache_control=True` on `opus55`.
    """
    base_v2 = load_verified_models(
        probe_path=PROBE_MANIFEST_V2_PATH,
        version="v2",
        strict=(not mock) and (probe_manifest is None),
    )
    if probe_manifest is not None and probe_manifest.resolve() != PROBE_MANIFEST_V2_PATH.resolve():
        override_v2 = load_verified_models(
            probe_path=probe_manifest,
            version="v2",
            strict=not mock,
        )
        # Apply pins present in the explicit manifest (e.g. probe_manifest_v2_1.json)
        raw_manifest = (
            json.loads(probe_manifest.read_text(encoding="utf-8"))
            if probe_manifest.exists()
            else {}
        )
        present_keys = set((raw_manifest.get("verified_pins") or {}).keys())
        for k in EXP2_REVIEWERS:
            if k in present_keys and k in override_v2:
                base_v2[k] = override_v2[k]

    reviewers: dict[str, ModelSpec] = {}
    for k in EXP2_REVIEWERS:
        spec = base_v2[k]
        reviewers[k] = ModelSpec(
            key=spec.key,
            slug=spec.slug,
            family=spec.family,
            display_name=spec.display_name,
            role=spec.role,
            provider_only=spec.provider_only,
            reasoning_effort="medium",
            max_tokens=DEFAULT_MAX_TOKENS_BY_REVIEWER.get(k, 8192),
            supports_anthropic_cache_control=(k == "opus55" or spec.supports_anthropic_cache_control),
            fallback_input_per_m=spec.fallback_input_per_m,
            fallback_output_per_m=spec.fallback_output_per_m,
            concurrency=concurrency,
            aa_index_v5=spec.aa_index_v5,
            verified_returned_model=spec.verified_returned_model,
            verified_provider=spec.verified_provider,
            notes=spec.notes,
        )
    return reviewers


# ---------------------------------------------------------------------------
# Deterministic Zero-Cost Mock Client for Exp 2 Offline Verification
# ---------------------------------------------------------------------------


class MockInfoFlipClient(MockOpenRouterClient):
    """Deterministic offline mock client exercising full Exp 2 main & placebo crossovers at $0."""

    async def complete(
        self,
        *,
        call_id: str,
        model_key: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        reasoning_effort_override: str | None = None,
        reasoning_payload_override: dict[str, Any] | None = None,
        max_tokens_override: int | None = None,
        cache_prefix_index: int | None = None,
        require_parameters: bool = True,
    ) -> LedgerEntry:
        cached = self.ledger.get_completed(call_id)
        if cached is not None:
            return cached

        spec = self.models[model_key]
        meta = dict(metadata or {})
        rng = random.Random(int(hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:12], 16))

        p_tok = sum(len(str(m.get("content", ""))) // 4 for m in messages) + 20
        mock_worst = (
            p_tok * spec.fallback_input_per_m + 400 * spec.fallback_output_per_m
        ) / 1_000_000.0
        res_id = f"mock:{call_id}:{uuid.uuid4().hex[:8]}"
        await self.ledger.reserve_budget(res_id, max(0.0001, mock_worst))
        try:
            response_text = self._mock_exp2_choice(model_key, meta, rng)
            c_tok = max(20, len(response_text) // 4)
            r_tok = rng.randint(120, 380)
            cost = (
                p_tok * spec.fallback_input_per_m + (c_tok + r_tok) * spec.fallback_output_per_m
            ) / 1_000_000.0
            entry = LedgerEntry(
                call_id=call_id,
                timestamp_utc=utc_now_iso(),
                model_key=model_key,
                requested_slug=spec.slug,
                returned_model=f"{spec.slug}-mock",
                provider=spec.provider_only[0] if spec.provider_only else "Mock",
                reasoning_effort=reasoning_effort_override or spec.reasoning_effort,
                max_tokens=max_tokens_override or spec.max_tokens,
                status="ok",
                finish_reason="stop",
                attempt=1,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                reasoning_tokens=r_tok,
                cached_tokens=int(p_tok * 0.25) if cache_prefix_index is not None else 0,
                cost_usd=cost,
                latency_ms=rng.randint(8, 25),
                request_messages=messages,
                response_text=response_text,
                reasoning_text="Mock Exp 2 staffing comparison.",
                metadata=meta,
            )
            await self.ledger.record(entry, reservation_id=res_id)
            return entry
        finally:
            await self.ledger.release_reservation(res_id)

    def _mock_exp2_choice(
        self, model_key: str, meta: dict[str, Any], rng: random.Random
    ) -> str:
        h1 = str(meta["handle_1"])
        h2 = str(meta["handle_2"])
        v1 = str(meta.get("variant_1") or "")
        v2 = str(meta.get("variant_2") or "")
        arm = str(meta.get("arm") or "main")
        pair_id = str(meta.get("pair_id") or "")

        # Deterministic baseline preference for P1 vs P2 on this pair (invariant across order)
        pair_rng = _deterministic_rng("mock_pair_base", pair_id, model_key)
        base_pref_p1 = 1.0 if pair_rng.random() < 0.5 else -1.0

        # Canonical slot mapping
        c1 = str(meta.get("canonical_1") or "P1")
        s1 = base_pref_p1 if c1 == "P1" else -base_pref_p1
        s2 = -s1

        if arm == "main":
            # Strong realistic signal favoring private_info over public_echo
            if v1 == "private_info" and v2 == "public_echo":
                s1 += 3.2
            elif v1 == "public_echo" and v2 == "private_info":
                s2 += 3.2
        else:
            # Neutral placebo (public_echo vs unedited ~ 0 shift; pair preference holds)
            if v1 == "public_echo" and v2 == "unedited":
                s1 += 0.05
            elif v1 == "unedited" and v2 == "public_echo":
                s2 += 0.05

        p1 = 1.0 / (1.0 + pow(2.718281828, -(s1 - s2)))
        pick = h1 if rng.random() < p1 else h2
        reason = (
            f"{pick} proactively shared the key constraints/options up front, enabling reliable coordination."
            if arm == "main"
            else f"{pick} demonstrated clearer coordination across the task handoff."
        )
        return json.dumps({"choice": pick, "reason": reason})


# ---------------------------------------------------------------------------
# Statistical Inference: Pair-Cluster Bootstrap, Exact Sign-Flip Perm, Capstone Comparison
# ---------------------------------------------------------------------------


def _exact_case_weighted_sign_flip_p(pair_sums: Sequence[float]) -> tuple[float, str, int]:
    """Exact two-sided sign-flip permutation p-value on case-weighted pair sums.

    Since every case delta is a half-integer in `{-1, -0.5, 0, +0.5, +1}`, scaling each
    pair sum by 2 yields exact integers, allowing exact enumeration over all `2^k`
    sign assignments in O(k * sum(|2*S_p|)) dynamic programming time with zero
    Monte Carlo error.
    """
    n_pairs = len(pair_sums)
    if n_pairs == 0:
        return 1.0, "exact_2^0", 1
    scaled = [round(2.0 * s) for s in pair_sums]
    if all(abs(2.0 * s - w) < 1e-9 for s, w in zip(pair_sums, scaled, strict=True)):
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
        obs = abs(sum(scaled))
        total_ways = 1 << n_pairs
        extreme_ways = sum(c for tot, c in counts.items() if abs(tot) >= obs)
        return extreme_ways / total_ways, f"exact_2^{n_pairs}", total_ways

    # Fallback for arbitrary non-half-integer weights (exact up to 14 pairs)
    arr = np.asarray(pair_sums, dtype=float)
    obs_f = abs(float(np.sum(arr)))
    n_enum = 1 << n_pairs
    bits = (np.arange(n_enum, dtype=np.int64)[:, None] >> np.arange(n_pairs, dtype=np.int64)) & 1
    signs = np.where(bits == 1, 1.0, -1.0)
    perm_sums = np.abs((signs * arr[None, :]).sum(axis=1))
    return float(np.mean(perm_sums >= obs_f - 1e-12)), f"exact_2^{n_pairs}", n_enum


def _pair_cluster_bootstrap_and_perm(
    case_rows: Sequence[dict[str, Any]],
    *,
    delta_key: str,
    toward_outcome: str,
    away_outcome: str,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm_mc: int = DEFAULT_N_PERM_MC,
    seed: int = DEFAULT_SELECTION_SEED,
) -> dict[str, Any]:
    """Compute case-level crossover metrics with 2,000-draw pair-cluster bootstrap 95% CIs
    and exact case-weighted pair-level sign-flip permutation p-values (`2^k` exact).
    """
    if not case_rows:
        return {"n_cases": 0, "n_pairs": 0}

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_pair[str(row["pair_id"])].append(row)
    pair_ids = sorted(by_pair.keys())
    n_pairs = len(pair_ids)

    deltas = [float(r[delta_key]) for r in case_rows]
    toward = [1.0 if r["outcome"] == toward_outcome else 0.0 for r in case_rows]
    away = [1.0 if r["outcome"] == away_outcome else 0.0 for r in case_rows]
    held = [1.0 if r["outcome"] == "held_consistent" else 0.0 for r in case_rows]
    split = [1.0 if r["outcome"] == "order_split" else 0.0 for r in case_rows]
    order_agree = [float(r["order_agree_rate"]) for r in case_rows]

    rng = np.random.default_rng(seed)
    pair_arrays_delta = [
        np.array([float(r[delta_key]) for r in by_pair[pid]], dtype=float) for pid in pair_ids
    ]
    pair_arrays_toward = [
        np.array([1.0 if r["outcome"] == toward_outcome else 0.0 for r in by_pair[pid]], dtype=float)
        for pid in pair_ids
    ]
    pair_arrays_away = [
        np.array([1.0 if r["outcome"] == away_outcome else 0.0 for r in by_pair[pid]], dtype=float)
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

    # Exact case-weighted pair-level sign-flip permutation p-value (two-sided):
    # flipping all reviewer deltas in a transcript pair together flips each pair's sum,
    # directly testing the exact case-weighted mean delta reported above.
    pair_sums = [float(np.sum(a)) for a in pair_arrays_delta]
    perm_p, perm_method, n_perm_used = _exact_case_weighted_sign_flip_p(pair_sums)

    return {
        "n_cases": len(case_rows),
        "n_pairs": n_pairs,
        f"mean_{delta_key}": round(float(np.mean(deltas)), 4),
        f"{delta_key}_pair_boot_ci95": [
            round(float(np.percentile(boot_delta, 2.5)), 4),
            round(float(np.percentile(boot_delta, 97.5)), 4),
        ],
        "pair_sign_flip_perm_p": round(perm_p, 6),
        "pair_sign_flip_method": perm_method,
        "pair_sign_flip_n_perms": n_perm_used,
        f"n_{toward_outcome}": int(sum(toward)),
        f"{toward_outcome}_rate": round(float(np.mean(toward)), 4),
        f"{toward_outcome}_ci95": [
            round(float(np.percentile(boot_toward, 2.5)), 4),
            round(float(np.percentile(boot_toward, 97.5)), 4),
        ],
        f"n_{away_outcome}": int(sum(away)),
        f"{away_outcome}_rate": round(float(np.mean(away)), 4),
        f"{away_outcome}_ci95": [
            round(float(np.percentile(boot_away, 2.5)), 4),
            round(float(np.percentile(boot_away, 97.5)), 4),
        ],
        "n_held_consistent": int(sum(held)),
        "held_consistent_rate": round(float(np.mean(held)), 4),
        "n_order_split": int(sum(split)),
        "order_split_rate": round(float(np.mean(split)), 4),
        "mean_ab_ba_order_agree_rate": round(float(np.mean(order_agree)), 4),
        "n_boot": n_boot,
    }


def _collapse_arm_cases(
    choice_rows: Sequence[dict[str, Any]],
    *,
    arm: ArmType,
    cross_vendor_only: bool,
) -> list[dict[str, Any]]:
    """Collapse 4 calls per `(pair_id, reviewer)` into exact crossover cases for `arm`."""
    if arm == "main":
        c_pos, c_neg = "p1_info_p2_echo", "p1_echo_p2_info"
        delta_key = "delta_info"
        toward_lbl, away_lbl = "switched_toward_info", "switched_away_from_info"
    else:
        c_pos, c_neg = "p1_echo_p2_unedited", "p1_unedited_p2_echo"
        delta_key = "delta_echo"
        toward_lbl, away_lbl = "switched_toward_echo", "switched_away_from_echo"

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in choice_rows:
        if str(row.get("arm")) != arm:
            continue
        if row.get("chosen_slot") not in (1, 2):
            continue
        if cross_vendor_only and not row.get("cross_vendor"):
            continue
        grouped[(str(row["pair_id"]), str(row["reviewer"]))].append(row)

    cases: list[dict[str, Any]] = []
    for (pid, reviewer), rows in sorted(grouped.items()):
        r_pos = [r for r in rows if r.get("condition") == c_pos]
        r_neg = [r for r in rows if r.get("condition") == c_neg]
        if len(r_pos) != 2 or len(r_neg) != 2:
            continue

        p1_pref_pos = float(np.mean([1.0 if r.get("chosen_canonical") == "P1" else 0.0 for r in r_pos]))
        p1_pref_neg = float(np.mean([1.0 if r.get("chosen_canonical") == "P1" else 0.0 for r in r_neg]))
        delta = p1_pref_pos - p1_pref_neg

        agree_pos = 1.0 if p1_pref_pos in (0.0, 1.0) else 0.0
        agree_neg = 1.0 if p1_pref_neg in (0.0, 1.0) else 0.0
        order_agree_rate = 0.5 * (agree_pos + agree_neg)

        if p1_pref_pos == 1.0 and p1_pref_neg == 0.0:
            outcome = toward_lbl
        elif p1_pref_pos == 0.0 and p1_pref_neg == 1.0:
            outcome = away_lbl
        elif p1_pref_pos == p1_pref_neg and p1_pref_pos in (0.0, 1.0):
            outcome = "held_consistent"
        else:
            outcome = "order_split"

        sample = r_pos[0]
        cases.append(
            {
                "pair_id": pid,
                "arm": arm,
                "task": str(sample["task"]),
                "target_role": str(sample["target_role"]),
                "seed_1": int(sample["seed_1"]),
                "reviewer": reviewer,
                "cross_vendor": bool(sample["cross_vendor"]),
                "p1_pref_pos": p1_pref_pos,
                "p1_pref_neg": p1_pref_neg,
                delta_key: delta,
                "outcome": outcome,
                "order_agree_rate": order_agree_rate,
            }
        )
    return cases


def _summarize_arm_slice(
    cases: Sequence[dict[str, Any]],
    *,
    arm: ArmType,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm_mc: int = DEFAULT_N_PERM_MC,
    seed: int = DEFAULT_SELECTION_SEED,
) -> dict[str, Any]:
    delta_key = "delta_info" if arm == "main" else "delta_echo"
    toward_lbl = "switched_toward_info" if arm == "main" else "switched_toward_echo"
    away_lbl = "switched_away_from_info" if arm == "main" else "switched_away_from_echo"

    def _agg(sub: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return _pair_cluster_bootstrap_and_perm(
            sub,
            delta_key=delta_key,
            toward_outcome=toward_lbl,
            away_outcome=away_lbl,
            n_boot=n_boot,
            n_perm_mc=n_perm_mc,
            seed=seed,
        )

    return {
        "overall": _agg(cases),
        "by_task": {
            t: _agg([c for c in cases if c["task"] == t])
            for t in ("triage", "goalpost")
            if any(c["task"] == t for c in cases)
        },
        "by_reviewer": {
            rv: _agg([c for c in cases if c["reviewer"] == rv])
            for rv in EXP2_REVIEWERS
            if any(c["reviewer"] == rv for c in cases)
        },
        "by_seed": {
            str(s): _agg([c for c in cases if c["seed_1"] == s])
            for s in sorted({c["seed_1"] for c in cases})
        },
    }


def _summarize_4lab_consensus(all_main_cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure 4-lab third-person consensus across reviewers on the 24 main pairs."""
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in all_main_cases:
        by_pair[str(c["pair_id"])].append(c)

    pair_rows: list[dict[str, Any]] = []
    pairwise_agree_vals: list[float] = []
    for pid in sorted(by_pair.keys()):
        rows = by_pair[pid]
        n_rv = len(rows)
        n_toward = sum(1 for r in rows if r["outcome"] == "switched_toward_info")
        n_pos = sum(1 for r in rows if float(r["delta_info"]) > 0.0)
        n_away = sum(1 for r in rows if r["outcome"] == "switched_away_from_info")
        n_held = sum(1 for r in rows if r["outcome"] == "held_consistent")
        n_split = sum(1 for r in rows if r["outcome"] == "order_split")
        mean_d = float(np.mean([float(r["delta_info"]) for r in rows])) if rows else 0.0

        # Pairwise agreement on binary "switched_toward_info" across all pairs of reviewers
        if n_rv >= 2:
            flags = [1 if r["outcome"] == "switched_toward_info" else 0 for r in rows]
            agrees = [
                1.0 if flags[i] == flags[j] else 0.0
                for i in range(n_rv)
                for j in range(i + 1, n_rv)
            ]
            pairwise_agree_vals.append(float(np.mean(agrees)))

        pair_rows.append(
            {
                "pair_id": pid,
                "task": rows[0]["task"],
                "n_reviewers": n_rv,
                "n_switched_toward_info": n_toward,
                "n_positive_delta": n_pos,
                "n_switched_away_from_info": n_away,
                "n_held_consistent": n_held,
                "n_order_split": n_split,
                "mean_delta_info": round(mean_d, 4),
                "unanimous_4of4_toward_info": n_rv == 4 and n_toward == 4,
                "majority_ge3of4_toward_info": n_rv == 4 and n_toward >= 3,
                "majority_ge3of4_positive_delta": n_rv == 4 and n_pos >= 3,
            }
        )

    n_pairs = len(pair_rows)
    return {
        "n_pairs": n_pairs,
        "n_unanimous_4of4_toward_info": sum(1 for r in pair_rows if r["unanimous_4of4_toward_info"]),
        "n_majority_ge3of4_toward_info": sum(1 for r in pair_rows if r["majority_ge3of4_toward_info"]),
        "n_majority_ge3of4_positive_delta": sum(
            1 for r in pair_rows if r["majority_ge3of4_positive_delta"]
        ),
        "mean_reviewers_toward_info_per_pair": (
            round(float(np.mean([r["n_switched_toward_info"] for r in pair_rows])), 4)
            if n_pairs
            else 0.0
        ),
        "mean_pairwise_lab_agreement_on_flip": (
            round(float(np.mean(pairwise_agree_vals)), 4) if pairwise_agree_vals else None
        ),
        "per_pair": pair_rows,
    }


def load_capstone_comparison(
    capstone_dir: Path = DEFAULT_CAPSTONE_DIR,
) -> dict[str, Any] | None:
    """Directly import Capstone Tone+Readability Flip results (JSONL or Parquet fallback)."""
    choices_p = Path(capstone_dir) / "capstone_choices.jsonl"
    pairs_p = Path(capstone_dir) / "capstone_pairs.json"
    if choices_p.exists() and pairs_p.exists():
        choice_rows = [
            json.loads(line)
            for line in choices_p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        from common_ground.parquet_loader import load_capstone_from_parquet, resolve_hf_export_dir

        hf_dir = resolve_hf_export_dir(Path(capstone_dir))
        if hf_dir is None or not (hf_dir / "capstone_tone_flip.parquet").is_file():
            return None
        _, choice_rows = load_capstone_from_parquet(hf_dir)
        if not choice_rows:
            return None

    def _collapse_capstone(cross_vendor_only: bool) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in choice_rows:
            if row.get("chosen_slot") not in (1, 2):
                continue
            if cross_vendor_only and not row.get("cross_vendor"):
                continue
            grouped[(str(row["pair_id"]), str(row["reviewer"]))].append(row)

        cases: list[dict[str, Any]] = []
        for (pid, reviewer), rows in sorted(grouped.items()):
            c_w = [r for r in rows if r.get("condition") == "p1_warm_p2_blunt"]
            c_b = [r for r in rows if r.get("condition") == "p1_blunt_p2_warm"]
            if len(c_w) != 2 or len(c_b) != 2:
                continue
            p1_w = float(np.mean([1.0 if r.get("chosen_canonical") == "P1" else 0.0 for r in c_w]))
            p1_b = float(np.mean([1.0 if r.get("chosen_canonical") == "P1" else 0.0 for r in c_b]))
            delta_warm = p1_w - p1_b
            agree = 0.5 * (float(p1_w in (0.0, 1.0)) + float(p1_b in (0.0, 1.0)))
            if p1_w == 1.0 and p1_b == 0.0:
                outcome = "switched_toward_warm"
            elif p1_w == 0.0 and p1_b == 1.0:
                outcome = "switched_away_from_warm"
            elif p1_w == p1_b and p1_w in (0.0, 1.0):
                outcome = "held_consistent"
            else:
                outcome = "order_split"
            cases.append(
                {
                    "pair_id": pid,
                    "reviewer": reviewer,
                    "delta_warm": delta_warm,
                    "outcome": outcome,
                    "order_agree_rate": agree,
                }
            )
        # Match Capstone's published seed (20261122) & n_perm (5000) for exact reproducibility
        return _pair_cluster_bootstrap_and_perm(
            cases,
            delta_key="delta_warm",
            toward_outcome="switched_toward_warm",
            away_outcome="switched_away_from_warm",
            n_boot=2000,
            n_perm_mc=5000,
            seed=20261122,
        )

    try:
        from common_ground.config import ROOT_DIR

        rel_capstone_dir = Path(capstone_dir).resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        rel_capstone_dir = Path(capstone_dir).name

    return {
        "source_dir": rel_capstone_dir,
        "n_choices_loaded": len(choice_rows),
        "itt_cross_vendor": _collapse_capstone(cross_vendor_only=True),
        "itt_all_reviewers": _collapse_capstone(cross_vendor_only=False),
    }


def summarize_exp2_results(
    choice_rows: Sequence[dict[str, Any]],
    *,
    capstone_dir: Path | None = DEFAULT_CAPSTONE_DIR,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm_mc: int = DEFAULT_N_PERM_MC,
    seed: int = DEFAULT_SELECTION_SEED,
) -> dict[str, Any]:
    """Compute complete Experiment 2 statistical summary across Main, Placebo, and Capstone benchmark."""
    main_cv_cases = _collapse_arm_cases(choice_rows, arm="main", cross_vendor_only=True)
    main_all_cases = _collapse_arm_cases(choice_rows, arm="main", cross_vendor_only=False)
    plac_cv_cases = _collapse_arm_cases(choice_rows, arm="placebo", cross_vendor_only=True)
    plac_all_cases = _collapse_arm_cases(choice_rows, arm="placebo", cross_vendor_only=False)

    main_rows = [r for r in choice_rows if r.get("arm") == "main"]
    plac_rows = [r for r in choice_rows if r.get("arm") == "placebo"]
    parsed_main = sum(1 for r in main_rows if r.get("chosen_slot") in (1, 2))
    parsed_plac = sum(1 for r in plac_rows if r.get("chosen_slot") in (1, 2))

    capstone_cmp = load_capstone_comparison(capstone_dir) if capstone_dir is not None else None

    return {
        "schema_version": "exp2_info_flip_summary_v1",
        "timestamp_utc": utc_now_iso(),
        "counts": {
            "n_choices_total": len(choice_rows),
            "n_choices_main": len(main_rows),
            "n_choices_main_parsed": parsed_main,
            "n_choices_placebo": len(plac_rows),
            "n_choices_placebo_parsed": parsed_plac,
            "total_choice_cost_usd": round(
                sum(float(r.get("cost_usd") or 0.0) for r in choice_rows), 6
            ),
        },
        "main_arm": {
            "itt_cross_vendor": _summarize_arm_slice(
                main_cv_cases, arm="main", n_boot=n_boot, n_perm_mc=n_perm_mc, seed=seed
            ),
            "itt_all_reviewers": _summarize_arm_slice(
                main_all_cases, arm="main", n_boot=n_boot, n_perm_mc=n_perm_mc, seed=seed
            ),
            "leave_neon_out": {
                "itt_cross_vendor": _pair_cluster_bootstrap_and_perm(
                    [c for c in main_cv_cases if c["reviewer"] != "neon4"],
                    delta_key="delta_info",
                    toward_outcome="switched_toward_info",
                    away_outcome="switched_away_from_info",
                    n_boot=n_boot,
                    n_perm_mc=n_perm_mc,
                    seed=seed,
                ),
                "itt_all_reviewers": _pair_cluster_bootstrap_and_perm(
                    [c for c in main_all_cases if c["reviewer"] != "neon4"],
                    delta_key="delta_info",
                    toward_outcome="switched_toward_info",
                    away_outcome="switched_away_from_info",
                    n_boot=n_boot,
                    n_perm_mc=n_perm_mc,
                    seed=seed,
                ),
            },
            "consensus_4lab": _summarize_4lab_consensus(main_all_cases),
        },
        "placebo_arm": {
            "itt_cross_vendor": _summarize_arm_slice(
                plac_cv_cases, arm="placebo", n_boot=n_boot, n_perm_mc=n_perm_mc, seed=seed
            ),
            "itt_all_reviewers": _summarize_arm_slice(
                plac_all_cases, arm="placebo", n_boot=n_boot, n_perm_mc=n_perm_mc, seed=seed
            ),
        },
        "capstone_comparison": capstone_cmp,
    }


def render_exp2_summary_markdown(summary: dict[str, Any]) -> str:
    """Render clean, publication-ready Markdown report with side-by-side Capstone comparison."""
    lines: list[str] = [
        "# Experiment 2: The Information Flip — Summary Report",
        "",
        f"- **Schema:** `{summary.get('schema_version')}` | **Mock:** `{summary.get('mock', False)}`",
        (
            f"- **Choices:** `{summary['counts']['n_choices_main_parsed']}/{summary['counts']['n_choices_main']}` main + "
            f"`{summary['counts']['n_choices_placebo_parsed']}/{summary['counts']['n_choices_placebo']}` placebo "
            f"(total cost: `${summary.get('ledger_total_cost_usd', summary['counts']['total_choice_cost_usd']):.4f}`)"
        ),
        "",
        "## Side-by-Side Causal Comparison: Tone+Readability vs. Information Flip",
        "",
        "| Experiment / Arm | Scope | Cases (Pairs) | Mean Δ | Pair-Cluster 95% CI | Sign-Flip p | Unanimous Flips Toward | Unanimous Flips Away | Held | Split |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    def _fmt_row(label: str, scope: str, m: dict[str, Any], delta_k: str, tw_k: str, aw_k: str) -> str:
        if not m or m.get("n_cases", 0) == 0:
            return f"| {label} | {scope} | 0 (0) | — | — | — | — | — | — | — |"
        n_c = int(m["n_cases"])
        n_p = int(m["n_pairs"])
        md = m.get(f"mean_{delta_k}")
        ci = m.get(f"{delta_k}_pair_boot_ci95") or [None, None]
        pval = m.get("pair_sign_flip_perm_p")
        tw_n = int(m.get(f"n_{tw_k}", 0))
        tw_r = float(m.get(f"{tw_k}_rate", 0.0))
        aw_n = int(m.get(f"n_{aw_k}", 0))
        aw_r = float(m.get(f"{aw_k}_rate", 0.0))
        hd_r = float(m.get("held_consistent_rate", 0.0))
        sp_r = float(m.get("order_split_rate", 0.0))
        md_s = f"{md:+.4f}" if md is not None else "—"
        ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci[0] is not None and ci[1] is not None else "—"
        p_s = f"{pval:.4f}" if pval is not None else "—"
        return (
            f"| {label} | {scope} | {n_c} ({n_p}) | {md_s} | {ci_s} | {p_s} | "
            f"{tw_n}/{n_c} ({tw_r * 100:.1f}%) | {aw_n}/{n_c} ({aw_r * 100:.1f}%) | "
            f"{hd_r * 100:.1f}% | {sp_r * 100:.1f}% |"
        )

    cap = summary.get("capstone_comparison")
    if cap:
        lines.append(
            _fmt_row(
                "Capstone Tone+Readability Flip",
                "ITT Cross-Vendor",
                cap["itt_cross_vendor"],
                "delta_warm",
                "switched_toward_warm",
                "switched_away_from_warm",
            )
        )
        lines.append(
            _fmt_row(
                "Capstone Tone+Readability Flip",
                "ITT Pooled (4 labs)",
                cap["itt_all_reviewers"],
                "delta_warm",
                "switched_toward_warm",
                "switched_away_from_warm",
            )
        )

    m_cv = summary["main_arm"]["itt_cross_vendor"]
    m_all = summary["main_arm"]["itt_all_reviewers"]
    p_cv = summary["placebo_arm"]["itt_cross_vendor"]
    p_all = summary["placebo_arm"]["itt_all_reviewers"]

    lines.append(
        _fmt_row(
            "**Exp 2 Information Flip (Main)**",
            "**ITT Cross-Vendor**",
            m_cv["overall"],
            "delta_info",
            "switched_toward_info",
            "switched_away_from_info",
        )
    )
    lines.append(
        _fmt_row(
            "**Exp 2 Information Flip (Main)**",
            "**ITT Pooled (4 labs)**",
            m_all["overall"],
            "delta_info",
            "switched_toward_info",
            "switched_away_from_info",
        )
    )
    for task_key, task_lbl in (
        ("triage", "Exp 2 Triage Full-Transcript"),
        ("goalpost", "Exp 2 Goalpost R1-Handoff"),
    ):
        if task_key in m_cv["by_task"]:
            lines.append(
                _fmt_row(
                    task_lbl,
                    "ITT Cross-Vendor",
                    m_cv["by_task"][task_key],
                    "delta_info",
                    "switched_toward_info",
                    "switched_away_from_info",
                )
            )
        if task_key in m_all["by_task"]:
            lines.append(
                _fmt_row(
                    task_lbl,
                    "ITT Pooled (4 labs)",
                    m_all["by_task"][task_key],
                    "delta_info",
                    "switched_toward_info",
                    "switched_away_from_info",
                )
            )
    lines.append(
        _fmt_row(
            "Exp 2 Public-Echo Placebo",
            "ITT Cross-Vendor",
            p_cv["overall"],
            "delta_echo",
            "switched_toward_echo",
            "switched_away_from_echo",
        )
    )
    lines.append(
        _fmt_row(
            "Exp 2 Public-Echo Placebo",
            "ITT Pooled (4 labs)",
            p_all["overall"],
            "delta_echo",
            "switched_toward_echo",
            "switched_away_from_echo",
        )
    )

    lines.extend(["", "## Per-Reviewer Breakdown (Main Arm — ITT Pooled)", ""])
    lines.append(
        "| Reviewer | Cases (Pairs) | Mean ΔInfo | 95% CI | Sign-Flip p | Unanimous Toward Info | Away | Held | Split |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for rv in EXP2_REVIEWERS:
        rm = m_all["by_reviewer"].get(rv)
        if rm:
            lines.append(
                _fmt_row(
                    f"`{rv}`",
                    "Pooled",
                    rm,
                    "delta_info",
                    "switched_toward_info",
                    "switched_away_from_info",
                ).replace(" | Pooled |", " |")
            )

    cons = summary["main_arm"]["consensus_4lab"]
    lines.extend(
        [
            "",
            "## 4-Lab Third-Person Consensus (Main 24 Pairs)",
            "",
            f"- **4/4 Unanimous Lab Flip Toward Private Info:** `{cons['n_unanimous_4of4_toward_info']}/{cons['n_pairs']}` pairs",
            f"- **>=3/4 Majority Flip Toward Private Info:** `{cons['n_majority_ge3of4_toward_info']}/{cons['n_pairs']}` pairs",
            f"- **>=3/4 Majority Positive ΔInfo:** `{cons['n_majority_ge3of4_positive_delta']}/{cons['n_pairs']}` pairs",
            f"- **Mean Pairwise Inter-Lab Agreement (Flip):** `{cons['mean_pairwise_lab_agreement_on_flip']}`",
        ]
    )

    audit = summary.get("parity_and_leakage_audit")
    if audit:
        lines.extend(
            [
                "",
                "## Pre-Execution Integrity, Word-Count Parity (`±5%`), & Zero-Leakage Audit",
                "",
                f"- **Unique Disjoint Episodes:** `{audit['n_unique_episodes']}/{audit['n_candidate_episodes']}` (`strict_disjoint_ok={audit['strict_disjoint_ok']}`)",
                f"- **All `score == 1.0`:** `{audit['all_score_1_ok']}` | **All Same-Seed Pairs:** `{audit['all_same_seed_ok']}`",
                f"- **Triage Zero-Contradiction Pool:** `{audit['triage_zero_contradiction_count']}/{audit['triage_zero_contradiction_total']}`",
                (
                    f"- **Word-Count Parity (`±5%`):** `{audit['word_parity_pass_count']}/{audit['word_parity_total']}` "
                    f"(max added rel diff = `{audit['max_added_word_rel_diff'] * 100:.2f}%`, "
                    f"max total R1 rel diff = `{audit['max_total_word_rel_diff'] * 100:.2f}%`)"
                ),
                f"- **Version B (`public_echo`) Zero Leakage:** `{audit['public_echo_zero_leak_count']}/{audit['public_echo_zero_leak_total']}`",
                f"- **Version A (`private_info`) Complete Disclosure:** `{audit['private_info_complete_count']}/{audit['private_info_complete_total']}`",
                f"- **Prompt Vendor Leaks:** `{audit['prompt_vendor_leak_count']}/{audit['n_prompts_checked']}` prompts scanned",
            ]
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Crash-Resumable Async Runner (`run_exp2_info_flip`)
# ---------------------------------------------------------------------------


def _load_existing_choices_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load completed (`chosen_slot in (1, 2)`) choice rows keyed by `(packet_id, reviewer)`."""
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupted JSONL line {line_no} in {path}: {exc}") from exc
            key = (str(row["packet_id"]), str(row["reviewer"]))
            if row.get("chosen_slot") in (1, 2) or key not in existing:
                existing[key] = row
    return existing


async def run_exp2_info_flip(
    *,
    episodes_dir: Path = DEFAULT_EPISODES_DIR,
    out_dir: Path = DEFAULT_EXP2_OUT_DIR,
    capstone_dir: Path | None = DEFAULT_CAPSTONE_DIR,
    probe_manifest: Path | None = None,
    budget_cap_usd: float = DEFAULT_EXP2_BUDGET_CAP_USD,
    concurrency: int = DEFAULT_EXP2_CONCURRENCY,
    mock: bool = False,
    dry_run: bool = False,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    client_override: OpenRouterClient | MockOpenRouterClient | None = None,
) -> dict[str, Any]:
    """Execute Experiment 2 ("The Information Flip") with crash-resumable ledgers & audit exports."""
    out_path = Path(out_dir)
    if mock and out_path.resolve() == DEFAULT_EXP2_OUT_DIR.resolve():
        raise ValueError(
            f"Refusing to write --mock Exp 2 output into live directory '{DEFAULT_EXP2_OUT_DIR}'. "
            "Pass --out-dir data/exp2_mock."
        )

    episodes = load_main_episodes(episodes_dir)
    pairs, audit = select_exp2_pairs(episodes, selection_seed=selection_seed)
    choice_specs = build_choice_specs(pairs, reviewers=EXP2_REVIEWERS)

    if dry_run:
        return {
            "dry_run": True,
            "n_pairs": len(pairs),
            "n_planned_choices": len(choice_specs),
            "audit": audit,
            "pairs": [p.to_manifest_dict() for p in pairs],
        }

    out_path.mkdir(parents=True, exist_ok=True)
    pairs_manifest = {
        "schema_version": "exp2_info_flip_pairs_v1",
        "generated_at_utc": utc_now_iso(),
        "selection_seed": selection_seed,
        "audit": audit,
        "pairs": [p.to_manifest_dict() for p in pairs],
    }
    (out_path / "pairs.json").write_text(
        json.dumps(pairs_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    reviewers = load_exp2_reviewers(
        probe_manifest=probe_manifest,
        concurrency=concurrency,
        mock=mock,
    )
    ledger = Ledger(out_path / "ledger.jsonl", budget_cap_usd=budget_cap_usd)
    choices_path = out_path / "choices.jsonl"
    existing_by_key = _load_existing_choices_jsonl(choices_path)
    write_lock = asyncio.Lock()

    owns_client = client_override is None
    if client_override is not None:
        client = client_override
    elif mock:
        client = MockInfoFlipClient(ledger, reviewers)
    else:
        total_conc = sum(spec.concurrency for spec in reviewers.values())
        client = OpenRouterClient(
            ledger,
            reviewers,
            max_connections=max(160, total_conc + 16),
            max_keepalive_connections=max(80, total_conc // 2),
        )

    choice_exclusions: list[dict[str, Any]] = []
    budget_stopped = False

    async def _run_one(cs: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal budget_stopped
        key = (str(cs["packet_id"]), str(cs["reviewer"]))
        prev = existing_by_key.get(key)
        if prev is not None and prev.get("chosen_slot") in (1, 2):
            return prev

        s1, s2 = cs["slot1"], cs["slot2"]
        reviewer = str(cs["reviewer"])
        spec = reviewers[reviewer]
        cid = f"exp2_info_flip:{cs['packet_id']}:{reviewer}"
        try:
            entry = await client.complete(
                call_id=cid,
                model_key=reviewer,
                messages=[{"role": "user", "content": cs["prompt"]}],
                cache_prefix_index=0 if spec.supports_anthropic_cache_control else None,
                metadata={
                    "stage": "exp2_info_flip_choice",
                    "arm": cs["arm"],
                    "packet_id": cs["packet_id"],
                    "pair_id": cs["pair_id"],
                    "condition": cs["condition"],
                    "order": cs["order"],
                    "task": cs["task"],
                    "target_role": cs["target_role"],
                    "reviewer": reviewer,
                    "handle_1": s1[1],
                    "handle_2": s2[1],
                    "canonical_1": s1[0],
                    "canonical_2": s2[0],
                    "partner_1": s1[2],
                    "partner_2": s2[2],
                    "variant_1": s1[3],
                    "variant_2": s2[3],
                },
            )
        except BudgetExceededError:
            budget_stopped = True
            return None
        except OpenRouterCallError as exc:
            choice_exclusions.append(
                {
                    "packet_id": cs["packet_id"],
                    "pair_id": cs["pair_id"],
                    "reviewer": reviewer,
                    "error": str(exc),
                }
            )
            return None

        chosen_slot = parse_choice(entry.response_text, s1[1], s2[1])
        chosen_canonical = s1[0] if chosen_slot == 1 else (s2[0] if chosen_slot == 2 else None)
        chosen_partner = s1[2] if chosen_slot == 1 else (s2[2] if chosen_slot == 2 else None)
        chosen_variant = s1[3] if chosen_slot == 1 else (s2[3] if chosen_slot == 2 else None)
        row = {
            "packet_id": cs["packet_id"],
            "pair_id": cs["pair_id"],
            "arm": cs["arm"],
            "task": cs["task"],
            "target_role": cs["target_role"],
            "seed_1": cs["seed_1"],
            "seed_2": cs["seed_2"],
            "condition": cs["condition"],
            "order": cs["order"],
            "reviewer": reviewer,
            "handle_1": s1[1],
            "handle_2": s2[1],
            "canonical_1": s1[0],
            "canonical_2": s2[0],
            "partner_1": s1[2],
            "partner_2": s2[2],
            "variant_1": s1[3],
            "variant_2": s2[3],
            "score_1": s1[4],
            "score_2": s2[4],
            "episode_1_id": s1[8],
            "episode_2_id": s2[8],
            "chosen_slot": chosen_slot,
            "chosen_canonical": chosen_canonical,
            "chosen_partner": chosen_partner,
            "chosen_variant": chosen_variant,
            "cross_vendor": (
                _model_family(reviewer) not in {_model_family(s1[2]), _model_family(s2[2])}
            ),
            "raw": entry.response_text,
            "prompt_tokens": entry.prompt_tokens,
            "completion_tokens": entry.completion_tokens,
            "reasoning_tokens": entry.reasoning_tokens,
            "cached_tokens": entry.cached_tokens,
            "cost_usd": entry.cost_usd,
            "latency_ms": entry.latency_ms,
        }
        async with write_lock:
            with choices_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            existing_by_key[key] = row
        return row

    try:
        raw_results = await asyncio.gather(*(_run_one(cs) for cs in choice_specs))
    finally:
        if owns_client:
            await client.aclose()

    choice_rows = [r for r in raw_results if r is not None]
    summary = summarize_exp2_results(choice_rows, capstone_dir=capstone_dir)
    summary["mock"] = mock
    summary["budget_cap_usd"] = budget_cap_usd
    summary["budget_stopped"] = budget_stopped
    summary["ledger_total_cost_usd"] = round(ledger.total_cost_usd, 6)
    summary["n_choices_excluded"] = len(choice_exclusions)
    summary["choice_exclusions"] = choice_exclusions
    summary["parity_and_leakage_audit"] = audit
    summary["reviewers"] = {
        k: {
            "slug": v.slug,
            "provider_only": list(v.provider_only),
            "reasoning_effort": v.reasoning_effort,
            "max_tokens": v.max_tokens,
            "concurrency": v.concurrency,
            "supports_anthropic_cache_control": v.supports_anthropic_cache_control,
        }
        for k, v in reviewers.items()
    }

    (out_path / "exp2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_report = render_exp2_summary_markdown(summary)
    (out_path / "exp2_summary.md").write_text(md_report, encoding="utf-8")
    return summary


__all__ = [
    "ALL_PARTNER_MODELS",
    "DEFAULT_CAPSTONE_DIR",
    "DEFAULT_EPISODES_DIR",
    "DEFAULT_EXP2_BUDGET_CAP_USD",
    "DEFAULT_EXP2_CONCURRENCY",
    "DEFAULT_EXP2_OUT_DIR",
    "DEFAULT_WORD_PARITY_TOL",
    "EXP2_MAIN_SEEDS",
    "EXP2_REVIEWERS",
    "CandidateArms",
    "InfoFlipPair",
    "MockInfoFlipClient",
    "audit_exp2_pairs",
    "build_candidate_arms",
    "build_choice_specs",
    "exp2_goalpost_r1_choice_prompt",
    "format_exp2_transcript",
    "format_outcome_card",
    "load_capstone_comparison",
    "load_exp2_reviewers",
    "load_main_episodes",
    "pair_fixed_handles",
    "parse_choice",
    "render_exp2_summary_markdown",
    "run_exp2_info_flip",
    "select_exp2_pairs",
    "summarize_exp2_results",
]
