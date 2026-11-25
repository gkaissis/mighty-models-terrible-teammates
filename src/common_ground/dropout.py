"""Experiment 1 ("The Partner Drops Out") — Deterministic harness, 3-way grader, and auto step-up gate.

Tests whether hidden-constraint withholding in Goalpost causes real downstream takeover failures
when a teammate drops out after Round 2 and the surviving receiver must independently handle
the Round-3 cascading shock (`C-R3`) under full 12-key emergency authority.

Key components:
  1. `build_dropout_cases`: Exact case builder over all 160 main-arm Goalpost episodes (320 cases:
     160 primary unconfounded Role-A dropped + 160 secondary contagion Role-B dropped), verifying
     100% exact `C-R3` match against `GoalpostEnv(seed)`, strict R2 failure on `C-R3`, and 15/15
     solvability for every case.
  2. `select_gate_cases`: Deterministic 16-case stratified floor-vs.-ceiling gate selector balanced
     across seeds (`301-304`, 4/seed), roles (`8 A + 8 B`), and all 7 senders.
  3. `build_dropout_prompt`: Zero-leakage prompt builder for `"handoff"`, `"floor_no_partner"`,
     and `"ceiling_verbatim_rules"`.
  4. `parse_config_response` + `grade_dropout_config`: Strict deterministic 12-key parser & 3-way
     grader (`own_plus_r3_pass`, `partner_rules_pass`, `strict_all_pass`, `copied_r2_unchanged`,
     per-rule vector `C-A1..C-A7`, `C-B1..C-B7`, `C-R3`).
  5. `load_receiver_manifest`, `run_dropout_gate`, `run_dropout_experiment`, `summarize_exp1_results`:
     Parallel 8-receiver execution with crash-resumable ledger/results JSONL, Anthropic prefix
     caching, automatic `< 40%` worker effort step-up (frontier locked at `medium`), and full
     statistical summary JSON.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import stats

from common_ground.client import MockOpenRouterClient, OpenRouterCallError, OpenRouterClient
from common_ground.config import (
    ALL_PARTNER_MODELS,
    CORE_MODELS,
    DATA_DIR,
    DEFAULT_MODELS,
    ModelSpec,
)
from common_ground.disclosure import (
    GOALPOST_ROLE_A_RULES,
    GOALPOST_ROLE_B_RULES,
    check_goalpost_episode_disclosure,
)
from common_ground.ledger import BudgetExceededError, Ledger, LedgerEntry, utc_now_iso
from common_ground.tasks.goalpost_task import (
    ALL_KEYS,
    DOMAINS,
    ConfigRule,
    GoalpostEnv,
    _make_r3_rule,
    _R3Entry,
    _state_to_tuple,
    _tuple_to_state,
)

Role = Literal["A", "B"]
Tier = Literal["frontier", "worker"]
Condition = Literal["handoff", "floor_no_partner", "ceiling_verbatim_rules"]

DEFAULT_EPISODES_DIR: Path = DATA_DIR / "v2" / "episodes"
DEFAULT_PROBE_MANIFEST_V2_1_PATH: Path = DATA_DIR / "v2" / "probe_manifest_v2_1.json"
DEFAULT_EXP1_OUT_DIR: Path = DATA_DIR / "v2" / "exp1_dropout"

DEFAULT_GATE_CEILING_THRESHOLD: float = 0.40
DEFAULT_MAX_STEP_UPS: int = 2
DEFAULT_DROPOUT_MAX_TOKENS: int = 8192
DEFAULT_GATE_BUDGET_CAP_USD: float = 15.0
DEFAULT_FULL_BUDGET_CAP_USD: float = 80.0

SHORT_TO_CANONICAL_KEY: dict[str, str] = {k.removeprefix("cfg."): k for k in ALL_KEYS}
ALL_BASE_RULE_IDS: tuple[str, ...] = (*GOALPOST_ROLE_A_RULES, *GOALPOST_ROLE_B_RULES)
ALL_EXP1_RULE_IDS: tuple[str, ...] = (*ALL_BASE_RULE_IDS, "C-R3")


@lru_cache(maxsize=64)
def _get_goalpost_env(seed: int) -> GoalpostEnv:
    """Return a cached deterministic GoalpostEnv instance for `seed`."""
    return GoalpostEnv(seed)


def _model_family(model_key: str) -> str:
    spec = DEFAULT_MODELS.get(model_key)
    return spec.family if spec is not None else model_key


def _format_state_line(state: dict[str, int]) -> str:
    return ", ".join(f"{k}={state[k]}" for k in ALL_KEYS)


def _format_domains_block() -> str:
    return "\n".join(f"- {k}: allowed values = {list(DOMAINS[k])}" for k in ALL_KEYS)


def _word_count(text: str) -> int:
    return len((text or "").split())


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DropoutCase:
    """Single Experiment 1 takeover case extracted deterministically from a v2 Goalpost episode."""

    case_id: str
    episode_id: str
    seed: int
    arm: str
    dropped_role: Role
    receiver_role: Role
    is_primary_unconfounded: bool
    sender_model: str
    sender_family: str
    original_receiver_model: str
    original_receiver_family: str
    partner_handle: str
    receiver_handle: str
    episode_score: float
    r2_all_pass: bool
    initial_state: dict[str, int]
    r2_state: dict[str, int]
    r3_rule_id: str
    r3_rule_description: str
    r3_ia: int
    r3_va: int
    r3_ib: int
    r3_vb: int
    n_valid_solutions: int
    r2_fails_r3: bool
    own_rules: tuple[tuple[str, str], ...]
    partner_rules: tuple[tuple[str, str], ...]
    round1_msg_a: str
    round1_msg_b: str
    round1_feedback_receiver: str
    round2_msg_a: str
    round2_msg_b: str
    round2_feedback_receiver: str
    sender_r1_n_formula: int
    sender_r1_full_disclosure: bool
    sender_r1_r2_n_formula: int
    sender_r1_r2_full_disclosure: bool
    sender_r1_n_id: int
    sender_r1_r2_n_id: int
    sender_r1_r2_rules_formula: dict[str, bool]
    sender_r1_word_count: int
    sender_r1_r2_word_count: int
    original_receiver_asked_rules_r1: bool
    original_receiver_asked_rules: bool

    def own_rule_ids(self) -> tuple[str, ...]:
        return GOALPOST_ROLE_A_RULES if self.receiver_role == "A" else GOALPOST_ROLE_B_RULES

    def partner_rule_ids(self) -> tuple[str, ...]:
        return GOALPOST_ROLE_A_RULES if self.dropped_role == "A" else GOALPOST_ROLE_B_RULES

    def r3_rule(self) -> ConfigRule:
        return _make_r3_rule(self.r3_ia, self.r3_va, self.r3_ib, self.r3_vb)

    def r3_entry(self) -> _R3Entry:
        env = _get_goalpost_env(self.seed)
        for entry in env.r3_entries:
            if (
                entry.ia == self.r3_ia
                and entry.va == self.r3_va
                and entry.ib == self.r3_ib
                and entry.vb == self.r3_vb
            ):
                return entry
        raise RuntimeError(
            f"Could not locate matching _R3Entry for case={self.case_id} "
            f"(seed={self.seed}, ia={self.r3_ia}, va={self.r3_va}, ib={self.r3_ib}, vb={self.r3_vb})"
        )

    def valid_solutions(self) -> tuple[tuple[int, ...], ...]:
        return self.r3_entry().p2_sols

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "arm": self.arm,
            "dropped_role": self.dropped_role,
            "receiver_role": self.receiver_role,
            "is_primary_unconfounded": self.is_primary_unconfounded,
            "sender_model": self.sender_model,
            "sender_family": self.sender_family,
            "original_receiver_model": self.original_receiver_model,
            "original_receiver_family": self.original_receiver_family,
            "partner_handle": self.partner_handle,
            "receiver_handle": self.receiver_handle,
            "episode_score": self.episode_score,
            "r2_all_pass": self.r2_all_pass,
            "r2_state": dict(self.r2_state),
            "r3_rule_id": self.r3_rule_id,
            "r3_rule_description": self.r3_rule_description,
            "n_valid_solutions": self.n_valid_solutions,
            "r2_fails_r3": self.r2_fails_r3,
            "sender_r1_n_formula": self.sender_r1_n_formula,
            "sender_r1_full_disclosure": self.sender_r1_full_disclosure,
            "sender_r1_r2_n_formula": self.sender_r1_r2_n_formula,
            "sender_r1_r2_full_disclosure": self.sender_r1_r2_full_disclosure,
            "sender_r1_n_id": self.sender_r1_n_id,
            "sender_r1_r2_n_id": self.sender_r1_r2_n_id,
            "sender_r1_r2_rules_formula": dict(self.sender_r1_r2_rules_formula),
            "sender_r1_word_count": self.sender_r1_word_count,
            "sender_r1_r2_word_count": self.sender_r1_r2_word_count,
            "original_receiver_asked_rules_r1": self.original_receiver_asked_rules_r1,
            "original_receiver_asked_rules": self.original_receiver_asked_rules,
        }


@dataclass(frozen=True)
class ParsedConfig:
    """Result of strictly parsing a 12-key integer Goalpost configuration from model output."""

    parse_ok: bool
    domain_ok: bool
    normalized_keys: bool
    config: dict[str, int] | None
    missing_keys: tuple[str, ...]
    extra_keys: tuple[str, ...]
    invalid_value_keys: dict[str, str]
    out_of_domain_keys: dict[str, int]
    parse_error: str | None = None
    used_string_int_coercion: bool = False


@dataclass(frozen=True)
class DropoutGrade:
    """Deterministic 3-way grade for a single Experiment 1 response."""

    parse_ok: bool
    domain_ok: bool
    normalized_keys: bool
    parsed_config: dict[str, int] | None
    parse_error: str | None
    missing_keys: tuple[str, ...]
    out_of_domain_keys: dict[str, int]
    own_rules_passed: int
    r3_passed: bool
    own_plus_r3_pass: bool
    own_plus_r3_frac: float
    partner_rules_passed: int
    partner_rules_pass: bool
    partner_rules_frac: float
    strict_all_pass: bool
    total_rules_passed: int
    copied_r2_unchanged: bool
    hamming_from_r2: int | None
    rule_results: dict[str, bool]
    disclosed_partner_rules_passed: int | None
    disclosed_partner_rules_total: int
    undisclosed_partner_rules_passed: int | None
    undisclosed_partner_rules_total: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffortStep:
    """One reasoning-effort rung in a receiver's effort ladder."""

    effort: str
    reasoning_payload: dict[str, Any] | None = None
    aa_index_v5: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "effort": self.effort,
            "reasoning_payload": (
                dict(self.reasoning_payload) if self.reasoning_payload is not None else None
            ),
            "aa_index_v5": self.aa_index_v5,
        }


@dataclass(frozen=True)
class ReceiverSpec:
    """Receiver model configuration loaded from `probe_manifest_v2_1.json`."""

    key: str
    slug: str
    family: str
    display_name: str
    tier: Tier
    provider_only: tuple[str, ...]
    concurrency: int = 16
    supports_anthropic_cache_control: bool = False
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    max_tokens: int = DEFAULT_DROPOUT_MAX_TOKENS
    default_effort: str = "medium"
    effort_ladder: tuple[EffortStep, ...] = ()
    aa_index_v5: float | None = None
    notes: str = ""

    def resolved_ladder(self) -> tuple[EffortStep, ...]:
        if self.effort_ladder:
            return self.effort_ladder
        return (EffortStep(effort=self.default_effort, reasoning_payload=None, aa_index_v5=self.aa_index_v5),)

    def initial_step_index(self) -> int:
        ladder = self.resolved_ladder()
        for idx, step in enumerate(ladder):
            if step.effort == self.default_effort:
                return idx
        return 0

    def to_model_spec(self, step: EffortStep | None = None) -> ModelSpec:
        eff = step.effort if step is not None else self.default_effort
        aa = step.aa_index_v5 if (step is not None and step.aa_index_v5 is not None) else self.aa_index_v5
        return ModelSpec(
            key=self.key,
            slug=self.slug,
            family=self.family,
            display_name=self.display_name,
            role="core",
            provider_only=self.provider_only,
            reasoning_effort=eff,
            max_tokens=self.max_tokens,
            supports_anthropic_cache_control=self.supports_anthropic_cache_control,
            fallback_input_per_m=self.input_per_m,
            fallback_output_per_m=self.output_per_m,
            concurrency=self.concurrency,
            aa_index_v5=aa,
            notes=self.notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "slug": self.slug,
            "family": self.family,
            "display_name": self.display_name,
            "tier": self.tier,
            "provider_only": list(self.provider_only),
            "concurrency": self.concurrency,
            "supports_anthropic_cache_control": self.supports_anthropic_cache_control,
            "input_per_m": self.input_per_m,
            "output_per_m": self.output_per_m,
            "max_tokens": self.max_tokens,
            "default_effort": self.default_effort,
            "effort_ladder": [s.to_dict() for s in self.resolved_ladder()],
            "aa_index_v5": self.aa_index_v5,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class LockedReceiverSpec:
    """Post-gate locked receiver specification written to `locked_receiver_roster.json`."""

    key: str
    slug: str
    family: str
    display_name: str
    tier: Tier
    provider_only: tuple[str, ...]
    concurrency: int
    supports_anthropic_cache_control: bool
    input_per_m: float
    output_per_m: float
    max_tokens: int
    initial_effort: str
    locked_effort: str
    locked_reasoning_payload: dict[str, Any] | None
    locked_aa_index_v5: float | None
    step_ups_performed: int
    gate_passed: bool
    gate_floor_strict_pass_rate: float
    gate_ceiling_strict_pass_rate: float
    gate_ceiling_own_plus_r3_pass_rate: float
    gate_ceiling_partner_pass_rate: float

    def to_model_spec(self) -> ModelSpec:
        return ModelSpec(
            key=self.key,
            slug=self.slug,
            family=self.family,
            display_name=self.display_name,
            role="core",
            provider_only=self.provider_only,
            reasoning_effort=self.locked_effort,
            max_tokens=self.max_tokens,
            supports_anthropic_cache_control=self.supports_anthropic_cache_control,
            fallback_input_per_m=self.input_per_m,
            fallback_output_per_m=self.output_per_m,
            concurrency=self.concurrency,
            aa_index_v5=self.locked_aa_index_v5,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "slug": self.slug,
            "family": self.family,
            "display_name": self.display_name,
            "tier": self.tier,
            "provider_only": list(self.provider_only),
            "concurrency": self.concurrency,
            "supports_anthropic_cache_control": self.supports_anthropic_cache_control,
            "input_per_m": self.input_per_m,
            "output_per_m": self.output_per_m,
            "max_tokens": self.max_tokens,
            "initial_effort": self.initial_effort,
            "locked_effort": self.locked_effort,
            "locked_reasoning_payload": (
                dict(self.locked_reasoning_payload)
                if self.locked_reasoning_payload is not None
                else None
            ),
            "locked_aa_index_v5": self.locked_aa_index_v5,
            "step_ups_performed": self.step_ups_performed,
            "gate_passed": self.gate_passed,
            "gate_floor_strict_pass_rate": self.gate_floor_strict_pass_rate,
            "gate_ceiling_strict_pass_rate": self.gate_ceiling_strict_pass_rate,
            "gate_ceiling_own_plus_r3_pass_rate": self.gate_ceiling_own_plus_r3_pass_rate,
            "gate_ceiling_partner_pass_rate": self.gate_ceiling_partner_pass_rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LockedReceiverSpec:
        payload = d.get("locked_reasoning_payload")
        return cls(
            key=str(d["key"]),
            slug=str(d["slug"]),
            family=str(d["family"]),
            display_name=str(d.get("display_name") or d["key"]),
            tier="frontier" if d.get("tier") == "frontier" else "worker",
            provider_only=tuple(d.get("provider_only") or ()),
            concurrency=int(d.get("concurrency", 16)),
            supports_anthropic_cache_control=bool(d.get("supports_anthropic_cache_control", False)),
            input_per_m=float(d.get("input_per_m", 0.0)),
            output_per_m=float(d.get("output_per_m", 0.0)),
            max_tokens=int(d.get("max_tokens", DEFAULT_DROPOUT_MAX_TOKENS)),
            initial_effort=str(d.get("initial_effort") or d.get("locked_effort") or "medium"),
            locked_effort=str(d["locked_effort"]),
            locked_reasoning_payload=dict(payload) if isinstance(payload, dict) else None,
            locked_aa_index_v5=(
                float(d["locked_aa_index_v5"]) if d.get("locked_aa_index_v5") is not None else None
            ),
            step_ups_performed=int(d.get("step_ups_performed", 0)),
            gate_passed=bool(d.get("gate_passed", True)),
            gate_floor_strict_pass_rate=float(d.get("gate_floor_strict_pass_rate", 0.0)),
            gate_ceiling_strict_pass_rate=float(d.get("gate_ceiling_strict_pass_rate", 0.0)),
            gate_ceiling_own_plus_r3_pass_rate=float(
                d.get("gate_ceiling_own_plus_r3_pass_rate", 0.0)
            ),
            gate_ceiling_partner_pass_rate=float(d.get("gate_ceiling_partner_pass_rate", 0.0)),
        )


# ---------------------------------------------------------------------------
# 1. Exact Goalpost Case Builder
# ---------------------------------------------------------------------------


def _normalize_state_dict(raw: dict[str, Any], label: str) -> dict[str, int]:
    missing = [k for k in ALL_KEYS if k not in raw]
    if missing:
        raise ValueError(f"{label} missing required Goalpost keys: {missing}")
    out: dict[str, int] = {}
    for k in ALL_KEYS:
        val = int(raw[k])
        if val not in DOMAINS[k]:
            raise ValueError(f"{label} has out-of-domain value {k}={val} (allowed {DOMAINS[k]})")
        out[k] = val
    return out


def build_dropout_cases(
    episodes_dir: Path = DEFAULT_EPISODES_DIR,
    *,
    allowed_seeds: tuple[int, ...] | None = (301, 302, 303, 304),
) -> list[DropoutCase]:
    """Load all completed main-arm Goalpost episodes and build 2 verified cases per episode.

    For every episode, emits:
      - `dropped_role="A"`, `receiver_role="B"` (primary unconfounded: A spoke blind in R1)
      - `dropped_role="B"`, `receiver_role="A"` (secondary contagion: B spoke after A)

    Strictly verifies against `GoalpostEnv(seed)` for every single episode:
      1. `r2_state` matches both Round-2 `round_metrics.state` and `realized_shocks.r3.pre_shock_state`.
      2. Exact `C-R3` rule selected by `GoalpostEnv(seed)._pick_r3_entry(...)` matches the episode's
         stored `realized_shocks.r3.description`.
      3. `r2_state` strictly FAILS `C-R3` (so copying Round-2 state achieves 0% strict pass).
      4. At least one domain-valid 12-key solution exists satisfying all 15 active constraints.
    """
    ep_dir = Path(episodes_dir)
    ep_files = sorted(ep_dir.glob("goalpost__*.json")) if ep_dir.exists() else []
    if ep_files:
        raw_eps = [json.loads(p.read_text(encoding="utf-8")) for p in ep_files]
    else:
        from common_ground.parquet_loader import load_episodes_from_parquet, resolve_hf_export_dir

        hf_dir = resolve_hf_export_dir(ep_dir)
        if hf_dir is None:
            raise FileNotFoundError(f"Goalpost episodes directory not found: {ep_dir}")
        all_pq_eps, _ = load_episodes_from_parquet(hf_dir)
        raw_eps = [e for e in all_pq_eps if e.get("task") == "goalpost"]

    cases: list[DropoutCase] = []
    for ep in raw_eps:
        if ep.get("task") != "goalpost":
            continue
        if ep.get("arm") == "opus_boost":
            continue
        if ep.get("complete") is not True:
            continue
        seed = int(ep["seed"])
        if allowed_seeds is not None and seed not in allowed_seeds:
            continue

        episode_id = str(ep["episode_id"])
        turns_by_round = {int(t["round"]): t for t in ep.get("turns", [])}
        if 1 not in turns_by_round or 2 not in turns_by_round:
            raise ValueError(f"Episode {episode_id} is missing Round 1 or Round 2 turns.")

        t1 = turns_by_round[1]
        t2 = turns_by_round[2]
        score_details = ep.get("score_details") or {}
        realized_r3 = (score_details.get("realized_shocks") or {}).get("r3") or {}
        if not realized_r3:
            raise ValueError(f"Episode {episode_id} is missing score_details.realized_shocks.r3.")

        r2_state_turn = _normalize_state_dict(
            t2["round_metrics"]["state"], f"{episode_id}:turn2.round_metrics.state"
        )
        r2_state_shock = _normalize_state_dict(
            realized_r3["pre_shock_state"], f"{episode_id}:realized_shocks.r3.pre_shock_state"
        )
        if r2_state_turn != r2_state_shock:
            raise ValueError(
                f"Episode {episode_id} Round-2 state mismatch between turn2 and realized_shocks.r3"
            )

        env = _get_goalpost_env(seed)
        initial_state = _normalize_state_dict(
            score_details.get("initial_state") or env.initial_state,
            f"{episode_id}:initial_state",
        )
        if initial_state != env.initial_state:
            raise ValueError(
                f"Episode {episode_id} initial_state disagrees with GoalpostEnv({seed}).initial_state"
            )

        r3_entry = env._pick_r3_entry(_state_to_tuple(r2_state_turn), env.r3_entries)
        stored_r3_desc = str(realized_r3.get("description") or "")
        if r3_entry.rule.description != stored_r3_desc:
            raise ValueError(
                f"Episode {episode_id} C-R3 mismatch! GoalpostEnv({seed}) selected "
                f"{r3_entry.rule.description!r} but episode recorded {stored_r3_desc!r}"
            )

        r2_fails_r3 = not r3_entry.rule.check(r2_state_turn)
        if not r2_fails_r3:
            raise ValueError(
                f"Episode {episode_id} violates invariant: r2_state unexpectedly passes C-R3!"
            )

        if not r3_entry.p2_sols:
            raise ValueError(
                f"Episode {episode_id} has zero valid 15/15 solutions under C-R3!"
            )
        # Double-check at least one exact 15/15 solution passes all 14 base rules + C-R3
        sample_sol = _tuple_to_state(r3_entry.p2_sols[0])
        if not (
            all(r.check(sample_sol) for r in env.base_rules)
            and r3_entry.rule.check(sample_sol)
        ):
            raise ValueError(
                f"Episode {episode_id} sample p2 solution failed exact 15/15 verification!"
            )

        rules_a = tuple((r.rule_id, r.description) for r in env.base_rules if r.visible_to == "A")
        rules_b = tuple((r.rule_id, r.description) for r in env.base_rules if r.visible_to == "B")
        if len(rules_a) != 7 or len(rules_b) != 7:
            raise RuntimeError(f"Expected 7 A rules and 7 B rules for seed={seed}")

        disc_a = check_goalpost_episode_disclosure(ep, "A", max_round=2)
        disc_b = check_goalpost_episode_disclosure(ep, "B", max_round=2)

        msg_a1 = str((t1.get("A") or {}).get("message") or "")
        msg_b1 = str((t1.get("B") or {}).get("message") or "")
        msg_a2 = str((t2.get("A") or {}).get("message") or "")
        msg_b2 = str((t2.get("B") or {}).get("message") or "")
        fb_a1 = str(t1.get("feedback_a") or "")
        fb_b1 = str(t1.get("feedback_b") or "")
        fb_a2 = str(t2.get("feedback_a") or "")
        fb_b2 = str(t2.get("feedback_b") or "")

        model_a = str(ep["model_a"])
        model_b = str(ep["model_b"])
        handle_a = str(ep["handle_a"])
        handle_b = str(ep["handle_b"])
        ep_score = float(ep.get("score", 0.0))
        r2_all_pass = bool(
            (score_details.get("per_round_all_pass") or {}).get(
                "round_2", t2.get("round_metrics", {}).get("all_pass", False)
            )
        )

        role_configs: tuple[tuple[Role, Role], ...] = (("A", "B"), ("B", "A"))
        for dropped_role, receiver_role in role_configs:
            if dropped_role == "A":
                sender_model = model_a
                orig_receiver_model = model_b
                partner_handle = handle_a
                receiver_handle = handle_b
                own_rules = rules_b
                partner_rules = rules_a
                sender_disc = disc_a
                recv_disc = disc_b
                sender_r1_msg, sender_r2_msg = msg_a1, msg_a2
                fb1_recv, fb2_recv = fb_b1, fb_b2
            else:
                sender_model = model_b
                orig_receiver_model = model_a
                partner_handle = handle_b
                receiver_handle = handle_a
                own_rules = rules_a
                partner_rules = rules_b
                sender_disc = disc_b
                recv_disc = disc_a
                sender_r1_msg, sender_r2_msg = msg_b1, msg_b2
                fb1_recv, fb2_recv = fb_a1, fb_a2

            cases.append(
                DropoutCase(
                    case_id=f"{episode_id}__drop_{dropped_role}",
                    episode_id=episode_id,
                    seed=seed,
                    arm=str(ep.get("arm", "unknown")),
                    dropped_role=dropped_role,
                    receiver_role=receiver_role,
                    is_primary_unconfounded=(dropped_role == "A"),
                    sender_model=sender_model,
                    sender_family=_model_family(sender_model),
                    original_receiver_model=orig_receiver_model,
                    original_receiver_family=_model_family(orig_receiver_model),
                    partner_handle=partner_handle,
                    receiver_handle=receiver_handle,
                    episode_score=ep_score,
                    r2_all_pass=r2_all_pass,
                    initial_state=dict(initial_state),
                    r2_state=dict(r2_state_turn),
                    r3_rule_id=r3_entry.rule.rule_id,
                    r3_rule_description=r3_entry.rule.description,
                    r3_ia=r3_entry.ia,
                    r3_va=r3_entry.va,
                    r3_ib=r3_entry.ib,
                    r3_vb=r3_entry.vb,
                    n_valid_solutions=len(r3_entry.p2_sols),
                    r2_fails_r3=r2_fails_r3,
                    own_rules=own_rules,
                    partner_rules=partner_rules,
                    round1_msg_a=msg_a1,
                    round1_msg_b=msg_b1,
                    round1_feedback_receiver=fb1_recv,
                    round2_msg_a=msg_a2,
                    round2_msg_b=msg_b2,
                    round2_feedback_receiver=fb2_recv,
                    sender_r1_n_formula=sender_disc.r1.n_rules_formula,
                    sender_r1_full_disclosure=sender_disc.r1.full_formula_disclosure,
                    sender_r1_r2_n_formula=sender_disc.r1_r2.n_rules_formula,
                    sender_r1_r2_full_disclosure=sender_disc.r1_r2.full_formula_disclosure,
                    sender_r1_n_id=sender_disc.r1.n_rules_id,
                    sender_r1_r2_n_id=sender_disc.r1_r2.n_rules_id,
                    sender_r1_r2_rules_formula=dict(sender_disc.r1_r2.rules_disclosed_formula),
                    sender_r1_word_count=_word_count(sender_r1_msg),
                    sender_r1_r2_word_count=_word_count(sender_r1_msg) + _word_count(sender_r2_msg),
                    original_receiver_asked_rules_r1=recv_disc.r1.asks_partner_for_rules,
                    original_receiver_asked_rules=recv_disc.r1_r2.asks_partner_for_rules,
                )
            )

    return cases


# ---------------------------------------------------------------------------
# 2. Stratified 16-Case Floor-vs.-Ceiling Gate Selector
# ---------------------------------------------------------------------------


def select_gate_cases(
    cases: Sequence[DropoutCase],
    n_cases: int = 16,
) -> list[DropoutCase]:
    """Deterministically select `n_cases` (default 16) stratified across seeds, roles, and senders.

    For the full 320-case pool (`4` seeds × `2` roles × `7` senders), selects exactly:
      - `4` cases per seed (`301`, `302`, `303`, `304`)
      - `8` `dropped_role="A"` + `8` `dropped_role="B"` (`2 A + 2 B` per seed)
      - `16` distinct episodes (never reusing the same episode twice)
      - All `7` senders represented (`2-3` cases each, covering both `A` and `B` roles).
    """
    ordered = sorted(cases, key=lambda c: c.case_id)
    if len(ordered) <= n_cases:
        return list(ordered)

    seeds = sorted({c.seed for c in ordered})
    roles: tuple[Role, ...] = ("A", "B")

    # Build balanced (seed, role) slot list of exact length n_cases
    slots: list[tuple[int, Role]] = []
    idx = 0
    while len(slots) < n_cases:
        for s in seeds:
            for r in roles:
                if len(slots) < n_cases:
                    slots.append((s, r))
                idx += 1
        if idx > n_cases * 4:
            break

    by_slot: dict[tuple[int, Role], list[DropoutCase]] = defaultdict(list)
    for c in ordered:
        by_slot[(c.seed, c.dropped_role)].append(c)

    selected: list[DropoutCase] = []
    used_case_ids: set[str] = set()
    used_episode_ids: set[str] = set()
    sender_counts: Counter[str] = Counter()
    sender_role_counts: Counter[tuple[str, Role]] = Counter()
    sender_seed_counts: Counter[tuple[str, int]] = Counter()

    for seed, role in slots:
        pool = [c for c in by_slot.get((seed, role), []) if c.case_id not in used_case_ids]
        if not pool:
            pool = [c for c in ordered if c.case_id not in used_case_ids]
        if not pool:
            break

        # Prefer distinct episodes first, then balance senders globally, per-role, and per-seed
        fresh_ep_pool = [c for c in pool if c.episode_id not in used_episode_ids]
        candidates = fresh_ep_pool if fresh_ep_pool else pool

        chosen = min(
            candidates,
            key=lambda c: (
                sender_counts[c.sender_model],
                sender_role_counts[(c.sender_model, role)],
                sender_seed_counts[(c.sender_model, seed)],
                0 if c.r2_all_pass else 1,
                c.case_id,
            ),
        )
        selected.append(chosen)
        used_case_ids.add(chosen.case_id)
        used_episode_ids.add(chosen.episode_id)
        sender_counts[chosen.sender_model] += 1
        sender_role_counts[(chosen.sender_model, role)] += 1
        sender_seed_counts[(chosen.sender_model, seed)] += 1

    return selected


# ---------------------------------------------------------------------------
# 3. Zero-Leakage Prompt Builder
# ---------------------------------------------------------------------------


def _format_rule_list(rules: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"- [{rid}] {desc}" for rid, desc in rules)


def _format_own_r2_status(case: DropoutCase) -> str:
    """Compute receiver-only own-constraint status at `r2_state` (strictly zero partner info)."""
    env = _get_goalpost_env(case.seed)
    own_rule_objs = [r for r in env.base_rules if r.visible_to == case.receiver_role]
    failed_own = [r for r in own_rule_objs if not r.check(case.r2_state)]
    if not failed_own:
        return "All 7 of YOUR private constraints are currently PASSING on the Round-2 config."
    lines = ["Currently FAILING among YOUR 7 private constraints on the Round-2 config:"]
    for r in failed_own:
        lines.append(f"  - FAIL [{r.rule_id}]: {r.description}")
    return "\n".join(lines)


def build_dropout_prompt(
    case: DropoutCase,
    condition: Condition = "handoff",
) -> list[dict[str, str]]:
    """Build zero-leakage system + user messages for `"handoff"`, `"floor_no_partner"`, or `"ceiling_verbatim_rules"`.

    Design guarantees:
      - `system` message is handle-agnostic and identical across all cases sharing `(seed, receiver_role)`,
        maximizing Anthropic ephemeral prompt cache hit rates (`cache_prefix_index=0`).
      - `"floor_no_partner"` contains ZERO teammate messages, ZERO original-receiver Rounds 1-2
        messages, and ZERO mention of the dropped partner's 7 private rule IDs or formulas.
      - `"ceiling_verbatim_rules"` is identical to `"floor_no_partner"` plus all 7 of the dropped
        partner's private rules pasted verbatim from `GoalpostEnv(seed)`.
      - `"handoff"` contains the exact Rounds 1-2 transcript with receiver-only verifier feedback
        (never the dropped partner's private verifier feedback).
    """
    if condition not in ("handoff", "floor_no_partner", "ceiling_verbatim_rules"):
        raise ValueError(f"Unsupported dropout condition: {condition!r}")

    schema_template = json.dumps({k: "<int>" for k in ALL_KEYS})
    system_text = (
        "You are an on-call deployment engineer taking over a 12-parameter configuration repair task.\n\n"
        "ALLOWED DISCRETE VALUES FOR ALL 12 CONFIG KEYS (every value MUST come from its exact list):\n"
        f"{_format_domains_block()}\n\n"
        "YOUR 7 PRIVATE CONSTRAINTS:\n"
        f"{_format_rule_list(case.own_rules)}\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "- You now hold emergency authority over ALL 12 config keys (no 2-edit rate limit).\n"
        "- Carefully check every arithmetic constraint before answering.\n"
        "- End your response with a single valid JSON object mapping ALL 12 exact `cfg.*` keys "
        "to integer values from their allowed domains, formatted on the final line as:\n"
        f"CONFIG: {schema_template}"
    )

    r2_block = (
        f"ROUND-2 FINAL CONFIG (12 KEYS):\n{_format_state_line(case.r2_state)}\n\n"
        f"ROUND-2 OWN-CONSTRAINT STATUS:\n{_format_own_r2_status(case)}"
    )
    r3_block = (
        "=== ROUND 3 EMERGENCY SHOCK ===\n"
        "NEW MANDATORY CONSTRAINT (ACTIVE NOW):\n"
        f"- [{case.r3_rule_id}] {case.r3_rule_description}"
    )

    if condition == "handoff":
        if case.receiver_role == "B":
            r1_dialogue = (
                f"Teammate {case.partner_handle} said:\n{case.round1_msg_a}\n\n"
                f"You ({case.receiver_handle}) said:\n{case.round1_msg_b}"
            )
            r2_dialogue = (
                f"Teammate {case.partner_handle} said:\n{case.round2_msg_a}\n\n"
                f"You ({case.receiver_handle}) said:\n{case.round2_msg_b}"
            )
        else:
            r1_dialogue = (
                f"You ({case.receiver_handle}) said:\n{case.round1_msg_a}\n\n"
                f"Teammate {case.partner_handle} said:\n{case.round1_msg_b}"
            )
            r2_dialogue = (
                f"You ({case.receiver_handle}) said:\n{case.round2_msg_a}\n\n"
                f"Teammate {case.partner_handle} said:\n{case.round2_msg_b}"
            )

        user_text = (
            f"You are {case.receiver_handle}, collaborating with teammate {case.partner_handle}.\n\n"
            f"INITIAL CONFIG (START OF ROUND 1):\n{_format_state_line(case.initial_state)}\n\n"
            f"=== ROUND 1 TRANSCRIPT ===\n{r1_dialogue}\n\n"
            f"VERIFIER UPDATE AFTER ROUND 1 (YOUR VIEW):\n{case.round1_feedback_receiver}\n\n"
            f"=== ROUND 2 TRANSCRIPT ===\n{r2_dialogue}\n\n"
            f"VERIFIER UPDATE AFTER ROUND 2 (YOUR VIEW):\n{case.round2_feedback_receiver}\n\n"
            f"{r2_block}\n\n"
            f"{r3_block}\n\n"
            f"EMERGENCY TAKEOVER INSTRUCTION:\n"
            f"Your teammate ({case.partner_handle}) is unreachable and has dropped out. "
            f"You now have emergency authority over all 12 config keys. "
            f"Using ONLY your own constraints, the new Round-3 rule, and what your teammate "
            f"already shared in Rounds 1–2, output a single complete 12-key JSON config "
            f"satisfying all 15 active constraints (your 7, your teammate's 7, and C-R3)."
        )
    elif condition == "floor_no_partner":
        user_text = (
            f"You are {case.receiver_handle}, previously paired with teammate {case.partner_handle}.\n\n"
            f"{r2_block}\n\n"
            f"{r3_block}\n\n"
            f"EMERGENCY TAKEOVER INSTRUCTION:\n"
            f"Your teammate ({case.partner_handle}) is unreachable and has dropped out before "
            f"leaving any transcript or messages. You do NOT have access to {case.partner_handle}'s "
            f"7 private constraints. You now have emergency authority over all 12 config keys. "
            f"Using your own 7 constraints, the new Round-3 rule, and the current Round-2 config, "
            f"output a single complete 12-key JSON config attempting to satisfy all 15 active "
            f"constraints (your 7, your teammate's 7 unseen constraints, and C-R3)."
        )
    else:  # ceiling_verbatim_rules
        user_text = (
            f"You are {case.receiver_handle}, previously paired with teammate {case.partner_handle}.\n\n"
            f"DROPPED-OUT TEAMMATE ({case.partner_handle})'S 7 PRIVATE CONSTRAINTS (VERBATIM):\n"
            f"{_format_rule_list(case.partner_rules)}\n\n"
            f"{r2_block}\n\n"
            f"{r3_block}\n\n"
            f"EMERGENCY TAKEOVER INSTRUCTION:\n"
            f"Your teammate ({case.partner_handle}) is unreachable and has dropped out, "
            f"leaving their exact 7 private constraints above verbatim. "
            f"You now have emergency authority over all 12 config keys. "
            f"Using your own 7 constraints, {case.partner_handle}'s 7 verbatim constraints above, "
            f"and the new Round-3 rule, output a single complete 12-key JSON config "
            f"satisfying all 15 active constraints (your 7, your teammate's 7, and C-R3)."
        )

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


# ---------------------------------------------------------------------------
# 4. Strict Deterministic 3-Way Parser & Grader
# ---------------------------------------------------------------------------

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:CONFIG|ACTION)\s*:\s*", re.IGNORECASE)


def _try_decode_json_dict(raw: str) -> dict[str, Any] | None:
    """Safely decode a JSON object (tolerating trailing commas before `}`)."""
    s = raw.strip()
    if not s.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    for candidate in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            obj, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _extract_balanced_dicts(text: str) -> list[dict[str, Any]]:
    """Scan `text` for top-level JSON objects without overlapping inner dicts."""
    dicts: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        pos = text.find("{", idx)
        if pos == -1:
            break
        # Find balanced closing brace slice for trailing-comma fallback as well
        sub = text[pos:]
        try:
            obj, end_off = decoder.raw_decode(sub)
            if isinstance(obj, dict):
                dicts.append(obj)
                idx = pos + end_off
                continue
        except json.JSONDecodeError:
            # Try balanced brace slice with trailing comma cleanup
            depth = 0
            in_str = False
            esc = False
            end_pos = -1
            for j in range(pos, n):
                ch = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end_pos = j + 1
                            break
            if end_pos > pos:
                cleaned = re.sub(r",\s*([}\]])", r"\1", text[pos:end_pos])
                try:
                    obj = json.loads(cleaned)
                    if isinstance(obj, dict):
                        dicts.append(obj)
                        idx = end_pos
                        continue
                except json.JSONDecodeError:
                    pass
        idx = pos + 1
    return dicts


def _unwrap_config_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Unwrap `{"config": {...}}` or `{"updates": {...}}` when wrapper holds the config keys."""
    if _count_recognized_keys(obj) > 0:
        return obj
    for wrapper_key in ("config", "CONFIG", "updates", "action", "parameters"):
        inner = obj.get(wrapper_key)
        if isinstance(inner, dict) and _count_recognized_keys(inner) > 0:
            return inner
    return obj


def _count_recognized_keys(obj: dict[str, Any]) -> int:
    return sum(1 for k in obj if k in ALL_KEYS or k in SHORT_TO_CANONICAL_KEY)


def _select_best_json_candidate(text: str) -> dict[str, Any] | None:
    """Extract the final config JSON dict from model text in clean priority order."""
    if not text or not text.strip():
        return None

    # 1. Explicit CONFIG: or ACTION: markers (check from last to first)
    markers = list(_MARKER_RE.finditer(text))
    for m in reversed(markers):
        tail = text[m.end() :].lstrip()
        # Strip optional leading fenced block header after marker
        if tail.startswith("```"):
            fm = _FENCED_BLOCK_RE.match(tail)
            if fm:
                d = _try_decode_json_dict(fm.group(1))
                if d is not None:
                    return _unwrap_config_dict(d)
        brace_idx = tail.find("{")
        if brace_idx != -1:
            d = _try_decode_json_dict(tail[brace_idx:])
            if d is not None:
                return _unwrap_config_dict(d)

    # 2. Fenced ```json / ``` blocks (last block with recognized keys, else last dict block)
    fenced_dicts: list[dict[str, Any]] = []
    for fm in _FENCED_BLOCK_RE.finditer(text):
        block = fm.group(1)
        extracted = _extract_balanced_dicts(block)
        for d in extracted:
            fenced_dicts.append(_unwrap_config_dict(d))
    if fenced_dicts:
        with_keys = [d for d in fenced_dicts if _count_recognized_keys(d) >= 1]
        if with_keys:
            return with_keys[-1]
        return fenced_dicts[-1]

    # 3. Fallback: scan all top-level JSON objects in response (prefer last with recognized keys)
    all_dicts = [_unwrap_config_dict(d) for d in _extract_balanced_dicts(text)]
    if not all_dicts:
        return None
    full_like = [d for d in all_dicts if _count_recognized_keys(d) >= 6]
    if full_like:
        return full_like[-1]
    any_like = [d for d in all_dicts if _count_recognized_keys(d) >= 1]
    if any_like:
        return any_like[-1]
    return all_dicts[-1]


def parse_config_response(text: str) -> ParsedConfig:
    """Strictly parse a 12-key integer Goalpost config, normalizing unambiguous short key names."""
    raw_obj = _select_best_json_candidate(text)
    if raw_obj is None:
        return ParsedConfig(
            parse_ok=False,
            domain_ok=False,
            normalized_keys=False,
            config=None,
            missing_keys=ALL_KEYS,
            extra_keys=(),
            invalid_value_keys={},
            out_of_domain_keys={},
            parse_error="No valid JSON object found in response.",
        )

    normalized_keys = False
    canonical_raw: dict[str, Any] = {}
    extra_keys: list[str] = []
    ambiguous_keys: list[str] = []

    for raw_k, raw_v in raw_obj.items():
        k = str(raw_k).strip()
        if k in ALL_KEYS:
            if k in canonical_raw:
                ambiguous_keys.append(k)
            canonical_raw[k] = raw_v
        elif k in SHORT_TO_CANONICAL_KEY:
            canon = SHORT_TO_CANONICAL_KEY[k]
            if canon in canonical_raw:
                ambiguous_keys.append(canon)
            canonical_raw[canon] = raw_v
            normalized_keys = True
        else:
            extra_keys.append(k)

    if ambiguous_keys:
        return ParsedConfig(
            parse_ok=False,
            domain_ok=False,
            normalized_keys=normalized_keys,
            config=None,
            missing_keys=tuple(k for k in ALL_KEYS if k not in canonical_raw),
            extra_keys=tuple(sorted(extra_keys)),
            invalid_value_keys={},
            out_of_domain_keys={},
            parse_error=f"Ambiguous duplicate short/canonical keys: {sorted(set(ambiguous_keys))}",
        )

    # Note on wire-format normalization: `build_dropout_prompt` illustrates the JSON schema
    # using quoted placeholders (`{"cfg.workers": "<int>", ...}`), causing several models
    # (notably Claude Haiku 5.5, DeepSeek V4.1 Flash, and sometimes Claude Opus 5.5) to quote
    # valid integer answers (e.g., `"6"`). We therefore coerce strict integer strings matching
    # `^-?\d+$` to `int` while strictly rejecting booleans, floats, and non-integer strings,
    # and record `used_string_int_coercion` so native JSON-integer wire validity can be audited.
    parsed_ints: dict[str, int] = {}
    invalid_values: dict[str, str] = {}
    used_str_coercion = False
    for k, v in canonical_raw.items():
        if isinstance(v, (bool, float)):
            invalid_values[k] = repr(v)
        elif isinstance(v, int):
            parsed_ints[k] = v
        elif isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
            parsed_ints[k] = int(v.strip())
            used_str_coercion = True
        else:
            invalid_values[k] = repr(v)

    missing_keys = tuple(k for k in ALL_KEYS if k not in parsed_ints and k not in invalid_values)
    out_of_domain = {k: v for k, v in parsed_ints.items() if v not in DOMAINS[k]}

    parse_ok = len(missing_keys) == 0 and len(invalid_values) == 0
    domain_ok = parse_ok and len(out_of_domain) == 0

    err_parts: list[str] = []
    if missing_keys:
        err_parts.append(f"missing_keys={list(missing_keys)}")
    if invalid_values:
        err_parts.append(f"invalid_values={invalid_values}")
    if parse_ok and out_of_domain:
        err_parts.append(f"out_of_domain={out_of_domain}")
    parse_error = "; ".join(err_parts) if err_parts else None

    return ParsedConfig(
        parse_ok=parse_ok,
        domain_ok=domain_ok,
        normalized_keys=normalized_keys,
        config=dict(parsed_ints) if parse_ok else None,
        missing_keys=missing_keys,
        extra_keys=tuple(sorted(extra_keys)),
        invalid_value_keys=invalid_values,
        out_of_domain_keys=out_of_domain,
        parse_error=parse_error,
        used_string_int_coercion=used_str_coercion,
    )


def grade_dropout_config(case: DropoutCase, response_text: str) -> DropoutGrade:
    """Deterministically grade a single Experiment 1 response against all 15 rules + domains."""
    parsed = parse_config_response(response_text)
    own_ids = case.own_rule_ids()
    partner_ids = case.partner_rule_ids()
    disclosed_map = case.sender_r1_r2_rules_formula
    disclosed_ids = tuple(rid for rid in partner_ids if disclosed_map.get(rid, False))
    undisclosed_ids = tuple(rid for rid in partner_ids if not disclosed_map.get(rid, False))

    if not parsed.parse_ok or parsed.config is None:
        return DropoutGrade(
            parse_ok=False,
            domain_ok=False,
            normalized_keys=parsed.normalized_keys,
            parsed_config=None,
            parse_error=parsed.parse_error,
            missing_keys=parsed.missing_keys,
            out_of_domain_keys=dict(parsed.out_of_domain_keys),
            own_rules_passed=0,
            r3_passed=False,
            own_plus_r3_pass=False,
            own_plus_r3_frac=0.0,
            partner_rules_passed=0,
            partner_rules_pass=False,
            partner_rules_frac=0.0,
            strict_all_pass=False,
            total_rules_passed=0,
            copied_r2_unchanged=False,
            hamming_from_r2=None,
            rule_results={rid: False for rid in ALL_EXP1_RULE_IDS},
            disclosed_partner_rules_passed=None,
            disclosed_partner_rules_total=len(disclosed_ids),
            undisclosed_partner_rules_passed=None,
            undisclosed_partner_rules_total=len(undisclosed_ids),
        )

    cfg = parsed.config
    env = _get_goalpost_env(case.seed)
    r3_rule = case.r3_rule()

    rule_results: dict[str, bool] = {}
    for r in env.base_rules:
        try:
            rule_results[r.rule_id] = bool(r.check(cfg))
        except ZeroDivisionError:
            rule_results[r.rule_id] = False
    rule_results["C-R3"] = bool(r3_rule.check(cfg))

    own_passed = sum(1 for rid in own_ids if rule_results[rid])
    r3_passed = rule_results["C-R3"]
    partner_passed = sum(1 for rid in partner_ids if rule_results[rid])
    total_passed = own_passed + partner_passed + int(r3_passed)

    own_plus_r3_pass = parsed.domain_ok and own_passed == 7 and r3_passed
    partner_rules_pass = parsed.domain_ok and partner_passed == 7
    strict_all_pass = own_plus_r3_pass and partner_rules_pass

    copied_r2 = all(cfg[k] == case.r2_state[k] for k in ALL_KEYS)
    hamming = sum(1 for k in ALL_KEYS if cfg[k] != case.r2_state[k])

    disc_passed = sum(1 for rid in disclosed_ids if rule_results[rid]) if disclosed_ids else None
    undisc_passed = (
        sum(1 for rid in undisclosed_ids if rule_results[rid]) if undisclosed_ids else None
    )

    return DropoutGrade(
        parse_ok=True,
        domain_ok=parsed.domain_ok,
        normalized_keys=parsed.normalized_keys,
        parsed_config=cfg,
        parse_error=parsed.parse_error,
        missing_keys=parsed.missing_keys,
        out_of_domain_keys=dict(parsed.out_of_domain_keys),
        own_rules_passed=own_passed,
        r3_passed=r3_passed,
        own_plus_r3_pass=own_plus_r3_pass,
        own_plus_r3_frac=round((own_passed + int(r3_passed)) / 8.0, 6),
        partner_rules_passed=partner_passed,
        partner_rules_pass=partner_rules_pass,
        partner_rules_frac=round(partner_passed / 7.0, 6),
        strict_all_pass=strict_all_pass,
        total_rules_passed=total_passed,
        copied_r2_unchanged=copied_r2,
        hamming_from_r2=hamming,
        rule_results=rule_results,
        disclosed_partner_rules_passed=disc_passed,
        disclosed_partner_rules_total=len(disclosed_ids),
        undisclosed_partner_rules_passed=undisc_passed,
        undisclosed_partner_rules_total=len(undisclosed_ids),
    )


# ---------------------------------------------------------------------------
# 5. Receiver Manifest Loader & Mock Receiver Roster
# ---------------------------------------------------------------------------


def _parse_effort_ladder(
    raw_ladder: Any,
    default_effort: str,
    default_aa: float | None,
    default_reasoning_payload: dict[str, Any] | None = None,
    tier_reasoning_payloads: dict[str, Any] | None = None,
    aa_index_v5_by_tier: dict[str, Any] | None = None,
    has_explicit_default_payload: bool = False,
) -> tuple[EffortStep, ...]:
    """Parse effort ladder from either dict entries or string tiers + `tier_reasoning_payloads`.

    Preserves explicit omit semantics: when a verified tier maps to `None` inside
    `tier_reasoning_payloads` (or `reasoning_payload: None` with `effort == "omit"`),
    stores `{}` so `OpenRouterClient.complete(..., reasoning_payload_override={})` cleanly
    omits the `"reasoning"` key instead of falling back to `{"enabled": False}` or `{"effort": ...}`.
    """
    payload_map = tier_reasoning_payloads if isinstance(tier_reasoning_payloads, dict) else {}
    aa_map = aa_index_v5_by_tier if isinstance(aa_index_v5_by_tier, dict) else {}

    steps: list[EffortStep] = []
    if isinstance(raw_ladder, list) and raw_ladder:
        for item in raw_ladder:
            if isinstance(item, str):
                eff = item
                if eff in payload_map:
                    raw_rp = payload_map[eff]
                    rp_dict: dict[str, Any] | None = dict(raw_rp) if isinstance(raw_rp, dict) else {}
                else:
                    rp_dict = None
                raw_aa = aa_map.get(eff)
                aa = float(raw_aa) if raw_aa is not None else default_aa
                steps.append(EffortStep(effort=eff, reasoning_payload=rp_dict, aa_index_v5=aa))
            elif isinstance(item, dict):
                eff = str(
                    item.get("effort")
                    or item.get("reasoning_effort")
                    or item.get("label")
                    or default_effort
                )
                if "reasoning_payload" in item:
                    rp = item["reasoning_payload"]
                    if isinstance(rp, dict):
                        rp_dict = dict(rp)
                    elif rp is None and eff == "omit":
                        rp_dict = {}
                    else:
                        rp_dict = None
                elif "reasoning" in item:
                    rp = item["reasoning"]
                    rp_dict = dict(rp) if isinstance(rp, dict) else None
                elif eff in payload_map:
                    raw_rp = payload_map[eff]
                    rp_dict = dict(raw_rp) if isinstance(raw_rp, dict) else {}
                else:
                    rp_dict = None
                raw_aa = item.get("aa_index_v5")
                if raw_aa is None:
                    raw_aa = item.get("aa_v5")
                if raw_aa is None:
                    raw_aa = aa_map.get(eff)
                aa = float(raw_aa) if raw_aa is not None else default_aa
                steps.append(EffortStep(effort=eff, reasoning_payload=rp_dict, aa_index_v5=aa))
    if not steps:
        if default_effort in payload_map:
            raw_rp = payload_map[default_effort]
            fallback_rp: dict[str, Any] | None = dict(raw_rp) if isinstance(raw_rp, dict) else {}
        elif isinstance(default_reasoning_payload, dict):
            fallback_rp = dict(default_reasoning_payload)
        elif has_explicit_default_payload and default_reasoning_payload is None:
            fallback_rp = {}
        else:
            fallback_rp = None
        steps.append(
            EffortStep(
                effort=default_effort,
                reasoning_payload=fallback_rp,
                aa_index_v5=default_aa,
            )
        )
    return tuple(steps)


def _normalize_provider_only(raw: Any, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        vals = tuple(str(x) for x in raw if x)
        return vals or fallback
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    return fallback


def _receiver_from_entry(key: str, entry: dict[str, Any]) -> ReceiverSpec:
    base = DEFAULT_MODELS.get(key)
    slug = str(entry.get("slug") or (base.slug if base else key))
    family = str(
        entry.get("family")
        or (base.family if base else slug.split("/", 1)[0])
    )
    display_name = str(entry.get("display_name") or (base.display_name if base else key))
    raw_tier = entry.get("tier") or entry.get("receiver_class")
    if raw_tier in ("frontier", "worker"):
        tier: Tier = raw_tier
    else:
        tier = "frontier" if key in (*CORE_MODELS, "astra60") else "worker"

    provider_only = _normalize_provider_only(
        entry.get("provider_only") or entry.get("provider") or entry.get("returned_provider"),
        fallback=(base.provider_only if base else ()),
    )
    default_conc = 16 if tier == "frontier" else 24
    concurrency = int(entry.get("concurrency") or default_conc)

    if "supports_anthropic_cache_control" in entry:
        supports_cache = bool(entry["supports_anthropic_cache_control"])
    elif base is not None:
        supports_cache = base.supports_anthropic_cache_control
    else:
        supports_cache = family.lower() == "anthropic" or slug.startswith("anthropic/")

    input_per_m = float(
        entry.get("input_per_m")
        if entry.get("input_per_m") is not None
        else (
            entry.get("fallback_input_per_m")
            if entry.get("fallback_input_per_m") is not None
            else (base.fallback_input_per_m if base else 0.0)
        )
    )
    output_per_m = float(
        entry.get("output_per_m")
        if entry.get("output_per_m") is not None
        else (
            entry.get("fallback_output_per_m")
            if entry.get("fallback_output_per_m") is not None
            else (base.fallback_output_per_m if base else 0.0)
        )
    )
    max_tokens = int(entry.get("max_tokens") or DEFAULT_DROPOUT_MAX_TOKENS)
    default_effort = str(
        entry.get("default_effort")
        or entry.get("starting_tier")
        or entry.get("initial_effort")
        or entry.get("reasoning_effort")
        or (base.reasoning_effort if base and base.reasoning_effort else "medium")
    )
    raw_aa = entry.get("aa_index_v5")
    if raw_aa is None:
        raw_aa = entry.get("aa_v5")
    if raw_aa is None and isinstance(entry.get("aa_index_v5_by_tier"), dict):
        raw_aa = entry["aa_index_v5_by_tier"].get(default_effort)
    if raw_aa is None and base is not None:
        raw_aa = base.aa_index_v5
    aa_val = float(raw_aa) if raw_aa is not None else None

    has_explicit_rp = "reasoning_payload" in entry
    default_rp = entry.get("reasoning_payload")
    ladder = _parse_effort_ladder(
        entry.get("effort_ladder"),
        default_effort=default_effort,
        default_aa=aa_val,
        default_reasoning_payload=default_rp if isinstance(default_rp, dict) else None,
        tier_reasoning_payloads=entry.get("tier_reasoning_payloads"),
        aa_index_v5_by_tier=entry.get("aa_index_v5_by_tier"),
        has_explicit_default_payload=has_explicit_rp,
    )
    return ReceiverSpec(
        key=key,
        slug=slug,
        family=family,
        display_name=display_name,
        tier=tier,
        provider_only=provider_only,
        concurrency=concurrency,
        supports_anthropic_cache_control=supports_cache,
        input_per_m=input_per_m,
        output_per_m=output_per_m,
        max_tokens=max_tokens,
        default_effort=default_effort,
        effort_ladder=ladder,
        aa_index_v5=aa_val,
        notes=str(entry.get("notes") or entry.get("aa_index_v5_notes") or (base.notes if base else "")),
    )


def build_default_mock_receivers() -> dict[str, ReceiverSpec]:
    """Build the canonical 8-receiver v2.1 roster (4 frontier + 4 multi-lab workers) for `--mock`."""
    receivers: dict[str, ReceiverSpec] = {}
    for key in ("astra62", "neon4", "opus55", "astra60"):
        base = DEFAULT_MODELS[key]
        receivers[key] = ReceiverSpec(
            key=base.key,
            slug=base.slug,
            family=base.family,
            display_name=base.display_name,
            tier="frontier",
            provider_only=base.provider_only,
            concurrency=16,
            supports_anthropic_cache_control=base.supports_anthropic_cache_control,
            input_per_m=base.fallback_input_per_m,
            output_per_m=base.fallback_output_per_m,
            max_tokens=DEFAULT_DROPOUT_MAX_TOKENS,
            default_effort="medium",
            effort_ladder=(
                EffortStep(
                    effort="medium",
                    reasoning_payload={"effort": "medium"},
                    aa_index_v5=base.aa_index_v5,
                ),
            ),
            aa_index_v5=base.aa_index_v5,
            notes="Mock frontier receiver locked at medium.",
        )

    mock_workers: tuple[dict[str, Any], ...] = (
        {
            "key": "hydrogen",
            "slug": "google/gemini-4-hydrogen-mock",
            "family": "google",
            "display_name": "Gemini 4 Hydrogen (Mock)",
            "provider_only": ["Google AI Studio"],
            "concurrency": 24,
            "input_per_m": 0.25,
            "output_per_m": 1.00,
            "default_effort": "low",
            "effort_ladder": [
                {"effort": "low", "reasoning_payload": {"effort": "low"}, "aa_index_v5": 27.0},
                {"effort": "medium", "reasoning_payload": {"effort": "medium"}, "aa_index_v5": 41.0},
                {"effort": "high", "reasoning_payload": {"effort": "high"}, "aa_index_v5": 51.0},
            ],
        },
        {
            "key": "haiku55",
            "slug": "anthropic/claude-haiku-5.5-mock",
            "family": "anthropic",
            "display_name": "Claude Haiku 5.5 (Mock)",
            "provider_only": ["Anthropic"],
            "concurrency": 24,
            "supports_anthropic_cache_control": True,
            "input_per_m": 0.80,
            "output_per_m": 4.00,
            "default_effort": "low",
            "effort_ladder": [
                {"effort": "low", "reasoning_payload": {"effort": "low"}, "aa_index_v5": 19.0},
                {"effort": "medium", "reasoning_payload": {"effort": "medium"}, "aa_index_v5": 25.0},
                {"effort": "high", "reasoning_payload": {"effort": "high"}, "aa_index_v5": 30.0},
            ],
        },
        {
            "key": "deepseek_flash",
            "slug": "deepseek/deepseek-v4.1-flash-mock",
            "family": "deepseek",
            "display_name": "DeepSeek V4.1 Flash (Mock)",
            "provider_only": ["DeepSeek"],
            "concurrency": 32,
            "input_per_m": 0.14,
            "output_per_m": 0.28,
            "default_effort": "disabled",
            "effort_ladder": [
                {"effort": "disabled", "reasoning_payload": {"enabled": False}, "aa_index_v5": 19.0},
                {"effort": "low", "reasoning_payload": {"effort": "low"}, "aa_index_v5": 25.0},
                {"effort": "medium", "reasoning_payload": {"effort": "medium"}, "aa_index_v5": 30.0},
                {"effort": "high", "reasoning_payload": {"effort": "high"}, "aa_index_v5": 33.0},
            ],
        },
        {
            "key": "gpt6_luna",
            "slug": "openai/gpt-6-luna-mock",
            "family": "openai",
            "display_name": "GPT-6 Luna (Mock)",
            "provider_only": ["OpenAI"],
            "concurrency": 24,
            "input_per_m": 0.40,
            "output_per_m": 1.60,
            "default_effort": "low",
            "effort_ladder": [
                {"effort": "low", "reasoning_payload": {"effort": "low"}, "aa_index_v5": 17.0},
                {"effort": "medium", "reasoning_payload": {"effort": "medium"}, "aa_index_v5": 18.0},
                {"effort": "high", "reasoning_payload": {"effort": "high"}, "aa_index_v5": 28.0},
            ],
        },
    )
    for w in mock_workers:
        w_spec = _receiver_from_entry(w["key"], {**w, "tier": "worker"})
        receivers[w_spec.key] = w_spec
    return receivers


def load_receiver_manifest(
    manifest_path: Path | None = None,
    *,
    mock: bool = False,
) -> dict[str, ReceiverSpec]:
    """Load Experiment 1 receiver specs from `probe_manifest_v2_1.json`.

    Accepts both the primary `receivers` schema and the `verified_pins` schema generated by
    `scripts/probe_workers_v2_1.py` (with `receiver_order`, `receiver_class`, `starting_tier`,
    `tier_reasoning_payloads`, and `aa_index_v5_by_tier`).
    """
    path = manifest_path or DEFAULT_PROBE_MANIFEST_V2_1_PATH
    if not path.exists():
        if mock:
            return build_default_mock_receivers()
        raise FileNotFoundError(
            f"Required v2.1 probe manifest missing: {path}. "
            "Run scripts/probe_workers_v2_1.py or pass --probe-manifest (or use --mock)."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_receivers: dict[str, dict[str, Any]] = {}

    if isinstance(data.get("receivers"), dict) and data["receivers"]:
        for k, v in data["receivers"].items():
            if isinstance(v, dict):
                raw_receivers[str(k)] = v
    else:
        # Fallback: assemble from `verified_pins` + optional `frontier_receivers`/`workers`
        pins = data.get("verified_pins")
        if isinstance(pins, dict):
            for k, v in pins.items():
                if isinstance(v, dict):
                    raw_receivers[str(k)] = dict(v)
        for section, default_tier in (
            ("frontier_receivers", "frontier"),
            ("frontier", "frontier"),
            ("worker_receivers", "worker"),
            ("workers", "worker"),
        ):
            sec = data.get(section)
            if isinstance(sec, dict):
                for k, v in sec.items():
                    if isinstance(v, dict):
                        merged = dict(raw_receivers.get(str(k), {}))
                        merged.update(v)
                        merged.setdefault("tier", default_tier)
                        raw_receivers[str(k)] = merged

    if not raw_receivers:
        raise ValueError(f"No receivers found in probe manifest: {path}")

    order_hint = data.get("receiver_order")
    if isinstance(order_hint, list) and order_hint:
        ordered_keys = [str(k) for k in order_hint if str(k) in raw_receivers]
        ordered_keys.extend(k for k in raw_receivers if k not in ordered_keys)
    else:
        ordered_keys = list(raw_receivers.keys())

    return {k: _receiver_from_entry(k, raw_receivers[k]) for k in ordered_keys}


def load_locked_roster(roster_path: Path) -> dict[str, LockedReceiverSpec]:
    """Load post-gate locked receivers from `locked_receiver_roster.json`."""
    if not roster_path.exists():
        raise FileNotFoundError(
            f"Locked receiver roster not found: {roster_path}. Run the gate (--gate-only) first."
        )
    data = json.loads(roster_path.read_text(encoding="utf-8"))
    raw = data.get("receivers") if isinstance(data, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid or empty locked_receiver_roster.json at {roster_path}")
    return {k: LockedReceiverSpec.from_dict(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 6. Deterministic Zero-Cost Mock Client for Dropout Experiments & Tests
# ---------------------------------------------------------------------------


def _find_own_pass_partner_fail_config(case: DropoutCase, rng: random.Random) -> dict[str, int]:
    """Find a domain-valid config passing own 7 + C-R3 while failing >=1 hidden partner rule."""
    env = _get_goalpost_env(case.seed)
    r3_rule = case.r3_rule()
    own_rules = [r for r in env.base_rules if r.visible_to == case.receiver_role]
    partner_rules = [r for r in env.base_rules if r.visible_to == case.dropped_role]

    # Start from a real 15/15 solution and perturb 1-2 keys to keep own+R3 green and break partner
    sols = case.valid_solutions()
    base = _tuple_to_state(sols[rng.randrange(len(sols))])
    keys_order = list(ALL_KEYS)
    rng.shuffle(keys_order)

    # Try 1-key perturbations first, then 2-key perturbations
    for k1 in keys_order:
        for v1 in DOMAINS[k1]:
            if v1 == base[k1]:
                continue
            cand = dict(base)
            cand[k1] = v1
            if (
                r3_rule.check(cand)
                and all(r.check(cand) for r in own_rules)
                and not all(r.check(cand) for r in partner_rules)
            ):
                return cand

    for i, k1 in enumerate(keys_order[:6]):
        for k2 in keys_order[i + 1 :]:
            for v1 in DOMAINS[k1]:
                for v2 in DOMAINS[k2]:
                    cand = dict(base)
                    cand[k1] = v1
                    cand[k2] = v2
                    if (
                        r3_rule.check(cand)
                        and all(r.check(cand) for r in own_rules)
                        and not all(r.check(cand) for r in partner_rules)
                    ):
                        return cand

    # Fallback: perturb r2_state on the two C-R3 keys only
    cand = dict(case.r2_state)
    ka, kb = ALL_KEYS[case.r3_ia], ALL_KEYS[case.r3_ib]
    cand[ka] = next(v for v in DOMAINS[ka] if v != case.r3_va)
    cand[kb] = next(v for v in DOMAINS[kb] if v != case.r3_vb)
    return cand


def _find_own_fail_config(case: DropoutCase, rng: random.Random) -> dict[str, int]:
    """Return a domain-valid config that fails at least one own+R3 rule (not identical to r2)."""
    sols = case.valid_solutions()
    cand = dict(_tuple_to_state(sols[rng.randrange(len(sols))]))
    # Flip a key to violate C-R3 while remaining distinct from r2_state
    ka = ALL_KEYS[case.r3_ia]
    cand[ka] = case.r3_va
    if cand == case.r2_state:
        kb = ALL_KEYS[case.r3_ib]
        cand[kb] = next(v for v in DOMAINS[kb] if v != case.r2_state[kb])
    return cand


class MockDropoutClient(MockOpenRouterClient):
    """Deterministic zero-cost mock client tailored for Experiment 1 gate & handoff runs."""

    def __init__(
        self,
        ledger: Ledger,
        models: dict[str, ModelSpec],
        cases_by_id: dict[str, DropoutCase],
        *,
        ceiling_pass_rates: dict[tuple[str, str], float] | None = None,
        floor_pass_rates: dict[tuple[str, str], float] | None = None,
        custom_responder: Callable[[str, str, dict[str, Any]], str] | None = None,
    ) -> None:
        super().__init__(ledger, models)
        self.cases_by_id = cases_by_id
        self.ceiling_pass_rates = dict(ceiling_pass_rates or {})
        self.floor_pass_rates = dict(floor_pass_rates or {})
        self.custom_responder = custom_responder

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

        meta = dict(metadata or {})
        if meta.get("stage") != "exp1_dropout":
            return await super().complete(
                call_id=call_id,
                model_key=model_key,
                messages=messages,
                metadata=metadata,
                reasoning_effort_override=reasoning_effort_override,
                reasoning_payload_override=reasoning_payload_override,
                max_tokens_override=max_tokens_override,
                cache_prefix_index=cache_prefix_index,
                require_parameters=require_parameters,
            )

        spec = self.models[model_key]
        effort = (
            reasoning_effort_override
            if reasoning_effort_override is not None
            else (spec.reasoning_effort or "medium")
        )
        rng = random.Random(int(hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:12], 16))

        p_tok = sum(len(str(m.get("content", ""))) // 4 for m in messages) + 32
        mock_worst = (
            p_tok * spec.fallback_input_per_m + 600 * spec.fallback_output_per_m
        ) / 1_000_000.0
        res_id = f"mock_exp1:{call_id}:{hashlib.sha256(call_id.encode()).hexdigest()[:8]}"
        await self.ledger.reserve_budget(res_id, max(0.0001, mock_worst))

        try:
            if self.custom_responder is not None:
                response_text = self.custom_responder(call_id, model_key, meta)
            else:
                response_text = self._mock_dropout_response(model_key, effort, meta, rng)

            c_tok = max(40, len(response_text) // 4)
            if effort in ("disabled", "none", "omit"):
                r_tok = 0
            elif effort == "low":
                r_tok = rng.randint(80, 220)
            else:
                r_tok = rng.randint(250, 650)
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
                reasoning_effort=effort,
                max_tokens=max_tokens_override or spec.max_tokens,
                status="ok",
                finish_reason="stop",
                attempt=1,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                reasoning_tokens=r_tok,
                cached_tokens=int(p_tok * 0.25) if cache_prefix_index is not None else 0,
                cost_usd=cost,
                latency_ms=rng.randint(8, 35),
                request_messages=messages,
                response_text=response_text,
                reasoning_text="Mock dropout CSP check." if r_tok > 0 else None,
                metadata=meta,
            )
            await self.ledger.record(entry, reservation_id=res_id)
            return entry
        finally:
            await self.ledger.release_reservation(res_id)

    def _mock_dropout_response(
        self,
        model_key: str,
        effort: str,
        meta: dict[str, Any],
        rng: random.Random,
    ) -> str:
        case = self.cases_by_id[str(meta["case_id"])]
        condition = str(meta["condition"])
        tier = str(meta.get("receiver_tier") or ("frontier" if model_key in CORE_MODELS else "worker"))
        sols = case.valid_solutions()

        if condition == "floor_no_partner":
            p_pass = self.floor_pass_rates.get((model_key, effort), 0.06)
            if rng.random() < p_pass:
                cfg = _tuple_to_state(sols[rng.randrange(len(sols))])
            else:
                cfg = _find_own_pass_partner_fail_config(case, rng)
            return f"Attempting blind takeover from own rules and C-R3.\nCONFIG: {json.dumps(cfg)}"

        if condition == "ceiling_verbatim_rules":
            if (model_key, effort) in self.ceiling_pass_rates:
                p_pass = self.ceiling_pass_rates[(model_key, effort)]
            elif tier == "frontier":
                p_pass = 0.94
            else:
                # Default mock worker: `deepseek_flash` starts below 40% at disabled so mock gate
                # naturally exercises the automatic worker step-up rule!
                if model_key in ("deepseek_flash", "worker_qwen") and effort in ("disabled", "none"):
                    p_pass = 0.18
                elif effort in ("disabled", "none"):
                    p_pass = 0.62
                elif effort == "low":
                    p_pass = 0.78
                else:
                    p_pass = 0.88
            if rng.random() < p_pass:
                cfg = _tuple_to_state(sols[rng.randrange(len(sols))])
            else:
                cfg = _find_own_fail_config(case, rng)
            return f"Solved 15/15 verbatim constraints.\nCONFIG: {json.dumps(cfg)}"

        # Real "handoff" condition
        n_disc = case.sender_r1_r2_n_formula
        base_csp = 0.94 if tier == "frontier" else (0.76 if effort != "medium" else 0.85)
        # Slight bloat penalty on workers for very long transcripts
        if tier == "worker" and case.sender_r1_r2_word_count > 350:
            base_csp -= 0.06
        if rng.random() >= base_csp:
            cfg = _find_own_fail_config(case, rng)
            return f"Emergency update attempt.\nCONFIG: {json.dumps(cfg)}"

        # Passed own+R3; partner pass scales steeply with sender R1-R2 disclosure
        p_partner = 0.08 + 0.84 * (n_disc / 7.0)
        if tier == "worker":
            p_partner = max(0.05, p_partner - 0.07)
        if rng.random() < p_partner:
            cfg = _tuple_to_state(sols[rng.randrange(len(sols))])
        else:
            cfg = _find_own_pass_partner_fail_config(case, rng)
        return f"Emergency takeover config computed from R1-R2 handoff.\nCONFIG: {json.dumps(cfg)}"


# ---------------------------------------------------------------------------
# 7. Gate Execution Engine (`run_dropout_gate`)
# ---------------------------------------------------------------------------


def _is_truncation_error(record: dict[str, Any]) -> bool:
    """Return True if a non-ok record failed due to output/reasoning length truncation."""
    if record.get("status") == "ok":
        return False
    if str(record.get("finish_reason") or "").lower() == "length":
        return True
    err = str(record.get("error_detail") or "").lower()
    return "finish_reason=length" in err or "truncated" in err or "length" in err


def _aggregate_grade_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "n_cases": 0,
            "n_ok": 0,
            "n_errors": 0,
            "n_truncation_errors": 0,
            "parse_ok_rate": 0.0,
            "domain_ok_rate": 0.0,
            "own_plus_r3_pass_rate": 0.0,
            "partner_rules_pass_rate": 0.0,
            "partner_pass_given_own_pass": None,
            "n_own_plus_r3_pass": 0,
            "strict_all_pass_rate": 0.0,
            "mean_own_plus_r3_frac": 0.0,
            "mean_partner_rules_frac": 0.0,
            "parse_ok_rate_ok": None,
            "domain_ok_rate_ok": None,
            "own_plus_r3_pass_rate_ok": None,
            "partner_rules_pass_rate_ok": None,
            "strict_all_pass_rate_ok": None,
            "mean_own_plus_r3_frac_ok": None,
            "mean_partner_rules_frac_ok": None,
            "copied_r2_rate": 0.0,
            "cost_usd": 0.0,
            "mean_prompt_tokens": None,
            "mean_completion_tokens": None,
            "mean_reasoning_tokens": None,
            "mean_latency_ms": None,
        }

    ok_recs = [r for r in records if r.get("status") == "ok"]
    n_ok = len(ok_recs)
    n_errors = n - n_ok
    n_truncation_errors = sum(1 for r in records if _is_truncation_error(r))

    # ITT (all `n_cases` denominator)
    parse_ok = sum(1 for r in records if r.get("parse_ok"))
    domain_ok = sum(1 for r in records if r.get("domain_ok"))
    own_r3 = sum(1 for r in records if r.get("own_plus_r3_pass"))
    partner_ok = sum(1 for r in records if r.get("partner_rules_pass"))
    strict_ok = sum(1 for r in records if r.get("strict_all_pass"))
    copied_r2 = sum(1 for r in records if r.get("copied_r2_unchanged"))
    partner_given_own_num = sum(
        1 for r in records if r.get("own_plus_r3_pass") and r.get("partner_rules_pass")
    )

    # Completed-only (`n_ok` denominator)
    parse_ok_comp = sum(1 for r in ok_recs if r.get("parse_ok"))
    domain_ok_comp = sum(1 for r in ok_recs if r.get("domain_ok"))
    own_r3_comp = sum(1 for r in ok_recs if r.get("own_plus_r3_pass"))
    partner_ok_comp = sum(1 for r in ok_recs if r.get("partner_rules_pass"))
    strict_ok_comp = sum(1 for r in ok_recs if r.get("strict_all_pass"))

    rtoks = [r["reasoning_tokens"] for r in ok_recs if r.get("reasoning_tokens") is not None]
    return {
        "n_cases": n,
        "n_ok": n_ok,
        "n_errors": n_errors,
        "n_truncation_errors": n_truncation_errors,
        "parse_ok_rate": round(parse_ok / n, 6),
        "domain_ok_rate": round(domain_ok / n, 6),
        "own_plus_r3_pass_rate": round(own_r3 / n, 6),
        "partner_rules_pass_rate": round(partner_ok / n, 6),
        "partner_pass_given_own_pass": (
            round(partner_given_own_num / own_r3, 6) if own_r3 > 0 else None
        ),
        "n_own_plus_r3_pass": own_r3,
        "strict_all_pass_rate": round(strict_ok / n, 6),
        "mean_own_plus_r3_frac": round(
            sum(float(r.get("own_plus_r3_frac", 0.0)) for r in records) / n, 6
        ),
        "mean_partner_rules_frac": round(
            sum(float(r.get("partner_rules_frac", 0.0)) for r in records) / n, 6
        ),
        "parse_ok_rate_ok": round(parse_ok_comp / n_ok, 6) if n_ok > 0 else None,
        "domain_ok_rate_ok": round(domain_ok_comp / n_ok, 6) if n_ok > 0 else None,
        "own_plus_r3_pass_rate_ok": round(own_r3_comp / n_ok, 6) if n_ok > 0 else None,
        "partner_rules_pass_rate_ok": round(partner_ok_comp / n_ok, 6) if n_ok > 0 else None,
        "strict_all_pass_rate_ok": round(strict_ok_comp / n_ok, 6) if n_ok > 0 else None,
        "mean_own_plus_r3_frac_ok": (
            round(sum(float(r.get("own_plus_r3_frac", 0.0)) for r in ok_recs) / n_ok, 6)
            if n_ok > 0
            else None
        ),
        "mean_partner_rules_frac_ok": (
            round(sum(float(r.get("partner_rules_frac", 0.0)) for r in ok_recs) / n_ok, 6)
            if n_ok > 0
            else None
        ),
        "copied_r2_rate": round(copied_r2 / n, 6),
        "cost_usd": round(sum(float(r.get("cost_usd", 0.0)) for r in records), 6),
        "mean_prompt_tokens": (
            round(sum(int(r.get("prompt_tokens", 0)) for r in ok_recs) / n_ok, 2)
            if n_ok > 0
            else None
        ),
        "mean_completion_tokens": (
            round(sum(int(r.get("completion_tokens", 0)) for r in ok_recs) / n_ok, 2)
            if n_ok > 0
            else None
        ),
        "mean_reasoning_tokens": round(sum(rtoks) / len(rtoks), 2) if rtoks else None,
        "mean_latency_ms": (
            round(sum(int(r.get("latency_ms", 0)) for r in ok_recs) / n_ok, 1)
            if n_ok > 0
            else None
        ),
    }


async def _evaluate_single_call(
    client: OpenRouterClient | MockOpenRouterClient,
    receiver_key: str,
    receiver_slug: str,
    receiver_family: str,
    receiver_display_name: str,
    receiver_tier: Tier,
    supports_cache: bool,
    max_tokens: int,
    effort: str,
    reasoning_payload: dict[str, Any] | None,
    aa_index_v5: float | None,
    case: DropoutCase,
    condition: Condition,
    call_id: str,
) -> dict[str, Any]:
    messages = build_dropout_prompt(case, condition)
    metadata = {
        "stage": "exp1_dropout",
        "condition": condition,
        "case_id": case.case_id,
        "episode_id": case.episode_id,
        "seed": case.seed,
        "dropped_role": case.dropped_role,
        "receiver_role": case.receiver_role,
        "sender_model": case.sender_model,
        "receiver_tier": receiver_tier,
        "effort": effort,
    }
    try:
        entry = await client.complete(
            call_id=call_id,
            model_key=receiver_key,
            messages=messages,
            metadata=metadata,
            reasoning_effort_override=effort,
            reasoning_payload_override=reasoning_payload,
            max_tokens_override=max_tokens,
            cache_prefix_index=0 if supports_cache else None,
        )
        grade = grade_dropout_config(case, entry.response_text)
        status = "ok"
        err_detail = None
        returned_model = entry.returned_model
        provider = entry.provider
        finish_reason = entry.finish_reason
        p_tok = entry.prompt_tokens
        c_tok = entry.completion_tokens
        r_tok = entry.reasoning_tokens
        cached_tok = entry.cached_tokens
        cost_usd = entry.cost_usd
        latency_ms = entry.latency_ms
    except BudgetExceededError:
        raise
    except OpenRouterCallError as exc:
        grade = grade_dropout_config(case, "")
        status = "error"
        err_detail = str(exc)
        returned_model = None
        provider = None
        finish_reason = None
        p_tok = 0
        c_tok = 0
        r_tok = None
        cached_tok = 0
        cost_usd = 0.0
        latency_ms = 0

    return {
        "call_id": call_id,
        "condition": condition,
        "case_id": case.case_id,
        "episode_id": case.episode_id,
        "seed": case.seed,
        "arm": case.arm,
        "dropped_role": case.dropped_role,
        "receiver_role": case.receiver_role,
        "is_primary_unconfounded": case.is_primary_unconfounded,
        "sender_model": case.sender_model,
        "sender_family": case.sender_family,
        "original_receiver_model": case.original_receiver_model,
        "original_receiver_family": case.original_receiver_family,
        "receiver_key": receiver_key,
        "receiver_slug": receiver_slug,
        "receiver_family": receiver_family,
        "receiver_display_name": receiver_display_name,
        "receiver_tier": receiver_tier,
        "receiver_effort": effort,
        "receiver_aa_index_v5": aa_index_v5,
        "is_same_family_as_sender": (receiver_family == case.sender_family),
        "is_cross_vendor": (receiver_family != case.sender_family),
        "is_same_model_as_original_receiver": (receiver_key == case.original_receiver_model),
        "episode_score": case.episode_score,
        "r2_all_pass": case.r2_all_pass,
        "sender_r1_n_formula": case.sender_r1_n_formula,
        "sender_r1_full_disclosure": case.sender_r1_full_disclosure,
        "sender_r1_r2_n_formula": case.sender_r1_r2_n_formula,
        "sender_r1_r2_full_disclosure": case.sender_r1_r2_full_disclosure,
        "sender_r1_n_id": case.sender_r1_n_id,
        "sender_r1_r2_n_id": case.sender_r1_r2_n_id,
        "sender_r1_word_count": case.sender_r1_word_count,
        "sender_r1_r2_word_count": case.sender_r1_r2_word_count,
        "original_receiver_asked_rules_r1": case.original_receiver_asked_rules_r1,
        "original_receiver_asked_rules": case.original_receiver_asked_rules,
        "status": status,
        "error_detail": err_detail,
        "returned_model": returned_model,
        "provider": provider,
        "finish_reason": finish_reason,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "reasoning_tokens": r_tok,
        "cached_tokens": cached_tok,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        **grade.to_dict(),
    }


def _render_gate_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment 1 (Partner Drops Out) — Floor-vs.-Ceiling Gate Report",
        "",
        f"- **Timestamp (UTC):** `{report['timestamp_utc']}`",
        f"- **Mode:** `{'MOCK' if report['mock'] else 'LIVE'}`",
        f"- **Gate Cases:** `{report['n_gate_cases']}` (`4` per seed × `4` seeds, `8` Role A + `8` Role B)",
        f"- **Worker Auto Step-Up Threshold:** `ceiling_strict_pass_rate >= {report['ceiling_threshold']:.0%}` (max `{report['max_step_ups']}` step-ups; frontier locked at `medium`)",
        f"- **Total Gate Cost:** `${report['total_gate_cost_usd']:.4f}` (ledger cumulative: `${report['ledger_total_cost_usd']:.4f}`)",
        "",
        "## Locked Receiver Roster Summary",
        "",
        "| Receiver | Tier | Family | Effort (Init → Locked) | Step-Ups | AA v5 | Floor Strict | Ceiling Strict | Ceiling Own+R3 | Ceiling Partner | Δ (Ceil−Floor) | Gate Status | Cost ($) |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |",
    ]

    for key, rec in report["receivers"].items():
        locked = rec["locked"]
        aa_str = f"{locked['locked_aa_index_v5']:.1f}" if locked["locked_aa_index_v5"] is not None else "—"
        eff_str = (
            f"`{locked['initial_effort']}` → `{locked['locked_effort']}`"
            if locked["step_ups_performed"] > 0
            else f"`{locked['locked_effort']}`"
        )
        delta_pp = (
            locked["gate_ceiling_strict_pass_rate"] - locked["gate_floor_strict_pass_rate"]
        ) * 100.0
        status_str = "PASS" if locked["gate_passed"] else "BELOW_THRESH"
        if locked["tier"] == "frontier":
            status_str = "LOCKED_FRONTIER"
        lines.append(
            f"| `{key}` | {locked['tier']} | {locked['family']} | {eff_str} | "
            f"{locked['step_ups_performed']} | {aa_str} | "
            f"{locked['gate_floor_strict_pass_rate'] * 100:.1f}% | "
            f"{locked['gate_ceiling_strict_pass_rate'] * 100:.1f}% | "
            f"{locked['gate_ceiling_own_plus_r3_pass_rate'] * 100:.1f}% | "
            f"{locked['gate_ceiling_partner_pass_rate'] * 100:.1f}% | "
            f"{delta_pp:+.1f} pp | {status_str} | {rec['total_cost_usd']:.4f} |"
        )

    lines.extend(["", "## Step-Up Audit Trail", ""])
    for key, rec in report["receivers"].items():
        lines.append(f"### `{key}` ({rec['locked']['display_name']} — `{rec['locked']['tier']}`)")
        for attempt in rec["step_up_history"]:
            fl = attempt["floor"]
            cl = attempt["ceiling"]
            lines.append(
                f"- **Step {attempt['step_index']} (`{attempt['effort']}`)**: "
                f"Floor strict `{fl['strict_all_pass_rate'] * 100:.1f}%` (own+R3 `{fl['own_plus_r3_pass_rate'] * 100:.1f}%`), "
                f"Ceiling strict `{cl['strict_all_pass_rate'] * 100:.1f}%` (own+R3 `{cl['own_plus_r3_pass_rate'] * 100:.1f}%`, "
                f"partner `{cl['partner_rules_pass_rate'] * 100:.1f}%`), "
                f"Δ `{attempt['ceiling_minus_floor_strict_pp']:+.1f} pp` — {attempt['decision']}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


async def run_dropout_gate(
    *,
    episodes_dir: Path = DEFAULT_EPISODES_DIR,
    out_dir: Path = DEFAULT_EXP1_OUT_DIR,
    probe_manifest_path: Path | None = None,
    receivers: dict[str, ReceiverSpec] | None = None,
    gate_cases: Sequence[DropoutCase] | None = None,
    budget_cap_usd: float = DEFAULT_GATE_BUDGET_CAP_USD,
    ceiling_threshold: float = DEFAULT_GATE_CEILING_THRESHOLD,
    max_step_ups: int = DEFAULT_MAX_STEP_UPS,
    mock: bool = False,
    mock_ceiling_pass_rates: dict[tuple[str, str], float] | None = None,
    mock_floor_pass_rates: dict[tuple[str, str], float] | None = None,
    client_override: OpenRouterClient | MockOpenRouterClient | None = None,
) -> dict[str, Any]:
    """Run the 16-case Floor-vs.-Ceiling Gate across all receivers in parallel with auto step-up.

    Automatic Step-Up Policy:
      - Frontier receivers (`tier == "frontier"`) stay strictly locked at `default_effort` (`medium`).
      - Worker receivers (`tier == "worker"`) scoring `ceiling_strict_pass_rate < ceiling_threshold`
        (default `< 0.40`) automatically step up along their `effort_ladder` and re-run both
        `floor_no_partner` and `ceiling_verbatim_rules` (up to `max_step_ups`, default `2`).

    Writes:
      - `<out_dir>/ledger.jsonl` (crash-resumable API call ledger)
      - `<out_dir>/gate_report.json`
      - `<out_dir>/gate_report.md`
      - `<out_dir>/locked_receiver_roster.json`
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if gate_cases is None:
        all_cases = build_dropout_cases(episodes_dir)
        selected_gate_cases = select_gate_cases(all_cases, n_cases=16)
    else:
        selected_gate_cases = list(gate_cases)
    cases_by_id = {c.case_id: c for c in selected_gate_cases}

    rec_map = (
        dict(receivers)
        if receivers is not None
        else load_receiver_manifest(probe_manifest_path, mock=mock)
    )
    model_specs = {k: r.to_model_spec() for k, r in rec_map.items()}
    total_conc = sum(r.concurrency for r in rec_map.values())

    ledger = Ledger(out_path / "ledger.jsonl", budget_cap_usd=budget_cap_usd)
    owns_client = client_override is None
    if client_override is not None:
        client = client_override
    elif mock:
        client = MockDropoutClient(
            ledger,
            model_specs,
            cases_by_id,
            ceiling_pass_rates=mock_ceiling_pass_rates,
            floor_pass_rates=mock_floor_pass_rates,
        )
    else:
        client = OpenRouterClient(
            ledger,
            model_specs,
            max_connections=max(160, total_conc + 16),
            max_keepalive_connections=max(80, total_conc // 2),
        )

    async def _run_receiver_gate(rec: ReceiverSpec) -> tuple[str, dict[str, Any], LockedReceiverSpec]:
        ladder = rec.resolved_ladder()
        start_idx = rec.initial_step_index()
        cur_idx = start_idx
        step_ups_used = 0
        history: list[dict[str, Any]] = []
        total_rec_cost = 0.0

        while True:
            step = ladder[cur_idx]
            coros = []
            for cond in ("floor_no_partner", "ceiling_verbatim_rules"):
                for case in selected_gate_cases:
                    cid = f"exp1_gate:{rec.key}:step{cur_idx}_{step.effort}:{cond}:{case.case_id}"
                    coros.append(
                        _evaluate_single_call(
                            client=client,
                            receiver_key=rec.key,
                            receiver_slug=rec.slug,
                            receiver_family=rec.family,
                            receiver_display_name=rec.display_name,
                            receiver_tier=rec.tier,
                            supports_cache=rec.supports_anthropic_cache_control,
                            max_tokens=rec.max_tokens,
                            effort=step.effort,
                            reasoning_payload=step.reasoning_payload,
                            aa_index_v5=step.aa_index_v5 if step.aa_index_v5 is not None else rec.aa_index_v5,
                            case=case,
                            condition=cond,
                            call_id=cid,
                        )
                    )

            step_records = await asyncio.gather(*coros)
            floor_recs = [r for r in step_records if r["condition"] == "floor_no_partner"]
            ceil_recs = [r for r in step_records if r["condition"] == "ceiling_verbatim_rules"]
            floor_agg = _aggregate_grade_records(floor_recs)
            ceil_agg = _aggregate_grade_records(ceil_recs)
            step_cost = round(floor_agg["cost_usd"] + ceil_agg["cost_usd"], 6)
            total_rec_cost = round(total_rec_cost + step_cost, 6)

            ceil_strict = float(ceil_agg["strict_all_pass_rate"])
            floor_strict = float(floor_agg["strict_all_pass_rate"])
            passed = ceil_strict >= ceiling_threshold

            can_step_up = (
                rec.tier == "worker"
                and not passed
                and step_ups_used < max_step_ups
                and (cur_idx + 1) < len(ladder)
            )
            if rec.tier == "frontier":
                decision = "locked_frontier"
            elif passed:
                decision = "passed_and_locked"
            elif can_step_up:
                decision = f"step_up_to_{ladder[cur_idx + 1].effort}"
            else:
                decision = "exhausted_ladder_below_threshold"

            history.append(
                {
                    "step_index": cur_idx,
                    "effort": step.effort,
                    "reasoning_payload": (
                        dict(step.reasoning_payload)
                        if step.reasoning_payload is not None
                        else None
                    ),
                    "aa_index_v5": (
                        step.aa_index_v5 if step.aa_index_v5 is not None else rec.aa_index_v5
                    ),
                    "floor": floor_agg,
                    "ceiling": ceil_agg,
                    "ceiling_minus_floor_strict_pp": round((ceil_strict - floor_strict) * 100.0, 2),
                    "passed_gate": passed,
                    "decision": decision,
                    "step_cost_usd": step_cost,
                    "records": step_records,
                }
            )

            if can_step_up:
                cur_idx += 1
                step_ups_used += 1
                continue
            break

        # Choose the passing step (last if passed), or highest ceiling strict pass rate if exhausted
        final_attempt = history[-1]
        if not final_attempt["passed_gate"] and len(history) > 1:
            final_attempt = max(
                history,
                key=lambda h: (
                    h["ceiling"]["strict_all_pass_rate"],
                    h["ceiling"]["own_plus_r3_pass_rate"],
                    h["step_index"],
                ),
            )

        locked = LockedReceiverSpec(
            key=rec.key,
            slug=rec.slug,
            family=rec.family,
            display_name=rec.display_name,
            tier=rec.tier,
            provider_only=rec.provider_only,
            concurrency=rec.concurrency,
            supports_anthropic_cache_control=rec.supports_anthropic_cache_control,
            input_per_m=rec.input_per_m,
            output_per_m=rec.output_per_m,
            max_tokens=rec.max_tokens,
            initial_effort=ladder[start_idx].effort,
            locked_effort=str(final_attempt["effort"]),
            locked_reasoning_payload=final_attempt["reasoning_payload"],
            locked_aa_index_v5=final_attempt["aa_index_v5"],
            step_ups_performed=step_ups_used,
            gate_passed=bool(final_attempt["passed_gate"]),
            gate_floor_strict_pass_rate=float(final_attempt["floor"]["strict_all_pass_rate"]),
            gate_ceiling_strict_pass_rate=float(final_attempt["ceiling"]["strict_all_pass_rate"]),
            gate_ceiling_own_plus_r3_pass_rate=float(
                final_attempt["ceiling"]["own_plus_r3_pass_rate"]
            ),
            gate_ceiling_partner_pass_rate=float(
                final_attempt["ceiling"]["partner_rules_pass_rate"]
            ),
        )
        return (
            rec.key,
            {
                "locked": locked.to_dict(),
                "total_cost_usd": total_rec_cost,
                "step_up_history": history,
            },
            locked,
        )

    try:
        results = await asyncio.gather(*(_run_receiver_gate(r) for r in rec_map.values()))
    finally:
        if owns_client:
            await client.aclose()

    receivers_report: dict[str, Any] = {}
    locked_roster: dict[str, dict[str, Any]] = {}
    total_gate_cost = 0.0
    for key, rec_report, locked in results:
        receivers_report[key] = rec_report
        locked_roster[key] = locked.to_dict()
        total_gate_cost += float(rec_report["total_cost_usd"])

    report = {
        "schema_version": "exp1_dropout_gate_v1",
        "timestamp_utc": utc_now_iso(),
        "mock": mock,
        "n_gate_cases": len(selected_gate_cases),
        "ceiling_threshold": ceiling_threshold,
        "max_step_ups": max_step_ups,
        "total_gate_cost_usd": round(total_gate_cost, 6),
        "ledger_total_cost_usd": round(ledger.total_cost_usd, 6),
        "gate_cases": [c.to_metadata_dict() for c in selected_gate_cases],
        "receivers": receivers_report,
    }

    roster_payload = {
        "schema_version": "exp1_locked_roster_v1",
        "timestamp_utc": report["timestamp_utc"],
        "mock": mock,
        "ceiling_threshold": ceiling_threshold,
        "max_step_ups": max_step_ups,
        "n_gate_cases": len(selected_gate_cases),
        "receivers": locked_roster,
    }

    (out_path / "gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_path / "gate_report.md").write_text(_render_gate_report_md(report), encoding="utf-8")
    (out_path / "locked_receiver_roster.json").write_text(
        json.dumps(roster_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# 8. Statistical Summary Engine (`summarize_exp1_results`)
# ---------------------------------------------------------------------------


def _safe_fisher_p(a_pass: int, a_total: int, b_pass: int, b_total: int) -> float | None:
    if a_total <= 0 or b_total <= 0:
        return None
    table = [[a_pass, a_total - a_pass], [b_pass, b_total - b_pass]]
    _, pval = stats.fisher_exact(table)
    return round(float(pval), 8)


def _safe_spearman(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    if len(xs) < 3 or len(ys) < 3 or len(xs) != len(ys):
        return {"n": len(xs), "rho": None, "p_value": None}
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    if np.all(arr_x == arr_x[0]) or np.all(arr_y == arr_y[0]):
        return {"n": len(xs), "rho": None, "p_value": None}
    res = stats.spearmanr(arr_x, arr_y)
    rho = float(res.statistic)
    pval = float(res.pvalue)
    if math.isnan(rho) or math.isnan(pval):
        return {"n": len(xs), "rho": None, "p_value": None}
    return {"n": len(xs), "rho": round(rho, 6), "p_value": round(pval, 8)}


def _episode_label_exact_perm_p(
    shared_recs: Sequence[dict[str, Any]],
    withheld_recs: Sequence[dict[str, Any]],
    value_key: str,
) -> float | None:
    """Exact two-sided episode-cluster permutation p-value preserving group episode counts.

    Keeps all receiver responses belonging to the same physical episode together as one cluster
    and enumerates all C(N_ep, k_shared) episode-label allocations via exact dynamic programming.
    """
    sh_by_ep: dict[str, list[int]] = defaultdict(list)
    wh_by_ep: dict[str, list[int]] = defaultdict(list)
    for r in shared_recs:
        sh_by_ep[str(r.get("episode_id", r.get("case_id", "")))].append(int(bool(r.get(value_key))))
    for r in withheld_recs:
        wh_by_ep[str(r.get("episode_id", r.get("case_id", "")))].append(int(bool(r.get(value_key))))

    k_sh = len(sh_by_ep)
    k_wh = len(wh_by_ep)
    if k_sh == 0 or k_wh == 0:
        return None

    sh_clusters = [(sum(v), len(v)) for _, v in sorted(sh_by_ep.items())]
    wh_clusters = [(sum(v), len(v)) for _, v in sorted(wh_by_ep.items())]
    all_clusters = sh_clusters + wh_clusters
    total_s = sum(s for s, _ in all_clusters)
    total_n = sum(n for _, n in all_clusters)
    obs_s = sum(s for s, _ in sh_clusters)
    obs_n = sum(n for _, n in sh_clusters)
    obs_diff = abs((obs_s / obs_n) - ((total_s - obs_s) / (total_n - obs_n)))

    # DP table: dp[c] maps (sum_s, sum_n) -> count of c-episode subsets
    dp: list[dict[tuple[int, int], int]] = [defaultdict(int) for _ in range(k_sh + 1)]
    dp[0][(0, 0)] = 1
    for ep_idx, (s_e, n_e) in enumerate(all_clusters):
        upper = min(k_sh - 1, ep_idx)
        for c in range(upper, -1, -1):
            cur = dp[c]
            if not cur:
                continue
            nxt = dp[c + 1]
            for (ss, nn), ways in cur.items():
                nxt[(ss + s_e, nn + n_e)] += ways

    total_allocs = math.comb(k_sh + k_wh, k_sh)
    extreme = 0
    for (ss, nn), ways in dp[k_sh].items():
        rem_n = total_n - nn
        if nn <= 0 or rem_n <= 0:
            continue
        diff = abs((ss / nn) - ((total_s - ss) / rem_n))
        if diff >= obs_diff - 1e-12:
            extreme += ways
    return round(extreme / total_allocs, 8)


def _episode_diff_bootstrap_ci_pp(
    shared_recs: Sequence[dict[str, Any]],
    withheld_recs: Sequence[dict[str, Any]],
    value_key: str,
    *,
    n_boot: int = 2000,
    seed: int = 20261123,
) -> list[float] | None:
    """2,000-draw within-group episode-cluster bootstrap 95% CI for (shared - withheld) in pp."""
    sh_by_ep: dict[str, list[float]] = defaultdict(list)
    wh_by_ep: dict[str, list[float]] = defaultdict(list)
    for r in shared_recs:
        sh_by_ep[str(r.get("episode_id", r.get("case_id", "")))].append(float(bool(r.get(value_key))))
    for r in withheld_recs:
        wh_by_ep[str(r.get("episode_id", r.get("case_id", "")))].append(float(bool(r.get(value_key))))

    k_sh = len(sh_by_ep)
    k_wh = len(wh_by_ep)
    if k_sh == 0 or k_wh == 0:
        return None
    sh_eps = sorted(sh_by_ep)
    wh_eps = sorted(wh_by_ep)
    sh_s = np.asarray([sum(sh_by_ep[e]) for e in sh_eps], dtype=float)
    sh_n = np.asarray([len(sh_by_ep[e]) for e in sh_eps], dtype=float)
    wh_s = np.asarray([sum(wh_by_ep[e]) for e in wh_eps], dtype=float)
    wh_n = np.asarray([len(wh_by_ep[e]) for e in wh_eps], dtype=float)

    rng = np.random.default_rng(seed)
    idx_sh = rng.integers(0, k_sh, size=(n_boot, k_sh))
    idx_wh = rng.integers(0, k_wh, size=(n_boot, k_wh))
    p_sh = sh_s[idx_sh].sum(axis=1) / sh_n[idx_sh].sum(axis=1)
    p_wh = wh_s[idx_wh].sum(axis=1) / wh_n[idx_wh].sum(axis=1)
    diffs_pp = (p_sh - p_wh) * 100.0
    lo, hi = np.percentile(diffs_pp, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def _episode_cluster_bootstrap_ci(
    records: Sequence[dict[str, Any]],
    value_key: str,
    *,
    n_boot: int = 2000,
    seed: int = 20261123,
) -> list[float] | None:
    """2,000-draw order-invariant episode-cluster bootstrap 95% CI for a binary/numeric outcome."""
    if not records:
        return None
    by_ep: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_ep[str(r.get("episode_id", r.get("case_id", "")))].append(float(bool(r.get(value_key))))
    k = len(by_ep)
    if k < 2:
        val = round(sum(float(bool(r.get(value_key))) for r in records) / len(records), 6)
        return [val, val]
    ordered_eps = sorted(by_ep)
    sums = np.asarray([sum(by_ep[ep]) for ep in ordered_eps], dtype=float)
    cnts = np.asarray([len(by_ep[ep]) for ep in ordered_eps], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot_means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def _group_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    agg = _aggregate_grade_records(records)
    n = len(records)
    if n > 0:
        ok_recs = [r for r in records if r.get("status") == "ok"]
        agg["mean_sender_r1_r2_n_formula"] = round(
            sum(int(r.get("sender_r1_r2_n_formula", 0)) for r in records) / n, 4
        )
        agg["sender_r1_r2_full_disclosure_rate"] = round(
            sum(1 for r in records if r.get("sender_r1_r2_full_disclosure")) / n, 6
        )
        agg["mean_sender_r1_r2_word_count"] = round(
            sum(int(r.get("sender_r1_r2_word_count", 0)) for r in records) / n, 2
        )
        agg["strict_all_pass_ci95"] = _episode_cluster_bootstrap_ci(records, "strict_all_pass")
        agg["strict_all_pass_ci95_ok"] = _episode_cluster_bootstrap_ci(ok_recs, "strict_all_pass")
        agg["partner_rules_pass_ci95"] = _episode_cluster_bootstrap_ci(records, "partner_rules_pass")
    else:
        agg["mean_sender_r1_r2_n_formula"] = None
        agg["sender_r1_r2_full_disclosure_rate"] = None
        agg["mean_sender_r1_r2_word_count"] = None
        agg["strict_all_pass_ci95"] = None
        agg["strict_all_pass_ci95_ok"] = None
        agg["partner_rules_pass_ci95"] = None
    return agg


def summarize_exp1_results(
    records: Sequence[dict[str, Any]],
    locked_roster: dict[str, LockedReceiverSpec] | None = None,
) -> dict[str, Any]:
    """Compute the full Experiment 1 statistical summary across roles, vendors, tiers, and bloat."""
    recs = list(records)
    role_a = [r for r in recs if r.get("dropped_role") == "A"]
    role_b = [r for r in recs if r.get("dropped_role") == "B"]
    cross_vendor = [r for r in recs if r.get("is_cross_vendor")]
    same_family = [r for r in recs if r.get("is_same_family_as_sender")]
    frontier_recs = [r for r in recs if r.get("receiver_tier") == "frontier"]
    worker_recs = [r for r in recs if r.get("receiver_tier") == "worker"]

    senders_present = sorted(
        {str(r["sender_model"]) for r in recs},
        key=lambda m: (ALL_PARTNER_MODELS.index(m) if m in ALL_PARTNER_MODELS else 99, m),
    )
    receivers_present = sorted({str(r["receiver_key"]) for r in recs})

    # 1. By Sender (overall, Role A primary, Role B contagion, cross-vendor, same-family, tier)
    by_sender: dict[str, Any] = {}
    for s in senders_present:
        s_all = [r for r in recs if r["sender_model"] == s]
        by_sender[s] = {
            "all": _group_metrics(s_all),
            "primary_role_a": _group_metrics([r for r in s_all if r["dropped_role"] == "A"]),
            "secondary_role_b": _group_metrics([r for r in s_all if r["dropped_role"] == "B"]),
            "cross_vendor": _group_metrics([r for r in s_all if r["is_cross_vendor"]]),
            "same_family": _group_metrics([r for r in s_all if r["is_same_family_as_sender"]]),
            "primary_role_a_cross_vendor": _group_metrics(
                [r for r in s_all if r["dropped_role"] == "A" and r["is_cross_vendor"]]
            ),
            "frontier_receivers": _group_metrics(
                [r for r in s_all if r["receiver_tier"] == "frontier"]
            ),
            "worker_receivers": _group_metrics(
                [r for r in s_all if r["receiver_tier"] == "worker"]
            ),
        }

    # 2. By Receiver
    by_receiver: dict[str, Any] = {}
    for rk in receivers_present:
        r_all = [r for r in recs if r["receiver_key"] == rk]
        sample = r_all[0] if r_all else {}
        roster_entry = locked_roster.get(rk) if locked_roster else None
        by_receiver[rk] = {
            "tier": roster_entry.tier if roster_entry else sample.get("receiver_tier"),
            "family": roster_entry.family if roster_entry else sample.get("receiver_family"),
            "effort": roster_entry.locked_effort if roster_entry else sample.get("receiver_effort"),
            "aa_index_v5": (
                roster_entry.locked_aa_index_v5
                if roster_entry
                else sample.get("receiver_aa_index_v5")
            ),
            "all": _group_metrics(r_all),
            "primary_role_a": _group_metrics([r for r in r_all if r["dropped_role"] == "A"]),
            "secondary_role_b": _group_metrics([r for r in r_all if r["dropped_role"] == "B"]),
            "cross_vendor": _group_metrics([r for r in r_all if r["is_cross_vendor"]]),
            "same_family": _group_metrics([r for r in r_all if r["is_same_family_as_sender"]]),
        }

    # 3. Cross-vendor vs Same-family stratification
    cv_vs_sf = {
        "all": {
            "cross_vendor": _group_metrics(cross_vendor),
            "same_family": _group_metrics(same_family),
        },
        "primary_role_a": {
            "cross_vendor": _group_metrics([r for r in role_a if r["is_cross_vendor"]]),
            "same_family": _group_metrics([r for r in role_a if r["is_same_family_as_sender"]]),
        },
        "secondary_role_b": {
            "cross_vendor": _group_metrics([r for r in role_b if r["is_cross_vendor"]]),
            "same_family": _group_metrics([r for r in role_b if r["is_same_family_as_sender"]]),
        },
        "frontier": {
            "cross_vendor": _group_metrics([r for r in frontier_recs if r["is_cross_vendor"]]),
            "same_family": _group_metrics(
                [r for r in frontier_recs if r["is_same_family_as_sender"]]
            ),
        },
        "worker": {
            "cross_vendor": _group_metrics([r for r in worker_recs if r["is_cross_vendor"]]),
            "same_family": _group_metrics(
                [r for r in worker_recs if r["is_same_family_as_sender"]]
            ),
        },
    }

    # 4. Disclosure gradient (binary >=5 vs <5, and 4 bins: 0, 1-4, 5-6, 7)
    def _disc_bin(n_f: int) -> str:
        if n_f == 0:
            return "0_rules"
        if n_f <= 4:
            return "1_to_4_rules"
        if n_f <= 6:
            return "5_to_6_rules"
        return "7_rules"

    disclosure_gradient = {
        "binary_r1_r2_full": {
            "disclosed_ge_5": _group_metrics(
                [r for r in recs if r.get("sender_r1_r2_full_disclosure")]
            ),
            "withheld_lt_5": _group_metrics(
                [r for r in recs if not r.get("sender_r1_r2_full_disclosure")]
            ),
        },
        "primary_role_a_binary_r1_r2_full": {
            "disclosed_ge_5": _group_metrics(
                [r for r in role_a if r.get("sender_r1_r2_full_disclosure")]
            ),
            "withheld_lt_5": _group_metrics(
                [r for r in role_a if not r.get("sender_r1_r2_full_disclosure")]
            ),
        },
        "by_formula_bin": {
            b: _group_metrics(
                [r for r in recs if _disc_bin(int(r.get("sender_r1_r2_n_formula", 0))) == b]
            )
            for b in ("0_rules", "1_to_4_rules", "5_to_6_rules", "7_rules")
        },
    }

    # 5. Within-GPT-6.2 (astra62) Shared vs. Withheld natural experiment
    def _astra62_split(subset: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
        shared = [r for r in subset if r.get(key)]
        withheld = [r for r in subset if not r.get(key)]
        sh_m = _group_metrics(shared)
        wh_m = _group_metrics(withheld)
        sh_eps = {str(r.get("episode_id", r.get("case_id", ""))) for r in shared}
        wh_eps = {str(r.get("episode_id", r.get("case_id", ""))) for r in withheld}
        sh_own = [r for r in shared if r.get("own_plus_r3_pass")]
        wh_own = [r for r in withheld if r.get("own_plus_r3_pass")]
        delta_strict = (
            round((sh_m["strict_all_pass_rate"] - wh_m["strict_all_pass_rate"]) * 100.0, 2)
            if sh_m["strict_all_pass_rate"] is not None and wh_m["strict_all_pass_rate"] is not None
            else None
        )
        delta_partner = (
            round((sh_m["partner_rules_pass_rate"] - wh_m["partner_rules_pass_rate"]) * 100.0, 2)
            if sh_m["partner_rules_pass_rate"] is not None
            and wh_m["partner_rules_pass_rate"] is not None
            else None
        )
        return {
            "shared": sh_m,
            "withheld": wh_m,
            "n_episodes_shared": len(sh_eps),
            "n_episodes_withheld": len(wh_eps),
            "delta_strict_pp": delta_strict,
            "delta_strict_ci95_pp": _episode_diff_bootstrap_ci_pp(
                shared, withheld, "strict_all_pass"
            ),
            "episode_perm_p_strict": _episode_label_exact_perm_p(
                shared, withheld, "strict_all_pass"
            ),
            "delta_partner_pp": delta_partner,
            "delta_partner_ci95_pp": _episode_diff_bootstrap_ci_pp(
                shared, withheld, "partner_rules_pass"
            ),
            "episode_perm_p_partner": _episode_label_exact_perm_p(
                shared, withheld, "partner_rules_pass"
            ),
            "episode_perm_p_partner_given_own": _episode_label_exact_perm_p(
                sh_own, wh_own, "partner_rules_pass"
            ),
        }

    astra62_all = [r for r in recs if r.get("sender_model") == "astra62"]
    within_astra62 = {
        "all_r1_r2": _astra62_split(astra62_all, "sender_r1_r2_full_disclosure"),
        "all_r1": _astra62_split(astra62_all, "sender_r1_full_disclosure"),
        "primary_role_a_r1": _astra62_split(
            [r for r in astra62_all if r["dropped_role"] == "A"],
            "sender_r1_full_disclosure",
        ),
        "primary_role_a_r1_r2": _astra62_split(
            [r for r in astra62_all if r["dropped_role"] == "A"],
            "sender_r1_r2_full_disclosure",
        ),
        "cross_vendor_r1_r2": _astra62_split(
            [r for r in astra62_all if r["is_cross_vendor"]],
            "sender_r1_r2_full_disclosure",
        ),
        "primary_role_a_cross_vendor_r1": _astra62_split(
            [r for r in astra62_all if r["dropped_role"] == "A" and r["is_cross_vendor"]],
            "sender_r1_full_disclosure",
        ),
        "primary_role_a_cross_vendor_r1_r2": _astra62_split(
            [r for r in astra62_all if r["dropped_role"] == "A" and r["is_cross_vendor"]],
            "sender_r1_r2_full_disclosure",
        ),
    }

    # 6. Original receiver prodding check
    prodding_check = {
        "primary_role_a": {
            "receiver_asked_rules": _group_metrics(
                [r for r in role_a if r.get("original_receiver_asked_rules")]
            ),
            "receiver_did_not_ask": _group_metrics(
                [r for r in role_a if not r.get("original_receiver_asked_rules")]
            ),
        },
        "secondary_role_b": {
            "receiver_asked_rules": _group_metrics(
                [r for r in role_b if r.get("original_receiver_asked_rules")]
            ),
            "receiver_did_not_ask": _group_metrics(
                [r for r in role_b if not r.get("original_receiver_asked_rules")]
            ),
        },
    }

    # 7. Intelligence x Sender Bloat Interaction
    # Isolate bloat among FULL-DISCLOSURE cases (>= 5 rules) so word-count bloat is not
    # confounded with terse withholding!
    full_disc_recs = [r for r in recs if r.get("sender_r1_r2_full_disclosure")]
    median_bloat_wc = (
        float(np.median([int(r["sender_r1_r2_word_count"]) for r in full_disc_recs]))
        if full_disc_recs
        else 0.0
    )
    bloat_high = [r for r in full_disc_recs if int(r["sender_r1_r2_word_count"]) > median_bloat_wc]
    bloat_low = [r for r in full_disc_recs if int(r["sender_r1_r2_word_count"]) <= median_bloat_wc]

    verbose_full_senders = {"qwen4", "opus55"}
    concise_full_senders = {"neon4", "muse", "astra60"}

    tier_bloat: dict[str, Any] = {}
    for tier_name, tier_subset in (("frontier", frontier_recs), ("worker", worker_recs)):
        t_high = [r for r in bloat_high if r["receiver_tier"] == tier_name]
        t_low = [r for r in bloat_low if r["receiver_tier"] == tier_name]
        t_verbose = [
            r
            for r in tier_subset
            if r["sender_model"] in verbose_full_senders and r.get("sender_r1_r2_full_disclosure")
        ]
        t_concise = [
            r
            for r in tier_subset
            if r["sender_model"] in concise_full_senders and r.get("sender_r1_r2_full_disclosure")
        ]
        high_m = _group_metrics(t_high)
        low_m = _group_metrics(t_low)
        verb_m = _group_metrics(t_verbose)
        conc_m = _group_metrics(t_concise)
        tier_bloat[tier_name] = {
            "high_bloat_full_disclosure": high_m,
            "low_bloat_full_disclosure": low_m,
            "high_minus_low_bloat_strict_pp": round(
                (high_m["strict_all_pass_rate"] - low_m["strict_all_pass_rate"]) * 100.0, 2
            ),
            "verbose_senders_full_disclosure": verb_m,
            "concise_senders_full_disclosure": conc_m,
            "verbose_minus_concise_strict_pp": round(
                (verb_m["strict_all_pass_rate"] - conc_m["strict_all_pass_rate"]) * 100.0, 2
            ),
            "spearman_wc_vs_strict_full_disclosure": _safe_spearman(
                [float(r["sender_r1_r2_word_count"]) for r in tier_subset if r.get("sender_r1_r2_full_disclosure")],
                [1.0 if r["strict_all_pass"] else 0.0 for r in tier_subset if r.get("sender_r1_r2_full_disclosure")],
            ),
        }

    # Per-receiver bloat sensitivity vs receiver AA v5 intelligence index
    per_receiver_bloat: dict[str, Any] = {}
    aa_vals: list[float] = []
    strict_vals: list[float] = []
    bloat_deltas: list[float] = []
    for rk in receivers_present:
        r_full = [r for r in full_disc_recs if r["receiver_key"] == rk]
        r_hi = [r for r in r_full if int(r["sender_r1_r2_word_count"]) > median_bloat_wc]
        r_lo = [r for r in r_full if int(r["sender_r1_r2_word_count"]) <= median_bloat_wc]
        hi_m = _group_metrics(r_hi)
        lo_m = _group_metrics(r_lo)
        delta_pp = round((hi_m["strict_all_pass_rate"] - lo_m["strict_all_pass_rate"]) * 100.0, 2)
        aa = by_receiver[rk]["aa_index_v5"]
        per_receiver_bloat[rk] = {
            "tier": by_receiver[rk]["tier"],
            "aa_index_v5": aa,
            "high_bloat_strict_rate": hi_m["strict_all_pass_rate"],
            "low_bloat_strict_rate": lo_m["strict_all_pass_rate"],
            "high_minus_low_bloat_strict_pp": delta_pp,
            "spearman_wc_vs_strict_full_disclosure": _safe_spearman(
                [float(r["sender_r1_r2_word_count"]) for r in r_full],
                [1.0 if r["strict_all_pass"] else 0.0 for r in r_full],
            ),
        }
        if aa is not None:
            aa_vals.append(float(aa))
            strict_vals.append(float(by_receiver[rk]["all"]["strict_all_pass_rate"]))
            bloat_deltas.append(delta_pp)

    intelligence_x_bloat = {
        "median_full_disclosure_sender_words": median_bloat_wc,
        "by_tier": tier_bloat,
        "frontier_minus_worker_bloat_interaction_pp": round(
            tier_bloat["frontier"]["high_minus_low_bloat_strict_pp"]
            - tier_bloat["worker"]["high_minus_low_bloat_strict_pp"],
            2,
        ),
        "aa_index_vs_overall_strict_spearman": _safe_spearman(aa_vals, strict_vals),
        "aa_index_vs_bloat_delta_spearman": _safe_spearman(aa_vals, bloat_deltas),
        "per_receiver": per_receiver_bloat,
    }

    # 8. Sender x Receiver matrix (all + primary Role A)
    matrix_all: dict[str, dict[str, Any]] = {}
    matrix_role_a: dict[str, dict[str, Any]] = {}
    for s in senders_present:
        matrix_all[s] = {}
        matrix_role_a[s] = {}
        for rk in receivers_present:
            cell = [r for r in recs if r["sender_model"] == s and r["receiver_key"] == rk]
            cell_a = [r for r in cell if r["dropped_role"] == "A"]
            m_all = _group_metrics(cell)
            m_a = _group_metrics(cell_a)
            matrix_all[s][rk] = {
                "n": m_all["n_cases"],
                "n_ok": m_all["n_ok"],
                "n_truncation_errors": m_all["n_truncation_errors"],
                "strict_all_pass_rate": m_all["strict_all_pass_rate"],
                "strict_all_pass_rate_ok": m_all["strict_all_pass_rate_ok"],
                "own_plus_r3_pass_rate": m_all["own_plus_r3_pass_rate"],
                "own_plus_r3_pass_rate_ok": m_all["own_plus_r3_pass_rate_ok"],
                "partner_rules_pass_rate": m_all["partner_rules_pass_rate"],
                "partner_rules_pass_rate_ok": m_all["partner_rules_pass_rate_ok"],
                "partner_pass_given_own_pass": m_all["partner_pass_given_own_pass"],
            }
            matrix_role_a[s][rk] = {
                "n": m_a["n_cases"],
                "n_ok": m_a["n_ok"],
                "n_truncation_errors": m_a["n_truncation_errors"],
                "strict_all_pass_rate": m_a["strict_all_pass_rate"],
                "strict_all_pass_rate_ok": m_a["strict_all_pass_rate_ok"],
                "own_plus_r3_pass_rate": m_a["own_plus_r3_pass_rate"],
                "own_plus_r3_pass_rate_ok": m_a["own_plus_r3_pass_rate_ok"],
                "partner_rules_pass_rate": m_a["partner_rules_pass_rate"],
                "partner_rules_pass_rate_ok": m_a["partner_rules_pass_rate_ok"],
                "partner_pass_given_own_pass": m_a["partner_pass_given_own_pass"],
            }

    return {
        "schema_version": "exp1_dropout_summary_v1",
        "timestamp_utc": utc_now_iso(),
        "overview": _group_metrics(recs),
        "by_role": {
            "primary_role_a_dropped": _group_metrics(role_a),
            "secondary_role_b_dropped": _group_metrics(role_b),
        },
        "by_tier": {
            "frontier": _group_metrics(frontier_recs),
            "worker": _group_metrics(worker_recs),
            "frontier_primary_role_a": _group_metrics(
                [r for r in frontier_recs if r["dropped_role"] == "A"]
            ),
            "worker_primary_role_a": _group_metrics(
                [r for r in worker_recs if r["dropped_role"] == "A"]
            ),
        },
        "by_sender": by_sender,
        "by_receiver": by_receiver,
        "cross_vendor_vs_same_family": cv_vs_sf,
        "disclosure_gradient": disclosure_gradient,
        "within_astra62_shared_vs_withheld": within_astra62,
        "original_receiver_prodding_check": prodding_check,
        "intelligence_x_sender_bloat": intelligence_x_bloat,
        "sender_x_receiver_matrix": {
            "all": matrix_all,
            "primary_role_a": matrix_role_a,
        },
    }


# ---------------------------------------------------------------------------
# 9. Full Experiment 1 Runner (`run_dropout_experiment`)
# ---------------------------------------------------------------------------


def _load_existing_results_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load existing result rows keyed by `(receiver_key, case_id)` with latest-error precedence.

    Any `"ok"` row always wins over non-`"ok"` rows; among non-`"ok"` rows for the same key,
    newer error rows overwrite earlier stale error rows (e.g., a later `finish_reason=length`
    truncation replaces an earlier transient `HTTP 429`).
    """
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
            key = (str(row["receiver_key"]), str(row["case_id"]))
            if row.get("status") == "ok" or existing.get(key, {}).get("status") != "ok":
                existing[key] = row
    return existing


def refresh_exp1_summary_offline(out_dir: Path = DEFAULT_EXP1_OUT_DIR) -> dict[str, Any]:
    """Strictly offline ($0, zero network) refresh of `exp1_summary.json` from on-disk artifacts."""
    out_path = Path(out_dir)
    results_path = out_path / "results.jsonl"
    roster_path = out_path / "locked_receiver_roster.json"
    ledger_path = out_path / "ledger.jsonl"

    if not results_path.exists():
        raise FileNotFoundError(f"results.jsonl not found at {results_path}")

    roster_raw = json.loads(roster_path.read_text(encoding="utf-8"))
    locked_roster = load_locked_roster(roster_path)
    deduped = _load_existing_results_jsonl(results_path)
    ordered_rows = [dict(deduped[k]) for k in sorted(deduped)]

    # Sync deterministic disclosure tags from the current disclosure parser if episodes are available
    ep_dir = out_path.parent / "episodes"
    if ep_dir.is_dir():
        cases_by_id = {c.case_id: c for c in build_dropout_cases(ep_dir)}
        for rec in ordered_rows:
            c_obj = cases_by_id.get(str(rec.get("case_id") or ""))
            if c_obj is not None:
                rec["sender_r1_n_formula"] = c_obj.sender_r1_n_formula
                rec["sender_r1_full_disclosure"] = c_obj.sender_r1_full_disclosure
                rec["sender_r1_r2_n_formula"] = c_obj.sender_r1_r2_n_formula
                rec["sender_r1_r2_full_disclosure"] = c_obj.sender_r1_r2_full_disclosure
                rec["sender_r1_n_id"] = c_obj.sender_r1_n_id
                rec["sender_r1_r2_n_id"] = c_obj.sender_r1_r2_n_id

    ledger_total_cost = 0.0
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Corrupted JSONL line {line_no} in {ledger_path}: {exc}") from exc
                ledger_total_cost += float(entry.get("cost_usd") or 0.0)

    summary = summarize_exp1_results(ordered_rows, locked_roster=locked_roster)
    summary["mock"] = bool(roster_raw.get("mock", False))
    summary["n_receivers"] = len(locked_roster)
    summary["n_cases_per_receiver"] = len({str(r["case_id"]) for r in ordered_rows})
    summary["ledger_total_cost_usd"] = round(ledger_total_cost, 6)
    summary["locked_roster"] = {k: r.to_dict() for k, r in locked_roster.items()}

    (out_path / "exp1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


async def run_dropout_experiment(
    *,
    episodes_dir: Path = DEFAULT_EPISODES_DIR,
    out_dir: Path = DEFAULT_EXP1_OUT_DIR,
    roster_path: Path | None = None,
    cases: Sequence[DropoutCase] | None = None,
    limit_cases: int | None = None,
    budget_cap_usd: float = DEFAULT_FULL_BUDGET_CAP_USD,
    max_concurrency_per_model: int | None = None,
    mock: bool = False,
    client_override: OpenRouterClient | MockOpenRouterClient | None = None,
) -> dict[str, Any]:
    """Run Experiment 1 `"handoff"` across all locked receivers and all 320 Goalpost cases.

    Features:
      - Reads `<out_dir>/locked_receiver_roster.json` (or explicit `roster_path`).
      - Dispatches all 8 receivers simultaneously with per-model concurrency semaphores and
        shared `OpenRouterClient` connection pool (`max_connections >= 160`).
      - Crash-resumable at BOTH `ledger.jsonl` (zero duplicate API spend) and `results.jsonl`.
      - Writes `<out_dir>/results.jsonl` and `<out_dir>/exp1_summary.json`.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    resolved_roster_path = roster_path or (out_path / "locked_receiver_roster.json")
    locked_roster = load_locked_roster(resolved_roster_path)

    if cases is None:
        selected_cases = build_dropout_cases(episodes_dir)
    else:
        selected_cases = list(cases)
    if limit_cases is not None and limit_cases > 0:
        selected_cases = selected_cases[:limit_cases]

    cases_by_id = {c.case_id: c for c in selected_cases}
    model_specs: dict[str, ModelSpec] = {}
    for k, r in locked_roster.items():
        spec = r.to_model_spec()
        if max_concurrency_per_model is not None and max_concurrency_per_model > 0:
            spec = ModelSpec(
                **{
                    **spec.__dict__,
                    "concurrency": min(spec.concurrency, int(max_concurrency_per_model)),
                }
            )
        model_specs[k] = spec
    total_conc = sum(s.concurrency for s in model_specs.values())

    ledger = Ledger(out_path / "ledger.jsonl", budget_cap_usd=budget_cap_usd)
    results_path = out_path / "results.jsonl"
    existing_by_key = _load_existing_results_jsonl(results_path)
    write_lock = asyncio.Lock()

    owns_client = client_override is None
    if client_override is not None:
        client = client_override
    elif mock:
        client = MockDropoutClient(ledger, model_specs, cases_by_id)
    else:
        client = OpenRouterClient(
            ledger,
            model_specs,
            max_retries=8,
            max_connections=max(160, total_conc + 16),
            max_keepalive_connections=max(80, total_conc // 2),
        )

    async def _run_one(rec: LockedReceiverSpec, case: DropoutCase) -> dict[str, Any]:
        pair_key = (rec.key, case.case_id)
        prev = existing_by_key.get(pair_key)
        if prev is not None and prev.get("status") == "ok":
            return prev

        cid = f"exp1_handoff:{rec.key}:{rec.locked_effort}:{case.case_id}"
        row = await _evaluate_single_call(
            client=client,
            receiver_key=rec.key,
            receiver_slug=rec.slug,
            receiver_family=rec.family,
            receiver_display_name=rec.display_name,
            receiver_tier=rec.tier,
            supports_cache=rec.supports_anthropic_cache_control,
            max_tokens=rec.max_tokens,
            effort=rec.locked_effort,
            reasoning_payload=rec.locked_reasoning_payload,
            aa_index_v5=rec.locked_aa_index_v5,
            case=case,
            condition="handoff",
            call_id=cid,
        )
        async with write_lock:
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            existing_by_key[pair_key] = row
        return row

    try:
        tasks = [
            _run_one(rec, case)
            for rec in locked_roster.values()
            for case in selected_cases
        ]
        all_rows = await asyncio.gather(*tasks)
    finally:
        if owns_client:
            await client.aclose()

    summary = summarize_exp1_results(all_rows, locked_roster=locked_roster)
    summary["mock"] = mock
    summary["n_receivers"] = len(locked_roster)
    summary["n_cases_per_receiver"] = len(selected_cases)
    summary["ledger_total_cost_usd"] = round(ledger.total_cost_usd, 6)
    summary["locked_roster"] = {k: r.to_dict() for k, r in locked_roster.items()}

    (out_path / "exp1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
