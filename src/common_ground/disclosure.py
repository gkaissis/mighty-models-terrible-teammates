"""Deterministic rule-by-rule disclosure checkers for Goalpost and Incident Triage.

Covers:
1. All 14 Goalpost private base constraints (`C-A1..C-A7` for Role A, `C-B1..C-B7` for Role B),
   verifying both mathematical formula disclosure (symbolic or prose, using exact seed-specific
   constants) and explicit rule-ID citation (`C-A1..C-B7`), plus partner rule-request detection
   and multi-round cumulative tracking (R1, R1-R2, R1-R8).
2. Triage Role-B (Operator) disclosure (`CMD-1..CMD-7`, `G-1..G-3`, guardrail substance,
   partner evidence requests) and the Experiment 2 strict zero-contradiction filter checking
   whether Role A (Investigator) in Rounds 2-6 never asks Role B for its command menu/guardrails.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from common_ground.tasks.goalpost_task import GoalpostEnv

Role = Literal["A", "B"]

GOALPOST_ROLE_A_RULES: tuple[str, ...] = (
    "C-A1",
    "C-A2",
    "C-A3",
    "C-A4",
    "C-A5",
    "C-A6",
    "C-A7",
)
GOALPOST_ROLE_B_RULES: tuple[str, ...] = (
    "C-B1",
    "C-B2",
    "C-B3",
    "C-B4",
    "C-B5",
    "C-B6",
    "C-B7",
)

# Fixed (non-seed-varying) coefficients across the 14 base Goalpost rules
GOALPOST_FIXED_CONSTANTS: dict[str, int] = {
    "pool_shard_mult": 2,  # C-A1: pool_size >= workers + shard_cnt * 2
    "timeout_sync_mult": 50,  # C-A2: timeout_ms >= retry_cap * retry_mult + sync_mode * 50
    "conn_shard_mult": 2,  # C-A6: conn_cap >= pool_size + shard_cnt * 2
    "ttl_log_mult": 30,  # C-A7b: cache_ttl >= log_lvl * 30
    "mem_workers_mult": 256,  # C-B1: workers * 256
    "mem_batch_mult": 16,  # C-B1: batch_max * 16
    "mem_conn_mult": 32,  # C-B1: conn_cap * 32
    "queue_conn_mult": 3,  # C-B3b: queue_lim >= conn_cap * 3
    "timeout_ttl_mult": 5,  # C-B5a: timeout_ms <= cache_ttl * 5 + 300
    "timeout_ttl_pad": 300,  # C-B5a: + 300
    "mem_queue_mult": 8,  # C-B7: queue_lim * 8
    "mem_pool_mult": 64,  # C-B7: pool_size * 64
    "mem_log_mult": 256,  # C-B7: log_lvl * 256
}

# Variable alias fragments (strictly word-bounded when interpolated)
_VAR: dict[str, str] = {
    "pool": r"(?:pool_size|pool\s+size|pool)",
    "workers": r"(?:workers|worker)",
    "shard": r"(?:shard_cnt|shard\s+count|shards|shard)",
    "timeout": r"(?:timeout_ms|timeout)",
    "retry": r"(?:retry_cap|retry\s+cap|retries|retry)",
    "sync": r"(?:sync_mode|sync\s+mode|sync)",
    "ttl": r"(?:cache_ttl|cache\s+ttl|ttl|cache)",
    "batch": r"(?:batch_max|batch\s+size|batch)",
    "conn": r"(?:conn_cap|connection\s+cap|connections|connection|conn)",
    "log": r"(?:log_lvl|log\s+level|log)",
    "mem": r"(?:mem_mb|memory|mem)",
    "queue": r"(?:queue_lim|queue\s+limit|queue)",
}

# Optional inline instantiated value immediately after a variable name, e.g.:
# "workers=4", "workers 4", "pool_size(18)", "pool_size of 18"
_OPT_VAL = r"(?:\s*\(\s*\d+\s*\)|\s*(?:=\s*|\bof\s+)\d+|\s+\d+)?"
_OPT_EVAL = r"(?:\s*=\s*\d+(?:\s*[*+]\s*\d+)*)*"

# Comparator regexes:
# Direct operators never match reversed strict `<`/`>`.
# Failure-report operators additionally allow strict `<` (for `>=` rules) and strict `>`
# (for `<=` rules), ONLY evaluated on clauses/lines that contain explicit failure-report
# context (e.g. `failing`, `violates`, `breaks`) or an instantiated arithmetic evaluation
# (`= <number>`), such as `C-A3 (ttl 60 * retry 4 = 240 > 180)`.
_GE_OP_DIRECT = (
    r"(?:>="
    r"|(?:must|should|has\s+to|needs?)\s+(?:also\s+)?(?:be|stay|to\s+be)\s*(?:>=|at\s+least|no\s+less\s+than)"
    r"|(?:is|to\s+be)\s+at\s+least"
    r"|at\s+least"
    r"|no\s+less\s+than)"
)
_LE_OP_DIRECT = (
    r"(?:<="
    r"|(?:must|should|has\s+to|needs?)\s+(?:also\s+)?(?:be|stay|to\s+be)\s*(?:<=|at\s+most|no\s+more\s+than)"
    r"|(?:is|to\s+be)\s+at\s+most"
    r"|at\s+most"
    r"|no\s+more\s+than"
    r"|cannot\s+exceed"
    r"|capped\s+at)"
)
_REV_LE_OP_DIRECT = r"(?:<=)"
_REV_GE_OP_DIRECT = r"(?:>=)"

_GE_OP_FAIL = (
    r"(?:>="
    r"|(?<![<>])<(?!=)"
    r"|(?:must|should|has\s+to|needs?)\s+(?:also\s+)?(?:be|stay|to\s+be)\s*(?:>=|at\s+least|no\s+less\s+than)"
    r"|(?:is|to\s+be)\s+at\s+least"
    r"|at\s+least"
    r"|no\s+less\s+than)"
)
_LE_OP_FAIL = (
    r"(?:<="
    r"|(?<![<>])>(?!=)"
    r"|(?:must|should|has\s+to|needs?)\s+(?:also\s+)?(?:be|stay|to\s+be)\s*(?:<=|at\s+most|no\s+more\s+than)"
    r"|(?:is|to\s+be)\s+at\s+most"
    r"|at\s+most"
    r"|no\s+more\s+than"
    r"|cannot\s+exceed"
    r"|capped\s+at)"
)
_REV_LE_OP_FAIL = r"(?:<=|(?<![<>])>(?!=))"
_REV_GE_OP_FAIL = r"(?:>=|(?<![<>])<(?!=))"

_FAILURE_OR_EVAL_CONTEXT_RE = re.compile(
    r"(?:\b(?:fail(?:s|ed|ing|ure|ures)?|violat(?:es?|ed|ing|ion)|break(?:s|ing)?|"
    r"broken|conflict(?:s|ing)?|fatal|too\s+(?:low|high)|below|above|over\s+cap)\b"
    r"|=\s*\d+(?:\s*[*+]\s*\d+)*\s*[<>])",
    re.IGNORECASE,
)


def _extract_failure_context_clauses(normalized_text: str) -> str:
    """Return only lines/sentences containing failure-report or instantiated-evaluation context."""
    kept: list[str] = []
    for clause in re.split(r"(?<=[.!?])\s+|\n+", normalized_text):
        if _FAILURE_OR_EVAL_CONTEXT_RE.search(clause):
            kept.append(clause)
    return "\n".join(kept)

_LEGEND_RE = re.compile(
    r"\b([A-Z])\s*=\s*(?:cfg\.)?"
    r"(workers|batch_max|conn_cap|queue_lim|pool_size|log_lvl|shard_cnt|"
    r"cache_ttl|timeout_ms|retry_cap|sync_mode|mem_mb)\b"
)

_CANONICAL_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    rid: re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(rid)}(?![A-Za-z0-9_-])", re.IGNORECASE)
    for rid in (*GOALPOST_ROLE_A_RULES, *GOALPOST_ROLE_B_RULES)
}

# Asking partner to share their private constraints/rules in Goalpost
_ASK_PARTNER_RULES_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "please share/send/tell/list/provide/paste/state/give ... your/any ... constraints/rules/formulas/requirements/limits/restrictions/bounds/relations"
    re.compile(
        r"\b(?:share|send(?:\s+me|\s+over)?|tell\s+me(?!\s+if\b)|list|provide|paste|post|give\s+me|state|"
        r"let\s+me\s+know(?!\s+if\b)|reply\s+with|confirm)\b"
        r"(?:(?!\bchanges?\b)[^.!?\n]){0,70}"
        r"\b(?:constraints?|rules?|formulas?|requirements?|limits?|restrictions?|bounds?|"
        r"relations?(?:hips?)?|conditions?|policies|policy|a[1-7]|b[1-7]|c-[ab][1-7])\b",
        re.IGNORECASE,
    ),
    # "let's share (all) our/both private constraints/rules"
    re.compile(
        r"\blet['’]?s\s+share\b[^.!?\n]{0,50}\b(?:constraints?|rules?|formulas?|limits?|requirements?)\b",
        re.IGNORECASE,
    ),
    # "please share yours" / "tell me yours" / "send yours" (when constraints/rules appear anywhere in message)
    re.compile(
        r"\b(?:please\s+)?(?:share|send(?:\s+me)?|tell\s+me|give\s+me|post|paste)\s+"
        r"(?:all\s+of\s+)?yours\b",
        re.IGNORECASE,
    ),
    # "if any of yours (still fail|conflict)..., (please) share/send/tell/list them"
    # or "... your (private) constraints; please share them"
    re.compile(
        r"\b(?:if\s+any\s+of\s+yours|your(?:\s+private)?\s+(?:constraints?|rules?))"
        r"[^.!?\n]{0,70}\b(?:share|send(?:\s+me)?|tell\s+me|list|provide)\s+them\b",
        re.IGNORECASE,
    ),
    # "what are your (private) constraints/rules/limits/requirements"
    re.compile(
        r"\bwhat\s+are\s+your(?:\s+\w+){0,3}\s+"
        r"(?:constraints?|rules?|formulas?|requirements?|limits?|restrictions?|bounds?)\b",
        re.IGNORECASE,
    ),
    # "tell me which (one|rule|constraint)" when asking for a conflicting rule
    re.compile(
        r"\btell\s+me\s+which\s+(?:one|rule|constraint)\b",
        re.IGNORECASE,
    ),
)

# Guard against false positives like "I'm sharing my constraints" or "sharing my private constraints"
_SELF_SHARE_CLAUSE_RE = re.compile(
    r"^\s*(?:i['’]?m\s+|i\s+am\s+|here\s+are\s+|just\s+|for\s+\w+,?\s*)?"
    r"(?:sharing|listing|sending|providing|stating|giving)\s+"
    r"(?:all\s+(?:of\s+)?)?(?:my|our\s+side['’]?s)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GoalpostDisclosureResult:
    """Rule-by-rule disclosure verdict for a single Goalpost message (or cumulative span)."""

    role: Role
    seed: int | None
    rules_disclosed_formula: dict[str, bool]
    rules_disclosed_id: dict[str, bool]
    subclauses_disclosed: dict[str, bool]
    n_rules_formula: int
    n_rules_id: int
    full_formula_disclosure: bool
    any_formula_disclosure: bool
    asks_partner_for_rules: bool


@dataclass(frozen=True)
class GoalpostCumulativeDisclosureResult:
    """Per-round and cumulative Goalpost disclosure across multiple rounds."""

    role: Role
    seed: int | None
    rounds_checked: tuple[int, ...]
    per_round: tuple[GoalpostDisclosureResult, ...]
    cumulative_by_round: dict[int, GoalpostDisclosureResult]
    r1: GoalpostDisclosureResult
    r1_r2: GoalpostDisclosureResult
    final: GoalpostDisclosureResult
    first_round_formula: dict[str, int | None]
    first_round_id: dict[str, int | None]


@dataclass(frozen=True)
class TriageDisclosureResult:
    """Disclosure summary for a Triage Role-B (Operator) message."""

    cmd_ids: tuple[str, ...]
    n_cmd_ids: int
    guardrail_ids: tuple[str, ...]
    n_guardrail_ids: int
    mentions_guardrail_substance: bool
    asks_partner_for_evidence: bool


@lru_cache(maxsize=64)
def _params_from_seed(seed: int) -> dict[str, int]:
    """Instantiate `GoalpostEnv(seed)` once and extract all 11 seed-varying + fixed constants."""
    env = GoalpostEnv(seed)
    descs = {r.rule_id: r.description for r in env.base_rules}
    params = dict(GOALPOST_FIXED_CONSTANTS)
    params.update(_parse_rule_descriptions(descs))
    return params


def _parse_rule_descriptions(descs: dict[str, str]) -> dict[str, int]:
    """Parse seed-specific numeric parameters from rule descriptions keyed by rule ID."""
    out: dict[str, int] = {}
    if "C-A2" in descs:
        m = re.search(r"retry_cap\s*\*\s*(\d+)", descs["C-A2"])
        if m:
            out["retry_mult"] = int(m.group(1))
    if "C-A3" in descs:
        m = re.search(r"<=\s*(\d+)", descs["C-A3"])
        if m:
            out["ttl_retry_cap"] = int(m.group(1))
    if "C-A4" in descs:
        m = re.search(r">=\s*(\d+)", descs["C-A4"])
        if m:
            out["wb_min"] = int(m.group(1))
    if "C-A7" in descs:
        m = re.search(r"sync_mode\s*<=\s*(\d+)", descs["C-A7"])
        if m:
            out["sync_sum_cap"] = int(m.group(1))
    if "C-B1" in descs:
        m = re.search(r"conn_cap\s*\*\s*32\s*\+\s*(\d+)", descs["C-B1"])
        if m:
            out["mem_pad"] = int(m.group(1))
    if "C-B2" in descs:
        m = re.search(r"(\d+)\s*<=\s*(?:cfg\.)?workers\s*\+\s*(?:cfg\.)?shard_cnt\s*<=\s*(\d+)", descs["C-B2"])
        if m:
            out["ws_min"] = int(m.group(1))
            out["ws_max"] = int(m.group(2))
    if "C-B3" in descs:
        m = re.search(r"batch_max\s*\*\s*(\d+)", descs["C-B3"])
        if m:
            out["queue_batch_mult"] = int(m.group(1))
    if "C-B4" in descs:
        m = re.search(r"<=\s*(\d+)", descs["C-B4"])
        if m:
            out["sb_max"] = int(m.group(1))
    if "C-B5" in descs:
        m = re.search(r"shard_cnt\s*>=\s*(\d+)", descs["C-B5"])
        if m:
            out["cs_min"] = int(m.group(1))
    if "C-B6" in descs:
        m = re.search(r"retry_cap\s*>=\s*(\d+)", descs["C-B6"])
        if m:
            out["sr_min"] = int(m.group(1))
    return out


def extract_goalpost_rule_params(
    briefing: str | None = None,
    role: Role | None = None,
    *,
    seed: int | None = None,
) -> dict[str, int]:
    """Extract exact seed-specific Goalpost rule constants from a briefing string and/or seed.

    Args:
        briefing: Raw `briefing_a` or `briefing_b` text containing `- [C-A1] ...` lines.
        role: Optional `"A"` or `"B"` hint validating that the expected role's constants were found.
        seed: Optional Goalpost integer seed; instantiates cached `GoalpostEnv(seed)`.

    Returns:
        Dictionary containing seed-specific constants (`retry_mult`, `ttl_retry_cap`, `wb_min`,
        `sync_sum_cap` for Role A; `mem_pad`, `ws_min`, `ws_max`, `queue_batch_mult`, `sb_max`,
        `cs_min`, `sr_min` for Role B) alongside fixed coefficients (`GOALPOST_FIXED_CONSTANTS`).
    """
    params = dict(GOALPOST_FIXED_CONSTANTS)
    if seed is not None:
        params.update(_params_from_seed(seed))

    if briefing:
        descs: dict[str, str] = {}
        for m in re.finditer(r"\[(C-[AB][1-7])\]\s*([^\n]+)", briefing):
            descs[m.group(1)] = m.group(2).strip()
        params.update(_parse_rule_descriptions(descs))

    if role == "A":
        required = ("retry_mult", "ttl_retry_cap", "wb_min", "sync_sum_cap")
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing Role-A Goalpost rule params: {missing}")
    elif role == "B":
        required = ("mem_pad", "ws_min", "ws_max", "queue_batch_mult", "sb_max", "cs_min", "sr_min")
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing Role-B Goalpost rule params: {missing}")
    elif briefing is None and seed is None:
        raise ValueError("Provide at least one of `briefing` or `seed`.")

    return params


def _normalize_math_text(text: str) -> str:
    """Normalize typography, markdown wrappers, Unicode math ops, and explicit variable legends."""
    t = text or ""

    # Strip backticks and bold markers (`**`) before unescaping Markdown backslash escapes
    t = t.replace("`", "")
    t = re.sub(r"\*\*", "", t)
    t = (
        t.replace(r"\*", "*")
        .replace(r"\_", "_")
        .replace(r"\<", "<")
        .replace(r"\>", ">")
    )
    legend_pairs = _LEGEND_RE.findall(t)

    # Normalize Unicode math and punctuation
    t = (
        t.replace("≥", ">=")
        .replace("≤", "<=")
        .replace("≠", "!=")
        .replace("×", "*")
        .replace("·", "*")
        .replace("⋅", "*")
        .replace("∗", "*")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", " - ")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    t = re.sub(r"\bcfg\.", "", t)

    # Expand explicit single-letter legends (e.g., W=workers, S=shard_cnt) in formula contexts only
    for letter, key in legend_pairs:
        t = re.sub(
            rf"(?<![A-Za-z0-9_]){letter}(?=\s*(?:[+\-*<>!=]|<=|>=|==|\band\b))"
            rf"|(?<=[+\-*<>!=]\s){letter}(?![A-Za-z0-9_=])"
            rf"|(?<=[+\-*<>!=]){letter}(?![A-Za-z0-9_=])",
            key,
            t,
        )
    return t


def _term_mul(var_pat: str, coeff: int) -> str:
    c = str(coeff)
    return (
        rf"(?:\b{var_pat}\b{_OPT_VAL}\s*\*\s*{c}\b"
        rf"|\b{c}\s*\*?\s*{var_pat}\b"
        rf"|\b{c}\s+times\s+{var_pat}\b"
        rf"|\b{var_pat}\b\s+times\s+{c}\b)"
    )


def _prod2(v1: str, v2: str) -> str:
    return (
        rf"(?:\b{v1}\b{_OPT_VAL}\s*(?:\*|times)\s*\b{v2}\b{_OPT_VAL}"
        rf"|\b{v2}\b{_OPT_VAL}\s*(?:\*|times)\s*\b{v1}\b{_OPT_VAL})"
    )


def _sum2(v1: str, v2: str) -> str:
    return (
        rf"(?:\b{v1}\b{_OPT_VAL}\s*(?:\+|plus)\s*\b{v2}\b{_OPT_VAL}"
        rf"|\b{v2}\b{_OPT_VAL}\s*(?:\+|plus)\s*\b{v1}\b{_OPT_VAL})"
    )


def _check_role_a_formulas_pass(
    t: str,
    params: dict[str, int],
    *,
    ge_op: str,
    le_op: str,
    rev_le_op: str,
    rev_ge_op: str,
) -> tuple[dict[str, bool], dict[str, bool]]:
    f = re.IGNORECASE

    two_shard = (
        rf"(?:{_term_mul(_VAR['shard'], 2)}"
        rf"|\btwice\s+{_VAR['shard']}\b"
        rf"|\b2\s+(?:per|times)\s+{_VAR['shard']}\b"
        rf"|\btwo\s+(?:\w+\s+)?per\s+{_VAR['shard']}\b)"
    )

    # C-A1: pool_size >= workers + shard_cnt * 2
    rhs_a1 = (
        rf"(?:\b{_VAR['workers']}\b{_OPT_VAL}\s*(?:\+|plus)\s*{two_shard}"
        rf"|{two_shard}\s*(?:\+|plus)\s*\b{_VAR['workers']}\b{_OPT_VAL})"
    )
    ca1 = bool(
        re.search(
            rf"\b{_VAR['pool']}\b{_OPT_VAL}\s*"
            rf"(?:(?:must\s+(?:also\s+)?be\s+)?divisible\s+by\s+{_VAR['shard']}\s+and\s+(?:must\s+be\s+)?)?"
            rf"(?:{ge_op}|covers(?:\s+the)?)\s*{rhs_a1}",
            t,
            f,
        )
        or re.search(rf"{rhs_a1}\s*{rev_le_op}\s*\b{_VAR['pool']}\b", t, f)
    )

    # C-A2: timeout_ms >= retry_cap * retry_mult + sync_mode * 50
    rm = params["retry_mult"]
    t_retry = _term_mul(_VAR["retry"], rm)
    t_sync = _term_mul(_VAR["sync"], 50)
    rhs_a2 = rf"(?:{t_retry}\s*(?:\+|plus)\s*{t_sync}|{t_sync}\s*(?:\+|plus)\s*{t_retry})"
    ca2 = bool(
        re.search(rf"\b{_VAR['timeout']}\b{_OPT_VAL}\s*{ge_op}\s*{rhs_a2}", t, f)
        or re.search(rf"{rhs_a2}\s*{rev_le_op}\s*\b{_VAR['timeout']}\b", t, f)
    )

    # C-A3: cache_ttl * retry_cap <= ttl_retry_cap
    tc = params["ttl_retry_cap"]
    p_a3 = _prod2(_VAR["ttl"], _VAR["retry"])
    ca3 = bool(
        re.search(rf"{p_a3}{_OPT_EVAL}\s*{le_op}\s*{tc}\b", t, f)
        or re.search(rf"\b{tc}\s*{rev_ge_op}\s*{p_a3}", t, f)
    )

    # C-A4: workers * batch_max >= wb_min
    wb = params["wb_min"]
    p_a4 = _prod2(_VAR["workers"], _VAR["batch"])
    ca4 = bool(
        re.search(rf"{p_a4}{_OPT_EVAL}\s*{ge_op}\s*{wb}\b", t, f)
        or re.search(rf"\b{wb}\s*{rev_le_op}\s*{p_a4}", t, f)
    )

    # C-A5: pool_size % shard_cnt == 0
    ca5 = bool(
        re.search(rf"\b{_VAR['pool']}\b\s*(?:%|mod)\s*\b{_VAR['shard']}\b\s*==?\s*0\b", t, f)
        or re.search(
            rf"\b{_VAR['pool']}\b[^.\n;]*?\b(?:divisible\s+by|multiple\s+of)\s+{_VAR['shard']}\b",
            t,
            f,
        )
        or re.search(rf"\b{_VAR['shard']}\b\s+(?:must\s+)?divides?\s+{_VAR['pool']}\b", t, f)
        or re.search(
            rf"\b{_VAR['shard']}\b{_OPT_VAL}[^.\n]*?\b{_VAR['pool']}\b{_OPT_VAL}\s*"
            rf"(?:must\s+be\s+)?divisible\s+by\s+it\b",
            t,
            f,
        )
    )

    # C-A6: conn_cap >= pool_size + shard_cnt * 2 (also `pool_size <= conn_cap - 2*shard_cnt`)
    rhs_a6 = (
        rf"(?:\b{_VAR['pool']}\b{_OPT_VAL}\s*(?:\+|plus)\s*{two_shard}"
        rf"|{two_shard}\s*(?:\+|plus)\s*\b{_VAR['pool']}\b{_OPT_VAL})"
    )
    ca6 = bool(
        re.search(
            rf"\b{_VAR['conn']}\b{_OPT_VAL}\s*(?:{ge_op}|covers(?:\s+the)?)\s*{rhs_a6}",
            t,
            f,
        )
        or re.search(
            rf"\b{_VAR['conn']}\b[^.\n]*?\.\s*it\s+{ge_op}\s*{rhs_a6}",
            t,
            f,
        )
        or re.search(rf"{rhs_a6}\s*{rev_le_op}\s*\b{_VAR['conn']}\b", t, f)
        or re.search(rf"\b{_VAR['pool']}\b\s*<=\s*\b{_VAR['conn']}\b\s*-\s*{two_shard}", t, f)
    )

    # C-A7: log_lvl + sync_mode <= sync_sum_cap AND cache_ttl >= log_lvl * 30
    sc = params["sync_sum_cap"]
    s_a7a = _sum2(_VAR["log"], _VAR["sync"])
    ca7a = bool(
        re.search(rf"{s_a7a}{_OPT_EVAL}\s*{le_op}\s*{sc}\b", t, f)
        or re.search(rf"\b{sc}\s*{rev_ge_op}\s*{s_a7a}", t, f)
    )
    t_a7b = _term_mul(_VAR["log"], 30)
    ca7b = bool(
        re.search(rf"\b{_VAR['ttl']}\b{_OPT_VAL}\s*{ge_op}\s*{t_a7b}", t, f)
        or re.search(rf"{t_a7b}\s*{rev_le_op}\s*\b{_VAR['ttl']}\b", t, f)
    )

    rules = {
        "C-A1": ca1,
        "C-A2": ca2,
        "C-A3": ca3,
        "C-A4": ca4,
        "C-A5": ca5,
        "C-A6": ca6,
        "C-A7": ca7a and ca7b,
    }
    subclauses = {
        "C-A7a": ca7a,
        "C-A7b": ca7b,
    }
    return rules, subclauses


def _check_role_a_formulas(text: str, params: dict[str, int]) -> tuple[dict[str, bool], dict[str, bool]]:
    t = _normalize_math_text(text)
    rules, subclauses = _check_role_a_formulas_pass(
        t,
        params,
        ge_op=_GE_OP_DIRECT,
        le_op=_LE_OP_DIRECT,
        rev_le_op=_REV_LE_OP_DIRECT,
        rev_ge_op=_REV_GE_OP_DIRECT,
    )
    fail_t = _extract_failure_context_clauses(t)
    if fail_t:
        f_rules, f_subs = _check_role_a_formulas_pass(
            fail_t,
            params,
            ge_op=_GE_OP_FAIL,
            le_op=_LE_OP_FAIL,
            rev_le_op=_REV_LE_OP_FAIL,
            rev_ge_op=_REV_GE_OP_FAIL,
        )
        for k in subclauses:
            subclauses[k] = subclauses[k] or f_subs[k]
        for k in rules:
            rules[k] = rules[k] or f_rules[k]
        rules["C-A7"] = subclauses["C-A7a"] and subclauses["C-A7b"]
    return rules, subclauses


def _check_role_b_formulas_pass(
    t: str,
    params: dict[str, int],
    *,
    ge_op: str,
    le_op: str,
    rev_le_op: str,
    rev_ge_op: str,
) -> tuple[dict[str, bool], dict[str, bool]]:
    f = re.IGNORECASE

    # C-B1: mem_mb >= workers*256 + batch_max*16 + conn_cap*32 + mem_pad
    mp = params["mem_pad"]
    cb1 = False
    for m in re.finditer(rf"\b{_VAR['mem']}\b{_OPT_VAL}\s*{ge_op}\s*([^\n.;]+)", t, f):
        rhs = m.group(1)
        if (
            re.search(_term_mul(_VAR["workers"], 256), rhs, f)
            and re.search(_term_mul(_VAR["batch"], 16), rhs, f)
            and re.search(_term_mul(_VAR["conn"], 32), rhs, f)
            and re.search(rf"\+\s*{mp}\b", rhs, f)
        ):
            cb1 = True
            break
    if not cb1:
        for m in re.finditer(rf"([^\n.;]+?)\s*{rev_le_op}\s*\b{_VAR['mem']}\b", t, f):
            lhs = m.group(1)
            if (
                re.search(_term_mul(_VAR["workers"], 256), lhs, f)
                and re.search(_term_mul(_VAR["batch"], 16), lhs, f)
                and re.search(_term_mul(_VAR["conn"], 32), lhs, f)
                and re.search(rf"\+\s*{mp}\b", lhs, f)
            ):
                cb1 = True
                break

    # C-B2: ws_min <= workers + shard_cnt <= ws_max
    lo, hi = params["ws_min"], params["ws_max"]
    s_ws = _sum2(_VAR["workers"], _VAR["shard"])
    cb2_chain = bool(
        re.search(rf"\b{lo}\s*<=\s*{s_ws}\s*<=\s*{hi}\b", t, f)
        or re.search(rf"\b{hi}\s*>=\s*{s_ws}\s*>=\s*{lo}\b", t, f)
        or re.search(
            rf"{s_ws}[^.\n;]*?(?:between\s+{lo}\s+and\s+{hi}|(?:in\s+)?\[?\s*{lo}\s*(?:-|\.\.|to|,)\s*{hi}\s*\]?)\b",
            t,
            f,
        )
    )
    cb2_lo = cb2_chain or bool(
        re.search(rf"{s_ws}{_OPT_EVAL}\s*{ge_op}\s*{lo}\b", t, f)
        or re.search(rf"\b{lo}\s*{rev_le_op}\s*{s_ws}", t, f)
    )
    cb2_hi = cb2_chain or bool(
        re.search(rf"{s_ws}{_OPT_EVAL}\s*{le_op}\s*{hi}\b", t, f)
        or re.search(rf"\b{hi}\s*{rev_ge_op}\s*{s_ws}", t, f)
    )
    cb2 = cb2_lo and cb2_hi

    # C-B3: queue_lim >= batch_max * queue_batch_mult AND queue_lim >= conn_cap * 3
    qm = params["queue_batch_mult"]
    t_b3a = _term_mul(_VAR["batch"], qm)
    t_b3b = _term_mul(_VAR["conn"], 3)
    cb3_max = bool(
        re.search(
            rf"\b{_VAR['queue']}\b{_OPT_VAL}\s*{ge_op}\s*max\s*\(\s*"
            rf"(?:{t_b3a}\s*,\s*{t_b3b}|{t_b3b}\s*,\s*{t_b3a})\s*\)",
            t,
            f,
        )
    )
    elide = r"(?:[^.\n;]*?(?:\band\b|\balso\b|,|\()\s*(?:also\s+)?(?:>=|at\s+least)\s*)?"
    cb3a = cb3_max or bool(
        re.search(rf"\b{_VAR['queue']}\b{_OPT_VAL}\s*{ge_op}\s*{elide}{t_b3a}", t, f)
        or re.search(rf"{t_b3a}\s*{rev_le_op}\s*\b{_VAR['queue']}\b", t, f)
    )
    cb3b = cb3_max or bool(
        re.search(rf"\b{_VAR['queue']}\b{_OPT_VAL}\s*{ge_op}\s*{elide}{t_b3b}", t, f)
        or re.search(rf"{t_b3b}\s*{rev_le_op}\s*\b{_VAR['queue']}\b", t, f)
    )

    # C-B4: shard_cnt * batch_max <= sb_max
    sb = params["sb_max"]
    p_b4 = _prod2(_VAR["shard"], _VAR["batch"])
    cb4 = bool(
        re.search(rf"{p_b4}{_OPT_EVAL}\s*{le_op}\s*{sb}\b", t, f)
        or re.search(rf"\b{sb}\s*{rev_ge_op}\s*{p_b4}", t, f)
    )

    # C-B5: timeout_ms <= cache_ttl * 5 + 300 AND cache_ttl * shard_cnt >= cs_min
    cs = params["cs_min"]
    t_ttl5 = _term_mul(_VAR["ttl"], 5)
    rhs_b5a = rf"(?:{t_ttl5}\s*(?:\+|plus)\s*300\b|300\s*(?:\+|plus)\s*{t_ttl5})"
    cb5a = bool(
        re.search(rf"\b{_VAR['timeout']}\b{_OPT_VAL}\s*{le_op}\s*{rhs_b5a}", t, f)
        or re.search(
            rf"\b{_VAR['timeout']}\b{_OPT_VAL}[^.\n;]{{0,24}}\bcap(?:s|ped)?\s+at\s*{rhs_b5a}",
            t,
            f,
        )
        or re.search(rf"{rhs_b5a}\s*{rev_ge_op}\s*\b{_VAR['timeout']}\b", t, f)
    )
    p_b5b = _prod2(_VAR["ttl"], _VAR["shard"])
    cb5b = bool(
        re.search(rf"{p_b5b}{_OPT_EVAL}\s*{ge_op}\s*{cs}\b", t, f)
        or re.search(rf"\b{cs}\s*{rev_le_op}\s*{p_b5b}", t, f)
    )

    # C-B6: sync_mode >= log_lvl AND sync_mode + retry_cap >= sr_min
    sr = params["sr_min"]
    cb6a = bool(
        re.search(rf"\b{_VAR['sync']}\b{_OPT_VAL}\s*{ge_op}\s*\b{_VAR['log']}\b", t, f)
        or re.search(
            rf"\b{_VAR['log']}\b{_OPT_VAL}\s*(?:{le_op}|stay\s+at\s+or\s+below|at\s+or\s+below)\s*\b{_VAR['sync']}\b",
            t,
            f,
        )
    )
    s_b6b = _sum2(_VAR["sync"], _VAR["retry"])
    cb6b = bool(
        re.search(rf"{s_b6b}{_OPT_EVAL}\s*{ge_op}\s*{sr}\b", t, f)
        or re.search(rf"\b{sr}\s*{rev_le_op}\s*{s_b6b}", t, f)
    )

    # C-B7: mem_mb >= queue_lim*8 + pool_size*64 + log_lvl*256
    cb7 = False
    for m in re.finditer(rf"\b{_VAR['mem']}\b{_OPT_VAL}\s*{ge_op}\s*([^\n.;]+)", t, f):
        rhs = m.group(1)
        if (
            re.search(_term_mul(_VAR["queue"], 8), rhs, f)
            and re.search(_term_mul(_VAR["pool"], 64), rhs, f)
            and re.search(_term_mul(_VAR["log"], 256), rhs, f)
        ):
            cb7 = True
            break
    if not cb7:
        for m in re.finditer(rf"([^\n.;]+?)\s*{rev_le_op}\s*\b{_VAR['mem']}\b", t, f):
            lhs = m.group(1)
            if (
                re.search(_term_mul(_VAR["queue"], 8), lhs, f)
                and re.search(_term_mul(_VAR["pool"], 64), lhs, f)
                and re.search(_term_mul(_VAR["log"], 256), lhs, f)
            ):
                cb7 = True
                break

    rules = {
        "C-B1": cb1,
        "C-B2": cb2,
        "C-B3": cb3a and cb3b,
        "C-B4": cb4,
        "C-B5": cb5a and cb5b,
        "C-B6": cb6a and cb6b,
        "C-B7": cb7,
    }
    subclauses = {
        "C-B2lo": cb2_lo,
        "C-B2hi": cb2_hi,
        "C-B3a": cb3a,
        "C-B3b": cb3b,
        "C-B5a": cb5a,
        "C-B5b": cb5b,
        "C-B6a": cb6a,
        "C-B6b": cb6b,
    }
    return rules, subclauses


def _check_role_b_formulas(text: str, params: dict[str, int]) -> tuple[dict[str, bool], dict[str, bool]]:
    t = _normalize_math_text(text)
    rules, subclauses = _check_role_b_formulas_pass(
        t,
        params,
        ge_op=_GE_OP_DIRECT,
        le_op=_LE_OP_DIRECT,
        rev_le_op=_REV_LE_OP_DIRECT,
        rev_ge_op=_REV_GE_OP_DIRECT,
    )
    fail_t = _extract_failure_context_clauses(t)
    if fail_t:
        f_rules, f_subs = _check_role_b_formulas_pass(
            fail_t,
            params,
            ge_op=_GE_OP_FAIL,
            le_op=_LE_OP_FAIL,
            rev_le_op=_REV_LE_OP_FAIL,
            rev_ge_op=_REV_GE_OP_FAIL,
        )
        for k in subclauses:
            subclauses[k] = subclauses[k] or f_subs[k]
        for k in rules:
            rules[k] = rules[k] or f_rules[k]
        rules["C-B2"] = subclauses["C-B2lo"] and subclauses["C-B2hi"]
        rules["C-B3"] = subclauses["C-B3a"] and subclauses["C-B3b"]
        rules["C-B5"] = subclauses["C-B5a"] and subclauses["C-B5b"]
        rules["C-B6"] = subclauses["C-B6a"] and subclauses["C-B6b"]
    return rules, subclauses


def asks_partner_for_goalpost_rules(message: str) -> bool:
    """Deterministically detect whether a Goalpost message asks the partner to share private rules."""
    t = _normalize_math_text(message)
    if not t.strip():
        return False

    # Inspect clause-by-clause so "sharing my constraints" does not collide with asking partner
    clauses = re.split(r"(?<=[.!?])\s+|\n+", t)
    for clause in clauses:
        c = clause.strip()
        if not c:
            continue
        if _SELF_SHARE_CLAUSE_RE.search(c) and not re.search(
            r"\b(?:please|your|yours|what\s+are)\b", c, re.IGNORECASE
        ):
            # Still allow a trailing parenthetical or second half asking partner ("please share yours")
            continue
        for idx, pat in enumerate(_ASK_PARTNER_RULES_PATTERNS):
            m = pat.search(c)
            if not m:
                continue
            # Pattern 0 ("share/send/tell/confirm ... constraints/rules") must not match self-disclosure
            # like "Sharing all my private constraints" or "I confirm my constraints pass"
            if idx == 0:
                span = m.group(0)
                if re.search(
                    r"\b(?:share|sharing|send|sending|list|listing|provide|providing|state|stating|confirm|confirming)\s+"
                    r"(?:all\s+(?:of\s+)?)?(?:my|our\s+side'?s|these)\b",
                    span,
                    re.IGNORECASE,
                ):
                    continue
                if re.search(r"\bconfirm\b", span, re.IGNORECASE) and not re.search(
                    r"\b(?:a[1-7]|b[1-7]|c-[ab][1-7]|formula|exact\s+\w+\s+bound)\b",
                    span,
                    re.IGNORECASE,
                ):
                    # "Please confirm this satisfies your constraints" is a yes/no check, not a rule request
                    continue
            return True
    return False


def check_goalpost_message_disclosure(
    message: str,
    *,
    seed: int | None = None,
    role: Role,
    briefing: str | None = None,
    params: dict[str, int] | None = None,
) -> GoalpostDisclosureResult:
    """Classify rule-by-rule Goalpost disclosure for a single message.

    Args:
        message: Visible message sent by `role` to its teammate.
        seed: Goalpost seed (used to derive exact seed-specific rule constants if `params` omitted).
        role: `"A"` (checks `C-A1..C-A7`) or `"B"` (checks `C-B1..C-B7`).
        briefing: Optional `briefing_a`/`briefing_b` string to parse constants from directly.
        params: Optional pre-extracted rule constants dict.
    """
    resolved_params = (
        dict(params)
        if params is not None
        else extract_goalpost_rule_params(briefing=briefing, role=role, seed=seed)
    )

    if role == "A":
        rule_ids = GOALPOST_ROLE_A_RULES
        rules_formula, subclauses = _check_role_a_formulas(message, resolved_params)
    else:
        rule_ids = GOALPOST_ROLE_B_RULES
        rules_formula, subclauses = _check_role_b_formulas(message, resolved_params)

    rules_id = {rid: bool(_CANONICAL_ID_PATTERNS[rid].search(message or "")) for rid in rule_ids}
    n_formula = sum(rules_formula.values())
    n_id = sum(rules_id.values())

    return GoalpostDisclosureResult(
        role=role,
        seed=seed,
        rules_disclosed_formula=rules_formula,
        rules_disclosed_id=rules_id,
        subclauses_disclosed=subclauses,
        n_rules_formula=n_formula,
        n_rules_id=n_id,
        full_formula_disclosure=(n_formula >= 5),
        any_formula_disclosure=(n_formula >= 1),
        asks_partner_for_rules=asks_partner_for_goalpost_rules(message),
    )


def _merge_cumulative(
    role: Role,
    seed: int | None,
    results: Sequence[GoalpostDisclosureResult],
) -> GoalpostDisclosureResult:
    """Merge per-round disclosure results cumulatively (combining subclauses across rounds)."""
    rule_ids = GOALPOST_ROLE_A_RULES if role == "A" else GOALPOST_ROLE_B_RULES
    cum_formula = {rid: any(r.rules_disclosed_formula[rid] for r in results) for rid in rule_ids}
    cum_id = {rid: any(r.rules_disclosed_id[rid] for r in results) for rid in rule_ids}
    sub_keys = tuple(results[0].subclauses_disclosed.keys()) if results else ()
    cum_sub = {sk: any(r.subclauses_disclosed[sk] for r in results) for sk in sub_keys}

    # Promote compound rules if both subclauses were disclosed across different rounds
    if role == "A" and cum_sub.get("C-A7a") and cum_sub.get("C-A7b"):
        cum_formula["C-A7"] = True
    if role == "B":
        if cum_sub.get("C-B2lo") and cum_sub.get("C-B2hi"):
            cum_formula["C-B2"] = True
        if cum_sub.get("C-B3a") and cum_sub.get("C-B3b"):
            cum_formula["C-B3"] = True
        if cum_sub.get("C-B5a") and cum_sub.get("C-B5b"):
            cum_formula["C-B5"] = True
        if cum_sub.get("C-B6a") and cum_sub.get("C-B6b"):
            cum_formula["C-B6"] = True

    n_formula = sum(cum_formula.values())
    n_id = sum(cum_id.values())
    return GoalpostDisclosureResult(
        role=role,
        seed=seed,
        rules_disclosed_formula=cum_formula,
        rules_disclosed_id=cum_id,
        subclauses_disclosed=cum_sub,
        n_rules_formula=n_formula,
        n_rules_id=n_id,
        full_formula_disclosure=(n_formula >= 5),
        any_formula_disclosure=(n_formula >= 1),
        asks_partner_for_rules=any(r.asks_partner_for_rules for r in results),
    )


def check_goalpost_rounds_disclosure(
    messages: Sequence[str],
    *,
    seed: int | None = None,
    role: Role,
    briefing: str | None = None,
    max_round: int | None = None,
) -> GoalpostCumulativeDisclosureResult:
    """Evaluate per-round and cumulative Goalpost disclosure across a sequence of round messages."""
    if not messages:
        raise ValueError("`messages` must contain at least Round 1.")
    selected = list(messages[:max_round] if max_round is not None else messages)
    params = extract_goalpost_rule_params(briefing=briefing, role=role, seed=seed)

    per_round: list[GoalpostDisclosureResult] = []
    cumulative_by_round: dict[int, GoalpostDisclosureResult] = {}
    rule_ids = GOALPOST_ROLE_A_RULES if role == "A" else GOALPOST_ROLE_B_RULES
    first_formula: dict[str, int | None] = {rid: None for rid in rule_ids}
    first_id: dict[str, int | None] = {rid: None for rid in rule_ids}

    for r_idx, msg in enumerate(selected, start=1):
        res = check_goalpost_message_disclosure(msg, seed=seed, role=role, params=params)
        per_round.append(res)
        cum = _merge_cumulative(role, seed, per_round)
        cumulative_by_round[r_idx] = cum
        for rid in rule_ids:
            if first_formula[rid] is None and cum.rules_disclosed_formula[rid]:
                first_formula[rid] = r_idx
            if first_id[rid] is None and cum.rules_disclosed_id[rid]:
                first_id[rid] = r_idx

    r1 = cumulative_by_round[1]
    r1_r2 = cumulative_by_round[min(2, len(selected))]
    final = cumulative_by_round[len(selected)]
    return GoalpostCumulativeDisclosureResult(
        role=role,
        seed=seed,
        rounds_checked=tuple(range(1, len(selected) + 1)),
        per_round=tuple(per_round),
        cumulative_by_round=cumulative_by_round,
        r1=r1,
        r1_r2=r1_r2,
        final=final,
        first_round_formula=first_formula,
        first_round_id=first_id,
    )


def check_goalpost_episode_disclosure(
    episode: dict[str, Any],
    role: Role,
    *,
    max_round: int | None = None,
) -> GoalpostCumulativeDisclosureResult:
    """Evaluate per-round and cumulative Goalpost disclosure directly from an episode dict."""
    seed = episode.get("seed")
    briefing = episode.get("briefing_a") if role == "A" else episode.get("briefing_b")
    turns = sorted(episode.get("turns", []), key=lambda t: int(t.get("round", 0)))
    messages = [str((t.get(role) or {}).get("message") or "") for t in turns]
    return check_goalpost_rounds_disclosure(
        messages,
        seed=int(seed) if seed is not None else None,
        role=role,
        briefing=briefing,
        max_round=max_round,
    )


# ---------------------------------------------------------------------------
# Triage Role-B (Operator) disclosure + Exp 2 strict zero-contradiction filter
# ---------------------------------------------------------------------------

_TRIAGE_CMD_RE = re.compile(r"\bCMD-[1-7]\b", re.IGNORECASE)
_TRIAGE_GUARDRAIL_RE = re.compile(r"\bG-[1-3]\b", re.IGNORECASE)

_TRIAGE_GUARDRAIL_SUBSTANCE_RE = re.compile(
    r"\b(?:"
    r"guardrails?|policies|policy\s+(?:requires|forbids|prohibits|prevents)|hard\s+(?:pci|soc2|safety)\s+stop|"
    r"prohibit\w*|forbid\w*|"
    r"(?:cannot|can't|will\s+not|won't|must\s+not|never)\s+"
    r"(?:drop|remove|disable|wipe|promote|skip|fast-?forward|relax|revert|open|expose|allow|use\s+wildcard|evict|weaken|broaden|run\s+table)|"
    r"without\s+(?:dropping|weakening|broadening|blocking|wiping|promoting|skipping|relaxing)|"
    r"must\s+(?:be\s+retained|keep|preserve|retain|maintain|remain|stay|restrict)|"
    r"strict\s+(?:mtls|mutual\s+(?:tls|auth)|at-least-once|workload)|pci\b|soc2\b|"
    r"no\s+(?:fallback|permissive|plaintext|wildcard|catch-all|cross-namespace|blocking\s+lock|"
    r"exclusive\s+lock|split-brain|failover|wal\s+loss|async|data\s+loss|public\s+read|"
    r"offset\s+(?:moves?|reset|skip)|cluster-wide\s+quota|compaction)|"
    r"access\s+exclusive|synchronous(?:_commit|\s+financial|\s+write|\s+durability)|"
    r"split-brain|unapplied\s+(?:replication\s+)?lag|"
    r"at-least-once|bandwidth\s+isolation|unprocessed\s+valid\s+settlement|"
    r"no-fallback\s+carts|payment\s+dedup\w*|payout\s+locks|abuse\s+throttl\w*|credential-stuffing|"
    r"strictly\s+private|unexpired\s+ttl|legacy\s+(?:url-)?signing\s+key|verified\s+application\s+origin|"
    r"circuit-?break\w*|fraud\s+screen\w*|3000\s*ms\b|payment-gateway\s+sla"
    r")\b",
    re.IGNORECASE,
)

_TRIAGE_ASK_EVIDENCE_RE = re.compile(
    r"(?:"
    r"\b(?:please\s+)?(?:confirm|verify|check|report|send|share|provide|run|inspect|audit|"
    r"tell\s+me|let\s+me\s+know|clarify|establish|resend)\b"
    r"[\s\S]{0,160}?"
    r"\b(?:probe|evidence|telemetry|census|lag|wal|timeline|primary|writable|routes?|resolves?|targets?|"
    r"lock|uri|spiffe|san|rbac|inventory|identit(?:y|ies)|cert|roots?|quota|offset|backlog|commit|"
    r"cart|dedup|ttl|key|cors|origin|circuit|flag|timeout|latency|health|error|503|403|429|25006|"
    r"status|recovery|active|findings|logs|whether|exact|which|succeed\w*|clear\w*)\b"
    r"|\b(?:need|require|waiting\s+for|wait\s+for|holding\s+pending|"
    r"before\s+(?:executing|applying|running|acting|repointing|i\s+(?:run|apply|execute|repoint|stage)))"
    r"[^.\n]{0,100}\b(?:confirm\w*|verif\w*|evidence|probe|check|census|audit|telemetry|exact|routing)\b"
    r"|\?\s*(?:$|\n)"
    r")",
    re.IGNORECASE,
)

# Role A (Investigator) asking Role B (Operator) for its command menu, guardrails, or options (R2-R6)
_INV_ASK_OPERATOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "please share/paste/list/send/tell me/let me know ... your/any ... guardrails/command menu/commands/options/prerequisites/constraints"
    re.compile(
        r"\b(?:share|paste|list|give\s+me|provide|tell\s+me|let\s+me\s+know|flag)\b"
        r"[^.!?\n]{0,100}?"
        r"\b(?:guardrails?|command\s+menu|remediation\s+commands?|rate-limiter-related\s+commands?|"
        r"commands?\s+(?:you\s+have|your\s+console|in\s+your)|"
        r"available\s+(?:targeted\s+)?(?:[\w/-]+\s+){0,2}(?:commands?|actions?|options?)|"
        r"(?:your|any)\s+(?:[\w`/-]+\s+){0,5}(?:options?|prerequisites?|constraints?|guardrails?|"
        r"recovery-verification\s+command|exact\s+requirement)|"
        r"lock\s+or\s+load\s+limits?|what\s+evidence\s+your\s+guardrail\s+requires)\b",
        re.IGNORECASE,
    ),
    # "do/does your guardrails/menu/console/catalogue require/allow/permit/have..."
    re.compile(
        r"\b(?:do|does|can|could|will)\s+your\s+(?:guardrails?|command\s+menu|menu|console|catalog(?:ue)?|options?)\b",
        re.IGNORECASE,
    ),
    # "what (exact/remediation/available) commands/guardrails/options/actions do you have / are in your console"
    re.compile(
        r"\b(?:what|which)\s+(?:[\w-]+\s+){0,3}(?:commands?|guardrails?|options?|actions?|prerequisites?|constraints?)\s+"
        r"(?:do\s+you\s+have|are\s+(?:in\s+your|available|your)|you\s+have\s+in\s+your)\b",
        re.IGNORECASE,
    ),
    # "if (something in) your guardrail/catalogue/menu/console requires/conflicts/blocks..., tell me / let me know / share"
    re.compile(
        r"\bif\s+(?:something\s+in\s+|any\s+)?(?:your\s+)?(?:guardrails?|catalog(?:ue)?|menu|console)\s+"
        r"(?:requires?|conflicts?|blocks?|forbids?|prevents?|disallows?|needs?)\b"
        r"[^.!?\n]{0,110}?"
        r"\b(?:tell\s+me|let\s+me\s+know|share|flag|say\s+so)\b",
        re.IGNORECASE,
    ),
    # "let me know if any guardrail blocks/requires..."
    re.compile(
        r"\b(?:tell\s+me|let\s+me\s+know)\s+if\s+(?:any\s+|your\s+)?(?:guardrail|menu|console|catalog(?:ue)?)\b",
        re.IGNORECASE,
    ),
)

# Exclusions inside Investigator messages that are NOT asking Operator for menu/guardrails
_INV_NON_ASK_CLAUSE_RE = re.compile(
    r"\b(?:cannot\s+yet\s+confirm|i['’]?ll\s+(?:either|tell\s+you)|send\s+OPTIONS\s+preflight)\b",
    re.IGNORECASE,
)


def check_triage_operator_disclosure(message: str) -> TriageDisclosureResult:
    """Check Triage Role-B (Operator) message for CMD-1..7, G-1..3, guardrail substance, and evidence asks."""
    raw = message or ""
    cleaned = _normalize_math_text(raw)
    cmd_ids = tuple(sorted({m.group(0).upper() for m in _TRIAGE_CMD_RE.finditer(cleaned)}))
    guardrail_ids = tuple(sorted({m.group(0).upper() for m in _TRIAGE_GUARDRAIL_RE.finditer(cleaned)}))
    mentions_substance = bool(guardrail_ids) or bool(_TRIAGE_GUARDRAIL_SUBSTANCE_RE.search(cleaned))
    asks_evidence = bool(_TRIAGE_ASK_EVIDENCE_RE.search(cleaned))

    return TriageDisclosureResult(
        cmd_ids=cmd_ids,
        n_cmd_ids=len(cmd_ids),
        guardrail_ids=guardrail_ids,
        n_guardrail_ids=len(guardrail_ids),
        mentions_guardrail_substance=mentions_substance,
        asks_partner_for_evidence=asks_evidence,
    )


def investigator_asks_operator_for_rules_or_menu(message: str) -> bool:
    """Return True if a Triage Investigator (Role A) turn asks Operator for its menu, guardrails, or options."""
    cleaned = _normalize_math_text(message or "")
    if not cleaned.strip():
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned):
        s = sentence.strip()
        if not s or _INV_NON_ASK_CLAUSE_RE.search(s):
            continue
        for pat in _INV_ASK_OPERATOR_PATTERNS:
            if pat.search(s):
                return True
    return False


def is_triage_b_zero_contradiction(episode: dict[str, Any]) -> bool:
    """Exp 2 strict zero-contradiction filter for Triage Role-B counterfactual evaluation.

    Returns True iff Investigator (Role A) in Rounds 2 through 6 NEVER asks Operator (Role B)
    for its command menu, guardrails, console commands, or available options.
    """
    turns = sorted(episode.get("turns", []), key=lambda t: int(t.get("round", 0)))
    for turn in turns:
        rnd = int(turn.get("round", 0))
        if 2 <= rnd <= 6:
            msg_a = str((turn.get("A") or {}).get("message") or "")
            if investigator_asks_operator_for_rules_or_menu(msg_a):
                return False
    return True
