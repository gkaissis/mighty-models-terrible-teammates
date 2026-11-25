"""Programmatic quote-to-episode, quote-to-choice, and numeric-claim verifier for v2.1.

Verifies structured references against raw v2 episode JSONs, forced-choice JSONL records,
episode debrief summaries/friction quotes, and Capstone Tone-Flip rationales:
- Exact file/record existence
- Seed, model, role, and round metadata match
- Whitespace-normalized verbatim substring verification (with ordered `[...]` ellipsis
  segment verification and transparent reporting of inline editorial `[Model]` annotations)
- Deterministic numeric-claim consistency diagnostics (e.g., catching hallucinated task
  counts like Claude Opus's `"all 26 constraints"` on Goalpost where active constraints
  are 14 base -> 17 max)
- End-to-end audit of draft blog quotes, quote cards, and key v2.1 candidate quotes
  saved to `data/v2/analysis/quote_audit_v2_1.json` (preserved frozen as the pre-publication
  audit record that flagged the Opus "26 constraints" claim prior to publication).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from common_ground.config import DATA_DIR, ROOT_DIR

SourceType = Literal[
    "episode_turn",
    "choice_reason",
    "debrief_summary",
    "capstone_reason",
    "capstone_rewrite",
    "exp2_reason",
]
DebriefField = Literal[
    "F1.summary",
    "F2.summary",
    "F1.friction_quote",
    "F1.helped_quote",
    "F2.friction_quote",
    "F2.helped_quote",
]

_ELLIPSIS_SPLIT_RE = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]")
_EDITORIAL_BRACKET_RE = re.compile(r"\[(?!\s*(?:\.\.\.|…)\s*\])([^\[\]]+)\]")

# Known task ground-truth structural bounds for numeric-claim diagnostics
TASK_STRUCTURAL_BOUNDS: dict[str, dict[str, tuple[int, int]]] = {
    "goalpost": {
        "active_constraints": (14, 17),
        "private_constraints_single_side": (7, 9),
        "keys": (12, 12),
        "keys_single_side": (6, 6),
        "rounds": (8, 8),
        "passable_rounds": (6, 6),
        "max_edits_per_round": (2, 2),
    },
    "calendar": {
        "meetings": (7, 7),
        "slots": (9, 9),
        "rounds": (10, 10),
        "disclosure_limit_cells": (2, 2),
    },
    "triage": {
        "rounds": (6, 6),
        "probes": (6, 6),
        "commands": (7, 7),
        "guardrails": (3, 3),
        "fix_steps": (2, 2),
    },
    "code_handoff": {
        "rounds": (6, 6),
        "ci_checks": (4, 4),
        "subsystems": (4, 4),
        "invariants": (4, 4),
        "baseline_checks": (4, 4),
    },
}


@dataclass(frozen=True)
class QuoteReference:
    """Structured pointer to a quote in a v2 episode, debrief, choice, or Capstone record."""

    source_type: SourceType
    locator: str
    quote_text: str
    expected_seed: int | tuple[int, int] | None = None
    expected_model: str | None = None
    expected_role: Literal["A", "B"] | None = None
    expected_round: int | None = None
    debrief_field: DebriefField = "F1.summary"
    label: str = ""


@dataclass(frozen=True)
class NumericClaimFlag:
    """Suspicious or inconsistent numeric task-count claim detected inside a quote or rationale."""

    claim_text: str
    claimed_value: int
    metric: str
    expected_range: tuple[int, int]
    severity: Literal["error", "warning"]
    explanation: str


@dataclass(frozen=True)
class QuoteVerificationResult:
    """Full verification verdict for a single `QuoteReference`."""

    label: str
    source_type: SourceType
    locator: str
    record_exists: bool
    seed_match: bool | None
    model_match: bool | None
    role_match: bool | None
    round_match: bool | None
    verbatim_match: bool
    verbatim_match_ignoring_editorial_brackets: bool
    editorial_annotations: tuple[str, ...]
    segments_checked: int
    segments_matched: int
    missing_segments: tuple[str, ...]
    actual_seed: int | tuple[int, int] | None
    actual_model: str | None
    actual_role: str | None
    actual_round: int | None
    actual_word_count: int | None
    numeric_claim_flags: tuple[NumericClaimFlag, ...]
    errors: tuple[str, ...]
    passed: bool
    passed_with_editorial_annotations: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single ASCII spaces and trim ends."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _ordered_segment_match(quote_text: str, source_text: str) -> tuple[bool, int, int, tuple[str, ...]]:
    """Split `quote_text` on explicit `[...]` ellipses and verify every segment verbatim in order."""
    haystack = normalize_whitespace(source_text)
    raw_segments = _ELLIPSIS_SPLIT_RE.split(quote_text or "")
    segments = [normalize_whitespace(seg) for seg in raw_segments if normalize_whitespace(seg)]
    if not segments:
        return False, 0, 0, ("<empty quote>",)

    cursor = 0
    matched = 0
    missing: list[str] = []
    for seg in segments:
        pos = haystack.find(seg, cursor)
        if pos != -1:
            matched += 1
            cursor = pos + len(seg)
        else:
            missing.append(seg)
    return (len(missing) == 0), len(segments), matched, tuple(missing)


def _strip_editorial_brackets(quote_text: str, source_text: str) -> tuple[str, tuple[str, ...]]:
    """Strip inline editorial annotations like `[Muse]` or `[GPT-6]` that do not appear in source."""
    haystack = normalize_whitespace(source_text)
    annotations: list[str] = []

    def _repl(m: re.Match[str]) -> str:
        full = m.group(0)
        if full in haystack:
            return full
        annotations.append(full)
        return ""

    stripped = _EDITORIAL_BRACKET_RE.sub(_repl, quote_text or "")
    # Clean any double spaces or space-before-punctuation introduced by removing `[Annot]`
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = normalize_whitespace(stripped)
    return stripped, tuple(annotations)


def _extract_capstone_reason(rec: dict[str, Any]) -> str:
    """Extract the `reason` field from a Capstone choice record's JSON `raw` string (or `reason`)."""
    if isinstance(rec.get("reason"), str) and rec["reason"].strip():
        return rec["reason"]
    raw = rec.get("raw")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("reason"), str):
                return parsed["reason"]
        except json.JSONDecodeError:
            pass
        return raw
    return ""


class V2CorpusIndex:
    """Lazy-cached index over `data/v2` episodes, forced choices, and Capstone choices (JSON or Parquet)."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else (DATA_DIR / "v2")
        self._episodes_cache: dict[str, dict[str, Any]] = {}
        self._parquet_episodes_loaded: bool = False
        self._choices_by_id: dict[str, dict[str, Any]] | None = None
        self._choices_excluded_by_id: dict[str, dict[str, Any]] | None = None
        self._capstone_by_key: dict[tuple[str, str], dict[str, Any]] | None = None
        self._capstone_by_packet: dict[str, list[dict[str, Any]]] | None = None
        self._capstone_rewrites_by_key: dict[tuple[str, str], dict[str, Any]] | None = None
        self._exp2_by_key: dict[tuple[str, str, str, str], dict[str, Any]] | None = None

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        if episode_id in self._episodes_cache:
            return self._episodes_cache[episode_id]
        path = self.data_dir / "episodes" / f"{episode_id}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._episodes_cache[episode_id] = data
            return data
        if not self._parquet_episodes_loaded:
            self._parquet_episodes_loaded = True
            from common_ground.parquet_loader import (
                load_episodes_from_parquet,
                resolve_hf_export_dir,
            )

            hf_dir = resolve_hf_export_dir(self.data_dir)
            if hf_dir is not None:
                eps, _ = load_episodes_from_parquet(hf_dir)
                for ep in eps:
                    self._episodes_cache[str(ep["episode_id"])] = ep
        return self._episodes_cache.get(episode_id)

    def _load_choices(self) -> None:
        if self._choices_by_id is not None:
            return
        self._choices_by_id = {}
        self._choices_excluded_by_id = {}
        choices_path = self.data_dir / "choices.jsonl"
        if choices_path.is_file():
            with choices_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    mid = rec.get("manifest_item_id")
                    if mid:
                        self._choices_by_id[str(mid)] = rec
                    iid = rec.get("item_id")
                    if iid and str(iid) not in self._choices_by_id:
                        self._choices_by_id[str(iid)] = rec
        else:
            from common_ground.parquet_loader import (
                load_choices_from_parquet,
                resolve_hf_export_dir,
            )

            hf_dir = resolve_hf_export_dir(self.data_dir)
            if hf_dir is not None:
                for rec in load_choices_from_parquet(hf_dir):
                    mid = rec.get("manifest_item_id")
                    if mid:
                        self._choices_by_id[str(mid)] = rec
                    iid = rec.get("item_id")
                    if iid and str(iid) not in self._choices_by_id:
                        self._choices_by_id[str(iid)] = rec

        excl_path = self.data_dir / "choices_excluded.jsonl"
        if excl_path.is_file():
            with excl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    mid = rec.get("manifest_item_id")
                    if mid:
                        self._choices_excluded_by_id[str(mid)] = rec
        else:
            from common_ground.parquet_loader import (
                load_excluded_choices_from_parquet,
                resolve_hf_export_dir,
            )

            hf_dir = resolve_hf_export_dir(self.data_dir)
            if hf_dir is not None:
                for rec in load_excluded_choices_from_parquet(hf_dir):
                    mid = rec.get("manifest_item_id")
                    if mid:
                        self._choices_excluded_by_id[str(mid)] = rec

    def get_choice(self, locator: str) -> tuple[dict[str, Any] | None, bool]:
        """Return `(choice_record, is_excluded)`."""
        self._load_choices()
        assert self._choices_by_id is not None
        assert self._choices_excluded_by_id is not None
        if locator in self._choices_by_id:
            return self._choices_by_id[locator], False
        if locator in self._choices_excluded_by_id:
            return self._choices_excluded_by_id[locator], True
        return None, False

    def iter_choices(self) -> list[dict[str, Any]]:
        self._load_choices()
        assert self._choices_by_id is not None
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for rec in self._choices_by_id.values():
            mid = str(rec.get("manifest_item_id") or "")
            if mid and mid not in seen:
                seen.add(mid)
                out.append(rec)
        return out

    def _load_capstone(self) -> None:
        if self._capstone_by_key is not None:
            return
        self._capstone_by_key = {}
        self._capstone_by_packet = {}
        cap_path = self.data_dir / "capstone_tone_flip" / "capstone_choices.jsonl"
        if cap_path.is_file():
            with cap_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pid = str(rec.get("packet_id") or "")
                    rev = str(rec.get("reviewer") or "")
                    if pid and rev:
                        self._capstone_by_key[(pid, rev)] = rec
                    if pid:
                        self._capstone_by_packet.setdefault(pid, []).append(rec)
        else:
            from common_ground.parquet_loader import (
                load_capstone_from_parquet,
                resolve_hf_export_dir,
            )

            hf_dir = resolve_hf_export_dir(self.data_dir)
            if hf_dir is not None:
                _, cap_choices = load_capstone_from_parquet(hf_dir)
                for rec in cap_choices:
                    pid = str(rec.get("packet_id") or "")
                    rev = str(rec.get("reviewer") or "")
                    if pid and rev:
                        self._capstone_by_key[(pid, rev)] = rec
                    if pid:
                        self._capstone_by_packet.setdefault(pid, []).append(rec)

    def get_capstone_choice(
        self,
        locator: str,
        reviewer: str | None = None,
    ) -> dict[str, Any] | None:
        self._load_capstone()
        assert self._capstone_by_key is not None
        assert self._capstone_by_packet is not None
        if reviewer and (locator, reviewer) in self._capstone_by_key:
            return self._capstone_by_key[(locator, reviewer)]
        # Also allow composite locator `"packet_id::reviewer"`
        if "::" in locator:
            pid, rev = locator.split("::", 1)
            if (pid, rev) in self._capstone_by_key:
                return self._capstone_by_key[(pid, rev)]
        recs = self._capstone_by_packet.get(locator, [])
        if len(recs) == 1:
            return recs[0]
        return None

    def iter_capstone_choices(self) -> list[dict[str, Any]]:
        self._load_capstone()
        assert self._capstone_by_key is not None
        return list(self._capstone_by_key.values())

    def _load_capstone_rewrites(self) -> None:
        if self._capstone_rewrites_by_key is not None:
            return
        self._capstone_rewrites_by_key = {}
        rw_path = self.data_dir / "capstone_tone_flip" / "capstone_rewrites.jsonl"
        if rw_path.is_file():
            with rw_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (str(rec.get("episode_id") or ""), str(rec.get("target_tone") or ""))
                    self._capstone_rewrites_by_key[key] = rec
        else:
            from common_ground.parquet_loader import (
                load_capstone_from_parquet,
                resolve_hf_export_dir,
            )

            hf_dir = resolve_hf_export_dir(self.data_dir)
            if hf_dir is not None:
                rewrites, _ = load_capstone_from_parquet(hf_dir)
                for rec in rewrites:
                    key = (str(rec.get("episode_id") or ""), str(rec.get("target_tone") or ""))
                    self._capstone_rewrites_by_key[key] = rec

    def get_capstone_rewrite(self, episode_id: str, target_tone: str) -> dict[str, Any] | None:
        self._load_capstone_rewrites()
        assert self._capstone_rewrites_by_key is not None
        return self._capstone_rewrites_by_key.get((episode_id, target_tone))

    def _load_exp2_choices(self) -> None:
        if self._exp2_by_key is not None:
            return
        self._exp2_by_key = {}
        e2_path = self.data_dir / "exp2_info_flip" / "choices.jsonl"
        if e2_path.is_file():
            with e2_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (
                        str(rec.get("pair_id") or ""),
                        str(rec.get("condition") or ""),
                        str(rec.get("order") or ""),
                        str(rec.get("reviewer") or ""),
                    )
                    self._exp2_by_key[key] = rec
        else:
            import pandas as pd

            from common_ground.parquet_loader import resolve_hf_export_dir

            hf_dir = resolve_hf_export_dir(self.data_dir)
            if hf_dir is not None and (hf_dir / "exp2_info_flip_choices.parquet").is_file():
                df = pd.read_parquet(hf_dir / "exp2_info_flip_choices.parquet")
                for row in df.to_dict(orient="records"):
                    key = (
                        str(row.get("pair_id") or ""),
                        str(row.get("condition") or ""),
                        str(row.get("order") or ""),
                        str(row.get("reviewer") or ""),
                    )
                    self._exp2_by_key[key] = row

    def get_exp2_choice(
        self, pair_id: str, condition: str, order: str, reviewer: str
    ) -> dict[str, Any] | None:
        self._load_exp2_choices()
        assert self._exp2_by_key is not None
        return self._exp2_by_key.get((pair_id, condition, order, reviewer))


def _infer_task_name(meta: dict[str, Any] | str) -> str | None:
    if isinstance(meta, str):
        low = meta.strip().lower()
        for t in TASK_STRUCTURAL_BOUNDS:
            if t in low:
                return t
        return None
    if isinstance(meta, dict):
        if isinstance(meta.get("task"), str):
            return meta["task"].strip().lower()
        for key in ("episode_id", "episode_1_id", "pair_id", "manifest_item_id"):
            val = str(meta.get(key) or "").lower()
            for t in TASK_STRUCTURAL_BOUNDS:
                if t in val:
                    return t
    return None


def check_numeric_claim_consistency(
    quote_text: str,
    episode_or_task_meta: dict[str, Any] | str,
) -> list[NumericClaimFlag]:
    """Flag obvious hallucinated task counts in quotes or model rationales.

    For example, flags Claude Opus's `"all 26 constraints"` on Goalpost (where active
    constraints range strictly from 14 base to 17 max after R3/R5/R7 shocks), impossible
    constraint fractions like `26/26` or `18/18`, wrong meeting counts on Calendar (`!= 7`),
    wrong guardrail counts on Triage (`!= 3`), or wrong CI-check counts on Code Handoff (`!= 4`).
    """
    text = normalize_whitespace(quote_text)
    task = _infer_task_name(episode_or_task_meta)
    if not text or not task:
        return []

    flags: list[NumericClaimFlag] = []
    seen_spans: list[tuple[int, int, str]] = []

    def _add(
        claim: str,
        val: int,
        metric: str,
        expected: tuple[int, int],
        severity: Literal["error", "warning"],
        explanation: str,
        span: tuple[int, int],
    ) -> None:
        for s0, s1, prev_metric in seen_spans:
            if prev_metric == metric and max(s0, span[0]) < min(s1, span[1]):
                return
        seen_spans.append((span[0], span[1], metric))
        flags.append(
            NumericClaimFlag(
                claim_text=claim,
                claimed_value=val,
                metric=metric,
                expected_range=expected,
                severity=severity,
                explanation=explanation,
            )
        )

    if task == "goalpost":
        # 1. "all N (active/total/joint/private)? constraints/rules"
        for m in re.finditer(
            r"\ball\s+(\d+)\s+(?:(?:active|total|joint|combined|coupled|base)\s+)?(?:constraints?|rules?)\b",
            text,
            re.IGNORECASE,
        ):
            n = int(m.group(1))
            # Check if immediately followed/preceded by "of my/your/our private" (single side = 7..9)
            ctx_before = text[max(0, m.start() - 25) : m.start()].lower()
            ctx_after = text[m.end() : min(len(text), m.end() + 25)].lower()
            is_single_side = bool(
                re.search(r"\b(?:my|your|their|its)\b", ctx_before)
                or re.search(r"^\s*of\s+(?:my|your|their|its)\b", ctx_after)
            )
            if is_single_side:
                if not (7 <= n <= 9):
                    _add(
                        m.group(0),
                        n,
                        "goalpost_single_side_constraints",
                        (7, 9),
                        "error",
                        f"Claimed {n} single-side constraints on Goalpost, but each owner has 7 base rules (8-9 with shock alerts).",
                        m.span(),
                    )
            elif not (14 <= n <= 17):
                _add(
                    m.group(0),
                    n,
                    "goalpost_active_constraints",
                    (14, 17),
                    "error",
                    f"Claimed '{m.group(0)}' on Goalpost, but total active constraints are 14 base -> 17 max (R3 +1, R5 +1, R7 +1).",
                    m.span(),
                )

        # 2. Bare "N constraints/rules" where N > 17 (always impossible on Goalpost)
        for m in re.finditer(
            r"\b(\d+)\s+(?:(?:active|total|joint|combined|private|visible)\s+)?(?:constraints?|rules?)\b",
            text,
            re.IGNORECASE,
        ):
            n = int(m.group(1))
            if n > 17:
                _add(
                    m.group(0),
                    n,
                    "goalpost_active_constraints",
                    (14, 17),
                    "error",
                    f"Claimed {n} constraints on Goalpost, exceeding the maximum 17 active constraints (14 base + 3 shocks).",
                    m.span(),
                )

        # 3. Constraint pass fractions like "17/17" or "26/26" (skip dates/unrelated numbers by checking context)
        for m in re.finditer(r"\b(\d+)/(\d+)\b", text):
            num, den = int(m.group(1)), int(m.group(2))
            window = text[max(0, m.start() - 40) : min(len(text), m.end() + 40)].lower()
            if re.search(
                r"\b(?:round|constraint|rule|pass|end|final|up|down|broke|fail|check)\b", window
            ) and (den not in (6, 7, 8, 9, 14, 15, 16, 17) or num > den):
                _add(
                    m.group(0),
                    den,
                    "goalpost_fraction_denominator",
                    (14, 17),
                    "error",
                    f"Fraction '{m.group(0)}' does not match Goalpost active constraint counts (14..17), single-side counts (7..9), or rounds (6/8).",
                    m.span(),
                )

        # 4. Total rounds claim ("all N rounds" / "N total rounds")
        for m in re.finditer(r"\b(?:all|over|across)\s+(\d+)\s+rounds?\b", text, re.IGNORECASE):
            n = int(m.group(1))
            if n not in (6, 8):
                _add(
                    m.group(0),
                    n,
                    "goalpost_rounds",
                    (8, 8),
                    "warning",
                    f"Claimed '{m.group(0)}' on Goalpost, which has 8 total rounds (6 causally passable).",
                    m.span(),
                )

    elif task == "calendar":
        for m in re.finditer(r"\b(\d+)/(\d+)\b", text):
            num, den = int(m.group(1)), int(m.group(2))
            window = text[max(0, m.start() - 40) : min(len(text), m.end() + 40)].lower()
            if re.search(
                r"\b(?:meeting|schedule|agreement|consensus|veto|optimal|perfect|result|outcome)\b",
                window,
            ) and (den != 7 or num > 7):
                _add(
                    m.group(0),
                    den,
                    "calendar_meetings",
                    (7, 7),
                    "error",
                    f"Fraction '{m.group(0)}' is inconsistent with Calendar's 7 meetings (MTG-1..MTG-7).",
                    m.span(),
                )
        for m in re.finditer(r"\ball\s+(\d+)\s+(?:meetings?|assignments?|slots?)\b", text, re.IGNORECASE):
            n = int(m.group(1))
            unit = m.group(0).lower()
            expected_n = 9 if "slot" in unit else 7
            if n != expected_n:
                _add(
                    m.group(0),
                    n,
                    "calendar_meetings",
                    (expected_n, expected_n),
                    "error",
                    f"Claimed '{m.group(0)}' on Calendar (expected {expected_n}).",
                    m.span(),
                )

    elif task == "triage":
        for m in re.finditer(r"\ball\s+(\d+)\s+guardrails?\b", text, re.IGNORECASE):
            n = int(m.group(1))
            if n != 3:
                _add(
                    m.group(0),
                    n,
                    "triage_guardrails",
                    (3, 3),
                    "error",
                    f"Claimed '{m.group(0)}' on Triage, which has exactly 3 guardrails (G-1..G-3).",
                    m.span(),
                )
        for m in re.finditer(r"\b(\d+)\s+commands?\b", text, re.IGNORECASE):
            n = int(m.group(1))
            if n > 7:
                _add(
                    m.group(0),
                    n,
                    "triage_commands",
                    (1, 7),
                    "error",
                    f"Claimed {n} commands on Triage, exceeding the 7 console commands (CMD-1..CMD-7).",
                    m.span(),
                )

    elif task == "code_handoff":
        for m in re.finditer(r"\b(\d+)/(\d+)\s+(?:ci|checks?|green|subsystems?|gates?)\b", text, re.IGNORECASE):
            num, den = int(m.group(1)), int(m.group(2))
            if den != 4 or num > 4:
                _add(
                    m.group(0),
                    den,
                    "code_handoff_ci_checks",
                    (4, 4),
                    "error",
                    f"Claimed '{m.group(0)}' on Code Handoff, which has 4 subsystem checks (4/4).",
                    m.span(),
                )
        for m in re.finditer(r"\ball\s+(\d+)\s+(?:invariants?|subsystems?|checks?)\b", text, re.IGNORECASE):
            n = int(m.group(1))
            if n != 4:
                _add(
                    m.group(0),
                    n,
                    "code_handoff_invariants",
                    (4, 4),
                    "error",
                    f"Claimed '{m.group(0)}' on Code Handoff, which has 4 invariants/subsystems/checks.",
                    m.span(),
                )

    return flags


def verify_quote(
    ref: QuoteReference,
    *,
    data_dir: Path | str | None = None,
    index: V2CorpusIndex | None = None,
) -> QuoteVerificationResult:
    """Verify a structured `QuoteReference` against the v2 corpus."""
    idx = index if index is not None else V2CorpusIndex(data_dir=data_dir)
    errors: list[str] = []

    record_exists = False
    source_text = ""
    meta_for_numeric: dict[str, Any] = {}
    actual_seed: int | tuple[int, int] | None = None
    actual_model: str | None = None
    actual_role: str | None = None
    actual_round: int | None = None

    if ref.source_type == "episode_turn":
        ep = idx.get_episode(ref.locator)
        if ep is None:
            errors.append(f"Episode file not found: {ref.locator}")
        else:
            record_exists = True
            meta_for_numeric = ep
            actual_seed = int(ep["seed"]) if ep.get("seed") is not None else None
            turns = ep.get("turns") or []
            turn = None
            if ref.expected_round is not None:
                for t in turns:
                    if int(t.get("round", -1)) == ref.expected_round:
                        turn = t
                        break
            if turn is None:
                errors.append(f"Round {ref.expected_round} not found in episode {ref.locator}")
            else:
                actual_round = int(turn["round"])
                role = ref.expected_role
                if role is None and ref.expected_model is not None:
                    if ep.get("model_a") == ref.expected_model:
                        role = "A"
                    elif ep.get("model_b") == ref.expected_model:
                        role = "B"
                if role not in ("A", "B"):
                    errors.append(f"Cannot resolve role for episode {ref.locator}")
                else:
                    actual_role = role
                    actual_model = str(ep.get(f"model_{role.lower()}") or "")
                    source_text = str((turn.get(role) or {}).get("message") or "")

    elif ref.source_type == "debrief_summary":
        ep = idx.get_episode(ref.locator)
        if ep is None:
            errors.append(f"Episode file not found: {ref.locator}")
        else:
            record_exists = True
            meta_for_numeric = ep
            actual_seed = int(ep["seed"]) if ep.get("seed") is not None else None
            debriefs = ep.get("debriefs") or {}
            role = ref.expected_role
            if role is None and ref.expected_model is not None:
                for r_key, d_val in debriefs.items():
                    if isinstance(d_val, dict) and d_val.get("subject_model") == ref.expected_model:
                        role = r_key  # type: ignore[assignment]
                        break
            if role not in debriefs:
                errors.append(f"Debrief for role {role!r} not found in episode {ref.locator}")
            else:
                d_entry = debriefs[role]
                actual_role = role
                actual_model = str(d_entry.get("subject_model") or "")
                form_key, field_key = ref.debrief_field.split(".", 1)
                form_obj = d_entry.get(form_key) or {}
                parsed = form_obj.get("parsed") or {}
                val = parsed.get(field_key)
                if val is None:
                    errors.append(f"Debrief field {ref.debrief_field} missing in {ref.locator} ({role})")
                else:
                    source_text = str(val)

    elif ref.source_type == "choice_reason":
        rec, is_excluded = idx.get_choice(ref.locator)
        if rec is None:
            errors.append(f"Choice record not found: {ref.locator}")
        else:
            record_exists = True
            meta_for_numeric = rec
            if is_excluded:
                errors.append(f"Choice record {ref.locator} is in choices_excluded.jsonl")
            s1 = rec.get("seed_1")
            s2 = rec.get("seed_2")
            if s1 is not None and s2 is not None:
                actual_seed = int(s1) if int(s1) == int(s2) else (int(s1), int(s2))
            actual_model = str(rec.get("subject_model") or "")
            actual_role = str(rec.get("role") or "") or None
            source_text = str(rec.get("reason") or "")

    elif ref.source_type == "capstone_reason":
        rec = idx.get_capstone_choice(ref.locator, reviewer=ref.expected_model)
        if rec is None:
            errors.append(f"Capstone choice record not found: {ref.locator} (reviewer={ref.expected_model})")
        else:
            record_exists = True
            meta_for_numeric = rec
            ep1 = idx.get_episode(str(rec.get("episode_1_id") or ""))
            ep2 = idx.get_episode(str(rec.get("episode_2_id") or ""))
            if ep1 and ep2 and ep1.get("seed") is not None and ep2.get("seed") is not None:
                s1, s2 = int(ep1["seed"]), int(ep2["seed"])
                actual_seed = s1 if s1 == s2 else (s1, s2)
            actual_model = str(rec.get("reviewer") or "")
            actual_role = str(rec.get("target_role") or "") or None
            source_text = _extract_capstone_reason(rec)

    elif ref.source_type == "capstone_rewrite":
        # locator format: "<episode_id>::<target_tone>" (e.g. "goalpost__s303__neon4__vs__muse::warm")
        if "::" not in ref.locator:
            errors.append(f"Invalid capstone_rewrite locator: {ref.locator}")
        else:
            ep_id, tone = ref.locator.split("::", 1)
            rec = idx.get_capstone_rewrite(ep_id, tone)
            if rec is None:
                errors.append(f"Capstone rewrite record not found: {ref.locator}")
            else:
                record_exists = True
                meta_for_numeric = rec
                actual_seed = int(rec["seed"]) if rec.get("seed") is not None else None
                actual_model = str(rec.get("partner") or "")
                actual_role = str(rec.get("target_role") or "") or None
                turns_list = rec.get("turns") or []
                matched_turn = None
                for t in turns_list:
                    if ref.expected_round is None or int(t.get("round", -1)) == ref.expected_round:
                        matched_turn = t
                        break
                if matched_turn is None:
                    errors.append(f"Round {ref.expected_round} not found in rewrite {ref.locator}")
                else:
                    actual_round = int(matched_turn["round"])
                    source_text = str(
                        matched_turn.get("rewritten")
                        or matched_turn.get("rewritten_message")
                        or matched_turn.get("text")
                        or ""
                    )

    elif ref.source_type == "exp2_reason":
        # locator format: "<pair_id>::<condition>::<order>::<reviewer>"
        parts = ref.locator.split("::")
        if len(parts) != 4:
            errors.append(f"Invalid exp2_reason locator: {ref.locator}")
        else:
            rec = idx.get_exp2_choice(*parts)
            if rec is None:
                errors.append(f"Exp2 choice record not found: {ref.locator}")
            else:
                record_exists = True
                meta_for_numeric = rec
                s1, s2 = rec.get("seed_1"), rec.get("seed_2")
                if s1 is not None and s2 is not None:
                    actual_seed = int(s1) if int(s1) == int(s2) else (int(s1), int(s2))
                actual_model = str(rec.get("reviewer") or "")
                actual_role = str(rec.get("target_role") or "") or None
                source_text = str(rec.get("rationale") or _extract_capstone_reason(rec))

    else:
        errors.append(f"Unsupported source_type: {ref.source_type}")

    # Metadata comparisons
    seed_match: bool | None = None
    if ref.expected_seed is not None:
        if isinstance(ref.expected_seed, tuple) and isinstance(actual_seed, int):
            seed_match = ref.expected_seed == (actual_seed, actual_seed)
        elif isinstance(ref.expected_seed, int) and isinstance(actual_seed, tuple):
            seed_match = actual_seed == (ref.expected_seed, ref.expected_seed)
        else:
            seed_match = actual_seed == ref.expected_seed
        if not seed_match:
            errors.append(f"Seed mismatch: expected {ref.expected_seed}, actual {actual_seed}")

    model_match: bool | None = None
    if ref.expected_model is not None:
        model_match = actual_model == ref.expected_model
        if not model_match:
            errors.append(f"Model mismatch: expected {ref.expected_model!r}, actual {actual_model!r}")

    role_match: bool | None = None
    if ref.expected_role is not None:
        role_match = actual_role == ref.expected_role
        if not role_match:
            errors.append(f"Role mismatch: expected {ref.expected_role!r}, actual {actual_role!r}")

    round_match: bool | None = None
    if ref.expected_round is not None:
        round_match = actual_round == ref.expected_round
        if not round_match:
            errors.append(f"Round mismatch: expected {ref.expected_round}, actual {actual_round}")

    # Verbatim checks (strict + editorial-bracket-stripped)
    verbatim_match = False
    verbatim_ignoring_editorial = False
    editorial_annotations: tuple[str, ...] = ()
    seg_checked = 0
    seg_matched = 0
    missing_segs: tuple[str, ...] = ()
    actual_word_count: int | None = None

    if record_exists and source_text:
        actual_word_count = len(source_text.split())
        verbatim_match, seg_checked, seg_matched, missing_segs = _ordered_segment_match(
            ref.quote_text, source_text
        )
        stripped_quote, editorial_annotations = _strip_editorial_brackets(ref.quote_text, source_text)
        if verbatim_match:
            verbatim_ignoring_editorial = True
        elif editorial_annotations:
            verbatim_ignoring_editorial, _, _, _ = _ordered_segment_match(stripped_quote, source_text)
        else:
            verbatim_ignoring_editorial = False

    numeric_flags = tuple(check_numeric_claim_consistency(ref.quote_text, meta_for_numeric))

    meta_ok = all(
        chk is not False for chk in (seed_match, model_match, role_match, round_match)
    )
    passed = record_exists and meta_ok and verbatim_match and len(errors) == 0
    passed_editorial = record_exists and meta_ok and verbatim_ignoring_editorial and len(errors) == 0

    return QuoteVerificationResult(
        label=ref.label,
        source_type=ref.source_type,
        locator=ref.locator,
        record_exists=record_exists,
        seed_match=seed_match,
        model_match=model_match,
        role_match=role_match,
        round_match=round_match,
        verbatim_match=verbatim_match,
        verbatim_match_ignoring_editorial_brackets=verbatim_ignoring_editorial,
        editorial_annotations=editorial_annotations,
        segments_checked=seg_checked,
        segments_matched=seg_matched,
        missing_segments=missing_segs,
        actual_seed=actual_seed,
        actual_model=actual_model,
        actual_role=actual_role,
        actual_round=actual_round,
        actual_word_count=actual_word_count,
        numeric_claim_flags=numeric_flags,
        errors=tuple(errors),
        passed=passed,
        passed_with_editorial_annotations=passed_editorial,
    )


def _strip_markdown_blockquote_wrapper(line: str) -> str:
    s = line.lstrip()
    if s.startswith(">"):
        s = s[1:].strip()
    if len(s) >= 2 and s[0] == "*" and s[-1] == "*":
        s = s[1:-1].strip()
    if len(s) >= 2 and s[0] in ('"', "“") and s[-1] in ('"', "”"):
        s = s[1:-1].strip()
    return s


def _audit_quote_cards(quote_cards_path: Path, idx: V2CorpusIndex) -> list[dict[str, Any]]:
    """Audit every blockquote and friction quote in `data/v2/analysis/quote_cards.md`."""
    text = quote_cards_path.read_text(encoding="utf-8")
    sections = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    results: list[dict[str, Any]] = []

    for sec in sections:
        lines = sec.splitlines()
        header = lines[0].strip()
        m_hdr = re.match(r"`([^`]+)`\s+[—-]\s+(\w+)\s+\(([AB])\)\s+reviewing\s+(\w+)", header)
        if not m_hdr:
            continue
        episode_id, subj_model, subj_role, partner_model = m_hdr.groups()
        ep = idx.get_episode(episode_id)
        seed = int(ep["seed"]) if ep and ep.get("seed") is not None else None
        total_rounds = int(ep["total_rounds"]) if ep and ep.get("total_rounds") is not None else None
        partner_role: Literal["A", "B"] = "B" if subj_role == "A" else "A"

        current_section = ""
        face_quotes: list[str] = []
        f2_quote: str | None = None
        f1_quote: str | None = None
        friction_quote: str | None = None

        for raw_line in lines[1:]:
            s = raw_line.strip()
            if s.startswith("- **What it said to their face"):
                current_section = "face"
                continue
            if s.startswith("- **Verbatim public review (F2):"):
                current_section = "f2"
                continue
            if s.startswith("- **Private review (F1):"):
                current_section = "f1"
                continue
            if s.startswith("- **Friction quote"):
                current_section = "friction"
                m_fq = re.search(r"`([^`]+)`", s)
                if m_fq and m_fq.group(1).strip().lower() != "none":
                    friction_quote = m_fq.group(1)
                continue
            if s.startswith(">"):
                q = _strip_markdown_blockquote_wrapper(s)
                if current_section == "face":
                    face_quotes.append(q)
                elif current_section == "f2":
                    f2_quote = q
                elif current_section == "f1":
                    f1_quote = q

        # Face-to-face quotes correspond to the final two rounds of subject_role
        if total_rounds is not None and len(face_quotes) == 2:
            for offset, q in enumerate(face_quotes):
                rnd = total_rounds - 1 + offset
                ref = QuoteReference(
                    source_type="episode_turn",
                    locator=episode_id,
                    quote_text=q,
                    expected_seed=seed,
                    expected_model=subj_model,
                    expected_role=subj_role,  # type: ignore[arg-type]
                    expected_round=rnd,
                    label=f"quote_cards:{episode_id}:R{rnd}:{subj_role}",
                )
                results.append(verify_quote(ref, index=idx).to_dict())

        if f2_quote is not None:
            ref = QuoteReference(
                source_type="debrief_summary",
                locator=episode_id,
                quote_text=f2_quote,
                expected_seed=seed,
                expected_model=subj_model,
                expected_role=subj_role,  # type: ignore[arg-type]
                debrief_field="F2.summary",
                label=f"quote_cards:{episode_id}:F2.summary",
            )
            results.append(verify_quote(ref, index=idx).to_dict())

        if f1_quote is not None:
            ref = QuoteReference(
                source_type="debrief_summary",
                locator=episode_id,
                quote_text=f1_quote,
                expected_seed=seed,
                expected_model=subj_model,
                expected_role=subj_role,  # type: ignore[arg-type]
                debrief_field="F1.summary",
                label=f"quote_cards:{episode_id}:F1.summary",
            )
            results.append(verify_quote(ref, index=idx).to_dict())

        if friction_quote is not None:
            # Verify friction quote appears in F1.friction_quote AND in one of partner's actual turns
            ref_debrief = QuoteReference(
                source_type="debrief_summary",
                locator=episode_id,
                quote_text=friction_quote,
                expected_seed=seed,
                expected_model=subj_model,
                expected_role=subj_role,  # type: ignore[arg-type]
                debrief_field="F1.friction_quote",
                label=f"quote_cards:{episode_id}:F1.friction_quote",
            )
            res_dict = verify_quote(ref_debrief, index=idx).to_dict()
            # Also find which partner turn contains the verbatim friction quote
            matched_partner_round: int | None = None
            if ep is not None:
                fq_norm = normalize_whitespace(friction_quote)
                for turn in ep.get("turns") or []:
                    pmsg = normalize_whitespace(str((turn.get(partner_role) or {}).get("message") or ""))
                    if fq_norm in pmsg:
                        matched_partner_round = int(turn["round"])
                        break
            res_dict["partner_model"] = partner_model
            res_dict["partner_role"] = partner_role
            res_dict["matched_partner_round_exact"] = matched_partner_round
            results.append(res_dict)

    return results


def _strip_editorial_markdown(text: str) -> str:
    """Remove author display-only emphasis (`_..._`) and surrounding quotes from a blog blockquote."""
    s = text.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("“") and s.endswith("”")):
        s = s[1:-1].strip()
    # Unwrap italic underscores added for author emphasis (e.g. `_shared its full..._`)
    s = re.sub(r"(?<![A-Za-z0-9_])_([^_\n]+)_(?![A-Za-z0-9_])", r"\1", s)
    return s


def _audit_blog_quotes(blog_path: Path, idx: V2CorpusIndex) -> dict[str, Any]:
    """Audit every displayed blockquote in `docs/blog_post.md` (and legacy draft quotes if present)."""
    lines = blog_path.read_text(encoding="utf-8").splitlines()
    raw_blockquotes: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(">"):
            start_ln = i + 1
            block_lines: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                stripped = re.sub(r"^\s*>\s?", "", lines[i])
                block_lines.append(stripped)
                i += 1
            joined = "\n".join(block_lines).strip()
            raw_blockquotes.append(
                {
                    "line": start_ln,
                    "raw_line": joined,
                    "extracted_quote": _strip_editorial_markdown(joined),
                }
            )
        else:
            i += 1

    # Map every displayed blockquote in docs/blog_post.md to its exact archive source
    blockquote_specs: list[tuple[int, QuoteReference, str]] = []
    for bq in raw_blockquotes:
        q = bq["extracted_quote"]
        ln = bq["line"]
        if "Keep batch size at 32 or 64 for throughput." in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="episode_turn",
                        locator="goalpost__s302__astra62__vs__opus55",
                        quote_text=q,
                        expected_seed=302,
                        expected_model="astra62",
                        expected_role="A",
                        expected_round=1,
                        label=f"blog_post.md:L{ln}:s302_astra62_opener",
                    ),
                    "Section 2 Candidate A: verbatim GPT-6.2 Astra Round-1 opener (83 words).",
                )
            )
        elif "pool_size divisible by shard_cnt" in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="episode_turn",
                        locator="goalpost__s302__astra60__vs__opus55",
                        quote_text=q,
                        expected_seed=302,
                        expected_model="astra60",
                        expected_role="A",
                        expected_round=1,
                        label=f"blog_post.md:L{ln}:s302_astra60_opener",
                    ),
                    "Section 2 Candidate B: verbatim GPT-6 Astra Round-1 opener (88 words).",
                )
            )
        elif "HPA burst scaling from 10 to 12 replicas" in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="episode_turn",
                        locator="code_handoff__s301__astra62__vs__opus55",
                        quote_text=q,
                        expected_seed=301,
                        expected_model="astra62",
                        expected_role="A",
                        expected_round=4,
                        label=f"blog_post.md:L{ln}:code_handoff_s301_r4_astra62",
                    ),
                    "Section 2 Code Handoff counterexample: verbatim GPT-6.2 Astra Round-4 excerpt.",
                )
            )
        elif "the same rounds DOWN and 17/17 at the end" in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="choice_reason",
                        locator="opus_boost_h2h__obh2h__opus55__goalpost__B__astra62_s306__vs__astra60_s306__BA",
                        quote_text=q,
                        expected_seed=306,
                        expected_model="opus55",
                        expected_role="B",
                        label=f"blog_post.md:L{ln}:opus_s306_BA_rationale",
                    ),
                    "Section 3 Opus s306 BA rationale: verbatim (with display emphasis stripped).",
                )
            )
        elif "Thanks so much for the excellent analysis, EN!" in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="capstone_rewrite",
                        locator="goalpost__s303__neon4__vs__muse::warm",
                        quote_text=q,
                        expected_seed=303,
                        expected_model="neon4",
                        expected_role="A",
                        expected_round=2,
                        label=f"blog_post.md:L{ln}:capstone_warm_rewrite_s303_r2",
                    ),
                    "Section 4 Warm rewrite example: verbatim from capstone_rewrites.",
                )
            )
        elif "EN: Double-checked all 14 constraints against your target configuration." in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="capstone_rewrite",
                        locator="goalpost__s303__neon4__vs__muse::blunt",
                        quote_text=q,
                        expected_seed=303,
                        expected_model="neon4",
                        expected_role="A",
                        expected_round=2,
                        label=f"blog_post.md:L{ln}:capstone_blunt_rewrite_s303_r2",
                    ),
                    "Section 4 Blunt rewrite example: verbatim from capstone_rewrites.",
                )
            )
        elif "optimal 7/7 veto-free result" in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="capstone_reason",
                        locator="cap_p01__calendar__A__tied__p1_warm_p2_blunt__AB",
                        quote_text=q,
                        expected_seed=(301, 303),
                        expected_model="opus55",
                        expected_role="A",
                        label=f"blog_post.md:L{ln}:cap_p01_p1_warm_p2_blunt_AB",
                    ),
                    "Section 4 Capstone cap_p01 (Kimi Warm vs Muse Blunt, AB): verbatim.",
                )
            )
        elif "perfect 7/7, veto-free, zero-regret outcome" in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="capstone_reason",
                        locator="cap_p01__calendar__A__tied__p1_blunt_p2_warm__AB",
                        quote_text=q,
                        expected_seed=(301, 303),
                        expected_model="opus55",
                        expected_role="A",
                        label=f"blog_post.md:L{ln}:cap_p01_p1_blunt_p2_warm_AB",
                    ),
                    "Section 4 Capstone cap_p01 (Kimi Blunt vs Muse Warm, AB): verbatim.",
                )
            )
        elif "HR kept its evidence well calibrated" in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="choice_reason",
                        locator="main_7p__m7p__opus55__triage__B__astra62_s304__vs__neon4_s303__BA",
                        quote_text=q,
                        expected_seed=(303, 304),
                        expected_model="opus55",
                        expected_role="B",
                        label=f"blog_post.md:L{ln}:opus_triage_upset_rationale",
                    ),
                    "Section 6 Incident Triage upset rationale: verbatim (with display emphasis stripped).",
                )
            )
        elif "contradicted its own statement about keeping constraints private" in q:
            blockquote_specs.append(
                (
                    ln,
                    QuoteReference(
                        source_type="exp2_reason",
                        locator="exp2_m20__goalpost__A__s303::p1_echo_p2_info::AB::neon4",
                        quote_text=q,
                        expected_seed=303,
                        expected_model="neon4",
                        expected_role="A",
                        label=f"blog_post.md:L{ln}:exp2_m20_neon4_reversal",
                    ),
                    "Appendix C exp2_m20 sole reversal rationale: verbatim.",
                )
            )

    audited_blockquotes: list[dict[str, Any]] = []
    for ln, ref, note in blockquote_specs:
        res = verify_quote(ref, index=idx).to_dict()
        res["blog_line"] = ln
        res["audit_note"] = note
        audited_blockquotes.append(res)

    return {
        "total_formal_blockquotes_found": len(raw_blockquotes),
        "audited_blockquotes": audited_blockquotes,
        "audited_inline_quotes": [],
    }


def _audit_v2_1_candidate_quotes(idx: V2CorpusIndex) -> dict[str, Any]:
    """Verify key v2.1 replacement & centerpiece quotes directly against raw v2 data."""
    # 1. Reverse-order Opus s306 rationale (BA) that gets 17/17 right, plus all matched s301-s308 ties
    opus_goalpost_ties: list[dict[str, Any]] = []
    for rec in idx.iter_choices():
        if (
            rec.get("task") == "goalpost"
            and rec.get("subject_model") == "opus55"
            and rec.get("same_seed") is True
            and {rec.get("partner_1"), rec.get("partner_2")} == {"astra60", "astra62"}
            and rec.get("score_1") == rec.get("score_2") == 1.0
        ):
            reason = str(rec.get("reason") or "")
            ref = QuoteReference(
                source_type="choice_reason",
                locator=str(rec["manifest_item_id"]),
                quote_text=reason,
                expected_seed=int(rec["seed_1"]),
                expected_model="opus55",
                expected_role=rec["role"],
                label=f"opus_goalpost_tie:s{rec['seed_1']}:{rec['role']}:{rec['order']}",
            )
            ver = verify_quote(ref, index=idx).to_dict()
            ver["chosen_partner"] = rec.get("chosen_partner")
            ver["chosen_handle"] = rec.get("chosen_handle")
            ver["partner_1"] = rec.get("partner_1")
            ver["handle_1"] = rec.get("handle_1")
            ver["partner_2"] = rec.get("partner_2")
            ver["handle_2"] = rec.get("handle_2")
            ver["episode_1_id"] = rec.get("episode_1_id")
            ver["episode_2_id"] = rec.get("episode_2_id")
            ver["reason_text"] = reason
            opus_goalpost_ties.append(ver)

    opus_goalpost_ties.sort(key=lambda x: (int(x["actual_seed"] or 0), str(x["actual_role"]), str(x["locator"])))

    # Spotlight exact s306 BA replacement
    s306_ba = next(
        (
            r
            for r in opus_goalpost_ties
            if r["locator"] == "opus_boost_h2h__obh2h__opus55__goalpost__B__astra62_s306__vs__astra60_s306__BA"
        ),
        None,
    )

    # 2. Opening hook: 83-word GPT-6.2 vs 88-word GPT-6 on goalpost__s302 (Role A vs Opus)
    ep_s302_62 = idx.get_episode("goalpost__s302__astra62__vs__opus55")
    ep_s302_60 = idx.get_episode("goalpost__s302__astra60__vs__opus55")
    msg_s302_62 = str(ep_s302_62["turns"][0]["A"]["message"]) if ep_s302_62 else ""
    msg_s302_60 = str(ep_s302_60["turns"][0]["A"]["message"]) if ep_s302_60 else ""

    hook_62 = verify_quote(
        QuoteReference(
            source_type="episode_turn",
            locator="goalpost__s302__astra62__vs__opus55",
            quote_text=msg_s302_62,
            expected_seed=302,
            expected_model="astra62",
            expected_role="A",
            expected_round=1,
            label="v2_1_hook:goalpost__s302__astra62__vs__opus55:R1:A",
        ),
        index=idx,
    ).to_dict()
    hook_62["quote_text"] = msg_s302_62
    hook_62["episode_score"] = ep_s302_62.get("score") if ep_s302_62 else None

    hook_60 = verify_quote(
        QuoteReference(
            source_type="episode_turn",
            locator="goalpost__s302__astra60__vs__opus55",
            quote_text=msg_s302_60,
            expected_seed=302,
            expected_model="astra60",
            expected_role="A",
            expected_round=1,
            label="v2_1_hook:goalpost__s302__astra60__vs__opus55:R1:A",
        ),
        index=idx,
    ).to_dict()
    hook_60["quote_text"] = msg_s302_60
    hook_60["episode_score"] = ep_s302_60.get("score") if ep_s302_60 else None

    # 3. Code Handoff counterexample: GPT-6.2 132-word Round-4 turn on code_handoff__s301__astra62__vs__opus55
    ep_ch_62 = idx.get_episode("code_handoff__s301__astra62__vs__opus55")
    msg_ch_r4 = str(ep_ch_62["turns"][3]["A"]["message"]) if ep_ch_62 else ""
    ch_counterexample = verify_quote(
        QuoteReference(
            source_type="episode_turn",
            locator="code_handoff__s301__astra62__vs__opus55",
            quote_text=msg_ch_r4,
            expected_seed=301,
            expected_model="astra62",
            expected_role="A",
            expected_round=4,
            label="v2_1_counterexample:code_handoff__s301__astra62__vs__opus55:R4:A",
        ),
        index=idx,
    ).to_dict()
    ch_counterexample["quote_text"] = msg_ch_r4
    ch_counterexample["episode_score"] = ep_ch_62.get("score") if ep_ch_62 else None

    # 4. Verbatim Kimi inline replacements (s301 R4 A, s303 R2 A, s304 R10 A)
    verbatim_kimi_replacements: list[dict[str, Any]] = []
    for ep_id, seed, rnd, role, snippet, lbl in [
        (
            "calendar__s301__kimi31__vs__qwen4",
            301,
            4,
            "A",
            "My full proposed list is below in the ACTION line.",
            "kimi_verbatim:calendar__s301__kimi31__vs__qwen4:R4:A",
        ),
        (
            "calendar__s303__kimi31__vs__astra60",
            303,
            2,
            "A",
            "My proposed schedule is in the ACTION line after my message; its total penalty for me is 2.",
            "kimi_verbatim:calendar__s303__kimi31__vs__astra60:R2:A",
        ),
        (
            "calendar__s304__kimi31__vs__kimi31",
            304,
            10,
            "A",
            "Confirmed. Submitting the identical, mutually valid, veto-free schedule with combined penalty 2.",
            "kimi_verbatim:calendar__s304__kimi31__vs__kimi31:R10:A",
        ),
    ]:
        v = verify_quote(
            QuoteReference(
                source_type="episode_turn",
                locator=ep_id,
                quote_text=snippet,
                expected_seed=seed,
                expected_model="kimi31",
                expected_role=role,  # type: ignore[arg-type]
                expected_round=rnd,
                label=lbl,
            ),
            index=idx,
        ).to_dict()
        v["quote_text"] = snippet
        verbatim_kimi_replacements.append(v)

    return {
        "opus_s306_reverse_order_ba": s306_ba,
        "opus_goalpost_matched_ties_s301_s308": opus_goalpost_ties,
        "goalpost_s302_opening_hook": {
            "astra62_r1_role_a": hook_62,
            "astra60_r1_role_a": hook_60,
        },
        "code_handoff_s301_r4_counterexample": ch_counterexample,
        "kimi_verbatim_calendar_quotes": verbatim_kimi_replacements,
    }


def audit_blog_and_quote_cards(
    *,
    root_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the complete v2.1 quote audit over `BLOG_POST_V2.md`, `quote_cards.md`, and key candidates."""
    root = Path(root_dir) if root_dir is not None else ROOT_DIR
    d_dir = Path(data_dir) if data_dir is not None else (root / "data" / "v2")
    idx = V2CorpusIndex(data_dir=d_dir)

    blog_path = root / "docs" / "blog_post.md"
    if not blog_path.exists():
        blog_path = root / "BLOG_POST_V2.md"
    blog_audit = _audit_blog_quotes(blog_path, idx)
    qc_path = d_dir / "analysis" / "quote_cards.md"
    quote_cards_audit = _audit_quote_cards(qc_path, idx) if qc_path.is_file() else []
    v2_1_candidates = _audit_v2_1_candidate_quotes(idx)

    qc_passed = sum(1 for r in quote_cards_audit if r["passed"])
    qc_numeric_flags = sum(len(r["numeric_claim_flags"]) for r in quote_cards_audit)
    blog_bq = blog_audit["audited_blockquotes"]
    blog_inline = blog_audit["audited_inline_quotes"]
    ties = v2_1_candidates["opus_goalpost_matched_ties_s301_s308"]
    ties_clean = [t for t in ties if t["passed"] and len(t["numeric_claim_flags"]) == 0]
    ties_flagged = [t for t in ties if len(t["numeric_claim_flags"]) > 0]

    report: dict[str, Any] = {
        "schema_version": "quote_audit_v2_1",
        "summary": {
            "blog_formal_blockquotes_total": len(blog_bq),
            "blog_formal_blockquotes_strict_verbatim": sum(1 for r in blog_bq if r["passed"]),
            "blog_formal_blockquotes_verbatim_ignoring_editorial_brackets": sum(
                1 for r in blog_bq if r["passed_with_editorial_annotations"]
            ),
            "blog_formal_blockquotes_with_numeric_flags": sum(
                1 for r in blog_bq if len(r["numeric_claim_flags"]) > 0
            ),
            "blog_inline_quotes_audited": len(blog_inline),
            "blog_inline_quotes_strict_verbatim": sum(1 for r in blog_inline if r["passed"]),
            "quote_cards_items_total": len(quote_cards_audit),
            "quote_cards_items_passed": qc_passed,
            "quote_cards_items_numeric_flags": qc_numeric_flags,
            "opus_goalpost_matched_ties_total": len(ties),
            "opus_goalpost_matched_ties_clean": len(ties_clean),
            "opus_goalpost_matched_ties_numeric_flagged": len(ties_flagged),
            "goalpost_s302_hook_astra62_words": v2_1_candidates["goalpost_s302_opening_hook"][
                "astra62_r1_role_a"
            ]["actual_word_count"],
            "goalpost_s302_hook_astra60_words": v2_1_candidates["goalpost_s302_opening_hook"][
                "astra60_r1_role_a"
            ]["actual_word_count"],
            "code_handoff_s301_r4_astra62_words": v2_1_candidates[
                "code_handoff_s301_r4_counterexample"
            ]["actual_word_count"],
        },
        "blog_post_v2_audit": blog_audit,
        "quote_cards_audit": quote_cards_audit,
        "v2_1_candidate_quotes": v2_1_candidates,
    }

    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return report
