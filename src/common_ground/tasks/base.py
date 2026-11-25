"""Shared task interface, blind handle generation, and free-text + ACTION wire parser."""

from __future__ import annotations

import hashlib
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["A", "B"]

# Task-disjoint neutral two-letter handles (zero collision with task tokens like C-R3, G-1, Q-1, R1..R10, etc.)
BLIND_HANDLES: tuple[str, ...] = ("AL", "BK", "CT", "DM", "EN", "FP", "GQ", "HR", "JT", "KV")


def make_blind_handles(seed_key: str) -> tuple[str, str]:
    """Deterministically generate two distinct neutral two-letter handles per episode."""
    h = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(h)
    h_a, h_b = rng.sample(list(BLIND_HANDLES), 2)
    return h_a, h_b


_ACTION_MARKER_RE = re.compile(r"(?im)^\s*ACTION\s*:\s*")


def _strip_earlier_action_blocks(text: str) -> str:
    """Remove any duplicate earlier line-start `ACTION: {...}` blocks from visible prose."""
    decoder = json.JSONDecoder()
    lines_out: list[str] = []
    pos = 0
    n = len(text)
    for m in _ACTION_MARKER_RE.finditer(text):
        if m.start() < pos:
            continue
        tail = text[m.end() :]
        stripped = tail.lstrip()
        fence_offset = len(tail) - len(stripped)
        end_pos: int | None = None

        if stripped.startswith("```"):
            m_fence = re.match(r"^```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
            if m_fence:
                try:
                    obj, _ = decoder.raw_decode(m_fence.group(1))
                    if isinstance(obj, dict):
                        end_pos = m.end() + fence_offset + m_fence.end()
                except json.JSONDecodeError:
                    pass
        elif stripped.startswith("{"):
            try:
                obj, rel_end = decoder.raw_decode(stripped)
                if isinstance(obj, dict):
                    end_pos = m.end() + fence_offset + rel_end
            except json.JSONDecodeError:
                pass

        if end_pos is not None:
            lines_out.append(text[pos : m.start()].rstrip())
            pos = end_pos
            # Consume trailing newline after stripped block
            while pos < n and text[pos] in " \t\r\n":
                pos += 1
    lines_out.append(text[pos:])
    cleaned = "\n".join(part for part in lines_out if part.strip())
    return cleaned.strip()


def parse_turn_response(raw_text: str) -> tuple[str, dict[str, Any] | None, str | None]:
    """Split a model's response into untouched free-text prose and trailing JSON ACTION object.

    Expected format:
        <free-text message to teammate>

        ACTION: {"key": "value"}

    Safety guarantees:
    - Strips any duplicate earlier line-start `ACTION: {...}` line/block from visible prose so
      structured actions never leak into teammate-visible text.
    - Rejects unexpected trailing prose after the final `ACTION:` JSON object (triggering a clean retry).
    - Tolerates markdown code fences around the JSON block after `ACTION:`.
    """
    if not raw_text or not raw_text.strip():
        return "", None, "empty response"

    matches = list(_ACTION_MARKER_RE.finditer(raw_text))
    if not matches:
        inline = list(re.finditer(r"ACTION\s*:\s*", raw_text))
        if not inline:
            return raw_text.strip(), None, "missing ACTION: marker"
        match = inline[-1]
    else:
        match = matches[-1]

    raw_message = raw_text[: match.start()].strip()
    message = _strip_earlier_action_blocks(raw_message) if len(matches) > 1 else raw_message
    tail = raw_text[match.end() :].strip()

    # Strip optional code fence around the JSON payload
    fenced = False
    if tail.startswith("```"):
        fenced = True
        tail = re.sub(r"^```(?:json)?\s*", "", tail)

    brace_idx = tail.find("{")
    if brace_idx == -1:
        return message, None, "no JSON object found after ACTION:"
    if tail[:brace_idx].strip():
        return message, None, "unexpected text between ACTION: and JSON object"

    decoder = json.JSONDecoder()
    try:
        obj, end_rel = decoder.raw_decode(tail[brace_idx:])
    except json.JSONDecodeError as exc:
        return message, None, f"invalid JSON after ACTION: {exc}"

    if not isinstance(obj, dict):
        return message, None, f"ACTION must be a JSON object, got {type(obj).__name__}"

    remainder = tail[brace_idx + end_rel :].strip()
    if fenced and remainder.startswith("```"):
        remainder = remainder[3:].strip()
    if remainder:
        return message, None, "unexpected trailing text after ACTION JSON"

    return message, obj, None


@dataclass
class RoundFeedback:
    round_num: int
    feedback_a: str
    feedback_b: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskScore:
    score: float  # Normalized [0.0, 1.0]
    details: dict[str, Any] = field(default_factory=dict)


class TaskEnv(ABC):
    """Abstract base class for multi-round deterministic collaborative environments."""

    task_name: str
    total_rounds: int = 10

    def __init__(self, seed: int) -> None:
        self.seed = seed

    @abstractmethod
    def system_briefing(self, role: Role, my_handle: str, partner_handle: str) -> str:
        """Role-specific private sheet and task rules (no model names, no style hints)."""

    @abstractmethod
    def turn_prompt(
        self,
        role: Role,
        round_num: int,
        partner_handle: str,
        partner_message: str | None,
        last_feedback: str | None,
    ) -> str:
        """Per-round user message delivering partner's message, verifier feedback, and round state."""

    @abstractmethod
    def step_round(
        self,
        round_num: int,
        action_a: dict[str, Any] | None,
        action_b: dict[str, Any] | None,
        msg_a: str,
        msg_b: str,
    ) -> RoundFeedback:
        """Apply both agents' actions for round_num and return per-role verifier feedback."""

    @abstractmethod
    def final_score(self) -> TaskScore:
        """Compute deterministic final score after final round."""

    @abstractmethod
    def oracle_actions(self, round_num: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return exact optimal (action_a, action_b) for unit-testing oracle = 1.0."""
