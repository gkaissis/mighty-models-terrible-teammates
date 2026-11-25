"""Append-only crash-resumable JSONL ledger with concurrency-safe budget reservation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BudgetExceededError(RuntimeError):
    """Raised before sending an API call that would breach the configured spend cap."""


@dataclass
class LedgerEntry:
    call_id: str
    timestamp_utc: str
    model_key: str
    requested_slug: str
    returned_model: str | None
    provider: str | None
    reasoning_effort: str | None
    max_tokens: int
    status: str  # "ok" | "truncated" | "empty" | "error"
    finish_reason: str | None
    attempt: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    cached_tokens: int
    cost_usd: float
    latency_ms: int
    request_messages: list[dict[str, Any]]
    response_text: str
    reasoning_text: str | None = None
    error_detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerEntry:
        return cls(**d)


class Ledger:
    """Async-safe append-only JSONL ledger with deterministic call_id dedup & in-flight budget reservations."""

    def __init__(self, path: Path, budget_cap_usd: float) -> None:
        self.path = path
        self.budget_cap_usd = budget_cap_usd
        self._lock = asyncio.Lock()
        self._ok_by_id: dict[str, LedgerEntry] = {}
        self._entries_by_id: dict[str, list[LedgerEntry]] = {}
        self._all_entries: list[LedgerEntry] = []
        self._total_cost_usd: float = 0.0
        self._reservations: dict[str, float] = {}
        self._reserved_cost_usd: float = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    entry = LedgerEntry.from_dict(raw)
                except Exception as exc:
                    raise ValueError(
                        f"Corrupted JSONL line {line_no} in {self.path}: {exc}"
                    ) from exc
                self._all_entries.append(entry)
                self._entries_by_id.setdefault(entry.call_id, []).append(entry)
                self._total_cost_usd += float(entry.cost_usd or 0.0)
                if entry.status == "ok":
                    self._ok_by_id[entry.call_id] = entry

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def reserved_cost_usd(self) -> float:
        return self._reserved_cost_usd

    @property
    def all_entries(self) -> list[LedgerEntry]:
        return list(self._all_entries)

    def get_completed(self, call_id: str) -> LedgerEntry | None:
        return self._ok_by_id.get(call_id)

    def total_cost_for_call_ids(self, call_ids: list[str]) -> float:
        """Sum cost_usd across all recorded attempts (including failed retries) for the given call IDs."""
        total = 0.0
        for cid in set(call_ids):
            for e in self._entries_by_id.get(cid, ()):
                total += float(e.cost_usd or 0.0)
        return round(total, 8)

    async def check_budget(self) -> None:
        async with self._lock:
            if self._total_cost_usd + self._reserved_cost_usd >= self.budget_cap_usd:
                raise BudgetExceededError(
                    f"Spend cap reached: committed=${self._total_cost_usd:.4f} + "
                    f"reserved=${self._reserved_cost_usd:.4f} >= ${self.budget_cap_usd:.2f}"
                )

    async def reserve_budget(self, reservation_id: str, worst_case_usd: float) -> None:
        """Reserve worst-case in-flight call cost before dispatch so concurrent workers cannot overshoot."""
        est = max(0.0, float(worst_case_usd))
        async with self._lock:
            if reservation_id in self._reservations:
                self._reserved_cost_usd -= self._reservations.pop(reservation_id)
            projected = self._total_cost_usd + self._reserved_cost_usd + est
            if projected > self.budget_cap_usd or (
                est == 0.0 and self._total_cost_usd + self._reserved_cost_usd >= self.budget_cap_usd
            ):
                raise BudgetExceededError(
                    f"Spend cap would be exceeded: committed=${self._total_cost_usd:.4f} + "
                    f"reserved=${self._reserved_cost_usd:.4f} + est=${est:.4f} > ${self.budget_cap_usd:.2f}"
                )
            self._reservations[reservation_id] = est
            self._reserved_cost_usd += est

    async def release_reservation(self, reservation_id: str) -> None:
        async with self._lock:
            amt = self._reservations.pop(reservation_id, 0.0)
            self._reserved_cost_usd = max(0.0, self._reserved_cost_usd - amt)

    async def record(self, entry: LedgerEntry, *, reservation_id: str | None = None) -> None:
        async with self._lock:
            if reservation_id is not None:
                amt = self._reservations.pop(reservation_id, 0.0)
                self._reserved_cost_usd = max(0.0, self._reserved_cost_usd - amt)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(entry.to_json() + "\n")
                fh.flush()
            self._all_entries.append(entry)
            self._entries_by_id.setdefault(entry.call_id, []).append(entry)
            self._total_cost_usd += float(entry.cost_usd or 0.0)
            if entry.status == "ok":
                self._ok_by_id[entry.call_id] = entry

    def summary_by_model(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for e in self._all_entries:
            s = stats.setdefault(
                e.model_key,
                {
                    "calls_total": 0,
                    "calls_ok": 0,
                    "calls_truncated": 0,
                    "calls_error": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "reasoning_tokens_missing_calls": 0,
                    "cached_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            s["calls_total"] += 1
            if e.status == "ok":
                s["calls_ok"] += 1
            elif e.status == "truncated":
                s["calls_truncated"] += 1
            else:
                s["calls_error"] += 1
            s["prompt_tokens"] += e.prompt_tokens
            s["completion_tokens"] += e.completion_tokens
            if e.reasoning_tokens is None:
                s["reasoning_tokens_missing_calls"] += 1
            else:
                s["reasoning_tokens"] += e.reasoning_tokens
            s["cached_tokens"] += e.cached_tokens
            s["cost_usd"] += e.cost_usd
        return stats


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
