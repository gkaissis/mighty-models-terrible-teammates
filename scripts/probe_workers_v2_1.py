"""v2.1 OpenRouter worker slug discovery, endpoint pinning, and reasoning-tier probe.

Discovers exact live OpenRouter slugs for the 4 Experiment 1 worker receivers across 4 labs
(Gemini 4 Hydrogen, Claude Haiku 5.5, DeepSeek V4.1 Flash, GPT-6 Luna), selects clean
standard-tier providers, probes each worker across its ordered reasoning-effort step-up ladder,
and copies the 4 frontier receiver pins from `data/v2/probe_manifest_v2.json` into a single
self-contained Exp 1 receiver manifest (`data/v2/probe_manifest_v2_1.json`).

Supports `--mock` mode via `httpx.MockTransport` for zero-cost offline verification without
touching `data/v2/probe_manifest_v2_1.json` unless `--out` is explicitly provided.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from common_ground.config import (
    DATA_DIR,
    DEFAULT_MODELS,
    OPENROUTER_BASE_URL,
    PROBE_MANIFEST_V2_PATH,
    load_api_key,
)

console = Console()

PROBE_MANIFEST_V2_1_PATH = DATA_DIR / "v2" / "probe_manifest_v2_1.json"

FRONTIER_RECEIVER_KEYS: tuple[str, ...] = ("astra62", "neon4", "opus55", "astra60")
WORKER_RECEIVER_KEYS: tuple[str, ...] = ("hydrogen", "haiku55", "deepseek_flash", "gpt6_luna")
FRONTIER_CONCURRENCY = 16
DEFAULT_WORKER_MAX_TOKENS = 16384

PROBE_PROMPT = 'Return ONLY JSON: {"ans": 17}'
PROBE_EXPECTED_ANS = 17
PROBE_MAX_TOKENS = 256
PROBE_TRUNCATION_RETRY_MAX_TOKENS = 1024
DEFAULT_BUDGET_CAP_USD = 1.0

# Hard-disqualified slug route/SKU tokens (never selected for worker pins)
_DISQUALIFIED_ROUTE_TAGS: frozenset[str] = frozenset(
    {"free", "online", "nitro", "floor", "exacto", "extended"}
)
_DISQUALIFIED_SKU_TOKENS: frozenset[str] = frozenset(
    {
        "free",
        "pro",
        "max",
        "ultra",
        "mini",
        "nano",
        "lite",
        "highspeed",
        "batch",
        "audio",
        "vision",
        "realtime",
        "codex",
        "coder",
    }
)
_DISQUALIFIED_SERVICE_TIERS: frozenset[str] = frozenset(
    {"highspeed", "batch", "priority", "fast", "ultrafast"}
)
_FP8_PLUS_QUANTS: frozenset[str] = frozenset(
    {"fp32", "bf16", "fp16", "fp8", "fp8_e4m3", "fp8_e5m2", "int8"}
)
_SUB_FP8_RE = re.compile(r"(?:(?:fp|int|nf|q|w)[1-6](?!\d)|mxfp4|awq|gptq)")

# Exact date-pinned slug suffixes: -20261017, -2026-10-17, -1017, -0813, -10-2026
_DATE_YYYYMMDD_RE = re.compile(r"^-(20\d{2})((?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))$")
_DATE_ISO_RE = re.compile(r"^-(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
_DATE_MMDD_RE = re.compile(r"^-((?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))$")
_DATE_MM_YYYY_RE = re.compile(r"^-(0[1-9]|1[0-2])-(20\d{2})$")


@dataclass(frozen=True)
class WorkerTarget:
    key: str
    display_name: str
    family: str
    expected_canonical_slug: str
    author_prefix: str
    id_regex: str
    name_regex: str
    near_tokens: tuple[str, ...]
    preferred_providers: tuple[str, ...]
    effort_ladder: tuple[str, ...]
    concurrency: int
    supports_anthropic_cache_control: bool = False
    require_fp8_plus_fallback: bool = False
    probe_none_diagnostic: bool = False
    aa_index_v5_by_tier: dict[str, float | None] = field(default_factory=dict)
    aa_index_v5_notes: str = ""


WORKER_TARGETS: dict[str, WorkerTarget] = {
    "hydrogen": WorkerTarget(
        key="hydrogen",
        display_name="Gemini 4 Hydrogen",
        family="google",
        expected_canonical_slug="google/gemini-4-hydrogen",
        author_prefix="google/",
        id_regex=r"^google/gemini-4(?:\.0)?-hydrogen(?=$|[-:])",
        name_regex=r"\bgemini\s*4(?:\.0)?\s*hydrogen\b",
        near_tokens=("hydrogen", "gemini-4"),
        preferred_providers=("Google AI Studio", "Google", "Google Vertex"),
        effort_ladder=("low", "medium", "high"),
        concurrency=24,
        supports_anthropic_cache_control=False,
        aa_index_v5_by_tier={"low": 27.0, "medium": 41.0, "high": 51.0},
        aa_index_v5_notes="AA v5: 27 (low), 41 (medium), 51 (high)",
    ),
    "haiku55": WorkerTarget(
        key="haiku55",
        display_name="Claude Haiku 5.5",
        family="anthropic",
        expected_canonical_slug="anthropic/claude-haiku-5.5",
        author_prefix="anthropic/",
        id_regex=r"^anthropic/claude-(?:haiku-5[.-]5|5[.-]5-haiku)(?=$|[-:])",
        name_regex=r"\bclaude\s*(?:haiku\s*5\.5|5\.5\s*haiku)\b",
        near_tokens=("haiku", "5.5"),
        preferred_providers=("Anthropic",),
        effort_ladder=("low", "medium", "high"),
        concurrency=24,
        supports_anthropic_cache_control=True,
        aa_index_v5_by_tier={"low": 19.0, "medium": 25.0, "high": 30.0},
        aa_index_v5_notes="AA v5: 19 (low), 25 (medium), 30 (high)",
    ),
    "deepseek_flash": WorkerTarget(
        key="deepseek_flash",
        display_name="DeepSeek V4.1 Flash",
        family="deepseek",
        expected_canonical_slug="deepseek/deepseek-v4.1-flash",
        author_prefix="deepseek/",
        id_regex=r"^deepseek/deepseek-(?:chat-)?v?4[.-]1-flash(?=$|[-:])",
        name_regex=r"\bdeepseek\s*v?4\.1\s*flash\b",
        near_tokens=("v4.1", "v4-1", "flash"),
        preferred_providers=("DeepSeek", "BaseTen", "Baseten"),
        effort_ladder=("disabled", "low", "medium", "high"),
        concurrency=32,
        supports_anthropic_cache_control=False,
        require_fp8_plus_fallback=True,
        aa_index_v5_by_tier={"disabled": 19.0, "low": None, "medium": None, "high": 32.5},
        aa_index_v5_notes="AA v5: 19 (non-reasoning), ~32-33 (max)",
    ),
    "gpt6_luna": WorkerTarget(
        key="gpt6_luna",
        display_name="GPT-6 Luna",
        family="openai",
        expected_canonical_slug="openai/gpt-6-luna",
        author_prefix="openai/",
        id_regex=r"^openai/gpt-6(?:\.0)?-luna(?=$|[-:])",
        name_regex=r"\bgpt-?6(?:\.0)?\s*luna\b",
        near_tokens=("luna", "gpt-6"),
        preferred_providers=("OpenAI",),
        effort_ladder=("low", "medium", "high"),
        concurrency=24,
        supports_anthropic_cache_control=False,
        probe_none_diagnostic=True,
        aa_index_v5_by_tier={"low": None, "medium": 18.0, "high": 28.0},
        aa_index_v5_notes="AA v5: unscored (low), 18 (medium), 28 (max)",
    ),
}


@dataclass
class BudgetState:
    cap_usd: float
    total_cost_usd: float = 0.0
    stopped: bool = False

    def can_spend(self) -> bool:
        if self.total_cost_usd >= self.cap_usd:
            self.stopped = True
            return False
        return True

    def add_cost(self, cost: float) -> None:
        self.total_cost_usd += max(0.0, float(cost))
        if self.total_cost_usd >= self.cap_usd:
            self.stopped = True


def _parse_date_suffix(rem: str) -> tuple[int, int] | None:
    """Return comparable `(year, mmdd)` if `rem` is an exact date-pinned revision suffix."""
    m = _DATE_YYYYMMDD_RE.match(rem)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = _DATE_ISO_RE.match(rem)
    if m:
        return (int(m.group(1)), int(m.group(2) + m.group(3)))
    m = _DATE_MMDD_RE.match(rem)
    if m:
        return (0, int(m.group(1)))
    m = _DATE_MM_YYYY_RE.match(rem)
    if m:
        return (int(m.group(2)), int(m.group(1)) * 100)
    return None


def _classify_slug_candidate(
    target: WorkerTarget,
    slug: str,
    rem: str,
) -> tuple[bool, int, tuple[int, int, str], str]:
    """Classify a matched catalog slug deterministically.

    Returns `(accepted, rank, sub_sort_key, reason)` where lower `(rank, sub_sort_key)` wins.
    """
    slug_l = slug.strip().lower()
    rem_l = rem.strip().lower()

    # Route suffix after ":" (e.g. :free, :beta, :nitro)
    route_tag = ""
    if ":" in slug_l:
        route_tag = slug_l.rsplit(":", 1)[1]
    if route_tag in _DISQUALIFIED_ROUTE_TAGS:
        return (
            False,
            99,
            (0, 0, slug_l),
            f"Disqualified route tag ':{route_tag}' (never pin free/routing modifiers).",
        )

    tail = slug_l.split("/", 1)[1] if "/" in slug_l else slug_l
    tail_tokens = [tok for tok in re.split(r"[-:/_]+", tail) if tok]
    rem_tokens = [tok for tok in re.split(r"[-:/_]+", rem_l) if tok]
    bad_skus = [tok for tok in tail_tokens if tok in _DISQUALIFIED_SKU_TOKENS]
    if bad_skus:
        return (
            False,
            99,
            (0, 0, slug_l),
            f"Disqualified variant/SKU token(s) {bad_skus} (prefer base worker slug).",
        )

    if slug_l == target.expected_canonical_slug.lower():
        return (
            True,
            0,
            (0, 0, slug_l),
            "Exact canonical base slug (non-free, non-pro, non-beta).",
        )

    if rem_l == "":
        return (
            True,
            1,
            (0, 0, slug_l),
            "Canonical alias base slug (non-free, non-pro, non-beta).",
        )

    parsed_date = _parse_date_suffix(rem_l)
    if parsed_date is not None:
        year, mmdd = parsed_date
        return (
            True,
            2,
            (-year, -mmdd, slug_l),
            "Exact date-pinned revision (non-free, non-pro, non-beta; latest date preferred).",
        )

    if route_tag == "beta" or "beta" in rem_tokens or "alpha" in rem_tokens:
        return (
            True,
            4,
            (0, 0, slug_l),
            "Beta/alpha fallback (used only if no stable canonical or dated slug exists).",
        )

    if "preview" in rem_tokens or "exp" in rem_tokens or "experimental" in rem_tokens:
        return (
            True,
            3,
            (0, 0, slug_l),
            "Preview/experimental revision (no stable canonical or dated base slug available).",
        )

    return (
        True,
        5,
        (0, 0, slug_l),
        f"Fallback base-prefix match with suffix '{rem_l}'.",
    )


def discover_worker_slug(
    target: WorkerTarget,
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Discover and deterministically select the best OpenRouter catalog slug for a worker."""
    id_re = re.compile(target.id_regex, re.IGNORECASE)
    name_re = re.compile(target.name_regex, re.IGNORECASE)

    matched_entries: list[tuple[str, dict[str, Any], str]] = []
    for slug in sorted(catalog):
        entry = catalog[slug]
        m = id_re.match(slug)
        if m:
            matched_entries.append((slug, entry, slug[m.end() :]))

    # Secondary fallback: same author + display-name match if ID pattern found nothing
    if not matched_entries:
        for slug in sorted(catalog):
            if not slug.lower().startswith(target.author_prefix.lower()):
                continue
            entry = catalog[slug]
            name_str = str(entry.get("name") or "")
            if name_re.search(name_str):
                rem = "" if slug.lower() == target.expected_canonical_slug.lower() else (
                    ":" + slug.rsplit(":", 1)[1] if ":" in slug else ""
                )
                matched_entries.append((slug, entry, rem))

    near_matches = sorted(
        slug
        for slug in catalog
        if slug.lower().startswith(target.author_prefix.lower())
        and any(tok in slug.lower() for tok in target.near_tokens)
    )[:12]

    evaluated: list[dict[str, Any]] = []
    accepted_ranked: list[tuple[int, tuple[int, int, str], str, str]] = []
    for slug, entry, rem in matched_entries:
        accepted, rank, sub_key, reason = _classify_slug_candidate(target, slug, rem)
        evaluated.append(
            {
                "id": slug,
                "name": entry.get("name"),
                "accepted": accepted,
                "rank": rank,
                "reason": reason,
            }
        )
        if accepted:
            accepted_ranked.append((rank, sub_key, slug, reason))

    if not accepted_ranked:
        return {
            "found": False,
            "selected_slug": None,
            "selection_reason": "No accepted non-free/non-pro catalog slug matched target pattern.",
            "all_catalog_matches": evaluated,
            "near_matches": near_matches,
        }

    accepted_ranked.sort(key=lambda item: (item[0], item[1]))
    best_rank, _, best_slug, best_reason = accepted_ranked[0]
    if best_rank == 2 and len([x for x in accepted_ranked if x[0] == 2]) > 1:
        best_reason = (
            "Latest exact date-pinned revision (multiple dated revisions, no undated base slug)."
        )

    return {
        "found": True,
        "selected_slug": best_slug,
        "selection_reason": best_reason,
        "all_catalog_matches": evaluated,
        "near_matches": near_matches,
    }


def _endpoint_pricing_per_m(ep: dict[str, Any]) -> tuple[float | None, float | None]:
    pricing = ep.get("pricing")
    if not isinstance(pricing, dict):
        return (None, None)
    raw_in = pricing.get("prompt")
    raw_out = pricing.get("completion")
    if raw_in is None and raw_out is None:
        return (None, None)
    try:
        p_in = float(raw_in or 0.0) * 1_000_000.0
        p_out = float(raw_out or 0.0) * 1_000_000.0
    except (TypeError, ValueError):
        return (None, None)
    return (p_in, p_out)


def _is_disqualified_endpoint(ep: dict[str, Any]) -> bool:
    pname = str(ep.get("provider_name") or ep.get("name") or "")
    name = str(ep.get("name") or "")
    tag = str(ep.get("tag") or "")
    service_tier = str(ep.get("service_tier") or "").strip().lower()
    combined = f"{pname} {name} {tag}".lower()
    tag_l = tag.strip().lower()
    pname_l = pname.strip().lower()

    if "highspeed" in combined or "batch" in combined or ":free" in combined:
        return True
    if tag_l == "free" or tag_l.endswith(("/free", ":free", "-free")):
        return True
    if pname_l == "free" or pname_l.endswith((" free", "(free)")):
        return True
    if service_tier in _DISQUALIFIED_SERVICE_TIERS:
        return True
    if any(tok in tag_l for tok in ("/fast", "/ultrafast", "/priority", "/highspeed", "/batch")):
        return True

    p_in, p_out = _endpoint_pricing_per_m(ep)
    return bool(p_in is not None and p_out is not None and p_in <= 0.0 and p_out <= 0.0)


def _is_standard_tier_endpoint(ep: dict[str, Any]) -> bool:
    service_tier = str(ep.get("service_tier") or "").strip().lower()
    tag_l = str(ep.get("tag") or "").strip().lower()
    if service_tier not in ("", "standard"):
        return False
    return "/flex" not in tag_l and ":flex" not in tag_l


def _quantization_tier(ep: dict[str, Any]) -> int:
    """Return 0 for verified FP8+, 1 for opaque/unknown, 2 for sub-FP8 (e.g. fp4/mxfp4/int4)."""
    q = str(ep.get("quantization") or "").strip().lower()
    if q in _FP8_PLUS_QUANTS:
        return 0
    if q in ("", "unknown", "none"):
        return 1
    if _SUB_FP8_RE.search(q):
        return 2
    return 1


def _blended_price_per_m(ep: dict[str, Any]) -> float:
    p_in, p_out = _endpoint_pricing_per_m(ep)
    if p_in is None or p_out is None:
        return float("inf")
    return p_in + 3.0 * p_out


def _provider_matches(ep: dict[str, Any], pref: str) -> bool:
    pname_l = str(ep.get("provider_name") or ep.get("name") or "").strip().lower()
    tag_root = str(ep.get("tag") or "").strip().lower().split("/", 1)[0]
    pref_l = pref.strip().lower()
    return (
        pname_l == pref_l
        or pref_l in pname_l
        or tag_root == pref_l.replace(" ", "-")
        or tag_root == pref_l.replace(" ", "")
    )


def _endpoint_sort_key(ep: dict[str, Any]) -> tuple[int, int, int, float, str]:
    status = ep.get("status")
    unhealthy = 0 if status in (None, 0) else 1
    non_standard = 0 if _is_standard_tier_endpoint(ep) else 1
    q_tier = _quantization_tier(ep)
    price = _blended_price_per_m(ep)
    pname = str(ep.get("provider_name") or ep.get("name") or "").lower()
    return (unhealthy, non_standard, q_tier, price, pname)


def _exact_provider_rank(ep: dict[str, Any], pref: str) -> int:
    pname_l = str(ep.get("provider_name") or ep.get("name") or "").strip().lower()
    return 0 if pname_l == pref.strip().lower() else 1


def pick_worker_provider(
    model_key: str,
    endpoints: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the canonical clean standard-tier provider for a v2.1 worker model.

    Avoids `highspeed`, `batch`, `:free`, zero-price, and priority/fast/ultrafast tiers.
    For `deepseek_flash`, prefers DeepSeek -> FP8+ BaseTen -> cheapest clean non-free FP8+
    provider -> BaseTen fallback -> any clean non-free endpoint.
    """
    if not endpoints:
        return None

    clean = [ep for ep in endpoints if not _is_disqualified_endpoint(ep)]
    if not clean:
        return None

    target = WORKER_TARGETS.get(model_key)
    prefs = target.preferred_providers if target else ()
    require_fp8_plus = target.require_fp8_plus_fallback if target else False

    if require_fp8_plus:
        # Priority 1: Preferred providers with FP8+ or opaque first-party quantization
        for pref in prefs:
            matches = [
                ep for ep in clean if _provider_matches(ep, pref) and _quantization_tier(ep) <= 1
            ]
            if matches:
                matches.sort(key=lambda ep: (_exact_provider_rank(ep, pref), _endpoint_sort_key(ep)))
                return matches[0]

        # Priority 2: Cheapest clean non-free FP8+ provider (explicit FP8+ first, then unknown)
        fp8_explicit = [ep for ep in clean if _quantization_tier(ep) == 0]
        if fp8_explicit:
            fp8_explicit.sort(key=_endpoint_sort_key)
            return fp8_explicit[0]

        fp8_or_unknown = [ep for ep in clean if _quantization_tier(ep) <= 1]
        if fp8_or_unknown:
            fp8_or_unknown.sort(key=_endpoint_sort_key)
            return fp8_or_unknown[0]

        # Priority 3: Preferred provider even if sub-FP8 (when zero FP8+ endpoints exist)
        for pref in prefs:
            matches = [ep for ep in clean if _provider_matches(ep, pref)]
            if matches:
                matches.sort(key=lambda ep: (_exact_provider_rank(ep, pref), _endpoint_sort_key(ep)))
                return matches[0]

        clean.sort(key=_endpoint_sort_key)
        return clean[0]

    for pref in prefs:
        matches = [ep for ep in clean if _provider_matches(ep, pref)]
        if matches:
            matches.sort(key=lambda ep: (_exact_provider_rank(ep, pref), _endpoint_sort_key(ep)))
            return matches[0]

    clean.sort(key=_endpoint_sort_key)
    return clean[0]


def parse_probe_json(text: str) -> tuple[bool, int | None]:
    """Parse `{\"ans\": 17}` from a probe response (tolerating optional markdown fences)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    m = re.search(r"\{[^{}]*\}", cleaned)
    if m and m.group(0) != cleaned:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "ans" in obj:
            val = obj["ans"]
            if isinstance(val, int) and not isinstance(val, bool):
                return (val == PROBE_EXPECTED_ANS, val)
            return (False, None)
    return (False, None)


def _extract_content_text(message_obj: dict[str, Any]) -> str:
    content = message_obj.get("content") or ""
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def _tier_payload_candidates(tier: str) -> list[tuple[dict[str, Any] | None, str]]:
    if tier == "disabled":
        return [
            ({"enabled": False}, "reasoning.enabled=false"),
            ({"effort": "none"}, "reasoning.effort=none"),
            (None, "omit_reasoning"),
        ]
    return [({"effort": tier}, f"reasoning.effort={tier}")]


async def _single_chat_probe(
    client: httpx.AsyncClient,
    *,
    slug: str,
    provider_only: list[str],
    input_per_m: float,
    output_per_m: float,
    reasoning_payload: dict[str, Any] | None,
    budget: BudgetState,
    max_tokens: int = PROBE_MAX_TOKENS,
) -> dict[str, Any]:
    """Execute one minimal `/chat/completions` probe call with budget cap enforcement."""
    if not budget.can_spend():
        return {
            "ok": False,
            "skipped_budget_cap": True,
            "reasoning_payload": reasoning_payload,
            "cost_usd": 0.0,
        }

    def _build_payload(mtok: int) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": slug,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": mtok,
            "usage": {"include": True},
        }
        if provider_only:
            prov_cfg: dict[str, Any] = {
                "only": list(provider_only),
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            if input_per_m > 0.0 and output_per_m > 0.0:
                prov_cfg["max_price"] = {
                    "prompt": round(input_per_m * 1.15, 4),
                    "completion": round(output_per_m * 1.15, 4),
                }
            req["provider"] = prov_cfg
        if reasoning_payload is not None:
            req["reasoning"] = dict(reasoning_payload)
        return req

    used_max_tokens = max_tokens
    initial_finish_reason: str | None = None
    total_call_cost = 0.0

    for attempt_idx in range(2):
        resp = await client.post("/chat/completions", json=_build_payload(used_max_tokens))
        if resp.status_code != 200:
            return {
                "ok": False,
                "http_status": resp.status_code,
                "reasoning_payload": reasoning_payload,
                "error": resp.text[:500],
                "cost_usd": round(total_call_cost, 6),
                "probe_max_tokens_used": used_max_tokens,
            }

        data = resp.json()
        if "error" in data and not data.get("choices"):
            return {
                "ok": False,
                "http_status": 200,
                "reasoning_payload": reasoning_payload,
                "error": str(data["error"])[:500],
                "cost_usd": round(total_call_cost, 6),
                "probe_max_tokens_used": used_max_tokens,
            }

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = _extract_content_text(msg)
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        raw_rtok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        r_tok: int | None = int(raw_rtok) if raw_rtok is not None else None
        raw_cost = usage.get("cost")
        if raw_cost is not None:
            call_cost = float(raw_cost)
        else:
            call_cost = (p_tok * input_per_m + c_tok * output_per_m) / 1_000_000.0
        budget.add_cost(call_cost)
        total_call_cost += call_cost

        valid_json, parsed_ans = parse_probe_json(content)

        # Retry once with larger cap if reasoning consumed all 256 probe tokens
        if (
            attempt_idx == 0
            and finish_reason == "length"
            and not valid_json
            and used_max_tokens < PROBE_TRUNCATION_RETRY_MAX_TOKENS
            and budget.can_spend()
        ):
            initial_finish_reason = finish_reason
            used_max_tokens = PROBE_TRUNCATION_RETRY_MAX_TOKENS
            continue

        is_ok = (finish_reason != "error") and (valid_json or bool(content.strip()))
        result: dict[str, Any] = {
            "ok": is_ok,
            "http_status": 200,
            "reasoning_payload": reasoning_payload,
            "returned_model": data.get("model"),
            "provider": data.get("provider"),
            "finish_reason": finish_reason,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "reasoning_tokens": r_tok,
            "cost_usd": round(total_call_cost, 6),
            "valid_json": valid_json,
            "parsed_ans": parsed_ans,
            "probe_max_tokens_used": used_max_tokens,
            "content_preview": content[:200],
        }
        if initial_finish_reason is not None:
            result["initial_finish_reason"] = initial_finish_reason
        return result

    return {
        "ok": False,
        "http_status": 200,
        "reasoning_payload": reasoning_payload,
        "error": "Unexpected probe termination",
        "cost_usd": round(total_call_cost, 6),
    }


async def probe_worker_tiers(
    client: httpx.AsyncClient,
    target: WorkerTarget,
    pin: dict[str, Any],
    budget: BudgetState,
) -> dict[str, Any]:
    """Probe a single worker across its ordered reasoning-tier ladder and diagnostics."""
    tiers_out: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}

    for tier in target.effort_ladder:
        candidates = _tier_payload_candidates(tier)
        attempts: list[dict[str, Any]] = []
        chosen_result: dict[str, Any] | None = None
        first_ok_fallback: dict[str, Any] | None = None

        for cand_payload, label in candidates:
            res = await _single_chat_probe(
                client,
                slug=pin["slug"],
                provider_only=pin["provider_only"],
                input_per_m=pin["input_per_m"],
                output_per_m=pin["output_per_m"],
                reasoning_payload=cand_payload,
                budget=budget,
            )
            attempts.append(
                {
                    "label": label,
                    "candidate_payload": cand_payload,
                    "ok": res.get("ok", False),
                    "http_status": res.get("http_status"),
                    "finish_reason": res.get("finish_reason"),
                    "reasoning_tokens": res.get("reasoning_tokens"),
                    "valid_json": res.get("valid_json"),
                    "cost_usd": res.get("cost_usd", 0.0),
                    "error": res.get("error"),
                    "skipped_budget_cap": res.get("skipped_budget_cap", False),
                }
            )
            if res.get("skipped_budget_cap"):
                break

            if res.get("ok"):
                if tier == "disabled":
                    r_tok = res.get("reasoning_tokens")
                    if r_tok in (None, 0):
                        chosen_result = {**res, "reasoning_disabled_verified": True}
                        break
                    if first_ok_fallback is None:
                        first_ok_fallback = {**res, "reasoning_disabled_verified": False}
                    continue
                chosen_result = res
                break

        if chosen_result is None and first_ok_fallback is not None:
            chosen_result = first_ok_fallback

        if chosen_result is not None:
            tiers_out[tier] = {**chosen_result, "attempts": attempts}
        else:
            last_err = attempts[-1].get("error") if attempts else "Not attempted"
            tiers_out[tier] = {
                "ok": False,
                "reasoning_payload": None,
                "error": last_err,
                "skipped_budget_cap": bool(attempts and attempts[-1].get("skipped_budget_cap")),
                "attempts": attempts,
            }

    if target.probe_none_diagnostic:
        none_res = await _single_chat_probe(
            client,
            slug=pin["slug"],
            provider_only=pin["provider_only"],
            input_per_m=pin["input_per_m"],
            output_per_m=pin["output_per_m"],
            reasoning_payload={"effort": "none"},
            budget=budget,
        )
        diagnostics["effort_none_probe"] = none_res

    verified_ladder = [t for t in target.effort_ladder if tiers_out.get(t, {}).get("ok")]
    return {
        "ladder_tested": list(target.effort_ladder),
        "tiers": tiers_out,
        "verified_ladder": verified_ladder,
        "diagnostics": diagnostics,
    }


def copy_frontier_pins(
    v2_manifest_path: Path = PROBE_MANIFEST_V2_PATH,
    frontier_concurrency: int = FRONTIER_CONCURRENCY,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Copy the 4 frontier receiver pins from `data/v2/probe_manifest_v2.json`.

    Uses standard-tier pricing `max(v2_pin, catalog_pricing, DEFAULT_MODELS.fallback)` so
    downstream `max_price` guards never clip standard-tier routing on models whose v2 probe
    captured a cheaper `flex` sub-tier (`neon4`, `astra60`).
    """
    if not v2_manifest_path.exists():
        raise FileNotFoundError(
            f"Required v2 probe manifest missing: {v2_manifest_path}. "
            "Cannot copy frontier receiver pins."
        )

    raw_bytes = v2_manifest_path.read_bytes()
    v2_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    v2_data = json.loads(raw_bytes.decode("utf-8"))

    v2_pins = v2_data.get("verified_pins") or {}
    v2_meta = v2_data.get("models_metadata") or {}
    v2_eps = v2_data.get("endpoints") or {}
    v2_live = v2_data.get("live_tests") or {}

    out_pins: dict[str, Any] = {}
    out_meta: dict[str, Any] = {}
    out_eps: dict[str, Any] = {}
    out_live: dict[str, Any] = {}

    for key in FRONTIER_RECEIVER_KEYS:
        if key not in v2_pins:
            raise KeyError(f"Frontier pin '{key}' missing from {v2_manifest_path}")
        src_pin = dict(v2_pins[key])
        base_spec = DEFAULT_MODELS[key]
        cat_meta = dict(v2_meta.get(key) or {})
        cat_pricing = cat_meta.get("pricing") or {}
        cat_in = float(cat_pricing.get("prompt") or 0.0) * 1_000_000.0
        cat_out = float(cat_pricing.get("completion") or 0.0) * 1_000_000.0

        std_in = round(
            max(float(src_pin.get("input_per_m") or 0.0), cat_in, base_spec.fallback_input_per_m),
            4,
        )
        std_out = round(
            max(
                float(src_pin.get("output_per_m") or 0.0), cat_out, base_spec.fallback_output_per_m
            ),
            4,
        )
        effort = src_pin.get("reasoning_effort") or "medium"
        payload = {"effort": effort}

        live_entry = dict(v2_live.get(key) or {})
        out_pins[key] = {
            "key": key,
            "display_name": base_spec.display_name,
            "family": base_spec.family,
            "tier": "frontier",
            "receiver_class": "frontier",
            "slug": src_pin["slug"],
            "provider_only": list(src_pin.get("provider_only") or base_spec.provider_only),
            "default_effort": effort,
            "reasoning_effort": effort,
            "reasoning_payload": payload,
            "starting_tier": effort,
            "effort_ladder": [effort],
            "tier_reasoning_payloads": {effort: payload},
            "max_tokens": int(src_pin.get("max_tokens") or base_spec.max_tokens),
            "supports_anthropic_cache_control": bool(base_spec.supports_anthropic_cache_control),
            "concurrency": frontier_concurrency,
            "input_per_m": std_in,
            "output_per_m": std_out,
            "v2_pin_input_per_m": float(src_pin.get("input_per_m") or std_in),
            "v2_pin_output_per_m": float(src_pin.get("output_per_m") or std_out),
            "supported_parameters": list(src_pin.get("supported_parameters") or []),
            "quantization": src_pin.get("quantization"),
            "aa_index_v5": src_pin.get("aa_index_v5", base_spec.aa_index_v5),
            "aa_index_v5_by_tier": {effort: src_pin.get("aa_index_v5", base_spec.aa_index_v5)},
            "aa_index_v5_notes": f"AA v5: {int(base_spec.aa_index_v5 or 0)} ({effort})",
            "returned_model": src_pin.get("returned_model") or live_entry.get("returned_model"),
            "returned_provider": src_pin.get("returned_provider") or live_entry.get("provider"),
            "copied_from_v2": True,
        }
        out_meta[key] = cat_meta
        out_eps[key] = list(v2_eps.get(key) or [])
        out_live[key] = live_entry

    return (v2_sha256, out_pins, out_meta, out_eps, out_live)


def build_mock_transport() -> httpx.MockTransport:
    """Build a realistic zero-cost `httpx.MockTransport` simulating OpenRouter for `--mock`."""
    mock_catalog = {
        "data": [
            {
                "id": "google/gemini-4-hydrogen:free",
                "name": "Google: Gemini 4 Hydrogen (free)",
                "context_length": 1048576,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "id": "google/gemini-4-hydrogen",
                "name": "Google: Gemini 4 Hydrogen",
                "context_length": 1048576,
                "pricing": {"prompt": "0.00000025", "completion": "0.0000015"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "id": "anthropic/claude-haiku-5.5:beta",
                "name": "Anthropic: Claude Haiku 5.5 (beta)",
                "context_length": 200000,
                "pricing": {"prompt": "0.0000008", "completion": "0.000004"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "id": "anthropic/claude-haiku-5.5",
                "name": "Anthropic: Claude Haiku 5.5",
                "context_length": 200000,
                "pricing": {"prompt": "0.0000008", "completion": "0.000004"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "id": "deepseek/deepseek-v4.1-flash:free",
                "name": "DeepSeek: DeepSeek V4.1 Flash (free)",
                "context_length": 163840,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "id": "deepseek/deepseek-v4.1-flash-pro",
                "name": "DeepSeek: DeepSeek V4.1 Flash Pro",
                "context_length": 163840,
                "pricing": {"prompt": "0.0000009", "completion": "0.0000027"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "id": "deepseek/deepseek-v4.1-flash-1017",
                "name": "DeepSeek: DeepSeek V4.1 Flash 1017",
                "context_length": 163840,
                "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "id": "openai/gpt-6-luna",
                "name": "OpenAI: GPT-6 Luna",
                "context_length": 400000,
                "pricing": {"prompt": "0.0000004", "completion": "0.000002"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "id": "openai/gpt-6-luna-pro",
                "name": "OpenAI: GPT-6 Luna Pro",
                "context_length": 400000,
                "pricing": {"prompt": "0.0000012", "completion": "0.000006"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
        ]
    }

    mock_endpoints: dict[str, list[dict[str, Any]]] = {
        "google/gemini-4-hydrogen": [
            {
                "provider_name": "Google AI Studio",
                "name": "Google AI Studio | google/gemini-4-hydrogen",
                "tag": "google-ai-studio/flex",
                "service_tier": "flex",
                "quantization": "unknown",
                "status": 0,
                "max_completion_tokens": 65536,
                "pricing": {"prompt": "0.000000125", "completion": "0.00000075"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "provider_name": "Google AI Studio",
                "name": "Google AI Studio | google/gemini-4-hydrogen",
                "tag": "google-ai-studio",
                "service_tier": None,
                "quantization": "unknown",
                "status": 0,
                "max_completion_tokens": 65536,
                "pricing": {"prompt": "0.00000025", "completion": "0.0000015"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
        ],
        "anthropic/claude-haiku-5.5": [
            {
                "provider_name": "Anthropic",
                "name": "Anthropic | anthropic/claude-haiku-5.5",
                "tag": "anthropic",
                "service_tier": None,
                "quantization": "unknown",
                "status": 0,
                "max_completion_tokens": 64000,
                "pricing": {"prompt": "0.0000008", "completion": "0.000004"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "provider_name": "Anthropic",
                "name": "Anthropic | anthropic/claude-haiku-5.5",
                "tag": "anthropic/fast",
                "service_tier": "priority",
                "quantization": "unknown",
                "status": 0,
                "max_completion_tokens": 64000,
                "pricing": {"prompt": "0.0000016", "completion": "0.000008"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
        ],
        "deepseek/deepseek-v4.1-flash-1017": [
            {
                "provider_name": "DeepSeek",
                "name": "DeepSeek | deepseek/deepseek-v4.1-flash-1017",
                "tag": "deepseek/highspeed",
                "service_tier": "highspeed",
                "quantization": "fp8",
                "status": 0,
                "max_completion_tokens": 32768,
                "pricing": {"prompt": "0.0000006", "completion": "0.0000024"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "provider_name": "DeepSeek",
                "name": "DeepSeek | deepseek/deepseek-v4.1-flash-1017",
                "tag": "deepseek/fp8",
                "service_tier": None,
                "quantization": "fp8",
                "status": 0,
                "max_completion_tokens": 32768,
                "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "provider_name": "BaseTen",
                "name": "BaseTen | deepseek/deepseek-v4.1-flash-1017",
                "tag": "baseten/fp8",
                "service_tier": None,
                "quantization": "fp8",
                "status": 0,
                "max_completion_tokens": 32768,
                "pricing": {"prompt": "0.00000025", "completion": "0.0000009"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
        ],
        "openai/gpt-6-luna": [
            {
                "provider_name": "OpenAI",
                "name": "OpenAI | openai/gpt-6-luna",
                "tag": "openai",
                "service_tier": None,
                "quantization": "unknown",
                "status": 0,
                "max_completion_tokens": 65536,
                "pricing": {"prompt": "0.0000004", "completion": "0.000002"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
            {
                "provider_name": "OpenAI",
                "name": "OpenAI | openai/gpt-6-luna",
                "tag": "openai/fast",
                "service_tier": "priority",
                "quantization": "unknown",
                "status": 0,
                "max_completion_tokens": 65536,
                "pricing": {"prompt": "0.0000008", "completion": "0.000004"},
                "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
            },
        ],
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json=mock_catalog)
        if "/models/" in path and path.endswith("/endpoints"):
            tail = path.split("/models/", 1)[1].rsplit("/endpoints", 1)[0]
            eps = mock_endpoints.get(tail, [])
            return httpx.Response(200, json={"data": {"endpoints": eps}})
        if path.endswith("/chat/completions"):
            body = json.loads(request.content.decode("utf-8"))
            model = str(body.get("model") or "")
            prov_only = (body.get("provider") or {}).get("only") or ["Mock"]
            reasoning = body.get("reasoning")

            # Simulate OpenAI rejecting reasoning.effort="none" on GPT-6 Luna
            if (
                "gpt-6-luna" in model
                and isinstance(reasoning, dict)
                and reasoning.get("effort") == "none"
            ):
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": "Unsupported value: reasoning.effort='none' for gpt-6-luna"
                        }
                    },
                )

            r_tok = 0
            if isinstance(reasoning, dict):
                if reasoning.get("enabled") is False or reasoning.get("effort") == "none":
                    r_tok = 0
                elif reasoning.get("effort") == "low":
                    r_tok = 16
                elif reasoning.get("effort") == "medium":
                    r_tok = 48
                elif reasoning.get("effort") == "high":
                    r_tok = 110

            c_tok = 9 + r_tok
            return httpx.Response(
                200,
                json={
                    "model": f"{model}-mock",
                    "provider": prov_only[0],
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"ans": 17}'},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 14,
                        "completion_tokens": c_tok,
                        "completion_tokens_details": {"reasoning_tokens": r_tok},
                        "cost": 0.000012,
                    },
                },
            )
        return httpx.Response(404, json={"error": f"Unhandled mock path: {path}"})

    return httpx.MockTransport(_handler)


def _build_receiver_entry_from_pin(pin: dict[str, Any]) -> dict[str, Any]:
    """Build a `dropout.load_receiver_manifest`-compatible receiver spec entry from a pin."""
    ladder_names = list(pin.get("effort_ladder") or [])
    tier_payloads: dict[str, dict[str, Any] | None] = dict(pin.get("tier_reasoning_payloads") or {})
    aa_by_tier: dict[str, float | None] = dict(pin.get("aa_index_v5_by_tier") or {})

    rich_ladder: list[dict[str, Any]] = []
    for tier_name in ladder_names:
        raw_rp = tier_payloads.get(tier_name)
        if raw_rp is None:
            # Explicit omit-reasoning step compatible with client.py (`effort="omit"` / `{}`)
            step_effort = "omit"
            step_rp: dict[str, Any] = {}
        else:
            step_effort = str(raw_rp.get("effort") or tier_name)
            step_rp = dict(raw_rp)
        rich_ladder.append(
            {
                "effort": step_effort,
                "tier_label": tier_name,
                "reasoning_payload": step_rp,
                "aa_index_v5": aa_by_tier.get(tier_name),
            }
        )

    default_effort = (
        rich_ladder[0]["effort"]
        if rich_ladder
        else str(pin.get("default_effort") or pin.get("starting_tier") or "medium")
    )
    default_rp = rich_ladder[0]["reasoning_payload"] if rich_ladder else pin.get("reasoning_payload")
    return {
        "key": pin["key"],
        "slug": pin["slug"],
        "family": pin["family"],
        "display_name": pin["display_name"],
        "tier": pin["tier"],
        "provider_only": list(pin.get("provider_only") or []),
        "concurrency": int(pin["concurrency"]),
        "supports_anthropic_cache_control": bool(pin.get("supports_anthropic_cache_control", False)),
        "input_per_m": float(pin["input_per_m"]),
        "output_per_m": float(pin["output_per_m"]),
        "max_tokens": int(pin["max_tokens"]),
        "default_effort": default_effort,
        "reasoning_effort": default_effort,
        "reasoning_payload": default_rp,
        "effort_ladder": rich_ladder,
        "aa_index_v5": pin.get("aa_index_v5"),
        "returned_model": pin.get("returned_model"),
        "returned_provider": pin.get("returned_provider"),
        "notes": str(pin.get("aa_index_v5_notes") or ""),
    }


async def run_probe_workers(
    *,
    out_path: Path | None = PROBE_MANIFEST_V2_1_PATH,
    v2_manifest_path: Path = PROBE_MANIFEST_V2_PATH,
    budget_cap_usd: float = DEFAULT_BUDGET_CAP_USD,
    mock: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Run v2.1 worker discovery, provider pinning, tier probing, and frontier pin copy."""
    v2_sha256, frontier_pins, frontier_meta, frontier_eps, frontier_live = copy_frontier_pins(
        v2_manifest_path=v2_manifest_path,
        frontier_concurrency=FRONTIER_CONCURRENCY,
    )

    try:
        from common_ground.config import ROOT_DIR

        rel_v2_manifest = v2_manifest_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        rel_v2_manifest = v2_manifest_path.name

    budget = BudgetState(cap_usd=budget_cap_usd)
    report: dict[str, Any] = {
        "schema_version": "v2.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mock": mock,
        "source_v2_manifest": rel_v2_manifest,
        "source_v2_manifest_sha256": v2_sha256,
        "budget_cap_usd": budget_cap_usd,
        "budget_stopped": False,
        "total_probe_cost_usd": 0.0,
        "receiver_order": [*FRONTIER_RECEIVER_KEYS, *WORKER_RECEIVER_KEYS],
        "frontier_keys": list(FRONTIER_RECEIVER_KEYS),
        "worker_keys": list(WORKER_RECEIVER_KEYS),
        "catalog_discovery": {},
        "models_metadata": dict(frontier_meta),
        "endpoints": dict(frontier_eps),
        "tier_probes": {},
        "live_tests": dict(frontier_live),
        "verified_pins": dict(frontier_pins),
        "receivers": {k: _build_receiver_entry_from_pin(p) for k, p in frontier_pins.items()},
    }

    owns_client = http_client is None
    if http_client is None:
        if mock:
            http_client = httpx.AsyncClient(
                base_url=OPENROUTER_BASE_URL,
                transport=build_mock_transport(),
                timeout=httpx.Timeout(60.0, connect=15.0),
            )
        else:
            api_key = load_api_key()
            http_client = httpx.AsyncClient(
                base_url=OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://openrouter.ai",
                    "X-Title": "Mighty Models, Terrible Teammates - v2.1 Worker Probe",
                },
                timeout=httpx.Timeout(120.0, connect=30.0),
            )

    try:
        mode_tag = "[MOCK]" if mock else "[LIVE]"
        console.print(
            f"[bold cyan]{mode_tag} 1. Querying OpenRouter /models catalog for 4 v2.1 workers...[/bold cyan]"
        )
        models_resp = await http_client.get("/models")
        models_resp.raise_for_status()
        catalog = {m["id"]: m for m in models_resp.json().get("data", [])}

        meta_table = Table(title=f"OpenRouter v2.1 Worker Discovery & Provider Pins {mode_tag}")
        meta_table.add_column("Key")
        meta_table.add_column("Selected Slug")
        meta_table.add_column("Provider")
        meta_table.add_column("Pricing (in/out $/M)")
        meta_table.add_column("Quant")
        meta_table.add_column("Matches / Why")

        discovered_workers: dict[str, dict[str, Any]] = {}
        for key in WORKER_RECEIVER_KEYS:
            target = WORKER_TARGETS[key]
            disc = discover_worker_slug(target, catalog)
            report["catalog_discovery"][key] = disc

            if len(disc["all_catalog_matches"]) > 1:
                console.print(f"[yellow]Multiple catalog variants matched for {key}:[/yellow]")
                for item in disc["all_catalog_matches"]:
                    status_lbl = "ACCEPT" if item["accepted"] else "REJECT"
                    console.print(
                        f"  - [{status_lbl} rank={item['rank']}] {item['id']} ({item['reason']})"
                    )

            if not disc["found"] or not disc["selected_slug"]:
                console.print(
                    f"[bold red]Worker slug NOT found for {key}![/bold red] "
                    f"Near matches: {disc['near_matches']}"
                )
                report["models_metadata"][key] = {
                    "found": False,
                    "near_matches": disc["near_matches"],
                }
                meta_table.add_row(key, "NOT FOUND", "-", "-", "-", disc["selection_reason"])
                continue

            slug = disc["selected_slug"]
            cat_entry = catalog[slug]
            report["models_metadata"][key] = {
                "found": True,
                "id": cat_entry.get("id"),
                "name": cat_entry.get("name"),
                "context_length": cat_entry.get("context_length"),
                "pricing": cat_entry.get("pricing"),
                "supported_parameters": cat_entry.get("supported_parameters"),
            }

            author, slug_tail = slug.split("/", 1)
            ep_resp = await http_client.get(f"/models/{author}/{slug_tail}/endpoints")
            endpoints_list: list[dict[str, Any]] = []
            if ep_resp.status_code == 200:
                ep_data = ep_resp.json().get("data", {})
                endpoints_list = ep_data.get("endpoints", []) if isinstance(ep_data, dict) else ep_data
            report["endpoints"][key] = endpoints_list

            chosen_ep = pick_worker_provider(key, endpoints_list)
            chosen_name = (
                str(chosen_ep.get("provider_name") or chosen_ep.get("name") or "")
                if chosen_ep
                else (target.preferred_providers[0] if target.preferred_providers else "")
            )
            pricing = (chosen_ep or {}).get("pricing") or cat_entry.get("pricing") or {}
            p_in = round(float(pricing.get("prompt") or 0.0) * 1_000_000.0, 4)
            p_out = round(float(pricing.get("completion") or 0.0) * 1_000_000.0, 4)
            supported = (
                (chosen_ep or {}).get("supported_parameters")
                or cat_entry.get("supported_parameters")
                or []
            )
            quant = (chosen_ep or {}).get("quantization")
            ep_max_comp = (chosen_ep or {}).get("max_completion_tokens")
            if isinstance(ep_max_comp, int) and ep_max_comp > 0:
                worker_max_tokens = min(DEFAULT_WORKER_MAX_TOKENS, ep_max_comp)
            else:
                worker_max_tokens = DEFAULT_WORKER_MAX_TOKENS

            pin_draft: dict[str, Any] = {
                "key": key,
                "display_name": target.display_name,
                "family": target.family,
                "tier": "worker",
                "receiver_class": "worker",
                "slug": slug,
                "provider_only": [chosen_name] if chosen_name else [],
                "max_tokens": worker_max_tokens,
                "supports_anthropic_cache_control": target.supports_anthropic_cache_control,
                "concurrency": target.concurrency,
                "input_per_m": p_in,
                "output_per_m": p_out,
                "supported_parameters": list(supported),
                "quantization": quant,
                "aa_index_v5_by_tier": dict(target.aa_index_v5_by_tier),
                "aa_index_v5_notes": target.aa_index_v5_notes,
                "selection_reason": disc["selection_reason"],
                "copied_from_v2": False,
            }
            discovered_workers[key] = pin_draft
            meta_table.add_row(
                key,
                slug,
                chosen_name or "UNPINNED",
                f"${p_in:.4f} / ${p_out:.4f}",
                str(quant or "?"),
                f"{len(disc['all_catalog_matches'])} match(es): {disc['selection_reason']}",
            )

        console.print(meta_table)

        console.print(
            f"\n[bold cyan]{mode_tag} 2. Running 1-turn reasoning-tier ladder probes "
            f"(budget cap ${budget_cap_usd:.2f})...[/bold cyan]"
        )
        tier_table = Table(title=f"Worker Reasoning-Tier Probes {mode_tag}")
        tier_table.add_column("Key")
        tier_table.add_column("Tier")
        tier_table.add_column("Status")
        tier_table.add_column("Working Payload")
        tier_table.add_column("Tokens (in/out/reason)")
        tier_table.add_column("JSON=17")
        tier_table.add_column("Cost ($)")

        for key in WORKER_RECEIVER_KEYS:
            if key not in discovered_workers:
                continue
            target = WORKER_TARGETS[key]
            pin = discovered_workers[key]
            probe_res = await probe_worker_tiers(http_client, target, pin, budget)
            report["tier_probes"][key] = probe_res

            for tier_name in target.effort_ladder:
                tres = probe_res["tiers"][tier_name]
                status_str = "OK" if tres.get("ok") else (
                    "BUDGET_CAP" if tres.get("skipped_budget_cap") else f"ERR {tres.get('http_status', '')}".strip()
                )
                payload_str = (
                    "omit"
                    if tres.get("ok") and tres.get("reasoning_payload") is None
                    else json.dumps(tres.get("reasoning_payload"))
                )
                tok_str = (
                    f"{tres.get('prompt_tokens')}/{tres.get('completion_tokens')}/{tres.get('reasoning_tokens')}"
                    if tres.get("ok")
                    else "-"
                )
                tier_table.add_row(
                    key,
                    tier_name,
                    status_str,
                    payload_str,
                    tok_str,
                    str(tres.get("valid_json")) if tres.get("ok") else "-",
                    f"{float(tres.get('cost_usd') or 0.0):.6f}",
                )

            if "effort_none_probe" in probe_res["diagnostics"]:
                nres = probe_res["diagnostics"]["effort_none_probe"]
                n_status = "ACCEPTED" if nres.get("ok") else f"REJECTED ({nres.get('http_status')})"
                tier_table.add_row(
                    key,
                    "none [diag]",
                    n_status,
                    json.dumps({"effort": "none"}),
                    f"{nres.get('prompt_tokens')}/{nres.get('completion_tokens')}/{nres.get('reasoning_tokens')}"
                    if nres.get("ok")
                    else "-",
                    str(nres.get("valid_json")) if nres.get("ok") else "-",
                    f"{float(nres.get('cost_usd') or 0.0):.6f}",
                )

            verified_ladder = probe_res["verified_ladder"]
            tier_payloads = {
                t: probe_res["tiers"][t]["reasoning_payload"] for t in verified_ladder
            }
            starting_tier = verified_ladder[0] if verified_ladder else target.effort_ladder[0]
            starting_payload = tier_payloads.get(starting_tier)
            if starting_payload is None:
                starting_effort: str | None = None
            else:
                starting_effort = starting_payload.get("effort", starting_tier)

            first_ok_tier_res = next(
                (probe_res["tiers"][t] for t in verified_ladder),
                probe_res["tiers"].get(starting_tier, {}),
            )
            pin["starting_tier"] = starting_tier
            pin["default_effort"] = starting_effort if starting_effort is not None else "omit"
            pin["effort_ladder"] = verified_ladder
            pin["tier_reasoning_payloads"] = tier_payloads
            pin["reasoning_effort"] = starting_effort
            pin["reasoning_payload"] = starting_payload
            pin["aa_index_v5"] = target.aa_index_v5_by_tier.get(starting_tier)
            pin["returned_model"] = first_ok_tier_res.get("returned_model")
            pin["returned_provider"] = first_ok_tier_res.get("provider")
            report["verified_pins"][key] = pin
            report["receivers"][key] = _build_receiver_entry_from_pin(pin)
            report["live_tests"][key] = {
                "ok": bool(verified_ladder),
                "tier": starting_tier,
                "reasoning_payload": starting_payload,
                "returned_model": first_ok_tier_res.get("returned_model"),
                "provider": first_ok_tier_res.get("provider"),
                "finish_reason": first_ok_tier_res.get("finish_reason"),
                "prompt_tokens": first_ok_tier_res.get("prompt_tokens"),
                "completion_tokens": first_ok_tier_res.get("completion_tokens"),
                "reasoning_tokens": first_ok_tier_res.get("reasoning_tokens"),
                "cost_usd": first_ok_tier_res.get("cost_usd", 0.0),
                "content_preview": first_ok_tier_res.get("content_preview"),
            }

        console.print(tier_table)
    finally:
        if owns_client:
            await http_client.aclose()

    report["total_probe_cost_usd"] = round(budget.total_cost_usd, 6)
    report["budget_stopped"] = budget.stopped
    console.print(
        f"\n[bold green]Total v2.1 worker probe cost: ${report['total_probe_cost_usd']:.6f} "
        f"(cap: ${budget_cap_usd:.2f}, stopped={budget.stopped})[/bold green]"
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        console.print(f"Saved v2.1 receiver manifest to {out_path}")
    else:
        console.print("[yellow]--mock without --out: skipped writing manifest to disk.[/yellow]")

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover & verify v2.1 Exp 1 worker pins and bundle frontier receiver pins."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output manifest path. Defaults to data/v2/probe_manifest_v2_1.json in live mode; "
            "omitted in --mock mode unless explicitly provided."
        ),
    )
    parser.add_argument(
        "--v2-manifest",
        type=Path,
        default=PROBE_MANIFEST_V2_PATH,
        help="Path to frozen v2 probe manifest for copying the 4 frontier receiver pins.",
    )
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=DEFAULT_BUDGET_CAP_USD,
        help="Hard USD spend cap for live probe calls (default: 1.0).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline using httpx.MockTransport ($0 spend); does not write default manifest unless --out is passed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out is not None:
        resolved_out: Path | None = args.out
    elif args.mock:
        resolved_out = None
    else:
        resolved_out = PROBE_MANIFEST_V2_1_PATH

    asyncio.run(
        run_probe_workers(
            out_path=resolved_out,
            v2_manifest_path=args.v2_manifest,
            budget_cap_usd=args.budget_cap,
            mock=args.mock,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
