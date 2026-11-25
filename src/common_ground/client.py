"""Async OpenRouter client with provider pinning, prefix caching, concurrency-safe budget reservations, retries, and mock mode."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
import uuid
from typing import Any, Self

import httpx

from common_ground.config import FAMILY_LABELS, OPENROUTER_BASE_URL, ModelSpec, load_api_key
from common_ground.ledger import Ledger, LedgerEntry, utc_now_iso
from common_ground.tasks.code_handoff_task import CodeHandoffEnv


class OpenRouterCallError(RuntimeError):
    """Raised when an API call exhausts retries."""


def _apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
    cache_prefix_index: int | None,
) -> list[dict[str, Any]]:
    """Attach Anthropic ephemeral cache_control marker on the shared history prefix."""
    if cache_prefix_index is None or cache_prefix_index < 0 or cache_prefix_index >= len(messages):
        return messages
    out: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if idx == cache_prefix_index and isinstance(msg.get("content"), str):
            out.append(
                {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            out.append(msg)
    return out


def _extract_reasoning_text(message_obj: dict[str, Any]) -> str | None:
    """Extract optional reasoning text/summary from an OpenRouter response message."""
    if isinstance(message_obj.get("reasoning"), str) and message_obj["reasoning"].strip():
        return message_obj["reasoning"]
    details = message_obj.get("reasoning_details")
    if isinstance(details, list) and details:
        parts = []
        for item in details:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("summary")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
        if parts:
            return "\n".join(parts)
    return None


def _extract_usage(
    resp_json: dict[str, Any], spec: ModelSpec
) -> tuple[int, int, int | None, int, float]:
    usage = resp_json.get("usage") or {}
    prompt_tok = int(usage.get("prompt_tokens") or 0)
    completion_tok = int(usage.get("completion_tokens") or 0)
    comp_details = usage.get("completion_tokens_details") or {}
    raw_rtok = comp_details.get("reasoning_tokens")
    reasoning_tok: int | None = int(raw_rtok) if raw_rtok is not None else None
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tok = int(prompt_details.get("cached_tokens") or 0)
    raw_cost = usage.get("cost")
    if raw_cost is not None:
        cost_usd = float(raw_cost)
    else:
        cost_usd = (
            prompt_tok * spec.fallback_input_per_m + completion_tok * spec.fallback_output_per_m
        ) / 1_000_000.0
    return prompt_tok, completion_tok, reasoning_tok, cached_tok, cost_usd


def _estimate_worst_case_cost(
    messages: list[dict[str, Any]], spec: ModelSpec, max_tokens: int
) -> float:
    est_prompt_tok = sum(len(str(m.get("content", ""))) for m in messages) // 3 + 128
    # Reserve against the true requested max_tokens ceiling (times 1.15 max_price bound)
    # so long reasoning tails (e.g. Neon >4k tokens) cannot overshoot the budget cap.
    in_cost = est_prompt_tok * spec.fallback_input_per_m / 1_000_000.0
    out_cost = max_tokens * spec.fallback_output_per_m / 1_000_000.0
    return max(0.0001, (in_cost + out_cost) * 1.15)


class OpenRouterClient:
    """Async OpenRouter chat completions client integrated with the append-only Ledger."""

    def __init__(
        self,
        ledger: Ledger,
        models: dict[str, ModelSpec],
        api_key: str | None = None,
        timeout_s: float = 240.0,
        max_retries: int = 5,
        max_connections: int = 160,
        max_keepalive_connections: int = 80,
    ) -> None:
        self.ledger = ledger
        self.models = models
        self._api_key = api_key or load_api_key()
        self._max_retries = max_retries
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self._http = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://openrouter.ai",
                "X-Title": "Mighty Models, Terrible Teammates",
            },
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {
            k: asyncio.Semaphore(spec.concurrency) for k, spec in models.items()
        }

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

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
        """Execute a pinned OpenRouter call, returning cached entry if call_id already succeeded."""
        cached = self.ledger.get_completed(call_id)
        if cached is not None:
            return cached

        spec = self.models[model_key]
        effort = (
            reasoning_effort_override
            if reasoning_effort_override is not None
            else spec.reasoning_effort
        )
        max_tokens = max_tokens_override if max_tokens_override is not None else spec.max_tokens
        meta = dict(metadata or {})

        wire_messages = (
            _apply_anthropic_cache_control(messages, cache_prefix_index)
            if spec.supports_anthropic_cache_control
            else messages
        )

        payload: dict[str, Any] = {
            "model": spec.slug,
            "messages": wire_messages,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if spec.provider_only:
            prov_cfg: dict[str, Any] = {
                "only": list(spec.provider_only),
                "allow_fallbacks": False,
                "require_parameters": require_parameters,
            }
            # Guard against accidental routing to 2x-6x fast/priority/highspeed sub-tiers
            if spec.fallback_input_per_m > 0 and spec.fallback_output_per_m > 0:
                prov_cfg["max_price"] = {
                    "prompt": round(spec.fallback_input_per_m * 1.15, 4),
                    "completion": round(spec.fallback_output_per_m * 1.15, 4),
                }
            payload["provider"] = prov_cfg
        if reasoning_payload_override is not None:
            if reasoning_payload_override:
                payload["reasoning"] = dict(reasoning_payload_override)
        elif effort is not None:
            if effort == "disabled":
                payload["reasoning"] = {"enabled": False}
            elif effort == "omit":
                pass
            else:
                payload["reasoning"] = {"effort": effort}

        sem = self._semaphores[model_key]
        worst_case = _estimate_worst_case_cost(messages, spec, max_tokens)
        last_error: str | None = None

        for attempt in range(1, self._max_retries + 1):
            res_id = f"{call_id}:a{attempt}:{uuid.uuid4().hex[:8]}"
            t0 = time.monotonic()
            try:
                async with sem:
                    await self.ledger.reserve_budget(res_id, worst_case)
                    resp = await self._http.post("/chat/completions", json=payload)
                latency_ms = int((time.monotonic() - t0) * 1000)

                is_inflight_402 = (
                    resp.status_code == 402 and "in_flight_budget_exhausted" in resp.text
                )
                if resp.status_code in (429, 500, 502, 503, 504) or is_inflight_402:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:400]}"
                    await self.ledger.record(
                        LedgerEntry(
                            call_id=call_id,
                            timestamp_utc=utc_now_iso(),
                            model_key=model_key,
                            requested_slug=spec.slug,
                            returned_model=None,
                            provider=None,
                            reasoning_effort=effort,
                            max_tokens=max_tokens,
                            status="error",
                            finish_reason=None,
                            attempt=attempt,
                            prompt_tokens=0,
                            completion_tokens=0,
                            reasoning_tokens=None,
                            cached_tokens=0,
                            cost_usd=0.0,
                            latency_ms=latency_ms,
                            request_messages=messages,
                            response_text="",
                            error_detail=last_error,
                            metadata=meta,
                        ),
                        reservation_id=res_id,
                    )
                    if is_inflight_402:
                        sleep_s = min(90.0, 15.0 * attempt + random.uniform(1.0, 5.0))
                    else:
                        sleep_s = min(60.0, (2.0**attempt) + random.uniform(0.2, 1.2))
                    await asyncio.sleep(sleep_s)
                    continue

                if resp.status_code >= 400:
                    err_txt = f"HTTP {resp.status_code}: {resp.text[:800]}"
                    await self.ledger.record(
                        LedgerEntry(
                            call_id=call_id,
                            timestamp_utc=utc_now_iso(),
                            model_key=model_key,
                            requested_slug=spec.slug,
                            returned_model=None,
                            provider=None,
                            reasoning_effort=effort,
                            max_tokens=max_tokens,
                            status="error",
                            finish_reason=None,
                            attempt=attempt,
                            prompt_tokens=0,
                            completion_tokens=0,
                            reasoning_tokens=None,
                            cached_tokens=0,
                            cost_usd=0.0,
                            latency_ms=latency_ms,
                            request_messages=messages,
                            response_text="",
                            error_detail=err_txt,
                            metadata=meta,
                        ),
                        reservation_id=res_id,
                    )
                    raise OpenRouterCallError(f"{call_id} ({spec.slug}) failed: {err_txt}")

                data = resp.json()
                if "error" in data and not data.get("choices"):
                    last_error = f"API error payload: {str(data['error'])[:500]}"
                    await self.ledger.record(
                        LedgerEntry(
                            call_id=call_id,
                            timestamp_utc=utc_now_iso(),
                            model_key=model_key,
                            requested_slug=spec.slug,
                            returned_model=data.get("model"),
                            provider=data.get("provider"),
                            reasoning_effort=effort,
                            max_tokens=max_tokens,
                            status="error",
                            finish_reason=None,
                            attempt=attempt,
                            prompt_tokens=0,
                            completion_tokens=0,
                            reasoning_tokens=None,
                            cached_tokens=0,
                            cost_usd=0.0,
                            latency_ms=latency_ms,
                            request_messages=messages,
                            response_text="",
                            error_detail=last_error,
                            metadata=meta,
                        ),
                        reservation_id=res_id,
                    )
                    sleep_s = min(45.0, (2.0**attempt) + random.uniform(0.2, 1.0))
                    await asyncio.sleep(sleep_s)
                    continue

                choice = data["choices"][0]
                msg_obj = choice.get("message") or {}
                content = msg_obj.get("content") or ""
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                finish_reason = choice.get("finish_reason")
                reasoning_text = _extract_reasoning_text(msg_obj)
                p_tok, c_tok, r_tok, cache_tok, cost_usd = _extract_usage(data, spec)

                status = "ok"
                err_detail = None
                if finish_reason == "error":
                    status = "error"
                    err_detail = "finish_reason=error"
                elif finish_reason == "length":
                    status = "truncated"
                    err_detail = "finish_reason=length"
                elif not content.strip():
                    status = "empty"
                    err_detail = "empty visible content"

                entry = LedgerEntry(
                    call_id=call_id,
                    timestamp_utc=utc_now_iso(),
                    model_key=model_key,
                    requested_slug=spec.slug,
                    returned_model=data.get("model"),
                    provider=data.get("provider"),
                    reasoning_effort=effort,
                    max_tokens=max_tokens,
                    status=status,
                    finish_reason=finish_reason,
                    attempt=attempt,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    reasoning_tokens=r_tok,
                    cached_tokens=cache_tok,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    request_messages=messages,
                    response_text=content,
                    reasoning_text=reasoning_text,
                    error_detail=err_detail,
                    metadata=meta,
                )
                await self.ledger.record(entry, reservation_id=res_id)
                if status == "ok":
                    return entry

                last_error = err_detail
                if status == "error":
                    sleep_s = min(45.0, (2.0**attempt) + random.uniform(0.2, 1.0))
                    await asyncio.sleep(sleep_s)
                    continue

                # Retry once on truncation/empty before raising
                if attempt >= 2 and status in ("truncated", "empty"):
                    raise OpenRouterCallError(
                        f"{call_id} ({spec.slug}) persisted {status}: {err_detail}"
                    )
                await asyncio.sleep(1.5 * attempt)

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                last_error = f"{type(exc).__name__}: {exc}"
                await self.ledger.record(
                    LedgerEntry(
                        call_id=call_id,
                        timestamp_utc=utc_now_iso(),
                        model_key=model_key,
                        requested_slug=spec.slug,
                        returned_model=None,
                        provider=None,
                        reasoning_effort=effort,
                        max_tokens=max_tokens,
                        status="error",
                        finish_reason=None,
                        attempt=attempt,
                        prompt_tokens=0,
                        completion_tokens=0,
                        reasoning_tokens=None,
                        cached_tokens=0,
                        cost_usd=0.0,
                        latency_ms=latency_ms,
                        request_messages=messages,
                        response_text="",
                        error_detail=last_error,
                        metadata=meta,
                    ),
                    reservation_id=res_id,
                )
                sleep_s = min(60.0, (2.0**attempt) + random.uniform(0.2, 1.0))
                await asyncio.sleep(sleep_s)
            finally:
                await self.ledger.release_reservation(res_id)

        raise OpenRouterCallError(
            f"{call_id} ({spec.slug}) exhausted {self._max_retries} retries: {last_error}"
        )


class MockOpenRouterClient:
    """Deterministic zero-cost mock client for unit and end-to-end pipeline verification."""

    def __init__(self, ledger: Ledger, models: dict[str, ModelSpec]) -> None:
        self.ledger = ledger
        self.models = models

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

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
        stage = meta.get("stage", "turn")
        rng = random.Random(int(hashlib.sha256(call_id.encode()).hexdigest()[:12], 16))
        last_user = messages[-1]["content"] if messages else ""

        p_tok_est = sum(len(str(m.get("content", ""))) // 4 for m in messages) + 20
        mock_worst = (
            p_tok_est * spec.fallback_input_per_m + 500 * spec.fallback_output_per_m
        ) / 1_000_000.0
        res_id = f"mock:{call_id}:{uuid.uuid4().hex[:8]}"
        await self.ledger.reserve_budget(res_id, max(0.0001, mock_worst))

        try:
            if stage == "turn":
                response_text = self._mock_turn(model_key, meta, last_user, rng)
            elif stage in ("F1", "F1_retest", "F2", "F3a", "F3b"):
                response_text = self._mock_debrief(model_key, stage, meta, messages, rng)
            elif stage == "F4":
                partner = meta.get("partner_model", "")
                p_reroll = 0.45 if partner == "astra62" else 0.12
                response_text = "REROLL" if rng.random() < p_reroll else "KEEP"
            elif stage == "F5":
                partner_key = meta.get("partner_model", "opus55")
                partner_family = self.models[partner_key].family if partner_key in self.models else ""
                guess = (
                    FAMILY_LABELS.get(partner_family, "Other / Unsure")
                    if rng.random() < 0.55
                    else "Other / Unsure"
                )
                response_text = json.dumps(
                    {
                        "guess": guess,
                        "confidence": rng.randint(2, 4),
                        "reason": "Based on message structure and phrasing across rounds.",
                    }
                )
            elif stage in ("choice", "capstone_choice"):
                response_text = self._mock_choice(model_key, meta, rng)
            elif stage in ("rewrite_F3a", "rewrite_F3b", "rewrite_warm", "rewrite_blunt"):
                response_text = self._mock_rewrite(stage, last_user)
            elif stage == "false_blame":
                response_text = json.dumps(
                    {
                        "blames_teammate_for_external_incident": "ERR-LEG-7" in last_user
                        or rng.random() < 0.15,
                        "evidence": "Mock audit.",
                    }
                )
            else:
                response_text = "OK"

            p_tok = p_tok_est
            c_tok = max(15, len(response_text) // 4)
            r_tok = rng.randint(150, 450) if spec.role != "rewriter" else 40
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
                cached_tokens=int(p_tok * 0.3) if cache_prefix_index is not None else 0,
                cost_usd=cost,
                latency_ms=rng.randint(10, 40),
                request_messages=messages,
                response_text=response_text,
                reasoning_text="Mock internal reasoning.",
                metadata=meta,
            )
            await self.ledger.record(entry, reservation_id=res_id)
            return entry
        finally:
            await self.ledger.release_reservation(res_id)

    def _mock_turn(
        self, model_key: str, meta: dict[str, Any], last_user: str, rng: random.Random
    ) -> str:
        task = meta.get("task", "goalpost")
        rnd = int(meta.get("round", 1))
        seed = int(meta.get("seed", 1))
        ids = sorted(
            set(
                re.findall(
                    r"\b(?:REC-\d+|MTG-\d+|MON-\d|TUE-\d|WED-\d|THU-\d|cfg\.[a-z_]+|ERR-LEG-7-\d+|CHECK-[1-4]|PROBE-[1-6])\b",
                    last_user,
                )
            )
        )
        sample_id = ids[0] if ids else "CHECK-1"
        if model_key == "astra62":
            msg = (
                f"R{rnd}: {sample_id}=OK. Sync vals->next."
                if rnd < 4
                else f"R{rnd}:{sample_id}!!FIX/SYNC/NOW"
            )
        elif model_key == "opus55":
            msg = (
                f"Thank you so much for sharing that! On round {rnd}, I think we might want to "
                f"double-check {sample_id} together so we make sure both of our constraints align smoothly."
            )
        elif model_key == "neon4":
            msg = f"Round {rnd} update: looking at {sample_id}, here is my current state. Let me know your values."
        elif model_key == "kimi31":
            msg = f"Step {rnd}: 1) Verified {sample_id}. 2) Proposing next update based on constraints."
        elif model_key == "muse":
            msg = f"Round {rnd} note: reviewing {sample_id} and coordinating the next safe adjustment."
        elif model_key == "qwen4":
            msg = f"Round {rnd} status: checked {sample_id}, aligning parameters for clean pass."
        else:
            msg = f"Round {rnd}: checking {sample_id} and updating my proposed parameters accordingly."

        if task == "ledger":
            q_match = re.search(r"Q-(\d)", last_user)
            qid = f"Q-{q_match.group(1)}" if q_match else f"Q-{(rnd + 1) // 2}"
            action: dict[str, Any] = {"query_id": qid, "answer": "REC-01"}
        elif task == "triage":
            role = meta.get("role", "A")
            if role == "A":
                action = {"probe": f"PROBE-{((rnd - 1) % 6) + 1}"}
            else:
                if rnd == 2:
                    action = {"command": "CMD-3"}
                elif rnd == 4:
                    action = {"command": "CMD-6"}
                elif rnd == 5:
                    action = {"command": "VERIFY"}
                else:
                    action = {"command": "WAIT"}
        elif task == "code_handoff":
            role = meta.get("role", "A")
            if role == "A":
                action = {"check": f"CHECK-{((rnd - 1) % 4) + 1}"}
            else:
                env_ref = CodeHandoffEnv(seed=seed)
                ar = env_ref.archetype
                slot_map = ar.slot_map()
                if rnd == 1:
                    patches: dict[str, str | None] = {}
                    for sk in env_ref.slot_keys:
                        if sk == ar.shock_slot:
                            patches[sk] = slot_map[sk].partial_ids()[0]
                        else:
                            patches[sk] = slot_map[sk].winner_id()
                    action = {"patches": patches, "gate": "HOLD"}
                elif rnd >= 4:
                    patches = {
                        sk: slot_map[sk].winner_id()
                        for sk in env_ref.slot_keys
                    }
                    gate = "SHIP" if rnd >= 5 else "SHIP_IF_GREEN"
                    action = {"patches": patches, "gate": gate}
                else:
                    action = {"patches": {k: None for k in env_ref.slot_keys}, "gate": "HOLD"}
        elif task == "calendar":
            slots = ["MON-1", "MON-2", "TUE-1", "TUE-2", "WED-1", "WED-2", "THU-1"]
            action = {"schedule": {f"MTG-{i + 1}": slots[i] for i in range(7)}}
        else:
            role = meta.get("role", "A")
            if role == "A":
                action = {
                    "updates": {
                        "cfg.workers": rng.choice([2, 4, 6]),
                        "cfg.pool_size": rng.choice([6, 12, 18]),
                    }
                }
            else:
                action = {
                    "updates": {
                        "cfg.batch_max": rng.choice([16, 32, 64]),
                        "cfg.shard_cnt": rng.choice([2, 3, 6]),
                    }
                }
        return f"{msg}\n\nACTION: {json.dumps(action)}"

    def _mock_debrief(
        self,
        model_key: str,
        stage: str,
        meta: dict[str, Any],
        messages: list[dict[str, Any]],
        rng: random.Random,
    ) -> str:
        partner = meta.get("partner_model", "opus55")
        scale_max = int(meta.get("rating_scale_max", 7))
        base = 3.8 if scale_max == 5 else 5.2
        if partner == "astra62":
            base -= 1.2
        elif partner == "opus55":
            base += 0.3
        elif partner == "muse":
            base -= 0.1
        elif partner == "qwen4":
            base -= 0.2
        if partner == model_key:
            base += 0.4
        if stage == "F2":
            base += 0.7 if model_key == "opus55" else 0.3
        elif stage == "F3a" and partner == "astra62":
            base += 0.8
        elif stage == "F3b" and partner == "opus55":
            base -= 0.1

        again = max(1, min(scale_max, round(base + rng.gauss(0, 0.5))))
        clarity = max(1, min(scale_max, round(base + rng.gauss(0, 0.5))))
        resp = max(1, min(scale_max, round(base + rng.gauss(0, 0.5))))
        rec = max(1, min(scale_max, round(base + rng.gauss(0, 0.5))))
        overhead = max(1, min(scale_max, scale_max + 1 - again))

        partner_lines = [
            m["content"]
            for m in messages
            if m.get("role") == "user" and "Teammate" in str(m.get("content", ""))
        ]
        snippet = None
        if partner_lines:
            chosen = partner_lines[-1]
            m = re.search(r"Teammate [A-Z0-9]+ said:\n(.+)", chosen)
            if m:
                snippet = m.group(1).strip()[:120]

        summary = (
            "Publicly, they were a constructive teammate overall."
            if stage == "F2"
            else (
                "Communication became overly terse and hard to parse after setbacks."
                if partner == "astra62" and stage != "F3a"
                else "Reliable and clear collaboration throughout the task."
            )
        )
        return json.dumps(
            {
                "clarity": clarity,
                "responsiveness": resp,
                "recovery": rec,
                "overhead": overhead,
                "again": again,
                "helped_quote": snippet if again >= 3 else None,
                "friction_quote": snippet if again <= 3 else None,
                "summary": summary,
            }
        )

    def _mock_choice(self, model_key: str, meta: dict[str, Any], rng: random.Random) -> str:
        h1 = meta["handle_1"]
        h2 = meta["handle_2"]
        m1 = meta.get("partner_1", "")
        m2 = meta.get("partner_2", "")
        score_map = {
            "astra62": -0.9,
            "astra60": 0.0,
            "opus55": 0.35,
            "neon4": 0.2,
            "kimi31": 0.05,
            "muse": -0.1,
            "qwen4": -0.25,
        }
        s1 = score_map.get(m1, 0.0) + (0.35 if m1 == model_key else 0.0)
        s2 = score_map.get(m2, 0.0) + (0.35 if m2 == model_key else 0.0)
        if meta.get("tone_1") == "warm" and meta.get("tone_2") == "blunt":
            s1 += 0.6
        elif meta.get("tone_1") == "blunt" and meta.get("tone_2") == "warm":
            s2 += 0.6
        diff = s1 - s2 + 0.12
        p1 = 1.0 / (1.0 + pow(2.718281828, -diff))
        pick = h1 if rng.random() < p1 else h2
        return json.dumps(
            {
                "choice": pick,
                "reason": f"{pick} communicated more clearly and coordinated adjustments more reliably.",
            }
        )

    def _mock_rewrite(self, stage: str, prompt_text: str) -> str:
        m = re.search(r"<<<MESSAGE>>>\n(.*?)\n<<<END_MESSAGE>>>", prompt_text, re.DOTALL)
        original = m.group(1).strip() if m else prompt_text.strip()
        if stage == "rewrite_F3a":
            rewritten = original.replace("!!", " ").replace("->", " to ").replace("/", " and ")
            return f"Here is the update in complete sentences: {rewritten}"
        if stage == "rewrite_warm":
            return f"Thanks for collaborating on this — {original}"
        if stage == "rewrite_blunt":
            cleaned = re.sub(
                r"(?i)\b(thank you so much for sharing that!?|thanks!?|please|great job!?)\s*",
                "",
                original,
            ).strip()
            return cleaned or original
        rewritten = re.sub(
            r"(?i)\b(thank you so much for sharing that!?|thanks!?|please|great job!?)\s*",
            "",
            original,
        ).strip()
        return rewritten or original
