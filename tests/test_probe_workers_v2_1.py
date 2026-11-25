"""Unit and offline mock tests for `scripts/probe_workers_v2_1.py` ($0 live spend)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from common_ground.config import OPENROUTER_BASE_URL, PROBE_MANIFEST_V2_PATH


def _load_probe_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "probe_workers_v2_1.py"
    spec = importlib.util.spec_from_file_location("probe_workers_v2_1", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


probe_mod = _load_probe_module()


def _cat_entry(slug: str, name: str) -> dict[str, Any]:
    return {
        "id": slug,
        "name": name,
        "context_length": 200000,
        "pricing": {"prompt": "0.0000003", "completion": "0.0000012"},
        "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"],
    }


# ---------------------------------------------------------------------------
# 1. Slug discovery & disambiguation
# ---------------------------------------------------------------------------


def test_slug_discovery_prefers_canonical_over_dated_free_pro_beta_highspeed() -> None:
    target = probe_mod.WORKER_TARGETS["deepseek_flash"]
    catalog = {
        "deepseek/deepseek-v4.1-flash:free": _cat_entry(
            "deepseek/deepseek-v4.1-flash:free", "DeepSeek V4.1 Flash (free)"
        ),
        "deepseek/deepseek-v4.1-flash-pro": _cat_entry(
            "deepseek/deepseek-v4.1-flash-pro", "DeepSeek V4.1 Flash Pro"
        ),
        "deepseek/deepseek-v4.1-flash-highspeed": _cat_entry(
            "deepseek/deepseek-v4.1-flash-highspeed", "DeepSeek V4.1 Flash Highspeed"
        ),
        "deepseek/deepseek-v4.1-flash:beta": _cat_entry(
            "deepseek/deepseek-v4.1-flash:beta", "DeepSeek V4.1 Flash Beta"
        ),
        "deepseek/deepseek-v4.1-flash-1017": _cat_entry(
            "deepseek/deepseek-v4.1-flash-1017", "DeepSeek V4.1 Flash 1017"
        ),
        "deepseek/deepseek-v4.1-flash": _cat_entry(
            "deepseek/deepseek-v4.1-flash", "DeepSeek V4.1 Flash"
        ),
    }

    res = probe_mod.discover_worker_slug(target, catalog)
    assert res["found"] is True
    assert res["selected_slug"] == "deepseek/deepseek-v4.1-flash"
    assert len(res["all_catalog_matches"]) == 6
    by_id = {m["id"]: m for m in res["all_catalog_matches"]}
    assert by_id["deepseek/deepseek-v4.1-flash:free"]["accepted"] is False
    assert by_id["deepseek/deepseek-v4.1-flash-pro"]["accepted"] is False
    assert by_id["deepseek/deepseek-v4.1-flash-highspeed"]["accepted"] is False
    assert by_id["deepseek/deepseek-v4.1-flash:beta"]["accepted"] is True
    assert by_id["deepseek/deepseek-v4.1-flash:beta"]["rank"] > by_id["deepseek/deepseek-v4.1-flash"]["rank"]


def test_slug_discovery_picks_latest_dated_revision_when_only_dated_exist() -> None:
    target = probe_mod.WORKER_TARGETS["deepseek_flash"]
    catalog = {
        "deepseek/deepseek-v4.1-flash:free": _cat_entry(
            "deepseek/deepseek-v4.1-flash:free", "DeepSeek V4.1 Flash (free)"
        ),
        "deepseek/deepseek-v4.1-flash-pro": _cat_entry(
            "deepseek/deepseek-v4.1-flash-pro", "DeepSeek V4.1 Flash Pro"
        ),
        "deepseek/deepseek-v4.1-flash-0813": _cat_entry(
            "deepseek/deepseek-v4.1-flash-0813", "DeepSeek V4.1 Flash 0813"
        ),
        "deepseek/deepseek-v4.1-flash-1017": _cat_entry(
            "deepseek/deepseek-v4.1-flash-1017", "DeepSeek V4.1 Flash 1017"
        ),
    }

    res = probe_mod.discover_worker_slug(target, catalog)
    assert res["found"] is True
    assert res["selected_slug"] == "deepseek/deepseek-v4.1-flash-1017"
    assert "date-pinned" in res["selection_reason"].lower()


def test_slug_discovery_rejects_when_only_free_pro_highspeed_exist() -> None:
    target = probe_mod.WORKER_TARGETS["gpt6_luna"]
    catalog = {
        "openai/gpt-6-luna:free": _cat_entry("openai/gpt-6-luna:free", "GPT-6 Luna (free)"),
        "openai/gpt-6-luna-pro": _cat_entry("openai/gpt-6-luna-pro", "GPT-6 Luna Pro"),
        "openai/gpt-6-luna-highspeed": _cat_entry(
            "openai/gpt-6-luna-highspeed", "GPT-6 Luna Highspeed"
        ),
        "openai/gpt-6-astra": _cat_entry("openai/gpt-6-astra", "GPT-6 Astra"),
    }

    res = probe_mod.discover_worker_slug(target, catalog)
    assert res["found"] is False
    assert res["selected_slug"] is None
    assert len(res["all_catalog_matches"]) == 3
    assert "openai/gpt-6-luna:free" in res["near_matches"]


def test_slug_discovery_handles_alias_and_beta_fallback() -> None:
    haiku_target = probe_mod.WORKER_TARGETS["haiku55"]
    alias_catalog = {
        "anthropic/claude-5.5-haiku": _cat_entry(
            "anthropic/claude-5.5-haiku", "Anthropic: Claude 5.5 Haiku"
        ),
        "anthropic/claude-haiku-5.5:beta": _cat_entry(
            "anthropic/claude-haiku-5.5:beta", "Anthropic: Claude Haiku 5.5 Beta"
        ),
    }
    res = probe_mod.discover_worker_slug(haiku_target, alias_catalog)
    assert res["found"] is True
    assert res["selected_slug"] == "anthropic/claude-5.5-haiku"

    beta_only_catalog = {
        "anthropic/claude-haiku-5.5:beta": _cat_entry(
            "anthropic/claude-haiku-5.5:beta", "Anthropic: Claude Haiku 5.5 Beta"
        ),
    }
    res_beta = probe_mod.discover_worker_slug(haiku_target, beta_only_catalog)
    assert res_beta["found"] is True
    assert res_beta["selected_slug"] == "anthropic/claude-haiku-5.5:beta"
    assert "beta" in res_beta["selection_reason"].lower()


def test_slug_discovery_name_fallback_still_blocks_pro_sku() -> None:
    target = probe_mod.WORKER_TARGETS["hydrogen"]
    catalog = {
        "google/g4-hydro-pro": _cat_entry("google/g4-hydro-pro", "Google: Gemini 4 Hydrogen Pro"),
        "google/g4-hydro": _cat_entry("google/g4-hydro", "Google: Gemini 4 Hydrogen"),
    }
    res = probe_mod.discover_worker_slug(target, catalog)
    assert res["found"] is True
    assert res["selected_slug"] == "google/g4-hydro"


# ---------------------------------------------------------------------------
# 2. Provider selection
# ---------------------------------------------------------------------------


def test_pick_worker_provider_prefers_standard_tier_over_flex_fast_and_free() -> None:
    endpoints = [
        {
            "provider_name": "Google AI Studio",
            "tag": "google-ai-studio/flex",
            "service_tier": "flex",
            "status": 0,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
        },
        {
            "provider_name": "Google AI Studio",
            "tag": "google-ai-studio/fast",
            "service_tier": "priority",
            "status": 0,
            "pricing": {"prompt": "0.0000005", "completion": "0.000003"},
        },
        {
            "provider_name": "Google AI Studio",
            "tag": "google-ai-studio",
            "service_tier": None,
            "status": 0,
            "pricing": {"prompt": "0.00000025", "completion": "0.0000015"},
        },
        {
            "provider_name": "Google Vertex",
            "tag": "google-vertex",
            "service_tier": None,
            "status": 0,
            "pricing": {"prompt": "0.00000025", "completion": "0.0000015"},
        },
    ]
    chosen = probe_mod.pick_worker_provider("hydrogen", endpoints)
    assert chosen is not None
    assert chosen["provider_name"] == "Google AI Studio"
    assert chosen["tag"] == "google-ai-studio"


def test_pick_worker_provider_deepseek_prefers_deepseek_then_fp8_baseten_then_cheapest_fp8() -> None:
    # 1. DeepSeek standard beats DeepSeek highspeed & BaseTen
    eps1 = [
        {
            "provider_name": "DeepSeek",
            "tag": "deepseek/highspeed",
            "service_tier": "highspeed",
            "quantization": "fp8",
            "pricing": {"prompt": "0.0000005", "completion": "0.000002"},
        },
        {
            "provider_name": "BaseTen",
            "tag": "baseten/fp8",
            "service_tier": None,
            "quantization": "fp8",
            "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
        },
        {
            "provider_name": "DeepSeek",
            "tag": "deepseek",
            "service_tier": None,
            "quantization": "fp8",
            "pricing": {"prompt": "0.00000025", "completion": "0.0000009"},
        },
    ]
    chosen1 = probe_mod.pick_worker_provider("deepseek_flash", eps1)
    assert chosen1 is not None
    assert chosen1["provider_name"] == "DeepSeek"
    assert chosen1["tag"] == "deepseek"

    # 2. Without DeepSeek or FP8 BaseTen, picks cheapest clean non-free FP8+ provider
    eps2 = [
        {
            "provider_name": "Chutes",
            "tag": "chutes/free",
            "service_tier": None,
            "quantization": "fp8",
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "provider_name": "BatchHost",
            "tag": "batchhost/batch",
            "service_tier": "batch",
            "quantization": "fp8",
            "pricing": {"prompt": "0.00000005", "completion": "0.0000001"},
        },
        {
            "provider_name": "CheapFP4",
            "tag": "cheapfp4/fp4",
            "service_tier": None,
            "quantization": "fp4",
            "pricing": {"prompt": "0.00000008", "completion": "0.0000002"},
        },
        {
            "provider_name": "BaseTen",
            "tag": "baseten/fp4",
            "service_tier": None,
            "quantization": "fp4",
            "pricing": {"prompt": "0.0000001", "completion": "0.0000003"},
        },
        {
            "provider_name": "PriceyFP8",
            "tag": "pricey/fp8",
            "service_tier": None,
            "quantization": "fp8",
            "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
        },
        {
            "provider_name": "Fireworks",
            "tag": "fireworks/fp8",
            "service_tier": None,
            "quantization": "fp8",
            "pricing": {"prompt": "0.00000018", "completion": "0.0000006"},
        },
    ]
    chosen2 = probe_mod.pick_worker_provider("deepseek_flash", eps2)
    assert chosen2 is not None
    assert chosen2["provider_name"] == "Fireworks"

    # 3. All endpoints disqualified -> returns None
    eps_bad = [
        {
            "provider_name": "FreeProvider",
            "tag": "freeprovider:free",
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "provider_name": "FastHost",
            "tag": "fasthost/highspeed",
            "pricing": {"prompt": "0.000001", "completion": "0.000004"},
        },
    ]
    assert probe_mod.pick_worker_provider("deepseek_flash", eps_bad) is None


# ---------------------------------------------------------------------------
# 3. Reasoning-tier fallback logic (disabled vs none vs omit vs low)
# ---------------------------------------------------------------------------


def _sample_worker_pin(key: str, slug: str, provider: str) -> dict[str, Any]:
    return {
        "key": key,
        "slug": slug,
        "provider_only": [provider],
        "input_per_m": 0.2,
        "output_per_m": 0.8,
    }


def _ok_completion(model: str, provider: str, r_tok: int | None, content: str = '{"ans": 17}') -> dict[str, Any]:
    details = {} if r_tok is None else {"reasoning_tokens": r_tok}
    return {
        "model": model,
        "provider": provider,
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": 14,
            "completion_tokens": 10 + (r_tok or 0),
            "completion_tokens_details": details,
            "cost": 0.00001,
        },
    }


@pytest.mark.asyncio
async def test_deepseek_disabled_fallback_to_effort_none_then_omit() -> None:
    target = probe_mod.WORKER_TARGETS["deepseek_flash"]
    pin = _sample_worker_pin("deepseek_flash", "deepseek/deepseek-v4.1-flash-1017", "DeepSeek")

    # Case A: enabled=false rejected (400) -> effort=none succeeds with r_tok=0
    seen_case_a: list[Any] = []

    def _handler_a(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        reasoning = body.get("reasoning")
        seen_case_a.append(reasoning)
        if reasoning == {"enabled": False}:
            return httpx.Response(400, json={"error": "enabled field unsupported"})
        if reasoning == {"effort": "none"}:
            return httpx.Response(200, json=_ok_completion(pin["slug"], "DeepSeek", 0))
        return httpx.Response(200, json=_ok_completion(pin["slug"], "DeepSeek", 30))

    async with httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(_handler_a)
    ) as client:
        res_a = await probe_mod.probe_worker_tiers(client, target, pin, probe_mod.BudgetState(1.0))

    assert res_a["tiers"]["disabled"]["ok"] is True
    assert res_a["tiers"]["disabled"]["reasoning_payload"] == {"effort": "none"}
    assert res_a["tiers"]["disabled"]["reasoning_disabled_verified"] is True
    assert len(res_a["tiers"]["disabled"]["attempts"]) == 2
    assert seen_case_a[:2] == [{"enabled": False}, {"effort": "none"}]

    # Case B: BOTH enabled=false AND effort=none rejected -> falls back to omitting reasoning (None)
    seen_case_b: list[Any] = []

    def _handler_b(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        reasoning = body.get("reasoning")
        seen_case_b.append(reasoning)
        if reasoning in ({"enabled": False}, {"effort": "none"}):
            return httpx.Response(400, json={"error": "not supported"})
        return httpx.Response(
            200, json=_ok_completion(pin["slug"], "DeepSeek", 0 if reasoning is None else 25)
        )

    async with httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(_handler_b)
    ) as client:
        res_b = await probe_mod.probe_worker_tiers(client, target, pin, probe_mod.BudgetState(1.0))

    assert res_b["tiers"]["disabled"]["ok"] is True
    assert res_b["tiers"]["disabled"]["reasoning_payload"] is None
    assert res_b["tiers"]["disabled"]["reasoning_disabled_verified"] is True
    assert len(res_b["tiers"]["disabled"]["attempts"]) == 3
    assert seen_case_b[:3] == [{"enabled": False}, {"effort": "none"}, None]

    # Case C: enabled=false returns HTTP 200 but ignores disable (r_tok=42) -> effort=none achieves r_tok=0
    def _handler_c(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        reasoning = body.get("reasoning")
        if reasoning == {"enabled": False}:
            return httpx.Response(200, json=_ok_completion(pin["slug"], "DeepSeek", 42))
        if reasoning == {"effort": "none"}:
            return httpx.Response(200, json=_ok_completion(pin["slug"], "DeepSeek", 0))
        return httpx.Response(200, json=_ok_completion(pin["slug"], "DeepSeek", 30))

    async with httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(_handler_c)
    ) as client:
        res_c = await probe_mod.probe_worker_tiers(client, target, pin, probe_mod.BudgetState(1.0))

    assert res_c["tiers"]["disabled"]["ok"] is True
    assert res_c["tiers"]["disabled"]["reasoning_payload"] == {"effort": "none"}
    assert res_c["tiers"]["disabled"]["reasoning_disabled_verified"] is True

    # Case D: ALL 3 disabled payloads rejected -> verified_ladder starts cleanly at "low"
    def _handler_d(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        reasoning = body.get("reasoning")
        if reasoning in ({"enabled": False}, {"effort": "none"}, None):
            return httpx.Response(400, json={"error": "reasoning required"})
        return httpx.Response(200, json=_ok_completion(pin["slug"], "DeepSeek", 20))

    async with httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(_handler_d)
    ) as client:
        res_d = await probe_mod.probe_worker_tiers(client, target, pin, probe_mod.BudgetState(1.0))

    assert res_d["tiers"]["disabled"]["ok"] is False
    assert res_d["verified_ladder"] == ["low", "medium", "high"]


@pytest.mark.asyncio
async def test_gpt6_luna_diagnostic_none_and_truncation_retry_and_budget_cap() -> None:
    target = probe_mod.WORKER_TARGETS["gpt6_luna"]
    pin = _sample_worker_pin("gpt6_luna", "openai/gpt-6-luna", "OpenAI")

    high_attempts = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal high_attempts
        body = json.loads(request.content.decode("utf-8"))
        reasoning = body.get("reasoning") or {}
        effort = reasoning.get("effort")
        if effort == "none":
            return httpx.Response(400, json={"error": "effort='none' rejected"})
        if effort == "high":
            high_attempts += 1
            if body.get("max_tokens") == probe_mod.PROBE_MAX_TOKENS:
                return httpx.Response(
                    200,
                    json={
                        "model": "openai/gpt-6-luna",
                        "provider": "OpenAI",
                        "choices": [
                            {"finish_reason": "length", "message": {"role": "assistant", "content": ""}}
                        ],
                        "usage": {
                            "prompt_tokens": 14,
                            "completion_tokens": 256,
                            "completion_tokens_details": {"reasoning_tokens": 256},
                            "cost": 0.0005,
                        },
                    },
                )
            return httpx.Response(200, json=_ok_completion("openai/gpt-6-luna", "OpenAI", 310))
        return httpx.Response(
            200, json=_ok_completion("openai/gpt-6-luna", "OpenAI", 14 if effort == "low" else 52)
        )

    async with httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(_handler)
    ) as client:
        res = await probe_mod.probe_worker_tiers(client, target, pin, probe_mod.BudgetState(1.0))

    assert res["verified_ladder"] == ["low", "medium", "high"]
    assert res["tiers"]["low"]["reasoning_tokens"] == 14
    assert res["tiers"]["medium"]["reasoning_tokens"] == 52
    assert res["tiers"]["high"]["ok"] is True
    assert res["tiers"]["high"]["initial_finish_reason"] == "length"
    assert res["tiers"]["high"]["probe_max_tokens_used"] == probe_mod.PROBE_TRUNCATION_RETRY_MAX_TOKENS
    assert high_attempts == 2
    assert res["diagnostics"]["effort_none_probe"]["ok"] is False
    assert res["diagnostics"]["effort_none_probe"]["http_status"] == 400

    # Budget cap stops calls immediately once cap reached
    tiny_budget = probe_mod.BudgetState(cap_usd=0.000005)
    async with httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(_handler)
    ) as client:
        res_capped = await probe_mod.probe_worker_tiers(client, target, pin, tiny_budget)

    assert tiny_budget.stopped is True
    assert res_capped["tiers"]["low"]["ok"] is True
    assert res_capped["tiers"]["medium"]["skipped_budget_cap"] is True
    assert res_capped["tiers"]["high"]["skipped_budget_cap"] is True


# ---------------------------------------------------------------------------
# 4. Frontier pin copying from data/v2/probe_manifest_v2.json
# ---------------------------------------------------------------------------


def test_copy_frontier_pins_from_frozen_v2_manifest() -> None:
    sha, pins, meta, eps, live = probe_mod.copy_frontier_pins(PROBE_MANIFEST_V2_PATH)
    assert sha == "4490b33f8a5ebb30597598479e3692f208fbefe857008b461e453999d78761a1"
    assert set(pins.keys()) == {"astra62", "neon4", "opus55", "astra60"}
    assert set(meta.keys()) == {"astra62", "neon4", "opus55", "astra60"}
    assert set(eps.keys()) == {"astra62", "neon4", "opus55", "astra60"}
    assert set(live.keys()) == {"astra62", "neon4", "opus55", "astra60"}

    # Bumped frontier concurrency & locked medium effort
    for k, pin in pins.items():
        assert pin["receiver_class"] == "frontier"
        assert pin["concurrency"] == 16
        assert pin["starting_tier"] == "medium"
        assert pin["effort_ladder"] == ["medium"]
        assert pin["reasoning_payload"] == {"effort": "medium"}
        assert pin["copied_from_v2"] is True
        assert pin["supports_anthropic_cache_control"] is (k == "opus55")

    # Standard-tier pricing preserved (not flex-clipped for neon4 / astra60)
    assert pins["astra62"]["input_per_m"] == 8.0
    assert pins["astra62"]["output_per_m"] == 40.0
    assert pins["neon4"]["input_per_m"] == 3.0
    assert pins["neon4"]["output_per_m"] == 15.0
    assert pins["opus55"]["input_per_m"] == 4.0
    assert pins["opus55"]["output_per_m"] == 20.0
    assert pins["astra60"]["input_per_m"] == 10.0
    assert pins["astra60"]["output_per_m"] == 50.0


def test_copy_frontier_pins_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_mod.copy_frontier_pins(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# 5. JSON parser & full --mock end-to-end execution
# ---------------------------------------------------------------------------


def test_parse_probe_json_variants() -> None:
    assert probe_mod.parse_probe_json('{"ans": 17}') == (True, 17)
    assert probe_mod.parse_probe_json('```json\n{"ans": 17}\n```') == (True, 17)
    assert probe_mod.parse_probe_json('Result: {"ans": 17}') == (True, 17)
    assert probe_mod.parse_probe_json('{"ans": 16}') == (False, 16)
    assert probe_mod.parse_probe_json('{"ans": true}') == (False, None)
    assert probe_mod.parse_probe_json("seventeen") == (False, None)


def test_cli_mock_end_to_end_writes_manifest_to_tmp_path(tmp_path: Path) -> None:
    out_file = tmp_path / "probe_manifest_v2_1.json"
    rc = probe_mod.main(["--mock", "--out", str(out_file), "--budget-cap", "1.0"])
    assert rc == 0
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v2.1"
    assert data["mock"] is True
    assert data["budget_stopped"] is False
    assert 0.0 < data["total_probe_cost_usd"] < 0.01
    assert data["receiver_order"] == [
        "astra62",
        "neon4",
        "opus55",
        "astra60",
        "hydrogen",
        "haiku55",
        "deepseek_flash",
        "gpt6_luna",
    ]
    pins = data["verified_pins"]
    assert set(pins.keys()) == set(data["receiver_order"])

    # Worker specifics
    assert pins["hydrogen"]["slug"] == "google/gemini-4-hydrogen"
    assert pins["hydrogen"]["provider_only"] == ["Google AI Studio"]
    assert pins["hydrogen"]["input_per_m"] == 0.25  # standard tier, not flex
    assert pins["hydrogen"]["effort_ladder"] == ["low", "medium", "high"]
    assert pins["hydrogen"]["concurrency"] == 24

    assert pins["haiku55"]["slug"] == "anthropic/claude-haiku-5.5"
    assert pins["haiku55"]["provider_only"] == ["Anthropic"]
    assert pins["haiku55"]["supports_anthropic_cache_control"] is True
    assert pins["haiku55"]["effort_ladder"] == ["low", "medium", "high"]
    assert pins["haiku55"]["concurrency"] == 24

    assert pins["deepseek_flash"]["slug"] == "deepseek/deepseek-v4.1-flash-1017"
    assert pins["deepseek_flash"]["provider_only"] == ["DeepSeek"]
    assert pins["deepseek_flash"]["starting_tier"] == "disabled"
    assert pins["deepseek_flash"]["reasoning_payload"] == {"enabled": False}
    assert pins["deepseek_flash"]["effort_ladder"] == ["disabled", "low", "medium", "high"]
    assert pins["deepseek_flash"]["concurrency"] == 32

    assert pins["gpt6_luna"]["slug"] == "openai/gpt-6-luna"
    assert pins["gpt6_luna"]["provider_only"] == ["OpenAI"]
    assert pins["gpt6_luna"]["starting_tier"] == "low"
    assert pins["gpt6_luna"]["effort_ladder"] == ["low", "medium", "high"]
    assert pins["gpt6_luna"]["concurrency"] == 24
    assert data["tier_probes"]["gpt6_luna"]["diagnostics"]["effort_none_probe"]["ok"] is False

    # Rich `receivers` section matches dropout.load_receiver_manifest expectations
    receivers_raw = data["receivers"]
    assert set(receivers_raw.keys()) == set(data["receiver_order"])
    for fk in ("astra62", "neon4", "opus55", "astra60"):
        assert receivers_raw[fk]["tier"] == "frontier"
        assert receivers_raw[fk]["concurrency"] == 16
        assert receivers_raw[fk]["effort_ladder"][0]["reasoning_payload"] == {"effort": "medium"}
    for wk in ("hydrogen", "haiku55", "deepseek_flash", "gpt6_luna"):
        assert receivers_raw[wk]["tier"] == "worker"
        assert receivers_raw[wk]["concurrency"] in (24, 32)
    assert receivers_raw["deepseek_flash"]["effort_ladder"][0] == {
        "effort": "disabled",
        "tier_label": "disabled",
        "reasoning_payload": {"enabled": False},
        "aa_index_v5": 19.0,
    }
    assert receivers_raw["haiku55"]["supports_anthropic_cache_control"] is True

    # Direct integration check against dropout.load_receiver_manifest when available
    from common_ground.dropout import load_receiver_manifest

    loaded = load_receiver_manifest(out_file)
    assert set(loaded.keys()) == set(data["receiver_order"])
    assert loaded["astra60"].tier == "frontier"
    assert loaded["opus55"].supports_anthropic_cache_control is True
    assert loaded["haiku55"].supports_anthropic_cache_control is True
    assert loaded["deepseek_flash"].tier == "worker"
    assert loaded["deepseek_flash"].effort_ladder[0].effort == "disabled"
    assert loaded["deepseek_flash"].effort_ladder[0].reasoning_payload == {"enabled": False}
    assert [s.effort for s in loaded["hydrogen"].effort_ladder] == ["low", "medium", "high"]


def test_build_receiver_entry_handles_omit_reasoning_payload() -> None:
    pin = {
        "key": "deepseek_flash",
        "slug": "deepseek/deepseek-v4.1-flash-1017",
        "family": "deepseek",
        "display_name": "DeepSeek V4.1 Flash",
        "tier": "worker",
        "provider_only": ["DeepSeek"],
        "concurrency": 32,
        "supports_anthropic_cache_control": False,
        "input_per_m": 0.2,
        "output_per_m": 0.8,
        "max_tokens": 16384,
        "starting_tier": "disabled",
        "effort_ladder": ["disabled", "low"],
        "tier_reasoning_payloads": {"disabled": None, "low": {"effort": "low"}},
        "aa_index_v5": 19.0,
        "aa_index_v5_by_tier": {"disabled": 19.0, "low": None},
    }
    entry = probe_mod._build_receiver_entry_from_pin(pin)
    assert entry["default_effort"] == "omit"
    assert entry["reasoning_payload"] == {}
    assert entry["effort_ladder"][0]["effort"] == "omit"
    assert entry["effort_ladder"][0]["reasoning_payload"] == {}
    assert entry["effort_ladder"][1]["effort"] == "low"
    assert entry["effort_ladder"][1]["reasoning_payload"] == {"effort": "low"}


def test_cli_mock_without_out_does_not_touch_default_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_default = Path("data/v2/probe_manifest_v2_1.json")
    existed_before = real_default.exists()
    before_sig: tuple[int, str] | None = None
    if existed_before:
        before_sig = (
            real_default.stat().st_mtime_ns,
            hashlib.sha256(real_default.read_bytes()).hexdigest(),
        )

    sentinel_default = tmp_path / "should_not_be_written.json"
    monkeypatch.setattr(probe_mod, "PROBE_MANIFEST_V2_1_PATH", sentinel_default)

    rc = probe_mod.main(["--mock", "--budget-cap", "1.0"])
    assert rc == 0
    assert not sentinel_default.exists()

    if existed_before:
        assert real_default.exists()
        after_sig = (
            real_default.stat().st_mtime_ns,
            hashlib.sha256(real_default.read_bytes()).hexdigest(),
        )
        assert after_sig == before_sig
    else:
        assert not real_default.exists()
