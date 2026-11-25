"""Model specifications, provider pins, decision panel specs, and v2 experiment configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROBE_REPORT_PATH = DATA_DIR / "probe_report.json"  # Legacy v1 probe (NEVER auto-loaded in v2)
PROBE_MANIFEST_V2_PATH = DATA_DIR / "v2" / "probe_manifest_v2.json"
DEFAULT_API_KEY_PATH = ROOT_DIR / "openrouter_key.txt"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    slug: str
    family: str
    display_name: str
    role: Literal["core", "partner_only", "predecessor", "rewriter"]
    provider_only: tuple[str, ...]
    reasoning_effort: str | None = "medium"
    max_tokens: int = 16384
    supports_anthropic_cache_control: bool = False
    fallback_input_per_m: float = 0.0
    fallback_output_per_m: float = 0.0
    concurrency: int = 8
    aa_index_v5: float | None = None
    verified_returned_model: str | None = None
    verified_provider: str | None = None
    notes: str = ""


# Verified v2 roster; provider_only and pricing are verified/pinned into data/v2/probe_manifest_v2.json
DEFAULT_MODELS: dict[str, ModelSpec] = {
    "astra62": ModelSpec(
        key="astra62",
        slug="openai/gpt-6.2-astra",
        family="openai",
        display_name="GPT-6.2 Astra",
        role="core",
        provider_only=("OpenAI",),
        reasoning_effort="medium",
        max_tokens=16384,
        fallback_input_per_m=8.0,
        fallback_output_per_m=40.0,
        concurrency=8,
        aa_index_v5=62.0,
        notes="Core chooser & partner (AA v5 = 62); rejects temperature/top_p and effort='none'.",
    ),
    "neon4": ModelSpec(
        key="neon4",
        slug="google/gemini-4-neon",
        family="google",
        display_name="Gemini 4 Neon",
        role="core",
        provider_only=("Google AI Studio",),
        reasoning_effort="medium",
        max_tokens=16384,
        fallback_input_per_m=3.0,
        fallback_output_per_m=15.0,
        concurrency=8,
        aa_index_v5=61.0,
        notes="Core chooser & partner (AA v5 = 61).",
    ),
    "opus55": ModelSpec(
        key="opus55",
        slug="anthropic/claude-opus-5.5",
        family="anthropic",
        display_name="Claude Opus 5.5",
        role="core",
        provider_only=("Anthropic",),
        reasoning_effort="medium",
        max_tokens=16384,
        supports_anthropic_cache_control=True,
        fallback_input_per_m=4.0,
        fallback_output_per_m=20.0,
        concurrency=8,
        aa_index_v5=59.0,
        notes="Core chooser & partner (AA v5 = 59); adaptive thinking mandatory.",
    ),
    "kimi31": ModelSpec(
        key="kimi31",
        slug="moonshotai/kimi-k3.1-pro",
        family="moonshot",
        display_name="Kimi K3.1 Pro",
        role="core",
        provider_only=("Moonshot AI",),
        reasoning_effort="medium",
        max_tokens=16384,
        fallback_input_per_m=2.0,
        fallback_output_per_m=10.0,
        concurrency=8,
        aa_index_v5=46.0,
        notes="Core chooser & partner (AA v5 = 46); open-weight MoE.",
    ),
    "astra60": ModelSpec(
        key="astra60",
        slug="openai/gpt-6-astra",
        family="openai",
        display_name="GPT-6 Astra",
        role="partner_only",
        provider_only=("OpenAI",),
        reasoning_effort="medium",
        max_tokens=16384,
        fallback_input_per_m=10.0,
        fallback_output_per_m=50.0,
        concurrency=8,
        aa_index_v5=57.0,
        notes="Partner-only predecessor control (AA v5 = 57), role-balanced A/B against all 4 core choosers.",
    ),
    "muse": ModelSpec(
        key="muse",
        slug="meta/muse-wonder",
        family="meta",
        display_name="Meta Muse Wonder",
        role="partner_only",
        provider_only=("Meta",),
        reasoning_effort="medium",
        max_tokens=16384,
        fallback_input_per_m=4.0,
        fallback_output_per_m=20.0,
        concurrency=8,
        aa_index_v5=44.0,
        notes="Partner-only 5th-lab frontier model @ Meta (AA v5 = 44), role-balanced A/B.",
    ),
    "qwen4": ModelSpec(
        key="qwen4",
        slug="qwen/qwen4-max",
        family="qwen",
        display_name="Qwen4 Max",
        role="partner_only",
        provider_only=("Alibaba",),
        reasoning_effort="medium",
        max_tokens=16384,
        fallback_input_per_m=1.425,
        fallback_output_per_m=4.275,
        concurrency=8,
        aa_index_v5=43.0,
        notes="Partner-only 6th-lab frontier model @ Alibaba (AA v5 = 43), role-balanced A/B.",
    ),
    "deepseek": ModelSpec(
        key="deepseek",
        slug="deepseek/deepseek-v4-pro-0813",
        family="deepseek",
        display_name="DeepSeek V4 Pro",
        role="rewriter",
        provider_only=("BaseTen",),
        reasoning_effort="low",
        max_tokens=4096,
        fallback_input_per_m=1.0,
        fallback_output_per_m=3.0,
        concurrency=10,
        aa_index_v5=None,
        notes="Neutral 7th-family rewriter @ BaseTen for Capstone tone-flip study; zero overlap with evaluated labs.",
    ),
}

CORE_MODELS: tuple[str, ...] = ("astra62", "opus55", "neon4", "kimi31")
PARTNER_ONLY_MODELS: tuple[str, ...] = ("astra60", "muse", "qwen4")
PREDECESSOR_MODEL: str = "astra60"
REWRITER_MODEL: str = "deepseek"
ALL_PARTNER_MODELS: tuple[str, ...] = (
    "astra62",
    "neon4",
    "opus55",
    "astra60",
    "kimi31",
    "muse",
    "qwen4",
)

AA_INDEX_V5: dict[str, float] = {
    k: float(spec.aa_index_v5)
    for k, spec in DEFAULT_MODELS.items()
    if spec.aa_index_v5 is not None
}

# 6 evaluated lab families (used in F5 with deterministic per-call randomized order + "Other / Unsure")
FAMILY_LABELS: dict[str, str] = {
    "openai": "OpenAI (GPT)",
    "anthropic": "Anthropic (Claude)",
    "google": "Google (Gemini)",
    "moonshot": "Moonshot (Kimi)",
    "meta": "Meta (Muse / Llama)",
    "qwen": "Alibaba (Qwen)",
}

# 3-Model Vendor-Independent Decision Panel (/api/alpha/decisions, zero overlap with evaluated labs)
DECISION_PANEL_SPECS: dict[str, dict[str, Any]] = {
    "liquid": {
        "slug": "liquid/d1.2",
        "provider": "Liquid",
        "fallback_input_per_m": 0.02,
    },
    "perplexity": {
        "slug": "perplexity/pplx-decider-v1.2",
        "provider": "Perplexity",
        "fallback_input_per_m": 0.02,
    },
    "typesafe": {
        "slug": "typesafe/jev-1.14",
        "provider": "TypeSafe",
        "fallback_input_per_m": 0.042,
    },
}


def load_api_key(key_path: Path | None = None) -> str:
    """Load OpenRouter API key from environment or local gitignored key file."""
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    path = key_path or DEFAULT_API_KEY_PATH
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise RuntimeError(
        f"OpenRouter API key not found in OPENROUTER_API_KEY or {path}. "
        "Place key in openrouter_key.txt or use --mock."
    )


def load_verified_models(
    probe_path: Path | None = None,
    *,
    version: str = "v2",
    strict: bool = False,
) -> dict[str, ModelSpec]:
    """Load model specs, applying verified pins strictly from the version-specific probe manifest.

    NEVER auto-loads legacy v1 `data/probe_report.json` when running `v2`.
    When `strict=True` (enforced on live v2 runs), raises FileNotFoundError if the probe manifest
    is missing rather than silently falling back to unpinned defaults.
    """
    models = dict(DEFAULT_MODELS)
    if probe_path is not None:
        path = probe_path
    elif version == "v2":
        path = PROBE_MANIFEST_V2_PATH
    else:
        path = PROBE_REPORT_PATH

    if not path.exists():
        if strict:
            raise FileNotFoundError(
                f"Required probe manifest missing: {path}. Run scripts/probe_models.py or pass --probe-manifest."
            )
        return models

    data = json.loads(path.read_text(encoding="utf-8"))
    pins = data.get("verified_pins", {})
    live_tests = data.get("live_tests", {})
    for key, override in pins.items():
        if key not in models:
            continue
        base = models[key]
        lt = live_tests.get(key) or {}
        models[key] = ModelSpec(
            key=base.key,
            slug=override.get("slug", base.slug),
            family=base.family,
            display_name=base.display_name,
            role=base.role,
            provider_only=tuple(override.get("provider_only", base.provider_only)),
            reasoning_effort=override.get("reasoning_effort", base.reasoning_effort),
            max_tokens=int(override.get("max_tokens", base.max_tokens)),
            supports_anthropic_cache_control=base.supports_anthropic_cache_control,
            fallback_input_per_m=float(
                max(base.fallback_input_per_m, override.get("input_per_m", base.fallback_input_per_m))
            ),
            fallback_output_per_m=float(
                max(base.fallback_output_per_m, override.get("output_per_m", base.fallback_output_per_m))
            ),
            concurrency=base.concurrency,
            aa_index_v5=base.aa_index_v5,
            verified_returned_model=override.get("returned_model") or lt.get("returned_model"),
            verified_provider=override.get("returned_provider") or lt.get("provider"),
            notes=base.notes,
        )
    return models


@dataclass
class RunConfig:
    run_name: str
    budget_cap_usd: float
    mock: bool = False
    max_concurrent_episodes: int = 16
    rating_scale_max: int = 7
    run_f3: bool = False
    probe_manifest_path: Path | None = None
    models: dict[str, ModelSpec] = field(default_factory=load_verified_models)
