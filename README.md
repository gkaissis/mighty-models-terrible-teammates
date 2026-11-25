# Mighty Models, Terrible Teammates

Code, deterministic verifiers, transcripts, causal crossover experiments, publication figures, and full offline reproducibility package accompanying the technical report **Mighty Models, Terrible Teammates** (George Kaissis and Florian A. Hölzl, Human-Centred Transformative AI Lab (HTAIL), Hasso-Plattner Institute, 25th of November 2026).

- **Web Report (GitHub Pages):** <https://gkaissis.github.io/mighty-models-terrible-teammates/> (Astro source in [`site/`](site/))
- **Markdown Companion:** [`docs/blog_post.md`](docs/blog_post.md)
- **Parquet Dataset (HuggingFace Hub):** <https://huggingface.co/datasets/gkaissis/mighty-models-terrible-teammates> (local copy in [`data/v2/hf_export/`](data/v2/hf_export/))

## Overview

This repository evaluates how live information flow between language models shapes multi-agent collaboration, downstream handoff reliability, partner compute costs, and peer teammate selection. Across four pure-text collaborative engineering tasks with private constraints and mid-run shocks, we collect:

- **702 completed two-agent episodes** (`638` core four-task benchmark runs across seeds `301–304` + `64` extra cross-model runs across seeds `305–308`, with `2` incomplete episodes logged separately) across **7 frontier models** from 6 providers.
- **3,178 live pairwise teammate choices**, **384 tone-rewrite crossover choices** (Warm vs. Blunt via DeepSeek V4 Pro), and **480 missing-state crossover choices** (`384` main + `96` length-matched public-recap placebo).
- **2,560 deterministic mid-task dropout handoffs** (`320` cases across **8 takeover models**: 4 frontier models + 4 low-cost worker models) plus `256` floor/ceiling calibration checks.
- **9,568 main-arm turns** scored by a 3-model vendor-independent decision panel (`liquid/d1.2`, `perplexity/pplx-decider-v1.2`, `typesafe/jev-1.14`).

## Repository Layout

- `site/`: Astro technical report (`src/pages/index.md`, `refs.bib`, `src/figures/*.svg`, `public/data/*.json`) deployed to GitHub Pages via `.github/workflows/deploy.yml`.
- `docs/`: Standalone Markdown version of the report (`blog_post.md`), publication figures (`assets/figures/*.svg`, `*.png`), and downloadable figure data (`assets/data/*.json`).
- `src/common_ground/`: Core evaluation harness, deterministic task environments (`triage`, `calendar`, `goalpost` / Config Repair, `code_handoff`), OpenRouter client, decision panel scorer, disclosure parsers, dropout takeover harness, missing-state crossover harness, meta-analysis, quote verifier, and Parquet exporter.
- `scripts/`: Reproducible entry points for probing endpoints, running experiments, running Phase 1 meta-analysis, generating publication figures, and verifying lossless Parquet replay.
- `tests/`: Deterministic unit, verifier, quote-audit, and Parquet-replay test suite.
- `data/v2/`: Frozen manifests (`manifest_v2.json`, `probe_manifest_v2.json`, `probe_manifest_v2_1.json`), analysis summaries (`analysis/`), experiment summaries (`exp1_dropout/`, `exp2_info_flip/`), and lossless zstd-compressed Parquet release (`hf_export/`, containing all episode transcripts, turn-level billing/latency telemetry, choices, and causal experiment tables).
- `data/v2_capstone/`: Summary results (`capstone_results.json`) for the Tone Rewrite crossover.

## Evaluated Models (Pinned via OpenRouter)

- **Core evaluators & partners (`reasoning.effort = "medium"`):** `openai/gpt-6.2-astra` (`OpenAI`), `google/gemini-4-neon` (`Google AI Studio`), `anthropic/claude-opus-5.5` (`Anthropic`), `moonshotai/kimi-k3.1-pro` (`Moonshot AI`).
- **Partner-only models (`reasoning.effort = "medium"`):** `openai/gpt-6-astra` (`OpenAI`), `meta/muse-wonder` (`Meta`), `qwen/qwen4-max` (`Alibaba`).
- **Dropout low-cost workers:** `google/gemini-4-hydrogen` (`low`), `anthropic/claude-haiku-5.5` (`low`), `deepseek/deepseek-v4.1-flash` (`disabled`, 0 reasoning tokens, `BaseTen`), `openai/gpt-6-luna` (`low`).
- **Independent decision models:** `liquid/d1.2` (`Liquid`), `perplexity/pplx-decider-v1.2` (`Perplexity`), `typesafe/jev-1.14` (`TypeSafe`).
- **Independent tone rewriter:** `deepseek/deepseek-v4-pro-0813` (`BaseTen`, `low`).

## Quickstart & Zero-Cost Local Reproduction

Install dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

All statistical tables, quote verifications, publication figures, and Parquet-only replays reproduce deterministically offline with zero API calls:

```bash
# 1. Lint and run the full deterministic test suite ($0)
uv run ruff check src tests scripts
uv run pytest -v

# 2. Verify every headline claim reproduces strictly from data/v2/hf_export/*.parquet ($0)
uv run python scripts/verify_hf_replay.py --run v2 --capstone-run v2_capstone

# 3. Re-run Phase 1 meta-analysis from raw data/v2/ transcripts & ledgers ($0)
uv run python scripts/run_phase1_meta_analysis.py --run-dir data/v2

# 4. Regenerate all eight SVG/PNG figures and downloadable JSON payloads ($0)
uv run python scripts/generate_v2_1_figures.py
```

### Re-Running Live API Experiments (Optional)

Set `OPENROUTER_API_KEY` in your environment (or place it in a gitignored `openrouter_key.txt`). Always isolate mock or fresh runs with a distinct `--run-name` / `--out-dir` so frozen `data/v2/` artefacts are never overwritten:

```bash
# Offline mock smoke test (isolated run directory)
uv run python -m common_ground.runner --phase v2 --run-name v2_mock --mock --budget-cap 5000

# Main live benchmark (enforces frozen manifest & probe pins)
uv run python -m common_ground.runner --phase v2 --run-name v2_repro --manifest data/v2/manifest_v2.json --budget-cap 620

# Control 1: Tone-rewrite crossover
uv run python scripts/run_capstone_tone_flip.py --source-run v2_repro --out-run v2_capstone_repro --n-pairs 24 --budget-cap 80

# Section 5: Mid-task dropout handoffs
uv run python scripts/run_exp1_dropout.py --full --episodes-dir data/v2_repro/episodes --out-dir data/v2_repro/exp1_dropout --budget-cap 120

# Control 2: Missing-state injection crossover + placebo
uv run python scripts/run_exp2_info_flip.py --episodes-dir data/v2_repro/episodes --capstone-dir data/v2_repro/capstone_tone_flip --out-dir data/v2_repro/exp2_info_flip --budget-cap 30

# Export lossless zstd Parquet dataset
uv run python -m common_ground.export_hf --run v2_repro
```

## Disclosures & License

- Bare chat-completion API endpoints are evaluated without vendor coding harnesses, tools, web search, or external memory.
- Author disclosure: George Kaissis is a research consultant and former member of technical staff at Google DeepMind. Two Google models (`google/gemini-4-neon` and `google/gemini-4-hydrogen`) appear in this study. Model endpoints, prompts, task seeds, and calibration checks were fixed before running each evaluation batch, and all raw transcripts, verdicts, and analysis scripts are released publicly.
- Licensed under Apache-2.0 (see [`LICENSE`](LICENSE)).
