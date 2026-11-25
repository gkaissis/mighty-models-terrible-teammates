---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - multi-agent
  - llm-evaluation
  - handoff-reliability
  - communication-style
  - cross-vendor
pretty_name: "Mighty Models, Terrible Teammates (v2)"
configs:
  - config_name: episodes
    data_files:
      - split: train
        path: episodes.parquet
  - config_name: turns
    data_files:
      - split: train
        path: turns.parquet
  - config_name: debriefs
    data_files:
      - split: train
        path: debriefs.parquet
  - config_name: choices
    data_files:
      - split: train
        path: choices.parquet
  - config_name: failed_episodes
    data_files:
      - split: train
        path: failed_episodes.parquet
  - config_name: choices_excluded
    data_files:
      - split: train
        path: choices_excluded.parquet
  - config_name: capstone_tone_flip
    data_files:
      - split: train
        path: capstone_tone_flip.parquet
  - config_name: capstone_rewrites
    data_files:
      - split: train
        path: capstone_rewrites.parquet
  - config_name: exp1_dropout_gate
    data_files:
      - split: train
        path: exp1_dropout_gate.parquet
  - config_name: exp1_dropout
    data_files:
      - split: train
        path: exp1_dropout.parquet
  - config_name: exp1_dropout_attempts
    data_files:
      - split: train
        path: exp1_dropout_attempts.parquet
  - config_name: exp2_info_flip_pairs
    data_files:
      - split: train
        path: exp2_info_flip_pairs.parquet
  - config_name: exp2_info_flip_choices
    data_files:
      - split: train
        path: exp2_info_flip_choices.parquet
---

# Mighty Models, Terrible Teammates: Dataset (`v2`)

Key-scrubbed, zstd-compressed Parquet release containing all trajectories, counterfactual choices, deterministic handoff evaluations, and surgical intervention artefacts required to independently reproduce the technical report.

- **Technical Report:** https://gkaissis.github.io/mighty-models-terrible-teammates/
- **Source Code and Reproducibility Package:** https://github.com/gkaissis/mighty-models-terrible-teammates

## Tables

- `episodes.parquet` (`702` rows): completed two-agent episodes across `calendar`, `code_handoff`, `goalpost`, `triage` (`638` main-arm + `64` isolated `opus_boost`), including role briefings, realized shocks, and deterministic scoring details.
- `failed_episodes.parquet` (`2` rows): incomplete episodes excluded from primary analysis for transparent completion accounting (`702/704`).
- `turns.parquet` (`10,528` rows; `9,568` main-arm turns, `10,528` ledger-joined): turn-level visible messages, structured actions, verifier feedback, linguistic metrics, 3-judge panel consensus and raw judge outputs, accepted/total token and cost telemetry, and per-turn API `latency_s`.
- `debriefs.parquet` (`2,137` rows): private post-episode evaluations (`F1` 1-7 Likert rating, `F1_retest` 25% stratified test-retest subset, and `F4` binary `KEEP`/`REROLL` retention).
- `choices.parquet` (`3,178` rows): order-balanced spliced-history forced teammate choices across all 7 partners, including full `choice_packet_json` prompts and free-text rationales.
- `choices_excluded.parquet` (`22` rows): choice items excluded due to incomplete upstream episodes (`3178/3200`).
- `capstone_tone_flip.parquet` (`384` rows): symmetric Warm vs. Blunt third-person staffing crossover choices with blinded transcript packets and objective outcome cards.
- `capstone_rewrites.parquet` (`96` rows across `48` candidate episodes): DeepSeek V4 Pro tone rewrites with turn-level automated and 3-judge demeanor/fidelity audits.
- `exp1_dropout_gate.parquet` (`256` rows): Experiment 1 calibration gate evaluations across 8 receivers x 16 cases x 2 conditions (`floor_no_partner` and `ceiling_verbatim_rules`), with deterministic 15-rule grading and raw model submissions.
- `exp1_dropout.parquet` (`2,560` rows): Experiment 1 Partner-Drops-Out single-turn handoff outcomes across 320 cases x 8 receivers, including deterministic grading (`own_r3_ok`, `partner_7_ok`, `strict_15_ok`), disclosure/bloat bins, token/latency/cost telemetry, and raw submissions.
- `exp1_dropout_attempts.parquet` (`5,321` rows): key-scrubbed attempt-level API ledger for Experiment 1 (calibration gate, accepted matrix calls, retries, and Gemini 4 Neon output-limit truncation attempts).
- `exp2_info_flip_pairs.parquet` (`30` rows: `24` main + `6` placebo): exact word-matched Round-1 surgical edit pairs (`private_info` vs. `public_echo` vs. `unedited`) with zero-leak and parity verification fields.
- `exp2_info_flip_choices.parquet` (`480` rows: `384` main + `96` placebo): symmetric Information-Flip third-person staffing crossover decisions across 4 frontier reviewers.

## Frozen Manifests

- `manifests/manifest_v2.json`
- `manifests/probe_manifest_v2.json`
- `manifests/probe_manifest_v2_1.json`
- `manifests/locked_receiver_roster.json`

## Pinned Endpoints and Evaluation Protocol

- **Main 7-partner collaboration league (OpenRouter first-party pins, `reasoning.effort = "medium"`):**
  - Core choosers and partners: `openai/gpt-6.2-astra` (`OpenAI`), `google/gemini-4-neon` (`Google AI Studio`), `anthropic/claude-opus-5.5` (`Anthropic`), `moonshotai/kimi-k3.1-pro` (`Moonshot AI`).
  - Partner-only controls: `openai/gpt-6-astra` (`OpenAI`), `meta/muse-wonder` (`Meta`), `qwen/qwen4-max` (`Alibaba`).
- **Vendor-independent 3-judge demeanor panel (`/api/alpha/decisions`):** `liquid/d1.2` (`Liquid`), `perplexity/pplx-decider-v1.2` (`Perplexity`), `typesafe/jev-1.14` (`TypeSafe`).
- **Neutral Capstone tone rewriter:** `deepseek/deepseek-v4-pro-0813` (`BaseTen`, `reasoning.effort = "low"`).
- **Experiment 1 worker-tier receivers:** `google/gemini-4-hydrogen` (`Google AI Studio`, `low`), `anthropic/claude-haiku-5.5` (`Anthropic`, `low`), `openai/gpt-6-luna` (`OpenAI`, `low`), and `deepseek/deepseek-v4.1-flash` (`BaseTen`, reasoning explicitly disabled with `reasoning.enabled = false`, verified `0` reasoning tokens across all gate and matrix calls).

## Independent Replay Verification

Every headline table, statistical test, deterministic dropout verdict, and causal intervention result in the report reproduces strictly from these Parquet files without accessing any external raw logs or making any API calls:

```bash
uv run python scripts/verify_hf_replay.py --run v2 --capstone-run v2_capstone
```

## Hygiene and Scope

- Format: zstd-compressed Parquet tables, frozen JSON manifests, and this dataset card only (no duplicate uncompressed `.jsonl` tables).
- Security: verified zero API keys or private credentials across all exported files.
