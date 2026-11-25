# Experiment 1 (Partner Drops Out) — Floor-vs.-Ceiling Gate Report

- **Timestamp (UTC):** `2026-11-23T11:28:46.839283+00:00`
- **Mode:** `LIVE`
- **Gate Cases:** `16` (`4` per seed × `4` seeds, `8` Role A + `8` Role B)
- **Worker Auto Step-Up Threshold:** `ceiling_strict_pass_rate >= 40%` (max `2` step-ups; frontier locked at `medium`)
- **Total Gate Cost:** `$6.3159` (ledger cumulative: `$6.3159`)

## Locked Receiver Roster Summary

| Receiver | Tier | Family | Effort (Init → Locked) | Step-Ups | AA v5 | Floor Strict | Ceiling Strict | Ceiling Own+R3 | Ceiling Partner | Δ (Ceil−Floor) | Gate Status | Cost ($) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| `astra62` | frontier | openai | `medium` | 0 | 62.0 | 31.2% | 100.0% | 100.0% | 100.0% | +68.8 pp | LOCKED_FRONTIER | 0.4873 |
| `neon4` | frontier | google | `medium` | 0 | 61.0 | 43.8% | 100.0% | 100.0% | 100.0% | +56.2 pp | LOCKED_FRONTIER | 2.9381 |
| `opus55` | frontier | anthropic | `medium` | 0 | 59.0 | 25.0% | 100.0% | 100.0% | 100.0% | +75.0 pp | LOCKED_FRONTIER | 1.4766 |
| `astra60` | frontier | openai | `medium` | 0 | 57.0 | 31.2% | 100.0% | 100.0% | 100.0% | +68.8 pp | LOCKED_FRONTIER | 0.7313 |
| `hydrogen` | worker | google | `low` | 0 | 27.0 | 31.2% | 100.0% | 100.0% | 100.0% | +68.8 pp | PASS | 0.5662 |
| `haiku55` | worker | anthropic | `low` | 0 | 19.0 | 6.2% | 100.0% | 100.0% | 100.0% | +93.8 pp | PASS | 0.0455 |
| `deepseek_flash` | worker | deepseek | `disabled` | 0 | 19.0 | 31.2% | 100.0% | 100.0% | 100.0% | +68.8 pp | PASS | 0.0581 |
| `gpt6_luna` | worker | openai | `low` | 0 | — | 18.8% | 93.8% | 93.8% | 93.8% | +75.0 pp | PASS | 0.0129 |

## Step-Up Audit Trail

### `astra62` (GPT-6.2 Astra — `frontier`)
- **Step 0 (`medium`)**: Floor strict `31.2%` (own+R3 `100.0%`), Ceiling strict `100.0%` (own+R3 `100.0%`, partner `100.0%`), Δ `+68.8 pp` — locked_frontier

### `neon4` (Gemini 4 Neon — `frontier`)
- **Step 0 (`medium`)**: Floor strict `43.8%` (own+R3 `100.0%`), Ceiling strict `100.0%` (own+R3 `100.0%`, partner `100.0%`), Δ `+56.2 pp` — locked_frontier

### `opus55` (Claude Opus 5.5 — `frontier`)
- **Step 0 (`medium`)**: Floor strict `25.0%` (own+R3 `100.0%`), Ceiling strict `100.0%` (own+R3 `100.0%`, partner `100.0%`), Δ `+75.0 pp` — locked_frontier

### `astra60` (GPT-6 Astra — `frontier`)
- **Step 0 (`medium`)**: Floor strict `31.2%` (own+R3 `100.0%`), Ceiling strict `100.0%` (own+R3 `100.0%`, partner `100.0%`), Δ `+68.8 pp` — locked_frontier

### `hydrogen` (Gemini 4 Hydrogen — `worker`)
- **Step 0 (`low`)**: Floor strict `31.2%` (own+R3 `100.0%`), Ceiling strict `100.0%` (own+R3 `100.0%`, partner `100.0%`), Δ `+68.8 pp` — passed_and_locked

### `haiku55` (Claude Haiku 5.5 — `worker`)
- **Step 0 (`low`)**: Floor strict `6.2%` (own+R3 `100.0%`), Ceiling strict `100.0%` (own+R3 `100.0%`, partner `100.0%`), Δ `+93.8 pp` — passed_and_locked

### `deepseek_flash` (DeepSeek V4.1 Flash — `worker`)
- **Step 0 (`disabled`)**: Floor strict `31.2%` (own+R3 `100.0%`), Ceiling strict `100.0%` (own+R3 `100.0%`, partner `100.0%`), Δ `+68.8 pp` — passed_and_locked

### `gpt6_luna` (GPT-6 Luna — `worker`)
- **Step 0 (`low`)**: Floor strict `18.8%` (own+R3 `100.0%`), Ceiling strict `93.8%` (own+R3 `93.8%`, partner `93.8%`), Δ `+75.0 pp` — passed_and_locked
