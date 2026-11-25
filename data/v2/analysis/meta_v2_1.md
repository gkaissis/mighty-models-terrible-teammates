# Mighty Models, Terrible Teammates — Phase 1 Meta-Analysis Report

- **Episodes:** `702` complete (`638` main arm + `64` isolated `opus_boost`; incomplete excluded: `2`)
- **Choices:** `3178` total (`2926` non-AA, `2670` `main_7p`, `1522` exact-score ties)
- **Capstone:** `96` rewrite runs, `384` decisions
- **Disclosure hook:** `complete_with_disclosure`

## 1. Construct Validity & Verbosity Confound (`main` arm only)

- OLS `R^2` (`~ log_wc + log_wc^2 + task + role`, `n = 9568` turns):
  - `clarity`: **`R^2 = 0.5351`**
  - `skips_why`: **`R^2 = 0.3458`**
  - `wooden_legalistic`: **`R^2 = 0.1389`**

| Model | Mean Words | Clarity Raw (Rank) | Clarity Resid (Rank) | Skips-Why Raw | Skips-Why Resid |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `opus55` | 244.9 | 3.0136 (#1) | +0.1819 (#1) | 0.1937 | -0.0356 |
| `qwen4` | 318.5 | 2.7447 (#2) | -0.1753 (#7) | 0.2274 | +0.0229 |
| `neon4` | 131.2 | 2.6802 (#3) | +0.0716 (#2) | 0.2351 | -0.0440 |
| `muse` | 124.6 | 2.6286 (#4) | +0.0201 (#3) | 0.2719 | -0.0071 |
| `astra60` | 90.6 | 2.4915 (#5) | -0.0427 (#5) | 0.3379 | +0.0409 |
| `kimi31` | 60.3 | 2.3922 (#6) | -0.0188 (#4) | 0.3310 | +0.0157 |
| `astra62` | 79.7 | 2.3866 (#7) | -0.1198 (#6) | 0.3338 | +0.0308 |

## 2. Visibility-Masked Cluster-Robust Choice Logits & Tone-Mention Audit

| Predictor (1-SD delta) | Multivariate (No FE) | Multivariate (+ Partner FE) | Exact Ties (+ Partner FE) |
| :--- | :---: | :---: | :---: |
| `d_score` | `+1.750` (p=< 0.001) | `+1.684` (p=< 0.001) | — |
| `d_clarity_resid` | `-0.092` (p=0.4441) | `-0.223` (p=0.1243) | `-0.129` (p=0.4343) |
| `d_log_wc` | `+0.549` (p=0.0068) | `+0.170` (p=0.5800) | `-0.050` (p=0.8978) |
| `d_skips_why` | `+0.166` (p=0.4283) | `-0.079` (p=0.7738) | `-0.035` (p=0.9027) |
| `d_wooden_legalistic` | `+0.384` (p=0.0034) | `+0.449` (p=0.0135) | `+0.253` (p=0.1744) |
| `d_warmth` | `+0.012` (p=0.9403) | `-0.062` (p=0.7870) | `-0.079` (p=0.7606) |
| `d_answers_directly` | `+0.442` (p=0.0053) | `+0.603` (p=< 0.001) | `+0.647` (p=< 0.001) |
| `d_condescending_pedantic` | `-0.408` (p=0.0295) | `-0.196` (p=0.3907) | `+0.033` (p=0.8909) |
| `d_s_conf` | `-0.328` (p=0.0056) | `-0.291` (p=0.0142) | `-0.203` (p=0.0687) |
| `d_ids_per_100` | `+0.197` (p=0.0746) | `+0.159` (p=0.1264) | `+0.241` (p=0.0408) |

- **Zero-tolerance politeness/warmth mention audit (live non-AA):** **`0/2926`** (`0.00%`); terse/tone mentions: `29/2926`.
- **Capstone tone-mention audit:** strict politeness `5/384`, terse/tone `7/384`.

## 3. Capstone Multi-Dimension Audit & 4-Lab Consensus

- **Turn-level Warm − Blunt deltas (`n = 360` paired turns):** `warmth = +1.077`, `clarity = +0.297`, `skips_why = -0.129`, `wooden_legalistic = -0.159` — while flipping **`0/96`** cases.
- **4-lab third-person consensus (all 24 pairs):** `4-0` unanimous = `21`, `3-1` strict = `1`, `3-0+split` = `2`, `2-2` = `0` (mean pairwise exact decision agreement = `0.9392`).
- **4-lab consensus (10 exact-tied pairs):** `4-0` = `7`, `3-1` strict = `1`, `3-0+split` = `2`, `2-2` = `0`.
- **Live `main_7p` inter-chooser Spearman contrast:** pooled mean pairwise `ρ = 0.5177`, within-task mean pairwise `ρ = 0.5117`.

## 4. Bloat Tax & Verbosity Contagion (`exposed_turns`, main arm)

| Partner | Prompt Tokens % | Completion Tokens % | Reasoning Tokens % | Latency % | Receiver Words % | Receiver Cost % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `astra62` | -17.0% | -5.5% | +1.5% | -4.7% | -10.2% | -9.0% |
| `neon4` | +1.3% | -2.9% | -13.0% | -4.1% | -1.3% | -2.5% |
| `opus55` | +17.7% | +4.5% | +8.5% | +3.5% | +11.9% | +8.8% |
| `astra60` | -12.5% | +2.8% | -3.9% | +2.5% | +0.0% | -3.0% |
| `kimi31` | -20.7% | -12.4% | -5.5% | -9.3% | -18.7% | -15.3% |
| `muse` | -1.6% | -0.9% | -0.3% | -1.0% | -0.6% | -1.1% |
| `qwen4` | +32.9% | +14.4% | +12.7% | +13.0% | +19.0% | +22.1% |

- **Exact-tie process efficiency (`n = 1522`):** faster convergence `73.2%` (`413/564`), shorter partner `42.9%`, cheaper episode cost `44.0%`.

## 5. Triage Paralysis, Score Upsets, Calendar 2-Cell Enforcement, & Neon Glitch

- **Triage `< 0.99` episodes (main arm):** `22` total; `astra60 × astra62` `0.00` `6/6 WAIT` deadlocks = `3`; `s304` `0.35` failures = `8`.
- **Score upsets (`chosen < other - 0.01`):** `292` total (`{'triage': 14, 'calendar': 184, 'goalpost': 24, 'code_handoff': 70}`); Opus `triage__s304` `0.35`-over-`1.00` receipts = `4`.
- **Calendar 2-cell enforcement:** `160` Calendar choice reasons cite the 2-cell rule (`opus55`: `87`, including `37` Opus score upsets).
- **Neon multi-ACTION self-play glitch:** `55` turns (`100%` unique to `neon4`, `s303`/`s304`: `31` turns); Opus win rate for Neon drops from `0.5903` (clean) to `0.0476` (glitched).

## 6. Disclosure Meta-Analysis (`common_ground.disclosure`)

### 6.1 Goalpost R1 Disclosure Table (Primary: `main` arm `160` eps / `320` R1 turns; sensitivity incl. `opus_boost`: `176` eps)

| Model | R1 Mean Rules | R1 >=5 All | R1 >=5 Role A | R1 >=5 Role B | Legacy ID >=5 | R1–R2 >=5 | R1–R8 >=5 | Role-A Ask-Without-Giving |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `astra62` | 2.82 | 39.3% | 25.0% | 53.6% | 1.8% | 44.6% | 44.6% | 42.9% |
| `neon4` | 6.82 | 96.4% | 100.0% | 92.9% | 96.4% | 96.4% | 100.0% | 0.0% |
| `opus55` | 5.12 | 67.9% | 75.0% | 60.7% | 19.6% | 82.1% | 83.9% | 14.3% |
| `astra60` | 6.66 | 93.8% | 100.0% | 87.5% | 9.4% | 93.8% | 93.8% | 0.0% |
| `kimi31` | 1.64 | 21.4% | 0.0% | 42.9% | 10.7% | 35.7% | 39.3% | 3.6% |
| `muse` | 6.75 | 96.9% | 100.0% | 93.8% | 90.6% | 96.9% | 100.0% | 0.0% |
| `qwen4` | 5.34 | 65.6% | 56.2% | 75.0% | 65.6% | 71.9% | 78.1% | 6.2% |

### 6.2 Within-GPT-6.2 Eval Blindness & Contagious Withholding

- **Within-GPT-6.2 (all main Goalpost runs):** Shared (`n=22`, score `0.9951`, ep words `49.8`) vs. Withheld (`n=34`, score `0.9777`, ep words `45.2`): CV choice win rate `43.3%` vs. `23.8%`; exact ties `39.1%` vs. `6.2%`.
- **Within-GPT-6.2 (`score == 1.0` only):** Shared (`n=21`, F1 `6.67`, F4 KEEP `100.0%`) vs. Withheld (`n=28`, F1 `6.44`, F4 KEEP `100.0%`): CV choice overall `48.1%` vs. `16.7%` (Fisher p=< 0.001, ep-perm p=0.0378); exact ties `39.1%` vs. `2.3%` (Fisher p=< 0.001, ep-perm p=< 0.001).
- **Contagious withholding (pooled Role B, `n = 160`):** mean R1 formulas after Role A disclosed = `6.13` vs. after Role A withheld = `3.56` (MWU p=< 0.001, perm p=< 0.001).
- **Role-B `opus55` after transparent (`neon4/muse/astra60/opus55`, `n=16`) vs. withholding (`astra62/kimi31`, `n=8`) Role-A partners:** `6.75` (93.8% full) vs. `2.38` (0.0% full), MWU p=< 0.001, perm p=< 0.001.

### 6.3 Producer–Consumer Hypocrisy & 14-Rule Exact-Tie Robustness

| Chooser | Producer Role-A R1 >=5 | Consumer Hires Discloser (Exact Ties) | Gap (pp) | Binom p | Case-Perm p |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `astra62` | 25.0% | 81.2% (`52/64`) | +56.2 | < 0.001 | 0.0015 |
| `opus55` | 75.0% | 88.2% (`60/68`) | +13.2 | < 0.001 | < 0.001 |
| `neon4` | 100.0% | 100.0% (`34/34`) | +0.0 | < 0.001 | < 0.001 |
| `kimi31` | 0.0% | 75.0% (`21/28`) | +75.0 | 0.0125 | 0.0365 |

- **14-rule robustness (`C-A1..C-A7`, `C-B1..C-B7`):** `14/14` rules favor the discloser on exact ties (win-rate range `70.4%`–`93.6%`).

### 6.4 Triage Role-B Disclosure & Exp-2 Pool Inventory

- **Triage Role-B Exp-2 zero-contradiction pool (`main` arm):** `142/160` zero-contradiction overall; `126` with `score == 1.0` (`{'301': 35, '302': 36, '303': 30, '304': 25}`); **intersected withheld pool (`<= 3 CMDs & score == 1.0 & zero_contradiction`): `89`** (`{'301': 26, '302': 28, '303': 20, '304': 15}`, strict `0 CMDs`: `68`, `ready_for_exp2_24_pairs=True`).

| Role-B Operator | Mean R1 CMDs | <=3 CMDs Rate | Mean R1 Guardrail IDs | Zero-Contradiction All | Zero-Contradiction `score==1.0` (All) | Withheld Exp-2 Pool (`<=3 CMDs & score==1.0 & ZC`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `astra62` | 0.79 | 92.9% | 0.00 | `21/28` | `17` (`{'301': 6, '302': 6, '303': 3, '304': 2}`) | `16` (`{'301': 6, '302': 6, '303': 2, '304': 2}`) |
| `neon4` | 2.68 | 64.3% | 0.64 | `26/28` | `25` (`{'301': 6, '302': 7, '303': 6, '304': 6}`) | `15` (`{'301': 3, '302': 5, '303': 4, '304': 3}`) |
| `opus55` | 3.75 | 46.4% | 0.64 | `28/28` | `25` (`{'301': 7, '302': 7, '303': 7, '304': 4}`) | `11` (`{'301': 5, '302': 2, '303': 3, '304': 1}`) |
| `astra60` | 2.25 | 75.0% | 0.00 | `14/16` | `8` (`{'301': 4, '302': 2, '303': 2}`) | `5` (`{'301': 3, '302': 2}`) |
| `kimi31` | 1.29 | 89.3% | 0.00 | `24/28` | `24` (`{'301': 5, '302': 6, '303': 6, '304': 7}`) | `21` (`{'301': 4, '302': 6, '303': 6, '304': 5}`) |
| `muse` | 0.38 | 100.0% | 0.00 | `14/16` | `13` (`{'301': 3, '302': 4, '303': 3, '304': 3}`) | `13` (`{'301': 3, '302': 4, '303': 3, '304': 3}`) |
| `qwen4` | 2.81 | 62.5% | 0.19 | `15/16` | `14` (`{'301': 4, '302': 4, '303': 3, '304': 3}`) | `8` (`{'301': 2, '302': 3, '303': 2, '304': 1}`) |

