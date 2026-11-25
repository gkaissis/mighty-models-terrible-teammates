# Experiment 2: The Information Flip — Summary Report

- **Schema:** `exp2_info_flip_summary_v1` | **Mock:** `False`
- **Choices:** `384/384` main + `96/96` placebo (total cost: `$15.4613`)

## Side-by-Side Causal Comparison: Tone+Readability vs. Information Flip

| Experiment / Arm | Scope | Cases (Pairs) | Mean Δ | Pair-Cluster 95% CI | Sign-Flip p | Unanimous Flips Toward | Unanimous Flips Away | Held | Split |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Capstone Tone+Readability Flip | ITT Cross-Vendor | 61 (24) | -0.0082 | [-0.0364, +0.0169] | 1.0000 | 0/61 (0.0%) | 0/61 (0.0%) | 91.8% | 8.2% |
| Capstone Tone+Readability Flip | ITT Pooled (4 labs) | 96 (24) | -0.0156 | [-0.0417, +0.0104] | 0.5000 | 0/96 (0.0%) | 0/96 (0.0%) | 92.7% | 7.3% |
| **Exp 2 Information Flip (Main)** | **ITT Cross-Vendor** | 57 (24) | +0.2456 | [+0.1053, +0.3818] | 0.0023 | 11/57 (19.3%) | 1/57 (1.8%) | 54.4% | 24.6% |
| **Exp 2 Information Flip (Main)** | **ITT Pooled (4 labs)** | 96 (24) | +0.2292 | [+0.1146, +0.3335] | 0.0011 | 17/96 (17.7%) | 1/96 (1.0%) | 53.1% | 28.1% |
| Exp 2 Triage Full-Transcript | ITT Cross-Vendor | 30 (12) | +0.1167 | [+0.0333, +0.2167] | 0.0625 | 1/30 (3.3%) | 0/30 (0.0%) | 73.3% | 23.3% |
| Exp 2 Triage Full-Transcript | ITT Pooled (4 labs) | 48 (12) | +0.1458 | [+0.0208, +0.2812] | 0.0938 | 5/48 (10.4%) | 0/48 (0.0%) | 68.8% | 20.8% |
| Exp 2 Goalpost R1-Handoff | ITT Cross-Vendor | 27 (12) | +0.3889 | [+0.1346, +0.6154] | 0.0186 | 10/27 (37.0%) | 1/27 (3.7%) | 33.3% | 25.9% |
| Exp 2 Goalpost R1-Handoff | ITT Pooled (4 labs) | 48 (12) | +0.3125 | [+0.1354, +0.4583] | 0.0122 | 12/48 (25.0%) | 1/48 (2.1%) | 37.5% | 35.4% |
| Exp 2 Public-Echo Placebo | ITT Cross-Vendor | 14 (6) | +0.0357 | [+0.0000, +0.1000] | 1.0000 | 0/14 (0.0%) | 0/14 (0.0%) | 92.9% | 7.1% |
| Exp 2 Public-Echo Placebo | ITT Pooled (4 labs) | 24 (6) | +0.0000 | [-0.0625, +0.0625] | 1.0000 | 0/24 (0.0%) | 0/24 (0.0%) | 91.7% | 8.3% |

## Per-Reviewer Breakdown (Main Arm — ITT Pooled)

| Reviewer | Cases (Pairs) | Mean ΔInfo | 95% CI | Sign-Flip p | Unanimous Toward Info | Away | Held | Split |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `opus55` | 24 (24) | +0.1458 | [+0.0000, +0.2917] | 0.0918 | 1/24 (4.2%) | 0/24 (0.0%) | 54.2% | 41.7% |
| `neon4` | 24 (24) | +0.5000 | [+0.2708, +0.7083] | 0.0010 | 12/24 (50.0%) | 1/24 (4.2%) | 37.5% | 8.3% |
| `astra62` | 24 (24) | +0.1667 | [+0.0000, +0.3333] | 0.1172 | 3/24 (12.5%) | 0/24 (0.0%) | 54.2% | 33.3% |
| `kimi31` | 24 (24) | +0.1042 | [-0.0208, +0.2500] | 0.2344 | 1/24 (4.2%) | 0/24 (0.0%) | 66.7% | 29.2% |

## 4-Lab Third-Person Consensus (Main 24 Pairs)

- **4/4 Unanimous Lab Flip Toward Private Info:** `0/24` pairs
- **>=3/4 Majority Flip Toward Private Info:** `0/24` pairs
- **>=3/4 Majority Positive ΔInfo:** `4/24` pairs
- **Mean Pairwise Inter-Lab Agreement (Flip):** `0.7014`
