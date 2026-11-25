"""Empirical budget projector and block-clustered Monte Carlo power simulator."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from typing import Any

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

logger = logging.getLogger(__name__)

from common_ground.config import CORE_MODELS, DATA_DIR, PREDECESSOR_MODEL, load_verified_models
from common_ground.ledger import Ledger


def simulate_block_clustered_choice(
    *,
    rng: np.random.Generator,
    n_seeds: int,
    beta_astra: float,
    both_roles: bool = True,
    n_tasks: int = 3,
    gamma: float = 0.5,
    delta: float = 0.8,
    eta: float = 0.4,
    sigma_ep: float = 0.7,
    sigma_item: float = 0.6,
    astra_score_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate forced choices WITH episode-level random effects & correlated scores."""
    beta = np.array([beta_astra, 0.0, 0.0, 0.0])
    pairs = list(itertools.combinations(range(4), 2))
    roles = (0, 1) if both_roles else (0,)
    seed_pairs = (
        [(0, 1)]
        if n_seeds == 2
        else [(i, (i + 1) % n_seeds) for i in range(n_seeds)]
    )

    rows, y, blocks = [], [], []
    block_id = 0
    for s in range(4):
        for t in range(n_tasks):
            for r in roles:
                # Episode-level random effects and scores shared across all pairs reusing an episode
                ep_u = rng.normal(0, sigma_ep, size=(4, n_seeds))
                ep_score = rng.normal(0, 1.0, size=(4, n_seeds))
                ep_score[0, :] += astra_score_shift
                for a, b in pairs:
                    for sa, sb in seed_pairs:
                        u_item = rng.normal(0, sigma_item)
                        dscore = ep_score[a, sa] - ep_score[b, sb]
                        ep_diff = ep_u[a, sa] - ep_u[b, sb]
                        for order in (+1, -1):
                            lin = (
                                beta[a]
                                - beta[b]
                                + gamma * ((a == s) - (b == s))
                                + delta * dscore
                                + eta * order
                                + ep_diff
                                + u_item
                            )
                            p = 1.0 / (1.0 + math.exp(-lin))
                            x = np.zeros(7)
                            x[a] += 1.0
                            x[b] -= 1.0
                            x[4] = (a == s) - (b == s)
                            x[5] = dscore
                            x[6] = order
                            rows.append(x)
                            y.append(float(rng.random() < p))
                            blocks.append(block_id)
                block_id += 1

    X = np.array(rows)[:, [0, 1, 2, 4, 5, 6]]  # col 3 (kimi31) is reference
    return X, np.array(y, float), np.array(blocks)


def estimate_power_h1(
    *,
    n_seeds: int = 3,
    beta_astra: float = -0.7,
    both_roles: bool = True,
    reps: int = 250,
    seed: int = 20261121,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    c = np.array([1.0, -1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0])
    hits = 0
    ok = 0
    for _ in range(reps):
        X, y, blocks = simulate_block_clustered_choice(
            rng=rng, n_seeds=n_seeds, beta_astra=beta_astra, both_roles=both_roles
        )
        try:
            fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": blocks}
            )
            est = float(c @ fit.params)
            se = math.sqrt(float(c @ fit.cov_params() @ c))
            if se > 0:
                ok += 1
                hits += int(abs(est / se) > 1.96)
        except (ValueError, np.linalg.LinAlgError, PerfectSeparationError, FloatingPointError) as exc:
            logger.debug("Skipping non-convergent simulation draw: %s", exc)
            continue
    return {
        "n_seeds": n_seeds,
        "both_roles": float(both_roles),
        "beta_astra": beta_astra,
        "p_choose_approx": round(1.0 / (1.0 + math.exp(-beta_astra)), 3),
        "power": round(hits / max(1, ok), 3),
        "reps_ok": ok,
    }


def project_budget_from_pilot(pilot_run: str = "pilot") -> dict[str, Any]:
    """Compute exact empirical cost per episode/fork/choice from pilot ledger and project main run."""
    ledger_path = DATA_DIR / pilot_run / "ledger.jsonl"
    if not ledger_path.exists():
        return {"error": f"Pilot ledger not found at {ledger_path}"}

    ledger = Ledger(ledger_path, budget_cap_usd=1e9)
    stage_costs: dict[tuple[str, str], list[float]] = {}
    stage_reason: dict[tuple[str, str], list[int]] = {}
    for e in ledger.all_entries:
        if e.status != "ok":
            continue
        stage = str(e.metadata.get("stage", "unknown"))
        key = (e.model_key, stage)
        stage_costs.setdefault(key, []).append(e.cost_usd)
        if e.reasoning_tokens is not None:
            stage_reason.setdefault(key, []).append(e.reasoning_tokens)

    def _mean(m: str, st: str, default: float = 0.0) -> float:
        vals = stage_costs.get((m, st))
        return float(np.mean(vals)) if vals else default

    # Empirical cost of one agent-episode (10 turns) + forks F1/F2/F3a/F3b/F4/F5
    turn_cost = {m: _mean(m, "turn") * 10.0 for m in (*CORE_MODELS, PREDECESSOR_MODEL)}
    fork_cost = {
        m: sum(_mean(m, f) for f in ("F1", "F2", "F3a", "F3b", "F4", "F5"))
        for m in CORE_MODELS
    }
    choice_cost = {m: _mean(m, "choice") for m in CORE_MODELS}
    rw_per_msg = float(
        np.mean(
            stage_costs.get(("rewriter", "rewrite_F3a"), [0.0005])
            + stage_costs.get(("rewriter", "rewrite_F3b"), [0.0005])
        )
    )

    projections: dict[str, Any] = {}
    for n_seeds in (1, 2, 3):
        for pred_both in (False, True):
            # Core episodes: 16 ordered pairs x 3 tasks x n_seeds -> each model plays 8*3*n_seeds agent-eps
            core_ae = 8 * 3 * n_seeds
            c_core_turns = sum(core_ae * turn_cost[m] for m in CORE_MODELS)

            # Predecessor episodes: (4 or 8) x 3 tasks x n_seeds
            pred_mult = 2 if pred_both else 1
            pred_eps = 4 * pred_mult * 3 * n_seeds
            c_pred_turns = pred_eps * turn_cost[PREDECESSOR_MODEL] + sum(
                (pred_mult * 3 * n_seeds) * turn_cost[m] for m in CORE_MODELS
            )

            # Forks: every core agent-ep + subject side of predecessor eps
            c_forks = sum(
                (core_ae + pred_mult * 3 * n_seeds) * fork_cost[m] for m in CORE_MODELS
            )

            # Rewriter: ~19 partner msgs per core ep * 2 modes + ~9.5 per pred ep * 2 modes
            n_rw_msgs = (16 * 3 * n_seeds * 19 + pred_eps * 9.5) * 2
            c_rw = n_rw_msgs * rw_per_msg

            # Choices (only exist for n_seeds >= 2)
            c_choices = 0.0
            if n_seeds >= 2:
                seed_pairs = 1 if n_seeds == 2 else n_seeds
                # Both roles: 6 core pairs x 3 tasks x seed_pairs x 2 orders x 2 roles
                n_core_ch = 6 * 3 * seed_pairs * 2 * 2
                # A/A: 4 partners x 3 tasks x 2 orders x 2 roles
                n_aa_ch = 4 * 3 * 2 * 2
                # H2H: pred_mult * 3 tasks x n_seeds x 2 orders
                n_h2h_ch = pred_mult * 3 * n_seeds * 2
                c_choices = sum(
                    (n_core_ch + n_aa_ch + n_h2h_ch) * choice_cost[m] for m in CORE_MODELS
                )

            subtotal = c_core_turns + c_pred_turns + c_forks + c_rw + c_choices
            label = f"seeds={n_seeds}_predBothRoles={pred_both}"
            projections[label] = {
                "episodes_total": 16 * 3 * n_seeds + pred_eps,
                "core_turns_usd": round(c_core_turns, 2),
                "pred_turns_usd": round(c_pred_turns, 2),
                "forks_usd": round(c_forks, 2),
                "rewriter_usd": round(c_rw, 2),
                "choices_usd": round(c_choices, 2),
                "subtotal_usd": round(subtotal, 2),
                "with_15pct_slack_usd": round(subtotal * 1.15, 2),
            }

    median_reasoning = {
        f"{m}:{st}": int(np.median(vals))
        for (m, st), vals in sorted(stage_reason.items())
        if vals
    }
    return {
        "pilot_run": pilot_run,
        "pilot_actual_cost_usd": round(ledger.total_cost_usd, 4),
        "empirical_turn_cost_per_10r_episode_usd": {k: round(v, 4) for k, v in turn_cost.items()},
        "empirical_all_6_forks_per_ae_usd": {k: round(v, 4) for k, v in fork_cost.items()},
        "empirical_choice_cost_usd": {k: round(v, 4) for k, v in choice_cost.items()},
        "median_reasoning_tokens_by_model_stage": median_reasoning,
        "main_run_projections": projections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Power simulation and pilot-calibrated budget projection.")
    parser.add_argument("--pilot-run", type=str, default="pilot")
    parser.add_argument("--skip-power", action="store_true")
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    out: dict[str, Any] = {}
    if not args.skip_power:
        out["power_h1_block_clustered"] = [
            estimate_power_h1(n_seeds=3, beta_astra=b, both_roles=r, reps=args.reps)
            for r in (False, True)
            for b in (0.0, -0.4, -0.7, -1.0)
        ]
    out["budget_projection"] = project_budget_from_pilot(args.pilot_run)
    print(json.dumps(out, indent=2))
    _ = load_verified_models  # keep import clean


if __name__ == "__main__":
    main()
