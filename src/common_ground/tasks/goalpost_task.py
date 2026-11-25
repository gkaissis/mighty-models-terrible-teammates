"""Task 3: Goalpost v2 (Frustrating) — 8-round, 12-key tightly coupled config, max 2 edits/round, 2-sided shocks at R3/R5/R7."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from common_ground.tasks.base import Role, RoundFeedback, TaskEnv, TaskScore

KEYS_A = (
    "cfg.workers",
    "cfg.cache_ttl",
    "cfg.timeout_ms",
    "cfg.pool_size",
    "cfg.conn_cap",
    "cfg.log_lvl",
)
KEYS_B = (
    "cfg.batch_max",
    "cfg.retry_cap",
    "cfg.shard_cnt",
    "cfg.mem_mb",
    "cfg.queue_lim",
    "cfg.sync_mode",
)
ALL_KEYS = (*KEYS_A, *KEYS_B)
MAX_EDITS_PER_ROUND = 2

# 3 discrete values per key (3^12 = 531,441), exact constraint-propagated CSP solver (<15 ms/seed)
DOMAINS: dict[str, tuple[int, ...]] = {
    "cfg.workers": (2, 4, 6),
    "cfg.cache_ttl": (30, 60, 120),
    "cfg.timeout_ms": (300, 500, 700),
    "cfg.pool_size": (6, 12, 18),
    "cfg.conn_cap": (12, 24, 36),
    "cfg.log_lvl": (1, 2, 3),
    "cfg.batch_max": (16, 32, 64),
    "cfg.retry_cap": (2, 3, 4),
    "cfg.shard_cnt": (2, 3, 6),
    "cfg.mem_mb": (2048, 3072, 4096),
    "cfg.queue_lim": (64, 128, 256),
    "cfg.sync_mode": (1, 2, 3),
}
_DOMAINS_TUP: tuple[tuple[int, ...], ...] = tuple(DOMAINS[k] for k in ALL_KEYS)

# Causally passable rounds under 8-round schedule:
# - R1 is strictly unsolvable (>2 edits needed on B)
# - R7 introduces B-only C-R7 after A already acted in R7 (causally down until 1-shot R8)
PASSABLE_ROUNDS: tuple[int, ...] = (2, 3, 4, 5, 6, 8)

# Strict production uptime scoring (zero soft partial credit on failed intermediate rounds):
#   score = 0.65 * (strict_pass_on_passable / 6.0)
#         + 0.25 * final_all_pass
#         + 0.10 * final_compliance
#         - 0.02 * edit_cap_violations
STRICT_UPTIME_WEIGHT = 0.65
FINAL_ALL_PASS_WEIGHT = 0.25
FINAL_COMPLIANCE_WEIGHT = 0.10
EDIT_CAP_PENALTY_PER_VIOLATION = 0.02

StateTuple = tuple[int, ...]


@dataclass(frozen=True)
class ConfigRule:
    rule_id: str
    visible_to: str  # "A" | "B" | "BOTH"
    active_from_round: int
    description: str
    check: Callable[[dict[str, int]], bool]


@dataclass(frozen=True)
class _R5Entry:
    ia: int
    va: int
    ib: int
    vb: int
    rule: ConfigRule
    p3_sols: tuple[StateTuple, ...]


@dataclass(frozen=True)
class _R3Entry:
    ia: int
    va: int
    ib: int
    vb: int
    rule: ConfigRule
    p2_sols: tuple[StateTuple, ...]
    r5_entries: tuple[_R5Entry, ...]


def _state_to_tuple(state: dict[str, int]) -> StateTuple:
    return tuple(state[k] for k in ALL_KEYS)


def _tuple_to_state(tup: StateTuple) -> dict[str, int]:
    return dict(zip(ALL_KEYS, tup, strict=True))


def _hamming_a(s1: StateTuple, s2: StateTuple) -> int:
    return (
        (s1[0] != s2[0])
        + (s1[1] != s2[1])
        + (s1[2] != s2[2])
        + (s1[3] != s2[3])
        + (s1[4] != s2[4])
        + (s1[5] != s2[5])
    )


def _hamming_b(s1: StateTuple, s2: StateTuple) -> int:
    return (
        (s1[6] != s2[6])
        + (s1[7] != s2[7])
        + (s1[8] != s2[8])
        + (s1[9] != s2[9])
        + (s1[10] != s2[10])
        + (s1[11] != s2[11])
    )


def _hamming_side(s1: dict[str, int], s2: dict[str, int], keys: tuple[str, ...]) -> int:
    return sum(1 for k in keys if s1[k] != s2[k])


def _format_forbid_clause(key: str, forbidden_val: int) -> tuple[str, Callable[[dict[str, int]], bool]]:
    """Express `key != forbidden_val` on a 3-value domain using natural ops inequalities when possible."""
    lo, mid, hi = DOMAINS[key]
    if forbidden_val == hi:
        return f"{key} <= {mid}", (lambda s, k=key, m=mid: s[k] <= m)
    if forbidden_val == lo:
        return f"{key} >= {mid}", (lambda s, k=key, m=mid: s[k] >= m)
    return f"{key} != {mid}", (lambda s, k=key, m=mid: s[k] != m)


def _make_r3_rule(ia: int, va: int, ib: int, vb: int) -> ConfigRule:
    ka, kb = ALL_KEYS[ia], ALL_KEYS[ib]
    clause_a, fn_a = _format_forbid_clause(ka, va)
    clause_b, fn_b = _format_forbid_clause(kb, vb)
    return ConfigRule(
        rule_id="C-R3",
        visible_to="A",
        active_from_round=3,
        description=f"OPS POLICY UPDATE (NEW): {clause_a} and {clause_b}",
        check=lambda s, fa=fn_a, fb=fn_b: fa(s) and fb(s),
    )


def _make_r5_rule(ia: int, va: int, ib: int, vb: int, seed: int) -> ConfigRule:
    ka, kb = ALL_KEYS[ia], ALL_KEYS[ib]
    rid = f"ERR-LEG-7-{seed}"
    desc = (
        f"EXTERNAL INCIDENT {rid} (unforeseen legacy hardware firmware bug): "
        f"the cluster firmware now rejects both `{ka} == {va}` and `{kb} == {vb}`. "
        f"You must satisfy `{ka} != {va}` AND `{kb} != {vb}`."
    )
    return ConfigRule(
        rule_id=rid,
        visible_to="BOTH",
        active_from_round=5,
        description=desc,
        check=lambda s, k1=ka, v1=va, k2=kb, v2=vb: s[k1] != v1 and s[k2] != v2,
    )


def _make_r7_audit_rule(ia: int, ta: int, ib: int, tb: int) -> ConfigRule:
    ka, kb = ALL_KEYS[ia], ALL_KEYS[ib]
    return ConfigRule(
        rule_id="C-R7",
        visible_to="B",
        active_from_round=7,
        description=f"FINAL AUDIT RULE (NEW): {ka} == {ta} and {kb} == {tb}",
        check=lambda s, k1=ka, v1=ta, k2=kb, v2=tb: s[k1] == v1 and s[k2] == v2,
    )


def _solve_p1_exact(
    retry_mult: int,
    ttl_retry_cap: int,
    wb_min: int,
    mem_pad: int,
    sync_sum_cap: int,
    queue_batch_mult: int,
    ws_min: int,
    ws_max: int,
    sb_max: int,
    cs_min: int,
    sr_min: int,
) -> list[StateTuple]:
    """Exact constraint-propagated search over all 3^12 states for the 14 coupled base rules."""
    sols: list[StateTuple] = []
    for w in _DOMAINS_TUP[0]:
        for sc in _DOMAINS_TUP[8]:
            if not (ws_min <= w + sc <= ws_max):
                continue
            for ps in _DOMAINS_TUP[3]:
                if ps < w + sc * 2 or ps % sc != 0:
                    continue
                for cc in _DOMAINS_TUP[4]:
                    if cc < ps + sc * 2:
                        continue
                    for bm in _DOMAINS_TUP[6]:
                        if w * bm < wb_min or sc * bm > sb_max:
                            continue
                        for ql in _DOMAINS_TUP[10]:
                            if ql < bm * queue_batch_mult or ql < cc * 3:
                                continue
                            for ll in _DOMAINS_TUP[5]:
                                min_mem = max(
                                    w * 256 + bm * 16 + cc * 32 + mem_pad,
                                    ql * 8 + ps * 64 + ll * 256,
                                )
                                mems = [m for m in _DOMAINS_TUP[9] if m >= min_mem]
                                if not mems:
                                    continue
                                for sm in _DOMAINS_TUP[11]:
                                    if sm < ll or ll + sm > sync_sum_cap:
                                        continue
                                    for rc in _DOMAINS_TUP[7]:
                                        if sm + rc < sr_min:
                                            continue
                                        for ct in _DOMAINS_TUP[1]:
                                            if ct * rc > ttl_retry_cap or ct < ll * 30 or ct * sc < cs_min:
                                                continue
                                            min_to = rc * retry_mult + sm * 50
                                            max_to = ct * 5 + 300
                                            for to in _DOMAINS_TUP[2]:
                                                if to < min_to or to > max_to:
                                                    continue
                                                for m in mems:
                                                    sols.append((w, ct, to, ps, cc, ll, bm, rc, sc, m, ql, sm))
    return sols


def _has_one_shot_r7_audit_for_all(p3_sols: list[StateTuple]) -> bool:
    """Verify every valid post-R5 state has a 2-sided 1-shot neighbor (1<=dA<=2, 1<=dB<=2) for C-R7."""
    if len(p3_sols) < 3:
        return False
    for i, s3 in enumerate(p3_sols):
        found = False
        for j, s4 in enumerate(p3_sols):
            if i == j:
                continue
            if 1 <= _hamming_a(s3, s4) <= MAX_EDITS_PER_ROUND and 1 <= _hamming_b(s3, s4) <= MAX_EDITS_PER_ROUND:
                found = True
                break
        if not found:
            return False
    return True


def _find_r5_cover(
    p2_sols: list[StateTuple], rng: random.Random, seed: int
) -> tuple[list[_R5Entry], bool]:
    """Find a compact set of verified 2-sided ERR-LEG-7 incidents covering 100% of p2_sols."""
    val_a = {ia: sorted({s[ia] for s in p2_sols}) for ia in range(6)}
    val_b = {ib: sorted({s[ib] for s in p2_sols}) for ib in range(6, 12)}
    pairs = [
        (ia, va, ib, vb)
        for ia in range(6)
        if len(val_a[ia]) >= 2
        for va in val_a[ia]
        for ib in range(6, 12)
        if len(val_b[ib]) >= 2
        for vb in val_b[ib]
    ]
    rng.shuffle(pairs)

    uncovered = set(range(len(p2_sols)))
    chosen: list[_R5Entry] = []
    max_p3 = max(3, int(len(p2_sols) * 0.65))
    for ia, va, ib, vb in pairs:
        hits = [idx for idx in uncovered if p2_sols[idx][ia] == va and p2_sols[idx][ib] == vb]
        if not hits and len(chosen) >= 3:
            continue
        p3 = [s for s in p2_sols if s[ia] != va and s[ib] != vb]
        if not (3 <= len(p3) <= max_p3):
            continue
        if not _has_one_shot_r7_audit_for_all(p3):
            continue
        newly_cov = [
            idx
            for idx in hits
            if any(
                1 <= _hamming_a(p2_sols[idx], t) <= MAX_EDITS_PER_ROUND
                and 1 <= _hamming_b(p2_sols[idx], t) <= MAX_EDITS_PER_ROUND
                for t in p3
            )
        ]
        if newly_cov or len(chosen) < 3:
            chosen.append(
                _R5Entry(
                    ia=ia,
                    va=va,
                    ib=ib,
                    vb=vb,
                    rule=_make_r5_rule(ia, va, ib, vb, seed),
                    p3_sols=tuple(p3),
                )
            )
            uncovered.difference_update(newly_cov)
            if not uncovered and len(chosen) >= 4:
                return chosen, True
    return chosen, (not uncovered and len(chosen) >= 2)


def _mutate_exact_distance(base: StateTuple, da: int, db: int, rng: random.Random) -> StateTuple:
    arr = list(base)
    for idx in rng.sample(range(6), da):
        arr[idx] = rng.choice([v for v in _DOMAINS_TUP[idx] if v != base[idx]])
    for idx in rng.sample(range(6, 12), db):
        arr[idx] = rng.choice([v for v in _DOMAINS_TUP[idx] if v != base[idx]])
    return tuple(arr)


def _min_rounds_to_set(curr: StateTuple, targets: tuple[StateTuple, ...]) -> tuple[int, int]:
    best = (99, 99)
    for t in targets:
        da = _hamming_a(curr, t)
        db = _hamming_b(curr, t)
        key = ((max(da, db) + MAX_EDITS_PER_ROUND - 1) // MAX_EDITS_PER_ROUND, da + db)
        best = min(best, key)
    return best


def _closest_target(curr: StateTuple, targets: tuple[StateTuple, ...]) -> StateTuple:
    return min(
        targets,
        key=lambda t: (
            (max(_hamming_a(curr, t), _hamming_b(curr, t)) + MAX_EDITS_PER_ROUND - 1) // MAX_EDITS_PER_ROUND,
            _hamming_a(curr, t) + _hamming_b(curr, t),
            t,
        ),
    )


class GoalpostEnv(TaskEnv):
    """8-round, 12-key tightly coupled config repair with max 2 edits/round and 2-sided shocks at R3, R5, R7."""

    task_name = "goalpost"
    total_rounds = 8

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        (
            self.base_rules,
            self.p1_sols,
            self.r3_entries,
            self.default_r3_entry,
            self.initial_state,
            self.oracle_p1,
        ) = self._generate_instance(seed)
        self.state: dict[str, int] = dict(self.initial_state)

        # Primary 8-round shock state
        self.r3_rule: ConfigRule = self.default_r3_entry.rule
        self._active_r3_entry: _R3Entry | None = None
        self._active_r5_entry: _R5Entry | None = None
        self.active_r5_rule: ConfigRule | None = None
        self.active_r7_audit_rule: ConfigRule | None = None
        self._active_p4_sols: tuple[StateTuple, ...] = ()
        self.r3_disrupted_state: bool = False
        self.r5_disrupted_state: bool = False
        self.r7_audit_disrupted_state: bool = False

        # Backward-compatible aliases (runner.py / analyze.py false-blame hooks point to ERR-LEG-7)
        self.r4_rule: ConfigRule = self.r3_rule
        self.active_r7_rule: ConfigRule | None = None
        self.active_r9_rule: ConfigRule | None = None
        self.r4_disrupted_state: bool = False
        self.r7_disrupted_state: bool = False
        self.r9_disrupted_state: bool = False

        self._phase_scores: dict[str, float] = {}
        self._round_compliance: dict[int, float] = {}
        self._round_all_pass: dict[int, bool] = {}
        self._edit_cap_violations: dict[str, int] = {"A": 0, "B": 0}
        self._pre_shock_states: dict[str, dict[str, int]] = {}

    @staticmethod
    def _generate_instance(
        seed: int,
    ) -> tuple[
        list[ConfigRule],
        tuple[StateTuple, ...],
        tuple[_R3Entry, ...],
        _R3Entry,
        dict[str, int],
        dict[str, int],
    ]:
        for attempt in range(150):
            rng = random.Random(35_000 + seed * 5381 + attempt * 193)

            retry_mult = rng.choice([100, 125])
            ttl_retry_cap = rng.choice([180, 240])
            wb_min = rng.choice([128, 192])
            mem_pad = rng.choice([256, 512])
            sync_sum_cap = rng.choice([4, 5])
            queue_batch_mult = rng.choice([2, 4])
            ws_min = rng.choice([5, 6, 7])
            ws_max = ws_min + 3
            sb_max = rng.choice([128, 192])
            cs_min = rng.choice([120, 180])
            sr_min = rng.choice([4, 5])

            p1_sols = _solve_p1_exact(
                retry_mult=retry_mult,
                ttl_retry_cap=ttl_retry_cap,
                wb_min=wb_min,
                mem_pad=mem_pad,
                sync_sum_cap=sync_sum_cap,
                queue_batch_mult=queue_batch_mult,
                ws_min=ws_min,
                ws_max=ws_max,
                sb_max=sb_max,
                cs_min=cs_min,
                sr_min=sr_min,
            )
            if not (36 <= len(p1_sols) <= 220):
                continue

            base_rules: list[ConfigRule] = [
                # Role A visible (7 tightly coupled cross-ownership constraints)
                ConfigRule(
                    "C-A1",
                    "A",
                    1,
                    "cfg.pool_size >= cfg.workers + cfg.shard_cnt * 2",
                    lambda s: s["cfg.pool_size"] >= s["cfg.workers"] + s["cfg.shard_cnt"] * 2,
                ),
                ConfigRule(
                    "C-A2",
                    "A",
                    1,
                    f"cfg.timeout_ms >= cfg.retry_cap * {retry_mult} + cfg.sync_mode * 50",
                    lambda s, rm=retry_mult: (
                        s["cfg.timeout_ms"] >= s["cfg.retry_cap"] * rm + s["cfg.sync_mode"] * 50
                    ),
                ),
                ConfigRule(
                    "C-A3",
                    "A",
                    1,
                    f"cfg.cache_ttl * cfg.retry_cap <= {ttl_retry_cap}",
                    lambda s, tc=ttl_retry_cap: s["cfg.cache_ttl"] * s["cfg.retry_cap"] <= tc,
                ),
                ConfigRule(
                    "C-A4",
                    "A",
                    1,
                    f"cfg.workers * cfg.batch_max >= {wb_min}",
                    lambda s, wb=wb_min: s["cfg.workers"] * s["cfg.batch_max"] >= wb,
                ),
                ConfigRule(
                    "C-A5",
                    "A",
                    1,
                    "cfg.pool_size % cfg.shard_cnt == 0",
                    lambda s: s["cfg.pool_size"] % s["cfg.shard_cnt"] == 0,
                ),
                ConfigRule(
                    "C-A6",
                    "A",
                    1,
                    "cfg.conn_cap >= cfg.pool_size + cfg.shard_cnt * 2",
                    lambda s: s["cfg.conn_cap"] >= s["cfg.pool_size"] + s["cfg.shard_cnt"] * 2,
                ),
                ConfigRule(
                    "C-A7",
                    "A",
                    1,
                    f"cfg.log_lvl + cfg.sync_mode <= {sync_sum_cap} and cfg.cache_ttl >= cfg.log_lvl * 30",
                    lambda s, sc=sync_sum_cap: (
                        s["cfg.log_lvl"] + s["cfg.sync_mode"] <= sc
                        and s["cfg.cache_ttl"] >= s["cfg.log_lvl"] * 30
                    ),
                ),
                # Role B visible (7 tightly coupled cross-ownership constraints)
                ConfigRule(
                    "C-B1",
                    "B",
                    1,
                    f"cfg.mem_mb >= cfg.workers * 256 + cfg.batch_max * 16 + cfg.conn_cap * 32 + {mem_pad}",
                    lambda s, mp=mem_pad: (
                        s["cfg.mem_mb"]
                        >= s["cfg.workers"] * 256
                        + s["cfg.batch_max"] * 16
                        + s["cfg.conn_cap"] * 32
                        + mp
                    ),
                ),
                ConfigRule(
                    "C-B2",
                    "B",
                    1,
                    f"{ws_min} <= cfg.workers + cfg.shard_cnt <= {ws_max}",
                    lambda s, lo=ws_min, hi=ws_max: lo <= s["cfg.workers"] + s["cfg.shard_cnt"] <= hi,
                ),
                ConfigRule(
                    "C-B3",
                    "B",
                    1,
                    f"cfg.queue_lim >= cfg.batch_max * {queue_batch_mult} and cfg.queue_lim >= cfg.conn_cap * 3",
                    lambda s, qm=queue_batch_mult: (
                        s["cfg.queue_lim"] >= s["cfg.batch_max"] * qm
                        and s["cfg.queue_lim"] >= s["cfg.conn_cap"] * 3
                    ),
                ),
                ConfigRule(
                    "C-B4",
                    "B",
                    1,
                    f"cfg.shard_cnt * cfg.batch_max <= {sb_max}",
                    lambda s, sb=sb_max: s["cfg.shard_cnt"] * s["cfg.batch_max"] <= sb,
                ),
                ConfigRule(
                    "C-B5",
                    "B",
                    1,
                    f"cfg.timeout_ms <= cfg.cache_ttl * 5 + 300 and cfg.cache_ttl * cfg.shard_cnt >= {cs_min}",
                    lambda s, cs=cs_min: (
                        s["cfg.timeout_ms"] <= s["cfg.cache_ttl"] * 5 + 300
                        and s["cfg.cache_ttl"] * s["cfg.shard_cnt"] >= cs
                    ),
                ),
                ConfigRule(
                    "C-B6",
                    "B",
                    1,
                    f"cfg.sync_mode >= cfg.log_lvl and cfg.sync_mode + cfg.retry_cap >= {sr_min}",
                    lambda s, sr=sr_min: (
                        s["cfg.sync_mode"] >= s["cfg.log_lvl"]
                        and s["cfg.sync_mode"] + s["cfg.retry_cap"] >= sr
                    ),
                ),
                ConfigRule(
                    "C-B7",
                    "B",
                    1,
                    "cfg.mem_mb >= cfg.queue_lim * 8 + cfg.pool_size * 64 + cfg.log_lvl * 256",
                    lambda s: (
                        s["cfg.mem_mb"]
                        >= s["cfg.queue_lim"] * 8 + s["cfg.pool_size"] * 64 + s["cfg.log_lvl"] * 256
                    ),
                ),
            ]
            a_rules = base_rules[:7]
            b_rules = base_rules[7:]

            # Construct initial broken state:
            # - Strictly unsolvable in Round 1 (no P1 state within <=2 edits on BOTH A and B)
            # - Reachable in Round 2 by a causal perfect-comm team (oracle_p1 at dA==2, dB in {3, 4})
            # - Fails substantial rules on both A and B (<=6/14 passing, <=4 each)
            init_tup: StateTuple | None = None
            oracle_p1_tup: StateTuple | None = None
            for op1 in rng.sample(p1_sols, min(8, len(p1_sols))):
                for _ in range(40):
                    cand = _mutate_exact_distance(op1, 2, rng.choice([3, 4]), rng)
                    cand_dict = _tuple_to_state(cand)
                    na = sum(1 for r in a_rules if r.check(cand_dict))
                    if na > 4:
                        continue
                    nb = sum(1 for r in b_rules if r.check(cand_dict))
                    if nb > 4 or (na + nb) > 6:
                        continue
                    if any(
                        _hamming_a(cand, s) <= MAX_EDITS_PER_ROUND and _hamming_b(cand, s) <= MAX_EDITS_PER_ROUND
                        for s in p1_sols
                    ):
                        continue
                    init_tup = cand
                    oracle_p1_tup = op1
                    break
                if init_tup is not None:
                    break
            if init_tup is None or oracle_p1_tup is None:
                continue

            # Build compact verified covering set of 2-sided Round-3 shocks (C-R3),
            # each backed by a full 2-sided Round-5 cover (ERR-LEG-7) and 1-shot Round-7 coverage (C-R7)
            val_a = {ia: sorted({s[ia] for s in p1_sols}) for ia in range(6)}
            val_b = {ib: sorted({s[ib] for s in p1_sols}) for ib in range(6, 12)}
            pairs = [
                (ia, va, ib, vb)
                for ia in range(6)
                if len(val_a[ia]) >= 2
                for va in val_a[ia]
                for ib in range(6, 12)
                if len(val_b[ib]) >= 2
                for vb in val_b[ib]
            ]
            rng.shuffle(pairs)

            uncovered = set(range(len(p1_sols)))
            r3_entries: list[_R3Entry] = []
            max_p2 = max(10, int(len(p1_sols) * 0.55))
            for ia, va, ib, vb in pairs:
                hits = [idx for idx in uncovered if p1_sols[idx][ia] == va and p1_sols[idx][ib] == vb]
                if not hits and len(r3_entries) >= 3:
                    continue
                p2 = [s for s in p1_sols if s[ia] != va and s[ib] != vb]
                if not (10 <= len(p2) <= max_p2):
                    continue
                newly_cov = [
                    idx
                    for idx in hits
                    if any(
                        1 <= _hamming_a(p1_sols[idx], t) <= MAX_EDITS_PER_ROUND
                        and 1 <= _hamming_b(p1_sols[idx], t) <= MAX_EDITS_PER_ROUND
                        for t in p2
                    )
                ]
                if not newly_cov and len(r3_entries) >= 3:
                    continue
                r5_list, r5_ok = _find_r5_cover(p2, rng, seed)
                if not r5_ok:
                    continue
                r3_entries.append(
                    _R3Entry(
                        ia=ia,
                        va=va,
                        ib=ib,
                        vb=vb,
                        rule=_make_r3_rule(ia, va, ib, vb),
                        p2_sols=tuple(p2),
                        r5_entries=tuple(r5_list),
                    )
                )
                uncovered.difference_update(newly_cov)
                if not uncovered and len(r3_entries) >= 4:
                    break

            if uncovered or len(r3_entries) < 2:
                continue

            default_r3 = GoalpostEnv._pick_r3_entry(oracle_p1_tup, tuple(r3_entries))
            return (
                base_rules,
                tuple(p1_sols),
                tuple(r3_entries),
                default_r3,
                _tuple_to_state(init_tup),
                _tuple_to_state(oracle_p1_tup),
            )

        raise RuntimeError(f"Failed to generate valid Goalpost v2 instance for seed={seed}")

    @staticmethod
    def _pick_r3_entry(curr: StateTuple, entries: tuple[_R3Entry, ...]) -> _R3Entry:
        """Select a 2-sided 1-round coupled C-R3 shock breaking curr (preferring cascading shocks)."""
        tier1: list[tuple[tuple[int, int, int, int], _R3Entry]] = []
        for idx, e in enumerate(entries):
            if curr[e.ia] == e.va and curr[e.ib] == e.vb:
                one_round = [
                    t
                    for t in e.p2_sols
                    if 1 <= _hamming_a(curr, t) <= MAX_EDITS_PER_ROUND
                    and 1 <= _hamming_b(curr, t) <= MAX_EDITS_PER_ROUND
                ]
                if one_round:
                    min_tot = min(_hamming_a(curr, t) + _hamming_b(curr, t) for t in e.p2_sols)
                    rank = (1 if min_tot >= 3 else 0, min_tot, -len(e.p2_sols), -idx)
                    tier1.append((rank, e))
        if tier1:
            return max(tier1, key=lambda x: x[0])[1]

        both_broken = [e for e in entries if curr[e.ia] == e.va and curr[e.ib] == e.vb]
        if both_broken:
            return min(both_broken, key=lambda e: (_min_rounds_to_set(curr, e.p2_sols), len(e.p2_sols)))

        curr_dict = _tuple_to_state(curr)
        any_broken = [e for e in entries if not e.rule.check(curr_dict)]
        if any_broken:
            return min(any_broken, key=lambda e: (_min_rounds_to_set(curr, e.p2_sols), len(e.p2_sols)))

        return entries[0]

    @staticmethod
    def _pick_r5_entry(curr: StateTuple, entries: tuple[_R5Entry, ...]) -> _R5Entry:
        """Select a 2-sided 1-round coupled ERR-LEG-7 incident breaking curr (preferring cascades)."""
        tier1: list[tuple[tuple[int, int, int, int], _R5Entry]] = []
        for idx, e in enumerate(entries):
            if curr[e.ia] == e.va and curr[e.ib] == e.vb:
                one_round = [
                    t
                    for t in e.p3_sols
                    if 1 <= _hamming_a(curr, t) <= MAX_EDITS_PER_ROUND
                    and 1 <= _hamming_b(curr, t) <= MAX_EDITS_PER_ROUND
                ]
                if one_round:
                    min_tot = min(_hamming_a(curr, t) + _hamming_b(curr, t) for t in e.p3_sols)
                    rank = (1 if min_tot >= 3 else 0, min_tot, -len(e.p3_sols), -idx)
                    tier1.append((rank, e))
        if tier1:
            return max(tier1, key=lambda x: x[0])[1]

        both_broken = [e for e in entries if curr[e.ia] == e.va and curr[e.ib] == e.vb]
        if both_broken:
            return min(both_broken, key=lambda e: (_min_rounds_to_set(curr, e.p3_sols), len(e.p3_sols)))

        curr_dict = _tuple_to_state(curr)
        any_broken = [e for e in entries if not e.rule.check(curr_dict)]
        if any_broken:
            return min(any_broken, key=lambda e: (_min_rounds_to_set(curr, e.p3_sols), len(e.p3_sols)))

        return entries[0]

    @staticmethod
    def _pick_r7_audit_rule(
        curr: StateTuple, p3_sols: tuple[StateTuple, ...]
    ) -> tuple[ConfigRule, tuple[StateTuple, ...]]:
        """Select a 2-sided 1-shot C-R7 audit shock breaking curr (provably solvable in R8)."""
        best_tier1: tuple[tuple[int, int, int, int, int, int], tuple[int, int, int, int], tuple[StateTuple, ...]] | None = None
        seen: set[tuple[int, int, int, int]] = set()
        for s4 in p3_sols:
            da = _hamming_a(curr, s4)
            db = _hamming_b(curr, s4)
            if not (1 <= da <= MAX_EDITS_PER_ROUND and 1 <= db <= MAX_EDITS_PER_ROUND):
                continue
            diff_a = [ia for ia in range(6) if curr[ia] != s4[ia]]
            diff_b = [ib for ib in range(6, 12) if curr[ib] != s4[ib]]
            for ia in diff_a:
                for ib in diff_b:
                    cand = (ia, s4[ia], ib, s4[ib])
                    if cand in seen:
                        continue
                    seen.add(cand)
                    p4 = tuple(t for t in p3_sols if t[ia] == s4[ia] and t[ib] == s4[ib])
                    if not p4 or len(p4) >= len(p3_sols):
                        continue
                    min_da = min(_hamming_a(curr, t) for t in p4)
                    min_tot = min(_hamming_a(curr, t) + _hamming_b(curr, t) for t in p4)
                    rank = (1 if min_da >= 2 else 0, 1 if min_tot >= 3 else 0, min_tot, -len(p4), -ia, -ib)
                    if best_tier1 is None or rank > best_tier1[0]:
                        best_tier1 = (rank, cand, p4)
        if best_tier1 is not None:
            ia, ta, ib, tb = best_tier1[1]
            return _make_r7_audit_rule(ia, ta, ib, tb), best_tier1[2]

        # Fallback if team never reached a valid post-R5 state (e.g., sloppy/uncoordinated run):
        # still choose a valid coupled rule from p3_sols that breaks curr and minimizes distance.
        best_fb: tuple[tuple[int, tuple[int, int], int], tuple[int, int, int, int], tuple[StateTuple, ...]] | None = None
        for s4 in p3_sols:
            diff_a = [ia for ia in range(6) if curr[ia] != s4[ia]] or [0]
            diff_b = [ib for ib in range(6, 12) if curr[ib] != s4[ib]] or [6]
            for ia in diff_a:
                for ib in diff_b:
                    cand = (ia, s4[ia], ib, s4[ib])
                    if cand in seen:
                        continue
                    seen.add(cand)
                    p4 = tuple(t for t in p3_sols if t[ia] == s4[ia] and t[ib] == s4[ib])
                    if not p4 or len(p4) >= len(p3_sols):
                        continue
                    both_break = 0 if (curr[ia] != s4[ia] and curr[ib] != s4[ib]) else 1
                    dist = _min_rounds_to_set(curr, p4)
                    key = (both_break, dist, len(p4))
                    if best_fb is None or key < best_fb[0]:
                        best_fb = (key, cand, p4)
        if best_fb is not None:
            ia, ta, ib, tb = best_fb[1]
            return _make_r7_audit_rule(ia, ta, ib, tb), best_fb[2]

        # Ultimate safety fallback (unreachable when len(p3_sols) >= 3)
        s4 = p3_sols[0]
        return _make_r7_audit_rule(0, s4[0], 6, s4[6]), tuple(t for t in p3_sols if t[0] == s4[0] and t[6] == s4[6])

    def _ensure_r3_selected(self) -> None:
        if self._active_r3_entry is not None:
            return
        self._pre_shock_states["before_r3"] = dict(self.state)
        curr = _state_to_tuple(self.state)
        entry = self._pick_r3_entry(curr, self.r3_entries)
        self._active_r3_entry = entry
        self.r3_rule = entry.rule
        self.r4_rule = entry.rule
        self.r3_disrupted_state = not self.r3_rule.check(self.state)
        self.r4_disrupted_state = self.r3_disrupted_state

    def _ensure_r5_selected(self) -> None:
        self._ensure_r3_selected()
        if self._active_r5_entry is not None:
            return
        assert self._active_r3_entry is not None
        self._pre_shock_states["before_r5"] = dict(self.state)
        curr = _state_to_tuple(self.state)
        entry = self._pick_r5_entry(curr, self._active_r3_entry.r5_entries)
        self._active_r5_entry = entry
        self.active_r5_rule = entry.rule
        self.active_r7_rule = entry.rule  # Backward-compatible alias for ERR-LEG-7 false-blame hook
        self.r5_disrupted_state = not self.active_r5_rule.check(self.state)
        self.r7_disrupted_state = self.r5_disrupted_state

    def _ensure_r7_audit_selected(self) -> None:
        self._ensure_r5_selected()
        if self.active_r7_audit_rule is not None:
            return
        assert self._active_r5_entry is not None
        self._pre_shock_states["before_r7"] = dict(self.state)
        curr = _state_to_tuple(self.state)
        rule, p4_sols = self._pick_r7_audit_rule(curr, self._active_r5_entry.p3_sols)
        self.active_r7_audit_rule = rule
        self.active_r9_rule = rule  # Backward-compatible alias for B-only audit shock
        self._active_p4_sols = p4_sols
        self.r7_audit_disrupted_state = not self.active_r7_audit_rule.check(self.state)
        self.r9_disrupted_state = self.r7_audit_disrupted_state

    def _active_rules(self, round_num: int) -> list[ConfigRule]:
        rules = list(self.base_rules)
        if round_num >= 3:
            rules.append(self.r3_rule)
        if round_num >= 5 and self.active_r5_rule is not None:
            rules.append(self.active_r5_rule)
        if round_num >= 7 and self.active_r7_audit_rule is not None:
            rules.append(self.active_r7_audit_rule)
        return rules

    def active_solutions(self, round_num: int) -> tuple[StateTuple, ...]:
        """Exact set of all valid configurations satisfying all active constraints in round_num."""
        if round_num < 3:
            return self.p1_sols
        if round_num < 5:
            self._ensure_r3_selected()
            assert self._active_r3_entry is not None
            return self._active_r3_entry.p2_sols
        if round_num < 7:
            self._ensure_r5_selected()
            assert self._active_r5_entry is not None
            return self._active_r5_entry.p3_sols
        self._ensure_r7_audit_selected()
        return self._active_p4_sols

    def _format_domains(self, keys: tuple[str, ...]) -> str:
        return "\n".join(f"- {k}: allowed values = {list(DOMAINS[k])}" for k in keys)

    def _format_visible_rules(self, role: Role, round_num: int) -> str:
        rules = [r for r in self._active_rules(round_num) if r.visible_to in (role, "BOTH")]
        return "\n".join(f"- [{r.rule_id}] {r.description}" for r in rules)

    def _format_state(self) -> str:
        return ", ".join(f"{k}={self.state[k]}" for k in ALL_KEYS)

    def system_briefing(self, role: Role, my_handle: str, partner_handle: str) -> str:
        my_keys = KEYS_A if role == "A" else KEYS_B
        partner_keys = KEYS_B if role == "A" else KEYS_A
        example = {my_keys[0]: DOMAINS[my_keys[0]][0], my_keys[1]: DOMAINS[my_keys[1]][1]}
        return (
            f"You are {my_handle}, collaborating with teammate {partner_handle} to repair a "
            f"12-parameter deployment configuration over {self.total_rounds} rounds.\n\n"
            f"OWNERSHIP:\n"
            f"- ONLY YOU can edit: {', '.join(my_keys)}\n"
            f"- ONLY {partner_handle} can edit: {', '.join(partner_keys)}\n"
            f"- RATE LIMIT: You may change AT MOST {MAX_EDITS_PER_ROUND} of your keys per round. "
            f"Additional key changes beyond {MAX_EDITS_PER_ROUND} in a round are ignored and incur a small score penalty.\n\n"
            f"ALLOWED VALUES FOR YOUR KEYS:\n{self._format_domains(my_keys)}\n\n"
            f"ALLOWED VALUES FOR {partner_handle}'S KEYS:\n{self._format_domains(partner_keys)}\n\n"
            f"YOUR PRIVATE CONSTRAINTS (your teammate CANNOT see them unless you share them):\n"
            f"{self._format_visible_rules(role, round_num=1)}\n\n"
            f"TASK RULES:\n"
            f"- The task runs for exactly {self.total_rounds} rounds. Production uptime is strict: "
            f"a round counts as UP only when 100% of active constraints pass (any failing constraint means DOWN for that round).\n"
            f"- Scoring combines strict production uptime across rounds (65%), a final Round-{self.total_rounds} "
            f"all-pass bonus (25%), and final Round-{self.total_rounds} constraint compliance (10%).\n"
            f"- After each round, the verifier reports the updated config, total passing constraints, "
            f"and which of YOUR visible constraints are failing.\n"
            f"- On every turn, write your message to {partner_handle}, then end with a final line:\n"
            f'  ACTION: {{"updates": {example}}}\n'
            f"Only the text before `ACTION:` is shown to {partner_handle}."
        )

    def turn_prompt(
        self,
        role: Role,
        round_num: int,
        partner_handle: str,
        partner_message: str | None,
        last_feedback: str | None,
    ) -> str:
        if round_num == 3:
            self._ensure_r3_selected()
        if round_num == 5:
            self._ensure_r5_selected()
        if round_num == 7:
            self._ensure_r7_audit_selected()

        parts = [f"=== ROUND {round_num} of {self.total_rounds} ==="]

        if round_num == 3 and role == "A":
            parts.append(
                f"PRIVATE OPS ALERT (sent ONLY to you; {partner_handle} was NOT notified):\n"
                f"A new mandatory constraint is now active:\n"
                f"- [{self.r3_rule.rule_id}] {self.r3_rule.description}"
            )
        if round_num == 5 and self.active_r5_rule is not None:
            parts.append(
                f"PLATFORM INCIDENT NOTICE (sent simultaneously to BOTH teammates):\n"
                f"An unforeseen legacy cluster hardware bug just triggered. Neither you nor "
                f"{partner_handle} could have known about this before Round 5.\n"
                f"- [{self.active_r5_rule.rule_id}] {self.active_r5_rule.description}"
            )
        if round_num == 7 and role == "B" and self.active_r7_audit_rule is not None:
            parts.append(
                f"PRIVATE AUDIT ALERT (sent ONLY to you; {partner_handle} was NOT notified):\n"
                f"A final pre-launch compliance constraint is now active (Round 7 of {self.total_rounds}; "
                f"Round {self.total_rounds} is the final round):\n"
                f"- [{self.active_r7_audit_rule.rule_id}] {self.active_r7_audit_rule.description}"
            )

        parts.append(f"CURRENT CONFIG:\n{self._format_state()}")
        if last_feedback:
            parts.append(f"VERIFIER UPDATE:\n{last_feedback}")
        if partner_message is not None:
            parts.append(f"Teammate {partner_handle} said:\n{partner_message}")
        else:
            parts.append(f"You speak first this round before {partner_handle}.")

        my_keys = KEYS_A if role == "A" else KEYS_B
        parts.append(
            f"Submit updates (AT MOST {MAX_EDITS_PER_ROUND} changed keys) for your keys "
            f"({', '.join(my_keys)}), ending with `ACTION: {{\"updates\": {{...}}}}`."
        )
        return "\n\n".join(parts)

    def _apply_updates(self, role: Role, action: dict[str, Any] | None) -> list[str]:
        warnings: list[str] = []
        if not action:
            return ["No valid ACTION parsed; your keys were unchanged."]
        updates = action.get("updates")
        if not isinstance(updates, dict):
            if any(k in ALL_KEYS for k in action):
                updates = action
            else:
                return ["ACTION missing 'updates' object; your keys were unchanged."]
        allowed = KEYS_A if role == "A" else KEYS_B
        changes_applied = 0
        for k, v in updates.items():
            if k not in ALL_KEYS:
                continue
            if k not in allowed:
                warnings.append(f"Ignored edit to partner-owned key `{k}`.")
                continue
            if isinstance(v, (bool, float)):
                warnings.append(f"Ignored non-integer value for `{k}`.")
                continue
            if isinstance(v, int):
                iv = v
            elif isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
                iv = int(v.strip())
            else:
                warnings.append(f"Ignored non-integer value for `{k}`.")
                continue
            if iv not in DOMAINS[k]:
                warnings.append(f"Ignored out-of-domain value {iv} for `{k}`.")
                continue
            if iv == self.state[k]:
                continue  # No-op reaffirmation doesn't consume edit budget
            if changes_applied >= MAX_EDITS_PER_ROUND:
                self._edit_cap_violations[role] += 1
                warnings.append(
                    f"Rate limit ({MAX_EDITS_PER_ROUND} edits/round): ignored change to `{k}`."
                )
                continue
            self.state[k] = iv
            changes_applied += 1
        return warnings

    def step_round(
        self,
        round_num: int,
        action_a: dict[str, Any] | None,
        action_b: dict[str, Any] | None,
        msg_a: str,
        msg_b: str,
    ) -> RoundFeedback:
        if round_num >= 3:
            self._ensure_r3_selected()
        if round_num >= 5:
            self._ensure_r5_selected()
        if round_num >= 7:
            self._ensure_r7_audit_selected()

        warn_a = self._apply_updates("A", action_a)
        warn_b = self._apply_updates("B", action_b)

        active = self._active_rules(round_num)
        passed = [r.rule_id for r in active if r.check(self.state)]
        failed_a = [r for r in active if r.visible_to in ("A", "BOTH") and not r.check(self.state)]
        failed_b = [r for r in active if r.visible_to in ("B", "BOTH") and not r.check(self.state)]

        frac = len(passed) / len(active)
        all_pass = len(passed) == len(active)
        self._round_compliance[round_num] = frac
        self._round_all_pass[round_num] = all_pass
        if round_num in (2, 4, 6, 8):
            self._phase_scores[f"round_{round_num}"] = round(frac, 6)

        status_str = "UP (all active constraints passing)" if all_pass else "DOWN (active constraint failure)"

        def _fmt_fb(failed: list[ConfigRule], warns: list[str]) -> str:
            lines = [f"Config after Round {round_num}: {self._format_state()}"]
            lines.append(f"Total active constraints passing: {len(passed)}/{len(active)} — Deployment status: {status_str}.")
            if failed:
                lines.append("YOUR visible failing constraints:")
                for r in failed:
                    lines.append(f"  - FAIL [{r.rule_id}]: {r.description}")
            else:
                lines.append("All of YOUR visible constraints are currently PASSING.")
            if warns:
                lines.append("Action notes: " + " ".join(warns))
            return "\n".join(lines)

        return RoundFeedback(
            round_num=round_num,
            feedback_a=_fmt_fb(failed_a, warn_a),
            feedback_b=_fmt_fb(failed_b, warn_b),
            metrics={
                "passed": len(passed),
                "total_active": len(active),
                "pass_fraction": round(frac, 6),
                "all_pass": all_pass,
                "state": dict(self.state),
            },
        )

    def _recovery_rounds(self, start_round: int, end_round: int) -> int | None:
        for r in range(start_round, end_round + 1):
            if self._round_all_pass.get(r, False):
                return r - start_round + 1
        return None

    def final_score(self) -> TaskScore:
        self._ensure_r7_audit_selected()
        active = self._active_rules(self.total_rounds)
        passed = [r.rule_id for r in active if r.check(self.state)]
        failed = [r.rule_id for r in active if not r.check(self.state)]

        final_compliance = len(passed) / len(active)
        final_all_pass = len(passed) == len(active)

        strict_uptime_count = sum(1 for r in PASSABLE_ROUNDS if self._round_all_pass.get(r, False))
        strict_uptime_rate = strict_uptime_count / len(PASSABLE_ROUNDS)

        total_violations = self._edit_cap_violations["A"] + self._edit_cap_violations["B"]
        edit_penalty = EDIT_CAP_PENALTY_PER_VIOLATION * total_violations

        raw_score = (
            STRICT_UPTIME_WEIGHT * strict_uptime_rate
            + FINAL_ALL_PASS_WEIGHT * (1.0 if final_all_pass else 0.0)
            + FINAL_COMPLIANCE_WEIGHT * final_compliance
        )
        score = round(max(0.0, min(1.0, raw_score - edit_penalty)), 6)

        per_round_compliance = {
            f"round_{r}": round(self._round_compliance.get(r, 0.0), 6)
            for r in range(1, self.total_rounds + 1)
        }
        per_round_all_pass = {
            f"round_{r}": bool(self._round_all_pass.get(r, False))
            for r in range(1, self.total_rounds + 1)
        }
        full_pass_rounds = [r for r in range(1, self.total_rounds + 1) if self._round_all_pass.get(r, False)]
        r5_rule_id = self.active_r5_rule.rule_id if self.active_r5_rule else None
        r7_audit_rule_id = self.active_r7_audit_rule.rule_id if self.active_r7_audit_rule else None

        realized_shocks = {
            "r3": {
                "round": 3,
                "rule_id": self.r3_rule.rule_id,
                "visible_to": self.r3_rule.visible_to,
                "description": self.r3_rule.description,
                "disrupted_state": self.r3_disrupted_state,
                "pre_shock_state": self._pre_shock_states.get("before_r3"),
            },
            "r5": {
                "round": 5,
                "rule_id": r5_rule_id,
                "visible_to": self.active_r5_rule.visible_to if self.active_r5_rule else "BOTH",
                "description": self.active_r5_rule.description if self.active_r5_rule else None,
                "disrupted_state": self.r5_disrupted_state,
                "pre_shock_state": self._pre_shock_states.get("before_r5"),
            },
            "r7": {
                "round": 7,
                "rule_id": r7_audit_rule_id,
                "visible_to": self.active_r7_audit_rule.visible_to if self.active_r7_audit_rule else "B",
                "description": self.active_r7_audit_rule.description if self.active_r7_audit_rule else None,
                "disrupted_state": self.r7_audit_disrupted_state,
                "pre_shock_state": self._pre_shock_states.get("before_r7"),
            },
        }

        return TaskScore(
            score=score,
            details={
                "passed": len(passed),
                "total_active": len(active),
                "failed_rule_ids": failed,
                "final_compliance": round(final_compliance, 6),
                "final_all_pass": final_all_pass,
                "per_round_compliance": per_round_compliance,
                "per_round_all_pass": per_round_all_pass,
                "full_pass_rounds": full_pass_rounds,
                "passable_rounds": list(PASSABLE_ROUNDS),
                "strict_uptime_count": strict_uptime_count,
                "strict_uptime_rate": round(strict_uptime_rate, 6),
                "recovery_rounds_r3": self._recovery_rounds(3, 4),
                "recovery_rounds_r5": self._recovery_rounds(5, 6),
                "recovery_rounds_r7": self._recovery_rounds(7, 8),
                # Clean 8-round shock fields + full realized clauses & pre-shock states for export
                "r3_rule_id": self.r3_rule.rule_id,
                "r5_rule_id": r5_rule_id,
                "r7_audit_rule_id": r7_audit_rule_id,
                "r3_disrupted_state": self.r3_disrupted_state,
                "r5_disrupted_state": self.r5_disrupted_state,
                "r7_audit_disrupted_state": self.r7_audit_disrupted_state,
                "realized_shocks": realized_shocks,
                "initial_state": dict(self.initial_state),
                # Backward-compatible aliases (r7_* points to ERR-LEG-7 external incident; r9_* to B-only audit)
                "r4_rule_id": self.r3_rule.rule_id,
                "r7_rule_id": r5_rule_id,
                "r9_rule_id": r7_audit_rule_id,
                "r4_disrupted_state": self.r3_disrupted_state,
                "r7_disrupted_state": self.r5_disrupted_state,
                "r9_disrupted_state": self.r7_audit_disrupted_state,
                "edit_cap_violations": dict(self._edit_cap_violations),
                "edit_cap_penalty": round(edit_penalty, 6),
                "phase_scores": dict(self._phase_scores),
                "solution_counts": {
                    "p1": len(self.p1_sols),
                    "p2": len(self._active_r3_entry.p2_sols) if self._active_r3_entry else None,
                    "p3": len(self._active_r5_entry.p3_sols) if self._active_r5_entry else None,
                    "p4": len(self._active_p4_sols) if self._active_p4_sols else None,
                },
                "final_state": dict(self.state),
            },
        )

    def _step_toward(self, target: dict[str, int], keys: tuple[str, ...]) -> dict[str, int]:
        """Move at most MAX_EDITS_PER_ROUND differing keys toward target."""
        diffs = {k: target[k] for k in keys if self.state[k] != target[k]}
        return dict(list(diffs.items())[:MAX_EDITS_PER_ROUND])

    def oracle_actions(self, round_num: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if round_num < 3:
            target = self.oracle_p1
        else:
            sols = self.active_solutions(round_num)
            curr = _state_to_tuple(self.state)
            target = _tuple_to_state(_closest_target(curr, sols))
        return (
            {"updates": self._step_toward(target, KEYS_A)},
            {"updates": self._step_toward(target, KEYS_B)},
        )
