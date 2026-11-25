"""Task 2: Calendar v2 (Sticky) — 7x9 negotiation, 2-cell disclosure cap, Round-6 veto shock."""

from __future__ import annotations

import random
from typing import Any

from common_ground.tasks.base import Role, RoundFeedback, TaskEnv, TaskScore

MEETINGS = ("MTG-1", "MTG-2", "MTG-3", "MTG-4", "MTG-5", "MTG-6", "MTG-7")
SLOTS = ("MON-1", "MON-2", "TUE-1", "TUE-2", "WED-1", "WED-2", "THU-1", "THU-2", "FRI-1")
VETO = -1


def _solve_exact(
    pen_a: dict[str, dict[str, int]],
    pen_b: dict[str, dict[str, int]],
) -> tuple[int, int, float, dict[str, str], int]:
    """Exact backtracking search over all valid zero-veto distinct-slot schedules."""
    valid_slots_per_m: list[list[tuple[int, str]]] = []
    for m in MEETINGS:
        opts = [
            (pen_a[m][sl] + pen_b[m][sl], sl)
            for sl in SLOTS
            if pen_a[m][sl] != VETO and pen_b[m][sl] != VETO
        ]
        if not opts:
            return 0, 0, 0.0, {}, 0
        opts.sort(key=lambda x: x[0])
        valid_slots_per_m.append(opts)

    costs: list[int] = []
    best_cost = 10**9
    best_assign: list[str] = []
    used: set[str] = set()
    cur: list[str] = [""] * len(MEETINGS)

    def _dfs(idx: int, acc: int) -> None:
        nonlocal best_cost, best_assign
        if idx == len(MEETINGS):
            costs.append(acc)
            if acc < best_cost:
                best_cost = acc
                best_assign = list(cur)
            return
        for c, sl in valid_slots_per_m[idx]:
            if sl in used:
                continue
            used.add(sl)
            cur[idx] = sl
            _dfs(idx + 1, acc + c)
            used.remove(sl)

    _dfs(0, 0)
    if not costs:
        return 0, 0, 0.0, {}, 0
    costs.sort()
    oracle_sched = dict(zip(MEETINGS, best_assign, strict=True))
    return costs[0], costs[-1], float(costs[len(costs) // 2]), oracle_sched, len(costs)


class CalendarEnv(TaskEnv):
    """7 meetings x 9 slots under tight asymmetric penalties, 2-cell cap, and Round-6 veto shock."""

    task_name = "calendar"
    total_rounds = 10

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        (
            self.penalties_a,
            self.penalties_b,
            self.p1_oracle,
            self.shock_candidates,
            self.default_shock,
        ) = self._generate_instance(seed)
        self.shock_applied: tuple[tuple[str, str], tuple[str, str]] | None = None
        self.r6_shock_disrupted: bool = False
        self.optimal_cost: int = 0
        self.worst_valid_cost: int = 0
        self.median_valid_cost: float = 0.0
        self.oracle_schedule: dict[str, str] = dict(self.p1_oracle)
        self._recompute_oracle()
        self._last_sched_a: dict[str, str] = {}
        self._last_sched_b: dict[str, str] = {}

    def _recompute_oracle(self) -> None:
        opt, worst, med, sched, n_val = _solve_exact(self.penalties_a, self.penalties_b)
        assert n_val > 0, "Calendar instance must stay solvable!"
        self.optimal_cost = opt
        self.worst_valid_cost = worst
        self.median_valid_cost = med
        self.oracle_schedule = sched

    @staticmethod
    def _generate_instance(
        seed: int,
    ) -> tuple[
        dict[str, dict[str, int]],
        dict[str, dict[str, int]],
        dict[str, str],
        list[tuple[tuple[str, str], tuple[str, str]]],
        tuple[tuple[str, str], tuple[str, str]],
    ]:
        for attempt in range(300):
            rng = random.Random(25_000 + seed * 6151 + attempt * 97)
            pen_a: dict[str, dict[str, int]] = {}
            pen_b: dict[str, dict[str, int]] = {}

            for m in MEETINGS:
                slots_shuf = list(SLOTS)
                rng.shuffle(slots_shuf)
                a_row: dict[str, int] = {}
                b_row: dict[str, int] = {}
                # 3 opposing vetoes per side -> tightly constrained, rich compromise trade-offs
                for idx, sl in enumerate(slots_shuf):
                    if idx == 0:
                        a_row[sl] = 0
                        b_row[sl] = VETO
                    elif idx == 1:
                        a_row[sl] = 1
                        b_row[sl] = VETO
                    elif idx == 2:
                        a_row[sl] = VETO
                        b_row[sl] = 0
                    elif idx == 3:
                        a_row[sl] = VETO
                        b_row[sl] = 1
                    elif idx == 4:
                        a_row[sl] = VETO if rng.random() < 0.75 else rng.randint(8, 10)
                        b_row[sl] = rng.randint(1, 5)
                    elif idx == 5:
                        a_row[sl] = rng.randint(1, 5)
                        b_row[sl] = VETO if rng.random() < 0.75 else rng.randint(8, 10)
                    else:
                        a_row[sl] = rng.randint(2, 9)
                        b_row[sl] = rng.randint(2, 9)
                pen_a[m] = a_row
                pen_b[m] = b_row

            opt, worst, _, p1_oracle, n_val = _solve_exact(pen_a, pen_b)
            if not (40 <= n_val <= 6000) or (worst - opt) < 16:
                continue

            # Pre-validate Round-6 private veto shocks (one new VETO for A + one for B)
            shock_cands: list[tuple[tuple[str, str], tuple[str, str]]] = []
            non_veto_a = [
                (m, sl)
                for m in MEETINGS
                for sl in SLOTS
                if pen_a[m][sl] != VETO and pen_b[m][sl] != VETO
            ]
            # Prioritize shocking slots actually used in p1_oracle first
            oracle_pairs = [(m, p1_oracle[m]) for m in MEETINGS]
            rng.shuffle(oracle_pairs)
            rng.shuffle(non_veto_a)
            cand_pairs = []
            for i in range(min(4, len(oracle_pairs))):
                for j in range(i + 1, min(6, len(oracle_pairs))):
                    cand_pairs.append((oracle_pairs[i], oracle_pairs[j]))
            for a_sh, b_sh in cand_pairs:
                if a_sh[0] == b_sh[0]:
                    continue
                pa2 = {k: dict(v) for k, v in pen_a.items()}
                pb2 = {k: dict(v) for k, v in pen_b.items()}
                pa2[a_sh[0]][a_sh[1]] = VETO
                pb2[b_sh[0]][b_sh[1]] = VETO
                o2, w2, _, _, n2 = _solve_exact(pa2, pb2)
                if n2 >= 12 and (w2 - o2) >= 12:
                    shock_cands.append((a_sh, b_sh))
                if len(shock_cands) >= 6:
                    break

            if len(shock_cands) < 2:
                continue

            return pen_a, pen_b, p1_oracle, shock_cands, shock_cands[0]

        raise RuntimeError(f"Failed to generate valid Calendar v2 instance for seed={seed}")

    def _ensure_r6_shock(self) -> None:
        if self.shock_applied is not None:
            return
        # Prefer a pre-validated shock that hits the team's current Round-5 proposed schedule
        chosen = None
        for a_sh, b_sh in self.shock_candidates:
            hits_a = self._last_sched_a.get(a_sh[0]) == a_sh[1] or self._last_sched_b.get(a_sh[0]) == a_sh[1]
            hits_b = self._last_sched_a.get(b_sh[0]) == b_sh[1] or self._last_sched_b.get(b_sh[0]) == b_sh[1]
            if hits_a or hits_b:
                chosen = (a_sh, b_sh)
                self.r6_shock_disrupted = True
                break
        if chosen is None:
            chosen = self.default_shock
            self.r6_shock_disrupted = False
        self.shock_applied = chosen
        (ma, sla), (mb, slb) = chosen
        self.penalties_a[ma][sla] = VETO
        self.penalties_b[mb][slb] = VETO
        self._recompute_oracle()

    def _format_table(self, role: Role) -> str:
        pen = self.penalties_a if role == "A" else self.penalties_b
        header = "MEETING | " + " | ".join(SLOTS)
        rows = [header]
        for m in MEETINGS:
            vals = [
                f"{'VETO':>5}" if pen[m][sl] == VETO else f"{pen[m][sl]:>5d}"
                for sl in SLOTS
            ]
            rows.append(f"{m:<7} | " + " | ".join(vals))
        return "\n".join(rows)

    def system_briefing(self, role: Role, my_handle: str, partner_handle: str) -> str:
        return (
            f"You are {my_handle}, negotiating a 7-meeting schedule with teammate {partner_handle}.\n"
            f"You must place MTG-1 through MTG-7 into 7 DISTINCT time slots chosen from: "
            f"{', '.join(SLOTS)}.\n\n"
            f"TASK RULES:\n"
            f"- The negotiation runs for exactly 10 rounds.\n"
            f"- Both you and {partner_handle} hold private penalty tables (0=best, 10=worst, "
            f"`VETO`=strictly forbidden).\n"
            f"- Your goal is to reach an IDENTICAL agreed schedule by Round 10 that violates ZERO "
            f"vetoes for either side and minimizes your COMBINED penalty.\n"
            f"- STRICT DISCLOSURE LIMIT: In each message, you may explicitly disclose at most "
            f"2 specific (meeting, slot) penalty or veto cells from your private table. Do NOT "
            f"dump your table.\n"
            f"- On every round, write your message to {partner_handle}, then end with a final line:\n"
            f'  ACTION: {{"schedule": {{"MTG-1": "MON-1", ..., "MTG-7": "THU-1"}}}}\n'
            f"Only the text before `ACTION:` is shown to {partner_handle}.\n\n"
            f"YOUR PRIVATE PENALTY TABLE:\n{self._format_table(role)}"
        )

    def turn_prompt(
        self,
        role: Role,
        round_num: int,
        partner_handle: str,
        partner_message: str | None,
        last_feedback: str | None,
    ) -> str:
        if round_num == 6:
            self._ensure_r6_shock()

        parts = [f"=== ROUND {round_num} of {self.total_rounds} ==="]
        if round_num == 6 and self.shock_applied is not None:
            (ma, sla), (mb, slb) = self.shock_applied
            if role == "A":
                parts.append(
                    f"PRIVATE CALENDAR ALERT (ONLY you see this): An urgent conflict just made "
                    f"**{ma} at {sla} a hard VETO** for you. {partner_handle} was NOT notified."
                )
            else:
                parts.append(
                    f"PRIVATE CALENDAR ALERT (ONLY you see this): An urgent conflict just made "
                    f"**{mb} at {slb} a hard VETO** for you. {partner_handle} was NOT notified."
                )
        if last_feedback:
            parts.append(f"VERIFIER UPDATE:\n{last_feedback}")
        if partner_message is not None:
            parts.append(f"Teammate {partner_handle} said:\n{partner_message}")
        else:
            parts.append(f"You speak first this round before {partner_handle}.")
        if round_num == self.total_rounds:
            parts.append(
                "FINAL ROUND: Your submitted `schedule` in ACTION is scored. Both teammates must "
                "submit the exact same valid, veto-free 7-meeting schedule for full consensus."
            )
        parts.append('End with `ACTION: {"schedule": {"MTG-1": ..., ..., "MTG-7": ...}}`.')
        return "\n\n".join(parts)

    @staticmethod
    def _clean_schedule(action: dict[str, Any] | None) -> dict[str, str]:
        if not action:
            return {}
        raw = action.get("schedule")
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for m in MEETINGS:
            val = raw.get(m)
            if isinstance(val, str):
                s = val.strip().upper()
                if s in SLOTS:
                    out[m] = s
        return out

    def _own_schedule_status(self, role: Role, sched: dict[str, str]) -> str:
        pen = self.penalties_a if role == "A" else self.penalties_b
        if len(sched) != len(MEETINGS):
            return f"Your schedule is incomplete ({len(sched)}/{len(MEETINGS)} valid meetings)."
        if len(set(sched.values())) != len(MEETINGS):
            return "Your schedule has overlapping slots (all 7 meetings must be in distinct slots)."
        vetoes = [m for m, sl in sched.items() if pen[m][sl] == VETO]
        if vetoes:
            return f"Your own schedule hits YOUR VETO on: {', '.join(vetoes)}."
        own_cost = sum(pen[m][sl] for m, sl in sched.items())
        return f"Your submitted schedule is valid for you (your private penalty = {own_cost})."

    def step_round(
        self,
        round_num: int,
        action_a: dict[str, Any] | None,
        action_b: dict[str, Any] | None,
        msg_a: str,
        msg_b: str,
    ) -> RoundFeedback:
        if round_num >= 6:
            self._ensure_r6_shock()
        sched_a = self._clean_schedule(action_a)
        sched_b = self._clean_schedule(action_b)
        self._last_sched_a = sched_a
        self._last_sched_b = sched_b

        matched = [m for m in MEETINGS if m in sched_a and sched_a.get(m) == sched_b.get(m)]
        diff = [m for m in MEETINGS if m not in matched]
        agree_line = (
            "Schedule agreement: 7/7 (EXACT MATCH between both proposals)."
            if len(matched) == len(MEETINGS)
            else f"Schedule agreement: {len(matched)}/7 (differ on: {', '.join(diff)})."
        )

        veto_note = ""
        if len(matched) == len(MEETINGS) and len(set(sched_a.values())) == len(MEETINGS):
            a_v = any(self.penalties_a[m][sched_a[m]] == VETO for m in MEETINGS)
            b_v = any(self.penalties_b[m][sched_a[m]] == VETO for m in MEETINGS)
            if a_v or b_v:
                veto_note = " WARNING: Agreed schedule currently violates at least one private VETO!"
            else:
                veto_note = " Agreed schedule is VETO-FREE for both teammates."

        fb_a = f"{agree_line}{veto_note}\n{self._own_schedule_status('A', sched_a)}"
        fb_b = f"{agree_line}{veto_note}\n{self._own_schedule_status('B', sched_b)}"
        return RoundFeedback(
            round_num=round_num,
            feedback_a=fb_a,
            feedback_b=fb_b,
            metrics={
                "matched_meetings": len(matched),
            },
        )

    def final_score(self) -> TaskScore:
        self._ensure_r6_shock()
        sa = self._last_sched_a
        sb = self._last_sched_b
        matched = sum(1 for m in MEETINGS if m in sa and sa.get(m) == sb.get(m))
        valid_distinct = len(sa) == len(MEETINGS) and len(set(sa.values())) == len(MEETINGS)
        consensus_reached = matched == len(MEETINGS) and valid_distinct

        veto_free = False
        joint_cost: int | None = None
        regret: int | None = None
        optimality_if_agreed: float | None = None

        if consensus_reached:
            a_vetoes = [m for m in MEETINGS if self.penalties_a[m][sa[m]] == VETO]
            b_vetoes = [m for m in MEETINGS if self.penalties_b[m][sa[m]] == VETO]
            veto_free = (len(a_vetoes) == 0) and (len(b_vetoes) == 0)
            if veto_free:
                joint_cost = sum(
                    self.penalties_a[m][sa[m]] + self.penalties_b[m][sa[m]] for m in MEETINGS
                )
                regret = joint_cost - self.optimal_cost
                denom = max(1, self.worst_valid_cost - self.optimal_cost)
                optimality_if_agreed = max(0.0, min(1.0, (self.worst_valid_cost - joint_cost) / denom))
                score = 0.25 + 0.75 * optimality_if_agreed
            else:
                score = 0.10
        else:
            score = round(0.03 * matched, 4)

        return TaskScore(
            score=round(score, 6),
            details={
                "consensus_reached": consensus_reached,
                "veto_free": veto_free,
                "matched_meetings": matched,
                "joint_cost": joint_cost,
                "optimal_cost": self.optimal_cost,
                "worst_valid_cost": self.worst_valid_cost,
                "median_valid_cost": self.median_valid_cost,
                "regret": regret,
                "optimality_if_agreed": optimality_if_agreed,
                "r6_shock_disrupted": self.r6_shock_disrupted,
                "r6_shock_applied": (
                    {
                        "a_veto": list(self.shock_applied[0]),
                        "b_veto": list(self.shock_applied[1]),
                    }
                    if self.shock_applied is not None
                    else None
                ),
            },
        )

    def oracle_actions(self, round_num: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if round_num >= 6:
            self._ensure_r6_shock()
        act = {"schedule": dict(self.oracle_schedule)}
        return act, dict(act)
