"""Task 1: Ledger (Smooth) — Private-shard join with 5 sequential consensus queries (10 rounds)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from common_ground.tasks.base import Role, RoundFeedback, TaskEnv, TaskScore

DEPTS = ("OPS", "FIN", "ENG", "SEC", "LOG")
STATUSES = ("ACTIVE", "AUDIT", "HOLD", "CLOSED")
OWNERS = ("P-1", "P-2", "P-3", "P-4", "P-5", "P-6")
REGIONS = ("EU-W", "US-E", "AP-S", "EU-N")
PRIORITIES = ("P1", "P2", "P3")
FLAGS = ("CLEAN", "REVIEW", "DUP")


@dataclass(frozen=True)
class LedgerRecord:
    rec_id: str
    # Role A shard
    dept: str
    cost: int
    status: str
    # Role B shard
    owner: str
    region: str
    priority: str
    flag: str


@dataclass(frozen=True)
class LedgerQuery:
    query_id: str
    prompt: str
    expected_answer: str


class LedgerEnv(TaskEnv):
    """24 split records, 5 exact multi-shard queries over 10 rounds (2 rounds per query)."""

    task_name = "ledger"
    total_rounds = 10

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.records, self.queries = self._generate_instance(seed)
        self._query_results: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _generate_instance(seed: int) -> tuple[list[LedgerRecord], list[LedgerQuery]]:
        rng = random.Random(10_000 + seed * 7919)

        # Generate 24 records ensuring rich coverage and clean unique answers for all 5 queries
        for attempt in range(200):
            r = random.Random(10_000 + seed * 7919 + attempt * 131)
            records: list[LedgerRecord] = []
            used_costs: set[int] = set()
            for i in range(1, 25):
                # Unique costs avoid ties on max/min cost queries
                while True:
                    cost = r.randrange(120, 990, 10)
                    if cost not in used_costs:
                        used_costs.add(cost)
                        break
                records.append(
                    LedgerRecord(
                        rec_id=f"REC-{i:02d}",
                        dept=r.choice(DEPTS),
                        cost=cost,
                        status=r.choices(STATUSES, weights=[4, 3, 2, 2], k=1)[0],
                        owner=r.choice(OWNERS),
                        region=r.choice(REGIONS),
                        priority=r.choices(PRIORITIES, weights=[3, 4, 3], k=1)[0],
                        flag=r.choices(FLAGS, weights=[6, 2, 2], k=1)[0],
                    )
                )

            # Build 5 deterministic cross-shard queries anchored to actual subsets
            # Q-1: Unique record match across both shards
            q1_candidates = [
                rec for rec in records if rec.status == "ACTIVE" and rec.flag == "CLEAN"
            ]
            q1_unique = [
                rec
                for rec in q1_candidates
                if sum(
                    1
                    for x in records
                    if x.dept == rec.dept
                    and x.status == "ACTIVE"
                    and x.region == rec.region
                    and x.priority == rec.priority
                    and x.flag == "CLEAN"
                )
                == 1
            ]
            if not q1_unique:
                continue
            t1 = r.choice(q1_unique)
            q1 = LedgerQuery(
                query_id="Q-1",
                prompt=(
                    f"Find the unique record ID (e.g. REC-01) with dept={t1.dept}, "
                    f"status=ACTIVE, region={t1.region}, priority={t1.priority}, and flag=CLEAN."
                ),
                expected_answer=t1.rec_id,
            )

            # Q-2: Sum of cost for records matching status + region + flag != DUP
            q2_opts = []
            for st in ("ACTIVE", "AUDIT", "HOLD"):
                for reg in REGIONS:
                    subset = [
                        x for x in records if x.status == st and x.region == reg and x.flag != "DUP"
                    ]
                    if 2 <= len(subset) <= 5:
                        q2_opts.append((st, reg, subset))
            if not q2_opts:
                continue
            st2, reg2, sub2 = r.choice(q2_opts)
            q2 = LedgerQuery(
                query_id="Q-2",
                prompt=(
                    f"Compute the exact integer sum of `cost` across all records with "
                    f"status={st2}, region={reg2}, and flag != DUP."
                ),
                expected_answer=str(sum(x.cost for x in sub2)),
            )

            # Q-3: Highest-cost record with cost >= 400, status != CLOSED, priority == p, flag == CLEAN
            q3_opts = []
            for prio in PRIORITIES:
                for owner_excl in OWNERS:
                    subset = [
                        x
                        for x in records
                        if x.cost >= 400
                        and x.status != "CLOSED"
                        and x.priority == prio
                        and x.flag == "CLEAN"
                        and x.owner != owner_excl
                    ]
                    if 2 <= len(subset) <= 6:
                        q3_opts.append((prio, owner_excl, subset))
            if not q3_opts:
                continue
            prio3, own3, sub3 = r.choice(q3_opts)
            best3 = max(sub3, key=lambda x: x.cost)
            q3 = LedgerQuery(
                query_id="Q-3",
                prompt=(
                    f"Among records with cost >= 400, status != CLOSED, priority={prio3}, "
                    f"flag=CLEAN, and owner != {own3}, which record ID has the highest `cost`?"
                ),
                expected_answer=best3.rec_id,
            )

            # Q-4: Count of records with dept in {d1, d2} AND priority in {P1, P2} AND flag != DUP AND status != CLOSED
            d1, d2 = r.sample(DEPTS, 2)
            sub4 = [
                x
                for x in records
                if x.dept in (d1, d2)
                and x.priority in ("P1", "P2")
                and x.flag != "DUP"
                and x.status != "CLOSED"
            ]
            if not (2 <= len(sub4) <= 8):
                continue
            q4 = LedgerQuery(
                query_id="Q-4",
                prompt=(
                    f"How many records have dept in {{{d1}, {d2}}}, status != CLOSED, "
                    f"priority in {{P1, P2}}, and flag != DUP? Reply with an integer."
                ),
                expected_answer=str(len(sub4)),
            )

            # Q-5: Composite SUM:MIN_REC_ID across a filtered join
            q5_opts = []
            for reg5 in REGIONS:
                for fl5 in ("CLEAN", "REVIEW"):
                    subset = [
                        x
                        for x in records
                        if x.region == reg5 and x.status in ("ACTIVE", "AUDIT") and x.flag != "DUP"
                    ]
                    if 2 <= len(subset) <= 5:
                        q5_opts.append((reg5, subset))
            if not q5_opts:
                continue
            reg5, sub5 = r.choice(q5_opts)
            total5 = sum(x.cost for x in sub5)
            lowest5 = min(sub5, key=lambda x: x.cost)
            q5 = LedgerQuery(
                query_id="Q-5",
                prompt=(
                    f"Consider all records with status in {{ACTIVE, AUDIT}}, region={reg5}, "
                    f"and flag != DUP. Compute their total `cost` and identify the record ID "
                    f"with the lowest `cost` in that set. Format your answer strictly as "
                    f"`TOTAL:REC-XX` (e.g. `1450:REC-04`)."
                ),
                expected_answer=f"{total5}:{lowest5.rec_id}",
            )

            # Sanity check: ensure rng was used deterministically
            _ = rng.random()
            return records, [q1, q2, q3, q4, q5]

        raise RuntimeError(f"Failed to generate valid Ledger instance for seed={seed}")

    def _format_shard(self, role: Role) -> str:
        lines = []
        if role == "A":
            lines.append("REC_ID | dept | cost | status")
            for rec in self.records:
                lines.append(f"{rec.rec_id} | {rec.dept} | {rec.cost} | {rec.status}")
        else:
            lines.append("REC_ID | owner | region | priority | flag")
            for rec in self.records:
                lines.append(
                    f"{rec.rec_id} | {rec.owner} | {rec.region} | {rec.priority} | {rec.flag}"
                )
        return "\n".join(lines)

    def system_briefing(self, role: Role, my_handle: str, partner_handle: str) -> str:
        other_cols = (
            "owner, region, priority, flag" if role == "A" else "dept, cost, status"
        )
        return (
            f"You are {my_handle}, collaborating with teammate {partner_handle} on a 10-round "
            f"data-reconciliation task.\n"
            f"There are 24 operational records (REC-01 through REC-24). Each of you holds half of "
            f"the columns for every record. Your teammate holds: {other_cols}.\n\n"
            f"TASK RULES:\n"
            f"- The task runs for exactly 10 rounds.\n"
            f"- Five queries (Q-1 through Q-5) are presented sequentially, 2 rounds per query:\n"
            f"  * Odd rounds (1, 3, 5, 7, 9): exchange shard information for the current query.\n"
            f"  * Even rounds (2, 4, 6, 8, 10): finalize and submit your answer for that query.\n"
            f"- A query scores 1 point ONLY if BOTH you and {partner_handle} submit the exact same "
            f"correct answer on its even resolution round.\n"
            f"- On every turn, write your message to {partner_handle}, then end your response with "
            f"a final line containing:\n"
            f'  ACTION: {{"query_id": "Q-X", "answer": "<exact string or null on odd rounds>"}}\n'
            f"Only the text before `ACTION:` is shown to {partner_handle}.\n\n"
            f"YOUR PRIVATE SHARD:\n"
            f"{self._format_shard(role)}"
        )

    def _active_query(self, round_num: int) -> LedgerQuery:
        return self.queries[(round_num - 1) // 2]

    def turn_prompt(
        self,
        role: Role,
        round_num: int,
        partner_handle: str,
        partner_message: str | None,
        last_feedback: str | None,
    ) -> str:
        q = self._active_query(round_num)
        phase = "EXCHANGE ROUND (1 of 2 for this query)" if round_num % 2 == 1 else "FINAL SUBMISSION ROUND (2 of 2 — both must submit matching exact answers)"
        parts = [f"=== ROUND {round_num} of {self.total_rounds} ==="]
        if last_feedback:
            parts.append(f"VERIFIER UPDATE:\n{last_feedback}")
        if partner_message is not None:
            parts.append(f"Teammate {partner_handle} said:\n{partner_message}")
        else:
            parts.append(f"You speak first this round before {partner_handle}.")
        parts.append(
            f"ACTIVE QUERY ({q.query_id}) — {phase}:\n{q.prompt}\n\n"
            f'End with `ACTION: {{"query_id": "{q.query_id}", "answer": ...}}`.'
        )
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_ans(val: Any) -> str | None:
        if val is None:
            return None
        s = str(val).strip().upper()
        # Allow optional whitespace around colon in Q-5
        s = s.replace(" : ", ":").replace(": ", ":").replace(" :", ":")
        return s

    def step_round(
        self,
        round_num: int,
        action_a: dict[str, Any] | None,
        action_b: dict[str, Any] | None,
        msg_a: str,
        msg_b: str,
    ) -> RoundFeedback:
        q = self._active_query(round_num)
        if round_num % 2 == 1:
            fb = f"{q.query_id} exchange round complete. Submit your agreed final answer on Round {round_num + 1}."
            return RoundFeedback(round_num=round_num, feedback_a=fb, feedback_b=fb)

        ans_a = self._normalize_ans((action_a or {}).get("answer"))
        ans_b = self._normalize_ans((action_b or {}).get("answer"))
        expected = self._normalize_ans(q.expected_answer)
        consensus = ans_a is not None and ans_a == ans_b
        correct = consensus and ans_a == expected

        self._query_results[q.query_id] = {
            "expected": expected,
            "ans_a": ans_a,
            "ans_b": ans_b,
            "consensus": consensus,
            "correct": correct,
            "a_correct": ans_a == expected,
            "b_correct": ans_b == expected,
        }

        if correct:
            status_line = f"{q.query_id} RESULT: CORRECT (consensus achieved: {ans_a})."
        elif consensus:
            status_line = f"{q.query_id} RESULT: INCORRECT (both submitted {ans_a}, which did not match the verifier)."
        else:
            status_line = (
                f"{q.query_id} RESULT: NO CONSENSUS (submissions differed: "
                f"Role A={ans_a}, Role B={ans_b}). 0 points."
            )
        return RoundFeedback(
            round_num=round_num,
            feedback_a=status_line,
            feedback_b=status_line,
            metrics=self._query_results[q.query_id],
        )

    def final_score(self) -> TaskScore:
        solved = sum(1 for r in self._query_results.values() if r["correct"])
        consensus_cnt = sum(1 for r in self._query_results.values() if r["consensus"])
        return TaskScore(
            score=solved / len(self.queries),
            details={
                "queries_solved": solved,
                "queries_total": len(self.queries),
                "queries_consensus": consensus_cnt,
                "per_query": dict(self._query_results),
            },
        )

    def oracle_actions(self, round_num: int) -> tuple[dict[str, Any], dict[str, Any]]:
        q = self._active_query(round_num)
        act = {"query_id": q.query_id, "answer": q.expected_answer}
        return act, dict(act)
