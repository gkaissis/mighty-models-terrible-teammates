"""Deterministic multi-turn collaborative task environments."""

from common_ground.tasks.base import (
    BLIND_HANDLES,
    RoundFeedback,
    TaskEnv,
    TaskScore,
    make_blind_handles,
    parse_turn_response,
)
from common_ground.tasks.calendar_task import CalendarEnv
from common_ground.tasks.code_handoff_task import CodeHandoffEnv
from common_ground.tasks.goalpost_task import GoalpostEnv
from common_ground.tasks.ledger_task import LedgerEnv
from common_ground.tasks.triage_task import TriageEnv

TASK_REGISTRY: dict[str, type[TaskEnv]] = {
    "triage": TriageEnv,
    "calendar": CalendarEnv,
    "goalpost": GoalpostEnv,
    "code_handoff": CodeHandoffEnv,
    "ledger": LedgerEnv,  # kept for backward compatibility with pilot v1 logs
}

DEFAULT_MAIN_TASKS: tuple[str, ...] = ("triage", "calendar", "goalpost", "code_handoff")


def make_task(task_name: str, seed: int) -> TaskEnv:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_name](seed=seed)


__all__ = [
    "BLIND_HANDLES",
    "DEFAULT_MAIN_TASKS",
    "TASK_REGISTRY",
    "CalendarEnv",
    "CodeHandoffEnv",
    "GoalpostEnv",
    "LedgerEnv",
    "RoundFeedback",
    "TaskEnv",
    "TaskScore",
    "TriageEnv",
    "make_blind_handles",
    "make_task",
    "parse_turn_response",
]
