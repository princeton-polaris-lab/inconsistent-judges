from ruleforge.tasks.base import (
    BUILTIN_TASKS,
    DEFAULT_TASK,
    FORCED_JSON,
    LABEL,
    RESPONSE_CONTRACTS,
    OptimizableTask,
    Task,
    TaskError,
    TaskSettings,
)
from ruleforge.tasks.history import RewardWeights
from ruleforge.tasks.legal import LegalTask
from ruleforge.tasks.modelspec import ModelSpecTask
from ruleforge.tasks.registry import (
    ENTRY_POINT_GROUP,
    TaskInfo,
    check_task_identity,
    list_tasks,
    load_task,
    register_task,
    task_locator,
)

__all__ = [
    "BUILTIN_TASKS", "DEFAULT_TASK", "ENTRY_POINT_GROUP", "FORCED_JSON", "LABEL", "RESPONSE_CONTRACTS", "LegalTask",
    "ModelSpecTask", "OptimizableTask", "RewardWeights", "Task", "TaskError", "TaskInfo", "TaskSettings",
    "check_task_identity", "list_tasks", "load_task", "register_task", "task_locator",
]
