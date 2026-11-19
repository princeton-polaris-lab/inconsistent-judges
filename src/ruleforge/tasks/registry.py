from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Union

from ruleforge.tasks.base import DEFAULT_TASK, OptimizableTask, Task, TaskError
from ruleforge.tasks.legal import LegalTask
from ruleforge.tasks.modelspec import ModelSpecTask

ENTRY_POINT_GROUP = "ruleforge.tasks"
BUILTIN_TASK_CLASSES: dict[str, type[Task]] = {"modelspec": ModelSpecTask, "legal": LegalTask}
_REGISTERED: dict[str, type[Task]] = {}
TaskLike = Union[str, Task, type]


def register_task(name: str, cls: type[Task]) -> type[Task]:
    """Register ``cls`` so ``load_task(name)`` finds it; a built-in name, or a class named otherwise, is refused."""
    if not (isinstance(cls, type) and issubclass(cls, Task)):
        raise TaskError(f"register_task({name!r}, ...) needs a Task subclass, got {cls!r}")
    if name in BUILTIN_TASK_CLASSES:
        raise TaskError(f"{name!r} is a built-in task's name; register the task under its own")
    if getattr(cls, "name", None) != name:
        raise TaskError(f"{cls.__qualname__} is named {getattr(cls, 'name', None)!r}, not {name!r}: a task is "
                        "registered under the name it records")
    _REGISTERED[name] = cls
    return cls


def _entry_point(name: str) -> Optional[type[Task]]:
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry.name == name:
            return entry.load()
    return None


def _dotted(path: str) -> Any:
    module, _, attribute = path.partition(":")
    try:
        found: Any = importlib.import_module(module)
        for part in attribute.split("."):
            found = getattr(found, part)
    except (ImportError, AttributeError) as error:
        raise TaskError(f"cannot load task {path!r}: {error}") from error
    return found


def _instance(found: Any, spec: str) -> Task:
    if isinstance(found, Task):
        return found
    if not (isinstance(found, type) and issubclass(found, Task)):
        raise TaskError(f"{spec!r} names {found!r}, which is not a Task")
    if found.__dict__.get("_builtin") is None and getattr(found, "name", None) in BUILTIN_TASK_CLASSES:
        raise TaskError(f"{spec!r} names a task called {found.name!r}, a built-in task's name")
    return found()


def load_task(spec: TaskLike = DEFAULT_TASK) -> Task:
    """Return the task ``spec`` names: a built-in, a registered class, an entry point or ``module:Class``;
    ``name@version`` checks the version."""
    if isinstance(spec, Task):
        return spec
    if isinstance(spec, type):
        return _instance(spec, spec.__qualname__)
    if not isinstance(spec, str) or not spec:
        raise TaskError(f"a task is named by a string, a Task or a Task subclass, got {spec!r}")
    name, _, version = spec.partition("@")
    if name in BUILTIN_TASK_CLASSES:
        task: Task = BUILTIN_TASK_CLASSES[name]()
    elif name in _REGISTERED:
        task = _REGISTERED[name]()
    elif ":" in name:
        task = _instance(_dotted(name), name)
    else:
        found = _entry_point(name)
        if found is None:
            raise TaskError(
                f"no task {name!r}: not built in ({', '.join(BUILTIN_TASK_CLASSES)}), registered, installed as a "
                f"{ENTRY_POINT_GROUP!r} entry point, or a dotted path package.module:Class")
        task = _instance(found, name)
    if ":" not in name and task.name != name:
        raise TaskError(f"{spec!r} found a task named {task.name!r}")
    if "@" in spec and task.version != version:
        raise TaskError(f"task {name!r} version {version!r} was asked for; the one installed is version "
                        f"{task.version!r}")
    return task


def task_locator(spec: TaskLike) -> str:
    """Return a string another process can load the task from: the name as given, or a class's ``module:qualname``."""
    if isinstance(spec, str):
        name, at, version = spec.partition("@")
        if name in BUILTIN_TASK_CLASSES or name not in _REGISTERED:
            return spec
        return f"{_class_path(_REGISTERED[name])}{at}{version}"
    return _class_path(spec if isinstance(spec, type) else type(spec))


def _class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def check_task_identity(task: Task, *, task_name: str, response_contract: Optional[str], prompt_style: str,
                        where: str) -> None:
    """Refuse a task whose identity or response contract differs from what a condition recorded."""
    if prompt_style not in task.styles:
        raise TaskError(f"{where}: the task loaded, {task.identifier!r}, has no prompt style {prompt_style!r}")
    contract = None if task.builtin else task.contract(prompt_style)
    if (task.identifier, contract) != (task_name, response_contract):
        raise TaskError(
            f"{where}: recorded task {task_name!r} (response contract {response_contract!r}), but the task loaded is "
            f"{task.identifier!r} (response contract {contract!r})")


@dataclass(frozen=True)
class TaskInfo:
    """One task found by name: its source, locator, identity, selectable styles, or the ``error`` loading it raised."""

    name: str
    source: str
    locator: str
    identifier: Optional[str] = None
    optimizable: Optional[bool] = None
    styles: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    default_style: Optional[str] = None
    error: Optional[str] = None


def _info(task: Task, *, source: str, locator: str) -> TaskInfo:
    return TaskInfo(name=task.name, source=source, locator=locator, identifier=task.identifier,
                    optimizable=isinstance(task, OptimizableTask),
                    styles=MappingProxyType({style: contract for style, contract in task.styles.items()
                                             if style not in task.record_only_styles}),
                    default_style=task.default_style)


def _listed(name: str, load: Callable[[], Task], *, source: str, locator: str) -> TaskInfo:
    try:
        task = load()
        if task.name != name:
            raise TaskError(f"it loads a task named {task.name!r}")
        return _info(task, source=source, locator=locator)
    except Exception as error:
        return TaskInfo(name=name, source=source, locator=locator, error=f"{type(error).__name__}: {error}")


def list_tasks() -> list[TaskInfo]:
    """Return every task found by name, in lookup order; one that fails to load is listed with its error."""
    found = [_info(cls(), source="built-in", locator=name) for name, cls in BUILTIN_TASK_CLASSES.items()]
    found += [_listed(name, cls, source="registered", locator=task_locator(name)) for name, cls in _REGISTERED.items()]
    seen = {info.name for info in found}
    for entry in sorted(metadata.entry_points(group=ENTRY_POINT_GROUP), key=lambda entry: entry.name):
        if entry.name in seen:
            continue
        seen.add(entry.name)
        found.append(_listed(entry.name, lambda entry=entry: _instance(entry.load(), entry.name),
                             source="entry point", locator=entry.name))
    return found
