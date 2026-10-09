"""Standard task iteration."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.samplers.base import TaskSampler


class StandardSampler(TaskSampler):
    """Advance the task iterator, cycling finite tasksets in source order."""

    def __init__(self, tasks: Sequence[vf.Task] | Iterator[vf.Task]) -> None:
        self.tasks = tuple(tasks) if isinstance(tasks, Sequence) else None
        if self.tasks is not None:
            if not self.tasks:
                raise ValueError("A finite curriculum needs at least one task")
            keys = [task.key for task in self.tasks]
            duplicates = {key for key, count in Counter(keys).items() if count > 1}
            if duplicates:
                raise ValueError(f"Task keys must be unique within a taskset: {sorted(duplicates)}")
            self.task_iterator = itertools.cycle(self.tasks)
        else:
            self.task_iterator = tasks
        self.task_keys = tuple(keys) if self.tasks is not None else ()
        self.excluded_keys: set[str] = set()
        self.cursor = 0

    def __next__(self) -> vf.Task:
        if not self.has_available_tasks:
            raise StopIteration
        while True:
            index = self.cursor % len(self.tasks) if self.tasks is not None else None
            task = next(self.task_iterator)
            self.cursor += 1
            key = self.task_keys[index] if index is not None else task.key
            if key not in self.excluded_keys:
                return task

    @property
    def has_available_tasks(self) -> bool:
        return self.tasks is None or len(self.excluded_keys) < len(self.tasks)

    def exclude_tasks(self, task_keys: set[str]) -> None:
        self.excluded_keys.update(task_keys if self.tasks is None else task_keys & set(self.task_keys))

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        cursor = state_dict["cursor"]
        if self.tasks is not None:
            self.task_iterator = itertools.cycle(self.tasks)
            offset = cursor % len(self.tasks)
        else:
            offset = cursor
        for _ in range(offset):
            next(self.task_iterator)
        self.cursor = cursor
