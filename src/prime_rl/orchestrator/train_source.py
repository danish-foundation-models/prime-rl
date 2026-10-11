"""Training source selection and curriculum lifecycle."""

from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum import Curriculum
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.types import TaskRequest
from prime_rl.orchestrator.utils import episode_env_name
from prime_rl.utils.logger import get_logger


class TrainSource:
    """Mix train envs and host one user-authored curriculum per env."""

    def __init__(self, train_envs: TrainEnvs, *, scheduler=None, quarantine_path: Path | None = None) -> None:
        self.scheduler = scheduler
        self.rng = random.Random(42)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.curricula: dict[str, Curriculum] = {}
        # Shared read-only task objects need their content identity computed only once.
        task_key_cache: dict[int, str] = {}
        for env in self.envs:
            if env.tasks is None:
                raise RuntimeError(f"env {env.name} not started")
            tasks = env.tasks if env.num_tasks is None else list(env.tasks)
            self.curricula[env.name] = Curriculum(env.config.curriculum, tasks, task_key_cache=task_key_cache)

        self._shared_difficulty: dict[str, list] = defaultdict(list)
        for curriculum in self.curricula.values():
            sampler = curriculum.sampler
            if getattr(sampler, "strategy", None) == "smoothed_replay" and sampler.shared_group:
                peers = self._shared_difficulty[sampler.shared_group]
                if peers and sampler.history_size != peers[0].history_size:
                    raise ValueError("Shared task difficulty requires matching history_size")
                peers.append(sampler)

        self.env_names = [env.name for env in self.envs]
        self.weights = [float(env.config.ratio) for env in self.envs]
        self._admitted: dict[str, int] = defaultdict(int)
        self._rejected: dict[str, int] = defaultdict(int)
        self.quarantine_path = quarantine_path
        self.quarantined_images: dict[tuple[str, str], str] = {}
        self.env_by_name = {env.name: env for env in self.envs}
        if quarantine_path is not None and quarantine_path.exists():
            self._restore_quarantine(json.loads(quarantine_path.read_text()))

    def next_task(self, *, step: int, allowed: set[str] | None = None) -> TaskRequest | None:
        selected = [
            (name, weight) for name, weight in zip(self.env_names, self.weights, strict=True)
            if (allowed is None or name in allowed) and self.curricula[name].sampler.has_available_tasks
        ]
        if not selected:
            return None
        names, weights = zip(*selected, strict=True)
        env_name = self.rng.choices(names, weights=weights, k=1)[0]
        # Finite samplers already exclude these tasks; bound rejection for streams.
        for _ in range(1024):
            task = next(self.curricula[env_name].sampler)
            if not self.is_quarantined(env_name, task):
                return TaskRequest(env_name=env_name, task=task, step=step)
        return None

    def upcoming_tasks(self, count: int) -> dict[str, list]:
        """Each env's next unseen tasks, where its sampler knows them ahead of sampling."""
        upcoming = {}
        for name, curriculum in self.curricula.items():
            peek = getattr(curriculum.sampler, "upcoming", None)
            if peek is not None:
                upcoming[name] = peek(count)
        return upcoming

    def image_identity(self, env_name: str, task) -> tuple[str, str] | None:
        config = getattr(self.env_by_name[env_name].config, "env", None)
        runtime = getattr(getattr(config, "agent", None), "runtime", None)
        image = getattr(task.data, "image", None) or getattr(runtime, "image", None)
        if not image:
            return None
        return str(getattr(runtime, "image_recipe_db", None) or ""), str(image)

    def is_quarantined(self, env_name: str, task) -> bool:
        return self.image_identity(env_name, task) in self.quarantined_images

    def quarantine_image(self, env_name: str, task, reason: str) -> bool:
        identity = self.image_identity(env_name, task)
        if identity is None or identity in self.quarantined_images:
            return False
        self._exclude_image(identity, reason)
        if self.quarantine_path is not None:
            self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.quarantine_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._quarantine_records(), indent=2))
            temporary.replace(self.quarantine_path)
        return True

    def _exclude_image(self, identity: tuple[str, str], reason: str) -> None:
        self.quarantined_images[identity] = reason
        affected = {}
        for name, curriculum in self.curricula.items():
            sampler = curriculum.sampler
            tasks = getattr(sampler, "tasks", None)
            if tasks is None:
                continue
            keys = {key for key, task in zip(sampler.task_keys, tasks, strict=True)
                    if self.image_identity(name, task) == identity}
            if keys:
                sampler.exclude_tasks(keys)
                affected[name] = len(keys)
        get_logger().warning(
            f"Quarantined task image {identity[1]} (recipe index {identity[0]}): {reason}; affected={affected}"
        )

    def _quarantine_records(self) -> list[dict[str, str]]:
        return [{"recipe_db": key[0], "image": key[1], "reason": reason}
                for key, reason in sorted(self.quarantined_images.items())]

    def _restore_quarantine(self, records) -> None:
        for record in records:
            self._exclude_image((record["recipe_db"], record["image"]), record["reason"])

    def on_result(self, group: list[vf.Episode]) -> bool:
        """Report a finalized group and return whether it should train."""
        if not group:
            raise ValueError("Cannot report an empty rollout group")
        env_name = episode_env_name(group[0])
        admitted = self.curricula[env_name].on_result(group)
        sampler = self.curricula[env_name].sampler
        for peer in self._shared_difficulty.get(getattr(sampler, "shared_group", None), []):
            if peer is not sampler:
                peer.observe(group)
        if not isinstance(admitted, bool):
            raise TypeError(f"Curriculum.on_result() must return bool, got {type(admitted).__name__}")
        if admitted:
            self._admitted[env_name] += 1
        else:
            self._rejected[env_name] += 1
        return admitted

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for env_name, curriculum in self.curricula.items():
            admitted = self._admitted.pop(env_name, 0)
            rejected = self._rejected.pop(env_name, 0)
            total = admitted + rejected
            if total:
                metrics[f"curriculum/{env_name}/admission_rate"] = admitted / total
            metrics |= {f"curriculum/{env_name}/{name}": float(value) for name, value in curriculum.metrics().items()}
        if self.quarantined_images:
            metrics["curriculum/quarantined_images"] = float(len(self.quarantined_images))
            for name, curriculum in self.curricula.items():
                metrics[f"curriculum/{name}/unavailable_tasks"] = float(len(curriculum.sampler.excluded_keys))
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "quarantined_images": self._quarantine_records(),
            "envs": {name: curriculum.state_dict() for name, curriculum in self.curricula.items()},
            **({"scheduler": self.scheduler.state_dict()} if self.scheduler is not None else {}),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected_fields = {"rng", "envs"}
        if set(state_dict) - {"scheduler", "quarantined_images"} != expected_fields:
            raise ValueError(f"Train-source checkpoint fields must be {sorted(expected_fields)}")
        env_states = state_dict["envs"]
        if set(env_states) != set(self.curricula):
            raise ValueError(
                f"Train-source checkpoint envs {sorted(env_states)} do not match configured envs "
                f"{sorted(self.curricula)}"
            )
        if "scheduler" in state_dict:
            if self.scheduler is None:
                raise ValueError("Checkpoint requires throughput_balancing")
            self.scheduler.load_state_dict(state_dict["scheduler"])
        self.rng.setstate(state_dict["rng"])
        for name, curriculum in self.curricula.items():
            curriculum.load_state_dict(env_states[name])
        self._restore_quarantine(state_dict.get("quarantined_images", []))
