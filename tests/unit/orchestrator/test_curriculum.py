import uuid
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import (
    AdvRangeGateConfig,
    CurriculumConfig,
    DifficultyPoolConfig,
    DifficultyPoolSamplerConfig,
)
from prime_rl.orchestrator.curriculum import (
    AdvRangeGate,
    Curriculum,
    StandardSampler,
)
from prime_rl.orchestrator.train_source import TrainSource


def make_task(idx: int) -> vf.Task:
    return vf.Task(vf.TaskData(idx=idx, prompt=f"task {idx}"))


def make_rollout(
    task: vf.Task,
    *,
    env_name: str = "test",
    reward: float = 0.0,
    advantages: list[float] | None = None,
) -> list[vf.Episode]:
    nodes = (
        [
            vf.MessageNode(
                message=vf.AssistantMessage(content="answer"),
                token_ids=list(range(len(advantages))),
                mask=[True] * len(advantages),
                logprobs=[0.0] * len(advantages),
                advantages=advantages,
                sampled=True,
            )
        ]
        if advantages is not None
        else []
    )
    trace = vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__,
            data=task.data,
            key=task.key,
            hash=task.hash,
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=reward)},
        ok=True,
    )
    episode = vf.Episode(
        env=vf.EnvInfo(id=env_name, name=env_name),
        task=trace.task,
        group=vf.GroupInfo(id=str(uuid.uuid4())),
        traces=[trace],
    )
    return [episode]


def test_default_curriculum_resumes_finite_and_infinite_tasksets() -> None:
    tasks = [make_task(i) for i in range(5)]
    finite = StandardSampler(tasks)
    for _ in range(3):
        next(finite)
    state = finite.state_dict()
    expected = [next(finite).key for _ in range(4)]

    restored = StandardSampler(tasks)
    restored.load_state_dict(state)
    assert [next(restored).key for _ in range(4)] == expected

    def task_stream() -> Iterator[vf.Task]:
        yield from (make_task(i) for i in range(10))

    infinite = StandardSampler(task_stream())
    next(infinite)
    next(infinite)
    restored_infinite = StandardSampler(task_stream())
    restored_infinite.load_state_dict(infinite.state_dict())
    assert next(restored_infinite).key == make_task(2).key


def test_curriculum_requires_unique_finite_task_keys() -> None:
    task = make_task(0)
    with pytest.raises(ValueError, match="Task keys must be unique"):
        StandardSampler([task, task])


def test_train_source_composes_sampler_and_all_gates_with_state_and_metrics() -> None:
    tasks = [make_task(i) for i in range(3)]
    pools = {"all": DifficultyPoolConfig(threshold=1.0, weight=1.0)}
    config = SimpleNamespace(
        ratio=1.0,
        curriculum=CurriculumConfig(
            sampler=DifficultyPoolSamplerConfig(pools=pools),
            gates={
                "reject": AdvRangeGateConfig(),
                "admit": AdvRangeGateConfig(reject_min=0.5, reject_max=0.5),
            },
        ),
    )
    env = SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)
    source = TrainSource([env])

    sampled = source.next_task(step=1).task
    assert source.on_result(make_rollout(sampled, reward=0.25, advantages=[0.0])) is False
    assert source.metrics() == {
        "curriculum/test/admission_rate": 0.0,
        "curriculum/test/sampler/pool/unseen": 2.0,
        "curriculum/test/sampler/pool/all": 1.0,
    }

    state = source.state_dict()
    restored = TrainSource([SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)])
    restored.load_state_dict(state)
    assert restored.curricula["test"].state_dict()["sampler"]["task_rewards"] == {sampled.key: 0.25}
    assert restored.curricula["test"].state_dict()["gates"] == {
        "reject": {},
        "admit": {},
    }


def test_difficulty_pools_stack_with_advantage_gate_and_resume_sampling() -> None:
    tasks = [make_task(i) for i in range(3)]
    pools = {
        "hard": DifficultyPoolConfig(threshold=0.25, weight=0.0),
        "normal": DifficultyPoolConfig(threshold=0.75, weight=1.0),
        "easy": DifficultyPoolConfig(threshold=1.0, weight=0.0),
    }
    sampler_config = DifficultyPoolSamplerConfig(pools=pools, seed=7)
    gate_config = AdvRangeGateConfig()
    config = CurriculumConfig(
        sampler=sampler_config,
        gates={"zero_advantage": gate_config},
    )
    curriculum = Curriculum(config, tasks)
    rewards = {0: 0.1, 1: 0.5, 2: 0.9}
    decisions = []
    for index, task in enumerate(tasks):
        group = make_rollout(task, reward=rewards[task.data.idx], advantages=[float(index > 0)])
        decisions.append(curriculum.on_result(group))

    assert decisions == [False, True, True]
    assert curriculum.metrics() == {
        "sampler/pool/unseen": 0.0,
        "sampler/pool/hard": 1.0,
        "sampler/pool/normal": 1.0,
        "sampler/pool/easy": 1.0,
    }
    state = curriculum.state_dict()
    expected = [next(curriculum.sampler).key for _ in range(10)]
    assert set(expected) == {tasks[1].key}
    restored = Curriculum(config, tasks)
    restored.load_state_dict(state)
    assert [next(restored.sampler).key for _ in range(10)] == expected


def test_advantage_range_gate_generalizes_zero_advantage_rejection() -> None:
    task = make_task(0)
    zero_gate = AdvRangeGate(AdvRangeGateConfig())
    assert zero_gate.admit(make_rollout(task, advantages=[0.0, 0.0])) is False
    assert zero_gate.admit(make_rollout(task, advantages=[0.0, 0.2])) is True
    assert zero_gate.admit(make_rollout(task)) is True

    tolerance_gate = AdvRangeGate(AdvRangeGateConfig(reject_min=-0.1, reject_max=0.1))
    assert tolerance_gate.admit(make_rollout(task, advantages=[-0.05, 0.0, 0.05])) is False

    positive_gate = AdvRangeGate(AdvRangeGateConfig(reject_min=0.5, reject_max=0.5))
    assert positive_gate.admit(make_rollout(task, advantages=[0.5])) is False


@pytest.mark.parametrize("component", ["ce", "ref_kl"])
def test_advantage_gate_preserves_auxiliary_supervision(component):
    group = make_rollout(make_task(0), advantages=[0.0, 0.0])
    node = group[0].traces[0].nodes[0]
    gate = AdvRangeGate(AdvRangeGateConfig())
    node.loss_weights = {component: [0.0, 0.02]}
    assert gate.admit(group)
    node.loss_weights = {component: [0.0, 0.0]}
    assert not gate.admit(group)
    node.loss_weights = {component: [0.0, 0.02]}
    # Weights on a context-only branch must not admit an otherwise empty group.
    node.mask = [False, False]
    assert not gate.admit(group)


@pytest.mark.parametrize("difficulty", [False, True])
def test_image_quarantine_spans_harnesses_and_survives_resume(tmp_path, difficulty):
    class ImageTaskData(vf.TaskData):
        image: str

    tasks = [vf.Task(ImageTaskData(idx=i, prompt=f"task {i}", image="bad" if i < 2 else "good"))
             for i in range(3)]
    config = CurriculumConfig(
        sampler=DifficultyPoolSamplerConfig(pools={"all": DifficultyPoolConfig(threshold=1.0, weight=1.0)})
    ) if difficulty else CurriculumConfig()

    def envs():
        return [SimpleNamespace(name=name, tasks=iter(tasks), num_tasks=3, config=SimpleNamespace(
            ratio=1.0, curriculum=config,
            env=SimpleNamespace(agent=SimpleNamespace(runtime=SimpleNamespace(image_recipe_db=db))),
        )) for name, db in [("bash", "/recipes/v1"), ("pi", "/recipes/v1"), ("repaired", "/recipes/v2")]]

    journal = tmp_path / "quarantined-images.json"
    source = TrainSource(envs(), quarantine_path=journal)
    assert source.quarantine_image("bash", tasks[0], "confirmed image failure")
    assert not source.quarantine_image("pi", tasks[1], "same image")
    assert source.is_quarantined("pi", tasks[1])
    assert not source.is_quarantined("repaired", tasks[0])
    for _ in range(20):
        assert source.next_task(step=1, allowed={"bash", "pi"}).task is tasks[2]
    assert source.curricula["bash"].sampler.state_dict().get("task_rewards", {}) == {}
    state = source.state_dict()
    restored = TrainSource(envs())
    restored.load_state_dict(state)
    assert restored.next_task(step=1, allowed={"pi"}).task is tasks[2]
    journal_restored = TrainSource(envs(), quarantine_path=journal)
    assert journal_restored.next_task(step=1, allowed={"bash"}).task is tasks[2]
    assert restored.quarantine_image("bash", tasks[2], "another image failure")
    assert restored.next_task(step=1, allowed={"bash", "pi"}) is None
    assert restored.next_task(step=1).env_name == "repaired"
    assert restored.metrics()["curriculum/pi/unavailable_tasks"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["ImageBuildFailure", "SandboxError"])
async def test_dispatcher_image_failure_cancels_only_matching_groups(error_type):
    import asyncio
    from prime_rl.orchestrator.dispatcher import Dispatcher
    from prime_rl.orchestrator.envs import Envs
    from prime_rl.orchestrator.types import GroupCancellation, GroupState, InflightEpisode, Policy, Progress

    class ImageTaskData(vf.TaskData):
        image: str

    tasks = [vf.Task(ImageTaskData(idx=i, prompt=f"task {i}", image=image))
             for i, image in enumerate(["bad", "good"])]
    envs = [SimpleNamespace(name=name, tasks=iter(tasks), num_tasks=2,
                           config=SimpleNamespace(group_size=8, ratio=1.0, curriculum=CurriculumConfig()))
            for name in ["bash", "pi"]]
    source = TrainSource(envs)
    collection = Envs()
    collection._envs = {env.name: env for env in envs}
    for env in envs:
        env.generation_source = SimpleNamespace(uses_live_policy=True)
    dispatcher = Dispatcher(train_envs=collection, eval_envs=None, train_source=source, eval_source=None,
                            policy_clients=None, policy=Policy(), progress=Progress(),
                            initial_max_inflight=64, max_inflight_ceiling=64, tasks_per_minute=None,
                            max_off_policy_steps=5, run_id="test", run_name="test")
    gids = [uuid.uuid4() for _ in range(3)]
    episode = make_rollout(tasks[0], reward=0.0, env_name="bash")[0]
    episode.ok = False
    episode.traces[0].ok = False
    episode.traces[0].errors.append(vf.Error(type=error_type, message="failure"))

    async def completed():
        return episode

    async def pending():
        await asyncio.Event().wait()

    current = asyncio.create_task(completed())
    await current
    for index, (name, task) in enumerate([("bash", tasks[0]), ("pi", tasks[0]), ("bash", tasks[1])]):
        gid = gids[index]
        dispatcher.groups[gid] = GroupState(kind="train", env_name=name, task=task, step=1,
                                            episodes_to_schedule=0, target_episodes=8)
        for member in range(8):
            request = current if index == 0 and member == 0 else asyncio.create_task(pending())
            dispatcher.inflight[request] = InflightEpisode(kind="train", env_name=name, task=task,
                                                           group_id=gid, policy_version=0, step=1)
            dispatcher.current_inflight += 1
    try:
        await dispatcher.handle_completed_request(current)
        emitted = dispatcher.out_q.get_nowait()
        assert emitted is episode
        if error_type == "ImageBuildFailure":
            cancellations = [dispatcher.out_q.get_nowait(), dispatcher.out_q.get_nowait()]
            assert all(isinstance(item, GroupCancellation) and item.reason == "unavailable_task"
                       for item in cancellations)
            assert [item.count for item in cancellations] == [7, 8]
            assert dispatcher.current_inflight == 8
            assert set(dispatcher.groups) == {gids[2]}
            assert source.is_quarantined("pi", tasks[0])
        else:
            assert dispatcher.current_inflight == 23
            assert len(dispatcher.groups) == 3
            assert not source.quarantined_images
        assert dispatcher.out_q.empty()
    finally:
        await dispatcher.cancel_inflight_episodes()


def test_smoothed_task_difficulty_shares_evidence_and_resumes_without_double_counting():
    from verifiers.v1.trace import Error

    tasks = [make_task(i) for i in range(40)]

    def envs():
        result = []
        for seed, name in enumerate(("bash", "pi", "separate")):
            sampler = DifficultyPoolSamplerConfig(
                strategy="smoothed_replay", shared_group="same-taskset" if name != "separate" else "other",
                seed=seed, history_size=16,
            )
            config = SimpleNamespace(ratio=1, curriculum=CurriculumConfig(sampler=sampler))
            result.append(SimpleNamespace(name=name, tasks=iter(tasks), num_tasks=len(tasks), config=config))
        return result

    source = TrainSource(envs())
    group = [episode for reward in [1.0, 1.0, 0, 0, 0, 0, 0, 0]
             for episode in make_rollout(tasks[0], env_name="bash", reward=reward)]
    source.on_result(group)
    key = tasks[0].key
    for name in ("bash", "pi"):
        sampler = source.curricula[name].sampler
        assert sampler.task_rewards[key] == pytest.approx(0.3)
        assert len(sampler.task_history[key]) == 8
    assert not source.curricula["separate"].sampler.task_rewards

    failed = make_rollout(tasks[0], env_name="pi")
    failed[0].traces[0].errors = [Error(type="SandboxError", message="unavailable")]
    failed[0].traces[0].ok = False
    source.on_result(failed)
    assert len(source.curricula["bash"].sampler.task_history[key]) == 8

    source.on_result([episode for _ in range(8) for episode in make_rollout(tasks[0], env_name="pi", reward=1)])
    assert source.curricula["bash"].sampler.task_rewards[key] == pytest.approx(11 / 18)
    state = source.state_dict()
    expected = [source.next_task(step=2).task.key for _ in range(20)]
    restored = TrainSource(envs())
    restored.load_state_dict(state)
    assert [restored.next_task(step=2).task.key for _ in range(20)] == expected
    restored.on_result([episode for _ in range(8) for episode in make_rollout(tasks[0], env_name="bash")])
    for name in ("bash", "pi"):
        assert restored.curricula[name].sampler.task_rewards[key] == pytest.approx(0.5)
        assert len(restored.curricula[name].sampler.task_history[key]) == 16


def test_smoothed_replay_keeps_exploration_and_does_not_let_one_task_dominate():
    from prime_rl.orchestrator.curriculum.samplers import DifficultyPoolSampler

    tasks = [make_task(i) for i in range(1000)]
    sampler = DifficultyPoolSampler(DifficultyPoolSamplerConfig(strategy="smoothed_replay", seed=7, replay_fraction=0.2), tasks)
    sampler.observe(make_rollout(tasks[0], reward=0.5))
    first = [next(sampler).key for _ in range(2000)]
    assert 5 < first.count(tasks[0].key) < 55  # 0.2 / 16 replay ramp.
    assert len(set(first)) > 500
    for task in tasks[1:16]:
        sampler.observe(make_rollout(task, reward=0.5))
    known = {task.key for task in tasks[:16]}
    draws = [next(sampler).key for _ in range(2000)]
    assert 300 < sum(key in known for key in draws) < 500
    sampler.exclude_tasks(known)
    assert all(next(sampler).key not in known for _ in range(100))
