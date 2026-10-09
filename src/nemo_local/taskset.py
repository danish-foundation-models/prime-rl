"""Native verifiers tasks backed by staged, revision-pinned JSONL pools."""

import asyncio
import atexit
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.dialects.responses import ResponsesDialect

from .grading import grade

Kind = Literal["pivot", "instruction", "calendar", "workplace"]


@lru_cache(maxsize=1)
def grading_pool():
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
    atexit.register(pool.shutdown, wait=False, cancel_futures=True)
    return pool


class NeMoLocalData(vf.TaskData):
    kind: Kind
    row: dict
    source_id: str


class NeMoLocalTaskConfig(vf.TaskConfig):
    pass


class NeMoLocalTask(vf.Task[NeMoLocalData, vf.State, NeMoLocalTaskConfig]):
    @property
    def key(self):
        return f"nemo-local:{self.data.kind}:{self.data.source_id}"

    @vf.reward(weight=1.0)
    async def correctness(self, trace: vf.Trace) -> float:
        calls = [
            {"name": call.name, "arguments": call.arguments}
            for message in trace.assistant_messages
            for call in message.tool_calls or []
        ]
        reward, details = await asyncio.get_running_loop().run_in_executor(
            grading_pool(),
            grade,
            self.data.kind,
            self.data.row,
            trace.last_reply,
            calls,
        )
        trace.info["nemo_local"] = {"task_reward": reward, **details}
        return reward


class NeMoLocalConfig(vf.TasksetConfig):
    kind: Kind
    dataset: Path
    task: NeMoLocalTaskConfig = NeMoLocalTaskConfig()


class NeMoLocalTaskset(vf.Taskset[NeMoLocalTask, NeMoLocalConfig]):
    def load(self):
        dialect = ResponsesDialect()
        with self.config.dataset.open() as file:
            for idx, line in enumerate(file):
                if not line.strip():
                    continue
                row = json.loads(line)
                request = dialect.parse_request(row["responses_create_params"])
                source_id = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                messages = request.messages
                if self.config.kind == "pivot":
                    # The dataset supervises exactly one next action, not an arbitrary batch.
                    from verifiers.v1.types import SystemMessage, content_text

                    instruction = "Produce exactly one next tool call for this step."
                    if messages and isinstance(messages[0], SystemMessage):
                        system = messages[0].model_copy(
                            update={
                                "content": content_text(messages[0].content) + "\n\n" + instruction,
                            }
                        )
                        messages = [system, *messages[1:]]
                    else:
                        messages = [SystemMessage(content=instruction), *messages]
                yield NeMoLocalTask(
                    NeMoLocalData(
                        idx=idx,
                        name=f"{self.config.kind}:{idx}",
                        kind=self.config.kind,
                        prompt=messages,
                        row=row,
                        source_id=source_id,
                    ),
                    self.config.task,
                )
