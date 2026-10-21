import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.shared import ControlConfig
from prime_rl.control.controller import Controller
from prime_rl.control.server import ControlServer
from prime_rl.entrypoints.sft import write_slurm_script

TEMPLATES = Path(__file__).parents[3] / "src" / "prime_rl" / "templates"


def render_helper(pause_before_time_limit: int | None) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), keep_trailing_newline=True)
    return env.get_template("_pause_before_time_limit.sh.j2").render(pause_before_time_limit=pause_before_time_limit)


def run_script(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, env=env)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("api_delay", [0.0, 2.0], ids=["api_up", "signal_during_startup"])
def test_time_limit_signal_pauses_trainer_and_exits_cleanly(api_delay):
    controller = Controller(requires_step=False)
    port = free_port()
    server = ControlServer(controller, ControlConfig(host="127.0.0.1", port=port))

    def trainer():
        # Stand-in for the trainer: serve the API after startup, then apply the pause at
        # the next step and save it.
        time.sleep(api_delay)
        server.start()
        while controller.pending is None:
            time.sleep(0.05)
        assert controller.update(7)
        controller.mark_saved(7)

    thread = threading.Thread(target=trainer, daemon=True)
    thread.start()
    script = f"""
set -e
export CONTROL_URL=http://127.0.0.1:{port}
{render_helper(60)}
sleep 60 &
MAIN_PID=$!
( sleep 0.5; kill -USR1 $$ ) &
wait_main
exit $EXIT_CODE
"""
    start = time.monotonic()
    result = run_script(script)
    thread.join(timeout=5)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "saved step 7" in result.stdout
    assert time.monotonic() - start < 30


@pytest.mark.parametrize("pause_before_time_limit", [None, 60])
def test_wait_main_propagates_exit_code(pause_before_time_limit):
    script = f"""
set -e
{render_helper(pause_before_time_limit)}
bash -c 'sleep 0.2; exit 3' &
MAIN_PID=$!
wait_main
exit $EXIT_CODE
"""
    assert run_script(script).returncode == 3


def sft_config(
    tmp_path: Path, deployment: dict, template: str | None, pause_before_time_limit: int | None
) -> SFTConfig:
    slurm = {"time": "01:00:00", "pause_before_time_limit": pause_before_time_limit}
    if template is not None:
        slurm["template_path"] = str(TEMPLATES / template)
    return SFTConfig.model_validate({"deployment": deployment, "slurm": slurm, "output_dir": str(tmp_path)})


@pytest.mark.parametrize(
    ("deployment", "template"),
    [
        ({"type": "single_node"}, None),
        ({"type": "multi_node", "num_train_nodes": 2}, None),
        ({"type": "single_node"}, "lumi/single_node_sft_lumi.sbatch.j2"),
        ({"type": "multi_node", "num_train_nodes": 2}, "lumi/multi_node_sft_lumi.sbatch.j2"),
    ],
    ids=["single_node", "multi_node", "lumi_single_node", "lumi_multi_node"],
)
@pytest.mark.parametrize("pause_before_time_limit", [None, 600])
def test_sft_templates_render(tmp_path, deployment, template, pause_before_time_limit):
    config = sft_config(tmp_path, deployment, template, pause_before_time_limit)
    assert (config.control is not None) == (pause_before_time_limit is not None)

    script_path = tmp_path / "job.sbatch"
    config_path = tmp_path / "configs" / "sft.toml"
    write_slurm_script(config, config_path, tmp_path / "logs", script_path, "run-id")
    script = script_path.read_text()

    assert subprocess.run(["bash", "-n", str(script_path)]).returncode == 0
    assert ("#SBATCH --signal=B:USR1@600" in script) == (pause_before_time_limit is not None)
    assert ("CONTROL_URL=" in script) == (pause_before_time_limit is not None)
    assert script.rstrip().endswith("exit $EXIT_CODE")
