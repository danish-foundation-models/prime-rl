import importlib.util
from pathlib import Path


def _cleanup_module():
    script = Path(__file__).parents[3] / "scripts" / "cleanup_ucloud_sandboxes.py"
    spec = importlib.util.spec_from_file_location("cleanup_ucloud_sandboxes", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ucloud_labels_reads_legacy_v0_env_args(tmp_path):
    config = tmp_path / "orchestrator.toml"
    config.write_text(
        """
[orchestrator]

[[orchestrator.train.env]]
name = "legacy"

[orchestrator.train.env.args]
sandbox_backend = "ucloud"
labels = ["legacy-label"]
""".strip()
    )

    cleanup = _cleanup_module()

    assert cleanup._ucloud_labels(config) == ["legacy-label"]


def test_ucloud_labels_reads_native_v1_harness_runtime(tmp_path):
    config = tmp_path / "orchestrator.toml"
    config.write_text(
        """
[orchestrator]

[[orchestrator.train.env]]
name = "native"

[orchestrator.train.env.harness.runtime]
type = "ucloud"
labels = ["native-label"]
""".strip()
    )

    cleanup = _cleanup_module()

    assert cleanup._ucloud_labels(config) == ["native-label"]
