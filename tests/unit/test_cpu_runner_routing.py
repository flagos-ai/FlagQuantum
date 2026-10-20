"""Trusted CPU CI uses the dedicated pool without exposing it to fork PRs."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"


@pytest.mark.parametrize("workflow", ["ci.yml", "pre-commit.yaml"])
def test_cpu_pool_routes_require_a_same_repository_pull_request(workflow: str) -> None:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    routed = [
        job
        for job in document["jobs"].values()
        if "'flagquantum-a100-cpu'" in str(job.get("runs-on", ""))
    ]
    assert routed, "The policy must discover at least one CPU-pool job"
    expected = (
        "${{ (github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository) "
        "&& 'flagquantum-a100-cpu' || 'ubuntu-latest' }}"
    )
    for job in routed:
        assert job["runs-on"] == expected
        steps = job["steps"]
        for index, step in enumerate(steps):
            if str(step.get("uses", "")).startswith("actions/setup-python@"):
                assert "cache" not in step.get("with", {})
                assert steps[index + 1].get("uses") == (
                    "./.github/actions/isolated-python"
                )
    assert "pull_request_target" not in (document.get("on") or document.get(True, {}))


def test_pool_validation_covers_every_cpu_slot_and_eight_gpu_ranks() -> None:
    document = yaml.safe_load((WORKFLOWS / "local-gpu.yml").read_text())
    cpu = document["jobs"]["validate-cpu-pool"]
    assert cpu["strategy"]["matrix"]["slot"] == [
        f"cpu{index:02d}" for index in range(1, 17)
    ]
    assert "${{ matrix.slot }}" in cpu["runs-on"]
    gpu = document["jobs"]["validate-eight-gpus"]
    assert "flagquantum-a100" in gpu["runs-on"]
    assert gpu["concurrency"]["group"] == "flagquantum-accelerator"
    commands = "\n".join(step.get("run", "") for step in gpu["steps"])
    assert "--nproc-per-node=8" in commands
    assert "statevector_forward_executor.py --backend nccl" in commands
    assert "statevector_reverse_executor.py --backend nccl" in commands
