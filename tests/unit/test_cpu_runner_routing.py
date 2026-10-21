"""Trusted CPU CI uses the dedicated pool without exposing it to fork PRs."""

from itertools import product
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
    assert gpu["concurrency"]["group"] == "flagquantum-accelerator-a100"
    commands = "\n".join(step.get("run", "") for step in gpu["steps"])
    assert "--nproc-per-node=8" in commands
    assert "statevector_forward_executor.py --backend nccl" in commands
    assert "statevector_reverse_executor.py --backend nccl" in commands


def test_trusted_cpu_jobs_and_matrix_legs_do_not_share_hosted_buckets() -> None:
    groups: set[str] = set()
    for workflow in ("ci.yml", "pre-commit.yaml"):
        jobs = yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"]
        for name, job in jobs.items():
            if "'flagquantum-a100-cpu'" not in str(job.get("runs-on", "")):
                continue
            selector = (
                "${{ (github.event_name != 'pull_request' || "
                "github.event.pull_request.head.repo.full_name == github.repository) "
                f"&& format('a100-cpu-{{0}}-{name}-{{1}}', "
                "github.run_id, strategy.job-index || 0) || 'gh' }}"
            )
            template = job["concurrency"]["group"]
            assert selector in template
            matrix = job.get("strategy", {}).get("matrix", {})
            rows = matrix.get("include") or [
                dict(zip(matrix, values, strict=True))
                for values in product(*matrix.values())
            ]
            for run_id in (100, 101):
                for index, row in enumerate(rows):
                    group = template.replace(
                        selector, f"a100-cpu-{run_id}-{name}-{index}"
                    )
                    group = group.replace(
                        "${{ github.event.pull_request.number || github.ref_name }}",
                        "659",
                    )
                    for key, value in row.items():
                        group = group.replace("${{ matrix." + key + " }}", str(value))
                    assert "${{" not in group
                    assert group not in groups, f"Unrelated CPU work shares {group}"
                    groups.add(group)
    assert len(groups) > 32, "Exercise more jobs than the 16-runner CPU pool"


def test_dependency_matrix_seeds_its_requested_torch_version() -> None:
    document = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
    steps = document["jobs"]["dependency-bounds"]["steps"]
    isolated = next(
        step
        for step in steps
        if step.get("uses") == "./.github/actions/isolated-python"
    )
    assert isolated["with"]["torch-spec"] == "${{ matrix.torch }}"
