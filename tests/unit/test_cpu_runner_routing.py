"""Trusted CPU CI uses the dedicated pool without exposing it to fork PRs."""

from itertools import product
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"


def _inventoried_r300_labels() -> set[str]:
    validation = yaml.safe_load((WORKFLOWS / "r300-validation.yml").read_text())
    return set(validation["jobs"]["resources"]["strategy"]["matrix"]["scale_set"])


@pytest.mark.parametrize("workflow", ["ci.yml", "pre-commit.yaml"])
def test_cpu_pool_routes_require_a_same_repository_pull_request(workflow: str) -> None:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    routed = [
        job
        for job in document["jobs"].values()
        if any(
            label in str(job.get("runs-on", ""))
            for label in (
                "flagquantum-a100-cpu",
                "flagquantum-cpu-general",
                "flagcicd-r300-",
            )
        )
    ]
    assert routed, "The policy must discover at least one CPU-pool job"
    expected = (
        "${{ (github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository) "
        "&& 'flagquantum-a100-cpu' || 'ubuntu-latest' }}"
    )
    for job in routed:
        assert job["runs-on"] in (
            expected,
            expected.replace("flagquantum-a100-cpu", "flagquantum-cpu-general"),
            *(
                expected.replace("flagquantum-a100-cpu", label)
                for label in _inventoried_r300_labels()
            ),
        )
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
            if not any(
                label in str(job.get("runs-on", ""))
                for label in (
                    "flagquantum-a100-cpu",
                    "flagquantum-cpu-general",
                    "flagcicd-r300-",
                )
            ):
                continue
            prefix = "self-hosted" if "flagcicd-r300-" in job["runs-on"] else "a100-cpu"
            selector = (
                "${{ (github.event_name != 'pull_request' || "
                "github.event.pull_request.head.repo.full_name == github.repository) "
                f"&& format('{prefix}-{{0}}-{name}-{{1}}', "
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


def test_dependency_matrix_separates_cpu_wheels_from_ordinary_dependencies() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    steps = jobs["dependency-bounds"]["steps"]
    installation = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses") == "./.github/actions/install-cpu-torch"
    )
    assert steps[installation]["with"]["torch-spec"] == "${{ matrix.torch }}"
    assert "--upgrade pip" in steps[installation - 1]["run"]
    assert len(jobs["dependency-bounds"]["strategy"]["matrix"]["include"]) == 7


def test_portable_cpu_work_uses_general_or_validated_scale_sets() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    portable = {
        "jax-optional",
        "azure-optional",
        "triton-optional",
        "qiskit-optional",
        "cirq-optional",
        "braket-optional",
        "pennylane-optional",
        "dependency-bounds",
        "supply-chain",
        "cpu-core-3-12",
        "ci-complete",
    }
    for name in portable:
        job = jobs[name]
        assert any(
            f"'{label}'" in job["runs-on"]
            for label in {"flagquantum-cpu-general", *_inventoried_r300_labels()}
        ), name
        assert "flagquantum-a100-cpu" not in job["runs-on"], name
        if "flagcicd-r300-" not in job["runs-on"]:
            assert not any(
                str(step.get("uses", "")).startswith("actions/checkout@")
                for step in job["steps"]
            ), f"{name} must not need A800's unavailable HTTPS checkout"
    # Long suites retain the measured A800 service budget; environment checks
    # above remain portable across the general pool.
    for name in ("distributed-cpu", "coverage-shard"):
        job = jobs[name]
        assert '\'["self-hosted","linux","x64","flagquantum-cpu"]\'' in job["runs-on"]
        assert "github.event_name != 'pull_request'" in job["runs-on"]
        assert "github.event.pull_request.head.repo.full_name == github.repository" in (
            job["runs-on"]
        )
        assert "'[\"ubuntu-latest\"]'" in job["runs-on"]
        assert not any(
            str(step.get("uses", "")).startswith("actions/checkout@")
            for step in job["steps"]
        )
    assert any(
        step.get("run") == "python -I tools/ci_cpu_budget.py --dedicated"
        for step in jobs["distributed-cpu"]["steps"]
    )
    # CUDA-Q stays hosted because cold SDK downloads time out on the shared pool.
    assert jobs["cudaq-optional"]["runs-on"] == "ubuntu-latest"
    assert "'flagquantum-a100-cpu'" in jobs["quality"]["runs-on"]
    assert any(
        step.get("with", {}).get("fetch-depth") == 0
        for step in jobs["quality"]["steps"]
    ), "Evidence provenance still needs full Git history"


def test_r300_compatibility_jobs_prepare_the_ephemeral_image() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    migrated = {
        name: job
        for name, job in jobs.items()
        if "flagcicd-r300-" in str(job.get("runs-on", ""))
    }
    assert set(migrated) == {"qiskit-optional", "cirq-optional", "braket-optional"}
    for job in migrated.values():
        assert job["timeout-minutes"] == 30
        steps = job["steps"]
        bootstrap = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses") == "./.github/actions/prepare-r300-cpu"
        )
        assert steps[bootstrap]["if"] == "startsWith(runner.name, 'flagcicd-r300-')"
        assert any(
            "git checkout --force --detach" in step.get("run", "")
            for step in steps[:bootstrap]
        )
        assert steps[bootstrap + 1]["uses"].startswith("actions/setup-python@")
        assert steps[bootstrap + 1]["if"] == (
            "${{ !startsWith(runner.name, 'flagcicd-r300-') }}"
        )
        assert steps[bootstrap + 2]["uses"] == "./.github/actions/isolated-python"


def test_r300_sdk_jobs_spread_across_distinct_inventory_targets() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    routes = [
        jobs[name]["runs-on"]
        for name in ("qiskit-optional", "cirq-optional", "braket-optional")
    ]
    assert len(set(routes)) == 3


def test_r300_source_failover_is_bounded_and_verifies_the_same_commit() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    for name in ("qiskit-optional", "cirq-optional", "braket-optional"):
        steps = jobs[name]["steps"]
        source = next(step for step in steps if step.get("id") == "source")
        assert source["timeout-minutes"] == 3
        assert (
            source["continue-on-error"]
            == "${{ startsWith(runner.name, 'flagcicd-r300-') }}"
        )
        restore = next(
            step
            for step in steps
            if step.get("name") == "Restore and verify this run's exact source commit"
        )
        assert restore["if"] == "steps.source.outcome == 'success'"
        checkout = next(
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        assert (
            checkout["if"]
            == "startsWith(runner.name, 'flagcicd-r300-') && steps.source.outcome == 'failure'"
        )
        assert checkout["timeout-minutes"] == 5
        assert checkout["with"] == {
            "ref": "${{ github.sha }}",
            "persist-credentials": False,
        }
        verify = next(
            step
            for step in steps
            if step.get("name") == "Verify the checked-out fallback commit"
        )
        assert verify["if"] == "steps.source.outcome == 'failure'"
        assert verify["run"] == 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"'
