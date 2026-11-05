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
            expected.replace("flagquantum-a100-cpu", "flagquantum-a100-cpu-entry"),
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
    for workflow_index, workflow in enumerate(("ci.yml", "pre-commit.yaml")):
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
            # GitHub run IDs are globally unique, including across workflows.
            for run_id in (100 + 2 * workflow_index, 101 + 2 * workflow_index):
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
    assert len(groups) > 16, "Exercise more jobs than the 16-runner CPU pool"


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
        "qiskit-optional",
        "braket-optional",
        "dependency-bounds",
        "distributed-cpu",
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
    for name in ("azure-optional", "pennylane-optional", "qiskit-compat"):
        job = jobs[name]
        assert job["runs-on"] == "ubuntu-latest"
        assert job["concurrency"]["group"] in {"flagquantum-gh-2", "flagquantum-gh-3"}
        assert job["concurrency"]["queue"] == "max"
    # Long suites and cache-dependent Triton setup use the isolated CPU pool;
    # the other portable environment checks stay on the general pool.
    for name in ("distributed-cpu-shard", "triton-optional"):
        job = jobs[name]
        labels = '["self-hosted","linux","x64","flagquantum-cpu"]'
        if name == "triton-optional":
            # The generic CPU label also exists on legacy A800 hosts without
            # the provisioned MC550 wheel directories and network path.
            labels = '["self-hosted","linux","x64","flagquantum-cpu","flagquantum-mc550-cpu"]'
        assert f"'{labels}'" in job["runs-on"]
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
        for step in jobs["distributed-cpu-shard"]["steps"]
    )
    # CUDA-Q stays hosted because cold SDK downloads time out on the shared pool.
    assert jobs["cudaq-optional"]["runs-on"] == "ubuntu-latest"
    assert jobs["quality"]["runs-on"] == "ubuntu-latest"
    assert any(
        step.get("with", {}).get("fetch-depth") == 0
        for step in jobs["quality"]["steps"]
    ), "Evidence provenance still needs full Git history"


def test_jax_and_cirq_use_preinstalled_self_hosted_profiles() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    for name in ("jax-optional", "cirq-optional"):
        job = jobs[name]
        assert "'flagquantum-cpu-sdk-preinstalled'" in job["runs-on"]
        assert "'ubuntu-latest'" in job["runs-on"]
        assert any(
            "fq_attach_preinstalled_sdk" in step.get("run", "") for step in job["steps"]
        )
    assert not any(
        "pip download" in step.get("run", "") for step in jobs["jax-optional"]["steps"]
    )


def test_sdk_jobs_use_accepted_fixed_capacity() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    for name in ("cirq-optional", "braket-optional"):
        job = jobs[name]
        label = (
            "flagquantum-cpu-sdk-preinstalled"
            if name == "cirq-optional"
            else "flagquantum-cpu-general"
        )
        assert f"'{label}'" in job["runs-on"]
        assert "'ubuntu-latest'" in job["runs-on"]
        assert job["timeout-minutes"] == 30
        assert job["env"]["MAX_JOBS"] == "2"
        installer = next(
            step
            for step in job["steps"]
            if step.get("uses") == "./.github/actions/install-cpu-torch"
        )
        assert installer["timeout-minutes"] == 8
        assert not any(
            step.get("continue-on-error") for step in job["steps"]
        ), "SDK preparation must fail closed"


def test_sdk_jobs_retain_native_preflight_before_certification() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    for name in ("cirq-optional", "braket-optional"):
        steps = jobs[name]["steps"]
        index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Validate the CPU runtime before SDK certification"
        )
        assert "adjoint.native_cpu_adjoint_available()" in steps[index]["run"]
        assert (
            "tests/test_native_circuit.py tests/test_backends.py tests/smoke/test_smoke_baseline.py -q"
            in steps[index]["run"]
        )
        sdk = name.removesuffix("-optional").removesuffix("-compat")
        assert any(
            f"python tools/check_{sdk}_interop_contract.py" in step.get("run", "")
            for step in steps[index + 1 :]
        )


def test_r300_manual_diagnostics_remain_available() -> None:
    workflow = yaml.safe_load((WORKFLOWS / "r300-validation.yml").read_text())
    assert len(_inventoried_r300_labels()) == 17
    assert workflow["jobs"]["resources"]["strategy"]["max-parallel"] == 17
    assert workflow["jobs"]["resources"]["timeout-minutes"] == 5


def test_r300_is_not_a_dependency_of_pr_source_preparation() -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    assert "r300-routes" not in jobs["prepare-source"]["outputs"]
    assert not any(
        "r300_pool.py" in step.get("run", "")
        for step in jobs["prepare-source"]["steps"]
    )
    assert all(
        "flagcicd-r300-" not in str(job.get("runs-on", ""))
        and "outputs.r300-routes" not in str(job.get("runs-on", ""))
        for job in jobs.values()
    )


@pytest.mark.parametrize(
    ("job", "profile"),
    [
        ("cpu-core", "core-py${{ matrix.python-version }}"),
        ("cpu-core-3-12-tests", "core-py3.12"),
        ("cpu-runtime", "core-py3.12"),
        ("coverage-shard", "coverage-py3.12"),
    ],
)
def test_cpu_suites_attach_the_locked_image(job: str, profile: str) -> None:
    jobs = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]
    suite = jobs[job]
    assert suite["runs-on"] == "ubuntu-latest"
    steps = suite["steps"]
    attach = next(
        step for step in steps if step.get("uses") == "./.github/actions/offline-cpu"
    )
    assert "if" not in attach
    assert attach["with"]["profile"] == profile
    assert attach["with"]["registry-token"] == "${{ github.token }}"
    assert not any(
        "setup-python" in step.get("uses", "") or "pip install" in step.get("run", "")
        for step in steps
    )


@pytest.mark.parametrize(
    "workflow,name",
    [
        ("ci.yml", "pre-commit"),
        ("ci.yml", "prepare-source"),
        ("local-gpu.yml", "prepare-source"),
        ("main-smoke.yml", "installed-wheel-smoke"),
    ],
)
def test_linux_prerequisites_keep_forks_off_private_cpu_runners(
    workflow: str, name: str
) -> None:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    job = document["jobs"][name]
    if workflow == "local-gpu.yml":
        assert job["runs-on"] == (
            "${{ (github.event_name != 'pull_request' || "
            "github.event.pull_request.head.repo.full_name == github.repository) "
            "&& 'flagquantum-a100-cpu-entry' || 'ubuntu-latest' }}"
        )
        assert "github.run_id" in job["concurrency"]["group"]
    else:
        assert job["runs-on"] == "ubuntu-latest"
    assert "pull_request_target" not in document.get("on", document.get(True, {}))
    assert job["concurrency"]["queue"] == "max"
    assert not job.get("continue-on-error", False)
    assert job["timeout-minutes"] <= 15
    steps = job["steps"]
    checkout = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert steps[checkout]["with"]["persist-credentials"] is False
    clearing = "\n".join(step.get("run", "") for step in steps[:checkout])
    assert "git sparse-checkout disable" in clearing
    assert "rm -f .git/info/sparse-checkout" in clearing
    assert "git config --local --unset core.sparseCheckout" in clearing
    for index, step in enumerate(steps):
        if str(step.get("uses", "")).startswith("actions/setup-python@"):
            assert "cache" not in step.get("with", {})
            assert steps[index + 1]["uses"] == "./.github/actions/isolated-python"
