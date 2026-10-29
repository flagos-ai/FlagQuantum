"""Invariants that keep the accelerator lanes from measuring each other.

The repository registers self-hosted runners onto hosts that other people also
use, and more than one lane asks those hosts for devices. Two lanes that overlap
take their measurements on a machine that the other one is loading, which is how
a threshold the code passes gets reported as a regression. These rules are about
the workflow's own job graph, which the workflow fully controls, rather than
about the containers a neighbour happens to be running, which it does not.

Pull requests may reach the explicitly provisioned A800 CPU and GPU jobs.
Other self-hosted hardware remains available only through its existing triggers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# The multi-node lane uses both A800 hosts, so their jobs still share a group.
# The A100 validation host is independent and must not wait for the A800 pair.
ACCELERATOR_GROUP = "flagquantum-accelerator"

A800_CPU_ROUTE = '${{ fromJSON((github.event_name != \'pull_request\' || github.event.pull_request.head.repo.full_name == github.repository) && \'["self-hosted","linux","x64","flagquantum-cpu"]\' || \'["ubuntu-latest"]\') }}'

MC550_CPU_ROUTE = A800_CPU_ROUTE.replace(
    '"flagquantum-cpu"]', '"flagquantum-cpu","flagquantum-mc550-cpu"]'
)

TRUSTED_CPU_ROUTE = (
    "${{ (github.event_name != 'pull_request' || "
    "github.event.pull_request.head.repo.full_name == github.repository) "
    "&& 'flagquantum-a100-cpu' || 'ubuntu-latest' }}"
)

ENTRY_CPU_ROUTE = TRUSTED_CPU_ROUTE.replace(
    "flagquantum-a100-cpu", "flagquantum-a100-cpu-entry"
)

PORTABLE_CPU_ROUTE = TRUSTED_CPU_ROUTE.replace(
    "flagquantum-a100-cpu", "flagquantum-cpu-general"
)

# The files that ask a host for its device state. They cannot share the
# constant: every tool here is a self-contained script, and a script under
# `tools/` cannot import a sibling module. The copies therefore have to be held
# to each other by test instead, because a query and a parser that drift apart
# answer with nothing rather than with an error, and nothing is also what an
# idle host answers.
DEVICE_QUERY_SOURCES = (
    ROOT / "benchmarks" / "continuous_performance.py",
    ROOT / "tools" / "evaluate_performance_artifact.py",
    ROOT / "tools" / "hardware_run_manifest.py",
    ROOT / "tools" / "multinode_launch_plan.py",
)

# Without `nounits` the answer reads `4 MiB` where the parser expects `4`, so
# the flag and the parser are one decision recorded in every copy.
UNITLESS_FORMAT = "--format=csv,noheader,nounits"

# Every event that lets a contributor's own code reach a workflow: a fork or a
# branch in this repository. `workflow_run` is here because it fires in response
# to a workflow that a pull request did trigger, so it inherits that run's reach
# into whatever runner it asks for; `repository_dispatch` is absent because it
# needs a token with write access, which a pull request does not have. The set
# is a lower bound -- an event this map does not know about is reported as
# unattributed rather than silently treated as safe.
UNTRUSTED_EVENTS = {
    "pull_request": "a pull request from a fork or a branch",
    "pull_request_target": "a pull request, with the base repository's secrets",
    "workflow_run": "a workflow that a pull request triggered",
}

# Events whose payload a contributor controls but whose trigger needs a write
# token or a maintainer, so they are not a route from a pull request on their
# own. They are named here so that `test_every_workflow_event_is_attributed`
# reports one of them going missing rather than staying quiet about it.
TRUSTED_EVENTS = frozenset(
    {
        "push",
        "schedule",
        "workflow_dispatch",
        "workflow_call",
        "release",
        "create",
        "delete",
        "deployment",
        "issue_comment",
        "merge_group",
    }
)


def test_scheduled_gpu_job_runs_single_device_runtime_regressions() -> None:
    job = _jobs(WORKFLOWS / "scheduled-hardware.yml")["gpu-scheduled"]
    assert "gpu" in _runs_on(job)
    steps = job["steps"]
    assert isinstance(steps, list)
    step = next(
        step
        for step in steps
        if step.get("name") == "Run single-device runtime regressions"
    )
    command = step["run"]
    assert "python -m pytest" in command
    assert (
        "tests/unit/test_density_matrix_output.py::"
        "test_noisy_density_keeps_accelerator_and_gradient" in command
    )
    assert (
        "tests/unit/test_mps_site_kernels.py::"
        "test_eager_two_site_contraction_preserves_precision_with_tf32_enabled"
        in command
    )


def _jobs(path: Path) -> dict[str, dict[str, object]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    jobs = document.get("jobs")
    assert isinstance(jobs, dict), f"{path.name}: no jobs mapping"
    return jobs


def _runs_on(job: dict[str, object]) -> tuple[str, ...]:
    value = job.get("runs-on")
    if isinstance(value, str):
        return (value,)
    assert isinstance(value, list), f"unexpected runs-on: {value!r}"
    return tuple(str(item) for item in value)


def _triggers(path: Path) -> frozenset[str]:
    """The event names this workflow declares, however the file spells them.

    `on` parses as the boolean `True` because YAML reads a bare `on` as a truthy
    key, and a workflow that names a single event may write it as a scalar
    rather than a mapping. Both spellings appear in this repository.
    """

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    declared = document.get(True, document.get("on"))
    if isinstance(declared, str):
        return frozenset({declared})
    assert isinstance(declared, dict), f"{path.name}: unexpected on: {declared!r}"
    return frozenset(str(event) for event in declared)


def _is_self_hosted(job: dict[str, object]) -> bool:
    labels = {label.lower() for label in _runs_on(job)}
    return any("self-hosted" in label for label in labels) or any(
        any(
            pool in label
            for pool in ("flagquantum-a100-cpu", "flagquantum-cpu-general")
        )
        for label in labels
    )


def _allows_untrusted_code(job: dict[str, object], reaching: frozenset[str]) -> bool:
    if not _is_self_hosted(job):
        return False
    return not (
        reaching == frozenset({"pull_request"})
        and job.get("runs-on")
        in (
            TRUSTED_CPU_ROUTE,
            ENTRY_CPU_ROUTE,
            PORTABLE_CPU_ROUTE,
            A800_CPU_ROUTE,
            MC550_CPU_ROUTE,
        )
    )


def _accelerator_jobs() -> dict[str, tuple[Path, dict[str, object]]]:
    """Every job that asks a self-hosted runner for a device."""

    found: dict[str, tuple[Path, dict[str, object]]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for name, job in _jobs(path).items():
            labels = {label.lower() for label in _runs_on(job)}
            if "self-hosted" in labels and "gpu" in labels:
                found[f"{path.name}:{name}"] = (path, job)
    return found


def test_the_scan_finds_the_accelerator_jobs_it_is_meant_to_police() -> None:
    # A policy scan that finds nothing passes every rule below while checking
    # nothing, so the discovery itself is asserted.
    found = set(_accelerator_jobs())

    assert {
        "local-gpu.yml:one-gpu-local",
        "local-gpu.yml:two-gpu-distributed-required",
        "scheduled-hardware.yml:crossover-scheduled",
        "scheduled-hardware.yml:gpu-scheduled",
        "scheduled-hardware.yml:local-scale-scheduled",
        "scheduled-hardware.yml:multinode",
    } <= found


def test_accelerator_jobs_serialize_only_with_their_shared_hosts() -> None:
    offenders: list[str] = []
    for key, (path, job) in sorted(_accelerator_jobs().items()):
        concurrency = job.get("concurrency")
        if not isinstance(concurrency, dict):
            offenders.append(f"{key} declares no concurrency block")
            continue
        labels = set(_runs_on(job))
        expected = (
            f"{ACCELERATOR_GROUP}-a100"
            if "flagquantum-a100" in labels
            else ACCELERATOR_GROUP
        )
        if key.startswith("scheduled-hardware.yml:"):
            # A generic GPU selector could choose the independent A100 host
            # while holding only the A800 lock, defeating device exclusion.
            assert "a800-172" in labels or "multinode" in labels
        if concurrency.get("group") != expected:
            offenders.append(
                f"{key} joins group {concurrency.get('group')!r} "
                f"instead of {expected!r}"
            )
        # The default keeps one waiting job and drops it when another arrives,
        # which would report nothing at all for the lane that was dropped.
        if concurrency.get("queue") != "max":
            offenders.append(f"{key} does not set `queue: max`")
    assert not offenders, (
        "jobs using the same hosts must share a concurrency group, while "
        "independent hosts must not block each other:\n" + "\n".join(offenders)
    )


def test_accelerator_matrix_uses_the_device_lock_instead_of_a_runner_cap() -> None:
    for key, (_path, job) in sorted(_accelerator_jobs().items()):
        strategy = job.get("strategy", {})
        if not strategy.get("matrix"):
            continue
        assert "max-parallel" not in strategy, key
        group = job["concurrency"]["group"]
        assert (
            "matrix." not in group
        ), f"{key}: matrix legs sharing devices must share the same device lock"


def _device_query(path: Path) -> str:
    """Read a module-level `DEVICE_QUERY` without importing the module.

    `benchmarks/` is not an installed package and a script cannot name a sibling
    `tools/` module, so the three copies are compared as parsed source.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "DEVICE_QUERY" in targets:
            value = ast.literal_eval(node.value)
            assert isinstance(value, str), f"{path}: DEVICE_QUERY is not a string"
            return value
    raise AssertionError(f"{path}: no module-level DEVICE_QUERY")


def test_the_device_query_copies_agree() -> None:
    queries = {
        path.relative_to(ROOT).as_posix(): _device_query(path)
        for path in DEVICE_QUERY_SOURCES
    }

    assert len(queries) >= 3, queries
    assert len(set(queries.values())) == 1, queries


def test_every_device_query_asks_for_columns_without_units() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in DEVICE_QUERY_SOURCES
        if UNITLESS_FORMAT not in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "a device-state answer carrying units parses as no devices at all, "
        "which reads the same as an idle host: " + ", ".join(offenders)
    )


def _untrusted_workflow_jobs() -> dict[str, tuple[Path, frozenset[str]]]:
    """Workflows a pull request can reach, with the events that get it there."""

    found: dict[str, tuple[Path, frozenset[str]]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        reaching = _triggers(path) & set(UNTRUSTED_EVENTS)
        if reaching:
            found[path.name] = (path, frozenset(reaching))
    return found


def test_every_workflow_event_is_attributed() -> None:
    # The rules below read an event set this module wrote down. An event that
    # arrives with a newer GitHub Actions and is not in either set would make a
    # workflow reachable by a route nobody classified, and the rules would stay
    # quiet about it. This is what speaks up instead.
    unattributed: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for event in sorted(_triggers(path)):
            if event not in UNTRUSTED_EVENTS and event not in TRUSTED_EVENTS:
                unattributed.append(f"{path.name}: {event}")
    assert (
        not unattributed
    ), "classify each event as reaching contributor code or not:\n" + "\n".join(
        unattributed
    )


def test_pull_requests_only_reach_provisioned_or_guarded_jobs() -> None:
    manual_pool_inputs = {
        "validate-cpu-pool": "validate_pool",
        "validate-eight-gpus": "validate_pool",
        "validate-mc550-pool": "validate_mc550_pool",
    }
    allowed = {
        ("local-gpu.yml", "one-gpu-local"): "a800-171",
        ("local-gpu.yml", "two-gpu-distributed-required"): "a800-172",
    }
    found = set()
    for name, (path, reaching) in sorted(_untrusted_workflow_jobs().items()):
        for job, definition in _jobs(path).items():
            if _is_self_hosted(definition):
                if not _allows_untrusted_code(definition, reaching):
                    continue
                if name == "local-gpu.yml" and job in manual_pool_inputs:
                    pool_input = manual_pool_inputs[job]
                    assert definition.get("if") == "${{ inputs." + pool_input + " }}"
                    inputs = yaml.safe_load(path.read_text())[True][
                        "workflow_dispatch"
                    ]["inputs"]
                    assert inputs[pool_input]["type"] == "boolean"
                    assert inputs[pool_input]["default"] is False
                    continue
                key = (name, job)
                assert key in allowed, f"unexpected self-hosted PR job: {key}"
                assert reaching == frozenset({"pull_request"})
                assert allowed[key] in _runs_on(definition)
                found.add(key)
    assert found == set(allowed)


def test_the_untrusted_scan_finds_the_workflows_it_is_meant_to_police() -> None:
    # Same reason as the discovery assertion above: a scan whose reachable set
    # came back empty would pass the rule while checking nothing.
    found = set(_untrusted_workflow_jobs())

    assert {"ci.yml", "publish-dev-container.yml", "local-gpu.yml"} <= found
    assert not {"flagos-reference.yml", "scheduled-hardware.yml"} & found


def test_the_self_hosted_detector_recognises_a_known_self_hosted_job() -> None:
    # A positive control for `_is_self_hosted`. The pull-request rule is written
    # as "no job may ...", so a detector that stopped recognising the label would
    # leave it with nothing to report and it would pass while checking nothing.
    # The discovery assertion above covers which workflows are reachable, not
    # whether a self-hosted job can be seen at all.
    self_hosted = _jobs(WORKFLOWS / "local-gpu.yml")["one-gpu-local"]
    github_hosted = _jobs(WORKFLOWS / "ci.yml")["package"]

    assert _is_self_hosted(self_hosted)
    assert not _is_self_hosted(github_hosted)


def test_trigger_parsing_reads_both_ways_a_workflow_can_spell_on(
    tmp_path: Path,
) -> None:
    # Every workflow here writes `on` as a mapping, which YAML hands back under
    # the boolean key `True`. A workflow naming a single event may write it as a
    # scalar instead, and reading that as no events at all would make the
    # pull-request rule pass by seeing nothing.
    mapping = tmp_path / "mapping.yml"
    mapping.write_text("on:\n  push:\n  pull_request:\n", encoding="utf-8")
    scalar = tmp_path / "scalar.yml"
    scalar.write_text("on: push\n", encoding="utf-8")

    assert _triggers(mapping) == frozenset({"push", "pull_request"})
    assert _triggers(scalar) == frozenset({"push"})


@pytest.mark.parametrize("route", [TRUSTED_CPU_ROUTE, MC550_CPU_ROUTE])
def test_cpu_route_rejects_missing_guards_and_other_untrusted_events(
    route: str,
) -> None:
    guarded = {"runs-on": route}
    assert _is_self_hosted(guarded)
    assert not _allows_untrusted_code(guarded, frozenset({"pull_request"}))
    for event in ("pull_request_target", "workflow_run"):
        assert _allows_untrusted_code(guarded, frozenset({event}))
    for unguarded in (
        "flagquantum-a100-cpu",
        "${{ 'flagquantum-a100-cpu' }}",
        route.replace(
            "github.event.pull_request.head.repo.full_name == github.repository",
            "true",
        ),
    ):
        assert _allows_untrusted_code(
            {"runs-on": unguarded}, frozenset({"pull_request"})
        )


def test_dedicated_cpu_lanes_keep_fork_code_on_hosted_runners() -> None:
    jobs = _jobs(WORKFLOWS / "ci.yml")
    for name in ("cpu-core", "cpu-core-3-12-tests", "cpu-runtime"):
        assert jobs[name]["runs-on"] == A800_CPU_ROUTE
        assert _is_self_hosted(jobs[name])
        assert not _allows_untrusted_code(jobs[name], frozenset({"pull_request"}))
    triton = jobs["triton-optional"]
    assert triton["runs-on"] == MC550_CPU_ROUTE
    assert _is_self_hosted(triton)
    assert not _allows_untrusted_code(triton, frozenset({"pull_request"}))
