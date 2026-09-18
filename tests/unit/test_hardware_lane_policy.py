"""Invariants that keep the accelerator lanes from measuring each other.

The repository registers self-hosted runners onto hosts that other people also
use, and more than one lane asks those hosts for devices. Two lanes that overlap
take their measurements on a machine that the other one is loading, which is how
a threshold the code passes gets reported as a regression. These rules are about
the workflow's own job graph, which the workflow fully controls, rather than
about the containers a neighbour happens to be running, which it does not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# The one group every accelerator job joins. A shared name is the point: the
# group is repository-wide, so the scheduled lane and the manual gate serialize
# against each other and not merely against themselves.
ACCELERATOR_GROUP = "flagquantum-accelerator"

# The three files that ask a host for its device state. They cannot share the
# constant: every tool here is a self-contained script, and a script under
# `tools/` cannot import a sibling module. The copies therefore have to be held
# to each other by test instead, because a query and a parser that drift apart
# answer with nothing rather than with an error, and nothing is also what an
# idle host answers.
DEVICE_QUERY_SOURCES = (
    ROOT / "benchmarks" / "continuous_performance.py",
    ROOT / "tools" / "evaluate_performance_artifact.py",
    ROOT / "tools" / "hardware_run_manifest.py",
)

# Without `nounits` the answer reads `4 MiB` where the parser expects `4`, so
# the flag and the parser are one decision recorded in three places.
UNITLESS_FORMAT = "--format=csv,noheader,nounits"


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
    } <= found


def test_every_accelerator_job_joins_the_serializing_group() -> None:
    offenders: list[str] = []
    for key, (path, job) in sorted(_accelerator_jobs().items()):
        concurrency = job.get("concurrency")
        if not isinstance(concurrency, dict):
            offenders.append(f"{key} declares no concurrency block")
            continue
        if concurrency.get("group") != ACCELERATOR_GROUP:
            offenders.append(
                f"{key} joins group {concurrency.get('group')!r} "
                f"instead of {ACCELERATOR_GROUP!r}"
            )
        # The default keeps one waiting job and drops it when another arrives,
        # which would report nothing at all for the lane that was dropped.
        if concurrency.get("queue") != "max":
            offenders.append(f"{key} does not set `queue: max`")
    assert not offenders, (
        "accelerator jobs must share one concurrency group so that no two of "
        "them measure the same devices at the same time:\n" + "\n".join(offenders)
    )


def test_no_accelerator_matrix_runs_its_legs_at_once() -> None:
    # A runner label says `gpu`, not how many devices it has, so a matrix leg
    # that asks for eight devices cannot tell whether its sibling asking for
    # four landed on the same host. One leg at a time is the only assumption the
    # label set supports.
    offenders: list[str] = []
    for key, (_path, job) in sorted(_accelerator_jobs().items()):
        strategy = job.get("strategy")
        if not isinstance(strategy, dict):
            continue
        matrix = strategy.get("matrix")
        if not isinstance(matrix, dict) or sum(len(v) for v in matrix.values()) < 2:
            continue
        if strategy.get("max-parallel") != 1:
            offenders.append(f"{key} runs its matrix legs in parallel")
    assert not offenders, "\n".join(offenders)


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


def test_the_three_device_query_copies_agree() -> None:
    queries = {
        path.relative_to(ROOT).as_posix(): _device_query(path)
        for path in DEVICE_QUERY_SOURCES
    }

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
