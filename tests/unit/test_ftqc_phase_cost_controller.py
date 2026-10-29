"""Synthetic observation/controller checks: no live probe or numerical worker."""

from __future__ import annotations

import ast
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tools import collect_ftqc_phase_cost as controller
from tools import observe_ftqc_phase_cost as worker

pytestmark = pytest.mark.unit


def test_structural_descriptions_match_frozen_reference_without_execution() -> None:
    path = Path("tests/integration/test_ftqc_coherent_reference.py")
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_matrix_description"
    )
    namespace: dict[str, Any] = {
        "_MATRIX": tuple(itertools.product((2, 3), (0, 1, 2, 4)))
    }
    exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
    for n, t in namespace["_MATRIX"]:
        assert worker.description(f"n{n}t{t}") == (
            n,
            namespace["_matrix_description"](n, t),
        )
    assert worker.description("hth") == (1, (("h", (0,)), ("t", (0,)), ("h", (0,))))
    assert worker.description("bell") == (
        2,
        (("h", (0,)), ("cx", (0, 1)), ("t", (0,)), ("h", (0,)), ("h", (1,))),
    )


def test_expansion_and_schedule_bound_the_approved_work() -> None:
    states = queries = 0
    for case in worker.CASES:
        n, word = worker.description(case)
        terms = worker.expansion(word)
        assert len(terms) == 2 ** sum(g == "t" for g, _ in word)
        assert all(
            len(w) <= 22 and all(g in ("h", "s", "cx") for g, _ in w) for _, w in terms
        )
        states += len(terms) + 2**n
        queries += len(terms) ** 2 + len(terms) * 2**n
    assert (4 * states, 4 * queries) == (416, 3400)
    matrix = controller.schedule()
    assert len(matrix) == 160
    assert sum(x["route"] != "candidate" for x in matrix) == 120
    for repeat in range(4):
        for index, case in enumerate(worker.CASES):
            point = matrix[repeat * 40 + index * 4 : repeat * 40 + index * 4 + 4]
            assert [x["route"] for x in point] == [
                worker.ROUTES[(offset + index + repeat) % 4] for offset in range(4)
            ]
            assert all(x["case"] == case and x["repetition"] == repeat for x in point)


def test_outside_domain_and_evidence_overwrite_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        worker.description("n4t8")
    path = tmp_path / "evidence.json"
    worker.atomic(path, {"original": True})
    with pytest.raises(FileExistsError):
        worker.atomic(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


def payload(directory: Path, item: dict[str, Any], **overrides: Any) -> None:
    data = {
        "case": item["case"],
        "route": item["route"],
        "status": "completed",
        "errors": dict.fromkeys(
            (
                (
                    "probability_error",
                    "normalization_error",
                    "complex_amplitude_error",
                    "complex_gram_error",
                )
                if item["route"] == "candidate"
                else ("probability_error", "normalization_error")
            ),
            0.0,
        ),
        **controller.expected_work(item),
        "performance_targets_enabled": False,
        "numerical_seconds": 0.001,
    }
    data.update(overrides)
    (directory / "producer.json").write_text(json.dumps(data))


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -0.1, 1.1e-12, True])
def test_synthetic_bad_comparison_never_passes(tmp_path: Path, error: float) -> None:
    item = {"case": "hth", "route": "statevector"}
    payload(tmp_path, item, errors={"one": error, "two": 0.0})
    result = controller.assess(tmp_path, {"status": "completed", "returncode": 0}, item)
    assert result["status"] == "failed"


def test_monitor_failure_primary_even_with_successful_producer(tmp_path: Path) -> None:
    item = {"case": "hth", "route": "candidate"}
    payload(tmp_path, item)
    report = controller.assess(tmp_path, {"status": "failed", "returncode": 0}, item)
    assert report["failure_domain"] == "supervision"


@pytest.mark.parametrize(
    "route,stage,expected",
    [
        ("mps", "planning", "refused"),
        ("candidate", "planning", "failed"),
        ("mps", "execution", "failed"),
    ],
)
def test_only_explicit_public_planning_refusal_is_nonfatal(
    tmp_path: Path, route: str, stage: str, expected: str
) -> None:
    item = {"case": "hth", "route": route}
    payload(
        tmp_path,
        item,
        status="refused",
        diagnostic="unsupported exact output",
        refusal_stage=stage,
    )
    assert (
        controller.assess(tmp_path, {"status": "completed", "returncode": 0}, item)[
            "status"
        ]
        == expected
    )


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        controller, "admission", lambda *_: {"python": "synthetic-python"}
    )
    monkeypatch.setattr(controller, "budget", lambda *_: None)

    def execute(
        command: list[str], directory: Path, output: Path, deadline: float
    ) -> dict[str, Any]:
        commands.append(command)
        if "--case" in command:
            item = {
                "case": command[command.index("--case") + 1],
                "route": command[command.index("--route") + 1],
            }
            payload(directory, item)
        return {
            "status": "completed",
            "returncode": 0,
            "outer_wall_seconds": 0.01,
            "sampled_group_rss_max_bytes": 4096,
        }

    monkeypatch.setattr(controller, "execute", execute)
    return commands


def test_full_synthetic_campaign_has_one_probe_and_160_workers(
    tmp_path: Path, synthetic: list[list[str]]
) -> None:
    output = tmp_path / "new"
    report = controller.campaign(
        tmp_path / "admission.json", output, "synthetic-source"
    )
    assert report["status"] == "completed"
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["points"]) == 40 and summary["selection_made"] is False
    assert all(
        x["complete_three_repeats"] and x["numerical_seconds"]["median"] == 0.001
        for x in summary["points"]
    )
    assert report["started_attempts"] == len(synthetic) == 161
    assert all(
        x["status"] == "completed" and x["supervision"]["outer_wall_seconds"] > 0
        for x in report["attempts"]
    )
    with pytest.raises(FileExistsError):
        controller.campaign(tmp_path / "admission.json", output, "synthetic-source")


def test_probe_failure_starts_no_worker(
    tmp_path: Path, synthetic: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller, "assess", lambda *_: {"status": "failed"})
    report = controller.campaign(
        tmp_path / "admission.json", tmp_path / "new", "source"
    )
    assert report["status"] == "failed" and len(synthetic) == 1
    assert all(x["status"] == "not_attempted" for x in report["attempts"])


def test_first_failure_retains_remaining_slots_without_retry(
    tmp_path: Path, synthetic: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = controller.assess
    monkeypatch.setattr(
        controller,
        "assess",
        lambda directory, supervision, item: (
            {"status": "failed"} if item else original(directory, supervision, item)
        ),
    )
    report = controller.campaign(
        tmp_path / "admission.json", tmp_path / "new", "source"
    )
    assert len(synthetic) == 2
    assert report["attempts"][0]["status"] == "failed"
    assert all(x["status"] == "not_attempted" for x in report["attempts"][1:])


def test_refused_repetitions_stay_unattempted_not_retried(
    tmp_path: Path, synthetic: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = controller.assess

    def assess(
        directory: Path, supervision: dict[str, Any], item: dict[str, Any] | None
    ) -> dict[str, Any]:
        if item and item["case"] == "n2t0" and item["route"] == "mps":
            return {"status": "refused"}
        return original(directory, supervision, item)

    monkeypatch.setattr(controller, "assess", assess)
    report = controller.campaign(
        tmp_path / "admission.json", tmp_path / "new", "source"
    )
    assert report["status"] == "completed_with_refusals" and len(synthetic) == 158
    refused = [
        x for x in report["attempts"] if x["case"] == "n2t0" and x["route"] == "mps"
    ]
    assert [x["status"] for x in refused] == [
        "refused",
        "not_attempted",
        "not_attempted",
        "not_attempted",
    ]


def test_monitor_exception_cleans_only_owned_group_and_retains_outer_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 4242
        returncode = None

        def poll(self) -> None:
            return None

    process = Process()
    stopped: list[object] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(controller, "budget", lambda *_: None)
    monkeypatch.setattr(
        controller,
        "_process_group_rss",
        lambda _group: (_ for _ in ()).throw(subprocess.TimeoutExpired("ps", 0.05)),
    )
    monkeypatch.setattr(controller, "_stop_group", stopped.append)
    directory = tmp_path / "attempt"
    directory.mkdir()
    report = controller.execute(["synthetic"], directory, tmp_path, float("inf"))
    assert report["status"] == "failed" and stopped == [process]
    assert report["owned_pgid"] == 4242 and report["outer_wall_seconds"] > 0
    assert "TimeoutExpired" in report["diagnostic"]


def test_delivery_receipt_refuses_pending_or_different_main(tmp_path: Path) -> None:
    path = tmp_path / "delivery.json"
    for data in (
        {"head": "source", "all_main_admission_gates_passed": False},
        {"head": "other", "all_main_admission_gates_passed": True},
    ):
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            controller.require_delivery(path, "source")
    path.write_text(
        json.dumps({"head": "source", "all_main_admission_gates_passed": True})
    )
    controller.require_delivery(path, "source")


def test_raw_trace_is_not_duplicated_in_ledger(tmp_path: Path) -> None:
    item = {"case": "hth", "route": "statevector"}
    payload(tmp_path, item)
    result = controller.assess(
        tmp_path,
        {"status": "completed", "returncode": 0, "samples_seconds_bytes": [(0.01, 42)]},
        item,
    )
    assert "samples_seconds_bytes" not in result["supervision"]


def test_installed_admission_rejects_wheel_or_member_drift_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    env = tmp_path / "environment"
    (env / "bin").mkdir(parents=True)
    python = env / "bin/python"
    python.write_bytes(b"synthetic interpreter, never executed")
    (env / "pyvenv.cfg").write_text("synthetic")
    site = env / "lib/python3.12/site-packages"
    package = site / "flagquantum"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"synthetic package, never imported")
    metadata = site / "torch-2.14.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch\nVersion: 2.14.1\n"
    )
    wheel = tmp_path / "retained.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(package / "__init__.py", "flagquantum/__init__.py")
    report = tmp_path / "installation.json"
    data: dict[str, Any] = {
        "source_commit": "source",
        "fresh_core_install": True,
        "python": str(python),
        "wheel": {"path": str(wheel), "sha256": controller.digest(wheel)},
        "runtime_versions": {"python": "synthetic", "torch": "2.14.1"},
    }
    report.write_text(json.dumps(data))

    def command(argv: list[str], **kwargs: Any) -> str:
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return "source\n"
        if argv[:3] == ["git", "status", "--porcelain"]:
            return ""
        if argv[0] == "sysctl":
            return "synthetic CPU"
        raise AssertionError(f"unexpected process: {argv}")

    monkeypatch.setattr(subprocess, "check_output", command)
    monkeypatch.setattr(platform, "processor", lambda: "synthetic CPU")
    admitted = controller.admission(report, "source")
    assert admitted["torch_version"] == "2.14.1"
    (package / "__init__.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="wheel member"):
        controller.admission(report, "source")
    data["wheel"]["sha256"] = "wrong"
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="wheel digest"):
        controller.admission(report, "source")


@pytest.mark.parametrize(
    "name", ["fts10-cpu-observations-20261025", "fts10-cpu-continuation-20261025"]
)
def test_old_raw_tree_descendants_are_refused_before_any_attempt(
    tmp_path: Path, synthetic: list[list[str]], name: str
) -> None:
    parent = tmp_path / name
    parent.mkdir()
    with pytest.raises(ValueError, match="historical"):
        controller.campaign(tmp_path / "admission.json", parent / "new-child", "source")
    assert list(parent.iterdir()) == [] and synthetic == []


@pytest.mark.parametrize(
    "rss,polled,expected",
    [
        ([4096, 0], [None, 0], "completed"),
        ([controller.RSS_BYTES + 1], [None], "failed"),
        ([4096, 4096], [None, 0], "failed"),
    ],
)
def test_synthetic_monitor_success_limit_and_surviving_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rss: list[int],
    polled: list[int | None],
    expected: str,
) -> None:
    class Process:
        pid = 4242
        returncode: int | None = None

        def poll(self) -> int | None:
            value = polled.pop(0)
            self.returncode = value
            return value

    process = Process()
    stopped: list[object] = []
    options: dict[str, Any] = {}

    def start(*args: Any, **kwargs: Any) -> Process:
        options.update(kwargs)
        return process

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(controller, "budget", lambda *_: None)
    monkeypatch.setattr(controller, "_process_group_rss", lambda _: rss.pop(0))
    monkeypatch.setattr(controller, "_stop_group", stopped.append)
    ticks = itertools.count(0.0, 0.001)
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    directory = tmp_path / "attempt"
    directory.mkdir()
    report = controller.execute(["synthetic"], directory, tmp_path, 1200)
    assert report["status"] == expected
    assert bool(stopped) == (expected == "failed")
    assert (
        options["start_new_session"]
        and options["preexec_fn"] is controller.limit_evidence_files
    )
    assert "PYTHONPATH" not in options["env"]
    assert (
        options["env"]["MAX_JOBS"] == "2"
        and options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    )


def test_changed_identity_counters_and_process_failure_are_retained(
    tmp_path: Path,
) -> None:
    item = {"case": "n2t0", "route": "candidate"}
    for changes in (
        {"case": "n3t0"},
        {"overlap_queries": 999},
        {"status": "failed", "diagnostic": "original numerical failure"},
    ):
        payload(tmp_path, item, **changes)
        result = controller.assess(
            tmp_path, {"status": "completed", "returncode": 0}, item
        )
        assert result["status"] == "failed" and result["failure_domain"] == "producer"
    result = controller.assess(tmp_path, {"status": "completed", "returncode": 1}, item)
    assert "original numerical failure" in result["diagnostic"]


def test_member_integrity_loop_cannot_replace_the_requested_case() -> None:
    tree = ast.parse(Path("tools/observe_ftqc_phase_cost.py").read_text())
    observe = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "observe"
    )
    for node in ast.walk(observe):
        if isinstance(node, ast.For):
            assigned = {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            assert not assigned.intersection({"name", "route", "provenance"})


@pytest.mark.parametrize(
    ("case", "failed_check"),
    [
        ("shape", "expected_shape"),
        ("dtype", "float64"),
        ("nan", "finite"),
        ("infinity", "finite"),
        ("negative", "nonnegative"),
    ],
)
def test_probability_failure_snapshot_preserves_rejected_values(
    case: str, failed_check: str
) -> None:
    import torch

    values = torch.tensor([0.0, 1.0], dtype=torch.float64)
    if case == "shape":
        values = values.reshape(1, 2)
    elif case == "dtype":
        values = values.float()
    elif case == "nan":
        values[0] = float("nan")
    elif case == "infinity":
        values[0] = float("inf")
    elif case == "negative":
        values[0] = -1e-16
    original = values.clone()
    snapshot = worker.probability_failure_snapshot(values, 1)
    assert snapshot["checks"][failed_check] is False
    assert snapshot["values_repr"] == [repr(v) for v in values.reshape(-1).tolist()]
    assert snapshot["element_count"] == 2
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    torch.testing.assert_close(values, original, rtol=0, atol=0, equal_nan=True)


def test_probability_failure_snapshot_bounds_values_and_avoids_device_copy() -> None:
    import torch

    oversized = torch.arange(20, dtype=torch.float64)
    snapshot = worker.probability_failure_snapshot(oversized, 3)
    assert snapshot["element_count"] == 20
    assert snapshot["values_repr"] == [repr(float(i)) for i in range(8)]
    device_snapshot = worker.probability_failure_snapshot(
        torch.empty(2, dtype=torch.float64, device="meta"), 1
    )
    assert device_snapshot["checks"]["cpu"] is False
    assert device_snapshot["checks"]["finite"] is None
    assert device_snapshot["checks"]["nonnegative"] is None
    assert device_snapshot["values_repr"] is None
    complex_snapshot = worker.probability_failure_snapshot(
        torch.tensor([1j, 1 + 0j], dtype=torch.complex128), 1
    )
    assert complex_snapshot["checks"]["float64"] is False
    assert complex_snapshot["checks"]["nonnegative"] is None
    assert complex_snapshot["values_repr"] == ["1j", "(1+0j)"]
    json.dumps(complex_snapshot, allow_nan=False)


def test_probability_guard_failure_crosses_worker_boundary_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}")
    output = tmp_path / "producer.json"
    validation = {"checks": {"nonnegative": False}, "values_repr": ["-1e-16"]}
    calls: list[str] = []

    def rejected(case: str, route: str, context: dict[str, Any]) -> dict[str, Any]:
        calls.append(case)
        assert route == "mps" and context == {}
        raise worker.ProbabilityValidationError(validation)

    monkeypatch.setattr(worker, "observe", rejected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "observer",
            "--case",
            "n2t1",
            "--route",
            "mps",
            "--provenance",
            str(provenance),
            "--output",
            str(output),
        ],
    )
    assert worker.main() == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["probability_validation"] == validation
    assert "require finite ordered CPU float64 probabilities" in report["diagnostic"]
    assert calls == ["n2t1"]
    assert output.with_name("report-io.json").exists()
    assessed = controller.assess(
        tmp_path,
        {"status": "completed", "returncode": 1},
        {"case": "n2t1", "route": "mps"},
    )
    assert assessed["status"] == "failed"
    assert assessed["failure_domain"] == "producer"
