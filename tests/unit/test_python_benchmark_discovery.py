"""Preserve complete physical benchmark discovery and script entry behavior."""

import json
import runpy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from benchmarks import physical_array_perf
from flagquantum.benchmarking._timing_suite import TimingSuite

pytestmark = pytest.mark.unit
_DATA: dict[str, Any] = json.loads(
    (Path(__file__).with_name("data") / "python_benchmark_workloads.json").read_text()
)


def test_complete_default_suite_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks import run_physical_perf

    observed = []

    def run(group: TimingSuite) -> None:
        observed.append([group.name, len(group.benchmarks)])

    monkeypatch.setattr(TimingSuite, "main", run)
    run_physical_perf.main()
    expected = [
        [name.replace("km.array", "OperandArray"), count]
        for name, count in _DATA["discovered"]
    ]
    assert sorted(observed) == sorted(expected)


def test_discovery_filters_suite_types_and_keeps_glob_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from benchmarks import run_physical_perf

    first = tmp_path / "physical_first_perf.py"
    second = tmp_path / "physical_second_perf.py"
    empty = tmp_path / "physical_empty_perf.py"
    wrong = tmp_path / "physical_wrong_perf.py"
    source = "from flagquantum.benchmarking._timing_suite import TimingSuite\nbenchmark = TimingSuite({name!r})\n"
    first.write_text(source.format(name="first"))
    second.write_text(source.format(name="second"))
    empty.write_text('print("loaded empty")\n')
    wrong.write_text('benchmark = object()\nprint("loaded wrong")\n')
    seen = []

    def observe(group: TimingSuite) -> None:
        seen.append(group.name)

    with (
        patch(
            "benchmarks.run_physical_perf.glob.glob",
            return_value=list(map(str, (second, empty, first, wrong))),
        ),
        patch.object(TimingSuite, "main", observe),
    ):
        run_physical_perf.run_directory(tmp_path)
    assert seen == ["second", "first"]
    assert capsys.readouterr().out == "loaded empty\nloaded wrong\n"
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "physical_nested_perf.py").write_text('raise RuntimeError("nested")\n')
    with patch.object(TimingSuite, "main", observe):
        run_physical_perf.run_directory(tmp_path)
    assert len(seen) == 4


def test_import_failure_and_empty_directory(tmp_path: Path) -> None:
    from benchmarks import run_physical_perf

    run_physical_perf.run_directory(tmp_path)
    error = tmp_path / "physical_error_perf.py"
    error.write_text('raise LookupError("source import failure")\n')
    with pytest.raises(LookupError, match="source import failure"):
        run_physical_perf.run_directory(tmp_path)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("physical_array_perf.py", [["OperandArray", 1]]),
        ("physical_builder_perf.py", [["CircuitBuilder", 5]]),
        ("run_physical_perf.py", [["CircuitBuilder", 5], ["OperandArray", 1]]),
    ],
)
def test_real_script_entry_paths(filename: str, expected: list[list[object]]) -> None:
    seen: list[list[object]] = []

    def observe(group: TimingSuite) -> None:
        seen.append([group.name, len(group.benchmarks)])

    path = Path(physical_array_perf.__file__).with_name(filename)
    with patch.object(TimingSuite, "main", observe):
        runpy.run_path(str(path), run_name="__main__")
    assert sorted(seen) == sorted(expected)
