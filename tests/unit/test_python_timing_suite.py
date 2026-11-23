"""Compare the complete Python timing source through native result ownership."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from flagquantum.benchmarking import _timing_suite as suite_module
from flagquantum.benchmarking._adaptive_timing import TimingSample, measure_suite
from flagquantum.benchmarking._timing_report import _rounded_si2, format_suite_timing
from flagquantum.benchmarking._timing_suite import TimingSuite

pytestmark = pytest.mark.unit
_DATA: dict[str, Any] = json.loads(
    (Path(__file__).with_name("data") / "python_timing_suite.json").read_text()
)


@pytest.mark.parametrize("value,expected", _DATA["si"])
def test_all_source_si_boundaries(value: float, expected: str) -> None:
    assert _rounded_si2(value) == expected


@pytest.mark.parametrize("seconds,reps,goal,expected", _DATA["reports"])
def test_complete_source_reports(
    seconds: float, reps: int, goal: float | None, expected: str
) -> None:
    sample = TimingSample(
        seconds,
        reps,
        (("ops", 1), ("ops", 2), ("items", 300)),
        goal_seconds=goal,
        name="group.case",
    )
    assert format_suite_timing(sample) == expected


@pytest.mark.parametrize("seconds,reps,goal,kind,message", _DATA["failures"])
def test_original_failure_types_and_diagnostics(
    seconds: float, reps: int, goal: str, kind: str, message: str
) -> None:
    sample = TimingSample(
        seconds,
        reps,
        (("ops", 1),),
        goal_seconds=None if goal == "None" else float(goal),
    )
    exceptions = {
        "ValueError": ValueError,
        "ZeroDivisionError": ZeroDivisionError,
        "OverflowError": OverflowError,
    }
    with pytest.raises(exceptions[kind]) as caught:
        format_suite_timing(sample)
    assert str(caught.value) == message


@pytest.mark.parametrize("target,ticks,calls,reps,seconds,rates,goal", _DATA["traces"])
def test_complete_source_batches_and_float_clock(
    target: float,
    ticks: list[float],
    calls: int,
    reps: int,
    seconds: float,
    rates: list[tuple[str, float]],
    goal: None,
) -> None:
    clock = iter(ticks)
    actual_calls = []
    actual = measure_suite(
        lambda: actual_calls.append(1),
        target_seconds=target,
        rates=(("ops", 1),),
        clock=lambda: next(clock),
    )
    assert len(actual_calls) == calls
    assert actual == TimingSample(
        seconds,
        reps,
        tuple((name, value) for name, value in rates),
        goal_seconds=goal,
    )


@pytest.mark.parametrize("units,name,goal,rates,size", _DATA["registrations"])
def test_complete_original_decorator_forms(
    units: dict[str, float],
    name: str,
    goal: float | None,
    rates: list[list[object]],
    size: int,
) -> None:
    group = TimingSuite("group")

    def body() -> TimingSample:
        return TimingSample(0.5, 3, (("ops", 2),), goal_seconds=3, name="inner")

    decorated = group(body, units={"ops": 1}, **units)
    actual = decorated()
    assert actual.name == name and actual.goal_seconds == goal
    assert list(map(list, actual.rates)) == rates and len(group.benchmarks) == size
    second = group(units={"ops": 1}, **units)(body)
    assert second() == actual and len(group.benchmarks) == 2


def test_mutable_registration_rates_and_main(
    capsys: pytest.CaptureFixture[str],
) -> None:
    group = TimingSuite("group")
    outer = {"ops": 1.0}

    @group(units=outer)
    def body() -> TimingSample:
        outer["ops"] = 4
        group.name = "changed"
        return TimingSample(0.5, 1, (("ops", 2),), goal_seconds=None)

    assert body().name == "changed.body"
    assert body().rates == (("ops", 4), ("ops", 2))
    group.main()
    assert capsys.readouterr().out == format_suite_timing(body()) + "\n"
    empty = TimingSuite("empty")
    empty.main()
    assert capsys.readouterr().out == ""
    assert empty.target_wait_time_seconds == 0.5


def test_go_snapshots_rate_mutation_after_body_and_preserves_exception() -> None:
    group = TimingSuite("group", target_wait_time_seconds=0.5)
    rates = {"ops": 1.0}
    calls = []

    def body() -> None:
        calls.append(1)
        rates["ops"] = 3

    with patch.object(suite_module, "monotonic", side_effect=[0, 0.5]):
        sample = group.go(body, rates=rates)
    assert sample == TimingSample(0.5, 1, (("ops", 3),), goal_seconds=None)
    assert calls == [1]
    with patch.object(suite_module, "monotonic", side_effect=[0, 0.5]):
        assert group.go(lambda: None).rates == ()

    def failing_body() -> None:
        raise LookupError("stop")

    with (
        patch.object(suite_module, "monotonic", side_effect=[0]),
        pytest.raises(LookupError, match="stop"),
    ):
        group.go(failing_body)
    assert group.benchmarks == []


def test_original_goal_conflicts_and_no_goal_name_format() -> None:
    group = TimingSuite("group")
    with pytest.raises(AssertionError, match="Specified multiple goal units"):
        group(goal_nanos=1, goal_micros=2)
    assert group.benchmarks == []
    assert format_suite_timing(TimingSample(0.5, 1, goal_seconds=None)) == "500 ms "
