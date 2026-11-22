"""Compare complete original benchmark support using scoped native ownership."""

import json
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks import reversible_phase_timing
from benchmarks.reversible_phase_timing import benchmark_options, run_increment
from flagquantum.benchmarking._adaptive_timing import (
    TimingCase,
    TimingSample,
    _next_repetitions,
    measure,
)
from flagquantum.benchmarking._timing_report import format_timing, select_benchmarks
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx_spin_report import _si2
from flagquantum.ecosystem._command_arguments import find_argument

pytestmark = pytest.mark.unit


def observations() -> dict[str, object]:
    value: object = json.loads(
        (Path(__file__).with_name("data") / "benchmark_support.json").read_text()
    )
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Invalid complete benchmark observations")
    return value


def test_complete_source_si_boundaries() -> None:
    cases = observations()["si"]
    assert isinstance(cases, list)
    for value, expected in cases:
        assert _si2(value, truncate_hundreds=True) + "\n" == expected
    assert _si2(123.456) == "123  "


def test_complete_source_result_reports() -> None:
    cases = observations()["reports"]
    assert isinstance(cases, list)
    for args, code, expected, error in cases:
        assert code == 0 and error == ""
        _, seconds, repetitions, goal, *rates = args
        sample = TimingSample(
            float(seconds), int(repetitions), goal_seconds=float(goal)
        )
        for index in range(0, len(rates), 2):
            sample = sample.with_rate(rates[index], float(rates[index + 1]))
        assert format_timing("custom", sample) == expected, args


def test_complete_goal_units_and_rate_overloads() -> None:
    cases = observations()["goals"]
    assert isinstance(cases, list)
    initial = TimingSample(0.001, 3)
    for args, code, output, error in cases:
        assert code == 0 and error == ""
        expected = list(map(float, output.split()))
        value = float(args[1])
        assert [
            initial.with_goal(value, unit).goal_seconds for unit in ("ns", "us", "ms")
        ] == expected
    assert initial.goal_seconds == -1 and initial.rates == ()
    rates = observations()["rates"]
    assert isinstance(rates, list)
    _, code, output, error = rates
    assert code == 0 and error == "" and isinstance(output, str)
    expected_rates = tuple(
        (line.split()[0], float(line.split()[1])) for line in output.splitlines()
    )
    integers = (
        -128,
        255,
        -32768,
        65535,
        -(1 << 31),
        (1 << 32) - 1,
        -(1 << 63),
        (1 << 64) - 1,
    )
    decorated = initial
    for (name, _), value in zip(
        expected_rates, (*integers, 1.23456789, 1234), strict=True
    ):
        decorated = decorated.with_rate(name, value)
    assert decorated.rates == expected_rates and initial.rates == ()


def test_complete_source_selection_and_empty_result_behavior() -> None:
    cases = observations()["cli"]
    assert isinstance(cases, list)
    names = ("timing_alpha", "timing_beta", "empty", "custom")
    samples = {
        "timing_alpha": TimingSample(0.002, 10)
        .with_rate("ops", 32)
        .with_goal(200, "us"),
        "timing_beta": TimingSample(0.01, 1).with_rate("items", 3),
        "empty": None,
        "custom": TimingSample(1, 1),
    }
    registry = tuple(TimingCase(name, lambda _: ()) for name in names)
    for args, code, expected, error in cases:
        argv = ["benchmark", *args]
        if "--unknown" in args or "-1" in args:
            continue  # Existing argument owner is checked independently below.
        filters = find_argument("--only", argv)
        if code:
            with pytest.raises(ValueError) as failure:
                select_benchmarks(registry, filters)
            assert str(failure.value) == error
        else:
            result = ""
            for case in select_benchmarks(registry, filters):
                sample = samples[case.name]
                if sample is not None:
                    result += format_timing(case.name, sample)
            assert result == expected


@dataclass
class ControlledClock:
    cost: int
    calls: int = 0
    nanos: int = 0

    def body(self) -> None:
        self.calls += 1
        self.nanos += self.cost

    def read(self) -> int:
        return self.nanos


def test_complete_source_adaptive_batches() -> None:
    cases = observations()["adaptive"]
    assert isinstance(cases, list)
    for args, code, output, error in cases:
        assert code == 0 and error == ""
        target, cost = float(args[1]), int(args[2])
        if output.startswith("error "):
            # The source optimizer collapses its empty callback's huge loops.
            # Replay the same planner states without invoking billions of bodies.
            total = 0
            limit = 1
            seconds = 0.0
            with pytest.raises(ValueError) as failure:
                while True:
                    reps = _next_repetitions("adaptive", target, seconds, total, limit)
                    assert reps is not None
                    total = (total + reps) & ((1 << 64) - 1)
                    seconds += reps * cost / 1e9
                    limit = (limit * 100) & ((1 << 64) - 1)
            assert str(failure.value) + "\n" == output.removeprefix("error ")
            continue
        clock = ControlledClock(cost)
        result = measure(
            "adaptive", clock.body, target_seconds=target, clock=clock.read
        )
        expected_calls, expected_reps, expected_seconds, result_count = output.split()
        assert clock.calls == int(expected_calls)
        assert result.total_repetitions == int(expected_reps)
        assert result.total_seconds == float(expected_seconds)
        assert result_count == "1"


@pytest.mark.parametrize("target", [-1, float("nan"), float("inf")])
def test_native_invalid_clock_and_target(target: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        measure("invalid", lambda: None, target_seconds=target)
    clock = iter((2, 1))
    with pytest.raises(ValueError, match="monotonic"):
        measure("invalid", lambda: None, clock=lambda: next(clock))


@pytest.mark.parametrize(
    "sample",
    [
        TimingSample(1, -1),
        TimingSample(1, 1, goal_seconds=0),
        TimingSample(1, 1, goal_seconds=float("inf")),
        TimingSample(0, 1, goal_seconds=1),
        TimingSample(0, 0, goal_seconds=1),
        TimingSample(float("nan"), 1, goal_seconds=1),
    ],
)
def test_native_refuses_undefined_goal_boundary(sample: TimingSample) -> None:
    with pytest.raises(ValueError):
        format_timing("invalid", sample)


def test_real_cli_argument_owner_and_physical_increment() -> None:
    assert benchmark_options(["benchmark"]) == (None, 0.5)
    assert benchmark_options(["benchmark", "--target_seconds", "0"]) == (None, 0.0)
    assert (
        benchmark_options(
            ["benchmark", "--only=increment_*", "--target_seconds", "0.001"]
        )[0]
        == "increment_*"
    )
    for args in (
        ["--unknown"],
        ["--target_seconds", "nan"],
        ["--target_seconds", "-1"],
    ):
        with pytest.raises(ValueError):
            benchmark_options(["benchmark", *args])
    for controlled in (False, True):
        sample = run_increment(controlled, 0.001)
        assert sample.total_repetitions >= 1 and sample.total_seconds > 0
        assert format_timing("increment", sample).endswith("s increment\n")


def test_real_main_and_record_count_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--only", "increment_unconditional", "--target_seconds", "0.001"],
    )
    runpy.run_path(str(Path(reversible_phase_timing.__file__)), run_name="__main__")
    assert capsys.readouterr().out.endswith("records/s) increment_unconditional\n")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--target_seconds", "0"])
    with pytest.raises(ValueError, match="Goal comparisons"):
        reversible_phase_timing.main()
    with (
        patch(
            "benchmarks.reversible_phase_timing.physical_increment",
            return_value=(ReversiblePhaseProgram(1, 0, ()), ()),
        ),
        pytest.raises(RuntimeError, match="instruction count changed"),
    ):
        run_increment(False, 0.001)


def test_body_exception_preserves_caller_state() -> None:
    calls = []

    def fail() -> None:
        calls.append(1)
        raise RuntimeError("body failure")

    with pytest.raises(RuntimeError, match="body failure"):
        measure("failing", fail)
    assert calls == [1]


def test_same_name_callbacks_and_multiple_results() -> None:
    first = TimingCase("same", lambda _: (TimingSample(1, 1), TimingSample(3, 1)))
    second = TimingCase("same", lambda _: (TimingSample(2, 1),))
    selected = select_benchmarks((first, second), "same")
    assert selected == (first, second)
    actual = "".join(
        format_timing(case.name, sample)
        for case in selected
        for sample in case.body(0.5)
    )
    expected = observations()["duplicates"]
    assert isinstance(expected, list) and expected[1] == 0 and actual == expected[2]
