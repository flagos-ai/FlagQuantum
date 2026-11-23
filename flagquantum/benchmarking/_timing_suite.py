# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/perf/benchmark_suite.py at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Group stateful Python benchmark callbacks with the original timing policy."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from time import monotonic
from typing import overload

from ._adaptive_timing import TimingSample, measure_suite
from ._timing_report import format_suite_timing

_Benchmark = Callable[[], TimingSample]


class TimingSuite:
    """Own decorator registration, ordered rates and fresh benchmark invocations."""

    def __init__(
        self, name: str, *, target_wait_time_seconds: float | None = None
    ) -> None:
        self.name = name
        self.benchmarks: list[_Benchmark] = []
        self.target_wait_time_seconds = (
            0.5 if target_wait_time_seconds is None else target_wait_time_seconds
        )

    @overload
    def __call__(
        self,
        original_method: _Benchmark,
        /,
        *,
        goal_nanos: float | None = None,
        goal_micros: float | None = None,
        goal_millis: float | None = None,
        units: Mapping[str, float] | None = None,
    ) -> _Benchmark: ...

    @overload
    def __call__(
        self,
        original_method: None = None,
        /,
        *,
        goal_nanos: float | None = None,
        goal_micros: float | None = None,
        goal_millis: float | None = None,
        units: Mapping[str, float] | None = None,
    ) -> Callable[[_Benchmark], _Benchmark]: ...

    def __call__(
        self,
        original_method: _Benchmark | None = None,
        /,
        *,
        goal_nanos: float | None = None,
        goal_micros: float | None = None,
        goal_millis: float | None = None,
        units: Mapping[str, float] | None = None,
    ) -> _Benchmark | Callable[[_Benchmark], _Benchmark]:
        assert (goal_micros is not None) + (goal_millis is not None) + (
            goal_nanos is not None
        ) <= 1, "Specified multiple goal units."
        goal: float | None = None
        if goal_micros is not None:
            goal = goal_micros * 1e-6
        if goal_millis is not None:
            goal = goal_millis * 1e-3
        if goal_nanos is not None:
            goal = goal_nanos * 1e-9

        def wrap(method: _Benchmark) -> _Benchmark:
            def run() -> TimingSample:
                result = method()
                return replace(
                    result,
                    name=self.name + "." + method.__name__,
                    rates=(*(() if units is None else units.items()), *result.rates),
                    goal_seconds=goal,
                )

            self.benchmarks.append(run)
            return run

        return wrap if original_method is None else wrap(original_method)

    def go(
        self, body: Callable[[], object], *, rates: Mapping[str, float] | None = None
    ) -> TimingSample:
        sample = measure_suite(
            body,
            target_seconds=self.target_wait_time_seconds,
            clock=monotonic,
        )
        return replace(sample, rates=tuple(() if rates is None else rates.items()))

    def main(self) -> None:
        for benchmark in self.benchmarks:
            print(format_suite_timing(benchmark()))
