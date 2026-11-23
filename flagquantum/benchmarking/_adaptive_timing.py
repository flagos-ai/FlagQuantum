# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix perf.perf.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Source-ordered adaptive timing with explicitly owned samples and clocks."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from time import monotonic, perf_counter_ns
from typing import Literal

_WORD_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class TimingSample:
    total_seconds: float
    total_repetitions: int
    rates: tuple[tuple[str, float], ...] = ()
    goal_seconds: float | None = -1.0
    name: str | None = None

    def with_rate(self, name: str, multiplier: int | float) -> "TimingSample":
        return replace(self, rates=(*self.rates, (name, float(multiplier))))

    def with_goal(
        self, value: float, unit: Literal["ns", "us", "ms"]
    ) -> "TimingSample":
        # Preserve the original successive division rounding for each unit.
        seconds = value / 1000
        if unit in ("ns", "us"):
            seconds /= 1000
        if unit == "ns":
            seconds /= 1000
        return replace(self, goal_seconds=seconds)


@dataclass(frozen=True)
class TimingCase:
    name: str
    body: Callable[[float], Sequence[TimingSample]]


def _next_repetitions(
    name: str, target: float, seconds: float, total_repetitions: int, limit: int
) -> int | None:
    if limit == 0:
        raise ValueError(
            f"During '{name}', rep_limit hit SIZE_MAX. This suggests the "
            "'benchmark_go' block is taking 0 time due to being compiled away."
        )
    if seconds <= 0:
        return limit
    expected = (target - seconds) * total_repetitions / seconds
    if expected > 1e15:
        raise ValueError(
            f"During '{name}', expected repetitions exceeded 1e15. This suggests "
            "the 'benchmark_go' block is taking 0 time due to being compiled away."
        )
    repetitions = int(expected)
    if repetitions < total_repetitions * 0.1:
        return None
    return max(1, min(repetitions, limit))


def measure(
    name: str,
    body: Callable[[], None],
    *,
    target_seconds: float = 0.5,
    clock: Callable[[], int] = perf_counter_ns,
) -> TimingSample:
    """Run the original adaptive batches using one monotonic nanosecond clock."""
    if not isfinite(target_seconds) or target_seconds < 0:
        raise ValueError("Target seconds must be finite and nonnegative")
    total_repetitions = 0
    total_seconds = 0.0
    limit = 1
    while total_seconds < target_seconds:
        repetitions = _next_repetitions(
            name, target_seconds, total_seconds, total_repetitions, limit
        )
        if repetitions is None:
            break
        start = clock()
        for _ in range(repetitions):
            body()
        elapsed = clock() - start
        if elapsed < 0:
            raise ValueError("Timing clock must be monotonic")
        total_repetitions = (total_repetitions + repetitions) & _WORD_MASK
        total_seconds += elapsed / 1_000_000_000
        limit = (limit * 100) & _WORD_MASK
    return TimingSample(total_seconds, total_repetitions)


def measure_suite(
    body: Callable[[], object],
    *,
    target_seconds: float = 0.5,
    rates: Sequence[tuple[str, float]] = (),
    clock: Callable[[], float] = monotonic,
) -> TimingSample:
    """Retain the Python source's float clock, floor division and loop batches."""
    repetitions = 0
    seconds = 0.0
    limit = 1
    while seconds < target_seconds:
        count = limit
        if seconds > 0:
            count = int((target_seconds - seconds) * repetitions // seconds)
            if count < repetitions * 0.1:
                break
            count = max(1, min(count, limit))
        if count > 50:
            groups = count >> 3
            count = groups << 3
            start = clock()
            for _ in range(groups):
                body()
                body()
                body()
                body()
                body()
                body()
                body()
                body()
            end = clock()
        else:
            start = clock()
            for _ in range(count):
                body()
            end = clock()
        seconds += end - start
        repetitions += count
        limit *= 100
    return TimingSample(seconds, repetitions, tuple(rates), goal_seconds=None)
