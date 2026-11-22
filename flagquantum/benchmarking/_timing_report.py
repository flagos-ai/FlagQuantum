# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix main.perf.cc at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Original benchmark selection and reports with scoped native result values."""

from collections.abc import Sequence
from math import ceil, copysign, floor, isfinite, isnan, log

from ..compiler.formats.kmx_spin_report import _si2
from ._adaptive_timing import TimingCase, TimingSample


def select_benchmarks(
    cases: Sequence[TimingCase], filters: str | None
) -> tuple[TimingCase, ...]:
    if filters is None:
        return tuple(cases)
    selected = []
    for pattern in filters.split(","):
        matches = [
            case
            for case in cases
            if (
                case.name.startswith(pattern[:-1])
                if pattern.endswith("*")
                else case.name == pattern
            )
        ]
        if not matches:
            available = "".join(f"    {case.name}\n" for case in cases)
            raise ValueError(
                f"No benchmark matching filter '{pattern}'. "
                f"Available benchmarks are:\n{available}"
            )
        selected.extend(matches)
    return tuple(selected)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator != 0:
        return numerator / denominator
    if isnan(numerator):
        return numerator
    if numerator == 0:
        return float("nan")
    return copysign(float("inf"), copysign(1, numerator) * copysign(1, denominator))


def format_timing(name: str, sample: TimingSample) -> str:
    if sample.total_repetitions < 0:
        raise ValueError("Timing repetitions must be unsigned")
    actual = _ratio(sample.total_seconds, float(sample.total_repetitions))
    if sample.goal_seconds != -1 and (
        not isfinite(sample.goal_seconds)
        or sample.goal_seconds <= 0
        or not isfinite(actual)
        or actual <= 0
    ):
        raise ValueError("Goal comparisons require positive finite time and goals")
    if sample.goal_seconds == -1:
        report = _si2(actual, truncate_hundreds=True) + "s "
    else:
        offset = (log(sample.goal_seconds) - log(actual)) / (log(10) / 10.0)
        deviation = floor(offset + 0.5) if offset >= 0 else ceil(offset - 0.5)
        marks = []
        for index in range(-20, 21):
            if (index < deviation and index < 0) or (index > deviation and index > 0):
                marks.append(".")
            elif index == deviation:
                marks.append("*")
            elif index == 0:
                marks.append("|")
            else:
                marks.append("<" if deviation < 0 else ">")
        report = (
            "["
            + "".join(marks)
            + "] "
            + _si2(actual, truncate_hundreds=True)
            + "s (vs "
            + _si2(sample.goal_seconds, truncate_hundreds=True)
            + "s) "
        )
    for unit, multiplier in sample.rates:
        report += (
            "("
            + _si2(
                _ratio(float(sample.total_repetitions), sample.total_seconds)
                * multiplier,
                truncate_hundreds=True,
            )
            + unit
            + "/s) "
        )
    return report + name + "\n"
