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
    goal = -1 if sample.goal_seconds is None else sample.goal_seconds
    if goal != -1 and (
        not isfinite(goal) or goal <= 0 or not isfinite(actual) or actual <= 0
    ):
        raise ValueError("Goal comparisons require positive finite time and goals")
    if goal == -1:
        report = _si2(actual, truncate_hundreds=True) + "s "
    else:
        offset = (log(goal) - log(actual)) / (log(10) / 10.0)
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
            + _si2(goal, truncate_hundreds=True)
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


def _rounded_si2(value: float) -> str:
    """Retain the Python source's rounding at SI and decimal boundaries."""
    unit = " "
    if value < 1:
        for prefix in "munp":
            if value < 1:
                value *= 1000
                unit = prefix
    else:
        for prefix in "kMGT":
            if value > 1000:
                value /= 1000
                unit = prefix
    if 1 <= value < 10:
        digits = str(round(value * 10))
        return f"{digits[:-1]}.{digits[-1]} {unit}"
    if 10 <= value < 100:
        digits_int = round(value)
        if digits_int == 0:
            digits_int = 1
        if digits_int == 100:
            digits_int = 99
        return f" {digits_int} {unit}"
    if 100 <= value < 1000:
        hundreds = round(value / 10) * 10
        if hundreds == 1000:
            hundreds = 990
        return f"{hundreds} {unit}"
    return f"{value} {unit}"


def format_suite_timing(sample: TimingSample) -> str:
    """Render Python source results, retaining zero-division and log failures."""
    actual = sample.total_seconds / sample.total_repetitions
    parts: list[str] = []
    if sample.goal_seconds is not None:
        deviation = round((log(sample.goal_seconds) - log(actual)) / (log(10) / 10.0))
        parts.append("[")
        for position in range(-20, 21):
            if (position < deviation and position < 0) or (
                position > deviation and position > 0
            ):
                parts.append(".")
            elif position == deviation:
                parts.append("*")
            elif position == 0:
                parts.append("|")
            elif deviation < 0:
                parts.append("<")
            else:
                parts.append(">")
        parts.extend(
            (
                "] ",
                _rounded_si2(actual),
                "s (vs ",
                _rounded_si2(sample.goal_seconds),
                "s) ",
            )
        )
    else:
        parts.extend((_rounded_si2(actual), "s "))
    for unit, multiplier in sample.rates:
        parts.extend(
            (
                "(",
                _rounded_si2(
                    sample.total_repetitions / sample.total_seconds * multiplier
                ),
                unit,
                "/s) ",
            )
        )
    if sample.name is not None:
        parts.append(sample.name)
    return "".join(parts)
