# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Capture complete locked Python timing, formatting and decorator observations."""

import hashlib
import importlib.util
import json
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch


def main() -> None:
    reference = Path(__file__).resolve().parents[4] / "ftqc-reference-env"
    path = reference / "kickmix/src/kickmix/py/perf/benchmark_suite.py"
    raw = path.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == "955c8e17d69b6ab106f0e8f3aa8e6bb66f664997"
    )
    spec = importlib.util.spec_from_file_location("frozen_timing_suite", path)
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    values = [
        -1,
        0,
        1e-15,
        1e-12,
        0.001,
        0.00995,
        0.995,
        1,
        1.05,
        1.15,
        9.95,
        10,
        12.5,
        98.5,
        99.5,
        100,
        105,
        115,
        995,
        999.9,
        1000,
        1000.1,
        1e16,
    ]
    si = [[value, source._si2(value)] for value in values]
    reports = []
    for seconds in (0.001, 0.5, 3, 30):
        for reps in (1, 3):
            for goal in (None, 1e-6, 0.5, 3):
                value = source._BenchmarkResult(
                    name="group.case",
                    total_seconds=seconds,
                    total_reps=reps,
                    rates=(("ops", 1), ("ops", 2), ("items", 300)),
                    goal_seconds=goal,
                )
                reports.append([seconds, reps, goal, str(value)])
    failures = []
    for seconds, reps, goal in (
        (0, 0, None),
        (0, 1, None),
        (1, 1, 0),
        (1, 1, -1),
        (1, 1, float("inf")),
    ):
        value = source._BenchmarkResult(
            name=None,
            total_seconds=seconds,
            total_reps=reps,
            rates=(("ops", 1),),
            goal_seconds=goal,
        )
        try:
            str(value)
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            failures.append(
                [seconds, reps, str(goal), type(error).__name__, str(error)]
            )
    traces = []
    for target, ticks in (
        (0, []),
        (-1, []),
        (0.5, [0, 0.01, 0.01, 0.49]),
        (0.5, [0, 0.4, 0.4, 0.5]),
        (0.5, [0, 0.001, 0.001, 0.501]),
        (0.5, [0, 0, 0, 0.5]),
        (0.5, [0, -0.1, 0, 0.6]),
    ):
        clock = iter(ticks)
        calls: list[int] = []
        with patch.object(source.time, "monotonic", partial(next, clock)):
            sample = source.BenchmarkSuite("group", target_wait_time_seconds=target).go(
                partial(calls.append, 1), rates={"ops": 1}
            )
        traces.append(
            [
                target,
                ticks,
                len(calls),
                sample.total_reps,
                sample.total_seconds,
                list(sample.rates),
                sample.goal_seconds,
            ]
        )
    registrations = []
    for units in (
        {},
        {"goal_nanos": 13},
        {"goal_micros": 2.8},
        {"goal_millis": 2.7},
        {"goal_nanos": -1e9},
    ):
        group = source.BenchmarkSuite("group")

        def body() -> Any:
            return source._BenchmarkResult(
                name="inner",
                total_seconds=0.5,
                total_reps=3,
                rates=(("ops", 2),),
                goal_seconds=3,
            )

        decorated = group(body, units={"ops": 1}, **units)
        result = decorated()
        registrations.append(
            [
                units,
                result.name,
                result.goal_seconds,
                list(result.rates),
                len(group.benchmarks),
            ]
        )
    output = {
        "source_blob": "955c8e17d69b6ab106f0e8f3aa8e6bb66f664997",
        "si": si,
        "reports": reports,
        "failures": failures,
        "traces": traces,
        "registrations": registrations,
    }
    destination = Path(__file__).resolve().parents[1] / "data/python_timing_suite.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
