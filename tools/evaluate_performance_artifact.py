#!/usr/bin/env python3
"""Re-evaluate committed performance artifacts against hardware thresholds."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from flagquantum.runtime.observability.performance import (
    PerformanceRecord,
    PerformanceThresholds,
    calibrate_cost_model,
    evaluate_performance,
)


def _record(payload: dict[str, object]) -> PerformanceRecord:
    names = {item.name for item in fields(PerformanceRecord)}
    return PerformanceRecord(**{name: payload[name] for name in names})


def evaluate_path(
    path: Path, *, write: bool, baseline_path: Path | None = None
) -> bool:
    payload = json.loads(path.read_text())
    record = _record(payload)
    baseline = (
        _record(json.loads(baseline_path.read_text()))
        if baseline_path is not None
        else None
    )
    thresholds = PerformanceThresholds(
        max_coefficient_of_variation=0.25 if record.world_size >= 8 else 0.10
    )
    payload.update(record.summary())
    payload["cost_model_calibration"] = calibrate_cost_model(record).__dict__
    payload["regression_thresholds"] = thresholds.__dict__
    payload["performance_gate"] = evaluate_performance(
        record, baseline=baseline, thresholds=thresholds
    ).__dict__
    payload["comparison_baseline"] = (
        str(baseline_path) if baseline_path is not None else None
    )
    if write:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return bool(payload["performance_gate"]["passed"])


def resolve_optional_baseline(path: Path | None) -> Path | None:
    """Return an optional baseline only when its artifact is available."""
    if path is None or path.exists():
        return path
    print(
        f"performance baseline not present: {path}; "
        "evaluating intrinsic thresholds only"
    )
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument("--baseline", type=Path)
    baseline_group.add_argument("--baseline-if-present", type=Path)
    args = parser.parse_args()
    requested_baseline = args.baseline or args.baseline_if_present
    if requested_baseline is not None and len(args.paths) != 1:
        parser.error("a baseline requires exactly one current artifact")
    baseline = (
        args.baseline
        if args.baseline is not None
        else resolve_optional_baseline(args.baseline_if_present)
    )
    passed = all(
        evaluate_path(path, write=args.write, baseline_path=baseline)
        for path in args.paths
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
