#!/usr/bin/env python3
"""Build bootstrap confidence intervals for matched value-and-gradient artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

SCHEMA = "flagquantum.statevector.mlsys_bootstrap.v1"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_median(
    samples: list[float], *, resamples: int, seed: int
) -> dict[str, Any]:
    if len(samples) < 3:
        raise ValueError("bootstrap requires at least three samples")
    generator = random.Random(seed)
    estimates = [
        statistics.median(generator.choices(samples, k=len(samples)))
        for _ in range(resamples)
    ]
    return {
        "median_seconds": statistics.median(samples),
        "confidence_level": 0.95,
        "confidence_interval_seconds": [
            percentile(estimates, 0.025),
            percentile(estimates, 0.975),
        ],
    }


def bootstrap_ratio(
    numerator: list[float],
    denominator: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        left = statistics.median(generator.choices(numerator, k=len(numerator)))
        right = statistics.median(
            generator.choices(denominator, k=len(denominator))
        )
        estimates.append(left / right)
    return {
        "ratio": statistics.median(numerator) / statistics.median(denominator),
        "confidence_level": 0.95,
        "confidence_interval": [
            percentile(estimates, 0.025),
            percentile(estimates, 0.975),
        ],
    }


def system_name(payload: dict[str, Any]) -> str:
    provider = payload.get("external_baseline_isolation", {}).get("provider", "")
    benchmark = str(payload.get("benchmark", "")).lower()
    if "torchquantum" in provider.lower() or "torchquantum" in benchmark:
        return "TorchQuantum-Dist"
    if "pennylane" in provider.lower() or "pennylane" in benchmark:
        return "PennyLane"
    return "FlagQuantum"


def world_size(payload: dict[str, Any]) -> int:
    return int(
        payload.get("world_size")
        or payload.get("environment", {}).get("world_size")
        or 1
    )


def build_report(
    paths: list[Path], *, resamples: int = 10_000, seed: int = 440044
) -> dict[str, Any]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = [
            float(value)
            for value in payload["value_and_grad"]["samples_seconds"]
        ]
        system, world = system_name(payload), world_size(payload)
        key = (system, world)
        if key in rows:
            raise ValueError(f"duplicate system/world pair: {key}")
        mean = statistics.fmean(samples)
        rows[key] = {
            "system": system,
            "world_size": world,
            "source": str(path),
            "sample_count": len(samples),
            "samples_seconds": samples,
            "coefficient_of_variation": (
                statistics.pstdev(samples) / mean if mean else 0.0
            ),
            "median": bootstrap_median(
                samples, resamples=resamples, seed=seed + world
            ),
            "stationarity_warning": (
                "high_cv_or_monotonic_tail; bootstrap assumes exchangeable samples"
                if (
                    statistics.pstdev(samples) / mean > 0.1
                    or (
                        len(samples) >= 3
                        and (
                            samples[-3:] == sorted(samples[-3:])
                            or samples[-3:] == sorted(samples[-3:], reverse=True)
                        )
                    )
                )
                else None
            ),
        }
    ratios = []
    for (system, world), row in sorted(rows.items()):
        baseline = rows.get(("FlagQuantum", world))
        if system == "FlagQuantum" or baseline is None:
            continue
        ratios.append(
            {
                "numerator": system,
                "denominator": "FlagQuantum",
                "world_size": world,
                **bootstrap_ratio(
                    row["samples_seconds"],
                    baseline["samples_seconds"],
                    resamples=resamples,
                    seed=seed + 1000 + world,
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "artifact_class": "derived_statistical_report",
        "claim_evidence_type": "development_uncertainty_analysis",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "bootstrap": {
            "method": "independent_nonparametric_bootstrap_of_medians",
            "resamples": resamples,
            "seed": seed,
            "confidence_level": 0.95,
        },
        "points": list(rows.values()),
        "ratios": ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=440044)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 1_000:
        raise ValueError("--resamples must be at least 1000")
    report = build_report(
        args.artifacts, resamples=args.resamples, seed=args.seed
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
