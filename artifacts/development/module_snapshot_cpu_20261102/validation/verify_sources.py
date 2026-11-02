"""Audit retained trials and reproduce aggregate development results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parents[1]


def aggregate():
    publication = json.loads((EVIDENCE / "publication.json").read_text())
    for relative, digest in publication["published_sha256"].items():
        assert hashlib.sha256((EVIDENCE / relative).read_bytes()).hexdigest() == digest
    phases = {}
    for phase in ("before", "after"):
        trials = []
        for trial in range(3):
            raw = gzip.decompress(
                (EVIDENCE / phase / f"trial-{trial}.json.gz").read_bytes()
            )
            summary = json.loads(raw)
            assert summary["revision"] == publication[phase]["source_commit"]
            assert summary["source_dirty"] is False
            assert summary["repeats"] == 5 and summary["iterations"] == 30
            assert len(summary["cases"]) == 10
            producer = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{summary['revision']}:benchmarks/module_execution_matrix.py",
                ],
                cwd=ROOT,
                timeout=20,
            )
            assert hashlib.sha256(producer).hexdigest() == summary["producer_sha256"]
            points = {}
            for case in summary["cases"]:
                assert case["status"] == "completed"
                shape = tuple(case["case"][key] for key in ("width", "depth", "ring"))
                assert len(case["measurements"]) == 15
                for point in case["measurements"]:
                    samples = point["seconds_per_call"]
                    assert len(samples) == 5 and all(
                        math.isfinite(x) and x > 0 for x in samples
                    )
                    assert statistics.median(samples) == point["median_seconds"]
                    key = (*shape, point["demand"], point["route"])
                    assert key not in points
                    points[key] = point["median_seconds"]
            assert len(points) == 150
            trials.append(points)
        assert trials[0].keys() == trials[1].keys() == trials[2].keys()
        phases[phase] = trials
    assert phases["before"][0].keys() == phases["after"][0].keys()
    points = []
    medians = {}
    for key in sorted(phases["before"][0]):
        record = dict(
            zip(("width", "depth", "ring", "demand", "route"), key, strict=True)
        )
        for phase in ("before", "after"):
            values = [trial[key] for trial in phases[phase]]
            median = statistics.median(values)
            record[phase] = {
                "process_medians_seconds": values,
                "median_seconds": median,
                "mad_seconds": statistics.median(abs(x - median) for x in values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
            medians[phase, key] = median
        record["before_over_after"] = (
            record["before"]["median_seconds"] / record["after"]["median_seconds"]
        )
        points.append(record)
    contexts = []
    for demand in ("parameters", "inputs", "both", "no_grad", "inference"):
        record = {"demand": demand}
        for route in ("default_module", "jax_module", "jax_kernel_diagnostic"):
            ratios = [
                p["before_over_after"]
                for p in points
                if p["demand"] == demand and p["route"] == route
            ]
            record[route] = {
                "median_ratio": statistics.median(ratios),
                "min_ratio": min(ratios),
                "max_ratio": max(ratios),
            }
        contexts.append(record)
    budgets = {}
    for phase in ("before", "after"):
        public_passes = default_passes = 0
        for key in phases[phase][0]:
            if key[-1] != "default_module":
                continue
            default = medians[phase, key]
            public = medians[phase, (*key[:-1], "jax_module")]
            kernel = medians[phase, (*key[:-1], "jax_kernel_diagnostic")]
            public_passes += int(public <= 1.15 * kernel + 5e-6)
            default_passes += int(default <= 1.25 * min(default, public) + 5e-6)
        budgets[phase] = {
            "public_overhead_passes": public_passes,
            "default_accessibility_passes": default_passes,
            "total": 50,
        }
    return {
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "closure_status": "not_ready_to_close_918",
        "numerical_checks": 900,
        "fresh_shape_processes": 60,
        "aggregation": "median of three process medians, then ratio; no confidence interval with n=3",
        "contexts": contexts,
        "proposed_development_budgets": budgets,
        "points": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = aggregate()
    target = EVIDENCE / "aggregate.json"
    if args.write:
        target.write_text(json.dumps(result, indent=2) + "\n")
    else:
        assert result == json.loads(target.read_text())
    print("Source hashes, 60 processes, 900 numerical checks and aggregate verified")
