"""Compare pinned PF receipts without treating failed samples as speedups."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .performance_bundle import objects
from .performance_results import read
from .results import _integer, _object


def _distribution(values: list[int]) -> dict[str, float]:
    if not values:
        raise ValueError("missing complete measurement cohort")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _metric(before: list[int], after: list[int], floor: int) -> dict[str, object]:
    if len(before) != len(after):
        raise ValueError("measurement cohort counts differ")
    left, right = _distribution(before), _distribution(after)
    delta = left["median"] - right["median"]
    fraction = delta / left["median"] if left["median"] else 0.0
    return {
        "before": left,
        "after": right,
        "reduction": {
            "absolute": delta,
            "fraction": fraction,
            "meets_review_line": fraction >= 0.1 and delta >= floor,
        },
    }


def _cohort(row: dict[str, object], mode: str) -> list[dict[str, object]]:
    samples = [s for s in objects(row["raw"]) if s["mode"] == mode]
    workload = _object(row["workload"])
    expected = "resource_refused" if workload["refused"] else "verified"
    if not samples or any(s["status"] != expected for s in samples):
        raise ValueError("incomplete primary cohort cannot establish a comparison")
    return samples


def _e1(samples: list[dict[str, object]]) -> list[int]:
    values = []
    for sample in samples:
        verification = _object(sample["verification"])
        if "e1" not in verification:
            continue  # Expected resource refusal has no executable result.
        e1 = _object(verification["e1"])
        if e1["status"] == "verified":
            values.append(_integer(e1["e0_plus_run_plus_read_ns"]))
        elif e1["status"] != "not_run_state_budget":
            raise ValueError("failed execution cannot establish an E1 comparison")
    if values and len(values) != len(samples):
        raise ValueError("partially verified execution cohort")
    return values


def compare(before: Path, after: Path) -> dict[str, object]:
    """Validate both complete receipts and compare identical request cohorts."""
    left, right = read(before), read(after)
    for field in ("configuration", "original_resource_hashes", "not_selected"):
        if left[field] != right[field]:
            raise ValueError("comparison protocol or workload selection differs")
    for name, digest in _object(left["source_hashes"]).items():
        if _object(right["source_hashes"]).get(name) != digest:
            raise ValueError("measurement implementation changed between cohorts")
    for key in (
        "python",
        "platform",
        "cpu_model",
        "logical_cpus",
        "threads",
        "compile_workers",
        "dependencies",
        "gc_enabled",
        "gc_thresholds",
        "quota",
        "affinity",
    ):
        if _object(left["environment"])[key] != _object(right["environment"])[key]:
            raise ValueError("comparison environment differs: " + key)
    before_rows = {str(r["id"]): r for r in objects(left["rows"])}
    after_rows = {str(r["id"]): r for r in objects(right["rows"])}
    if before_rows.keys() != after_rows.keys():
        raise ValueError("comparison workload IDs differ")
    rows: dict[str, object] = {}
    for identifier, old in before_rows.items():
        new = after_rows[identifier]
        old_timing, new_timing = _cohort(old, "timing"), _cohort(new, "timing")
        # Includes canonical circuit hash, width, workspace, histogram and depth.
        old_verified = _object(old_timing[0]["verification"])
        for sample in (*objects(old["raw"]), *objects(new["raw"])):
            if sample["status"] == "verified":
                if _object(sample["verification"])["resources"] != old_verified.get(
                    "resources"
                ):
                    raise ValueError("selected circuit or resources changed")
            elif sample["status"] == "resource_refused":
                if sample["diagnostic_codes"] != old_timing[0]["diagnostic_codes"]:
                    raise ValueError("refusal diagnostic changed")
        old_e1, new_e1 = _e1(old_timing), _e1(new_timing)
        if bool(old_e1) != bool(new_e1):
            raise ValueError("execution verification scope changed")
        rows[identifier] = {
            "e0_ns": _metric(
                [_integer(s["e0_ns"]) for s in old_timing],
                [_integer(s["e0_ns"]) for s in new_timing],
                20_000_000,
            ),
            "e1_ns": _metric(old_e1, new_e1, 20_000_000) if old_e1 else None,
            "sampled_compile_rss_bytes": _metric(
                [
                    _integer(_object(s["memory"])["window_sampled_peak_rss_bytes"])
                    for s in _cohort(old, "memory")
                ],
                [
                    _integer(_object(s["memory"])["window_sampled_peak_rss_bytes"])
                    for s in _cohort(new, "memory")
                ],
                16 * 1024**2,
            ),
        }
    return {
        "source_revisions": {
            "before": left["source_revision"],
            "after": right["source_revision"],
        },
        "claims": "Scoped local PF-01 development comparison; not release, Rust or hardware evidence",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(compare(args.before, args.after), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
