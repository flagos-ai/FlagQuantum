"""Freeze both PF-00 cohorts and derive their explicit configuration selection."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

from .performance_results import digest_summary, read
from .results import _integer, _object


def objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("expected receipt records")
    return [_object(item) for item in value]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(source: Path, target: Path) -> None:
    with (
        target.open("wb") as output,
        gzip.GzipFile(
            filename="", fileobj=output, mode="wb", compresslevel=6, mtime=0
        ) as compressed,
        source.open("rb") as raw,
    ):
        shutil.copyfileobj(raw, compressed)


def bundle(initial: Path, correction: Path, output: Path) -> dict[str, object]:
    inputs = {"initial": initial, "line_correction": correction}
    data = {name: read(path) for name, path in inputs.items()}
    primary, repaired = data["initial"], data["line_correction"]
    if (
        primary["production_tree"] != repaired["production_tree"]
        or primary["original_resource_hashes"] != repaired["original_resource_hashes"]
    ):
        raise ValueError("cohorts use different production trees or resource baselines")
    if primary["configuration"] != repaired["configuration"]:
        raise ValueError("cohort measurement protocols differ")
    for key in (
        "python",
        "platform",
        "cpu_model",
        "logical_cpus",
        "threads",
        "compile_workers",
        "dependencies",
        "gc_enabled",
        "quota",
    ):
        if (
            _object(primary["environment"])[key]
            != _object(repaired["environment"])[key]
        ):
            raise ValueError("cohort environment differs: " + key)
    driver = "benchmarks/synthesis/performance.py"
    if (
        _object(primary["source_hashes"])[driver]
        != _object(repaired["source_hashes"])[driver]
    ):
        raise ValueError("timing driver changed across the oracle correction")
    original_rows = objects(primary["rows"])
    corrected_rows = objects(repaired["rows"])
    expected = {
        f"energy-2x2:{frontend}:inline:line:success"
        for frontend in ("scalar", "qfunc", "qmod")
    }
    if len(original_rows) != 125 or {row["id"] for row in corrected_rows} != expected:
        raise ValueError(
            "expected complete initial inventory and exactly three corrected line targets"
        )
    if any(
        sample["status"] != "verified"
        for row in corrected_rows
        for sample in objects(row["raw"])
    ):
        raise ValueError("corrected line cohort is not completely verified")
    selected = {str(row["id"]): ("initial", row) for row in original_rows}
    selected.update(
        {str(row["id"]): ("line_correction", row) for row in corrected_rows}
    )
    output.mkdir(parents=True, exist_ok=True)
    cohorts = {}
    summaries = {}
    for name, payload in data.items():
        target = output / f"pf00-{name}.json.gz"
        freeze(inputs[name], target)
        cohorts[name] = {
            "file": target.name,
            "source_revision": payload["source_revision"],
            "production_tree": payload["production_tree"],
            "uncompressed_sha256": sha256(inputs[name]),
            "gzip_sha256": sha256(target),
            "uncompressed_bytes": inputs[name].stat().st_size,
            "gzip_bytes": target.stat().st_size,
            "attempted_configurations": len(objects(payload["rows"])),
            "raw_statuses": dict(
                Counter(
                    str(s["status"])
                    for row in objects(payload["rows"])
                    for s in objects(row["raw"])
                )
            ),
        }
        summaries[name] = {
            str(row["id"]): row for row in objects(digest_summary(payload)["rows"])
        }
    rows = []
    incomplete = []
    for key, (cohort, row) in selected.items():
        summary = summaries[cohort][key]
        summary.update(cohort=cohort, source_revision=data[cohort]["source_revision"])
        verified = [
            _object(s["verification"])["resources"]
            for s in objects(row["raw"])
            if s["status"] == "verified"
        ]
        summary["resources"] = verified[0] if verified else None
        rows.append(summary)
        gaps = [
            {"mode": s["mode"], "status": s["status"], "error": s.get("error")}
            for s in objects(row["raw"])
            if s["status"] not in {"verified", "resource_refused"}
        ]
        if gaps:
            incomplete.append({"id": key, "cohort": cohort, "gaps": gaps})

    def statuses(mode: str) -> dict[str, int]:
        return dict(
            Counter(
                str(s["status"])
                for _, row in selected.values()
                for s in objects(row["raw"])
                if s["mode"] == mode
            )
        )

    coverage = {
        "unique_configurations": len(selected),
        "attempted_configurations": len(original_rows) + len(corrected_rows),
        "original_corpus_configurations": 90,
        "selected_primary_statuses": statuses("timing"),
        "selected_memory_statuses": statuses("memory"),
        "selected_diagnostic_statuses": statuses("diagnostic"),
        "complete_primary_configurations": sum(
            _object(row["timing"])["sample_count"] == 5 for _, row in selected.values()
        ),
    }
    manifest = {
        "schema": "flagquantum.synthesis.python_performance_manifest.v1",
        "production_baseline_revision": primary["production_baseline_revision"],
        "production_tree": primary["production_tree"],
        "cohorts": cohorts,
        "selection": {key: cohort for key, (cohort, _) in selected.items()},
        "coverage": coverage,
        "preserved_initial_oracle_failures": sorted(expected),
        "incomplete_selected_records": incomplete,
        "invariants": [
            "Same accepted production tree and byte-identical timing driver.",
            "Original 601f179 data, including failed line-oracle records, are preserved.",
            "Correction changes only oracle coordinates, source validation and diagnostic exception cleanup; original diagnostic failures are not retagged.",
            "No speedup, Rust necessity, hardware or release claim.",
        ],
    }
    cold = {}
    for name, payload in data.items():
        values = [
            _integer(s["cold_start_to_import_ns"])
            for s in objects(payload["cold"])
            if s["status"] == "cold"
        ]
        cold[name] = {
            "n": len(values),
            "median_ns": statistics.median(values) if values else None,
            "min_ns": min(values) if values else None,
            "max_ns": max(values) if values else None,
        }
    overview = {
        "schema": "flagquantum.synthesis.python_performance_overview.v1",
        "cohort_source_revisions": {
            name: p["source_revision"] for name, p in data.items()
        },
        "production_baseline_revision": primary["production_baseline_revision"],
        "coverage": coverage,
        "environment": primary["environment"],
        "configuration": primary["configuration"],
        "cold_startup": cold,
        "rows": rows,
        "incomplete_selected_records": incomplete,
        "interpretation": "Primary E0/RSS are unprofiled. Stage spans/RSS are separate intrusive diagnostics. Full matrix has one primary RSS observation per configuration; no percentile or reliable memory-benefit claim.",
    }
    for name, payload in [
        ("pf00-manifest.json", manifest),
        ("pf00-summary.json", overview),
    ]:
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--line-correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bundle(args.initial, args.line_correction, args.output)
    print(json.dumps(result["coverage"], indent=2))


if __name__ == "__main__":
    main()
