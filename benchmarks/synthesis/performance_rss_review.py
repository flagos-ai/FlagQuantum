"""Interleave fresh RSS workers from two pinned Git snapshots for PF review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from benchmarks.synthesis.performance import environment, trial
from benchmarks.synthesis.performance_compare import _metric
from benchmarks.synthesis.performance_results import pinned_revision, validate_sample
from benchmarks.synthesis.performance_workloads import Workload, workloads
from benchmarks.synthesis.results import _integer, _object


def review(
    repository: Path,
    before: str,
    after: str,
    output: Path,
    *,
    qmod_line: bool = False,
    case: str | None = None,
) -> None:
    repository = repository.resolve()
    sources = {"before": pinned_revision(before), "after": pinned_revision(after)}
    cases: tuple[tuple[str, Workload], ...]
    if case is not None:
        selected = next((w for w in workloads() if w.identifier == case), None)
        if qmod_line or selected is None or selected.refused:
            raise ValueError("select one fixed successful workload for RSS review")
        cases = ((case, selected),)
    else:
        cases = (
            (("qmod_line", Workload("energy-2x2", frontend="qmod", target="line")),)
            if qmod_line
            else tuple(
                (target, Workload("energy-2x2", target=target))
                for target in ("logical", "line")
            )
        )
    snapshots: dict[str, Path] = {}
    records: dict[str, list[dict[str, object]]] = {}
    payload: dict[str, object] = {
        "scope": "Supplementary interleaved local RSS review, not replacement primary data",
        "sources": sources,
        "measurement_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "environment": environment(),
        "protocol": "Five fresh RSS workers per source/target, two warmups; AB/BA alternating rounds; unchanged PF-00 worker",
        "snapshots": {},
        "raw": records,
    }
    witnesses: dict[str, object] = {}
    payload["snapshots"] = witnesses
    with tempfile.TemporaryDirectory(prefix="fq-pf01-review-") as folder:
        for label, revision in sources.items():
            archive = Path(folder) / (label + ".tar")
            subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    "--output=" + str(archive),
                    revision,
                    "flagquantum",
                    "benchmarks/__init__.py",
                    "benchmarks/synthesis",
                ],
                cwd=repository,
                check=True,
                timeout=30,
            )
            snapshot = (Path(folder) / label).resolve()
            snapshot.mkdir()
            with tarfile.open(archive) as source:
                source.extractall(snapshot, filter="data")
            imported = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import flagquantum; print(flagquantum.__file__)",
                ],
                cwd=snapshot,
                text=True,
                timeout=30,
            ).strip()
            if Path(imported).resolve() != snapshot / "flagquantum/__init__.py":
                raise ValueError("worker import did not resolve to pinned snapshot")
            snapshots[label] = snapshot
            witnesses[label] = {
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "verified_import": imported,
                "production_tree": subprocess.check_output(
                    ["git", "rev-parse", revision + ":flagquantum"],
                    cwd=repository,
                    text=True,
                    timeout=15,
                ).strip(),
            }
        for name in (
            "performance.py",
            "performance_probe.py",
            "performance_workloads.py",
            "oracle.py",
            "verification.py",
            "arithmetic_baseline.py",
            "expressions.py",
        ):
            relative = Path("benchmarks/synthesis") / name
            reference = (snapshots["after"] / relative).read_bytes()
            if (snapshots["before"] / relative).read_bytes() != reference or (
                repository / relative
            ).read_bytes() != reference:
                raise ValueError(
                    "RSS review requires the unchanged measurement implementation"
                )
        for target, workload in cases:
            for label in sources:
                records[target + ":" + label] = []
            for repetition in range(5):
                order = (
                    ("before", "after") if repetition % 2 == 0 else ("after", "before")
                )
                for label in order:
                    print(target, repetition + 1, label, flush=True)
                    previous = Path.cwd()
                    try:
                        os.chdir(snapshots[label])
                        sample = trial(
                            workload,
                            "memory",
                            warmups=2,
                            seconds=30,
                        )
                    finally:
                        os.chdir(previous)
                    records[target + ":" + label].append(sample)
                    output.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n"
                    )
                    validate_sample(sample)
                    if sample["status"] != "verified":
                        raise ValueError(
                            "RSS review requires every independent worker to verify"
                        )
        summary: dict[str, object] = {}
        for target, _ in cases:
            left, right = records[target + ":before"], records[target + ":after"]
            expected = _object(left[0]["verification"])["resources"]
            if any(
                _object(s["verification"])["resources"] != expected
                for s in (*left, *right)
            ):
                raise ValueError("RSS review changed circuit or costs")
            summary[target] = {
                "compile_window_rss_bytes": _metric(
                    [
                        _integer(_object(s["memory"])["window_sampled_peak_rss_bytes"])
                        for s in left
                    ],
                    [
                        _integer(_object(s["memory"])["window_sampled_peak_rss_bytes"])
                        for s in right
                    ],
                    16 * 1024**2,
                ),
                "lifetime_high_water_bytes": _metric(
                    [
                        _integer(_object(s["compile_lifetime_high_water"])["bytes"])
                        for s in left
                    ],
                    [
                        _integer(_object(s["compile_lifetime_high_water"])["bytes"])
                        for s in right
                    ],
                    16 * 1024**2,
                ),
            }
        payload["summary"] = summary
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--qmod-line", action="store_true", help="Review Qmod line only"
    )
    selection.add_argument("--case", help="One fixed successful workload identifier")
    args = parser.parse_args()
    review(
        args.repository,
        args.before,
        args.after,
        args.output,
        qmod_line=args.qmod_line,
        case=args.case,
    )


if __name__ == "__main__":
    main()
