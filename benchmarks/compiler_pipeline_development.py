"""Inspect one complete compiler source tree with unchanged independent acceptance."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import flagquantum

Case = tuple[int, str, str, str]
_HARNESS = "9aa9260920984e2ca05bcb293abb885fccd9c144"
_SOURCES = (
    "flagquantum/compiler/lowering/native_block_optimization.py",
    "flagquantum/compiler/lowering/native_gate_legalization.py",
    "flagquantum/compiler/optimization/two_qubit_optimization.py",
    "flagquantum/compiler/topology/sabre.py",
    "flagquantum/core/ir/__init__.py",
)


class Observation(TypedDict):
    case: Case
    metrics: dict[str, object]
    normalized_output_sha256: str
    samples_seconds: list[float]
    median_seconds: float


class DevelopmentReport(TypedDict):
    evidence_class: str
    implementation_commit: str
    source_commit: str
    source_sha256: dict[str, str]
    runner_sha256: str
    python: str
    platform: str
    limits: str
    applications: list[Observation]
    latency: list[Observation]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, timeout=15
    ).strip()


def _observe(case: Case, harness: Path, *, samples: int, warmups: int) -> Observation:
    from sdkbench.candidates.adapters import load_adapter, workflow_configuration
    from sdkbench.candidates.applications import fixture
    from sdkbench.p0 import Bundle
    from sdkbench.p0.evaluators import routing

    width, family, topology, workflow = case
    bundle = Bundle(harness / "docs/implementation")
    acceptance = bundle.cases[
        "ROUTE-GRID-08" if width == 8 else f"ROUTE-LINE-{width:02d}"
    ]
    selected = fixture(width, family, topology, "01" * 32)
    seed = bundle.plan["execution_profile"]["compiler_seeds"][0]
    config = workflow_configuration(workflow)
    adapter = load_adapter(
        SimpleNamespace(
            configuration={
                "options": {
                    "sdk": config["sdk"],
                    "workflow": workflow,
                    "compiler_options": config,
                }
            }
        )
    )
    expected = None
    elapsed: list[float] = []
    for index in range(samples + warmups):
        start = time.perf_counter()
        payload = adapter.execute("route", selected, seed)
        seconds = time.perf_counter() - start
        metrics = routing(payload, selected, None, acceptance, 0)
        if payload["final_layout"] != list(range(width)):
            raise AssertionError("Output layout is not restored")
        normalized = json.dumps(
            {key: payload[key] for key in ("operations", "final_layout")},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if expected is None:
            expected = normalized
        if normalized != expected:
            raise AssertionError("Repeated compilation changed normalized output")
        if index >= warmups:
            elapsed.append(seconds)
    assert expected is not None
    return {
        "case": case,
        "metrics": metrics,
        "normalized_output_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "samples_seconds": elapsed,
        "median_seconds": statistics.median(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sdkbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    harness = args.sdkbench_root.resolve()
    if Path(flagquantum.__file__).resolve().parents[1] != source:
        parser.error("Imported FlagQuantum does not match --source-root")
    if _git(harness, "rev-parse", "HEAD") != _HARNESS or _git(
        harness, "status", "--porcelain"
    ):
        parser.error("The pinned SDKBench checkout must be unchanged")
    if _git(source, "diff", "HEAD", "--", "flagquantum"):
        parser.error("Compiler source must match its recorded commit")
    sys.path.insert(0, str(harness / "src"))
    report: DevelopmentReport = {
        "evidence_class": "development",
        "implementation_commit": _git(source, "rev-parse", "HEAD"),
        "source_commit": _HARNESS,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in _SOURCES
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "limits": "Inspected inputs on a shared macOS CPU host, one numerical thread; separate baseline-first processes, no qualified latency, memory, significance, held-out, hardware or SDK ranking claim. Width six checks complete operators; widths eight/twelve check three probes. Timing excludes independent evaluation and startup.",
        "applications": [],
        "latency": [],
    }
    workflows = ("flagquantum-ordinary-restored", "flagquantum-target-artifact")
    for case in itertools.product(
        (6, 8, 12),
        ("qaoa", "ising", "hopping", "qft", "repetition"),
        ("line", "ring", "grid", "sparse"),
        workflows,
    ):
        report["applications"].append(_observe(case, harness, samples=1, warmups=0))
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(len(report["applications"]), case, "passed", flush=True)
    for width, family in ((8, "ising"), (12, "qft")):
        for workflow in workflows:
            case = (width, family, "line", workflow)
            report["latency"].append(_observe(case, harness, samples=21, warmups=2))
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print("latency", case, "passed", flush=True)


if __name__ == "__main__":
    main()
