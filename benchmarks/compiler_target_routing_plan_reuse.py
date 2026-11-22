"""Compare selected-plan reuse with diagnostic replanning in the target workflow."""

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
from unittest.mock import patch

import flagquantum
from flagquantum.compiler.topology import routing as topology_routing
from flagquantum.compiler.topology.routing_cost import _TopologyCostModel
from flagquantum.compiler.topology.routing_strategy import RoutingStrategySelection
from flagquantum.core.ir import CircuitIR

Case = tuple[int, str, str]
_HARNESS = "9aa9260920984e2ca05bcb293abb885fccd9c144"
_SOURCE = "flagquantum/compiler/topology/topology_legalization.py"
_SELECT = topology_routing._select_routing_strategy


class Observation(TypedDict):
    case: Case
    mode: str
    metrics: dict[str, object]
    normalized_output_sha256: str
    seconds: float


class Report(TypedDict):
    evidence_class: str
    implementation_commit: str
    source_commit: str
    source_sha256: str
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


def _discard_plan(
    program: CircuitIR, topology: _TopologyCostModel
) -> tuple[RoutingStrategySelection, None]:
    selection, _ = _SELECT(program, topology)
    return selection, None


def _observe(case: Case, harness: Path, mode: str) -> Observation:
    from sdkbench.candidates.adapters import load_adapter, workflow_configuration
    from sdkbench.candidates.applications import fixture
    from sdkbench.p0 import Bundle
    from sdkbench.p0.evaluators import routing

    width, family, topology = case
    bundle = Bundle(harness / "docs/implementation")
    acceptance = bundle.cases[
        "ROUTE-GRID-08" if width == 8 else f"ROUTE-LINE-{width:02d}"
    ]
    selected = fixture(width, family, topology, "01" * 32)
    config = workflow_configuration("flagquantum-target-artifact")
    adapter = load_adapter(
        SimpleNamespace(
            configuration={
                "options": {
                    "sdk": config["sdk"],
                    "workflow": "flagquantum-target-artifact",
                    "compiler_options": config,
                }
            }
        )
    )
    selector = _SELECT if mode == "reuse" else _discard_plan
    with patch.object(topology_routing, "_select_routing_strategy", selector):
        start = time.perf_counter()
        payload = adapter.execute(
            "route", selected, bundle.plan["execution_profile"]["compiler_seeds"][0]
        )
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
    return {
        "case": case,
        "mode": mode,
        "metrics": metrics,
        "normalized_output_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdkbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(flagquantum.__file__).resolve().parents[1]
    harness = args.sdkbench_root.resolve()
    if _git(harness, "rev-parse", "HEAD") != _HARNESS or _git(
        harness, "status", "--porcelain"
    ):
        parser.error("The pinned SDKBench checkout must be unchanged")
    if _git(source, "diff", "HEAD", "--", "flagquantum"):
        parser.error("Compiler source must match its recorded commit")
    sys.path.insert(0, str(harness / "src"))
    report: Report = {
        "evidence_class": "development",
        "implementation_commit": _git(source, "rev-parse", "HEAD"),
        "source_commit": _HARNESS,
        "source_sha256": hashlib.sha256((source / _SOURCE).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "limits": "Inspected inputs on shared macOS CPU, one numerical thread. In-process diagnostic replanning discards only the selected plan. Alternating order; two warmup pairs, 21 measured pairs per latency case. No significance, memory, held-out, qualified speedup or SDK ranking claim. Six-qubit complete operators; three probes at eight/twelve qubits. Timed adapter execution excludes independent evaluation, setup and startup.",
        "applications": [],
        "latency": [],
    }

    def pair(case: Case, index: int) -> list[Observation]:
        modes = ("replan", "reuse") if index % 2 == 0 else ("reuse", "replan")
        observations = [_observe(case, harness, mode) for mode in modes]
        first, second = observations
        if (
            first["normalized_output_sha256"] != second["normalized_output_sha256"]
            or first["metrics"] != second["metrics"]
        ):
            raise AssertionError(f"Plan reuse changed accepted output for {case}")
        return observations

    def checkpoint() -> None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for index, case in enumerate(
        itertools.product(
            (6, 8, 12),
            ("qaoa", "ising", "hopping", "qft", "repetition"),
            ("line", "ring", "grid", "sparse"),
        )
    ):
        report["applications"].extend(pair(case, index))
        checkpoint()
        print(index + 1, case, "passed", flush=True)
    for case in ((8, "ising", "line"), (12, "qft", "line")):
        for index in range(23):
            observations = pair(case, index)
            if index >= 2:
                report["latency"].extend(observations)
                checkpoint()
        for mode in ("replan", "reuse"):
            samples = [
                row["seconds"]
                for row in report["latency"]
                if row["case"] == case and row["mode"] == mode
            ]
            print(case, mode, statistics.median(samples), flush=True)


if __name__ == "__main__":
    main()
