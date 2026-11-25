"""Fixed complete-routing controls for selection-local dependency reuse."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.compiler.topology import routing_strategy, sabre

from .compiler_sabre_incidence import Measurement, Observation, _program

BASELINE = "2ea5b78b6449b837db151e0d29c90895f6b51a4e"
SOURCE_PATH = "flagquantum/compiler/topology/sabre.py"
_FUNCTIONS = ("_plan_sabre_trial", "_select_sabre_trials")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, timeout=15).strip()


def _functions(source: str) -> dict[str, types.FunctionType]:
    nodes: list[ast.stmt] = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCTIONS
    ]
    assert len(nodes) == len(_FUNCTIONS)
    namespace = vars(sabre).copy()
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), SOURCE_PATH, "exec"), namespace
    )
    return {name: namespace[name] for name in _FUNCTIONS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _git("diff", "HEAD", "--", "flagquantum"):
        parser.error("Compiler implementation must match its recorded commit")
    baseline_source = _git("show", f"{BASELINE}:{SOURCE_PATH}") + "\n"
    candidate_source = Path(sabre.__file__).read_text()
    variants = {
        "baseline": _functions(baseline_source),
        "candidate": _functions(candidate_source),
    }
    observations: list[Observation] = []
    report: dict[str, object] = {
        "evidence_class": "development",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "base_commit": BASELINE,
        "implementation_commit": _git("rev-parse", "HEAD"),
        "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(
            candidate_source.encode()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "method": "Equal AST-cloned trial/selector patches, fresh coupling maps; full explicit SABRE route materialization and evidence; one warmup plus seven alternating samples; GC outside timers; separate traced peaks; identical output hashes in all calls.",
        "limits": "Inspected constructed inputs on shared macOS CPU with one numerical thread, 600-second process-group bound. Not qualified RSS, latency, statistical significance, unseen tasks, whole default compilation or a cross-SDK ranking.",
        "baseline_source": baseline_source,
        "candidate_source": candidate_source,
        "runner_source": Path(__file__).read_text(),
        "complete": False,
        "rows": [],
    }
    for width in (6, 8, 12, 16, 24, 32):
        for family in ("fourier", "parity_ladder", "graph_cost", "matching_layers"):
            program = _program(family, width)
            measurements = {label: Measurement([], 0, "") for label in variants}
            expected = None
            for block in range(9):
                labels = (
                    ("baseline", "candidate")
                    if block % 2 == 0
                    else ("candidate", "baseline")
                )
                for label in labels:
                    topology = CouplingMap.grid(
                        2 if width == 6 else 4, width // (2 if width == 6 else 4)
                    )
                    with ExitStack() as stack:
                        for module in (sabre, routing_strategy):
                            for name, function in variants[label].items():
                                stack.enter_context(
                                    patch.object(module, name, function)
                                )
                        gc.collect()
                        if block == 8:
                            tracemalloc.start()
                        try:
                            start = time.perf_counter()
                            result = route_to_topology(
                                program, topology, strategy="sabre"
                            )
                            elapsed = time.perf_counter() - start
                            if block == 8:
                                measurements[label].traced_peak_bytes = (
                                    tracemalloc.get_traced_memory()[1]
                                )
                        finally:
                            if block == 8:
                                tracemalloc.stop()
                    digest = result.content_hash
                    if expected is None:
                        expected = digest
                    assert digest == expected, (family, width, block, label)
                    if 0 < block < 8:
                        measurements[label].seconds.append(elapsed)
                    measurements[label].output_hash = digest
            row = Observation(family, width, len(program.instructions), measurements)
            observations.append(row)
            report["rows"] = [asdict(item) for item in observations]
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "width": width,
                        "family": family,
                        "time_ratio": statistics.median(
                            measurements["candidate"].seconds
                        )
                        / statistics.median(measurements["baseline"].seconds),
                    }
                ),
                flush=True,
            )
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
