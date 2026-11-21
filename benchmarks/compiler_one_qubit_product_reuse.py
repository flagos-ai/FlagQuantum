"""Compare native local-run implementations using a selected Git baseline.

This is an inspected CPU development diagnostic, not qualified SDKBench evidence.
It compares complete outputs before recording alternating local timing samples.
Python allocation peaks are not process RSS or a memory-capacity measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TypedDict, cast

from flagquantum.compiler.optimization import one_qubit_optimization as current
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import get_operator_schema

_SOURCE = "flagquantum/compiler/optimization/one_qubit_optimization.py"
_ROOT = Path(__file__).resolve().parents[1]
_Rewrite = Callable[[CircuitIR], CircuitIR]


class Measurement(TypedDict):
    samples_seconds: list[float]
    median_seconds: float
    python_peak_bytes: int


class DevelopmentReport(TypedDict):
    scope: str
    baseline: str
    candidate_source_sha256: str
    python: str
    platform: str
    matched_complete_outputs: int
    patterns: dict[str, dict[str, Measurement]]


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=_ROOT, text=True, timeout=30
    ).strip()


def _baseline(revision: str) -> tuple[str, _Rewrite]:
    commit = _git("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
    source = _git("show", f"{commit}:{_SOURCE}")
    module = ModuleType("flagquantum.compiler.optimization._product_reuse_baseline")
    module.__package__ = "flagquantum.compiler.optimization"
    exec(compile(source, f"{commit}:{_SOURCE}", "exec"), module.__dict__)
    return commit, cast(_Rewrite, module.collapse_one_qubit_runs)


def _compare_outputs(baseline: _Rewrite) -> int:
    rng = random.Random(71236)
    opcodes = sorted(current._EULER_ANGLES)
    for case in range(2400):
        width = (2, 4, 6)[case % 3]
        instructions = []
        for position in range(4 + case % 93):
            wire = rng.randrange(width)
            if rng.random() < 0.18:
                target = (wire + 1 + rng.randrange(width - 1)) % width
                instructions.append(Instruction("cx", (wire, target)))
            else:
                opcode = rng.choice(opcodes)
                schema = get_operator_schema(opcode)
                assert schema is not None
                instructions.append(
                    Instruction(
                        opcode,
                        (wire,),
                        {
                            name: rng.choice((0.0, 0.1, -0.7, 1.25, 3.141592653589793))
                            for name in schema.parameters
                        },
                        metadata={"source": position},
                    )
                )
        source = CircuitIR(width, tuple(instructions))
        if baseline(source) != current.collapse_one_qubit_runs(source):
            raise AssertionError(f"Complete output differs for generated case {case}")
    return 2400


def _measure(baseline: _Rewrite, source: CircuitIR) -> dict[str, Measurement]:
    implementations = {
        "baseline": baseline,
        "candidate": current.collapse_one_qubit_runs,
    }
    samples: dict[str, list[float]] = {name: [] for name in implementations}
    expected = baseline(source)
    for rewrite in implementations.values():
        if rewrite(source) != expected:
            raise AssertionError("The timing workload has different complete outputs")
    for round_index in range(41):
        order = list(implementations)
        if round_index % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            actual = implementations[name](source)
            samples[name].append(time.perf_counter() - start)
            if actual != expected:
                raise AssertionError("A measured rewrite changed its complete output")
    result: dict[str, Measurement] = {}
    for name, rewrite in implementations.items():
        tracemalloc.start()
        try:
            rewrite(source)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        result[name] = {
            "samples_seconds": samples[name],
            "median_seconds": statistics.median(samples[name]),
            "python_peak_bytes": peak,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True, help="Locally available Git revision"
    )
    arguments = parser.parse_args()
    commit, baseline = _baseline(arguments.baseline)
    matched = _compare_outputs(baseline)
    mixed = CircuitIR(
        4,
        tuple(
            Instruction(
                opcode, ((index // 5) % 4,), {"theta": 0.37} if opcode == "ry" else {}
            )
            for index, opcode in enumerate(("h", "ry", "t", "x", "h") * 200)
        ),
    )
    singletons = CircuitIR(
        2,
        tuple(
            gate
            for _ in range(500)
            for gate in (Instruction("h", (0,)), Instruction("cx", (0, 1)))
        ),
    )
    report: DevelopmentReport = {
        "scope": "Inspected local CPU diagnostic; not qualified or cross-SDK evidence",
        "baseline": commit,
        "candidate_source_sha256": hashlib.sha256(
            (_ROOT / _SOURCE).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "matched_complete_outputs": matched,
        "patterns": {
            "mixed_runs": _measure(baseline, mixed),
            "singletons": _measure(baseline, singletons),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
