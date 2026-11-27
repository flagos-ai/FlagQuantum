"""Measure removal of duplicate two-qubit solves against a retained Git revision.

Inspected local CPU inputs are development diagnostics, not qualified SDKBench
performance evidence. Python allocation peaks are not process memory peaks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TypedDict, cast

import torch

from benchmarks.compiler_one_qubit_product_reuse import Measurement, _git
from flagquantum.compiler.optimization import two_qubit_optimization as current
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operators.schema import get_operator_schema

_SOURCE = "flagquantum/compiler/optimization/two_qubit_optimization.py"
_Rewrite = Callable[[CircuitIR], CircuitIR]


class DevelopmentReport(TypedDict):
    scope: str
    baseline: str
    baseline_source_sha256: str
    candidate_source_sha256: str
    python: str
    platform: str
    matched_complete_outputs: int
    patterns: dict[str, dict[str, Measurement]]


def _baseline(revision: str) -> tuple[str, str, _Rewrite]:
    commit = _git("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
    source = _git("show", f"{commit}:{_SOURCE}")
    module = ModuleType("flagquantum.compiler.optimization._fallback_baseline")
    module.__package__ = "flagquantum.compiler.optimization"
    exec(compile(source, f"{commit}:{_SOURCE}", "exec"), module.__dict__)
    return (
        commit,
        hashlib.sha256((source + "\n").encode()).hexdigest(),
        cast(_Rewrite, module.collapse_two_qubit_blocks),
    )


def _compare_outputs(baseline: _Rewrite) -> int:
    rng = random.Random(81927)
    opcodes = [*current._FAMILIES, "h", "rx", "rz", "measure", "barrier"]
    for case in range(2400):
        width = (2, 4, 6)[case % 3]
        instructions: list[Instruction] = []
        for position in range(4 + case % 37):
            opcode = rng.choice(opcodes)
            if opcode in {"measure", "barrier"}:
                wires = (
                    (rng.randrange(width),)
                    if opcode == "measure"
                    else tuple(range(width))
                )
                instructions.append(
                    Instruction(
                        opcode, wires, metadata={"source": position, "is_dynamic": True}
                    )
                )
                continue
            schema = get_operator_schema(opcode)
            assert schema is not None
            arity = schema.arity if schema.arity is not None else width
            wires = tuple(rng.sample(range(width), arity))
            params: dict[str, float | torch.Tensor] = {}
            for name in schema.parameters:
                angle = rng.choice((0.0, 0.17, -0.23, 1.25, 3.141592653589793))
                params[name] = (
                    angle
                    if case % 3 == 0
                    else torch.tensor(
                        angle, dtype=torch.float64, requires_grad=case % 3 == 2
                    )
                )
            instructions.append(
                Instruction(opcode, wires, params, metadata={"source": position})
            )
        source = CircuitIR(width, tuple(instructions))
        if baseline(source) != current.collapse_two_qubit_blocks(source):
            raise AssertionError(f"Complete output differs for generated case {case}")
    return 2400


def _measure(baseline: _Rewrite, source: CircuitIR) -> dict[str, Measurement]:
    implementations = {
        "baseline": baseline,
        "candidate": current.collapse_two_qubit_blocks,
    }
    samples: dict[str, list[float]] = {name: [] for name in implementations}
    expected = baseline(source)
    for rewrite in implementations.values():
        if rewrite(source) != expected:
            raise AssertionError("Timing inputs have different complete outputs")
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
    args = parser.parse_args()
    commit, parent_hash, baseline = _baseline(args.baseline)
    matched = _compare_outputs(baseline)
    patterns: dict[str, CircuitIR] = {}
    for scalar_tensor in (False, True):
        operations = []
        for _ in range(100):
            for opcode, value in (("rxx", 0.17), ("rzz", 0.23)):
                theta = (
                    torch.tensor(value, dtype=torch.float64) if scalar_tensor else value
                )
                operations.append(Instruction(opcode, (0, 1), {"theta": theta}))
            operations.append(
                Instruction("barrier", (0, 1), metadata={"is_dynamic": True})
            )
        patterns["scalar_tensor_blocks" if scalar_tensor else "float_blocks"] = (
            CircuitIR(2, tuple(operations))
        )
    patterns["mixed_narrower_folds"] = CircuitIR(
        2,
        tuple(
            item
            for _ in range(100)
            for item in (
                Instruction(
                    "rzz", (0, 1), {"theta": torch.tensor(0.17, dtype=torch.float64)}
                ),
                Instruction(
                    "rzz", (0, 1), {"theta": torch.tensor(-0.17, dtype=torch.float64)}
                ),
                Instruction("h", (0,)),
                Instruction(
                    "rxx", (0, 1), {"theta": torch.tensor(0.23, dtype=torch.float64)}
                ),
                Instruction(
                    "rxx", (0, 1), {"theta": torch.tensor(-0.23, dtype=torch.float64)}
                ),
                Instruction("barrier", (0, 1), metadata={"is_dynamic": True}),
            )
        ),
    )
    report: DevelopmentReport = {
        "scope": "Inspected local CPU pass diagnostic, one warmup and 41 alternating samples; not held-out, whole-compiler, process RSS, cross-SDK, or qualified SDKBench evidence",
        "baseline": commit,
        "baseline_source_sha256": parent_hash,
        "candidate_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / _SOURCE).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "matched_complete_outputs": matched,
        "patterns": {
            name: _measure(baseline, source) for name, source in patterns.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
