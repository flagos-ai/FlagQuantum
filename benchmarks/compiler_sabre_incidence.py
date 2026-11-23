"""Balanced routing development controls; no cross-SDK performance claim."""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.compiler.topology import sabre
from flagquantum.core.ir import CircuitIR, Instruction

BASELINE = "7d2e13c4160c0c1546d61a41551189fa11615465"
SOURCE_PATH = "flagquantum/compiler/topology/sabre.py"


@dataclass
class Measurement:
    seconds: list[float]
    traced_peak_bytes: int
    output_hash: str


@dataclass
class Observation:
    family: str
    width: int
    source_operations: int
    measurements: dict[str, Measurement]


def _scorer(source: str) -> types.FunctionType:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_relative_score_swap"
    )
    namespace = vars(sabre).copy()
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), SOURCE_PATH, "exec"),
        namespace,
    )
    scorer = namespace["_relative_score_swap"]
    assert isinstance(scorer, types.FunctionType)
    return scorer


def _program(family: str, width: int) -> CircuitIR:
    instructions: list[Instruction] = []
    if family == "fourier":
        for target in range(width):
            instructions.append(Instruction("h", (target,)))
            for control in range(target + 1, width):
                instructions.append(
                    Instruction(
                        "cphase", (control, target), {"theta": 2.0 ** (target - control)}
                    )
                )
    elif family == "parity_ladder":
        for step in (1, 2, 4):
            for wire in range(width - step):
                instructions.extend(
                    (
                        Instruction("cx", (wire, wire + step)),
                        Instruction("rz", (wire + step,), {"theta": 0.31}),
                        Instruction("cx", (wire, wire + step)),
                    )
                )
    elif family == "graph_cost":
        rng = random.Random(1729 + width)
        edges = sorted(
            {tuple(sorted(rng.sample(range(width), 2))) for _ in range(width * 3)}
        )
        for _ in range(3):
            instructions.extend(
                Instruction("rzz", pair, {"theta": 0.23}) for pair in edges
            )
            instructions.extend(
                Instruction("rx", (wire,), {"theta": 0.19}) for wire in range(width)
            )
    else:
        assert family == "matching_layers"
        rng = random.Random(500 + width)
        for _ in range(8):
            order = list(range(width))
            rng.shuffle(order)
            instructions.extend(
                Instruction("cx", tuple(order[i : i + 2])) for i in range(0, width, 2)
            )
    return CircuitIR(width, tuple(instructions))


def main() -> None:
    baseline_source = subprocess.run(
        ["git", "show", f"{BASELINE}:{SOURCE_PATH}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    candidate_source = Path(sabre.__file__).read_text()
    scorers = {
        "baseline": _scorer(baseline_source),
        "candidate": _scorer(candidate_source),
    }
    observations: list[Observation] = []
    for width in (16, 24, 32):
        for family in ("fourier", "parity_ladder", "graph_cost", "matching_layers"):
            program = _program(family, width)
            topology = CouplingMap.grid(4, width // 4)
            measurements = {label: Measurement([], 0, "") for label in scorers}
            expected: str | None = None
            for block in range(8):
                labels = (
                    ("baseline", "candidate")
                    if block % 2 == 0
                    else ("candidate", "baseline")
                )
                for label in labels:
                    with patch.object(sabre, "_relative_score_swap", scorers[label]):
                        gc.collect()
                        start = time.perf_counter()
                        result = route_to_topology(program, topology, strategy="sabre")
                        elapsed = time.perf_counter() - start
                    digest = result.content_hash
                    if expected is None:
                        expected = digest
                    assert digest == expected, (family, width, block, label)
                    if block:
                        measurements[label].seconds.append(elapsed)
                    measurements[label].output_hash = digest
            for label in scorers:
                with patch.object(sabre, "_relative_score_swap", scorers[label]):
                    gc.collect()
                    tracemalloc.start()
                    try:
                        result = route_to_topology(program, topology, strategy="sabre")
                        measurements[label].traced_peak_bytes = (
                            tracemalloc.get_traced_memory()[1]
                        )
                    finally:
                        tracemalloc.stop()
                assert result.content_hash == expected
            observations.append(
                Observation(family, width, len(program.instructions), measurements)
            )
            print(
                json.dumps(
                    {
                        "phase": "routing",
                        "family": family,
                        "width": width,
                        "candidate_time_ratio": statistics.median(
                            measurements["candidate"].seconds
                        )
                        / statistics.median(measurements["baseline"].seconds),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    print(
        json.dumps(
            {
                "evidence_class": "development",
                "distribution_semantics": "single_device_fast_path",
                "scalability_claim_allowed": False,
                "baseline_revision": BASELINE,
                "baseline_source_sha256": hashlib.sha256(
                    baseline_source.encode()
                ).hexdigest(),
                "candidate_source_sha256": hashlib.sha256(
                    candidate_source.encode()
                ).hexdigest(),
                "python": sys.version,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "method": "Equal AST-cloned scorer patches; grid; one warmup and seven alternating samples; GC before both; separate full-routing tracemalloc; identical output hashes required.",
                "rows": [asdict(row) for row in observations],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
