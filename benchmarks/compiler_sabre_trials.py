"""Measure bounded routing trials on generic development inputs, not SDK rankings."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.topology import sabre
from flagquantum.core.ir import CircuitIR, Instruction


def _program(width: int, seed: int) -> CircuitIR:
    rng = random.Random(seed)
    return CircuitIR(
        width,
        tuple(
            Instruction("cx", tuple(rng.sample(range(width), 2)))
            for _ in range(4 * width)
        ),
    )


def _metrics(
    program: CircuitIR, coupling: CouplingMap, plan: sabre.SabrePlan
) -> dict[str, int]:
    operations: list[tuple[str, tuple[int, ...]]] = []
    swap_index = 0
    for count, source_index in enumerate(plan.sequence):
        while (
            swap_index < len(plan.swaps) and plan.swap_placements[swap_index] == count
        ):
            operations.append(("swap", plan.swaps[swap_index]))
            swap_index += 1
        operations.append(("cx", plan.placements[source_index]))
    if swap_index != len(plan.swaps):
        raise AssertionError("Plan contains unemitted swaps")
    restore = sabre.plan_restore_swaps(plan, coupling)
    operations.extend(("swap", (left, right)) for left, right, _ in restore)
    # Exact linear maps cover every computational input for CX/SWAP programs.
    expected = [1 << q for q in range(program.n_wires)]
    for instruction in program:
        expected[instruction.wires[1]] ^= expected[instruction.wires[0]]
    actual = [1 << q for q in range(program.n_wires)]
    levels = [0] * program.n_wires
    for name, (left, right) in operations:
        if not coupling.has_edge(left, right):
            raise AssertionError("Routed interaction is not on a device edge")
        if name == "swap":
            actual[left], actual[right] = actual[right], actual[left]
        else:
            actual[right] ^= actual[left]
        levels[left] = levels[right] = max(levels[left], levels[right]) + 1
    if actual != expected:
        raise AssertionError("Restored route changes the complete linear operator")
    return {
        "forward_swaps": len(plan.swaps),
        "restore_swaps": len(restore),
        "total_swaps": len(plan.swaps) + len(restore),
        "logical_two_qubit_depth": max(levels, default=0),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for width in (6, 8, 12, 16):
        for topology in ("line", "grid"):
            coupling = (
                CouplingMap.line(width)
                if topology == "line"
                else CouplingMap.grid(2, width // 2)
            )
            for seed in range(10):
                program = _program(width, seed)
                measurements: dict[str, dict[str, object]] = {}
                for label in (
                    ("baseline", "trials") if seed % 2 == 0 else ("trials", "baseline")
                ):
                    start = time.perf_counter()
                    if label == "baseline":
                        # Replace only trial selection; progress recovery and scoring stay identical.
                        with patch.object(
                            sabre,
                            "_select_sabre_trials",
                            side_effect=lambda p, c, baseline: baseline,
                        ):
                            plan = sabre.plan_sabre_swaps(program, coupling)
                    else:
                        plan = sabre.plan_sabre_swaps(program, coupling)
                    seconds = time.perf_counter() - start
                    measurements[label] = {
                        **_metrics(program, coupling, plan),
                        "seconds": seconds,
                    }
                rows.append(
                    {
                        "width": width,
                        "topology": topology,
                        "seed": seed,
                        "measurements": measurements,
                    }
                )
    print(
        json.dumps(
            {
                "evidence_class": "development",
                "python": sys.version,
                "platform": platform.platform(),
                "source_sha256": hashlib.sha256(
                    Path(sabre.__file__).read_bytes()
                ).hexdigest(),
                "acceptance": "exact complete GF(2) operator, restored layout, every interaction on a coupling edge",
                "timing_limit": "shared host, one sample, no warmup; diagnostic only",
                "depth_definition": "dependency depth counting each logical CX and SWAP as one; not target CZ depth",
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
