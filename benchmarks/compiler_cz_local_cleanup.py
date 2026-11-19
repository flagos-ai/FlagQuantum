"""Reproduce the local CZ cleanup development probe (not an SDK ranking).

Run from the repository root with ``python -m benchmarks.compiler_cz_local_cleanup``.
The fifty seeds are development inputs and must never be called held-out cases.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import torch

from flagquantum.compiler.optimization._cz_local_cleanup import compact_native_cz_locals
from flagquantum.compiler.optimization.two_qubit_synthesis import synthesize_two_qubit
from flagquantum.core.ir import Instruction


@dataclass(frozen=True)
class ProbeResult:
    seed: int
    baseline_operations: int
    cleanup_operations: int
    residual: float


def _operator(instructions: tuple[Instruction, ...]) -> torch.Tensor:
    # Independent closed-form gate matrices; no compiler matrix reader is used.
    identity = torch.eye(2, dtype=torch.complex128)
    result = torch.eye(4, dtype=torch.complex128)
    for instruction in instructions:
        if instruction.name == "cz":
            leaf = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex128))
        else:
            angle = float(instruction.params["theta"]) / 2
            c, s = math.cos(angle), math.sin(angle)
            if instruction.name == "rx":
                local = torch.tensor(
                    [[c, -1j * s], [-1j * s, c]], dtype=torch.complex128
                )
            elif instruction.name == "rz":
                local = torch.diag(
                    torch.tensor(
                        [complex(c, -s), complex(c, s)], dtype=torch.complex128
                    )
                )
            else:
                raise ValueError(f"Unexpected generated gate: {instruction.name}")
            leaf = (
                torch.kron(local, identity)
                if instruction.wires == (0,)
                else torch.kron(identity, local)
            )
        result = leaf @ result
    return result


def run_probe() -> tuple[ProbeResult, ...]:
    rows: list[ProbeResult] = []
    for seed in range(50):
        generator = torch.Generator().manual_seed(seed)
        matrix, _ = torch.linalg.qr(
            torch.randn((4, 4), generator=generator, dtype=torch.complex128)
        )
        source = [[complex(matrix[i, j].item()) for j in range(4)] for i in range(4)]
        baseline = synthesize_two_qubit(
            source, qubits=(0, 1), entangler="cz", z_rotation="rz", pulse_opcode="rx"
        )
        if baseline is None:
            raise RuntimeError(f"Synthesis refused seed {seed}")
        candidate = compact_native_cz_locals(baseline, (0, 1))
        actual = _operator(candidate)
        overlap = torch.vdot(actual.flatten(), matrix.flatten())
        phase = overlap / overlap.abs()
        residual = float((matrix - phase * actual).abs().max().item())
        if residual > 1e-11:
            raise RuntimeError(
                f"Operator verification failed for seed {seed}: {residual}"
            )
        if sum(item.name == "cz" for item in baseline) != sum(
            item.name == "cz" for item in candidate
        ):
            raise RuntimeError(f"Cleanup changed CZ count for seed {seed}")
        rows.append(ProbeResult(seed, len(baseline), len(candidate), residual))
    return tuple(rows)


if __name__ == "__main__":
    print(json.dumps([asdict(row) for row in run_probe()], indent=2))
