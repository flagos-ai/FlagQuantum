"""Print a bounded CX/SX/RZ gate-count diagnostic with optimized Qiskit.

This is local development evidence, not a qualified SDKBench performance run.
Install Qiskit and NumPy explicitly to run the optional independent comparator.
"""

from __future__ import annotations

import json
from collections import Counter
from importlib.metadata import version

import numpy as np
import torch
from numpy.typing import NDArray
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer

from flagquantum.compiler.optimization.two_qubit_synthesis import synthesize_two_qubit
from flagquantum.core.ir import Instruction
from flagquantum.simulation.gate_matrix import gate_matrix


def _operator(instructions: tuple[Instruction, ...]) -> NDArray[np.complex128]:
    product = torch.eye(4, dtype=torch.complex128)
    identity = torch.eye(2, dtype=torch.complex128)
    exchange = torch.tensor([0, 2, 1, 3])
    for instruction in instructions:
        size = 2 ** len(instruction.wires)
        matrix = gate_matrix(
            instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
        ).reshape(size, size)
        if size == 2:
            matrix = (
                torch.kron(matrix, identity)
                if instruction.wires == (0,)
                else torch.kron(identity, matrix)
            )
        elif instruction.wires == (1, 0):
            matrix = matrix[exchange][:, exchange]
        product = matrix @ product
    return product.numpy()


def _error(source: NDArray[np.complex128], actual: NDArray[np.complex128]) -> float:
    overlap = np.vdot(actual, source)
    if abs(overlap) < 1e-15:
        raise ValueError("The operators have no usable phase overlap")
    return float(np.max(np.abs(source - overlap / abs(overlap) * actual)))


def main() -> None:
    decomposer = TwoQubitBasisDecomposer(
        CXGate(), euler_basis="ZSX", pulse_optimize=True
    )
    counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("flagquantum", "qiskit")
    }
    errors: dict[str, float] = dict.fromkeys(counts, 0.0)
    for seed in range(27000, 27012):
        rng = np.random.default_rng(seed)
        source = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
        native = synthesize_two_qubit(
            source.tolist(), qubits=(0, 1), entangler="cx", z_rotation="rz"
        )
        if native is None:
            raise ValueError("The published CX/SX/RZ basis was refused")
        reference = decomposer(source, approximate=False)
        counts["flagquantum"].update(item.name for item in native)
        counts["qiskit"].update(reference.count_ops())
        errors["flagquantum"] = max(
            errors["flagquantum"], _error(source, _operator(native))
        )
        errors["qiskit"] = max(
            errors["qiskit"], _error(source, Operator(reference).data)
        )
    if any(error > 1e-12 for error in errors.values()):
        raise ValueError(f"Complete-operator acceptance failed: {errors}")
    print(
        json.dumps(
            {
                "scope": "local development; twelve fixed numeric matrices; no ranking",
                "versions": {
                    name: version(name) for name in ("qiskit", "numpy", "torch")
                },
                "counts": {name: dict(value) for name, value in counts.items()},
                "maximum_phase_aligned_operator_error": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
