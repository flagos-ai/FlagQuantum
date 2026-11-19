"""Small independent Circuit and channel oracles for the tutorial."""

import math
from itertools import product
from typing import cast

import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR

from .workflow import Calibration


def square_state() -> torch.Tensor:
    expected = torch.zeros(1024, dtype=torch.complex128)
    for value in range(8):
        index = (value << 7) | ((value * value + 1) << 1)
        expected[index] = 1 / math.sqrt(8)
    return cast(
        torch.Tensor, fq.Circuit(10, inputs=expected, dtype=torch.complex128).state()
    ).reshape(-1)


def ghz_state() -> torch.Tensor:
    circuit = fq.Circuit(3, dtype=torch.complex128).h(0).cx(0, 1).cx(1, 2)
    return cast(torch.Tensor, circuit.state()).reshape(-1)


def tensor_product(*matrices: torch.Tensor) -> torch.Tensor:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.kron(result, matrix)
    return result


def noisy_state(
    circuit: CircuitIR, profile: Calibration
) -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    paulis = (identity, x, y, z)
    sx = ((1 + 1j) * identity + (1 - 1j) * x) / 2
    cz = torch.eye(8, dtype=torch.complex128)
    rho = torch.zeros((8, 8), dtype=torch.complex128)
    rho[0, 0] = 1

    def embed(assignments: dict[int, torch.Tensor]) -> torch.Tensor:
        return tensor_product(*(assignments.get(wire, identity) for wire in range(3)))

    for op in circuit.instructions:
        if op.name == "cz":
            first, second = (1 << (2 - wire) for wire in op.wires)
            unitary = cz.clone()
            for word in range(8):
                if word & first and word & second:
                    unitary[word, word] = -1
        else:
            if op.name == "rz":
                angle = float(op.params["theta"])
                gate = torch.diag(
                    torch.exp(
                        torch.tensor(
                            [-0.5j * angle, 0.5j * angle], dtype=torch.complex128
                        )
                    )
                )
            else:
                gate = {"id": identity, "x": x, "sx": sx}[op.name]
            unitary = embed({op.wires[0]: gate})
        rho = unitary @ rho @ unitary.mH

        error = profile.errors[op.name, op.wires]
        if not error:
            continue
        local_dimension = 2 ** len(op.wires)
        probability = (local_dimension + 1) * error / local_dimension
        weight = probability / (local_dimension**2 - 1)
        mixture = (1 - probability) * rho
        for choice in product(range(4), repeat=len(op.wires)):
            if any(choice):
                pauli = embed(
                    {
                        wire: paulis[index]
                        for wire, index in zip(op.wires, choice, strict=True)
                    }
                )
                mixture += weight * (pauli @ rho @ pauli.mH)
        rho = mixture

    confusion = tensor_product(
        *(
            torch.tensor([[1 - p01, p10], [p01, 1 - p10]], dtype=torch.float64)
            for p01, p10 in profile.readout
        )
    )
    return rho, confusion @ rho.diagonal().real
