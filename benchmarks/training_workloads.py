"""Shared small training tasks and an independent dense correctness oracle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional

import flagquantum as fq
from flagquantum import algorithms as fqa

Task = Literal["classifier", "heisenberg"]


@dataclass(frozen=True)
class Gate:
    """One fixed gate or direct parameter/feature occurrence."""

    name: str
    wires: tuple[int, ...]
    parameter: int | None = None
    feature: int | None = None
    scale: float = 1.0

    def angle(self, parameters: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if self.parameter is not None:
            return parameters[self.parameter] * self.scale
        if self.feature is not None:
            return features[self.feature] * self.scale
        raise ValueError(f"fixed gate {self.name} has no angle")


@dataclass(frozen=True)
class Split:
    features: torch.Tensor
    labels: torch.Tensor


def dataset() -> tuple[Split, Split, Split]:
    """Return a fixed disjoint 64/32/32 XOR split without simulator labels."""
    generator = torch.Generator().manual_seed(20261026)
    features = 2 * torch.rand((128, 2), generator=generator, dtype=torch.float64) - 1
    labels = (features[:, 0] * features[:, 1] > 0).to(torch.float64)
    return (
        Split(features[:64], labels[:64]),
        Split(features[64:96], labels[64:96]),
        Split(features[96:], labels[96:]),
    )


@lru_cache(maxsize=2)
def recipe(task: Task) -> tuple[Gate, ...]:
    """Reuse the classifier gate order and canonical four-site HVA builder."""
    if task == "classifier":
        return (
            Gate("ry", (0,), feature=0, scale=1.4),
            Gate("ry", (1,), feature=1, scale=1.4),
            Gate("rz", (0,), parameter=0),
            Gate("rz", (1,), parameter=1),
            Gate("cx", (0, 1)),
            Gate("ry", (0,), parameter=2),
            Gate("ry", (1,), parameter=3),
            Gate("cx", (1, 0)),
        )
    count = fqa.heisenberg_hva_parameter_count(4, 2)
    circuit = fqa.heisenberg_hva(4, 2, torch.arange(count, dtype=torch.float64))
    gates = []
    for instruction in circuit.to_ir().instructions:
        index = None
        if instruction.params:
            angle = float(instruction.params["theta"])
            if not angle.is_integer() or not 0 <= angle < count:
                raise ValueError("HVA recipe requires direct indexed parameters")
            index = int(angle)
        gates.append(Gate(instruction.name, instruction.wires, parameter=index))
    return tuple(gates)


def initial_parameters(
    task: Task, seed: int, *, classical: bool = False
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if task == "classifier":
        parameters = torch.zeros(6, dtype=torch.float64)
        if not classical:
            parameters[:4] = 0.1 * torch.randn(
                4, generator=generator, dtype=torch.float64
            )
            parameters[4] = 1
    else:
        count = fqa.heisenberg_hva_parameter_count(4, 2)
        parameters = 0.02 * torch.randn(count, generator=generator, dtype=torch.float64)
    return parameters.requires_grad_()


@lru_cache(maxsize=2)
def terms(task: Task) -> tuple[tuple[float, str, tuple[int, ...]], ...]:
    if task == "classifier":
        return ((1.0, "Z", (0,)),)
    hamiltonian = fqa.heisenberg_chain_hamiltonian(4)
    # Keep the Hamiltonian definition shared with the maintained physics example.
    return tuple((float(t.coefficient), t.pauli, t.qubits) for t in hamiltonian.terms)


def classical_logits(parameters: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    x, y = features.unbind(1)
    basis = torch.stack((torch.ones_like(x), x, y, x.square(), x * y, y.square()), 1)
    return basis @ parameters


def fq_state(
    task: Task, parameters: torch.Tensor, features: torch.Tensor
) -> torch.Tensor:
    circuit = fq.Circuit(2 if task == "classifier" else 4, dtype=torch.complex128)
    for gate in recipe(task):
        options = (
            {"theta": gate.angle(parameters, features)}
            if gate.parameter is not None or gate.feature is not None
            else {}
        )
        circuit.gate(gate.name, gate.wires, **options)
    return circuit.state().reshape(-1)


PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
}


def pauli_matrix(word: str, wires: tuple[int, ...], qubits: int) -> np.ndarray:
    local = dict(zip(wires, word, strict=True))
    matrix: NDArray[np.complex128] = np.ones((1, 1), dtype=np.complex128)
    for wire in range(qubits):
        matrix = np.kron(matrix, PAULI[local.get(wire, "I")])
    return matrix


def dense_state(task: Task, parameters: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Apply explicit full matrices, independently of production simulators."""
    n = 2 if task == "classifier" else 4
    dimension = 2**n
    state = np.zeros(dimension, dtype=np.complex128)
    state[0] = 1
    for gate in recipe(task):
        if gate.name == "cx":
            matrix = np.zeros((dimension, dimension), dtype=np.complex128)
            control, target = (1 << (n - 1 - w) for w in gate.wires)
            for index in range(dimension):
                output = index ^ target if index & control else index
                matrix[output, index] = 1
        elif gate.name == "h":
            matrix = (
                pauli_matrix("X", gate.wires, n) + pauli_matrix("Z", gate.wires, n)
            ) / np.sqrt(2)
        elif gate.name in {"x", "z"}:
            matrix = pauli_matrix(gate.name.upper(), gate.wires, n)
        else:
            if gate.parameter is not None:
                theta = parameters[gate.parameter] * gate.scale
            elif gate.feature is not None:
                theta = features[gate.feature] * gate.scale
            else:
                raise ValueError(f"unsupported oracle gate: {gate.name}")
            generator = pauli_matrix(gate.name[1:].upper(), gate.wires, n)
            matrix = (
                np.cos(theta / 2) * np.eye(dimension)
                - 1j * np.sin(theta / 2) * generator
            )
        state = matrix @ state
    return state


def dense_value(task: Task, parameters: np.ndarray, features: np.ndarray) -> float:
    state = dense_state(task, parameters, features)
    n = 2 if task == "classifier" else 4
    observable = sum(
        coefficient * pauli_matrix(word, wires, n)
        for coefficient, word, wires in terms(task)
    )
    return float(np.vdot(state, observable @ state).real)


def dense_loss(task: Task, parameters: np.ndarray, split: Split) -> float:
    if task == "heisenberg":
        return dense_value(task, parameters, np.zeros(2))
    values = np.array(
        [dense_value(task, parameters, row) for row in split.features.numpy()]
    )
    logits = parameters[4] * values + parameters[5]
    return float(np.mean(np.logaddexp(0, logits) - split.labels.numpy() * logits))


def finite_difference(task: Task, parameters: torch.Tensor, split: Split) -> np.ndarray:
    values = parameters.detach().numpy().copy()
    gradient = np.zeros_like(values)
    for index in range(values.size):
        upper, lower = values.copy(), values.copy()
        upper[index] += 1e-5
        lower[index] -= 1e-5
        gradient[index] = (
            dense_loss(task, upper, split) - dense_loss(task, lower, split)
        ) / 2e-5
    return gradient


def binary_loss(logits: torch.Tensor, split: Split) -> torch.Tensor:
    return functional.binary_cross_entropy_with_logits(logits, split.labels)
