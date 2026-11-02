"""Benchmark-only HVA geometry adapters and an independent dense oracle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from benchmarks.training_workloads import pauli_matrix
from flagquantum import algorithms as fqa
from flagquantum.core.runtime_config import runtime_config


@dataclass(frozen=True)
class Workload:
    """The maintained bond-resolved HVA with width and depth varied separately."""

    qubits: int
    depth: int
    precision: str = "complex128"
    device: str = "cpu"

    @property
    def count(self) -> int:
        return fqa.heisenberg_hva_parameter_count(self.qubits, self.depth)

    @property
    def block(self) -> int:
        return self.count // self.depth

    def state(self, parameters: torch.Tensor) -> torch.Tensor:
        with runtime_config(complex_dtype=self.precision):
            return (
                fqa.heisenberg_hva(self.qubits, self.depth, parameters)
                .state()
                .reshape(-1)
            )

    def initial(self, parameters: torch.Tensor) -> torch.Tensor:
        return self.state(torch.zeros_like(parameters)).detach()

    def rotations(self) -> list[tuple[str, tuple[int, ...], int]]:
        """Extract parameter occurrences from the canonical builder's IR."""
        circuit = fqa.heisenberg_hva(
            self.qubits, self.depth, torch.arange(self.count, dtype=torch.float64)
        )
        return [
            (gate.name[1:].upper(), gate.wires, int(gate.params["theta"]))
            for gate in circuit.to_ir().instructions
            if gate.params
        ]

    def hamiltonian(self) -> np.ndarray:
        return np.stack(
            [
                pauli_matrix(axis * 2, (left, left + 1), self.qubits)
                for left in range(self.qubits - 1)
                for axis in "XYZ"
            ]
        ).sum(axis=0)


def block_metric(
    state: torch.Tensor, tangents: torch.Tensor, block: int
) -> torch.Tensor:
    """Fubini-Study metric, projected to identical layer blocks for every route."""
    connection = state.conj() @ tangents.T
    full = (
        tangents.conj() @ tangents.T - torch.outer(connection.conj(), connection)
    ).real
    full = (full + full.T) / 2
    groups = torch.arange(len(full), device=full.device) // block
    return full * (groups[:, None] == groups[None, :])


def primal(
    initial: torch.Tensor,
    parameters: torch.Tensor,
    layout: list[tuple[int, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """The same eager Pauli rotation primitive used by forward tangents."""
    state = initial
    for index, permutation, factors in layout:
        cosine, sine = torch.cos(parameters[index] / 2), torch.sin(
            parameters[index] / 2
        )
        state = cosine * state - 1j * sine * factors * state[permutation]
    return state


def forward_layout(
    workload: Workload, initial: torch.Tensor
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Prepare static Pauli permutations outside timed geometry calls."""
    indices = torch.arange(initial.numel(), device=initial.device)
    layout = []
    for word, wires, index in workload.rotations():
        permutation = indices.clone()
        factors = torch.ones_like(initial)
        for axis, wire in zip(word, wires, strict=True):
            bit = 1 << (workload.qubits - 1 - wire)
            signs = 1 - 2 * ((indices & bit) != 0).to(initial.real.dtype)
            if axis in "XY":
                permutation = permutation ^ bit
            if axis == "Y":
                factors = factors * (-1j * signs)
            elif axis == "Z":
                factors = factors * signs
        layout.append((index, permutation, factors))
    return layout


def unfused(
    initial: torch.Tensor,
    parameters: torch.Tensor,
    layout: list[tuple[int, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched forward lanes with separate eager rotation/derivative injection.

    Static layout and preparation are outside this call for both forward routes.
    No JIT, gate fusion or Python loop per parameter inflates the baseline.
    """
    state = initial
    tangents = torch.zeros(
        (parameters.numel(), state.numel()), dtype=state.dtype, device=state.device
    )
    for index, permutation, factors in layout:
        cosine, sine = torch.cos(parameters[index] / 2), torch.sin(
            parameters[index] / 2
        )
        state = cosine * state - 1j * sine * factors * state[permutation]
        tangents = cosine * tangents - 1j * sine * factors * tangents[:, permutation]
        tangents[index] += -0.5j * factors * state[permutation]
    return state, tangents


def independent_reference(
    workload: Workload, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Explicit NumPy matrices and finite differences; no simulator derivative."""
    n = workload.qubits
    dimension = 1 << n
    # Singlets have big-endian |01> - |10>; an unpaired final wire is |0>.
    initial: np.ndarray = np.ones(1, dtype=np.complex128)
    for _ in range(n // 2):
        initial = np.kron(initial, np.array([0, 1, -1, 0]) / np.sqrt(2))
    if n % 2:
        initial = np.kron(initial, [1, 0])
    generators = []
    # Deliberately derive the HVA ordering directly, without reading its IR.
    for _ in range(workload.depth):
        for axis in "XYZ":
            for left in range(n - 1):
                generators.append(pauli_matrix(axis * 2, (left, left + 1), n))
        generators.extend(pauli_matrix("Z", (wire,), n) for wire in range(n))

    def state_at(values: np.ndarray) -> np.ndarray:
        state = initial.copy()
        for angle, generator in zip(values, generators, strict=True):
            state = (
                np.cos(angle / 2) * np.eye(dimension)
                - 1j * np.sin(angle / 2) * generator
            ) @ state
        return state

    state = state_at(parameters)
    epsilon = 1e-5
    derivatives = []
    for index in range(len(parameters)):
        delta = np.zeros_like(parameters)
        delta[index] = epsilon
        derivatives.append(
            (state_at(parameters + delta) - state_at(parameters - delta))
            / (2 * epsilon)
        )
    tangents = np.stack(derivatives)
    connection = state.conj() @ tangents.T
    metric = (
        tangents.conj() @ tangents.T - np.outer(connection.conj(), connection)
    ).real
    groups = np.arange(len(parameters)) // workload.block
    metric *= groups[:, None] == groups[None, :]
    gradient = 2 * (tangents.conj() @ (workload.hamiltonian() @ state)).real
    return state, tangents, metric, gradient


class UnsupportedRouteError(RuntimeError):
    """A declared route is unavailable; never substitute another implementation."""


class Geometry:
    """A geometry route with explicit capability and method identity."""

    def __init__(self, method: str, workload: Workload):
        self.method, self.workload = method, workload
        self.framework_metric: Callable[[torch.Tensor], Any] | None = None
        if method == "fused_forward":
            if workload.device != "cuda" or workload.precision != "complex64":
                raise UnsupportedRouteError(
                    "the existing Triton HVA kernel requires CUDA complex64"
                )
            try:
                from flagquantum.kernels.triton import heisenberg_hva_forward_tangents
            except ImportError as exc:
                raise UnsupportedRouteError(
                    f"optional Triton is unavailable: {exc}"
                ) from exc
            self.fused = heisenberg_hva_forward_tangents
        elif method == "pennylane_metric":
            if workload.device != "cpu":
                raise UnsupportedRouteError(
                    "PennyLane adjoint_metric_tensor/default.qubit is a CPU route"
                )
            try:
                import pennylane as qml
            except ImportError as exc:
                raise UnsupportedRouteError(
                    f"optional PennyLane is unavailable: {exc}"
                ) from exc
            device = qml.device("default.qubit", wires=workload.qubits, shots=None)
            names = {
                "x": "PauliX",
                "z": "PauliZ",
                "h": "Hadamard",
                "cx": "CNOT",
                "rz": "RZ",
                "rxx": "IsingXX",
                "ryy": "IsingYY",
                "rzz": "IsingZZ",
            }
            circuit = fqa.heisenberg_hva(
                workload.qubits,
                workload.depth,
                torch.arange(workload.count, dtype=torch.float64),
            )
            instructions = circuit.to_ir().instructions

            @qml.qnode(device, interface="torch", diff_method="backprop")
            def qnode(values: torch.Tensor) -> Any:
                for gate in instructions:
                    operation = getattr(qml, names[gate.name])
                    if gate.params:
                        operation(values[int(gate.params["theta"])], wires=gate.wires)
                    else:
                        operation(wires=gate.wires)
                return qml.state()

            self.framework_metric = qml.adjoint_metric_tensor(qnode)
        elif method not in ("reverse_jacobian", "unfused_forward"):
            raise ValueError(f"unknown geometry method: {method}")

        dtype = torch.float64 if workload.precision == "complex128" else torch.float32
        zero = torch.zeros(workload.count, dtype=dtype, device=workload.device)
        self.initial_state = workload.initial(zero)
        self.layout = (
            forward_layout(workload, self.initial_state)
            if method in ("unfused_forward", "reverse_jacobian")
            else []
        )

    def tangent(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "reverse_jacobian":

            def state_at(p: torch.Tensor) -> torch.Tensor:
                return primal(self.initial_state, p, self.layout)

            real = torch.autograd.functional.jacobian(
                lambda p: state_at(p).real, parameters, vectorize=True
            )
            imag = torch.autograd.functional.jacobian(
                lambda p: state_at(p).imag, parameters, vectorize=True
            )
            return state_at(parameters).detach(), torch.complex(real, imag).T.detach()
        if self.method == "fused_forward":
            return self.fused(
                self.initial_state,
                parameters.detach(),
                n_qubits=self.workload.qubits,
                depth=self.workload.depth,
            )
        if self.method == "unfused_forward":
            return unfused(self.initial_state, parameters.detach(), self.layout)
        raise UnsupportedRouteError(
            "framework metric transform does not expose state tangents"
        )

    def metric(self, parameters: torch.Tensor) -> torch.Tensor:
        if self.framework_metric is not None:
            full = self.framework_metric(parameters).detach().to(parameters)
            groups = (
                torch.arange(self.workload.count, device=parameters.device)
                // self.workload.block
            )
            return full * (groups[:, None] == groups[None, :])
        state, tangents = self.tangent(parameters)
        return block_metric(state, tangents, self.workload.block)
