"""Frozen complete-circuit recipes and independent NumPy precision oracles."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import flagquantum as fq
from benchmarks.training_workloads import pauli_matrix
from flagquantum import algorithms as fqa
from flagquantum.core.ir import Instruction
from flagquantum.simulation.matrices import GATE_MAT_DICT
from flagquantum.simulation.statevector.double_single_device_gates import (
    encode_device_double_single_matrix,
)
from flagquantum.simulation.statevector.double_single_host_gates import (
    encode_host_double_single_matrix,
)
from flagquantum.simulation.statevector.split_real_imag import instruction_matrix_pair

Gate = tuple[str, tuple[int, ...], float | None]


@dataclass(frozen=True)
class Case:
    qubits: int
    depth: int
    family: str
    seed: int

    @property
    def intended_theta(self) -> float:
        return math.pi / 7 + self.seed / 1024

    @property
    def theta(self) -> float:
        return float(np.float32(self.intended_theta))

    @property
    def name(self) -> str:
        return f"q{self.qubits}-d{self.depth}-{self.family}-{self.seed}"

    def recipe(self) -> list[Gate]:
        gates: list[Gate] = [("h", (wire,), None) for wire in range(self.qubits)]
        gates.append(("ry", (0,), None))  # The sole named parameter occurrence.
        for layer in range(self.depth):
            a = (self.seed % 13 + layer % 7 + 1) / 32
            block: list[Gate] = [
                ("h", (0,), None),
                ("rx", (0,), a),
                ("rz", (0,), -a / 2),
                ("rxx", (0, 1), a / 4),
            ]
            if self.family == "cancellation_echo":
                inverse = [
                    (name, wires, -angle if angle is not None else None)
                    for name, wires, angle in reversed(block)
                ]
                gates.extend(block + inverse)
            elif self.family == "noncommuting":
                gates.extend(block)
                for wire in range(1, self.qubits):
                    gates.extend(
                        [("rz", (wire,), a / 2), ("cx", (wire - 1, wire), None)]
                    )
            else:
                raise ValueError(f"unknown frozen circuit family: {self.family}")
        return gates

    def circuit(
        self,
        *,
        device: str,
        fp64: bool = False,
        symbolic: bool = False,
        theta: float | None = None,
    ) -> fq.Circuit:
        dtype = torch.float64 if fp64 else torch.float32
        circuit = fq.Circuit(
            self.qubits,
            device=device,
            dtype=torch.complex128 if fp64 else torch.complex64,
        )
        parameter_index = self.qubits
        for index, (name, wires, angle) in enumerate(self.recipe()):
            options: dict[str, Any] = {}
            if index == parameter_index:
                options["theta"] = (
                    fq.Parameter("theta")
                    if symbolic
                    else torch.tensor(
                        self.theta if theta is None else theta,
                        device=device,
                        dtype=dtype,
                    )
                )
            elif angle is not None:
                options["theta"] = torch.tensor(angle, device=device, dtype=dtype)
            circuit.gate(name, wires, **options)
        return circuit

    def terms(self) -> list[tuple[float, str, tuple[int, ...]]]:
        leading = [(1.0, "Z", (0,))]
        if self.family == "cancellation_echo":
            leading = [(1e8, "Z", (0,)), (1.0, "Z", (0,)), (-1e8, "Z", (0,))]
        return leading + [(0.5, "X", (1,)), (0.25, "ZZ", (0, 1))]

    def observable(self) -> Any:
        return fqa.Hamiltonian(
            [
                fqa.pauli_term(coefficient, word, wires)
                for coefficient, word, wires in self.terms()
            ]
        )


def local_reference(name: str, angle: float | None = None) -> np.ndarray:
    """Explicit matrices independent of FlagQuantum matrix generation."""
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.diag([1, -1]).astype(np.complex128)
    if name == "h":
        return (x + z) / np.sqrt(2)
    if name == "cx":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        )
    if angle is None:
        raise ValueError("rotation reference requires an angle")
    generator = {"rx": x, "ry": y, "rz": z, "rxx": np.kron(x, x)}[name]
    return (
        np.cos(angle / 2) * np.eye(len(generator)) - 1j * np.sin(angle / 2) * generator
    )


def embed(matrix: np.ndarray, wires: tuple[int, ...], qubits: int) -> np.ndarray:
    """Explicit big-endian basis enumeration, independent of tensor gate kernels."""
    size = 1 << qubits
    output = np.zeros((size, size), dtype=np.complex128)
    masks = [1 << (qubits - wire - 1) for wire in wires]
    for column in range(size):
        local_column = sum(
            int(bool(column & mask)) << (len(wires) - index - 1)
            for index, mask in enumerate(masks)
        )
        cleared = column
        for mask in masks:
            cleared &= ~mask
        for local_row in range(len(matrix)):
            row = cleared
            for index, mask in enumerate(masks):
                if local_row & (1 << (len(wires) - index - 1)):
                    row |= mask
            output[row, column] = matrix[local_row, local_column]
    return output


def reference(
    case: Case, theta: float | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
    """Dense complex128 state and analytic initial-RY tangent, with coalesced H."""
    angle = case.theta if theta is None else theta
    state = np.zeros(1 << case.qubits, dtype=np.complex128)
    state[0] = 1
    tangent = np.zeros_like(state)
    for index, (name, wires, fixed_angle) in enumerate(case.recipe()):
        parameterized = index == case.qubits
        matrix = embed(
            local_reference(name, angle if parameterized else fixed_angle),
            wires,
            case.qubits,
        )
        state = matrix @ state
        tangent = matrix @ tangent
        if parameterized:
            tangent = -0.5j * pauli_matrix("Y", (0,), case.qubits) @ state
    observables = {
        label: pauli_matrix(word, wires, case.qubits)
        for label, word, wires in (
            ("Z0", "Z", (0,)),
            ("X1", "X", (1,)),
            ("Z0Z1", "ZZ", (0, 1)),
        )
    }
    # The repeated Z terms are mathematically one Z term. Coalescing with fsum
    # avoids contaminating the reference with the cancellation under test.
    z_coefficient = math.fsum(
        coefficient
        for coefficient, word, wires in case.terms()
        if word == "Z" and wires == (0,)
    )
    hamiltonian = (
        z_coefficient * observables["Z0"]
        + 0.5 * observables["X1"]
        + 0.25 * observables["Z0Z1"]
    )
    values = {
        name: float(np.vdot(state, matrix @ state).real)
        for name, matrix in observables.items()
    }
    values["objective"] = float(np.vdot(state, hamiltonian @ state).real)
    gradient = float(2 * np.vdot(tangent, hamiltonian @ state).real)
    return state, tangent, values, gradient


def state_quality(state: np.ndarray, reference_state: np.ndarray) -> dict[str, float]:
    norm = float(np.vdot(state, state).real)
    reference_norm = float(np.vdot(reference_state, reference_state).real)
    fidelity = float(
        abs(np.vdot(reference_state, state)) ** 2 / (norm * reference_norm)
    )
    return {
        "state_max_absolute_error": float(np.max(np.abs(state - reference_state))),
        "state_l2_error": float(np.linalg.norm(state - reference_state)),
        "state_fidelity": fidelity,
        "state_infidelity": abs(1 - fidelity),
        "norm_drift": abs(norm - 1),
    }


def output_hash(words: list[torch.Tensor]) -> str:
    """Hash dtype, shape and actual output bytes, never scalar error equality."""
    digest = hashlib.sha256()
    for word in words:
        value = word.detach().cpu().contiguous()
        digest.update(f"{value.dtype}:{tuple(value.shape)}:".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def synthesis_error(case: Case, mode: str, device: str) -> dict[str, Any]:
    """Observe existing gate encoders; no alternate precision implementation."""
    maximum = 0.0
    fp64_diagnostic = (
        "device_Gate_MAT_DICT" if mode == "fp64" else "actual_stage_encoder"
    )
    for index, (name, wires, angle) in enumerate(case.recipe()):
        if index == case.qubits:
            angle = case.theta
        options = (
            {}
            if angle is None
            else {
                "theta": torch.tensor(
                    angle,
                    dtype=torch.float64 if mode == "fp64" else torch.float32,
                    device=device,
                )
            }
        )
        instruction = Instruction(name=name, wires=wires, params=options)
        if mode == "double_single_p4":
            encoded, _ = encode_device_double_single_matrix(
                instruction, bindings={}, device=torch.device(device)
            )
            actual = encoded.to("cpu").to_complex128().numpy()
        elif mode == "double_single_p3":
            actual = (
                encode_host_double_single_matrix(
                    instruction, device=torch.device(device)
                )
                .to("cpu")
                .to_complex128()
                .numpy()
            )
        elif mode in ("fp32", "selective_p2"):
            real, imag = instruction_matrix_pair(
                instruction, device=torch.device(device)
            )
            actual = real.cpu().double().numpy() + 1j * imag.cpu().double().numpy()
        else:
            generator = GATE_MAT_DICT[name]
            actual_tensor = (
                generator(torch.tensor([angle], dtype=torch.float64, device=device))
                if callable(generator)
                else generator.to(device=device, dtype=torch.complex128)
            )
            actual = (
                actual_tensor.detach()
                .cpu()
                .numpy()
                .reshape(local_reference(name, angle).shape)
            )
        maximum = max(
            maximum, float(np.max(np.abs(actual - local_reference(name, angle))))
        )
    intended_state = reference(case, case.intended_theta)[0]
    matched_state = reference(case)[0]
    coefficients = [coefficient for coefficient, _, _ in case.terms()]
    return {
        "gate_synthesis_max_entry_error": maximum,
        "gate_synthesis_scope": fp64_diagnostic,
        "parameter_conversion_absolute_error": abs(case.intended_theta - case.theta),
        "reference_state_parameter_conversion_max_error": float(
            np.max(np.abs(intended_state - matched_state))
        ),
        "coefficient_conversion_max_error": max(
            abs(float(np.float32(c)) - c) for c in coefficients
        ),
        "output_conversion": "FP32_words_transferred_then_reconstructed_on_CPU_float64_only_outside_timer",
        "normalization_interval": 0,
    }
