"""Measurement helpers for the local statevector executor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..matrices import X_MATRIX, Y_MATRIX, Z_MATRIX
from ..numerics.complex_arithmetic import complex_conj, complex_mul

if TYPE_CHECKING:
    from ...circuit import Circuit


def _expectation_z(circuit: Circuit, qubits: tuple[int, ...]) -> torch.Tensor:
    """Evaluate per-qubit Z expectations for a local statevector circuit."""

    from .local import state

    for qubit in qubits:
        if not 0 <= qubit < circuit.n_qubits:
            raise ValueError("wire is outside the statevector")
    probabilities = torch.abs(state(circuit)) ** 2
    key = (qubits, str(probabilities.device), probabilities.dtype)
    signs = circuit._statevector_z_signs.get(key)
    if signs is None:
        basis = torch.arange(
            probabilities.shape[-1], dtype=torch.int64, device=probabilities.device
        )
        signs = torch.stack(
            tuple(
                1 - 2 * ((basis >> (circuit.n_qubits - 1 - qubit)) & 1)
                for qubit in qubits
            ),
            dim=-1,
        ).to(dtype=probabilities.dtype)
        circuit._statevector_z_signs[key] = signs
    return probabilities @ signs


def _expectation_pauli_string(
    circuit: Circuit,
    *,
    x: tuple[int, ...],
    y: tuple[int, ...],
    z: tuple[int, ...],
) -> torch.Tensor:
    """Evaluate one X/Y/Z product observable for a local statevector circuit."""

    from .local import _apply_gate_matrix, state
    from .pauli_expectation_dispatch import (
        _try_apply_cataloged_statevector_pauli_expectation,
    )

    current_state = state(circuit)
    operators = tuple(
        (qubit, axis)
        for axis, qubits in (("X", x), ("Y", y), ("Z", z))
        for qubit in qubits
    )
    if (
        current_state.device.type == "flagos"
        and current_state.dtype == torch.complex128
        and current_state.shape[-1] >= 1 << 22
    ):
        from .expectation_autograd import _apply_pauli_expectation

        return _apply_pauli_expectation(current_state, operators, circuit.n_qubits)
    dispatched = _try_apply_cataloged_statevector_pauli_expectation(
        current_state, operators
    )
    if dispatched is not None:
        return dispatched
    transformed = current_state
    for operator, qubits in ((X_MATRIX, x), (Y_MATRIX, y), (Z_MATRIX, z)):
        matrix = operator.to(device=current_state.device, dtype=current_state.dtype)
        for qubit in qubits:
            transformed = _apply_gate_matrix(
                transformed,
                matrix,
                (qubit,),
                circuit.n_qubits,
                uses_diagonal_kernel=False,
            )
    value = complex_mul(complex_conj(current_state), transformed).sum(dim=-1)
    return torch.real(value)


def _sample_statevector(
    circuit: Circuit,
    *,
    shots: int,
    generator: torch.Generator | None,
    return_bits: bool,
) -> torch.Tensor:
    """Sample local statevector probabilities as indices or bitstrings."""

    from .local import state
    from .operations import _bits_from_indices

    probabilities = torch.abs(state(circuit)) ** 2
    samples = torch.multinomial(
        probabilities, num_samples=shots, replacement=True, generator=generator
    )
    return _bits_from_indices(samples, circuit.n_qubits) if return_bits else samples


def _expectation_from_operators(
    *ops: tuple[Any, Sequence[int]],
    ket: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a dense operator product against an explicit statevector."""

    from ...circuit import Circuit
    from .local import state

    input_state = ket.reshape(1, -1) if ket.ndim == 1 else ket
    n_qubits = int(
        torch.log2(torch.tensor(input_state.shape[-1], dtype=torch.float32)).item()
    )
    circuit = Circuit(
        n_qubits,
        bsz=input_state.shape[0],
        device=input_state.device,
        dtype=input_state.dtype if input_state.is_complex() else None,
        inputs=input_state,
    )
    for matrix, qubits in ops:
        circuit.any(*qubits, matrix=matrix)
    return complex_mul(complex_conj(input_state), state(circuit)).sum(dim=-1)


def _measure_wire(
    state: torch.Tensor,
    wire: int,
    n_wires: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    shaped = state.reshape((2,) * n_wires)
    axes = tuple(axis for axis in range(n_wires) if axis != wire)
    magnitudes = torch.abs(shaped) ** 2
    probabilities = magnitudes.sum(dim=axes) if axes else magnitudes
    bit = int(torch.multinomial(probabilities, 1, generator=generator).item())
    selector = torch.arange(2, device=state.device) == bit
    view_shape = [1] * n_wires
    view_shape[wire] = 2
    collapsed = shaped * selector.reshape(view_shape)
    norm = torch.linalg.vector_norm(collapsed)
    return (collapsed / norm).reshape(1, -1), bit


def _measure_wires_batched(
    state: torch.Tensor,
    wire: int,
    n_wires: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    shaped = state.reshape((state.shape[0],) + (2,) * n_wires)
    axes = tuple(axis + 1 for axis in range(n_wires) if axis != wire)
    probabilities = (
        (torch.abs(shaped) ** 2).sum(dim=axes) if axes else (torch.abs(shaped) ** 2)
    )
    bits = torch.multinomial(probabilities, 1, generator=generator).reshape(-1)
    selector = torch.arange(2, device=state.device).reshape(1, 2) == bits.reshape(-1, 1)
    view_shape = [state.shape[0]] + [1] * n_wires
    view_shape[wire + 1] = 2
    collapsed = shaped * selector.reshape(view_shape)
    norms = torch.linalg.vector_norm(collapsed.reshape(state.shape[0], -1), dim=1)
    return collapsed.reshape_as(state) / norms.reshape(-1, 1), bits
