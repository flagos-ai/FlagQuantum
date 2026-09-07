"""Pure Double-Single statevector gate application and normalization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch

from ..numerics.double_single import (
    DoubleSingleComplexTensor,
    DoubleSingleTensor,
    double_single_sum,
)


def _component_view(
    value: torch.Tensor,
    *,
    logical_shape: tuple[int, ...],
    permutation: tuple[int, ...],
    gate_dimension: int,
) -> torch.Tensor:
    return value.reshape(logical_shape).permute(permutation).reshape(-1, gate_dimension)


def _restore_component(
    value: torch.Tensor,
    *,
    logical_shape: tuple[int, ...],
    inverse: tuple[int, ...],
) -> torch.Tensor:
    return value.reshape(logical_shape).permute(inverse).reshape(-1)


def _view_state(
    state: DoubleSingleComplexTensor,
    *,
    logical_shape: tuple[int, ...],
    permutation: tuple[int, ...],
    gate_dimension: int,
) -> DoubleSingleComplexTensor:
    def view(value: torch.Tensor) -> torch.Tensor:
        return _component_view(
            value,
            logical_shape=logical_shape,
            permutation=permutation,
            gate_dimension=gate_dimension,
        )

    return DoubleSingleComplexTensor(
        DoubleSingleTensor(view(state.real.high), view(state.real.low)),
        DoubleSingleTensor(view(state.imag.high), view(state.imag.low)),
    )


def _complex_column(
    value: DoubleSingleComplexTensor, index: int
) -> DoubleSingleComplexTensor:
    return DoubleSingleComplexTensor(
        DoubleSingleTensor(value.real.high[..., index], value.real.low[..., index]),
        DoubleSingleTensor(value.imag.high[..., index], value.imag.low[..., index]),
    )


def _broadcast_matrix_entry(
    matrix: DoubleSingleComplexTensor,
    row: int,
    column: int,
    like: torch.Tensor,
) -> DoubleSingleComplexTensor:
    def word(value: torch.Tensor) -> torch.Tensor:
        return value[row, column].expand_as(like)

    return DoubleSingleComplexTensor(
        DoubleSingleTensor(word(matrix.real.high), word(matrix.real.low)),
        DoubleSingleTensor(word(matrix.imag.high), word(matrix.imag.low)),
    )


def apply_double_single_gate(
    state: DoubleSingleComplexTensor,
    matrix: DoubleSingleComplexTensor,
    wires: Sequence[int],
    *,
    n_wires: int,
) -> DoubleSingleComplexTensor:
    """Apply one Double-Single gate matrix to a flat statevector."""

    wires = tuple(int(wire) for wire in wires)
    remaining = tuple(wire for wire in range(n_wires) if wire not in wires)
    permutation = remaining + wires
    inverse = tuple(permutation.index(wire) for wire in range(n_wires))
    gate_dimension = 2 ** len(wires)
    logical_shape = (2,) * n_wires
    values = _view_state(
        state,
        logical_shape=logical_shape,
        permutation=permutation,
        gate_dimension=gate_dimension,
    )
    rows = []
    for row in range(gate_dimension):
        accumulator = DoubleSingleComplexTensor(
            DoubleSingleTensor.zeros_like(values.real.high[..., 0]),
            DoubleSingleTensor.zeros_like(values.imag.high[..., 0]),
        )
        for column in range(gate_dimension):
            amplitude = _complex_column(values, column)
            factor = _broadcast_matrix_entry(matrix, row, column, amplitude.real.high)
            accumulator = accumulator.add(factor.multiply(amplitude))
        rows.append(accumulator)

    def stack(component: str, word: str) -> torch.Tensor:
        return torch.stack(
            [getattr(getattr(value, component), word) for value in rows], dim=-1
        )

    updated = DoubleSingleComplexTensor(
        DoubleSingleTensor(stack("real", "high"), stack("real", "low")),
        DoubleSingleTensor(stack("imag", "high"), stack("imag", "low")),
    )

    def restore(value: torch.Tensor) -> torch.Tensor:
        return _restore_component(value, logical_shape=logical_shape, inverse=inverse)

    return DoubleSingleComplexTensor(
        DoubleSingleTensor(restore(updated.real.high), restore(updated.real.low)),
        DoubleSingleTensor(restore(updated.imag.high), restore(updated.imag.low)),
    )


def _scale_state(
    state: DoubleSingleComplexTensor, factor: DoubleSingleTensor
) -> DoubleSingleComplexTensor:
    def expand(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return value.expand_as(like)

    scalar = DoubleSingleComplexTensor(
        DoubleSingleTensor(
            expand(factor.high, state.real.high),
            expand(factor.low, state.real.low),
        ),
        DoubleSingleTensor.zeros_like(state.imag.high),
    )
    return state.multiply(scalar)


def normalize_double_single_state(
    state: DoubleSingleComplexTensor,
) -> DoubleSingleComplexTensor:
    """Normalize a Double-Single statevector without changing representation."""

    norm_squared = double_single_sum(state.abs_squared())
    return _scale_state(state, norm_squared.reciprocal_sqrt())


def run_double_single_statevector(
    encoded_gates: Iterable[tuple[DoubleSingleComplexTensor, Sequence[int]]],
    *,
    n_wires: int,
    device: torch.device,
    renormalize_every: int,
) -> tuple[DoubleSingleComplexTensor, int]:
    """Evolve a zero state from lazily encoded Double-Single gates."""

    if renormalize_every < 0:
        raise ValueError("renormalize_every must be non-negative")
    high = torch.zeros(2**n_wires, dtype=torch.float32, device=device)
    high[0] = 1.0
    zero = torch.zeros_like(high)
    state = DoubleSingleComplexTensor(
        DoubleSingleTensor(high, zero),
        DoubleSingleTensor(torch.zeros_like(high), torch.zeros_like(high)),
    )
    normalization_count = 0
    for gate_number, (matrix, wires) in enumerate(encoded_gates, start=1):
        state = apply_double_single_gate(state, matrix, wires, n_wires=n_wires)
        if renormalize_every and gate_number % renormalize_every == 0:
            state = normalize_double_single_state(state)
            normalization_count += 1
    return state, normalization_count


def double_single_pauli_term_expectation(
    state: DoubleSingleComplexTensor,
    ops: Sequence[tuple[int, str]],
    coefficient: DoubleSingleTensor,
    *,
    n_wires: int,
    matrix_for_op: Callable[[int, str], DoubleSingleComplexTensor],
) -> DoubleSingleTensor:
    """Evaluate one real-coefficient Pauli term in Double-Single arithmetic."""

    transformed = state
    for wire, name in ops:
        transformed = apply_double_single_gate(
            transformed,
            matrix_for_op(wire, name),
            (wire,),
            n_wires=n_wires,
        )
    products = state.real.multiply(transformed.real).add(
        state.imag.multiply(transformed.imag)
    )
    return double_single_sum(products).multiply(coefficient)


__all__ = (
    "apply_double_single_gate",
    "double_single_pauli_term_expectation",
    "normalize_double_single_state",
    "run_double_single_statevector",
)
