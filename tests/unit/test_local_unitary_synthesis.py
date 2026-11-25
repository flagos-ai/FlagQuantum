"""Check absolute complex operators of local matrix synthesis and its controls."""

import sys
from itertools import product

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from flagquantum.compiler.optimization.local_unitary_synthesis import local_unitary_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "width,controlled,seed", tuple(product((1, 2), (False, True), (17, 23)))
)
def test_complete_complex_local_operators(
    width: int, controlled: bool, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(1 << width, 1 << width)) + 1j * rng.normal(
        size=(1 << width, 1 << width)
    )
    matrix, _ = np.linalg.qr(raw)
    matrix *= np.exp(0.37j)
    wires = tuple(range(1, width + 1)) if controlled else tuple(range(width))
    saved = matrix.copy()
    gates = local_unitary_ir(matrix, wires, control=0 if controlled else None)
    assert all(g.matrix is None for g in gates)
    assert not any(
        g.name in {"measure", "reset", "classical_xor", "any"} for g in gates
    )
    ir = CircuitIR(width + int(controlled), gates)
    assert CircuitIR.from_json(ir.to_json()) == ir
    expected = np.eye((1 << width) * (2 if controlled else 1), dtype=np.complex128)
    expected[-len(matrix) :, -len(matrix) :] = matrix
    torch.testing.assert_close(
        get_unitary(ir, dtype=torch.complex128),
        torch.tensor(expected),
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_array_equal(matrix, saved)


@pytest.mark.parametrize("width,controlled", tuple(product((1, 2), (False, True))))
@pytest.mark.parametrize("phase", [0.0, 0.31, -np.pi, np.pi / 2])
def test_scalar_and_degenerate_local_inputs(
    width: int, controlled: bool, phase: float
) -> None:
    matrix = np.eye(1 << width, dtype=np.complex128) * np.exp(1j * phase)
    wires = tuple(range(1, width + 1)) if controlled else tuple(range(width))
    gates = local_unitary_ir(matrix, wires, control=0 if controlled else None)
    expected = np.eye((1 << width) * (2 if controlled else 1), dtype=np.complex128)
    expected[-len(matrix) :, -len(matrix) :] = matrix
    torch.testing.assert_close(
        get_unitary(CircuitIR(width + int(controlled), gates), dtype=torch.complex128),
        torch.tensor(expected),
        atol=2e-10,
        rtol=0,
    )


@pytest.mark.parametrize(
    "matrix,wires,control",
    [
        ([[1, 0], [0, 2]], (0,), None),
        ([[1, 0], [0, float("nan")]], (0,), None),
        ([[1, 0], [0, 1]], (0, 1), None),
        (np.eye(8), (0, 1, 2), None),
        (np.eye(2), (0,), 0),
    ],
)
def test_invalid_local_payloads_are_refused(
    matrix: list[list[complex]] | NDArray[np.complex128],
    wires: tuple[int, ...],
    control: int | None,
) -> None:
    with pytest.raises(ValueError):
        local_unitary_ir(matrix, wires, control=control)


@pytest.mark.parametrize("delta", [1e-7, 1e-9])
@pytest.mark.parametrize("controlled", [False, True])
def test_small_interactions_are_not_dropped(delta: float, controlled: bool) -> None:
    matrix = np.diag(np.exp(1j * np.asarray([delta, -delta, -delta, delta])))
    wires = (1, 2) if controlled else (0, 1)
    gates = local_unitary_ir(matrix, wires, control=0 if controlled else None)
    expected = np.eye(8 if controlled else 4, dtype=np.complex128)
    expected[-4:, -4:] = matrix
    torch.testing.assert_close(
        get_unitary(CircuitIR(3 if controlled else 2, gates), dtype=torch.complex128),
        torch.tensor(expected),
        atol=2e-11,
        rtol=0,
    )


def test_reversed_targets_and_sdk_free_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("qualtran", "cirq", "quimb", "qiskit", "kickmix"):
        monkeypatch.setitem(sys.modules, name, None)
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    matrix, _ = np.linalg.qr(raw)
    gates = local_unitary_ir(matrix, (1, 0), control=2)
    expected = np.zeros((8, 8), dtype=np.complex128)
    for column in range(8):
        if column & 1 == 0:
            expected[column, column] = 1
            continue
        word = ((column >> 1) & 1) * 2 + ((column >> 2) & 1)
        for target in range(4):
            row = ((target & 1) << 2) | ((target >> 1) << 1) | 1
            expected[row, column] = matrix[target, word]
    torch.testing.assert_close(
        get_unitary(CircuitIR(3, gates), dtype=torch.complex128),
        torch.tensor(expected),
        atol=2e-10,
        rtol=0,
    )
