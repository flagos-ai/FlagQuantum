"""Independent dense Circuit for the complete declared product formula."""

from collections.abc import Sequence

import numpy as np
import torch

import flagquantum as fq

from .model import ComplexArray, LinearSystem, PauliCoefficient


def _dense(axis_word: Sequence[int]) -> ComplexArray:
    axes = (
        ((1, 0), (0, 1)),
        ((0, 1), (1, 0)),
        ((0, -1j), (1j, 0)),
        ((1, 0), (0, -1)),
    )
    a, b, c = (np.asarray(axes[axis], dtype=np.complex128) for axis in axis_word)
    return np.asarray(np.kron(a, np.kron(b, c)), dtype=np.complex128)


def reference_circuit(
    model: LinearSystem,
    groups: tuple[tuple[PauliCoefficient, ...], tuple[PauliCoefficient, ...]],
    duration: int,
    steps: int,
) -> fq.Circuit:
    """Apply an independent dense preparation and every source time slice."""
    identity = np.eye(8, dtype=np.complex128)
    difference = identity[:, 0] - model.initial
    prepare = identity - 2 * np.outer(difference, difference.conj()) / np.vdot(
        difference, difference
    )
    circuit = fq.Circuit(3, dtype=torch.complex128).any(
        0, 1, 2, matrix=torch.as_tensor(prepare)
    )
    operators = [
        [(term.coefficient, _dense(term.axes)) for term in terms] for terms in groups
    ]
    for index in range(steps):
        fraction = index / steps
        for time, repeats, terms in zip(
            (duration / steps * (1 - fraction), duration / steps * fraction),
            (1, 10),
            operators,
            strict=True,
        ):
            product = identity
            for coefficient, pauli in terms:
                energies, basis = np.linalg.eigh(pauli)
                rotation = (
                    basis * np.exp(-1j * time * coefficient * energies / repeats)
                ) @ basis.conj().T
                product = rotation @ product
            circuit.any(
                0,
                1,
                2,
                matrix=torch.as_tensor(np.linalg.matrix_power(product, repeats)),
            )
    return circuit
