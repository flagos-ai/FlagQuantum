# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 for_testing Pauli walk fixtures.

"""Pauli walk fixtures built from the existing PREPARE, SELECT and walk owners."""

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.skip("Pauli walk fixtures require NumPy", allow_module_level=True)

from flagquantum.algorithms.core import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler.registers.lookup.ordered_pauli_select import (
    ordered_pauli_select_ir,
)
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    uniform_preparation_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.matrix_register_fixture import MatrixRegisterFixture

PauliWords = tuple[str, ...]


def uniform_ring_words(sites: int) -> PauliWords:
    """Build the circular two-site products in original Hamiltonian order."""
    if sites < 2:
        raise ValueError("A one-site XY product is not Hermitian")
    terms = []
    for i in range(sites):
        following = (i + 1) % sites
        word = ["I"] * sites
        word[i] = "XYZ"[i % 3]
        word[following] = "XYZ"[(i + 1) % 3]
        terms.append("".join(word))
    return tuple(terms)


def fixture_hamiltonian(
    words: PauliWords, coefficients: tuple[complex, ...]
) -> Hamiltonian:
    return Hamiltonian(
        HamiltonianTerm(complex(coefficient), word.lower(), range(len(word)))
        for word, coefficient in zip(words, coefficients, strict=True)
    )


def uniform_prepare(
    count: int, cvs: int | tuple[int, ...] = (), qlambda: float = 0.0
) -> BlackBoxPrepare:
    width = (count - 1).bit_length()
    values = (cvs,) if isinstance(cvs, int) else tuple(cvs)

    def prepare(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        readouts: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        if values:
            raise ValueError("The source PREPARE wrapper has no ctrl register")
        return uniform_preparation_ir(
            count, ports["selection"], work, readouts, adjoint=adjoint
        )

    return BlackBoxPrepare((("selection", width),), (), prepare, qlambda)


def uniform_ring_walk(sites: int) -> QubitizationWalk:
    words = uniform_ring_words(sites)
    signal = uniform_prepare(len(words), qlambda=float(len(words)))
    return _walk(signal, words, sites, little_endian=False)


@dataclass(frozen=True)
class RandomPauliFixture:
    prepare: MatrixRegisterFixture
    words: PauliWords
    target_width: int

    def __post_init__(self) -> None:
        np.testing.assert_almost_equal(self.prepare.unitary()[:, 0].imag, 0)

    @classmethod
    def random(
        cls, selection_width: int, target_width: int, rng: np.random.RandomState
    ) -> "RandomPauliFixture":
        matrix = MatrixRegisterFixture.random(
            selection_width, random_state=rng
        ).unitary()
        column = matrix[:, 0]
        matrix *= (column.conj() / np.abs(column))[:, None]
        prepare = MatrixRegisterFixture(
            selection_width, tuple(tuple(row) for row in matrix)
        )
        words = tuple(
            "".join("IXYZ"[item] for item in rng.randint(0, 4, size=target_width))
            for _ in range(1 << selection_width)
        )
        return cls(prepare, words, target_width)

    @property
    def alphas(self) -> tuple[complex, ...]:
        return tuple(self.prepare.unitary()[:, 0] ** 2)

    def selector(self, control: int | None = None) -> SelectBlockEncoding:
        encoding = self.walk().block_encoding
        assert isinstance(encoding, SelectBlockEncoding)
        if control is None:
            return encoding

        def controlled(
            value: int,
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            readouts: tuple[int, ...],
            adjoint: bool,
        ) -> tuple[Instruction, ...]:
            return ordered_pauli_select_ir(
                self.words,
                ports["selection"],
                ports["target"],
                control=ports["control"],
                control_value=value,
                adjoint=adjoint,
            )

        return encoding.with_control(control, controlled)

    def hamiltonian(self) -> Hamiltonian:
        return fixture_hamiltonian(self.words, self.alphas)

    def walk(self) -> QubitizationWalk:
        def prepare(
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            readouts: tuple[int, ...],
            adjoint: bool,
        ) -> tuple[Instruction, ...]:
            matrix = self.prepare.adjoint() if adjoint else self.prepare
            return (
                Instruction(
                    "matrix_fixture", ports["selection"], matrix=matrix.tensor()
                ),
            )

        signal = BlackBoxPrepare(
            (("selection", self.prepare.bitsize),), (), prepare, 1.0
        )
        return _walk(signal, self.words, self.target_width, little_endian=True)


def _walk(
    signal: BlackBoxPrepare,
    words: PauliWords,
    target_width: int,
    *,
    little_endian: bool,
) -> QubitizationWalk:
    selector = SelectPauliLCU(signal.selection_width, target_width, words)

    def select(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        readouts: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        address = ports["selection"]
        if little_endian:
            return ordered_pauli_select_ir(
                words, address, ports["target"], adjoint=adjoint
            )
        return selector.make_bounded_on(
            selection=address,
            target=ports["target"],
            workspace=work,
            classical_bits=readouts,
        ).emit()

    return QubitizationWalk(
        SelectBlockEncoding(
            signal.selection_layout, (("target", target_width),), select, signal
        )
    )
