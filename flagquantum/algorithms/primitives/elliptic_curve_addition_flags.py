# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ECAddStepOne.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add.py

"""Bind the first reversible elliptic-curve addition stage."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.comparison.equality import equality_ir
from ...compiler.registers.control.multi_control import multi_control_x_ir
from ...compiler.registers.modular.modular_negation import (
    modular_negate_ir,
    modular_negation_requirements,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...dynamic import DynamicCircuit
from ...simulation.elliptic_curve_values import addition_flags_values

_FLAG_NAMES = ("f1", "f2", "f3", "f4", "ctrl")
_COORDINATE_NAMES = ("a", "b", "x", "y")


def addition_flags_signature(width: SymbolicInt) -> Signature:
    """Declare the source's five allocated bits and four Montgomery through ports."""
    return Signature(
        (
            *(
                Register(name, quantum_bit_type(), side=RegisterSide.RIGHT)
                for name in _FLAG_NAMES
            ),
            *(
                Register(name, montgomery_register_type(width))
                for name in _COORDINATE_NAMES
            ),
        )
    )


@dataclass(frozen=True)
class _BoundEllipticCurveAdditionFlags:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Curve addition flags require a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveAdditionFlags:
    """Compute flags onto five clean wires while retaining the four input words.

    Coordinates are big-endian Montgomery residues; (0, 0) encodes infinity.
    Flag order is f1 (equal x), f2 (opposite y), f3/f4 (infinity), ctrl
    (none of f2/f3/f4). Scratch returns to zero. Classical slots are reused
    between stages and retain the latest measurement outcomes.
    """

    width: int
    modulus: int

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 2:
            raise ValueError("Curve addition flags require a width of at least two")
        if (
            type(self.modulus) is not int
            or not 0 < self.modulus < (1 << self.width) - 1
        ):
            raise ValueError("Curve addition flags require modulus+1 to fit")

    @property
    def source_signature(self) -> Signature:
        return addition_flags_signature(self.width)

    def classical_values(
        self, a: int, b: int, x: int, y: int
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        """Return the original four predicates and combined control with the words."""
        return addition_flags_values(self.modulus, a, b, x, y)

    def workspace_requirements(self) -> tuple[int, int]:
        """Return the reused quantum and classical workspace sizes."""
        n = self.width
        negation_work, negation_bits = modular_negation_requirements(n)
        return max(negation_work, 2 * n - 1), max(negation_bits, 2 * n - 1)

    def make_on(
        self,
        *,
        a: tuple[int, ...],
        b: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        flags: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundEllipticCurveAdditionFlags:
        n = self.width
        q, c = self.workspace_requirements()
        if any(len(word) != n for word in (a, b, x, y)):
            raise ValueError("Each curve coordinate must have the declared width")
        if (
            len(flags) != len(_FLAG_NAMES)
            or len(workspace) != q
            or len(classical_bits) != c
        ):
            raise ValueError(
                "Curve addition flags require five outputs and exact workspace sizes"
            )
        wires = (*a, *b, *x, *y, *flags, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Curve addition flags require distinct nonnegative wires and slots"
                )

        a, b, x, y = (tuple(range(i * n, (i + 1) * n)) for i in range(4))
        f1, f2, f3, f4, ctrl = range(4 * n, 4 * n + 5)
        work = tuple(range(4 * n + 5, len(wires)))
        eq_work, eq_bits = work[: n - 1], classical_bits[: n - 1]
        negate = modular_negate_ir(y, self.modulus, work, eq_bits)
        inf_work, inf_bits = work[: 2 * n - 1], classical_bits[: 2 * n - 1]
        return _BoundEllipticCurveAdditionFlags(
            wires,
            (
                *equality_ir(a, x, f1, eq_work, eq_bits),
                *negate,
                *equality_ir(b, y, f2, eq_work, eq_bits),
                *negate,
                *multi_control_x_ir(a + b, (0,) * (2 * n), f3, inf_work, inf_bits),
                *multi_control_x_ir(x + y, (0,) * (2 * n), f4, inf_work, inf_bits),
                *multi_control_x_ir(
                    (f2, f3, f4), (0, 0, 0), ctrl, work[:2], classical_bits[:2]
                ),
            ),
        )
