# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ECAddStepThree.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add.py

"""Controlled coordinate update and exact cleanup of the temporary triple."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.bits.bitwise import xor_ir
from ...compiler.registers.modular.controlled_modular_addition import (
    controlled_modular_add_ir,
    controlled_modular_requirements,
)
from ...compiler.registers.modular.controlled_modular_subtraction import (
    controlled_modular_subtract_ir,
)
from ...compiler.registers.modular.modular_addition import (
    modular_add_adjoint_ir,
    modular_add_ir,
    modular_add_requirements,
)
from ...compiler.registers.modular.modular_doubling import (
    modular_double_ir,
    modular_double_outputs,
    modular_double_requirements,
)
from ...compiler.registers.modular.modular_halving import modular_half_ir
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...dynamic import DynamicCircuit
from ...simulation.elliptic_curve_values import addition_coordinates_values
from .elliptic_curve_window_product import CurveWindowProduct


def addition_coordinates_signature(width: SymbolicInt) -> Signature:
    """Declare the source control and five unchanged port identities."""
    word = montgomery_register_type(width)
    return Signature(
        (
            Register("ctrl", quantum_bit_type()),
            *(Register(name, word) for name in ("a", "b", "x", "y", "lam")),
        )
    )


@dataclass(frozen=True)
class _BoundEllipticCurveAdditionCoordinates:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Curve addition coordinates require a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveAdditionCoordinates:
    """Subtract the product from y and add 3a to x when ctrl is set.

    All words are big-endian Montgomery residues. The physical product must
    satisfy the residue promise of the controlled subtraction. With the source
    slope relation, the updated y is zero. Work starts and ends zero.
    """

    width: int
    modulus: int
    window: int = 1

    def __post_init__(self) -> None:
        CurveWindowProduct(self.width, self.modulus, self.window)

    @property
    def source_signature(self) -> Signature:
        return addition_coordinates_signature(self.width)

    def classical_values(
        self, ctrl: int, a: int, b: int, x: int, y: int, lam: int
    ) -> tuple[int, int, int, int, int, int]:
        """Evaluate the source controlled coordinate callback."""
        return addition_coordinates_values(self.modulus, ctrl, a, b, x, y, lam)

    def workspace_requirements(self) -> tuple[int, int]:
        """Reserve independent product history and controlled arithmetic work."""
        n = self.width
        product = CurveWindowProduct(n, self.modulus, self.window)
        arithmetic, bits = controlled_modular_requirements(n)
        double, double_bits = modular_double_requirements(n)
        return max(product.work_size, n + double) + arithmetic, max(
            bits, 2 * n, double_bits
        )

    def make_on(
        self,
        *,
        ctrl: int,
        a: tuple[int, ...],
        b: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        lam: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundEllipticCurveAdditionCoordinates:
        n = self.width
        q, c = self.workspace_requirements()
        if any(len(word) != n for word in (a, b, x, y, lam)):
            raise ValueError(
                "Each coordinate update operand must have the declared width"
            )
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Curve addition coordinates require exact workspace sizes")
        wires = (ctrl, *a, *b, *x, *y, *lam, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Curve addition coordinates require distinct nonnegative wires and slots"
                )
        a, _, x, y, lam = (tuple(range(1 + i * n, 1 + (i + 1) * n)) for i in range(5))
        work = tuple(range(1 + 5 * n, len(wires)))
        return _BoundEllipticCurveAdditionCoordinates(
            wires, self._emit(a, x, y, lam, work, classical_bits)
        )

    def _emit(
        self,
        a: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        lam: tuple[int, ...],
        work: tuple[int, ...],
        bits: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        n, p = self.width, self.modulus
        product = CurveWindowProduct(n, p, self.window)
        control = 0
        c_q, c_c = controlled_modular_requirements(n)
        add_q, add_c = modular_add_requirements(n)
        dbl_q, dbl_c = modular_double_requirements(n)
        arith, arith_bits = work[-c_q:], bits[:c_c]
        forward_mul, result, reverse_mul = product.emit(
            x, lam, work[: product.work_size], bits[: 2 * n]
        )
        temporary, double_work = work[:n], work[n : n + dbl_q]
        triple, double_returned_work = modular_double_outputs(temporary, double_work)
        copy = xor_ir(a, temporary)
        return (
            *forward_mul,
            *controlled_modular_subtract_ir(control, result, y, p, arith, arith_bits),
            *reverse_mul,
            *copy,
            *modular_double_ir(temporary, p, double_work, bits[:dbl_c]),
            *modular_add_ir(a, triple, p, arith[:add_q], bits[:add_c]),
            *controlled_modular_add_ir(control, triple, x, p, arith, arith_bits),
            *modular_add_adjoint_ir(a, triple, p, arith[:add_q], bits[:add_c]),
            *modular_half_ir(triple, p, double_returned_work, bits[:dbl_c]),
            *copy,
        )
