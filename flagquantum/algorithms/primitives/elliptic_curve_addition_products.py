# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ECAddStepFour.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add.py

"""Subtract the squared slope and XOR the updated coordinate product."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.bits.bitwise import xor_ir
from ...compiler.registers.modular.modular_addition import modular_add_requirements
from ...compiler.registers.modular.modular_subtraction import modular_subtract_ir
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import montgomery_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...dynamic import DynamicCircuit
from ...simulation.elliptic_curve_values import addition_products_values
from .elliptic_curve_window_product import CurveWindowProduct


def addition_products_signature(width: SymbolicInt) -> Signature:
    """Declare the three Montgomery through words of the source stage."""
    word = montgomery_register_type(width)
    return Signature(Register(name, word) for name in ("x", "y", "lam"))


@dataclass(frozen=True)
class _BoundEllipticCurveAdditionProducts:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Curve addition products require a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveAdditionProducts:
    """Apply the locked fourth stage with caller-owned clean work.

    Words are big-endian. The input x and lam, the lowered square and updated
    x must satisfy the residue promises of the arithmetic. The final product
    is XORed into an arbitrary y. Work returns to zero, and classical slots
    retain the latest outcomes. This preserves the source's window reduction.
    """

    width: int
    modulus: int
    window: int = 1

    def __post_init__(self) -> None:
        CurveWindowProduct(self.width, self.modulus, self.window)

    @property
    def source_signature(self) -> Signature:
        return addition_products_signature(self.width)

    def classical_values(self, x: int, y: int, lam: int) -> tuple[int, int, int]:
        """Evaluate the source square and overwrite-product callback."""
        return addition_products_values(self.width, self.modulus, x, y, lam)

    def workspace_requirements(self) -> tuple[int, int]:
        n = self.width
        multiply = CurveWindowProduct(n, self.modulus, self.window)
        sub_work, sub_bits = modular_add_requirements(n)
        return n + multiply.work_size + sub_work, max(2 * n, sub_bits)

    def make_on(
        self,
        *,
        x: tuple[int, ...],
        y: tuple[int, ...],
        lam: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundEllipticCurveAdditionProducts:
        n, p = self.width, self.modulus
        q, c = self.workspace_requirements()
        if any(len(word) != n for word in (x, y, lam)):
            raise ValueError("Each curve product operand must have the declared width")
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Curve addition products require exact workspace sizes")
        wires = (*x, *y, *lam, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Curve addition products require distinct nonnegative wires and slots"
                )
        x = tuple(range(n))
        y = tuple(range(n, 2 * n))
        lam = tuple(range(2 * n, 3 * n))
        work = tuple(range(3 * n, len(wires)))
        slope_copy = work[:n]
        multiply = CurveWindowProduct(n, p, self.window)
        product_work = work[n : n + multiply.work_size]
        sub_work = work[n + multiply.work_size :]
        bits = classical_bits[: 2 * n]
        forward_square, square, reverse_square = multiply.emit(
            slope_copy, lam, product_work, bits
        )
        forward_product, product, reverse_product = multiply.emit(
            x, lam, product_work, bits
        )
        copy = xor_ir(lam, slope_copy)
        return _BoundEllipticCurveAdditionProducts(
            wires,
            (
                *copy,
                *forward_square,
                *modular_subtract_ir(square, x, p, sub_work, classical_bits[:n]),
                *reverse_square,
                *copy,
                *forward_product,
                *xor_ir(product, y),
                *reverse_product,
            ),
        )
