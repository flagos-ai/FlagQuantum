# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ECAddStepTwo.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add.py

"""Reversible slope selection with explicit inverse and multiplication history."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.bits.controlled_register_xor import controlled_xor_ir
from ...compiler.registers.comparison.equality import equality_ir
from ...compiler.registers.control.multi_control import multi_control_x_ir
from ...compiler.registers.modular.controlled_modular_addition import (
    controlled_modular_requirements,
)
from ...compiler.registers.modular.controlled_modular_subtraction import (
    controlled_modular_subtract_ir,
)
from ...compiler.registers.modular.modular_addition import modular_add_requirements
from ...compiler.registers.modular.modular_inverse import (
    modular_inverse_ir,
    modular_inverse_outputs,
)
from ...compiler.registers.modular.modular_subtraction import modular_subtract_ir
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...dynamic import DynamicCircuit
from ...simulation.elliptic_curve_values import addition_slope_values
from .elliptic_curve_window_product import CurveWindowProduct


def addition_slope_signature(width: SymbolicInt) -> Signature:
    """Declare the two flags, through words and fresh Montgomery slope output."""
    word = montgomery_register_type(width)
    return Signature(
        (
            Register("f1", quantum_bit_type()),
            Register("ctrl", quantum_bit_type()),
            *(Register(name, word) for name in ("a", "b", "x", "y")),
            Register("lam", word, side=RegisterSide.RIGHT),
            Register("lam_r", word),
        )
    )


@dataclass(frozen=True)
class _BoundEllipticCurveAdditionSlope:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Curve addition slope requires a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveAdditionSlope:
    """Bind the locked slope circuit with zero-initialized output and workspace.

    The four coordinate words, tangent and slope are big-endian Montgomery
    words. Coordinates and tangent are residues; nonzero x-a must be a unit
    modulo p. The operation leaves both coordinate differences in their source
    roles, then returns inverse history, multiplication history and flag scratch
    to zero. It retains the source reduction and f1 update semantics.
    """

    width: int
    modulus: int
    window: int = 1

    def __post_init__(self) -> None:
        CurveWindowProduct(self.width, self.modulus, self.window)

    @property
    def source_signature(self) -> Signature:
        return addition_slope_signature(self.width)

    def classical_values(
        self, f1: int, ctrl: int, a: int, b: int, x: int, y: int, lam_r: int
    ) -> tuple[int, int, int, int, int, int, int, int]:
        """Evaluate the original high-level slope selection callback."""
        return addition_slope_values(
            self.width, self.modulus, f1, ctrl, a, b, x, y, lam_r
        )

    def workspace_requirements(self) -> tuple[int, int]:
        """Count clean quantum workspace and reusable classical measurement slots."""
        n = self.width
        multiply = CurveWindowProduct(n, self.modulus, self.window)
        _, sub_slots = controlled_modular_requirements(n)
        return 9 * n + 8 + multiply.work_size + 1, max(sub_slots, 2 * n)

    def make_on(
        self,
        *,
        f1: int,
        ctrl: int,
        a: tuple[int, ...],
        b: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        lam: tuple[int, ...],
        lam_r: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundEllipticCurveAdditionSlope:
        n = self.width
        q, c = self.workspace_requirements()
        if any(len(word) != n for word in (a, b, x, y, lam, lam_r)):
            raise ValueError("Each slope operand must have the declared width")
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Curve addition slope requires exact workspace sizes")
        wires = (f1, ctrl, *a, *b, *x, *y, *lam, *lam_r, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Curve addition slope requires distinct nonnegative wires and slots"
                )
        f1, ctrl = 0, 1
        a, b, x, y, lam, lam_r = (
            tuple(range(2 + i * n, 2 + (i + 1) * n)) for i in range(6)
        )
        work = tuple(range(2 + 6 * n, len(wires)))
        return _BoundEllipticCurveAdditionSlope(
            wires, self._emit(f1, ctrl, a, b, x, y, lam, lam_r, work, classical_bits)
        )

    def _emit(
        self,
        f1: int,
        ctrl: int,
        a: tuple[int, ...],
        b: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        lam: tuple[int, ...],
        lam_r: tuple[int, ...],
        work: tuple[int, ...],
        bits: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        n, p, w = self.width, self.modulus, self.window
        history, inverse_work = work[: 4 * n], work[4 * n : 9 * n + 8]
        multiply = work[9 * n + 8 : -1]
        mul_work = multiply[-(n + w - 1) :]
        equal_flag = work[-1]
        inv_bits = bits[: max(2 * n - 1, n + 1)]
        inv_x, inv_work_returned = modular_inverse_outputs(x, inverse_work)
        forward_mul, computed, reverse_mul = CurveWindowProduct(n, p, w).emit(
            inv_x, y, multiply, bits[: 2 * n]
        )
        equal = equality_ir(
            computed, lam_r, equal_flag, mul_work[: n - 1], bits[: n - 1]
        )
        select = tuple(
            gate
            for flag_value, slope in ((0, computed), (1, lam_r))
            for operand, result in zip(slope, lam, strict=True)
            for gate in multi_control_x_ir(
                (f1, ctrl, operand), (flag_value, 1, 1), result, mul_work[:2], bits[:2]
            )
        )
        controlled_xor = controlled_xor_ir(equal_flag, lam, lam_r)
        flip = Instruction("x", (equal_flag,))
        sub_q, sub_c = modular_add_requirements(n)
        csub_q, csub_c = controlled_modular_requirements(n)
        return (
            *modular_subtract_ir(a, x, p, work[:sub_q], bits[:sub_c]),
            *controlled_modular_subtract_ir(
                ctrl, b, y, p, work[:csub_q], bits[:csub_c]
            ),
            *modular_inverse_ir(x, history, p, inverse_work, inv_bits),
            *forward_mul,
            *equal,
            *select,
            flip,
            *controlled_xor,
            *multi_control_x_ir(
                (equal_flag, *lam_r), (1, *((0,) * n)), f1, mul_work[:n], bits[:n]
            ),
            *controlled_xor,
            flip,
            *equal,
            *reverse_mul,
            *modular_inverse_ir(
                inv_x, history, p, inv_work_returned, inv_bits, adjoint=True
            ),
        )
