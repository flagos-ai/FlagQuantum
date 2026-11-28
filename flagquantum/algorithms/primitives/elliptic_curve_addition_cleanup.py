# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ECAddStepFive.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add.py

"""Restore the coordinate representation and uncompute the selected slope."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.bits.controlled_register_xor import grouped_controlled_xor_ir
from ...compiler.registers.control.control_predicate import ControlRegister
from ...compiler.registers.control.multi_control import multi_control_x_ir
from ...compiler.registers.modular.controlled_modular_addition import (
    controlled_modular_requirements,
)
from ...compiler.registers.modular.controlled_modular_negation import (
    controlled_modular_negate_ir,
    controlled_modular_negation_requirements,
)
from ...compiler.registers.modular.controlled_modular_subtraction import (
    controlled_modular_subtract_ir,
)
from ...compiler.registers.modular.modular_addition import (
    modular_add_ir,
    modular_add_requirements,
)
from ...compiler.registers.modular.modular_inverse import (
    modular_inverse_ir,
    modular_inverse_outputs,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...dynamic import DynamicCircuit
from ...simulation.elliptic_curve_values import addition_cleanup_values
from .elliptic_curve_window_product import CurveWindowProduct


def addition_cleanup_signature(width: SymbolicInt) -> Signature:
    """Declare the source cleanup stage with a consumed slope word."""
    word = montgomery_register_type(width)
    return Signature(
        (
            Register("ctrl", quantum_bit_type()),
            *(Register(name, word) for name in ("a", "b", "x", "y", "lam_r")),
            Register("lam", word, side=RegisterSide.LEFT),
        )
    )


@dataclass(frozen=True)
class _BoundEllipticCurveAdditionCleanup:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Curve addition cleanup requires a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveAdditionCleanup:
    """Clean an incoming slope satisfying the locked source's Free promise.

    Coordinates and slopes use big-endian Montgomery residues. Nonzero x
    must be a unit modulo p. The supplied slope must equal the bitwise XOR
    of the two values that the source selects under the active controls.
    This includes zero slope in the inactive branch. The slope and all work
    return to zero on that domain. Classical slots retain latest outcomes.
    """

    width: int
    modulus: int
    window: int = 1

    def __post_init__(self) -> None:
        CurveWindowProduct(self.width, self.modulus, self.window)

    @property
    def source_signature(self) -> Signature:
        return addition_cleanup_signature(self.width)

    def classical_values(
        self, ctrl: int, a: int, b: int, x: int, y: int, lam_r: int, lam: int
    ) -> tuple[int, int, int, int, int, int]:
        """Return source coordinate restoration after consuming the slope value."""
        return addition_cleanup_values(self.modulus, ctrl, a, b, x, y, lam_r, lam)

    def workspace_requirements(self) -> tuple[int, int]:
        n = self.width
        multiply = CurveWindowProduct(n, self.modulus, self.window)
        _, sub_bits = controlled_modular_requirements(n)
        return 9 * n + 8 + multiply.work_size, max(2 * n, sub_bits)

    def make_on(
        self,
        *,
        ctrl: int,
        a: tuple[int, ...],
        b: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        lam_r: tuple[int, ...],
        lam: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundEllipticCurveAdditionCleanup:
        n = self.width
        q, c = self.workspace_requirements()
        if any(len(word) != n for word in (a, b, x, y, lam_r, lam)):
            raise ValueError("Each curve cleanup operand must have the declared width")
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Curve addition cleanup requires exact workspace sizes")
        wires = (ctrl, *a, *b, *x, *y, *lam_r, *lam, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Curve addition cleanup requires distinct nonnegative wires and slots"
                )
        a, b, x, y, lam_r, lam = (
            tuple(range(1 + i * n, 1 + (i + 1) * n)) for i in range(6)
        )
        work = tuple(range(1 + 6 * n, len(wires)))
        return _BoundEllipticCurveAdditionCleanup(
            wires, self._emit(a, b, x, y, lam_r, lam, work, classical_bits)
        )

    def _emit(
        self,
        a: tuple[int, ...],
        b: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        lam_r: tuple[int, ...],
        lam: tuple[int, ...],
        work: tuple[int, ...],
        bits: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        n, p, w = self.width, self.modulus, self.window
        ctrl = 0
        history, inverse_work = work[: 4 * n], work[4 * n : 9 * n + 8]
        inv_bits = bits[: max(2 * n - 1, n + 1)]
        inv_x, inv_returned_work = modular_inverse_outputs(x, inverse_work)
        mul_work = work[9 * n + 8 :]
        forward, result, reverse = CurveWindowProduct(n, p, w).emit(
            inv_x, y, mul_work, bits[: 2 * n]
        )
        # The multiply carry bank is clean while its output history is live.
        select_work = mul_work[-(n + w - 1) :]
        select = tuple(
            gate
            for product_bit, slope_bit in zip(result, lam, strict=True)
            for gate in multi_control_x_ir(
                (ctrl, product_bit), (1, 1), slope_bit, select_work[:1], bits[:1]
            )
        )
        tangent = grouped_controlled_xor_ir(
            ((ctrl,), inv_x),
            (ControlRegister(1, (1,)), ControlRegister(n, (0,))),
            lam_r,
            lam,
            select_work[:n],
            bits[:n],
        )
        neg_q, neg_c = controlled_modular_negation_requirements(n)
        add_q, add_c = modular_add_requirements(n)
        sub_q, sub_c = controlled_modular_requirements(n)
        return (
            *modular_inverse_ir(x, history, p, inverse_work, inv_bits),
            *forward,
            *select,
            *tangent,
            *reverse,
            *modular_inverse_ir(
                inv_x, history, p, inv_returned_work, inv_bits, adjoint=True
            ),
            *controlled_modular_negate_ir(ctrl, x, p, work[:neg_q], bits[:neg_c]),
            *modular_add_ir(a, x, p, work[:add_q], bits[:add_c]),
            *controlled_modular_subtract_ir(ctrl, b, y, p, work[:sub_q], bits[:sub_c]),
        )
