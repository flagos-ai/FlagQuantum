# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ECAddStepSix.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add.py

"""Handle exceptional points and uncompute the source addition flags."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.bits.bitwise import fanout_ir
from ...compiler.registers.bits.controlled_register_xor import controlled_xor_ir
from ...compiler.registers.comparison.equality import equality_ir
from ...compiler.registers.control.multi_control import multi_and_ir, multi_control_x_ir
from ...compiler.registers.modular.controlled_modular_addition import (
    controlled_modular_add_ir,
    controlled_modular_requirements,
)
from ...compiler.registers.modular.controlled_modular_subtraction import (
    controlled_modular_subtract_ir,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...dynamic import DynamicCircuit
from ...simulation.elliptic_curve_values import addition_finish_values


def addition_finish_signature(width: SymbolicInt) -> Signature:
    """Declare the five consumed flags and four through coordinates."""
    word = montgomery_register_type(width)
    return Signature(
        (
            *(
                Register(name, quantum_bit_type(), side=RegisterSide.LEFT)
                for name in ("f1", "f2", "f3", "f4", "ctrl")
            ),
            *(Register(name, word) for name in ("a", "b", "x", "y")),
        )
    )


@dataclass(frozen=True)
class _BoundEllipticCurveAdditionFinish:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Curve addition finish requires a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveAdditionFinish:
    """Emit the source exceptional-point and flag-erasure sequence.

    Coordinates are big-endian Montgomery words. The source requires all
    five flags to be zero at release. The caller owns those wires; live
    residual flags indicate a violated source promise.
    """

    width: int
    modulus: int

    def __post_init__(self) -> None:
        n, p = self.width, self.modulus
        if type(n) is not int or n < 2:
            raise ValueError("Curve addition finish requires width >= 2")
        if type(p) is not int or not 0 < p < (1 << n) - 1:
            raise ValueError("Curve addition finish requires modulus+1 to fit")

    @property
    def source_signature(self) -> Signature:
        return addition_finish_signature(self.width)

    def classical_values(
        self,
        f1: int,
        f2: int,
        f3: int,
        f4: int,
        ctrl: int,
        a: int,
        b: int,
        x: int,
        y: int,
    ) -> tuple[int, int, int, int]:
        """Preserve the source infinity selection and opposite-point override."""
        return addition_finish_values(f1, f2, f3, f4, ctrl, a, b, x, y)

    def workspace_requirements(self) -> tuple[int, int]:
        n = self.width
        q, c = controlled_modular_requirements(n)
        return max(3 * n - 1, q + 1), max(3 * n - 1, c)

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
    ) -> _BoundEllipticCurveAdditionFinish:
        n = self.width
        q, c = self.workspace_requirements()
        if any(len(word) != n for word in (a, b, x, y)):
            raise ValueError("Each curve finish operand must have the declared width")
        if len(flags) != 5 or len(workspace) != q or len(classical_bits) != c:
            raise ValueError(
                "Curve addition finish requires five flags and exact workspace sizes"
            )
        wires = (*a, *b, *x, *y, *flags, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Curve addition finish requires distinct nonnegative wires and slots"
                )
        a, b, x, y = (tuple(range(i * n, (i + 1) * n)) for i in range(4))
        f1, f2, f3, f4, ctrl = range(4 * n, 4 * n + 5)
        work = tuple(range(4 * n + 5, len(wires)))
        bits = classical_bits
        arithmetic_q, arithmetic_c = controlled_modular_requirements(n)
        arithmetic = work[1 : 1 + arithmetic_q]
        arith_bits = bits[:arithmetic_c]
        apply = Instruction("ccx", (f1, f2, work[0]))
        wide = tuple(
            gate
            for controls, target in (
                (a + x + y, f1),
                (x + a + b, f1),
                (a + b + y, f2),
                (x + y + b, f2),
            )
            for gate in multi_control_x_ir(
                controls, (0,) * (3 * n), target, work[: 3 * n - 1], bits[: 3 * n - 1]
            )
        )
        side_work, side_bits = work[: 2 * n - 1], bits[: 2 * n - 1]
        zero_pattern = (0,) * (2 * n)
        return _BoundEllipticCurveAdditionFinish(
            wires,
            (
                *multi_control_x_ir((f2, f3, f4), (0, 0, 0), ctrl, work[:2], bits[:2]),
                *wide,
                *controlled_xor_ir(f4, a, x),
                *controlled_xor_ir(f4, b, y),
                apply,
                *controlled_modular_subtract_ir(
                    work[0], a, x, self.modulus, arithmetic, arith_bits
                ),
                apply,
                apply,
                *controlled_modular_add_ir(
                    work[0], b, y, self.modulus, arithmetic, arith_bits
                ),
                apply,
                *equality_ir(a + b, x + y, f4, side_work, side_bits),
                *multi_control_x_ir(a + b, zero_pattern, f3, side_work, side_bits),
                *multi_and_ir(x + y, zero_pattern, side_work),
                *fanout_ir(side_work[-1], (f1, f2)),
                *multi_and_ir(
                    x + y,
                    zero_pattern,
                    side_work,
                    classical_bits=side_bits,
                    uncompute=True,
                ),
            ),
        )
