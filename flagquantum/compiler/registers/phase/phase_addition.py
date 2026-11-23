# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the fixed-point action of Qualtran v0.7.0 AddIntoPhaseGrad.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/phase_gradient.py
# Changes: lower the arithmetic through the shared measured Add and CAdd.

"""Add or subtract a fixed-point word into a phase-gradient register."""

from __future__ import annotations

from ....core.ir import Instruction
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..arithmetic.addition import add_adjoint_ir, add_ir
from ..arithmetic.controlled_addition import (
    controlled_add_adjoint_ir,
    controlled_add_ir,
)
from ..bits.bitwise import _validate_wires
from ..resources.symbolic_resource_model import SymbolicResourceFormula


def source_phase_add_signature(
    input_width: SymbolicInt,
    gradient_width: SymbolicInt,
    *,
    controlled: bool = False,
    input_fraction_bits: SymbolicInt | None = None,
) -> Signature:
    """Declare the source's through ports, including mixed fixed-point input."""
    fraction = input_width if input_fraction_bits is None else input_fraction_bits
    ports: tuple[Register, ...] = (
        (Register("ctrl", quantum_bit_type()),) if controlled else ()
    )
    if input_width != 0:
        ports += (Register("x", fixed_point_register_type(input_width, fraction)),)
    if gradient_width != 0:
        ports += (
            Register(
                "phase_grad", fixed_point_register_type(gradient_width, gradient_width)
            ),
        )
    return Signature(ports)


def source_phase_add_formula(
    input_width: SymbolicInt, gradient_width: SymbolicInt, *, controlled: bool = False
) -> SymbolicResourceFormula:
    """Report the locked abstract estimate, including negative small-width values."""
    control_width = int(controlled)
    return SymbolicResourceFormula(
        and_bloq=0,
        clifford=0,
        measurement=0,
        logical_qubits=smax(3, input_width + gradient_width + control_width),
        toffoli=(1 + control_width) * (gradient_width - 2),
    )


def _active_widths(
    input_width: int, gradient_width: int, right_shift: int
) -> tuple[int, int]:
    if (
        type(input_width) is not int
        or type(gradient_width) is not int
        or input_width < 0
        or gradient_width < 1
    ):
        raise ValueError(
            "Phase addition requires nonnegative input and positive gradient widths"
        )
    if type(right_shift) is not int or right_shift < 0:
        raise ValueError("Phase addition right shift must be a nonnegative integer")
    active = min(input_width, max(0, gradient_width - right_shift))
    return (active, active + right_shift) if active else (0, 0)


def phase_add_workspace_size(
    input_width: int,
    gradient_width: int,
    *,
    right_shift: int = 0,
    controlled: bool = False,
) -> int:
    """Count clean work qubits and equally many reusable classical slots."""
    if type(controlled) is not bool:
        raise ValueError("Phase addition controlled mode must be boolean")
    active, prefix = _active_widths(input_width, gradient_width, right_shift)
    return max(0, prefix - 1) + active * controlled


def phase_add_ir(
    source: tuple[int, ...],
    gradient: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    right_shift: int = 0,
    sign: int = 1,
    control: int | None = None,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Accumulate a shifted unsigned fraction in a big-endian phase word.

    Truncate the shifted fraction at the gradient precision. Only the
    destination prefix containing its retained bits and higher carries is
    updated. The remaining low bits keep their original values. Use sign=-1
    for the adjoint, and control_value=0 or 1 for a coherent control.
    The workspace starts and ends at zero. When controlled, the loaded
    input bits occupy the workspace prefix, followed by carries.
    """
    active, prefix = _active_widths(len(source), len(gradient), right_shift)
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("Phase addition sign must be -1 or 1")
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Phase addition control value must be zero or one")
    enable = () if control is None else (control,)
    _validate_wires((*source, *gradient, *enable, *workspace))
    loaded = active if control is not None else 0
    required = max(0, prefix - 1) + loaded
    if len(workspace) != required:
        raise ValueError(f"Phase addition requires {required} clean work wires")
    if (
        len(classical_bits) != required
        or len(set(classical_bits)) != required
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(f"Phase addition requires {required} distinct classical slots")
    if not active:
        return ()
    left, right = source[:active], gradient[:prefix]
    if control is None:
        emit = add_ir if sign == 1 else add_adjoint_ir
        return emit(left, right, workspace, classical_bits)
    controlled_emit = controlled_add_ir if sign == 1 else controlled_add_adjoint_ir
    return controlled_emit(
        control,
        left,
        right,
        workspace[:loaded],
        workspace[loaded:],
        classical_bits[:loaded],
        classical_bits[loaded:],
        control_value=control_value,
    )
