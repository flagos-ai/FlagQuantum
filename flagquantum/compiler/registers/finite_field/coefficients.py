# Adapted from Qualtran v0.7.0 and the pinned reversible-circuit generator.
# Copyright 2024 Google LLC (gf2_addition.py and gf2_add_k.py)
# Copyright 2026 Google LLC (gen_gf_iadd.cc)
# SPDX-License-Identifier: Apache-2.0
# Sources:
# https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs/gf_arithmetic
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_iadd.cc
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.cc
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/control_helper.pybind.cc
# Changes: bind polynomial-basis registers and emit native instructions.

"""Register operations over binary extension fields."""

from dataclasses import dataclass, replace

from ....core.classical_control import instruction_condition_clauses
from ....core.ir import Instruction


@dataclass(frozen=True)
class ClassicalFieldCoefficient:
    """Reference a classical slot supplying one polynomial coefficient."""

    slot: int

    def __post_init__(self) -> None:
        if type(self.slot) is not int or self.slot < 0:
            raise ValueError("Field coefficient slot must be a nonnegative integer")


def gf2_add_coefficients_ir(
    source: tuple[int | bool | ClassicalFieldCoefficient, ...],
    target: tuple[int, ...],
    *,
    control: int | bool | ClassicalFieldCoefficient | None = None,
) -> tuple[Instruction, ...]:
    """XOR aligned quantum wires, classical slots, and constants into target.

    Both words use big-endian coefficient order. Emit from the least
    significant coefficient in the locked generator's broadcast order.
    Classical slots must have been defined by the caller. A classical
    control guards the full addition, including classical coefficients.
    """
    if len(source) != len(target):
        raise ValueError("Binary-field addition requires equal register widths")
    quantum_control = control if type(control) is int else None
    targets = target if quantum_control is None else (*target, quantum_control)
    if len(set(targets)) != len(targets) or any(
        type(wire) is not int or wire < 0 for wire in targets
    ):
        raise ValueError(
            "Binary-field target and control require distinct quantum wires"
        )
    if control is False:
        return ()
    homogeneous_quantum = all(type(coefficient) is int for coefficient in source)
    gates: list[Instruction] = []
    for coefficient, wire in zip(source[::-1], target[::-1], strict=True):
        if coefficient is False:
            continue
        if isinstance(coefficient, ClassicalFieldCoefficient):
            conditions = {"conditions": ((coefficient.slot, 1),)}
            gates.append(
                Instruction("x", (wire,), metadata=conditions)
                if quantum_control is None
                else Instruction("cx", (quantum_control, wire), metadata=conditions)
            )
        elif coefficient is True:
            gates.append(
                Instruction("x", (wire,))
                if quantum_control is None
                else Instruction("cx", (quantum_control, wire))
            )
        else:
            if (
                type(coefficient) is not int
                or coefficient < 0
                or coefficient in targets
            ):
                raise ValueError(
                    "Binary-field source wires must be disjoint from target and control"
                )
            if quantum_control is None:
                gates.append(Instruction("cx", (coefficient, wire)))
            elif homogeneous_quantum:
                gates.append(Instruction("ccx", (coefficient, quantum_control, wire)))
            else:
                # Kickmix's mixed dispatch puts the common control first.
                gates.append(Instruction("ccx", (quantum_control, coefficient, wire)))
    if isinstance(control, ClassicalFieldCoefficient):
        guard = (control.slot, 1)
        for i, gate in enumerate(gates):
            (coefficient_conditions,) = instruction_condition_clauses(gate) or ((),)
            gates[i] = replace(
                gate,
                metadata={
                    "conditions": tuple(sorted({guard, *coefficient_conditions}))
                },
            )
    return tuple(gates)


def gf2_add_ir(
    source: tuple[int, ...], target: tuple[int, ...], *, control: int | None = None
) -> tuple[Instruction, ...]:
    """XOR equal-width polynomial-basis registers, optionally under control."""
    if len(source) != len(target):
        raise ValueError("Binary-field addition requires equal register widths")
    wires = (*source, *target) if control is None else (*source, *target, control)
    if len(set(wires)) != len(wires):
        raise ValueError("Binary-field addition requires disjoint operands")
    pairs = zip(source, target, strict=True)
    if control is None:
        return tuple(Instruction("cx", (a, b)) for a, b in pairs)
    return tuple(Instruction("ccx", (a, control, b)) for a, b in pairs)


def gf2_add_constant_ir(target: tuple[int, ...], value: int) -> tuple[Instruction, ...]:
    """XOR a concrete polynomial coefficient word into a big-endian register."""
    if type(value) is not int or not 0 <= value < 1 << len(target):
        raise ValueError("Binary-field constant must fit the target register")
    if len(set(target)) != len(target):
        raise ValueError("Binary-field addition requires disjoint operands")
    return tuple(
        Instruction("x", (wire,))
        for index, wire in enumerate(target)
        if value & (1 << (len(target) - index - 1))
    )
