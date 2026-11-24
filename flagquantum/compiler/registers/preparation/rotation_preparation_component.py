# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Typed stages follow Qualtran v0.7.0 StatePreparationViaRotations.

"""Compose source amplitude and phase tables on a caller-owned gradient."""

from math import pi

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..phase.phase_gradient_lookup import phase_gradient_lookup_table
from ..phase.phase_gradient_lookup_component import phase_gradient_lookup_program


def rotation_preparation_program(
    rows: tuple[tuple[int, ...], ...],
    phases: tuple[int, ...],
    phase_width: int,
    *,
    control_count: int = 0,
    uncompute: bool = False,
    adjoint: bool = False,
) -> ProgramRecord:
    """Keep source Rx signs, prefix/control joins and independent exact Adjoint.

    Source uncompute selects table values and stage order before this call;
    adjoint instead reverses the resulting graph using explicit exact children.
    Logical words are MSB-first. Only PRGA angle/carry and phase-flag work is
    private. No gradient state is prepared or replaced by this transformation.
    """
    if not rows:
        raise ValueError("Original singleton rotation preparation has no target body")
    if (
        type(control_count) is not int
        or control_count < 0
        or type(uncompute) is not bool
        or type(adjoint) is not bool
    ):
        raise ValueError("Invalid rotation preparation controls or direction")
    signature = Signature.build(
        prepare_control=control_count,
        target_state=len(rows),
        phase_gradient=phase_width,
    )
    bit = quantum_bit_type()
    rx = tuple(
        ProgramRecord(
            (
                bit_gate_function(
                    "main",
                    (Register("q", bit),),
                    CircuitIR(1, (Instruction("rx", (0,), {"theta": sign * pi / 2}),)),
                ),
            )
        )
        for sign in (1, -1)
    )
    x = ProgramRecord(
        (
            bit_gate_function(
                "main", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
            ),
        )
    )
    builder = ComponentProgramBuilder(signature)
    words = dict(builder.inputs)
    pairs = [(rx[0], rx[1]), (rx[1], rx[0]), (x, x)]

    def lookup(selection_width: int, values: tuple[int, ...]) -> None:
        table = phase_gradient_lookup_table(
            control_count + 1, selection_width, phase_width, values
        )
        forward = phase_gradient_lookup_program(table)
        if adjoint:
            pairs.append((forward, phase_gradient_lookup_program(table, adjoint=True)))
        result = builder.call_program(forward, words)
        words.clear()
        words.update(result)

    def combine(flag: RegisterBinding) -> None:
        if control_count:
            control = words.pop("prepare_control")
            assert isinstance(control, RegisterBinding)
            bits = list(builder.split(control)) if control_count > 1 else [control]
            words["control"] = builder.join([*bits, flag])
        else:
            words["control"] = flag

    def separate() -> RegisterBinding:
        control = words.pop("control")
        assert isinstance(control, RegisterBinding)
        separated = builder.split(control)
        if control_count:
            words["prepare_control"] = builder.join(separated[:-1])
        flag = separated[-1]
        assert isinstance(flag, RegisterBinding)
        return flag

    for stage in (("phases", "amplitudes") if uncompute else ("amplitudes", "phases")):
        if stage == "amplitudes":
            target = words.pop("target_state")
            assert isinstance(target, RegisterBinding)
            bits = builder.split(target)
            levels = reversed(range(len(rows))) if uncompute else range(len(rows))
            for level in levels:
                flag = builder.call_program(rx[0], {"q": bits[level]})["q"]
                assert isinstance(flag, RegisterBinding)
                if level:
                    words["selection"] = builder.join(bits[:level])
                combine(flag)
                lookup(level, rows[level])
                bits[level] = separate()
                if level:
                    prefix = words.pop("selection")
                    assert isinstance(prefix, RegisterBinding)
                    bits[:level] = builder.split(prefix)
                bits[level] = builder.call_program(rx[1], {"q": bits[level]})["q"]
            words["target_state"] = builder.join(bits)
        else:
            flag = builder.allocate()
            changed = builder.call_program(x, {"q": flag})["q"]
            assert isinstance(changed, RegisterBinding)
            combine(changed)
            words["selection"] = words.pop("target_state")
            lookup(len(rows), phases)
            words["target_state"] = words.pop("selection")
            changed = builder.call_program(x, {"q": separate()})["q"]
            assert isinstance(changed, RegisterBinding)
            builder.free(changed)
    record = builder.finalize(words)
    return adjoint_component_calls(record, tuple(pairs)) if adjoint else record
