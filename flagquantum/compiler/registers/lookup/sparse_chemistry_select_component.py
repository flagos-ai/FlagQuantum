# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 chemistry/sparse/select_bloq.py.

"""Compose the five source sparse SELECT calls with exact Majorana children."""

from typing import Literal

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .majorana_component import majorana_program

SPARSE_SELECT_STAGES: tuple[tuple[str, str, Literal["X", "Y"], bool], ...] = (
    ("p", "alpha", "Y", True),
    ("q", "alpha", "X", True),
    ("r", "beta", "Y", False),
    ("s", "beta", "X", False),
)


def sparse_select_signature(
    num_spin_orb: int, *, control_value: int | None = None
) -> Signature:
    """Keep the existing source metadata, including its unused outer control."""
    if type(num_spin_orb) is not int or num_spin_orb < 2 or num_spin_orb % 2:
        raise ValueError(
            "Sparse SELECT requires a positive even number of spin orbitals"
        )
    if control_value is not None and (
        type(control_value) is not int or control_value not in (0, 1)
    ):
        raise ValueError("Sparse SELECT control value must be zero, one or None")
    spatial = num_spin_orb // 2
    orbital = bounded_unsigned_register_type((spatial - 1).bit_length(), spatial)
    bit = bounded_unsigned_register_type(1)
    return Signature(
        (
            *(
                (Register("control", quantum_bit_type()),)
                if control_value is not None
                else ()
            ),
            *(Register(name, orbital) for name in ("p", "q", "r", "s")),
            *(Register(name, bit) for name in ("alpha", "beta", "flag_1b")),
            Register("sys", opaque_register_type(num_spin_orb)),
        )
    )


def sparse_select_component_pair(
    num_spin_orb: int, *, control_value: int | None = None
) -> tuple[ProgramRecord, ProgramRecord]:
    """Preserve S, flagged Y/X, unconditional Y/X and the source control defect.

    All named words are caller-owned MSB storage. Shared Majorana owners clear
    their private accumulators/predicates using measured erasure. The external
    control is a source bystander for either polarity, not a corrected control.
    The source's zero-width spatial axis is not an executable typed recipe.
    """
    signature = sparse_select_signature(num_spin_orb, control_value=control_value)
    if num_spin_orb == 2:
        raise ValueError(
            "Source Majorana zero-width spatial axis has no supported typed decomposition"
        )
    bit = quantum_bit_type()
    phase = ProgramRecord(
        (
            bit_gate_function(
                "sparse_s",
                (Register("q", bit),),
                CircuitIR(1, (Instruction("s", (0,)),)),
            ),
        )
    )
    inverse_phase = ProgramRecord(
        (
            bit_gate_function(
                "sparse_s_inverse",
                (Register("q", bit),),
                CircuitIR(1, (Instruction("sdg", (0,)),)),
            ),
        )
    )
    pairs = [(phase, inverse_phase)]
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    state["flag_1b"] = builder.call_program(
        phase, {"q": state.pop("flag_1b")}, entry=phase.functions[0].name
    )["q"]
    for orbital, spin, pauli, flagged in SPARSE_SELECT_STAGES:
        selection = (signature.get_left(orbital), signature.get_left(spin))
        control = Register("control", bit) if flagged else None
        pair = tuple(
            majorana_program(selection, pauli, control=control, adjoint=inverse)
            for inverse in (False, True)
        )
        pairs.append((pair[0], pair[1]))
        inputs = {
            orbital: state.pop(orbital),
            spin: state.pop(spin),
            "target": state.pop("sys"),
        }
        if flagged:
            inputs["control"] = state.pop("flag_1b")
        returned = builder.call_program(pair[0], inputs)
        state["sys"] = returned.pop("target")
        if flagged:
            state["flag_1b"] = returned.pop("control")
        state.update(returned)
    forward = builder.finalize(state)
    return forward, adjoint_component_calls(forward, tuple(pairs))
