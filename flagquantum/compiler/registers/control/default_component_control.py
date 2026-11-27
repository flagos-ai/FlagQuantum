# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_ctrl_system_with_correct_metabloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Dispatch default controls to the existing typed component owners."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import classical_bit_type, quantum_bit_type
from ....core.ir.program.registers import Register
from .component_control_selection import ComponentControlSystem
from .component_extension import StructuredControlAdder
from .control_predicate import ControlRegister


def default_component_control_system(
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...] | None,
    total: ComponentControlSystem,
    predicate: ComponentControlSystem,
    classical: ComponentControlSystem,
) -> tuple[ProgramRecord, StructuredControlAdder]:
    """Select the locked default implementation and return its actual callback.

    Only a scalar positive QBit selects total control. Other quantum descriptors
    select the shared predicate owner, including one-bit words and singleton
    arrays. Classical storage selects the feedback owner. Storage markers retain
    zero-width classical identity. Mixed storage and empty specifications fail.

    Owners receive the original descriptors and retain admission, child-body,
    workspace and execution responsibilities. Their failures propagate. Dispatch
    itself does not admit symbolic or additional physical register encodings.
    """
    if patterns is not None and len(controls) != len(patterns):
        raise ValueError("Control registers require one activation pattern each")
    if (
        len(controls) == 1
        and controls[0].dtype == quantum_bit_type()
        and controls[0].shape_symbolic == ()
        and patterns == (ControlRegister(1, (1,)),)
    ):
        return total(controls, patterns)

    storage = {
        register.dtype == classical_bit_type()
        or ("storage", "classical") in register.dtype.parameters
        for register in controls
    }
    if len(storage) > 1:
        raise NotImplementedError(
            "Mixed quantum and classical controls are unsupported"
        )
    if storage == {False}:
        return predicate(controls, patterns)
    if storage == {True}:
        return classical(controls, patterns)
    raise ValueError("A control specification requires at least one register")
