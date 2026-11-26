# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_add.py, _ECAddStepTwo.

"""Thread slope selection and consume the original arithmetic histories."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder


def addition_slope_program(
    signature: Signature,
    *,
    zero: ProgramRecord,
    subtract: ProgramRecord,
    controlled_subtract: ProgramRecord,
    inverse: ProgramRecord,
    multiply: ProgramRecord,
    equals: ProgramRecord,
    select_general: ProgramRecord,
    select_tangent: ProgramRecord,
    controlled_equals: ProgramRecord,
    multiply_adjoint: ProgramRecord,
    inverse_adjoint: ProgramRecord,
) -> ProgramRecord:
    """Preserve the source rare-flag behavior and privately erase both histories.

    Inputs are MSB-first words. The new logical lam belongs to the caller;
    the equality bit, inverse history and multiplication history are private.
    Exact adjoints must consume the corresponding forward outputs.
    """
    width = executable_register_width(signature.get_left("a").dtype)
    builder = ComponentProgramBuilder(signature)
    (
        zero_entry,
        subtract_entry,
        csub_entry,
        inverse_entry,
        multiply_entry,
        equals_entry,
        general_entry,
        tangent_entry,
        cequals_entry,
        multiply_adjoint_entry,
        inverse_adjoint_entry,
    ) = (
        builder.import_program(child, entry=child.functions[0].name)
        for child in (
            zero,
            subtract,
            controlled_subtract,
            inverse,
            multiply,
            equals,
            select_general,
            select_tangent,
            controlled_equals,
            multiply_adjoint,
            inverse_adjoint,
        )
    )
    f1, ctrl, a, b, x, y, lam_r = (
        builder.inputs[name] for name in ("f1", "ctrl", "a", "b", "x", "y", "lam_r")
    )
    lam = builder.call_dict(zero_entry, {})["val"]
    a, x = builder.call_tuple(subtract_entry, {"x": a, "y": x})
    ctrl, b, y = builder.call_tuple(csub_entry, {"ctrl": ctrl, "x": b, "y": y})
    x, junk = builder.call_tuple(inverse_entry, {"x": x})
    x, y, z4, z3, reduced = builder.call_tuple(multiply_entry, {"x": x, "y": y})
    ancilla: BindingCollection = builder.allocate(dtype=quantum_bit_type())
    z4, lam_r, ancilla = builder.call_tuple(
        equals_entry, {"x": z4, "y": lam_r, "target": ancilla}
    )
    assert isinstance(z4, RegisterBinding) and isinstance(lam, RegisterBinding)
    z4_bits, lam_bits = builder.split(z4), builder.split(lam)
    for i in range(width):
        controls, lam_bits[i] = builder.call_tuple(
            general_entry,
            {"controls": np.array([f1, ctrl, z4_bits[i]]), "target": lam_bits[i]},
        )
        assert isinstance(controls, np.ndarray)
        f1, ctrl, z4_bits[i] = controls
    z4 = builder.join(z4_bits, signature.get_right("lam").dtype)
    assert isinstance(lam_r, RegisterBinding)
    tangent_bits = builder.split(lam_r)
    for i in range(width):
        controls, lam_bits[i] = builder.call_tuple(
            tangent_entry,
            {"controls": np.array([f1, ctrl, tangent_bits[i]]), "target": lam_bits[i]},
        )
        assert isinstance(controls, np.ndarray)
        f1, ctrl, tangent_bits[i] = controls
    lam_r = builder.join(tangent_bits, signature.get_left("lam_r").dtype)
    lam = builder.join(lam_bits, signature.get_right("lam").dtype)
    ancilla, lam, lam_r, f1 = builder.call_tuple(
        cequals_entry, {"ctrl": ancilla, "x": lam, "y": lam_r, "target": f1}
    )
    z4, lam_r, ancilla = builder.call_tuple(
        equals_entry, {"x": z4, "y": lam_r, "target": ancilla}
    )
    assert isinstance(ancilla, RegisterBinding)
    builder.free(ancilla)
    x, y = builder.call_tuple(
        multiply_adjoint_entry,
        {"x": x, "y": y, "target": z4, "qrom_indices": z3, "reduced": reduced},
    )
    x = builder.call_dict(inverse_adjoint_entry, {"x": x, "junk": junk})["x"]
    return builder.finalize(
        {
            "f1": f1,
            "ctrl": ctrl,
            "a": a,
            "b": b,
            "x": x,
            "y": y,
            "lam": lam,
            "lam_r": lam_r,
        }
    )
