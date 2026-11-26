# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal root calls from Qualtran v0.7.0 cryptography/ecc/ec_add_r.py.

"""Compose point-table lookup, owned curve addition and explicit bank erasure."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def windowed_curve_addition_program(
    signature: Signature,
    *,
    lookup: ProgramRecord,
    addition: ProgramRecord,
    erasure: ProgramRecord,
) -> ProgramRecord:
    """Preserve all three optional junk banks until the matching erase call.

    Address bits are MSB-first. Coordinates contain raw Montgomery encodings.
    Lookup RIGHT words become private to this root and erasure consumes them;
    addition owns its arithmetic scratch. The supplied erasure may be a source
    model-only declaration or an explicitly chosen native measured recipe.
    """
    builder = ComponentProgramBuilder(signature)
    lookup_ports = Signature(lookup.functions[0].registers or ())
    ctrl = builder.join(
        builder.inputs["ctrl"],
        lookup_ports.get_left("selection").dtype,
    )
    loaded = builder.call_program(
        lookup, {"selection": ctrl}, entry=lookup.functions[0].name
    )
    names = ("a", "b", "lam_r")
    values = {name: loaded[f"target{i}_"] for i, name in enumerate(names)}
    values.update(x=builder.inputs["x"], y=builder.inputs["y"])
    result = builder.call_program(addition, values, entry=addition.functions[0].name)
    for i, name in enumerate(names):
        loaded[f"target{i}_"] = result[name]
    erased = builder.call_program(erasure, loaded, entry=erasure.functions[0].name)
    selection = erased["selection"]
    assert isinstance(selection, RegisterBinding)
    return builder.finalize(
        {"ctrl": builder.split(selection), "x": result["x"], "y": result["y"]}
    )
