# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal root calls from Qualtran v0.7.0 cryptography/ecc/ec_add_r.py.

"""Compose point-table lookup, owned curve addition and explicit bank erasure."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..layout.component_cast import cast_word_function


def _addition_views(program: ProgramRecord, signature: Signature) -> ProgramRecord:
    """Retain raw Montgomery bits across nominal word connections.

    The original builder connects equal-width QUInt and QMontgomeryUInt ports.
    Native nominal typing needs explicit, zero-gate views on these five words.
    These casts do not convert a number to or from Montgomery representation.
    """
    ports = program.functions[0].registers
    assert ports is not None
    builder = ComponentProgramBuilder(signature)
    values = {}
    for port in ports:
        view = cast_word_function(
            f"input_{port.name}", signature.get_left(port.name).dtype, port.dtype
        )
        builder.define(view)
        values[port.name] = builder.call_dict(
            view.name, {"reg": builder.inputs[port.name]}
        )["reg"]
    returned = builder.call_program(program, values, entry=program.functions[0].name)
    outputs = {}
    for port in ports:
        view = cast_word_function(
            f"output_{port.name}", port.dtype, signature.get_right(port.name).dtype
        )
        builder.define(view)
        outputs[port.name] = builder.call_dict(view.name, {"reg": returned[port.name]})[
            "reg"
        ]
    return builder.finalize(outputs)


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
    dtypes = {
        name: lookup_ports.get_right(f"target{i}_").dtype
        for i, name in enumerate(names)
    }
    dtypes.update({name: signature.get_left(name).dtype for name in ("x", "y")})
    views = Signature(
        Register(port.name, dtypes[port.name])
        for port in addition.functions[0].registers or ()
    )
    result = builder.call_program(_addition_views(addition, views), values)
    for i, name in enumerate(names):
        loaded[f"target{i}_"] = result[name]
    erased = builder.call_program(erasure, loaded, entry=erasure.functions[0].name)
    selection = erased["selection"]
    assert isinstance(selection, RegisterBinding)
    return builder.finalize(
        {"ctrl": builder.split(selection), "x": result["x"], "y": result["y"]}
    )
