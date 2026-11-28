# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 KaliskiModInverse.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Inline the source inverse algorithm behind its owned history interface."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import montgomery_register_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_builder import ComponentProgramBuilder
from .inversion.inverse_algorithm_component import inverse_algorithm_program


def modular_inverse_program(
    width: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Transform x with a RIGHT forward or LEFT inverse 4n-bit history word.

    x and history are MSB-first. Forward accepts zero or units modulo p and
    retains the original marker/terminal histories. Inverse consumes their
    exact forward image. Temporary words clear only on that promised domain.
    """
    algorithm = inverse_algorithm_program(width, modulus, adjoint=adjoint)
    return modular_inverse_component_program(width, modulus, algorithm, adjoint=adjoint)


def modular_inverse_component_program(
    width: int, modulus: SymbolicInt, algorithm: ProgramRecord, *, adjoint: bool = False
) -> ProgramRecord:
    """Keep directional history ports and inline the explicit source algorithm.

    Histories belong to the caller. Private words retain exact source releases;
    an unavailable descendant never grants a physical cleanup or price claim.
    """
    word = montgomery_register_type(width, modulus)
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", word),
                Register("junk", RegisterType("opaque", 4 * width), side=side),
            )
        )
    )
    u = builder.allocate(dtype=montgomery_register_type(width))
    r = builder.allocate(dtype=montgomery_register_type(width))
    s = builder.allocate(dtype=montgomery_register_type(width))
    f = builder.allocate()
    if adjoint:
        junk = builder.inputs["junk"]
        assert isinstance(junk, RegisterBinding)
        pieces = builder.split(junk)
        m, terminal = builder.join(pieces[: 2 * width]), builder.join(
            pieces[2 * width :]
        )
    else:
        m, terminal = builder.allocate(2 * width), builder.allocate(2 * width)
    values = builder.inline_calls(
        algorithm,
        {
            "u": u,
            "v": r if adjoint else builder.inputs["x"],
            "r": builder.inputs["x"] if adjoint else r,
            "s": s,
            "m": m,
            "f": f,
            "terminal_condition": terminal,
        },
    )
    output = values.pop("v" if adjoint else "r")
    releases = (
        ("u", "r", "s", "m", "f", "terminal_condition")
        if adjoint
        else ("u", "v", "s", "f")
    )
    for name in releases:
        value = values.pop(name)
        assert isinstance(value, RegisterBinding)
        builder.free(value)
    if adjoint:
        return builder.finalize({"x": output})
    markers, terminals = values["m"], values["terminal_condition"]
    assert isinstance(markers, RegisterBinding) and isinstance(
        terminals, RegisterBinding
    )
    junk = builder.join([*builder.split(markers), *builder.split(terminals)])
    return builder.finalize({"x": output, "junk": junk})
