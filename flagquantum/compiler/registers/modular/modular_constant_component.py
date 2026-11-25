# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_addition.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py

"""Own the source constant ket until its modular-add effect releases it."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_basis_vectors import integer_basis_function
from ..component_builder import ComponentProgramBuilder
from .controlled_modular_addition_component import controlled_modular_addition_program


def controlled_modular_constant_program(
    width: int, modulus: int, value: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Add an encoded constant under positive control, retaining its exact load.

    The source constant is not reduced modulo modulus. Its unsigned width must
    fit. The source residue-domain and clean-effect promises apply even for an
    inactive control. All word ports are MSB first.
    The adjoint retains the original constant ket/effect and exact add inverse.
    """
    if type(adjoint) is not bool:
        raise ValueError("Modular constant direction requires a boolean")
    word = unsigned_register_type(width)
    addition = controlled_modular_addition_program(word, modulus)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=quantum_bit_type(), x=word),
        (
            integer_basis_function("mod_constant_state", word, value),
            integer_basis_function("mod_constant_effect", word, value, state=False),
        ),
    )
    constant = builder.call_dict("mod_constant_state", {})["val"]
    added = builder.call_program(
        addition,
        {"ctrl": builder.inputs["ctrl"], "x": constant, "y": builder.inputs["x"]},
    )
    builder.call_dict("mod_constant_effect", {"val": added["x"]})
    record = builder.finalize({"ctrl": added["ctrl"], "x": added["y"]})
    if not adjoint:
        return record
    state = ProgramRecord((integer_basis_function("mod_constant_state", word, value),))
    effect = ProgramRecord(
        (integer_basis_function("mod_constant_effect", word, value, state=False),)
    )
    return adjoint_component_calls(
        record,
        (
            (state, effect),
            (effect, state),
            (
                addition,
                controlled_modular_addition_program(word, modulus, adjoint=True),
            ),
        ),
    )
