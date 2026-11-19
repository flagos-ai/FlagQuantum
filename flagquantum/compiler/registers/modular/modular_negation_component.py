# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_subtraction.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py

"""Compose the source nonzero flag, fanout and constant modular correction."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.constant_addition_component import constant_addition_program
from ..bits.fanout_component import fanout_component_program
from ..comparison._component_gates import comparison_bit_functions
from ..component_builder import ComponentProgramBuilder
from ..control.multi_control_component import multi_control_x_program


def _zero_word_flip_program(dtype: RegisterType) -> ProgramRecord:
    width = executable_register_width(dtype)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=dtype, q=quantum_bit_type())
    )
    control = builder.inputs["ctrl"]
    assert isinstance(control, RegisterBinding)
    result = builder.call_program(
        multi_control_x_program((0,) * width),
        {"controls": builder.split(control), "target": builder.inputs["q"]},
    )
    return builder.finalize(
        {"ctrl": builder.join(result["controls"], dtype), "q": result["target"]}
    )


def modular_negation_program(dtype: RegisterType, modulus: int) -> ProgramRecord:
    """Negate one MSB-first residue word with the original clean-flag promise.

    The root retains uint or Montgomery ports. Original generic zero controls
    bind a whole typed word. Fanout and AddK keep their source word views.
    The source cannot encode an all-ones modulus correction. Inputs outside
    the promised residue domain may retain a dirty flag at release.
    """
    width = executable_register_width(dtype)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError("Modular negation requires unsigned or Montgomery words")
    if type(modulus) is not int or not 0 < modulus < (1 << width) - 1:
        raise ValueError("Modular negation requires modulus+1 to fit the operand")
    correction = constant_addition_program(
        unsigned_register_type(width), modulus + 1, control_value=1
    )
    zero = _zero_word_flip_program(dtype)
    fanout = fanout_component_program(width)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype), comparison_bit_functions()
    )
    flag = builder.allocate()
    active = builder.call_dict("comparison_x", {"q": flag})["q"]
    tested = builder.call_program(zero, {"ctrl": builder.inputs["x"], "q": active})
    spread = builder.call_program(
        fanout, {"control": tested["q"], "targets": tested["ctrl"]}
    )
    added = builder.call_program(
        correction, {"ctrl": spread["control"], "x": spread["targets"]}
    )
    tested = builder.call_program(zero, {"ctrl": added["x"], "q": added["ctrl"]})
    cleared = builder.call_dict("comparison_x", {"q": tested["q"]})["q"]
    assert isinstance(cleared, RegisterBinding)
    builder.free(cleared)
    return builder.finalize({"x": tested["ctrl"]})
