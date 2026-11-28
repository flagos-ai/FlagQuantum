# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_subtraction.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py

"""Compose the original controlled nonzero flag and measured apply-bit erase."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.constant_addition_component import constant_addition_program
from ..bits.fanout_component import fanout_component_program
from ..comparison._component_gates import comparison_bit_functions
from ..component_builder import ComponentProgramBuilder
from ..control.conjunction_program import and_component_program
from ..control.multi_control_component import multi_control_x_program


def controlled_modular_negation_program(
    dtype: RegisterType, modulus: int, *, control_value: int = 1
) -> ProgramRecord:
    """Negate an MSB-first residue under either source control polarity.

    The explicit zero-test arrays and untyped Joins retain source views. One
    flag spans both zero tests. The conjunction owns an apply bit until its
    measured inverse. All work is clean on the promised residue domain.
    """
    width = executable_register_width(dtype)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError("Modular negation requires unsigned or Montgomery words")
    if type(modulus) is not int or not 0 < modulus < (1 << width) - 1:
        raise ValueError("Modular negation requires modulus+1 to fit the operand")
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Modular negation control value must be zero or one")
    correction = constant_addition_program(dtype, modulus + 1, control_value=1)
    return controlled_modular_negation_component_program(
        dtype, correction, control_value=control_value
    )


def controlled_modular_negation_component_program(
    dtype: RegisterType, correction: ProgramRecord, *, control_value: int = 1
) -> ProgramRecord:
    """Compose source calls around an explicitly owned correction child.

    Unresolved corrections remain typed declarations. Their constructor owner
    supplies the true source boundary; this template infers no executable
    arithmetic or resource price from an opaque signature.
    """
    width = executable_register_width(dtype)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError("Modular negation requires unsigned or Montgomery words")
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Modular negation control value must be zero or one")
    zero = multi_control_x_program((0,) * width)
    compute = and_component_program((control_value, 1), name="mod_neg_and")
    erase = and_component_program(
        (control_value, 1), name="mod_neg_erase", uncompute=True
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=quantum_bit_type(), x=dtype),
        comparison_bit_functions(),
    )
    flag = builder.allocate()
    active = builder.call_dict("comparison_x", {"q": flag})["q"]
    x = builder.inputs["x"]
    assert isinstance(x, RegisterBinding)
    tested = builder.call_program(
        zero, {"controls": builder.split(x), "target": active}
    )
    x = builder.join(tested["controls"])
    result = builder.call_program(
        compute,
        {"ctrl": np.array([builder.inputs["ctrl"], tested["target"]])},
        entry="mod_neg_and",
    )
    spread = builder.call_program(
        fanout_component_program(width), {"control": result["target"], "targets": x}
    )
    added = builder.call_program(
        correction,
        {"ctrl": spread["control"], "x": spread["targets"]},
        entry=correction.functions[0].name,
    )
    restored = builder.call_program(
        erase, {"ctrl": result["ctrl"], "target": added["ctrl"]}, entry="mod_neg_erase"
    )
    ctrl, active = np.asarray(restored["ctrl"])
    x_word = added["x"]
    assert isinstance(x_word, RegisterBinding)
    tested = builder.call_program(
        zero, {"controls": builder.split(x_word), "target": active}
    )
    x = builder.join(tested["controls"])
    cleared = builder.call_dict("comparison_x", {"q": tested["target"]})["q"]
    assert isinstance(cleared, RegisterBinding)
    builder.free(cleared)
    return builder.finalize({"ctrl": ctrl, "x": x})
