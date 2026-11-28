# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_subtraction.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py

"""Compose the source nonzero flag, fanout and constant modular correction."""

from dataclasses import replace

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import unsigned_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.constant_addition_component import constant_addition_program
from ..bits.fanout_component import fanout_component_program
from ..comparison._component_gates import comparison_bit_functions
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import control_component_via_predicate
from ..control.control_predicate import ControlRegister


def _zero_word_flip_program(dtype: RegisterType) -> ProgramRecord:
    width = executable_register_width(dtype)
    gates = {f.name: f for f in comparison_bit_functions()}
    base = replace(gates["comparison_x"], name="zero_base")
    builder = ComponentProgramBuilder(
        Signature(base.registers or ()), (base, gates["comparison_cx"])
    )
    record = builder.finalize(builder.call_dict(base.name, builder.inputs))

    def positive(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        result = builder.call_dict(
            "comparison_cx", {"ctrl": control, "target": values["q"]}
        )
        return result["ctrl"], (result["target"],)

    result = control_component_via_predicate(
        record,
        {"main": positive},
        (Register("ctrl", dtype),),
        (ControlRegister(width, (0,)),),
    )
    return replace(
        result, functions=tuple(f for f in result.functions if f.name != base.name)
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
    return modular_negation_component_program(dtype, correction)


def modular_negation_component_program(
    dtype: RegisterType,
    correction: ProgramRecord,
) -> ProgramRecord:
    """Retain the eight source calls with explicit constant/control owners.

    A logical constant child may truthfully lack a source decomposition. The
    constructor owns its actual modulus-plus-one identity; this template infers
    no arithmetic action or price from an opaque signature. The shared predicate
    owner preserves the actual source control graph.
    """
    width = executable_register_width(dtype)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError("Modular negation requires unsigned or Montgomery words")
    zero = _zero_word_flip_program(dtype)
    fanout = fanout_component_program(width)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype), comparison_bit_functions()
    )
    flag = builder.allocate()
    active = builder.call_dict("comparison_x", {"q": flag})["q"]
    tested = builder.call_program(
        zero,
        {"ctrl": builder.inputs["x"], "q": active},
        entry=zero.functions[0].name,
    )
    spread = builder.call_program(
        fanout, {"control": tested["q"], "targets": tested["ctrl"]}
    )
    added = builder.call_program(
        correction,
        {"ctrl": spread["control"], "x": spread["targets"]},
        entry=correction.functions[0].name,
    )
    tested = builder.call_program(
        zero,
        {"ctrl": added["x"], "q": added["ctrl"]},
        entry=zero.functions[0].name,
    )
    cleared = builder.call_dict("comparison_x", {"q": tested["q"]})["q"]
    assert isinstance(cleared, RegisterBinding)
    builder.free(cleared)
    return builder.finalize({"x": tested["ctrl"]})
