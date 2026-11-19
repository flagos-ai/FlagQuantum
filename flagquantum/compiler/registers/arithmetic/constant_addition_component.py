# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 AddK.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Compose an encoded constant load, unconditional Add and exact unload."""

from functools import partial

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component import bitwise_word_program
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import (
    control_component_program,
    control_component_via_predicate,
)
from ..control.component_bookkeeping import call_with_unchanged_controls
from ..control.control_predicate import ControlRegister
from ..layout.component_workspace import allocate_word_function, free_word_function
from ..primitives.always_component import always_component_program
from .addition_component import addition_component_program


def _unconditional(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
    *,
    callee: str,
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    controls, outputs = call_with_unchanged_controls(
        builder, callee, (control,), values
    )
    return controls[0], outputs


def _controlled_load(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
    *,
    record: ProgramRecord,
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_program(record, {"ctrl": control, **values})
    return outputs["ctrl"], (outputs["x"],)


def constant_addition_program(
    dtype: RegisterType,
    pattern: int,
    *,
    adjoint: bool = False,
    control_value: int | None = None,
) -> ProgramRecord:
    """Add an encoded unsigned pattern into a through word, MSB first.

    Caller-owned value encoding supplies the pattern for signed constants.
    The dtype is a concrete source integer descriptor. Loaded constant work
    and Add carries start and end clean. The adjoint retains the original
    load/Add-adjoint/unload path. Optional binary control affects only loading,
    while Always(Add) executes in both sectors. Invalid inputs raise ValueError.
    """
    if type(adjoint) is not bool:
        raise ValueError("Constant addition direction requires a boolean")
    if control_value is not None and (
        type(control_value) is not int or control_value not in (0, 1)
    ):
        raise ValueError("Constant addition control value must be zero, one or None")
    width = executable_register_width(dtype)
    addition = addition_component_program(
        width, name="constant_add", left_dtype=dtype, right_dtype=dtype, adjoint=adjoint
    )
    load = bitwise_word_program(dtype, "xor_constant", pattern=pattern)
    always = always_component_program(addition, "constant_add", "constant_always")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype),
        (
            allocate_word_function("constant_allocate", dtype, False),
            free_word_function("constant_free", dtype, False),
        ),
    )
    loaded = builder.call_dict("constant_allocate", {})["reg"]
    loaded = builder.call_program(load, {"x": loaded})["x"]
    result = builder.call_program(
        always, {"a": loaded, "b": builder.inputs["x"]}, entry="constant_always"
    )
    loaded = builder.call_program(load, {"x": result["a"]})["x"]
    assert isinstance(loaded, RegisterBinding)
    builder.call_dict("constant_free", {"reg": loaded})
    record = builder.finalize({"x": result["b"]})
    if control_value is None:
        return record
    operations = record.functions[0].blocks[0].operations
    load_name = operations[1].callee
    assert load_name is not None
    controlled = controlled_bitwise_word_program(dtype, "xor_constant", pattern=pattern)
    adders = {
        name: partial(_unconditional, callee=name)
        for name in ("constant_allocate", "constant_free", "constant_always")
    }
    adders[load_name] = partial(_controlled_load, record=controlled)
    if control_value == 1:
        return control_component_program(record, adders)
    return control_component_via_predicate(
        record,
        adders,
        (Register("ctrl", quantum_bit_type()),),
        (ControlRegister(1, (control_value,)),),
    )
