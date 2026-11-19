# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Negate.build_composite_bloq and Controlled.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/negate.py

"""Compose typed complement and constant increment with literal control paths."""

from functools import partial

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component import bitwise_word_program
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import (
    ControlAdder,
    control_component_program,
    control_component_via_predicate,
)
from ..control.control_predicate import ControlRegister
from .constant_addition_component import constant_addition_program


def _controlled_word(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
    *,
    record: ProgramRecord,
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_program(record, {"ctrl": control, **values})
    return outputs["ctrl"], (outputs["x"],)


def negation_program(
    dtype: RegisterType, *, adjoint: bool = False, control_value: int | None = None
) -> ProgramRecord:
    """Negate a through integer word, MSB first, retaining source component calls.

    The exact adjoint applies AddK-adjoint before the complement. Optional
    binary control acts on complement and constant loads, leaving Always(Add)
    unconditional. AddK owns its clean constant/carry work and measured cleanup.
    One-bit signed execution uses the encoded increment bit despite the source
    signed +1 value-encoding failure. Invalid descriptors/flags raise ValueError.
    """
    increment = constant_addition_program(dtype, 1, adjoint=adjoint, control_value=None)
    if control_value is not None and (
        type(control_value) is not int or control_value not in (0, 1)
    ):
        raise ValueError("Negation control value must be zero, one or None")
    complement = bitwise_word_program(dtype, "bitwise_not")
    children = (increment, complement) if adjoint else (complement, increment)
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(x=dtype))
    value = builder.inputs["x"]
    for child in children:
        value = builder.call_program(child, {"x": value})["x"]
    record = builder.finalize({"x": value})
    if control_value is None:
        return record
    adders = _control_adders(record, dtype, adjoint)
    if control_value == 1:
        return control_component_program(record, adders)
    return control_component_via_predicate(
        record,
        adders,
        (Register("ctrl", quantum_bit_type()),),
        (ControlRegister(1, (0,)),),
    )


def _control_adders(
    record: ProgramRecord, dtype: RegisterType, adjoint: bool
) -> dict[str, ControlAdder]:
    controlled_increment = constant_addition_program(
        dtype, 1, adjoint=adjoint, control_value=1
    )
    controlled_complement = controlled_bitwise_word_program(dtype, "bitwise_not")
    controlled_children = (
        (controlled_increment, controlled_complement)
        if adjoint
        else (controlled_complement, controlled_increment)
    )
    adders: dict[str, ControlAdder] = {}
    for operation, controlled in zip(
        record.functions[0].blocks[0].operations, controlled_children, strict=True
    ):
        assert operation.callee is not None
        adders[operation.callee] = partial(_controlled_word, record=controlled)
    return adders


def controlled_negation_program(
    dtype: RegisterType,
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...] | None,
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Negate under existing row-major register predicates and owned erasure.

    Controls retain their descriptors and shapes. Predicate conjunction work
    and AddK work have distinct clean lifetimes and measurements. Symbolic data
    follow the existing physical admission boundary. Unknown activation retains
    the shared source HasLength treatment: two bits leave an unresolved
    conjunction, while larger concrete conjunctions resolve to all ones.
    The supplied controls must use the source-compatible names for graph parity.
    """
    record = negation_program(dtype, adjoint=adjoint)
    return control_component_via_predicate(
        record,
        _control_adders(record, dtype, adjoint),
        controls,
        patterns,
    )
