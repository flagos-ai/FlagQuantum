# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OnEach.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/on_each.py

"""Build the owned register body for a repeated single-bit component."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def _validate_on_each_child(child: FunctionRecord) -> None:
    registers = child.registers
    if (
        registers is None
        or len(registers) != 1
        or registers[0].name != "q"
        or registers[0].shape_symbolic
        or registers[0].side != RegisterSide.THRU
    ):
        raise ValueError("OnEach requires one scalar quantum THRU port named q")
    if executable_register_width(registers[0].dtype) != 1:
        raise ValueError("OnEach child requires exactly one physical quantum bit")


def on_each_word_program(
    width: int, child: FunctionRecord, *, target_dtype: RegisterType | None = None
) -> ProgramRecord:
    """Split a concrete word, call a self-contained one-bit child in order, and join.

    The child has a single through-going port named q. The returned word uses
    the input dtype. Physical binding and any work required by the child are
    handled by the existing Program binder.
    """
    _validate_on_each_child(child)
    if type(width) is not int or width < 0:
        raise ValueError("OnEach requires a concrete nonnegative integer width")
    dtype = opaque_register_type(width) if target_dtype is None else target_dtype
    if dtype.num_qubits != width or dtype.num_cbits != 0:
        raise ValueError("OnEach target dtype must match the quantum word width")

    builder = ComponentProgramBuilder(Signature((Register("q", dtype),)), (child,))
    target = builder.inputs["q"]
    assert isinstance(target, RegisterBinding)
    bits = builder.split(target)
    for i, binding in enumerate(bits):
        bits[i] = builder.call_dict(child.name, {"q": binding})["q"]
    return builder.finalize({"q": builder.join(bits, dtype)})
