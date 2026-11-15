# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OnEach.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/on_each.py

"""Build the owned register body for a repeated single-bit component."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def on_each_word_program(
    width: int, child: FunctionRecord, *, target_dtype: RegisterType | None = None
) -> ProgramRecord:
    """Split a concrete word, call a self-contained QBit child in order, and join.

    The child has a single through-going port named q. The returned word uses
    the input dtype. Physical binding and any work required by the child are
    handled by the existing Program binder.
    """
    bit = Register("q", quantum_bit_type())
    if child.registers != (bit,):
        raise ValueError("OnEach requires a single scalar QBit THRU port named q")
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
