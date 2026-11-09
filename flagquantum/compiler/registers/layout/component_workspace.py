# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Workspace signatures adapted from Qualtran v0.7.0 Allocate and Free.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping

"""Real typed lifetime functions for finite register construction."""

from ....core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType


def allocate_word_function(
    name: str, dtype: RegisterType, dirty: bool
) -> FunctionRecord:
    """Define one fresh owned word, preserving the explicit borrowing flag."""
    executable_register_width(dtype)
    if type(dirty) is not bool:
        raise ValueError("workspace dirty flag must be a built-in boolean")
    word = ValueRef("entry", 0, "register")
    operation = Operation(
        "allocate_register", results=(word,), register_type=dtype, workspace_dirty=dirty
    )
    return FunctionRecord(
        name,
        (BlockRecord("entry", operations=(operation,), return_values=(word,)),),
        registers=(Register("reg", dtype, side=RegisterSide.RIGHT),),
    )


def free_word_function(name: str, dtype: RegisterType, dirty: bool) -> FunctionRecord:
    """Consume one owned word without claiming numerical restoration."""
    executable_register_width(dtype)
    if type(dirty) is not bool:
        raise ValueError("workspace dirty flag must be a built-in boolean")
    word = ValueRef("entry", 0, "register")
    operation = Operation("free_register", operands=(word,), workspace_dirty=dirty)
    return FunctionRecord(
        name,
        (BlockRecord("entry", (word,), operations=(operation,), return_values=()),),
        registers=(Register("reg", dtype, side=RegisterSide.LEFT),),
    )
