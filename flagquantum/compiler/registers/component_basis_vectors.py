# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _IntVector state/effect decomposition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py

"""Construct typed integer basis preparation and checked-release functions."""

from ...core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from ...core.ir.program.register_verifier import executable_register_width
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from .layout.int_vector import int_state_ir


def integer_basis_function_data(function: FunctionRecord) -> tuple[int, bool] | None:
    """Read a canonical integer ket/effect without interpreting function names.

    Return (unsigned value, is_state) only for the exact definition produced by
    integer_basis_function. The sole scalar val port, clean lifetime, bit order,
    X operands, SSA returns and all function fields must match that constructor.
    Other definitions, including equivalent noncanonical SSA, return None.
    This Compiler query creates no numerical tensor and does not execute code.
    """
    if function.registers is None or len(function.registers) != 1:
        return None
    register = function.registers[0]
    if (
        register.name != "val"
        or register.shape_symbolic
        or register.side not in (RegisterSide.LEFT, RegisterSide.RIGHT)
        or len(function.blocks) != 1
    ):
        return None
    try:
        width = executable_register_width(register.dtype)
    except ValueError:
        return None
    splits = [
        operation
        for operation in function.blocks[0].operations
        if operation.opcode == "split_register"
    ]
    if len(splits) != 1 or len(splits[0].results) != width:
        return None
    positions = {bit: index for index, bit in enumerate(splits[0].results)}
    value = 0
    for operation in function.blocks[0].operations:
        if operation.opcode != "x":
            continue
        if len(operation.operands) != 1 or operation.operands[0] not in positions:
            return None
        value |= 1 << (width - 1 - positions[operation.operands[0]])
    state = register.side is RegisterSide.RIGHT
    canonical = integer_basis_function(
        function.name, register.dtype, value, state=state
    )
    return (value, state) if function == canonical else None


def integer_basis_function(
    name: str, dtype: RegisterType, value: int, *, state: bool = True
) -> FunctionRecord:
    """Represent a concrete unsigned ket/effect in existing owned-word IR.

    Preparation allocates a clean word then applies the existing integer loader.
    An effect applies the same X gates then releases the clean word. The latter
    is a projection in tensor inspection; physical binding requires the caller's
    explicit owned pool and clean-release promise. It is never a reset or trace.
    Widths must be positive and constants built-in integers, as in int_state_ir.
    The original symbolic, NumPy-scalar and zero-width interfaces are wider.
    """
    width = executable_register_width(dtype)
    if type(state) is not bool:
        raise ValueError("integer basis direction requires a built-in boolean")
    gates = int_state_ir(tuple(range(width)), value)
    word = ValueRef("entry", 0, "register")
    bits = tuple(ValueRef("entry", 1 + i, "qubit") for i in range(width))
    operations = []
    if state:
        operations.append(
            Operation(
                "allocate_register",
                results=(word,),
                register_type=dtype,
                workspace_dirty=False,
            )
        )
    operations.append(Operation("split_register", (word,), bits))
    current = list(bits)
    for position, gate in enumerate(gates):
        wire = gate.wires[0]
        result = ValueRef("entry", 1 + width + position, "qubit")
        operations.append(Operation("x", (current[wire],), (result,)))
        current[wire] = result
    output = ValueRef("entry", 1 + width + len(gates), "register")
    operations.append(
        Operation("join_register", tuple(current), (output,), register_type=dtype)
    )
    if not state:
        operations.append(Operation("free_register", (output,), workspace_dirty=False))
    return FunctionRecord(
        name,
        (
            BlockRecord(
                "entry",
                () if state else (word,),
                operations=tuple(operations),
                return_values=(output,) if state else (),
            ),
        ),
        registers=(
            Register(
                "val", dtype, side=RegisterSide.RIGHT if state else RegisterSide.LEFT
            ),
        ),
    )
