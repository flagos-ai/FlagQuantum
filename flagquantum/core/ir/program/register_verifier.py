"""Infer owned word types at the finite Program register boundary.

Proposal 103 prerequisite 1.2 uses the existing SSA and linearity verifier.
This module supplies register port compatibility and typed lifetime signatures.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..diagnostics import LevelDiagnostic, _diagnostic
from .model import BlockRecord, FunctionRecord, OpcodeSignature, ValueRef
from .register_compatibility import check_register_types_consistent
from .register_types import classical_bit_type, quantum_bit_type
from .registers import Register, RegisterSide, RegisterType
from .signatures import Signature


def executable_register_width(dtype: RegisterType) -> int:
    """Return the concrete width admitted for a packed quantum word.

    Metadata constructors have wider support. Execution currently admits only
    unparameterized bit, opaque and unsigned quantum encodings.
    """
    if not isinstance(dtype, RegisterType):
        raise ValueError("an executable word needs a neutral RegisterType")
    width = dtype.quantum_bits
    if (
        dtype.encoding not in {"bit", "opaque", "uint"}
        or type(width) is not int
        or width < 0
        or type(dtype.classical_bits) is not int
        or dtype.classical_bits != 0
        or dtype.parameters
        or (dtype.encoding == "bit" and width != 1)
    ):
        raise ValueError(f"unsupported executable register descriptor {dtype!r}")
    return width


def register_port_types(
    function: FunctionRecord, side: RegisterSide
) -> tuple[RegisterType, ...]:
    """Flatten quantum-word and owned CBit ports in row-major element order."""
    if not isinstance(function.registers, tuple):
        raise ValueError("a register function requires immutable declared ports")
    if any(not isinstance(register, Register) for register in function.registers):
        raise ValueError("register ports must use neutral Register descriptors")
    signature = Signature(function.registers)
    for register in signature:
        dtype = register.dtype
        if not (
            isinstance(dtype, RegisterType)
            and type(dtype.quantum_bits) is int
            and type(dtype.classical_bits) is int
            and dtype == classical_bit_type()
        ):
            executable_register_width(dtype)
        if register.side not in (
            RegisterSide.LEFT,
            RegisterSide.RIGHT,
            RegisterSide.THRU,
        ):
            raise ValueError(f"unsupported executable register side {register.side!r}")
        if any(type(size) is not int or size < 0 for size in register.shape_symbolic):
            raise ValueError("executable array dimensions must be nonnegative integers")
    ports = signature.lefts() if side is RegisterSide.LEFT else signature.rights()
    return tuple(register.dtype for register in ports for _ in register.all_idxs())


def verify_register_block(
    function: FunctionRecord,
    block: BlockRecord,
    functions: Mapping[str, FunctionRecord],
) -> tuple[dict[int, OpcodeSignature], list[LevelDiagnostic]]:
    """Derive word types from definitions and supply signatures to Core typing."""
    signatures: dict[int, OpcodeSignature] = {}
    diagnostics: list[LevelDiagnostic] = []
    register_values = tuple(
        value
        for value in (
            *block.arguments,
            *(value for operation in block.operations for value in operation.results),
            *(block.return_values or ()),
        )
        if value.kind == "register"
    )
    rebindings = any(
        op.opcode
        in {
            "split_register",
            "join_register",
            "allocate_register",
            "free_register",
            "measure_register",
        }
        or op.register_type is not None
        or op.workspace_dirty is not None
        for op in block.operations
    )
    if function.registers is None:
        if register_values or rebindings:
            diagnostics.append(
                _diagnostic(
                    "program.register_structure",
                    "word values and rebindings require declared register ports",
                )
            )
        return signatures, diagnostics
    try:
        inputs = register_port_types(function, RegisterSide.LEFT)
        outputs = register_port_types(function, RegisterSide.RIGHT)
    except (ValueError, TypeError) as exc:
        diagnostics.append(_diagnostic("program.register_type", str(exc)))
        return signatures, diagnostics
    _check_port_kinds(block.arguments, inputs, diagnostics)
    types = dict(zip(block.arguments, inputs, strict=False))
    for position, operation in enumerate(block.operations):
        if operation.opcode == "call":
            callee = functions.get(operation.callee or "")
            if callee is not None and callee.registers is not None:
                try:
                    left = register_port_types(callee, RegisterSide.LEFT)
                    right = register_port_types(callee, RegisterSide.RIGHT)
                except (ValueError, TypeError):
                    # The callee's own verification reports its invalid ports.
                    continue
                _check_port_types(operation.operands, left, types, diagnostics)
                types.update(zip(operation.results, right, strict=False))
        elif operation.opcode == "measure_register":
            signatures[position] = OpcodeSignature(
                "measure_register",
                operand_kinds=("register",),
                result_kinds=("register",),
            )
            dtype = types.get(operation.operands[0]) if operation.operands else None
            if dtype != quantum_bit_type():
                diagnostics.append(
                    _diagnostic(
                        "program.register_type",
                        "destructive measurement requires an owned QBit register",
                    )
                )
            types.update((value, classical_bit_type()) for value in operation.results)
        elif operation.opcode == "split_register":
            dtype = types.get(operation.operands[0]) if operation.operands else None
            try:
                if dtype is None:
                    raise ValueError("split needs a producer-defined word type")
                width = executable_register_width(dtype)
            except ValueError as exc:
                diagnostics.append(_diagnostic("program.register_type", str(exc)))
            else:
                signatures[position] = OpcodeSignature(
                    "split_register",
                    operand_kinds=("register",),
                    result_kinds=("qubit",) * width,
                )
        elif operation.opcode in {"join_register", "allocate_register"}:
            dtype = operation.register_type
            try:
                if dtype is None:
                    raise ValueError(
                        "join/allocation requires an explicit destination descriptor"
                    )
                width = executable_register_width(dtype)
            except ValueError as exc:
                diagnostics.append(_diagnostic("program.register_type", str(exc)))
            else:
                signatures[position] = OpcodeSignature(
                    operation.opcode,
                    operand_kinds=(
                        ("qubit",) * width
                        if operation.opcode == "join_register"
                        else ()
                    ),
                    result_kinds=("register",),
                )
                types.update((value, dtype) for value in operation.results)
        elif operation.opcode == "free_register":
            signatures[position] = OpcodeSignature(
                "free_register", operand_kinds=("register",)
            )
            dtype = types.get(operation.operands[0]) if operation.operands else None
            try:
                if dtype is None:
                    raise ValueError("free needs a producer-defined word type")
                executable_register_width(dtype)
            except ValueError as exc:
                diagnostics.append(_diagnostic("program.register_type", str(exc)))
        workspace = operation.opcode in {"allocate_register", "free_register"}
        if workspace:
            if type(operation.workspace_dirty) is not bool:
                diagnostics.append(
                    _diagnostic(
                        "program.register_structure",
                        "workspace operations require an explicit boolean dirty flag",
                    )
                )
        elif operation.workspace_dirty is not None:
            diagnostics.append(
                _diagnostic(
                    "program.register_structure",
                    "only workspace operations can state a dirty flag",
                )
            )
        if operation.opcode in {
            "split_register",
            "join_register",
            "allocate_register",
            "free_register",
            "measure_register",
        }:
            if (
                operation.parameters
                or operation.literal is not None
                or operation.condition_clauses
                or operation.physical_data is not None
                or operation.callee is not None
            ):
                diagnostics.append(
                    _diagnostic(
                        "program.register_structure",
                        "word structure operations cannot carry gate, call or physical metadata",
                    )
                )
        if (
            operation.opcode not in {"join_register", "allocate_register"}
            and operation.register_type is not None
        ):
            diagnostics.append(
                _diagnostic(
                    "program.register_structure",
                    "only join/allocation can state a register type",
                )
            )
    returned = block.return_values or ()
    _check_port_kinds(returned, outputs, diagnostics)
    _check_port_types(returned, outputs, types, diagnostics)
    return signatures, diagnostics


def _check_port_kinds(
    values: tuple[ValueRef, ...],
    ports: tuple[RegisterType, ...],
    diagnostics: list[LevelDiagnostic],
) -> None:
    if len(values) != len(ports) or any(value.kind != "register" for value in values):
        diagnostics.append(
            _diagnostic(
                "program.register_structure",
                "register arguments and returns need one owned word per declared element",
            )
        )


def _check_port_types(
    values: tuple[ValueRef, ...],
    expected: tuple[RegisterType, ...],
    types: Mapping[ValueRef, RegisterType],
    diagnostics: list[LevelDiagnostic],
) -> None:
    for value, dtype in zip(values, expected, strict=False):
        actual = types.get(value)
        if (
            actual is None
            or actual.classical_bits != dtype.classical_bits
            or not check_register_types_consistent(actual, dtype)
        ):
            diagnostics.append(
                _diagnostic(
                    "program.register_type",
                    f"word {value!r} has producer type {actual!r}, expected {dtype!r}",
                )
            )
