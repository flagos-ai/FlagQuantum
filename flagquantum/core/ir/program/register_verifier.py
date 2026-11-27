"""Infer owned word types at the finite Program register boundary.

Proposal 103 prerequisite 1.2 uses the existing SSA and linearity verifier.
This module supplies register port compatibility and typed lifetime signatures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from ...value_math import SymbolicInt
from ...value_parameters import is_symbolic_value
from ..diagnostics import LevelDiagnostic, _diagnostic
from .field_register_validation import (
    _field_register_width,
    logical_field_register_width,
)
from .model import BlockRecord, FunctionRecord, OpcodeSignature, ValueRef
from .register_compatibility import check_register_types_consistent
from .register_types import (
    bounded_unsigned_register_type,
    classical_bit_type,
    fixed_point_register_type,
    montgomery_register_type,
    ones_complement_register_type,
    quantum_bit_type,
)
from .registers import Register, RegisterSide, RegisterType
from .signatures import Signature


def executable_register_width(dtype: RegisterType) -> int:
    """Return the concrete width admitted for a packed quantum word.

    Execution admits concrete bit, opaque, integer and raw field storage,
    including packed polynomials (Proposals 109 and 125). Nonbinary storage
    does not admit numerical field arithmetic or control interpretations.
    Proposals 113 through 116 admit concrete unsigned fixed-point storage,
    including empty words, and signed fixed-point words. Proposal 124 admits
    signed empty words only when their concrete fractional width is negative.
    Canonical parameters retain their metadata identity during binding.
    """
    if not isinstance(dtype, RegisterType):
        raise ValueError("an executable word needs a neutral RegisterType")
    width = dtype.quantum_bits
    if dtype.encoding in {"finite_field", "finite_field_polynomial"}:
        return _field_register_width(dtype)
    if dtype.encoding == "fixed_point":
        logical_register_width(dtype)
        parameters = dict(dtype.parameters)
        if (
            type(width) is int
            and width >= 0
            and type(parameters["fractional_width"]) is int
            and parameters["fractional_width"] <= width
        ):
            return width
        raise ValueError(f"unsupported executable register descriptor {dtype!r}")
    if (
        dtype.encoding
        not in {"bit", "opaque", "uint", "signed", "ones_complement", "montgomery"}
        or type(width) is not int
        or width < 0
        or type(dtype.classical_bits) is not int
        or dtype.classical_bits != 0
        or any(type(value) is not int for _, value in dtype.parameters)
        or (dtype.encoding == "bit" and width != 1)
    ):
        raise ValueError(f"unsupported executable register descriptor {dtype!r}")
    try:
        logical_register_width(dtype)
    except ValueError as error:
        raise ValueError(
            f"unsupported executable register descriptor {dtype!r}"
        ) from error
    return width


def logical_register_width(dtype: RegisterType) -> SymbolicInt:
    """Validate a logical quantum descriptor without claiming physical support."""
    if not isinstance(dtype, RegisterType):
        raise ValueError("a logical word needs a neutral RegisterType")
    if dtype.encoding in {"finite_field", "finite_field_polynomial"}:
        return logical_field_register_width(dtype)
    width = dtype.quantum_bits
    if (
        (not is_symbolic_value(width) and (type(width) is not int or width < 0))
        or type(dtype.classical_bits) is not int
        or dtype.classical_bits != 0
        or not isinstance(dtype.parameters, tuple)
        or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            for pair in dtype.parameters
        )
    ):
        raise ValueError(f"unsupported logical register descriptor {dtype!r}")
    if dtype.encoding in {"bit", "opaque", "uint", "signed"} and not dtype.parameters:
        if dtype.encoding == "bit" and (type(width) is not int or width != 1):
            raise ValueError("a logical quantum bit requires width one")
        return width
    names = tuple(name for name, _ in dtype.parameters)
    parameters = dict(dtype.parameters)
    if dtype.encoding == "ones_complement" and not names:
        expected = ones_complement_register_type(width)
    elif dtype.encoding == "montgomery" and names in ((), ("modulus",)):
        modulus = parameters.get("modulus")
        if (
            modulus is not None
            and not is_symbolic_value(modulus)
            and type(modulus) is not int
        ):
            raise ValueError(
                "a logical Montgomery modulus must be integral or symbolic"
            )
        expected = montgomery_register_type(width, cast("SymbolicInt | None", modulus))
    elif dtype.encoding == "uint" and names == ("bound",):
        bound = parameters["bound"]
        if not is_symbolic_value(bound) and type(bound) is not int:
            raise ValueError("a logical unsigned bound must be integral or symbolic")
        expected = bounded_unsigned_register_type(width, cast(SymbolicInt, bound))
    elif dtype.encoding == "fixed_point" and names == ("fractional_width", "signed"):
        fractional = parameters["fractional_width"]
        signed = parameters["signed"]
        if (not is_symbolic_value(fractional) and type(fractional) is not int) or type(
            signed
        ) is not bool:
            raise ValueError(
                "logical fixed-point parameters require a width and boolean sign"
            )
        expected = fixed_point_register_type(
            width, cast(SymbolicInt, fractional), signed
        )
    else:
        raise ValueError(f"unsupported logical register descriptor {dtype!r}")
    if dtype != expected:
        raise ValueError(f"noncanonical logical register descriptor {dtype!r}")
    return width


def concrete_logical_register_width(dtype: RegisterType) -> int:
    """Resolve logical bit arity for split/join without physical admission."""
    width = logical_register_width(dtype)
    if type(width) is not int:
        raise ValueError("split/join requires a concrete logical register width")
    return width


def logical_classical_register_width(dtype: RegisterType) -> SymbolicInt:
    """Validate canonical owned classical storage without binding slots.

    Proposal 108 preserves CBit identity and constructor parameter ordering.
    A quantum metadata view reuses existing constructor validation. Proposal 110
    admits concrete binary fields and polynomials. Proposal 111 reuses structural
    field metadata for nonbinary classical words. Proposal 125 shares this
    structural validation with raw quantum storage without changing classical words.
    """
    if (
        dtype == classical_bit_type()
        and type(dtype.quantum_bits) is int
        and type(dtype.classical_bits) is int
    ):
        return 1
    if (
        not isinstance(dtype, RegisterType)
        or type(dtype.quantum_bits) is not int
        or dtype.quantum_bits != 0
        or dtype.parameters[:1] != (("storage", "classical"),)
        or dtype.encoding
        not in {
            "uint",
            "signed",
            "ones_complement",
            "montgomery",
            "fixed_point",
            "finite_field",
            "finite_field_polynomial",
        }
    ):
        raise ValueError(f"unsupported classical register descriptor {dtype!r}")
    quantum_view = replace(
        dtype,
        quantum_bits=dtype.classical_bits,
        classical_bits=0,
        parameters=dtype.parameters[1:],
    )
    if dtype.encoding in {"finite_field", "finite_field_polynomial"}:
        return _field_register_width(quantum_view)
    return logical_register_width(quantum_view)


def executable_classical_register_width(dtype: RegisterType) -> int:
    """Resolve a canonical classical element's storage width for a feedback read."""
    width = logical_classical_register_width(dtype)
    if type(width) is not int or any(
        type(value) is not int and not (name == "signed" and type(value) is bool)
        for name, value in dtype.parameters
        if name != "storage"
    ):
        raise ValueError("classical execution requires concrete integral metadata")
    return width


def validate_register_declaration(function: FunctionRecord) -> Signature:
    """Validate logical ports without enumerating symbolic array dimensions."""
    if not isinstance(function.registers, tuple):
        raise ValueError("a register function requires immutable declared ports")
    if any(not isinstance(register, Register) for register in function.registers):
        raise ValueError("register ports must use neutral Register descriptors")
    signature = Signature(function.registers)
    for register in signature:
        dtype = register.dtype
        if dtype == classical_bit_type() or dtype.parameters[:1] == (
            ("storage", "classical"),
        ):
            logical_classical_register_width(dtype)
        else:
            logical_register_width(dtype)
        if register.side not in (
            RegisterSide.LEFT,
            RegisterSide.RIGHT,
            RegisterSide.THRU,
        ):
            raise ValueError(f"unsupported executable register side {register.side!r}")
        if any(
            not is_symbolic_value(size) and (type(size) is not int or size < 0)
            for size in register.shape_symbolic
        ):
            raise ValueError(
                "logical array dimensions must be nonnegative integers or symbolic"
            )
    return signature


def register_port_types(
    function: FunctionRecord, side: RegisterSide
) -> tuple[RegisterType, ...]:
    """Flatten concrete quantum-word and owned CBit ports in row-major order."""
    signature = validate_register_declaration(function)
    if any(
        type(size) is not int or size < 0
        for register in signature
        for size in register.shape_symbolic
    ):
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
            "parity_register",
            "read_classical_register",
            "write_classical_register",
            "cast_classical_register",
            "discard_classical_register",
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
                except (ValueError, TypeError) as exc:
                    diagnostics.append(_diagnostic("program.register_type", str(exc)))
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
        elif operation.opcode == "discard_classical_register":
            signatures[position] = OpcodeSignature(
                "discard_classical_register", operand_kinds=("register",)
            )
            dtype = types.get(operation.operands[0]) if operation.operands else None
            if dtype != classical_bit_type():
                diagnostics.append(
                    _diagnostic(
                        "program.register_type",
                        "classical discard requires an owned CBit register",
                    )
                )
        elif operation.opcode == "cast_classical_register":
            signatures[position] = OpcodeSignature(
                operation.opcode,
                operand_kinds=("register",),
                result_kinds=("register",),
            )
            source = types.get(operation.operands[0]) if operation.operands else None
            target = operation.register_type
            try:
                if source is None or target is None:
                    raise ValueError("classical cast requires source and target types")
                if logical_classical_register_width(
                    source
                ) != logical_classical_register_width(target):
                    raise ValueError("classical cast must preserve raw bit width")
            except ValueError as exc:
                diagnostics.append(_diagnostic("program.register_type", str(exc)))
            else:
                types.update((value, target) for value in operation.results)
        elif operation.opcode in {
            "read_classical_register",
            "write_classical_register",
        }:
            dtype = types.get(operation.operands[0]) if operation.operands else None
            width = 1
            try:
                if dtype is None:
                    raise ValueError(
                        "classical access needs a producer-defined word type"
                    )
                width = executable_classical_register_width(dtype)
            except ValueError as exc:
                diagnostics.append(_diagnostic("program.register_type", str(exc)))
            writing = operation.opcode == "write_classical_register"
            signatures[position] = OpcodeSignature(
                operation.opcode,
                operand_kinds=("register",) + (("bool",) * width if writing else ()),
                result_kinds=("register",) + (() if writing else ("bool",) * width),
            )
            if dtype is not None and operation.results:
                types[operation.results[0]] = dtype
        elif operation.opcode == "parity_register":
            count = len(operation.operands)
            signatures[position] = OpcodeSignature(
                operation.opcode,
                operand_kinds=("register",) * count,
                result_kinds=("register",) * (count + 1),
            )
            if any(
                types.get(value) != classical_bit_type() for value in operation.operands
            ):
                diagnostics.append(
                    _diagnostic(
                        "program.register_type",
                        "classical parity and feedback reads require owned CBit registers",
                    )
                )
            types.update(
                (value, classical_bit_type())
                for value in operation.results
                if value.kind == "register"
            )
        elif operation.opcode == "split_register":
            dtype = types.get(operation.operands[0]) if operation.operands else None
            try:
                if dtype is None:
                    raise ValueError("split needs a producer-defined word type")
                width = concrete_logical_register_width(dtype)
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
                logical_register_width(dtype)
                width = (
                    concrete_logical_register_width(dtype)
                    if operation.opcode == "join_register"
                    else 0
                )
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
                logical_register_width(dtype)
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
            "parity_register",
            "read_classical_register",
            "write_classical_register",
            "cast_classical_register",
            "discard_classical_register",
        }:
            if (
                operation.parameters
                or operation.literal is not None
                or (
                    operation.condition_clauses
                    and operation.opcode != "write_classical_register"
                )
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
            operation.opcode
            not in {"join_register", "allocate_register", "cast_classical_register"}
            and operation.register_type is not None
        ):
            diagnostics.append(
                _diagnostic(
                    "program.register_structure",
                    "only join/allocation/classical cast can state a register type",
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
