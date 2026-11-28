# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 mod_arithmetic/mod_addition.py.

"""Resolve modular addition constructors through existing typed graph owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register, RegisterType
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.arithmetic.addition_component import addition_component_program
from ..registers.arithmetic.controlled_addition_component import (
    controlled_addition_program,
)
from ..registers.comparison._component_gates import comparison_bit_functions
from ..registers.comparison.controlled_component import controlled_greater_program
from ..registers.comparison.linear_component import linear_greater_program
from ..registers.component_adjoint import adjoint_component_calls
from ..registers.modular.controlled_modular_addition_component import (
    controlled_modular_addition_component_program,
    controlled_modular_addition_program,
)
from ..registers.modular.modular_addition_component import (
    modular_addition_component_program,
    modular_addition_program,
)
from ..registers.modular.modular_constant_component import (
    controlled_modular_constant_program,
)
from ..registers.modular.scaled_modular_addition_component import (
    scaled_modular_addition_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_constant_control_factories import unresolved_constant_addition_program
from .legacy_multi_control_factories import _literal_values
from .legacy_word_factories import _rename_program
from .montgomery_port_views import restore_montgomery_ports, source_modulus_omission

_Kind = Literal["ModAdd", "ModAddK", "CModAdd", "CModAddK", "CtrlScaleModAdd"]


def _symbolic_addition(
    dtype: RegisterType,
    modulus: sympy.Expr,
    *,
    controlled: bool,
    control_value: int = 1,
    adjoint: bool = False,
) -> ProgramRecord:
    width = dtype.num_qubits
    assert type(width) is int
    word = (
        unsigned_register_type(width) if controlled else montgomery_register_type(width)
    )
    wide = (
        unsigned_register_type(width + 1)
        if controlled
        else montgomery_register_type(width + 1)
    )
    subtraction = unresolved_constant_addition_program(
        wide, -modulus, name="subtract_modulus"
    )
    correction = unresolved_constant_addition_program(
        word, modulus, name="restore_modulus", controlled=True
    )
    if controlled:
        addition = controlled_addition_program(word, wide, control_value=control_value)
        comparison = controlled_greater_program(word, control_value=control_value)
        record = controlled_modular_addition_component_program(
            dtype,
            addition,
            subtraction,
            correction,
            comparison,
        )
    else:
        addition = addition_component_program(
            width + 1, left_dtype=wide, right_dtype=wide
        )
        comparison = linear_greater_program(width)
        record = modular_addition_component_program(dtype, subtraction, correction)
    if not adjoint:
        return record
    inverse_addition = (
        controlled_addition_program(
            word, wide, control_value=control_value, adjoint=True
        )
        if controlled
        else addition_component_program(
            width + 1, left_dtype=wide, right_dtype=wide, adjoint=True
        )
    )
    inverse_comparison = (
        controlled_greater_program(word, control_value=control_value, adjoint=True)
        if controlled
        else linear_greater_program(width, adjoint=True)
    )
    x_gate = ProgramRecord((comparison_bit_functions()[0],))
    return adjoint_component_calls(
        record,
        (
            (addition, inverse_addition),
            (comparison, inverse_comparison),
            (
                subtraction,
                unresolved_constant_addition_program(
                    wide, -modulus, name="subtract_modulus_inverse", adjoint=True
                ),
            ),
            (
                correction,
                unresolved_constant_addition_program(
                    word,
                    modulus,
                    name="restore_modulus_inverse",
                    controlled=True,
                    adjoint=True,
                ),
            ),
            (x_gate, x_gate),
        ),
    )


def _integer(arguments: Mapping[str, ResolvedArgument], key: str) -> SymbolicInt:
    value = arguments.get(key)
    if type(value) is not int and not isinstance(value, sympy.Expr):
        raise SerializationError(f"Modular {key} requires integer or symbolic metadata")
    assert isinstance(value, (int, sympy.Expr))
    return value


def _modular_factory(
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    required = {"dtype", "mod"} if kind == "CModAdd" else {"bitsize", "mod"}
    if kind in {"CModAddK", "CtrlScaleModAdd"}:
        required = required | {"k"}
    optional = (
        {"cv"}
        if kind == "CModAdd"
        else {"add_val", "cvs"} if kind == "ModAddK" else set()
    )
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError(f"Invalid constructor arguments for {kind}")
    modulus = _integer(arguments, "mod")
    if kind == "CModAdd":
        dtype = arguments["dtype"]
        if not isinstance(dtype, RegisterType) or dtype.encoding not in {
            "uint",
            "montgomery",
        }:
            raise SerializationError("CModAdd requires an unsigned or Montgomery dtype")
        width = dtype.num_qubits
    else:
        width = _integer(arguments, "bitsize")
        dtype = (
            montgomery_register_type(width, modulus)
            if kind == "ModAdd"
            else unsigned_register_type(width)
        )
    ports = []
    if kind == "ModAddK":
        values = _literal_values(arguments.get("cvs", ()))
        if values:
            ports.append(Register("ctrl", quantum_bit_type(), (len(values),)))
    elif kind != "ModAdd":
        ports.append(Register("ctrl", quantum_bit_type()))
    ports.append(Register("x", dtype))
    if kind in {"ModAdd", "CModAdd", "CtrlScaleModAdd"}:
        ports.append(Register("y", dtype))
    signature = tuple(ports)
    if registers is not None and registers != signature:
        if (
            kind != "ModAdd"
            or len(registers) != len(signature)
            or not all(
                outer == inner or source_modulus_omission(outer, inner)
                for outer, inner in zip(registers, signature, strict=True)
            )
        ):
            raise SerializationError(f"Invalid register signature for {kind}")
    value = _integer(arguments, "k") if kind in {"CModAddK", "CtrlScaleModAdd"} else 0
    if kind == "ModAddK":
        _integer({"add_val": arguments.get("add_val", 1)}, "add_val")
        if (
            isinstance(width, int)
            and isinstance(modulus, int)
            and not 1 <= modulus <= 2**width
        ):
            raise SerializationError("ModAddK modulus must fit the operand width")
    if kind == "CModAdd":
        cv = arguments.get("cv", 1)
        if type(cv) is not int or cv not in (0, 1):
            raise SerializationError("CModAdd requires a binary control value")
    else:
        cv = 1
    if (
        kind in {"ModAdd", "CModAdd"}
        and type(width) is int
        and isinstance(modulus, sympy.Expr)
    ):
        program = _symbolic_addition(
            dtype, modulus, controlled=kind == "CModAdd", control_value=cv
        )
        if registers is not None and registers != signature:
            program = restore_montgomery_ports(program, registers)
        return _rename_program(program, name)
    if kind == "ModAddK" or any(
        isinstance(item, sympy.Expr) for item in (width, modulus, value)
    ):
        # The original ModAddK supplies a value permutation and a model, no body.
        return FunctionRecord(
            name, (), registers=signature if registers is None else registers
        )
    assert (
        isinstance(width, int) and isinstance(modulus, int) and isinstance(value, int)
    )
    if kind == "ModAdd":
        program = modular_addition_program(width, modulus)
    elif kind == "CModAdd":
        cv = arguments.get("cv", 1)
        if type(cv) is not int or cv not in (0, 1):
            raise SerializationError("CModAdd requires a binary control value")
        program = controlled_modular_addition_program(dtype, modulus, control_value=cv)
    elif kind == "CModAddK":
        program = controlled_modular_constant_program(width, modulus, value)
    else:
        program = scaled_modular_addition_program(width, modulus, value)
    if registers is not None and registers != signature:
        program = restore_montgomery_ports(program, registers)
    return _rename_program(program, name)


def modular_addition_factories() -> dict[str, ComponentFactory]:
    """Supply all five source constructors, retaining the atomic model boundary."""
    kinds: tuple[_Kind, ...] = (
        "ModAdd",
        "ModAddK",
        "CModAdd",
        "CModAddK",
        "CtrlScaleModAdd",
    )
    factories: dict[str, ComponentFactory] = {
        f"qualtran.bloqs.mod_arithmetic.mod_addition.{kind}": partial(
            _modular_factory, kind
        )
        for kind in kinds
    }
    # These exact package aliases are present in the frozen source resolver.
    for kind in ("ModAdd", "CModAddK"):
        factories[f"qualtran.bloqs.mod_arithmetic.{kind}"] = factories[
            f"qualtran.bloqs.mod_arithmetic.mod_addition.{kind}"
        ]
    return factories
