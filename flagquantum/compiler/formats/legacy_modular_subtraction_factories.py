# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 mod_arithmetic/mod_subtraction.py.

"""Resolve modular negation and subtraction through their shared graph owners."""

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
from ...errors import SerializationError
from ..registers.modular.controlled_modular_negation_component import (
    controlled_modular_negation_component_program,
    controlled_modular_negation_program,
)
from ..registers.modular.controlled_modular_subtraction_component import (
    controlled_modular_subtraction_component_program,
    controlled_modular_subtraction_program,
)
from ..registers.modular.modular_negation_component import (
    modular_negation_component_program,
    modular_negation_program,
)
from ..registers.modular.modular_subtraction_component import (
    modular_subtraction_component_program,
    modular_subtraction_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_constant_control_factories import unresolved_constant_addition_program
from .legacy_modular_addition_factories import _integer, _symbolic_addition
from .legacy_word_factories import _rename_program

_Kind = Literal["ModNeg", "CModNeg", "ModSub", "CModSub"]


def _symbolic_negation(
    dtype: RegisterType,
    modulus: sympy.Expr,
    name: str,
    *,
    controlled: bool = False,
    control_value: int = 1,
) -> ProgramRecord:
    # The source Controlled(AddK(k=p+1)) explicitly refuses symbolic k.
    # The remaining available control/fanout layers retain their complete bodies.
    correction = unresolved_constant_addition_program(
        dtype if controlled else unsigned_register_type(dtype.num_qubits),
        modulus + 1,
        name="controlled_correction",
        controlled=True,
    )
    program = (
        controlled_modular_negation_component_program(
            dtype, correction, control_value=control_value
        )
        if controlled
        else modular_negation_component_program(dtype, correction)
    )
    return _rename_program(program, name)


def _modular_factory(
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    controlled = kind in {"CModNeg", "CModSub"}
    allowed = {"dtype", "mod", "cv"} if controlled else {"dtype", "mod"}
    if not {"dtype", "mod"} <= arguments.keys() or arguments.keys() - allowed:
        raise SerializationError(f"Invalid constructor arguments for {kind}")
    dtype = arguments["dtype"]
    if not isinstance(dtype, RegisterType) or dtype.encoding not in {
        "uint",
        "montgomery",
    }:
        raise SerializationError(
            "Modular subtraction requires unsigned or Montgomery words"
        )
    modulus = _integer(arguments, "mod")
    cv = arguments.get("cv", 1)
    if type(cv) is not int or cv not in (0, 1):
        raise SerializationError("Modular subtraction requires a binary control value")
    ports = [Register("ctrl", quantum_bit_type())] if controlled else []
    ports.append(Register("x", dtype))
    if kind in {"ModSub", "CModSub"}:
        ports.append(Register("y", dtype))
    signature = tuple(ports)
    if registers is not None and registers != signature:
        raise SerializationError(f"Invalid register signature for {kind}")
    if (
        kind in {"ModNeg", "CModNeg"}
        and not dtype.is_symbolic()
        and isinstance(modulus, sympy.Expr)
    ):
        return _symbolic_negation(
            dtype, modulus, name, controlled=controlled, control_value=cv
        )
    if (
        kind in {"ModSub", "CModSub"}
        and not dtype.is_symbolic()
        and isinstance(modulus, sympy.Expr)
        and modulus.free_symbols
    ):
        load = unresolved_constant_addition_program(
            dtype, modulus + 1, name="load_complement"
        )
        unload = unresolved_constant_addition_program(
            dtype, modulus + 1, name="unload_complement", adjoint=True
        )
        addition = _symbolic_addition(
            (
                dtype
                if controlled
                else montgomery_register_type(dtype.num_qubits, modulus)
            ),
            modulus,
            controlled=controlled,
            control_value=cv,
        )
        program = (
            controlled_modular_subtraction_component_program(
                dtype, load, unload, addition
            )
            if controlled
            else modular_subtraction_component_program(dtype, load, unload, addition)
        )
        return _rename_program(program, name)
    if dtype.is_symbolic() or isinstance(modulus, sympy.Expr):
        return FunctionRecord(name, (), registers=signature)
    if kind == "ModNeg":
        program = modular_negation_program(dtype, modulus)
    elif kind == "CModNeg":
        program = controlled_modular_negation_program(dtype, modulus, control_value=cv)
    elif kind == "ModSub":
        program = modular_subtraction_program(dtype, modulus)
    else:
        program = controlled_modular_subtraction_program(
            dtype, modulus, control_value=cv
        )
    return _rename_program(program, name)


def modular_subtraction_factories() -> dict[str, ComponentFactory]:
    """Supply all four source constructors with exact dependency ownership."""
    kinds: tuple[_Kind, ...] = ("ModNeg", "CModNeg", "ModSub", "CModSub")
    factories: dict[str, ComponentFactory] = {
        f"qualtran.bloqs.mod_arithmetic.mod_subtraction.{kind}": partial(
            _modular_factory, kind
        )
        for kind in kinds
    }
    # These exact package aliases are present in the frozen source resolver.
    for kind in ("ModSub", "CModSub"):
        factories[f"qualtran.bloqs.mod_arithmetic.{kind}"] = factories[
            f"qualtran.bloqs.mod_arithmetic.mod_subtraction.{kind}"
        ]
    return factories
