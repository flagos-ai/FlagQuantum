# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 mod_arithmetic/mod_subtraction.py.

"""Resolve modular negation and subtraction through their shared graph owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_control import quantum_control_signature
from ..registers.modular.controlled_modular_negation_component import (
    controlled_modular_negation_program,
)
from ..registers.modular.controlled_modular_subtraction_component import (
    controlled_modular_subtraction_program,
)
from ..registers.modular.modular_negation_component import (
    modular_negation_component_program,
    modular_negation_program,
)
from ..registers.modular.modular_subtraction_component import (
    modular_subtraction_program,
)
from .legacy_arithmetic_factories import arithmetic_factories
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program

_Kind = Literal["ModNeg", "CModNeg", "ModSub", "CModSub"]


def _unencoded_constant(dtype: RegisterType, value: object) -> tuple[int, ...]:
    raise SerializationError("Symbolic modular correction has no concrete bit encoding")


def _symbolic_negation(
    dtype: RegisterType, modulus: sympy.Expr, name: str
) -> ProgramRecord:
    # The source Controlled(AddK(k=p+1)) explicitly refuses symbolic k.
    # The remaining available control/fanout layers retain their complete bodies.
    constant = arithmetic_factories(encode_constant=_unencoded_constant)[
        "qualtran.bloqs.arithmetic.addition.AddK"
    ](
        "correction_constant",
        None,
        {"dtype": unsigned_register_type(dtype.num_qubits), "k": modulus + 1},
    )
    assert isinstance(constant, FunctionRecord) and not constant.blocks
    assert constant.registers is not None
    correction = ProgramRecord(
        (
            FunctionRecord(
                "controlled_correction",
                (),
                registers=tuple(
                    quantum_control_signature(
                        Signature(constant.registers), ((quantum_bit_type(), ()),)
                    )
                ),
            ),
        )
    )
    return _rename_program(modular_negation_component_program(dtype, correction), name)


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
    if kind == "ModNeg" and not dtype.is_symbolic() and isinstance(modulus, sympy.Expr):
        return _symbolic_negation(dtype, modulus, name)
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
