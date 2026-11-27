# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from Qualtran v0.7.0 arithmetic/hamming_weight.py.

"""Bind Hamming metadata to existing compression widths and executable trees."""

from collections.abc import Mapping

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.bits.hamming_weight import hamming_weight_resource_widths
from ..registers.bits.hamming_weight_component import source_hamming_weight_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _hamming_weight_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if arguments.keys() != {"bitsize"}:
        raise SerializationError("HammingWeightCompute requires bitsize")
    width = _integer(arguments, "bitsize")
    junk, output = hamming_weight_resource_widths(width)
    ports = (
        Register("x", unsigned_register_type(width)),
        Register("junk", RegisterType("opaque", junk), side=RegisterSide.RIGHT),
        Register("out", unsigned_register_type(output), side=RegisterSide.RIGHT),
    )
    if registers is not None:
        # Serialized dimension symbols lose their original integer assumptions.
        normalized = (
            output.xreplace({sympy.floor(width): width})
            if isinstance(output, sympy.Expr)
            else output
        )
        integer_ports = (
            *ports[:-1],
            Register(
                "out", unsigned_register_type(normalized), side=RegisterSide.RIGHT
            ),
        )
        if registers not in (ports, integer_ports):
            raise SerializationError("Invalid HammingWeightCompute register signature")
        ports = registers
    if not isinstance(width, (int, sympy.Integer)) or width <= 0:
        return FunctionRecord(name, (), registers=ports)
    return _rename_program(source_hamming_weight_program(width), name)


def hamming_weight_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Supply Hamming constructors and compilation-scoped exact inverse recipes.

    Keep one table per compilation. Only complete resolved functions produced
    by this owner qualify for its inverse; unrelated children require an
    explicitly supplied owner. Nonliteral symbolic/zero widths remain declarations.
    """
    recipes: dict[FunctionRecord, tuple[SymbolicInt, bool]] = {}

    def remember(
        result: FunctionRecord | ProgramRecord, width: SymbolicInt, inverse: bool
    ) -> FunctionRecord | ProgramRecord:
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        recipes[root] = width, inverse
        return result

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        result = _hamming_weight_factory(name, registers, arguments)
        return remember(result, _integer(arguments, "bitsize"), False)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) != {"subbloq"}
            or not isinstance(child, FunctionRecord)
            or child not in recipes
        ):
            if adjoint_factory is not None:
                return adjoint_factory(name, registers, arguments)
            raise SerializationError(
                "Missing exact Hamming adjoint constructor context"
            )
        width, inverse = recipes[child]
        assert child.registers is not None
        expected = tuple(Signature(child.registers).adjoint())
        if registers is not None and registers != expected:
            raise SerializationError("Invalid Hamming adjoint register signature")
        result: FunctionRecord | ProgramRecord
        if isinstance(width, (int, sympy.Integer)) and width > 0:
            result = _rename_program(
                source_hamming_weight_program(width, adjoint=not inverse), name
            )
        else:
            result = FunctionRecord(name, (), registers=expected)
        return remember(result, width, not inverse)

    return {
        "qualtran.bloqs.arithmetic.hamming_weight.HammingWeightCompute": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
