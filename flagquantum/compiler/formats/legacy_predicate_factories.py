# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 mcmt/ctrl_spec_and.py.

"""Resolve grouped predicates and their exact measured adjoints."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.control.component_predicate import (
    control_predicate_program,
    control_predicate_signature,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_control_specs import control_registers_from_spec
from .legacy_word_factories import _rename_program


def predicate_factories(
    *, adjoint_factory: ComponentFactory
) -> dict[str, ComponentFactory]:
    """Supply grouped integer predicates and compose their adjoint resolver.

    Keep one catalogue per compilation. The supplied existing adjoint resolver
    retains ownership of unrelated children such as MultiAnd. Exact predicate
    recipes are keyed by the complete resolved function, never by callee names.
    """
    recipes: dict[FunctionRecord, tuple[ControlSpecification, bool]] = {}

    def make_program(
        name: str,
        registers: tuple[Register, ...] | None,
        spec: ControlSpecification,
        uncompute: bool,
    ) -> FunctionRecord | ProgramRecord:
        if not spec.clauses or any(
            dtype.encoding
            not in {"bit", "opaque", "uint", "signed", "ones_complement", "montgomery"}
            or dtype.classical_bits != 0
            for dtype, _ in spec.clauses
        ):
            raise SerializationError(
                "CtrlSpecAnd requires quantum integer control storage"
            )
        controls = tuple(
            Register(f"ctrl_{index}", dtype, values.shape)
            for index, (dtype, values) in enumerate(spec.clauses)
        )
        signature = tuple(control_predicate_signature(controls, uncompute=uncompute))
        if registers is not None and registers != signature:
            raise SerializationError("Invalid register signature for grouped predicate")
        result: FunctionRecord | ProgramRecord
        if any(control.is_symbolic() for control in controls):
            result = FunctionRecord(name, (), registers=signature)
        else:
            patterns = control_registers_from_spec(spec.clauses)
            result = _rename_program(
                control_predicate_program(controls, patterns, uncompute=uncompute), name
            )
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        recipes[root] = spec, uncompute
        return result

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if set(arguments) != {"ctrl_spec"} or not isinstance(
            arguments["ctrl_spec"], ControlSpecification
        ):
            raise SerializationError("CtrlSpecAnd requires only ctrl_spec")
        return make_program(name, registers, arguments["ctrl_spec"], False)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if (
            set(arguments) == {"subbloq"}
            and isinstance(arguments["subbloq"], FunctionRecord)
            and arguments["subbloq"] in recipes
        ):
            spec, uncompute = recipes[arguments["subbloq"]]
            return make_program(name, registers, spec, not uncompute)
        return adjoint_factory(name, registers, arguments)

    return {
        "qualtran.bloqs.mcmt.ctrl_spec_and.CtrlSpecAnd": factory,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
