# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 qubitization_walk_operator.py.

"""Resolve walks using explicit signal-state, block and inverse ownership."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_control import (
    ControlAdder,
    _stored_control_adder,
    quantum_control_signature,
)
from ..registers.preparation.qubitization_walk_component import (
    qubitization_walk_component_pair,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_word_factories import _rename_program

ExactPair = tuple[ProgramRecord, ProgramRecord]


@dataclass(frozen=True)
class WalkContext:
    """Retain the actual root order and independently owned signal reflection.

    The binding owner is responsible for correspondence to its assigned source
    definition. Neither signal state nor exact child semantics can be inferred
    from a shallow block FunctionRecord or a positional register signature.
    """

    signature: Signature
    block_pair: ExactPair
    reflection_pair: ExactPair
    positive_block_pair: ExactPair | None = None
    positive_reflection_pair: ExactPair | None = None
    source_signature: Signature | None = None
    positive_source_block_pair: ExactPair | None = None

    def pair(
        self, *, controlled: bool = False, source_control_recipes: bool = False
    ) -> ExactPair:
        """Compose owned calls, optionally selecting literal source control recipes.

        Default pairs retain the stored graph's reverse ownership. The source
        option preserves a separately supplied constructor specialization;
        it makes no mathematical inverse claim for an upstream defect.
        """
        signature = self.signature
        block, reflection = self.block_pair, self.reflection_pair
        if controlled:
            if (
                self.positive_block_pair is None
                or self.positive_reflection_pair is None
            ):
                raise SerializationError("Missing exact positive walk dependencies")
            block, reflection = self.positive_block_pair, self.positive_reflection_pair
            if source_control_recipes and self.positive_source_block_pair is not None:
                # A source constructor may change specialization when forming
                # its inverse. Keep that literal recipe distinct from the
                # stored forward graph's exact mathematical inverse.
                block = self.positive_source_block_pair
            signature = quantum_control_signature(
                signature, ((quantum_bit_type(), ()),)
            )
        return qubitization_walk_component_pair(
            signature, block, reflection, controlled=controlled
        )


def qubitization_walk_factories(
    context_owner: Callable[[FunctionRecord], WalkContext],
    *,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Keep constructor and control recipes scoped to resolved child identity."""
    pairs: dict[FunctionRecord, ExactPair] = {}
    roles: dict[FunctionRecord, tuple[WalkContext, bool]] = {}
    controls: dict[FunctionRecord, tuple[FunctionRecord, ControlSpecification]] = {}

    def store(
        name: str, registers: tuple[Register, ...] | None, pair: ExactPair
    ) -> ProgramRecord:
        forward, inverse = pair
        if registers is not None and registers != forward.functions[0].registers:
            raise SerializationError("Invalid source walk register signature")
        if forward.functions[0].registers != inverse.functions[0].registers:
            raise SerializationError("Walk requires matching exact root ports")
        result = _rename_program(forward, name)
        pairs[result.functions[0]] = result, inverse
        return result

    def walk(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child = arguments.get("block_encoding")
        if arguments.keys() != {"block_encoding"} or not isinstance(
            child, FunctionRecord
        ):
            raise SerializationError("Walk requires its explicit native block context")
        context = context_owner(child)
        if (
            context.source_signature is not None
            and context.source_signature != context.signature
        ):
            raise SerializationError(
                "Source walk ports omit retained signal junk; bind the native context pair directly"
            )
        if child.registers != context.block_pair[0].functions[0].registers:
            raise SerializationError(
                "Walk context disagrees with the resolved block ports"
            )
        result = store(name, registers, context.pair())
        roles[result.functions[0]] = context, False
        return result

    def positive_pair(child: FunctionRecord) -> ExactPair:
        if child not in roles:
            raise SerializationError("Missing exact positive walk context")
        context, inverse = roles[child]
        pair = context.pair(controlled=True, source_control_recipes=True)
        return (pair[1], pair[0]) if inverse else pair

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        record = _rename_program(positive_pair(child)[0], child.name + "_positive")
        return record, _stored_control_adder(record.functions[0])

    via_and = controlled_via_and_factories(control_system=positive)[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        if (
            arguments.keys() != {"subbloq", "ctrl_spec"}
            or not isinstance(child, FunctionRecord)
            or not isinstance(spec, ControlSpecification)
            or len(spec.clauses) != 1
        ):
            raise SerializationError(
                "Controlled walk requires one positive scalar QBit"
            )
        dtype, values = spec.clauses[0]
        if dtype != quantum_bit_type() or values.shape or int(values) != 1:
            raise SerializationError(
                "Controlled walk requires one positive scalar QBit"
            )
        return store(name, registers, positive_pair(child))

    def predicate(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        result = via_and(name, registers, arguments)
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        assert isinstance(child, FunctionRecord) and isinstance(
            spec, ControlSpecification
        )
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        controls[root] = child, spec
        return result

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if arguments.keys() == {"subbloq"} and isinstance(child, FunctionRecord):
            if child in pairs:
                forward, inverse = pairs[child]
                result = store(name, registers, (inverse, forward))
                if child in roles:
                    context, direction = roles[child]
                    roles[result.functions[0]] = context, not direction
                return result
            if child in controls:
                original, spec = controls[child]
                reverse = adjoint(name + "_inverse", None, {"subbloq": original})
                function = (
                    reverse.functions[0]
                    if isinstance(reverse, ProgramRecord)
                    else reverse
                )
                return predicate(
                    name, registers, {"subbloq": function, "ctrl_spec": spec}
                )
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact walk adjoint context")

    return {
        "qualtran.bloqs.qubitization.qubitization_walk_operator.QubitizationWalkOperator": walk,
        "qualtran._infra.adjoint.Adjoint": adjoint,
        "qualtran._infra.controlled.Controlled": controlled,
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd": predicate,
    }
