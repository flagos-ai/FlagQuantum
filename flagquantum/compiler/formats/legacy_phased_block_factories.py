# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 block_encoding/phase.py.

"""Resolve Phase through existing scalar carriers and explicit native child pairs."""

from collections.abc import Callable, Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicReal
from ...errors import SerializationError
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.phase.controlled_block_phase_component import (
    controlled_block_phase_program,
)
from ..registers.phase.phased_block_component import phased_block_component_pair
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_rotation_factories import _real_argument
from .legacy_scalar_factories import _global_phase_factory
from .legacy_unitary_block_factories import unitary_block_factories
from .legacy_word_factories import _rename_program

ExactPair = tuple[ProgramRecord, ProgramRecord]
PairOwner = Callable[[FunctionRecord], ExactPair]
_Role = tuple[FunctionRecord, SymbolicReal, SymbolicReal, bool]


def phased_block_factories(
    *,
    child_owner: PairOwner | None = None,
    positive_child_owner: PairOwner | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Keep exact pairs scoped to one compilation and never guess child bodies.

    Existing Unitary constructors supply ordinary fixed-gate blocks. Other
    children and their source-positive specializations require explicit native
    owners. The original scalar remains a zero-port call with privately owned
    carrier execution, independently of its zero-cost source resource model.
    """
    pairs: dict[FunctionRecord, ExactPair] = {}
    roles: dict[FunctionRecord, _Role] = {}
    controls: dict[FunctionRecord, tuple[FunctionRecord, ControlSpecification]] = {}
    units = unitary_block_factories(
        on_pair=lambda child, pair: pairs.__setitem__(child, pair)
    )

    def store(
        name: str, registers: tuple[Register, ...] | None, pair: ExactPair
    ) -> ProgramRecord:
        forward, inverse = pair
        if registers is not None and registers != forward.functions[0].registers:
            raise SerializationError("Invalid phased block register signature")
        if inverse.functions[0].registers != forward.functions[0].registers:
            raise SerializationError(
                "Phased block requires matching exact child pair ports"
            )
        result = _rename_program(forward, name)
        pairs[result.functions[0]] = result, inverse
        return result

    def phase(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child = arguments.get("block_encoding")
        if arguments.keys() != {"block_encoding", "phi", "eps"} or not isinstance(
            child, FunctionRecord
        ):
            raise SerializationError("Phase requires its child, phi and eps")
        if child not in pairs:
            if child_owner is None:
                raise SerializationError("Missing exact phased block child owner")
            pairs[child] = child_owner(child)
        exponent, precision = _real_argument(arguments, "phi"), _real_argument(
            arguments, "eps"
        )
        scalar = tuple(
            _global_phase_factory(
                "scalar", (), {"exponent": sign * exponent, "eps": precision}
            )
            for sign in (1, -1)
        )
        pair = phased_block_component_pair(*pairs[child], scalar[0], scalar[1])
        result = store(name, registers, pair)
        roles[result.functions[0]] = child, exponent, precision, False
        return result

    def positive_pair(child: FunctionRecord) -> ExactPair:
        if child not in roles or positive_child_owner is None:
            raise SerializationError(
                "Missing explicit positive phased block child owner"
            )
        original, exponent, _, inverse = roles[child]
        forward, backward = positive_child_owner(original)
        if inverse:
            forward, backward, exponent = backward, forward, -exponent
        return (
            controlled_block_phase_program(forward, exponent),
            controlled_block_phase_program(
                forward, exponent, inverse=backward, adjoint=True
            ),
        )

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
                "Controlled Phase requires child and one positive scalar QBit"
            )
        dtype, values = spec.clauses[0]
        if dtype != quantum_bit_type() or values.shape or int(values) != 1:
            raise SerializationError(
                "Controlled Phase requires one positive scalar QBit"
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
                    original, exponent, precision, direction = roles[child]
                    roles[result.functions[0]] = (
                        original,
                        exponent,
                        precision,
                        not direction,
                    )
                return result
            if child in controls:
                original, spec = controls[child]
                reverse = adjoint(name + "_child_inverse", None, {"subbloq": original})
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
        raise SerializationError("Missing exact phased block adjoint context")

    return {
        **units,
        "qualtran.bloqs.block_encoding.phase.Phase": phase,
        "qualtran._infra.adjoint.Adjoint": adjoint,
        "qualtran._infra.controlled.Controlled": controlled,
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd": predicate,
    }
