# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 chemistry/sparse/select_bloq.py.

"""Resolve source sparse SELECT with independently owned exact inverse recipes."""

from collections.abc import Mapping
from typing import Literal

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.control.specialized_component_control import (
    specialized_component_control_system,
)
from ..registers.lookup.sparse_chemistry_select_component import (
    sparse_select_component_pair,
)
from ..registers.lookup.sparse_selection_control import generic_sparse_select_program
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_word_factories import _rename_program


def sparse_chemistry_factories(
    *,
    adjoint_factory: ComponentFactory | None = None,
    cy_profile: Literal["cirq", "scalar"] = "cirq",
) -> dict[str, ComponentFactory]:
    """Bind the constructor and generic adjoint without importing source classes."""
    pairs: dict[FunctionRecord, tuple[ProgramRecord, ProgramRecord]] = {}
    roles: dict[FunctionRecord, tuple[int, bool, int | None]] = {}
    predicates: dict[
        FunctionRecord, tuple[FunctionRecord, ControlSpecification, bool]
    ] = {}

    def store(
        name: str,
        registers: tuple[Register, ...] | None,
        pair: tuple[ProgramRecord, ProgramRecord],
    ) -> ProgramRecord:
        forward, inverse = pair
        if registers is not None and registers != forward.functions[0].registers:
            raise SerializationError("Invalid source sparse SELECT register signature")
        record = _rename_program(forward, name)
        pairs[record.functions[0]] = record, inverse
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        size, control = arguments.get("num_spin_orb"), arguments.get("control_val")
        if (
            arguments.keys() - {"num_spin_orb", "control_val"}
            or type(size) is not int
            or control is not None
            and (type(control) is not int or control not in (0, 1))
        ):
            raise SerializationError(
                "Sparse SELECT requires concrete size and control metadata"
            )
        assert isinstance(size, int) and (control is None or isinstance(control, int))
        record = store(
            name, registers, sparse_select_component_pair(size, control_value=control)
        )
        roles[record.functions[0]] = size, False, control
        return record

    def positive(
        child: FunctionRecord, *, inverse: bool = False
    ) -> tuple[ProgramRecord, ControlAdder]:
        if child not in roles:
            raise SerializationError(
                "Missing sparse SELECT control context for an uncontrolled root"
            )
        size, generic_inverse, existing_control = roles[child]
        if generic_inverse:
            module = generic_sparse_select_program(
                size,
                control_value=existing_control,
                adjoint=not inverse,
                cy_profile=cy_profile,
            )
        else:
            pair = sparse_select_component_pair(size, control_value=1)
            module = pair[int(inverse)]
        record = _rename_program(module, child.name + "_positive")
        if existing_control is not None and not generic_inverse:
            return specialized_component_control_system(
                record, "control", existing_control, 1, record.functions[0].name
            )
        return record, _stored_control_adder(record.functions[0])

    via_and = controlled_via_and_factories(control_system=positive)[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]

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
        predicates[root] = child, spec, False
        return result

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in pairs
        ):
            forward, inverse = pairs[child]
            record = store(name, registers, (inverse, forward))
            size, direction, controlled = roles[child]
            roles[record.functions[0]] = size, not direction, controlled
            return record
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in predicates
        ):
            original, spec, direction = predicates[child]
            backward = controlled_via_and_factories(
                control_system=lambda target: positive(target, inverse=not direction)
            )["qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"]
            result = backward(name, registers, {"subbloq": original, "ctrl_spec": spec})
            root = result.functions[0] if isinstance(result, ProgramRecord) else result
            predicates[root] = original, spec, not direction
            return result
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact sparse SELECT inverse context")

    return {
        "qualtran.bloqs.chemistry.sparse.select_bloq.SelectSparse": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd": predicate,
    }
