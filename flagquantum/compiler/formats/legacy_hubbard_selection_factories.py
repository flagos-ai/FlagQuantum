# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 select_hubbard.py.

"""Resolve Hubbard selection through shared typed call and scalar owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.component_builder import BindingCollection, ComponentProgramBuilder
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.control.hubbard_select_component import hubbard_select_program
from ..registers.control.specialized_component_control import (
    multi_controlled_component_pair,
    specialized_component_control_system,
)
from ..registers.layout.hubbard_signatures import hubbard_signature
from ..registers.lookup.hubbard_majorana_component import hubbard_majorana_program
from ..registers.lookup.hubbard_spin_up_component import hubbard_spin_up_program
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_modular_addition_factories import _integer
from .legacy_multi_control_factories import _literal_values
from .legacy_word_factories import _rename_program

_Kind = Literal["select", "majorana", "spin_up"]


def _program(
    kind: _Kind, arguments: Mapping[str, ResolvedArgument], *, adjoint: bool
) -> ProgramRecord:
    allowed = {"x_dim", "y_dim", "control_val"} | (
        {"gate"} if kind == "majorana" else set()
    )
    if not {"x_dim", "y_dim"} <= arguments.keys() or arguments.keys() - allowed:
        raise SerializationError("Invalid Hubbard selection constructor arguments")
    x, y = (_integer(arguments, key) for key in ("x_dim", "y_dim"))
    if x != y:
        raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")
    cv = arguments.get("control_val")
    if cv is not None and (type(cv) is not int or cv not in (0, 1)):
        raise SerializationError("Hubbard control must be zero, one or None")
    if kind == "select" and cv == 0:
        raise NotImplementedError(
            "control_val=0 not supported, use an outer controlled operation"
        )
    signature = hubbard_signature(kind, x, controlled=cv is not None)
    pauli = arguments.get("gate", "Y")
    if (
        type(x) is not int
        or x == 1
        or kind == "majorana"
        and (cv == 0 or pauli not in ("X", "Y"))
    ):
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    assert isinstance(x, int)
    if kind == "select":
        return hubbard_select_program(x, controlled=cv is not None, adjoint=adjoint)
    if kind == "majorana":
        assert pauli in ("X", "Y")
        assert isinstance(pauli, str)
        return hubbard_majorana_program(
            x, "X" if pauli == "X" else "Y", controlled=cv is not None, adjoint=adjoint
        )
    # The original spin-up body tests presence, not polarity, of control_val.
    # Preserve its positive body for zero, separately from the native validator.
    return hubbard_spin_up_program(x, controlled=cv is not None, adjoint=adjoint)


def hubbard_selection_factories() -> dict[str, ComponentFactory]:
    """Scope exact constructor inverse recipes to one library resolution."""
    recipes: dict[
        FunctionRecord, tuple[_Kind, Mapping[str, ResolvedArgument], bool]
    ] = {}
    controls: dict[FunctionRecord, tuple[FunctionRecord, ControlSpecification]] = {}
    specialized: dict[
        FunctionRecord, tuple[FunctionRecord, tuple[int, ...], str, bool]
    ] = {}

    def make(
        kind: _Kind,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        *,
        adjoint: bool = False,
    ) -> ProgramRecord:
        record = _rename_program(_program(kind, arguments, adjoint=adjoint), name)
        root = record.functions[0]
        if registers is not None and registers != root.registers:
            raise SerializationError("Invalid Hubbard selection register signature")
        recipes[root] = kind, dict(arguments), adjoint
        return record

    def inverse(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if arguments.keys() == {"subbloq"} and isinstance(child, FunctionRecord):
            if child in specialized:
                original, values, control_name, direction = specialized[child]
                return make_specialized(
                    name, registers, original, values, control_name, not direction
                )
            if child in controls:
                original, spec = controls[child]
                backward = inverse(name + "_child_inverse", None, {"subbloq": original})
                return controlled(
                    name,
                    registers,
                    {
                        "subbloq": (
                            backward.functions[0]
                            if isinstance(backward, ProgramRecord)
                            else backward
                        ),
                        "ctrl_spec": spec,
                    },
                )
        if (
            set(arguments) != {"subbloq"}
            or not isinstance(child, FunctionRecord)
            or child not in recipes
        ):
            raise SerializationError("Missing exact Hubbard selection adjoint context")
        kind, metadata, direction = recipes[child]
        return make(kind, name, registers, metadata, adjoint=not direction)

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child in specialized:
            original, values, control_name, direction = specialized[child]
            record = make_specialized(
                child.name + "_positive",
                None,
                original,
                (1, *values),
                control_name,
                direction,
            )

            def add(
                builder: ComponentProgramBuilder,
                control: BindingCollection,
                inputs: dict[str, BindingCollection],
            ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
                import numpy as np

                old = inputs.pop(control_name)
                assert isinstance(old, np.ndarray) and old.shape == (len(values),)
                inputs[control_name] = np.array([control, *old], dtype=object)
                packed, *data = builder.call_tuple(record.functions[0].name, inputs)
                assert isinstance(packed, np.ndarray)
                return packed[0], (packed[1:], *data)

            return record, add
        if child not in recipes or recipes[child][0] != "select":
            raise SerializationError("Missing owned Hubbard SELECT control context")
        _, metadata, direction = recipes[child]
        record = _rename_program(
            _program("select", {**metadata, "control_val": 1}, adjoint=direction),
            child.name + "_positive",
        )
        if metadata.get("control_val") is not None:
            return specialized_component_control_system(
                record, "control", 1, 1, entry=record.functions[0].name
            )
        return record, _stored_control_adder(record.functions[0])

    via_and = controlled_via_and_factories(control_system=positive)[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if isinstance(child, FunctionRecord) and (
            child not in controls
            and child not in specialized
            and (child not in recipes or recipes[child][0] != "select")
        ):
            raise SerializationError("Missing owned Hubbard SELECT control child")
        result = via_and(name, registers, arguments)
        child, spec = arguments["subbloq"], arguments["ctrl_spec"]
        assert isinstance(child, FunctionRecord) and isinstance(
            spec, ControlSpecification
        )
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        controls[root] = child, spec
        return result

    def make_specialized(
        name: str,
        registers: tuple[Register, ...] | None,
        child: FunctionRecord,
        values: tuple[int, ...],
        control_name: str,
        direction: bool,
    ) -> ProgramRecord:
        if child not in recipes:
            raise SerializationError("Missing specialized Hubbard SELECT context")
        kind, metadata, child_direction = recipes[child]
        if (
            kind != "select"
            or metadata.get("control_val") != 1
            or control_name != "control"
        ):
            raise SerializationError(
                "Specialized Hubbard SELECT needs its positive control port"
            )
        pair = multi_controlled_component_pair(
            _program(kind, metadata, adjoint=child_direction),
            _program(kind, metadata, adjoint=not child_direction),
            control_name,
            values,
        )
        result = _rename_program(pair[int(direction)], name)
        if registers is not None and registers != result.functions[0].registers:
            raise SerializationError("Invalid specialized Hubbard SELECT signature")
        specialized[result.functions[0]] = child, values, control_name, direction
        return result

    def specialize(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child, control_name = arguments.get("ctrl_bloq"), arguments.get("ctrl_reg_name")
        if (
            arguments.keys() != {"ctrl_bloq", "ctrl_reg_name", "cvs"}
            or not isinstance(child, FunctionRecord)
            or not isinstance(control_name, str)
        ):
            raise SerializationError("Invalid specialized Hubbard SELECT arguments")
        return make_specialized(
            name,
            registers,
            child,
            _literal_values(arguments["cvs"]),
            control_name,
            False,
        )

    prefix = "qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard."
    return {
        prefix + "SelectHubbard": partial(make, "select"),
        prefix + "HubbardMajorannaOperator": partial(make, "majorana"),
        prefix + "HubbardSpinUpZ": partial(make, "spin_up"),
        "qualtran._infra.adjoint.Adjoint": inverse,
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd": controlled,
        "qualtran.bloqs.mcmt.specialized_ctrl._MultiControlledFromSinglyControlled": specialize,
    }
