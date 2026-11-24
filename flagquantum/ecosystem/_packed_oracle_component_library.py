# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 Hubbard/packed oracles.

"""Supply packed source examples through existing native Hubbard graph owners."""

from collections.abc import Mapping
from typing import Literal

from ..algorithms.primitives.hubbard_prepare import PrepareHubbard
from ..algorithms.primitives.hubbard_prepare_component import hubbard_prepare_program
from ..algorithms.primitives.hubbard_select import SelectHubbard
from ..algorithms.primitives.hubbard_select_component import hubbard_select_program
from ..compiler.formats.legacy_block_encoding_factories import ControlledSelect
from ..compiler.formats.legacy_component_arguments import ControlSpecification
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_controlled_factories import (
    PositiveControlProvider,
    controlled_via_and_factories,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_packed_oracle_factories import (
    OracleContext,
    packed_oracle_factories,
)
from ..compiler.formats.legacy_rotation_factories import _real_argument
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from ..compiler.registers.component_control import ControlAdder
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ..core.ir.program.registers import Register
from ..core.value_math import ssum
from ..errors import SerializationError

_Context = tuple[
    tuple[Register, ...], tuple[Register, ...], tuple[ProgramRecord, ProgramRecord]
]
_PREPARE = (
    "qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard.PrepareHubbard"
)
_SELECT = (
    "qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard.SelectHubbard"
)


def _packed_oracle_library_context(
    *,
    prepare_context: OracleContext | None = None,
    select_context: OracleContext | None = None,
) -> tuple[dict[str, ComponentFactory], OracleContext, OracleContext, ControlledSelect]:
    """Track complete example metadata, with explicit native owners for other oracles.

    The uncontrolled Hubbard SELECT keeps its scalar-node declaration until
    Proposal 119 is approved. Its packed wrapper may be resolved logically,
    but physical execution still refuses the missing body. No phase is omitted.
    """
    preparations: dict[FunctionRecord, _Context] = {}
    selections: dict[FunctionRecord, _Context] = {}
    select_recipes: dict[FunctionRecord, tuple[int, int | None]] = {}
    pending: dict[
        FunctionRecord, tuple[Literal["prepare", "select"], _Context, ProgramRecord]
    ] = {}

    def dimension(arguments: Mapping[str, ResolvedArgument]) -> int:
        x, y = _integer(arguments, "x_dim"), _integer(arguments, "y_dim")
        if type(x) is not int or type(y) is not int or x < 1 or x != y:
            raise SerializationError(
                "Hubbard example requires a positive concrete square"
            )
        return x

    def checked(
        name: str, registers: tuple[Register, ...] | None, record: ProgramRecord
    ) -> ProgramRecord:
        if registers is not None and registers != record.functions[0].registers:
            raise SerializationError("Invalid Hubbard oracle register signature")
        return _rename_program(record, name)

    def prepare(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if arguments.keys() != {"x_dim", "y_dim", "t", "u"}:
            raise SerializationError(
                "Hubbard preparation requires dimensions and weights"
            )
        t, u = _real_argument(arguments, "t"), _real_argument(arguments, "u")
        if not isinstance(t, (int, float)) or not isinstance(u, (int, float)):
            raise SerializationError(
                "Stored Hubbard preparation requires concrete weights"
            )
        oracle = PrepareHubbard(dimension(arguments), float(t), float(u))
        if oracle.dimension == 1:
            forward = ProgramRecord(
                (FunctionRecord("main", (), registers=tuple(oracle.signature)),)
            )
            inverse = forward
        else:
            forward, inverse = (
                hubbard_prepare_program(oracle, adjoint=direction)
                for direction in (False, True)
            )
        result = checked(name, registers, forward)
        preparations[result.functions[0]] = (
            oracle.selection_registers,
            oracle.junk_registers,
            (result, inverse),
        )
        return result

    def select(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if not {"x_dim", "y_dim"} <= arguments.keys() or arguments.keys() - {
            "x_dim",
            "y_dim",
            "control_val",
        }:
            raise SerializationError(
                "Hubbard SELECT requires dimensions and optional control"
            )
        size = dimension(arguments)
        control = arguments.get("control_val")
        if control is not None and (type(control) is not int or control != 1):
            raise SerializationError(
                "Original Hubbard SELECT refuses this control value"
            )
        # Selection descriptors depend on the square geometry, not weights.
        selection = PrepareHubbard(size, 1.0, 4.0).selection_registers
        target = (Register("target", opaque_register_type(2 * size**2)),)
        ports = (
            *(
                (Register("control", quantum_bit_type()),)
                if control is not None
                else ()
            ),
            *selection,
            *target,
        )
        if control is None or size == 1:
            forward = ProgramRecord((FunctionRecord("main", (), registers=ports),))
            inverse = forward
        else:
            oracle = SelectHubbard(size, controlled=True)
            forward, inverse = (
                hubbard_select_program(oracle, adjoint=direction)
                for direction in (False, True)
            )
        result = checked(name, registers, forward)
        selections[result.functions[0]] = (selection, target, (result, inverse))
        select_recipes[result.functions[0]] = size, control
        return result

    def prepare_owner(child: FunctionRecord) -> _Context:
        if child in pending:
            materialize(child)
        if child in preparations:
            return preparations[child]
        if prepare_context is not None:
            return prepare_context(child)
        raise SerializationError("Missing native PREPARE role context")

    def select_owner(child: FunctionRecord) -> _Context:
        if child in pending:
            materialize(child)
        if child in selections:
            return selections[child]
        if select_context is not None:
            return select_context(child)
        raise SerializationError("Missing native SELECT role context")

    raw = packed_oracle_factories(
        prepare_context=prepare_owner, select_context=select_owner
    )

    def track(
        kind: Literal["prepare", "select"], factory: ComponentFactory
    ) -> ComponentFactory:
        def construct(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
        ) -> FunctionRecord | ProgramRecord:
            result = factory(name, registers, arguments)
            assert isinstance(result, ProgramRecord)
            original = arguments[kind]
            assert isinstance(original, FunctionRecord)
            roles = (
                prepare_owner(original) if kind == "prepare" else select_owner(original)
            )
            # Source wrapper properties use QAny, even when its one-bit signature
            # uses QBit or omits a zero-width port. Keep those roles explicitly.
            selection = (
                Register(
                    "selection",
                    opaque_register_type(ssum(r.total_bits() for r in roles[0])),
                ),
            )
            auxiliary = (
                Register(
                    "junk" if kind == "prepare" else "system",
                    opaque_register_type(ssum(r.total_bits() for r in roles[1])),
                ),
            )
            pending[result.functions[0]] = (
                kind,
                (selection, auxiliary, roles[2]),
                result,
            )
            return result

        return construct

    def materialize(child: FunctionRecord) -> None:
        kind, roles, record = pending.pop(child)
        inverse = raw["qualtran._infra.adjoint.Adjoint"](
            child.name + "_paired_inverse", None, {"subbloq": child}
        )
        assert isinstance(inverse, ProgramRecord)
        context = roles[0], roles[1], (record, inverse)
        (preparations if kind == "prepare" else selections)[child] = context

    def control_owner(
        child: FunctionRecord, value: int
    ) -> tuple[ControlAdder, tuple[tuple[ProgramRecord, ProgramRecord], ...]]:
        if child not in select_recipes or value not in (0, 1):
            raise SerializationError("Missing source SELECT specialized-control owner")
        size, previous = select_recipes[child]
        if previous is not None:
            raise SerializationError(
                "LCU source signature omits the original SELECT control"
            )
        selection, targets, _ = selections[child]
        ports = (Register("control", quantum_bit_type()), *selection, *targets)
        pair = tuple(
            (
                ProgramRecord((FunctionRecord("main", (), registers=ports),))
                if size == 1
                else hubbard_select_program(
                    SelectHubbard(size, controlled=True), adjoint=direction
                )
            )
            for direction in (False, True)
        )

        def positive(record: ProgramRecord) -> ControlAdder:
            def add(
                owner: ComponentProgramBuilder,
                control: BindingCollection,
                inputs: dict[str, BindingCollection],
            ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
                output = owner.call_program(record, {"control": control, **inputs})
                control = output.pop("control")
                return control, tuple(output[reg.name] for reg in child.registers or ())

            return add

        if value == 1:
            return positive(pair[0]), ((pair[0], pair[1]),)
        # Original zero-control fallback is ControlledViaAnd, whose X flips
        # stay inside that child, not directly in the LCU root.
        try:
            import numpy as np
        except ModuleNotFoundError as error:
            raise SerializationError(
                "Controlled oracle metadata requires NumPy"
            ) from error

        def provider_for(module: ProgramRecord) -> PositiveControlProvider:
            def provider(
                original: FunctionRecord,
            ) -> tuple[ProgramRecord, ControlAdder]:
                assert original == child
                return module, positive(module)

            return provider

        controlled = []
        for direction, module in enumerate(pair):
            factory = controlled_via_and_factories(control_system=provider_for(module))[
                "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
            ]
            result = factory(
                child.name + f"_zero_control_{direction}",
                None,
                {
                    "subbloq": child,
                    "ctrl_spec": ControlSpecification(
                        ((quantum_bit_type(), np.array(0)),)
                    ),
                },
            )
            assert isinstance(result, ProgramRecord)
            controlled.append(result)

        def add_zero(
            owner: ComponentProgramBuilder,
            control: BindingCollection,
            inputs: dict[str, BindingCollection],
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            module = controlled[0]
            output = owner.call_program(
                module, {"ctrl": control, **inputs}, entry=module.functions[0].name
            )
            control = output.pop("ctrl")
            return control, tuple(output[reg.name] for reg in child.registers or ())

        return add_zero, ((controlled[0], controlled[1]),)

    factories = dict(raw)
    wrappers: tuple[tuple[Literal["prepare", "select"], str], ...] = (
        (
            "prepare",
            "qualtran.bloqs.state_preparation.black_box_prepare.BlackBoxPrepare",
        ),
        ("select", "qualtran.bloqs.multiplexers.black_box_select.BlackBoxSelect"),
    )
    for kind, identifier in wrappers:
        factories[identifier] = track(kind, raw[identifier])
    factories.update({_PREPARE: prepare, _SELECT: select})
    return factories, prepare_owner, select_owner, control_owner


def packed_oracle_library_factories(
    *,
    prepare_context: OracleContext | None = None,
    select_context: OracleContext | None = None,
) -> dict[str, ComponentFactory]:
    """Supply the original packed constructors through native oracle owners."""
    return _packed_oracle_library_context(
        prepare_context=prepare_context, select_context=select_context
    )[0]
