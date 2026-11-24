# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 Hubbard/packed oracles.

"""Supply packed source examples through existing native Hubbard graph owners."""

from collections.abc import Mapping

from ..algorithms.primitives.hubbard_prepare import PrepareHubbard
from ..algorithms.primitives.hubbard_prepare_component import hubbard_prepare_program
from ..algorithms.primitives.hubbard_select import SelectHubbard
from ..algorithms.primitives.hubbard_select_component import hubbard_select_program
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_packed_oracle_factories import (
    OracleContext,
    packed_oracle_factories,
)
from ..compiler.formats.legacy_rotation_factories import _real_argument
from ..compiler.formats.legacy_word_factories import _rename_program
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ..core.ir.program.registers import Register
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


def packed_oracle_library_factories(
    *,
    prepare_context: OracleContext | None = None,
    select_context: OracleContext | None = None,
) -> dict[str, ComponentFactory]:
    """Track complete example metadata, with explicit native owners for other oracles.

    The uncontrolled Hubbard SELECT keeps its scalar-node declaration until
    Proposal 119 is approved. Its packed wrapper may be resolved logically,
    but physical execution still refuses the missing body. No phase is omitted.
    """
    preparations: dict[FunctionRecord, _Context] = {}
    selections: dict[FunctionRecord, _Context] = {}

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
        return result

    def prepare_owner(child: FunctionRecord) -> _Context:
        if child in preparations:
            return preparations[child]
        if prepare_context is not None:
            return prepare_context(child)
        raise SerializationError("Missing native PREPARE role context")

    def select_owner(child: FunctionRecord) -> _Context:
        if child in selections:
            return selections[child]
        if select_context is not None:
            return select_context(child)
        raise SerializationError("Missing native SELECT role context")

    return {
        **packed_oracle_factories(
            prepare_context=prepare_owner, select_context=select_owner
        ),
        _PREPARE: prepare,
        _SELECT: select,
    }
