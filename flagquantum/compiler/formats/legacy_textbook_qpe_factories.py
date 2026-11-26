# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 phase_estimation/text_book_qpe.py.

"""Resolve stored phase estimation through owned window and phase-call recipes."""

from collections.abc import Mapping

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt, SymbolicReal
from ...errors import SerializationError
from ..registers.phase.phase_estimation_component import (
    rectangular_z_phase_estimation_program,
)
from ..registers.phase.textbook_qft_component import textbook_qft_program
from .legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from .legacy_phase_window_factories import (
    phase_window_library_factories,
)
from .legacy_qft_factories import qft_factories
from .legacy_rotation_factories import (
    _real_argument,
    rotation_factories,
)
from .legacy_word_factories import _rename_program
from .symbolic_parameters import symbolic_parameter_value

_Z = "qualtran.bloqs.basic_gates.rotation.ZPowGate"
_WINDOW = "qualtran.bloqs.phase_estimation.qpe_window_state.RectangularWindowState"
_QFT = "qualtran.bloqs.qft.qft_text_book.QFTTextBook"
_QPE = "qualtran.bloqs.phase_estimation.text_book_qpe.TextbookQPE"
_ADJOINT = "qualtran._infra.adjoint.Adjoint"


def textbook_qpe_library_factories() -> dict[str, ComponentFactory]:
    """Retain constructor provenance instead of inferring a child from its ports.

    The four original examples use ZPow and rectangular preparation. Other
    unitary/window owners compose their explicit powered Programs through the
    shared Compiler owner; they are not replaced by same-width ZPow recipes.
    Symbolic declarations remain logical and cannot be bound for execution.
    """
    phases: dict[FunctionRecord, SymbolicReal] = {}
    windows: dict[FunctionRecord, ProgramRecord] = {}
    fourier: dict[FunctionRecord, tuple[SymbolicInt, bool, bool]] = {}
    estimates: dict[
        FunctionRecord, tuple[ProgramRecord, SymbolicReal, bool, bool, bool]
    ] = {}

    def checked(
        name: str, registers: tuple[Register, ...] | None, record: ProgramRecord
    ) -> ProgramRecord:
        if registers is not None and registers != record.functions[0].registers:
            raise SerializationError("Invalid phase-estimation register signature")
        return _rename_program(record, name)

    def z(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        record = rotation_factories()[_Z](name, registers, arguments)
        root = record.functions[0] if isinstance(record, ProgramRecord) else record
        phases[root] = _real_argument(arguments, "exponent", 1.0)
        return record

    def window(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        record = phase_window_library_factories()[_WINDOW](name, registers, arguments)
        assert isinstance(record, ProgramRecord)
        windows[record.functions[0]] = record
        return record

    def qft(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        record = qft_factories()[_QFT](name, registers, arguments)
        root = record.functions[0] if isinstance(record, ProgramRecord) else record
        assert root.registers is not None
        fourier[root] = (
            root.registers[0].bitsize,
            bool(arguments.get("with_reverse", True)),
            False,
        )
        return record

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        preparation: ProgramRecord,
        exponent: SymbolicReal,
        reverse: bool,
        qft_inverse: bool,
        inverse: bool,
    ) -> ProgramRecord:
        ports = preparation.functions[0].registers
        assert ports is not None
        width = ports[0].bitsize
        if type(width) is not int:
            # The source carries symbolic widths without decomposing a circuit.
            from ...core.ir.program.register_types import quantum_bit_type

            target = Register("q", quantum_bit_type())
            signature = (ports[0].adjoint() if inverse else ports[0], target)
            record = ProgramRecord((FunctionRecord(name, (), registers=signature),))
        else:
            angle = (
                symbolic_parameter_value(exponent)
                if isinstance(exponent, sympy.Expr)
                else exponent
            )
            record = rectangular_z_phase_estimation_program(
                preparation,
                angle,
                with_reverse=reverse,
                qft_adjoint=qft_inverse,
                adjoint=inverse,
            )
        record = checked(name, registers, record)
        estimates[record.functions[0]] = (
            preparation,
            exponent,
            reverse,
            qft_inverse,
            inverse,
        )
        return record

    def estimate(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if not {
            "unitary",
            "ctrl_state_prep",
        } <= arguments.keys() or arguments.keys() - {
            "unitary",
            "ctrl_state_prep",
            "qft_inv",
        }:
            raise SerializationError(
                "Phase estimation requires unitary, preparation and optional inverse QFT"
            )
        unitary, prep = arguments["unitary"], arguments["ctrl_state_prep"]
        if (
            not isinstance(unitary, FunctionRecord)
            or unitary not in phases
            or not isinstance(prep, FunctionRecord)
            or prep not in windows
        ):
            raise SerializationError("Missing exact unitary/window constructor context")
        assert prep.registers is not None
        width = prep.registers[0].bitsize
        inverse_qft = arguments.get("qft_inv")
        if "qft_inv" not in arguments:
            reverse, direction = True, True
        elif isinstance(inverse_qft, FunctionRecord) and inverse_qft in fourier:
            qft_width, reverse, direction = fourier[inverse_qft]
            if qft_width != width:
                raise SerializationError("Phase word and Fourier widths differ")
        else:
            raise SerializationError("Missing exact Fourier constructor context")
        return make(
            name, registers, windows[prep], phases[unitary], reverse, direction, False
        )

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if arguments.keys() != {"subbloq"} or not isinstance(child, FunctionRecord):
            raise SerializationError(
                "Phase-estimation adjoint requires a resolved child"
            )
        if child in fourier:
            width, reverse, inverse = fourier[child]
            if type(width) is int and width > 0:
                record = checked(
                    name,
                    registers,
                    textbook_qft_program(
                        width, with_reverse=reverse, adjoint=not inverse
                    ),
                )
                root = record.functions[0]
            else:
                root = FunctionRecord(name, (), registers=child.registers)
                record = ProgramRecord((root,))
                if registers is not None and registers != root.registers:
                    raise SerializationError(
                        "Invalid symbolic Fourier register signature"
                    )
            fourier[root] = (width, reverse, not inverse)
            return record
        if child in estimates:
            preparation, exponent, reverse, direction, inverse = estimates[child]
            return make(
                name, registers, preparation, exponent, reverse, direction, not inverse
            )
        raise SerializationError("Missing exact phase-estimation adjoint context")

    return {_Z: z, _WINDOW: window, _QFT: qft, _QPE: estimate, _ADJOINT: adjoint}
