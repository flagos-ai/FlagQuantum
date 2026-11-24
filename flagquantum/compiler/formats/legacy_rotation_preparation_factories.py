# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 state_preparation_via_rotation.py.

"""Bind coefficient/angle data to owned typed rotation-preparation graphs."""

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import bit_length
from ...errors import SerializationError
from ..registers.lookup.qrom_table import QromTable
from ..registers.phase.phase_gradient_lookup import PhaseGradientLookupCall
from ..registers.phase.phase_gradient_lookup_component import (
    phase_gradient_lookup_program,
)
from ..registers.phase.rotation_table_words import rotation_table_words
from ..registers.preparation.rotation_preparation_component import (
    rotation_preparation_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def rotation_preparation_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Own concrete state/PRGA constructors and exact inverses per compilation.

    Symbolic precision preserves a logical declaration. Source uncompute is a
    distinct table/stage choice; Adjoint reverses that chosen complete graph.
    Unknown inverse children delegate only to an explicit owner or refuse.
    """
    recipes: dict[FunctionRecord, tuple[Callable[[bool], ProgramRecord], bool]] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        build: Callable[[bool], ProgramRecord],
        inverse: bool = False,
    ) -> ProgramRecord:
        record = _rename_program(build(inverse), name)
        if registers is not None and record.functions[0].registers != registers:
            raise SerializationError("Invalid rotation preparation register signature")
        recipes[record.functions[0]] = build, inverse
        return record

    def state(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if not {
            "state_coefficients",
            "phase_bitsize",
        } <= arguments.keys() or arguments.keys() - {
            "state_coefficients",
            "phase_bitsize",
            "control_bitsize",
            "uncompute",
        }:
            raise SerializationError("Invalid rotation state constructor arguments")
        values = arguments["state_coefficients"]
        if (
            not isinstance(values, (tuple, list, np.ndarray))
            or np.asarray(values).ndim != 1
        ):
            raise SerializationError("Rotation coefficients require a concrete vector")
        coefficients = tuple(
            complex(value) for value in np.asarray(values, dtype=complex).flat
        )
        width = _integer(arguments, "phase_bitsize")
        controls = _integer(
            {"control_bitsize": arguments.get("control_bitsize", 0)}, "control_bitsize"
        )
        if type(controls) is not int or controls < 0:
            raise SerializationError(
                "Rotation control width must be concrete and nonnegative"
            )
        uncompute = bool(arguments.get("uncompute", False))
        ports = tuple(
            Signature.build(
                prepare_control=controls,
                target_state=bit_length(len(coefficients) - 1),
                phase_gradient=width,
            )
        )
        if type(width) is int:
            rows, phases = rotation_table_words(
                coefficients, width, uncompute=uncompute
            )
        else:
            rows, phases = (), ()

        def build(inverse: bool) -> ProgramRecord:
            if type(width) is not int or len(coefficients) == 1:
                return ProgramRecord((FunctionRecord("main", (), registers=ports),))
            return rotation_preparation_program(
                rows,
                phases,
                width,
                control_count=controls,
                uncompute=uncompute,
                adjoint=inverse,
            )

        return make(name, registers, build)

    def prga(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if arguments.keys() != {
            "selection_bitsize",
            "phase_bitsize",
            "rom_values",
            "control_bitsize",
        }:
            raise SerializationError(
                "Invalid phase-gradient lookup constructor arguments"
            )
        selection = _integer(arguments, "selection_bitsize")
        phase = _integer(arguments, "phase_bitsize")
        controls = _integer(arguments, "control_bitsize")
        values = arguments["rom_values"]
        if (
            not isinstance(values, (tuple, list, np.ndarray))
            or np.asarray(values).ndim != 1
        ):
            raise SerializationError("Phase lookup requires concrete angle words")
        table = replace(
            QromTable.from_data(
                np.asarray(values), target_bitsizes=(phase,), num_controls=controls
            ),
            selection_bitsizes=(selection,),
        )

        def build(inverse: bool) -> ProgramRecord:
            if type(phase) is not int:
                return ProgramRecord(
                    (
                        FunctionRecord(
                            "main",
                            (),
                            registers=tuple(PhaseGradientLookupCall(table).signature),
                        ),
                    )
                )
            return phase_gradient_lookup_program(table, adjoint=inverse)

        return make(name, registers, build)

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
                "Missing exact rotation adjoint constructor context"
            )
        build, inverse = recipes[child]
        return make(name, registers, build, not inverse)

    prefix = "qualtran.bloqs.state_preparation.state_preparation_via_rotation."
    return {
        prefix + "StatePreparationViaRotations": state,
        prefix + "PRGAViaPhaseGradient": prga,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
