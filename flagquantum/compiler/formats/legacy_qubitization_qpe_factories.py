# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor follows Qualtran v0.7.0 phase_estimation/qubitization_qpe.py.

"""Bind walk-QPE constructors to explicitly owned directional dependencies."""

from collections.abc import Callable, Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.phase.qubitization_phase_estimation_component import (
    qubitization_phase_estimation_program,
)
from ..registers.phase.textbook_qft_component import textbook_qft_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program

ExactPair = tuple[ProgramRecord, ProgramRecord]
WalkDependencies = tuple[ExactPair, ExactPair, tuple[ExactPair, ...]]


def qubitization_qpe_factories(
    *,
    preparation_pair: Callable[[FunctionRecord], ExactPair],
    walk_dependencies: Callable[[FunctionRecord, int], WalkDependencies],
    fourier_pair: Callable[[FunctionRecord], ExactPair],
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Require original child owners, with complete helper and inverse closures.

    Walk dependencies supply positive-controlled walk, zero-controlled reflection
    and powers 2/4/8/... in that order. No graph is recovered from resource counts
    or matching port widths. An omitted qft_inv uses the source default; an
    explicit child always goes to its owning provider. Providers contain native
    records only and remain scoped to the caller's library compilation.
    """
    recipes: dict[
        FunctionRecord,
        tuple[ExactPair, WalkDependencies, ExactPair, tuple[Register, ...], bool],
    ] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        preparation: ExactPair,
        walk: WalkDependencies,
        fourier: ExactPair,
        targets: tuple[Register, ...],
        inverse: bool,
    ) -> ProgramRecord:
        control, reflection, powers = walk
        pairs = (preparation, control, reflection, *powers, fourier)
        record = qubitization_phase_estimation_program(
            preparation[0],
            control[0],
            reflection[0],
            tuple(p[0] for p in powers),
            fourier[0],
            targets,
            adjoint=inverse,
            inverse_pairs=pairs,
        )
        if registers is not None and registers != record.functions[0].registers:
            raise SerializationError("Invalid qubitization QPE register signature")
        record = _rename_program(record, name)
        recipes[record.functions[0]] = (preparation, walk, fourier, targets, inverse)
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if not {"walk", "ctrl_state_prep"} <= arguments.keys() or arguments.keys() - {
            "walk",
            "ctrl_state_prep",
            "qft_inv",
        }:
            raise SerializationError(
                "Walk QPE requires walk, preparation and optional inverse QFT"
            )
        walk, prepare = arguments["walk"], arguments["ctrl_state_prep"]
        if not isinstance(walk, FunctionRecord) or not isinstance(
            prepare, FunctionRecord
        ):
            raise SerializationError("Walk QPE requires resolved native children")
        preparation = preparation_pair(prepare)
        phase = preparation[0].functions[0].registers or ()
        if len(phase) != 1 or type(phase[0].bitsize) is not int or phase[0].bitsize < 1:
            raise SerializationError(
                "Stored walk QPE needs a positive finite phase width"
            )
        width = phase[0].bitsize
        children = walk_dependencies(walk, width)
        if "qft_inv" in arguments:
            qft = arguments["qft_inv"]
            if not isinstance(qft, FunctionRecord):
                raise SerializationError(
                    "Explicit inverse QFT requires its native owner"
                )
            fourier = fourier_pair(qft)
        else:
            fourier = (
                textbook_qft_program(width, adjoint=True),
                textbook_qft_program(width),
            )
        if walk.registers is None:
            raise SerializationError("Walk QPE requires typed walk targets")
        return make(
            name, registers, preparation, children, fourier, walk.registers, False
        )

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            preparation, walk, fourier, targets, inverse = recipes[child]
            return make(
                name, registers, preparation, walk, fourier, targets, not inverse
            )
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact walk QPE adjoint constructor context")

    return {
        "qualtran.bloqs.phase_estimation.qubitization_qpe.QubitizationQPE": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
