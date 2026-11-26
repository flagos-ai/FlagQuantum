# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Dependency order follows locked Qualtran v0.7.0 MeanEstimationOperator.

"""Compose native code context without inventing a missing upstream code codec."""

from math import pi

from ...algorithms.primitives.mean_estimation_operator import CodeForRandomVariable
from ...compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ...compiler.registers.preparation.qubitization_walk_component import (
    qubitization_walk_component_pair,
)
from ...compiler.registers.primitives.global_phase_component import scalar_phase_program
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ._complex_phase_library import complex_phase_library_factories


def mean_estimation_component_pair(
    code: CodeForRandomVariable,
    prepare_pair: tuple[ProgramRecord, ProgramRecord],
    encoder_pair: tuple[ProgramRecord, ProgramRecord],
    *,
    arctan_bitsize: int = 32,
    signal_junk_names: tuple[str, ...] | None = None,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Use exact owned children; the caller establishes clean reflection junk.

    The original source cannot serialize its code object. This native context
    preserves that refusal and does not register a fabricated shallow factory.
    Explicit signal junk remains caller-owned when private cleanup is not valid;
    its expanded logical signature is a separate native physical context.
    """
    encoder = encoder_pair[0].functions[0]
    expected = (*code.encoder_selection_registers, *code.encoder_target_registers)
    if encoder.registers is None or {r.name: r for r in encoder.registers} != {
        r.name: r for r in expected
    }:
        raise SerializationError("Mean-estimation encoder disagrees with code context")
    prepare_ports = prepare_pair[0].functions[0].registers
    selections = tuple(r.name for r in code.synthesizer_selection_registers)
    if prepare_ports is None or {
        r.name: r for r in prepare_ports if r.name in selections
    } != {r.name: r for r in code.synthesizer_selection_registers}:
        raise SerializationError("Mean-estimation PREPARE disagrees with code context")
    pairs: list[tuple[ProgramRecord, ProgramRecord]] = []

    def remember(
        child: FunctionRecord, pair: tuple[ProgramRecord, ProgramRecord]
    ) -> None:
        pairs.append(pair)

    factory = next(
        iter(
            complex_phase_library_factories(
                encoder_owner=lambda _: encoder_pair,
                target_names=lambda _: tuple(
                    r.name for r in code.encoder_target_registers
                ),
                on_pair=remember,
            ).values()
        )
    )
    factory("mean_phase", None, {"encoder": encoder, "arctan_bitsize": arctan_bitsize})
    (phase_pair,) = pairs
    reflection = prepared_reflection_program(
        *prepare_pair,
        selections,
        global_phase=-1,
        scalar_phase=scalar_phase_program(pi),
        signal_junk_names=signal_junk_names,
    )
    reflection_ports = reflection.functions[0].registers
    assert reflection_ports is not None
    return qubitization_walk_component_pair(
        Signature(
            (
                *code.encoder_selection_registers,
                *(r for r in reflection_ports if r.name not in selections),
            )
        ),
        phase_pair,
        (reflection, reflection),
    )
