# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Dependency order follows locked Qualtran v0.7.0 MeanEstimationOperator.

"""Compose native code context without inventing a missing upstream code codec."""

from math import pi

from ...algorithms.primitives.mean_estimation_operator import CodeForRandomVariable
from ...compiler.formats.legacy_component_arguments import ControlSpecification
from ...compiler.registers.component_control import ControlAdder
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


def controlled_mean_estimation_component_pair(
    code: CodeForRandomVariable,
    prepare_pair: tuple[ProgramRecord, ProgramRecord],
    encoder_pair: tuple[ProgramRecord, ProgramRecord],
    positive_encoder_pair: tuple[ProgramRecord, ProgramRecord],
    controls: ControlSpecification,
    *,
    arctan_bitsize: int = 32,
    signal_junk_names: tuple[str, ...] | None = None,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Reduce typed controls once around the actual source-positive mean calls.

    All encoder inverses and the positive encoder are explicit owners. The
    original smaller mean ABI is retained in private-junk mode; caller-owned
    signal junk remains a separate expanded context. No control or inverse is
    inferred from matrix equivalence, and no code-object codec is fabricated.
    """
    from ...compiler.formats.legacy_controlled_factories import (
        controlled_via_and_factories,
    )
    from ...compiler.formats.legacy_word_factories import _rename_program
    from ...compiler.registers.component_control import (
        _stored_control_adder,
        quantum_control_signature,
    )
    from ...core.ir.program.register_types import quantum_bit_type
    from ._complex_phase_library import positive_complex_phase_pair

    base = mean_estimation_component_pair(
        code,
        prepare_pair,
        encoder_pair,
        arctan_bitsize=arctan_bitsize,
        signal_junk_names=signal_junk_names,
    )
    ports = positive_encoder_pair[0].functions[0].registers
    if ports is None or ports[1:] != encoder_pair[0].functions[0].registers:
        raise SerializationError(
            "Positive mean encoder must retain original data ports"
        )
    phase_pair = positive_complex_phase_pair(
        positive_encoder_pair,
        tuple(r.name for r in code.encoder_target_registers),
        arctan_bitsize,
    )
    selection_names = tuple(r.name for r in code.synthesizer_selection_registers)
    reflection = prepared_reflection_program(
        *prepare_pair,
        selection_names,
        control_value=1,
        global_phase=-1,
        signal_junk_names=signal_junk_names,
    )
    signature = quantum_control_signature(
        Signature(base[0].functions[0].registers or ()), ((quantum_bit_type(), ()),)
    )
    positive = qubitization_walk_component_pair(
        signature, phase_pair, (reflection, reflection), controlled=True
    )
    if (
        len(controls.clauses) == 1
        and controls.clauses[0][0] == quantum_bit_type()
        and controls.clauses[0][1].shape == ()
        and controls.clauses[0][1].item() == 1
    ):
        return positive
    result = []
    for child, controlled in zip(base, positive, strict=True):
        module = _rename_program(controlled, "positive_mean")
        root = child.functions[0]

        def owner(
            function: FunctionRecord,
            *,
            expected: FunctionRecord = root,
            positive_program: ProgramRecord = module,
        ) -> tuple[ProgramRecord, ControlAdder]:
            if function != expected:
                raise SerializationError("Unknown mean-estimation control child")
            return positive_program, _stored_control_adder(
                positive_program.functions[0]
            )

        factory = controlled_via_and_factories(control_system=owner)[
            "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
        ]
        record = factory("main", None, {"subbloq": root, "ctrl_spec": controls})
        result.append(
            record if isinstance(record, ProgramRecord) else ProgramRecord((record,))
        )
    return result[0], result[1]
