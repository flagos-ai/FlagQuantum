# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal phase gates follow locked Qualtran v0.7.0 ComplexPhaseOracle.

"""Translate literal signed phase gates and reuse the bounded native ArcTan owner."""

from collections.abc import Callable

from ...compiler.formats.legacy_complex_phase_factories import (
    ComplexPhaseContext,
    complex_phase_factories,
)
from ...compiler.formats.legacy_component_resolution import ComponentFactory
from ...compiler.formats.legacy_unitary_block_factories import UnitaryPairOwner
from ...compiler.registers.lookup.arctan_lookup_component import arctan_lookup_program
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...errors import SerializationError
from ._component_library import _gate_program
from ._gate_metadata import cirq_gate_signature
from .conversion import _cirq


def complex_phase_library_factories(
    *,
    encoder_owner: UnitaryPairOwner,
    target_names: Callable[[FunctionRecord], tuple[str, ...]],
    on_pair: (
        Callable[[FunctionRecord, tuple[ProgramRecord, ProgramRecord]], None] | None
    ) = None,
) -> dict[str, ComponentFactory]:
    """Select native lookup explicitly; preserve source SDK phase constructors."""

    def context(encoder: FunctionRecord, precision: int) -> ComplexPhaseContext:
        names = target_names(encoder)
        if encoder.registers is None or not set(names) <= {
            r.name for r in encoder.registers
        }:
            raise SerializationError("Unknown complex phase encoder target ports")
        if any(r.shape_symbolic for r in encoder.registers if r.name in names):
            raise SerializationError(
                "Source complex phase allocation requires scalar target words"
            )
        width = sum(int(r.total_bits()) for r in encoder.registers if r.name in names)
        lookup = arctan_lookup_program(width, precision)
        cirq = _cirq()
        phases = []
        inverses = []
        for index in range(precision):
            for sign, value in ((0, 1), (1, -1)):
                gate = (cirq.Z ** (value / 2 ** (1 + index))).controlled(
                    control_values=[sign]
                )
                record = _gate_program(
                    gate, cirq_gate_signature(gate), f"signed_phase_{index}_{sign}"
                )
                phases.append(
                    record
                    if isinstance(record, ProgramRecord)
                    else ProgramRecord((record,))
                )
                if on_pair is not None:
                    inverse_gate = cirq.inverse(gate)
                    inverse = _gate_program(
                        inverse_gate,
                        cirq_gate_signature(inverse_gate),
                        f"signed_phase_inverse_{index}_{sign}",
                    )
                    inverses.append(
                        inverse
                        if isinstance(inverse, ProgramRecord)
                        else ProgramRecord((inverse,))
                    )
        return ComplexPhaseContext(
            encoder_owner(encoder),
            (lookup, lookup),
            tuple(phases),
            names,
            tuple(inverses) if on_pair is not None else None,
        )

    return complex_phase_factories(context, on_pair=on_pair)


def positive_complex_phase_pair(
    encoder_pair: tuple[ProgramRecord, ProgramRecord],
    target_names: tuple[str, ...],
    precision: int,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Compose source-positive child calls from an explicit controlled encoder.

    The native controlled ArcTan is an explicit bounded lookup replacement for
    the source's opaque arithmetic leaf. Signed phase controls retain the source
    merged S/T controls and raw controlled-CZPow views, not a whole-phase matrix.
    """
    import numpy as np

    from ...compiler.formats.legacy_component_arguments import (
        ControlSpecification,
        ExternalGatePayload,
    )
    from ...compiler.formats.legacy_controlled_factories import (
        controlled_via_and_factories,
    )
    from ...compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
    from ...compiler.registers.component_adjoint import adjoint_component_calls
    from ...compiler.registers.phase.complex_phase_component import (
        complex_phase_oracle_program,
    )
    from ...core.ir.program.register_types import quantum_bit_type
    from ._component_library import cirq_gate_library_factories

    if type(precision) is not int or precision < 0:
        raise SerializationError(
            "Controlled complex phase requires a concrete precision"
        )
    registers = encoder_pair[0].functions[0].registers
    if registers is None:
        raise SerializationError("Controlled encoder requires typed ports")
    width = sum(int(r.total_bits()) for r in registers if r.name in target_names)
    lookup = arctan_lookup_program(width, precision, controlled=True)
    cirq = _cirq()
    fixed = fixed_gate_factories()
    native_controls = controlled_via_and_factories()
    gates = cirq_gate_library_factories()

    def program(record: FunctionRecord | ProgramRecord) -> ProgramRecord:
        return record if isinstance(record, ProgramRecord) else ProgramRecord((record,))

    def phase(index: int, sign: int, inverse: bool) -> ProgramRecord:
        exponent = (-1 if inverse else 1) / 2 ** (1 + index)
        if sign == 0 and index < 2:
            kind = "s_gate.SGate" if index == 0 else "t_gate.TGate"
            child = fixed["qualtran.bloqs.basic_gates." + kind](
                "phase_atom", None, {"is_adjoint": int(inverse)}
            )
            factory = native_controls[
                "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
            ]
        else:
            raw = cirq.Z**exponent if sign == 0 else cirq.CZ**-exponent
            child = gates["qualtran.cirq_interop._cirq_to_bloq.CirqGateAsBloq"](
                "phase_atom",
                None,
                {"gate": ExternalGatePayload(cirq.to_json_gzip(raw))},
            )
            factory = gates[
                (
                    "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
                    if sign == 0
                    else "qualtran._infra.controlled.Controlled"
                )
            ]
        root = child.functions[0] if isinstance(child, ProgramRecord) else child
        return program(
            factory(
                f"signed_{index}_{sign}_{int(inverse)}",
                None,
                {
                    "subbloq": root,
                    "ctrl_spec": ControlSpecification(
                        (
                            (
                                quantum_bit_type(),
                                np.array([1, 0]) if sign == 0 else np.array(1),
                            ),
                        )
                    ),
                },
            )
        )

    phases = tuple(phase(i, sign, False) for i in range(precision) for sign in (0, 1))
    inverse_phases = tuple(
        phase(i, sign, True) for i in range(precision) for sign in (0, 1)
    )
    forward = complex_phase_oracle_program(
        *encoder_pair,
        lookup,
        lookup,
        phases,
        target_names=target_names,
        controlled=True,
    )
    inverse = adjoint_component_calls(
        forward,
        (
            encoder_pair,
            (encoder_pair[1], encoder_pair[0]),
            (lookup, lookup),
            *tuple(zip(phases, inverse_phases, strict=True)),
        ),
    )
    return forward, inverse
