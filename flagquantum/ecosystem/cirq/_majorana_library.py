# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Gate payload adaptation follows Qualtran v0.7.0 serialization/bloq.py.

"""End Majorana gate JSON at the existing native gate-definition boundary."""

from typing import Any

import numpy as np

from ...compiler.formats.legacy_component_arguments import (
    ControlSpecification,
    ExternalGatePayload,
)
from ...compiler.formats.legacy_component_resolution import ComponentFactory
from ...compiler.formats.legacy_controlled_factories import controlled_via_and_factories
from ...compiler.formats.legacy_majorana_factories import majorana_factories
from ...compiler.formats.legacy_word_factories import _rename_program
from ...compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from ...compiler.registers.component_control import (
    ControlAdder,
    quantum_control_signature,
)
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ._component_library import _gate_program
from ._gate_metadata import cirq_gate_signature
from ._named_gate_definitions import _NamedGateDefinitions
from .conversion import _cirq


def majorana_library_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Decode one-qubit gates and return only owned native Program dependencies.

    Exact inverse recipes are selected independently for controlled target
    action and the initial multi-controlled accumulator flip. No source Bloq
    constructor is imported or reconstructed.
    """

    def program(gate: Any, name: str) -> ProgramRecord:
        definitions = _NamedGateDefinitions()
        known = definitions.get(gate)
        if known is not None:
            return _rename_program(definitions._module(known), name)
        cirq = _cirq()
        signature = cirq_gate_signature(gate)
        if isinstance(gate, cirq.ControlledGate):
            base = definitions.get(gate.sub_gate)
            base_signature = (
                Signature(base.registers)
                if base is not None and base.registers is not None
                else cirq_gate_signature(gate.sub_gate)
            )
            count = gate.num_controls()
            signature = quantum_control_signature(
                base_signature,
                ((quantum_bit_type(), () if count == 1 else (count,)),),
            )
        record = _gate_program(gate, signature, name)
        return record if isinstance(record, ProgramRecord) else ProgramRecord((record,))

    def multiple_control_initializer(count: int) -> ProgramRecord:
        """Retain the source X.get_ctrl_system multi-control predicate graph."""
        cirq = _cirq()
        definitions = _NamedGateDefinitions()
        child = definitions[cirq.X]
        positive = definitions[cirq.CNOT]
        module = definitions._module(positive)

        def control_system(
            original: FunctionRecord,
        ) -> tuple[ProgramRecord, ControlAdder]:
            if original != child:
                raise SerializationError("Unexpected Majorana initializer child")

            def add(
                builder: ComponentProgramBuilder,
                control: BindingCollection,
                values: dict[str, BindingCollection],
            ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
                result = builder.call_dict(
                    positive.name, {"ctrl": control, "target": values["q"]}
                )
                return result["ctrl"], (result["target"],)

            return module, add

        factory = controlled_via_and_factories(control_system=control_system)[
            "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
        ]
        record = factory(
            "initial",
            None,
            {
                "subbloq": child,
                "ctrl_spec": ControlSpecification(
                    ((quantum_bit_type(), np.ones(count, dtype=int)),)
                ),
            },
        )
        return record if isinstance(record, ProgramRecord) else ProgramRecord((record,))

    def children(
        payload: ExternalGatePayload | None, count: int
    ) -> tuple[
        tuple[ProgramRecord, ProgramRecord], tuple[ProgramRecord, ProgramRecord]
    ]:
        cirq = _cirq()
        try:
            gate = (
                cirq.Y
                if payload is None
                else cirq.read_json_gzip(gzip_raw=payload.gzip_json)
            )
        except (ValueError, TypeError, OSError, EOFError) as error:
            raise SerializationError(
                "Invalid serialized Majorana gate payload"
            ) from error
        if not isinstance(gate, cirq.Gate) or cirq.num_qubits(gate) != 1:
            raise SerializationError("Majorana target must be a single-qubit gate")
        if not cirq.is_parameterized(gate) and not cirq.has_unitary(gate):
            raise SerializationError("Majorana target must have unitary action")
        target = gate.controlled()
        initial = cirq.X.controlled(num_controls=count) if count else cirq.X
        if count > 2:
            initializer = multiple_control_initializer(count)
            initial_pair = (initializer, initializer)
        else:
            initial_pair = (
                program(initial, "initial"),
                program(cirq.inverse(initial), "initial_inverse"),
            )
        return (
            (
                program(target, "target"),
                program(cirq.inverse(target), "target_inverse"),
            ),
            initial_pair,
        )

    return majorana_factories(gate_programs=children, adjoint_factory=adjoint_factory)
