# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Payload adaptation from Qualtran v0.7.0 serialization/bloq.py and _cirq_to_bloq.py.

"""End serialized external gate payloads at the Cirq interoperability boundary."""

from collections.abc import Mapping
from typing import Any

from ...compiler.formats.legacy_component_arguments import (
    ControlSpecification,
    ExternalGatePayload,
)
from ...compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ...compiler.formats.legacy_control_specs import control_registers_from_spec
from ...compiler.formats.legacy_controlled_factories import (
    _fixed_positive_control,
    controlled_via_and_factories,
)
from ...compiler.formats.legacy_word_factories import _rename_program
from ...compiler.registers.component_control import (
    ControlAdder,
    _stored_control_adder,
    quantum_control_signature,
)
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ._gate_metadata import cirq_gate_signature
from ._named_gate_definitions import _NamedGateDefinitions
from ._named_gate_inverse import named_inverse_program
from ._register_import import (
    _operation_ports,
    import_cirq_register_gate,
    import_cirq_register_optree,
)
from .conversion import _cirq


def _gate_program(
    gate: Any, signature: Signature, name: str
) -> FunctionRecord | ProgramRecord:
    cirq = _cirq()
    definitions = _NamedGateDefinitions()
    program = import_cirq_register_gate(
        gate, signature=signature, gate_definitions=definitions
    )
    if program is None:
        if not cirq.is_parameterized(gate) and not cirq.has_unitary(gate):
            return FunctionRecord(name, (), registers=tuple(signature))
        operation = gate.on(*cirq.LineQid.for_gate(gate))
        inputs, outputs = _operation_ports(signature, operation.qubits)
        program = import_cirq_register_optree(
            cirq.Circuit(operation),
            signature=signature,
            in_quregs=inputs,
            out_quregs=outputs,
            gate_definitions=definitions,
        )
    return _rename_program(program, name)


def cirq_gate_library_factories(
    *,
    controlled_factory: ComponentFactory | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Resolve gate JSON through existing decomposition and explicit-leaf owners.

    SDK objects end at this boundary. Returned Programs own their parameters,
    matrices and workspace. No source-framework constructor is imported. A
    per-compilation catalogue retains SDK recipes for generic controls and
    explicit adjoints.
    Other child controls may delegate to an independently supplied owner.
    """
    recipes: dict[FunctionRecord, Any] = {}
    adjoint_sources: set[FunctionRecord] = set()

    def remember(
        gate: Any,
        signature: Signature,
        name: str,
        *,
        adjoint_source: bool = True,
    ) -> FunctionRecord | ProgramRecord:
        program = _gate_program(gate, signature, name)
        root = program.functions[0] if isinstance(program, ProgramRecord) else program
        recipes[root] = gate
        if adjoint_source:
            adjoint_sources.add(root)
        return program

    def gate_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if set(arguments) != {"gate"} or not isinstance(
            arguments["gate"], ExternalGatePayload
        ):
            raise SerializationError(
                "External gate constructor requires one gate payload"
            )
        cirq = _cirq()
        try:
            gate = cirq.read_json_gzip(gzip_raw=arguments["gate"].gzip_json)
        except (ValueError, TypeError, OSError, EOFError) as error:
            raise SerializationError("Invalid serialized Cirq gate payload") from error
        if not isinstance(gate, cirq.Gate):
            raise SerializationError("External gate payload must decode to a Cirq gate")
        signature = cirq_gate_signature(gate)
        if registers is not None and tuple(signature) != registers:
            raise SerializationError("External gate signature differs from its payload")
        return remember(gate, signature, name)

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if set(arguments) != {"subbloq", "ctrl_spec"}:
            raise SerializationError("Controlled gate requires subbloq and ctrl_spec")
        child, spec = arguments["subbloq"], arguments["ctrl_spec"]
        if not isinstance(child, FunctionRecord) or child not in recipes:
            if controlled_factory is not None:
                return controlled_factory(name, registers, arguments)
            raise SerializationError(
                "Missing external controlled-gate constructor context"
            )
        if not isinstance(spec, ControlSpecification) or child.registers is None:
            raise SerializationError(
                "External controlled gate requires typed control metadata"
            )
        signature = quantum_control_signature(
            Signature(child.registers),
            tuple((dtype, values.shape) for dtype, values in spec.clauses),
        )
        if registers is not None and tuple(signature) != registers:
            raise SerializationError("External controlled-gate signature differs")
        if not child.blocks or any(register.is_symbolic() for register in signature):
            return FunctionRecord(name, (), registers=tuple(signature))
        groups = control_registers_from_spec(spec.clauses)
        bits = tuple(bit for group in groups for bit in group.bits())
        gate = recipes[child].controlled(num_controls=len(bits), control_values=bits)
        # Source Controlled owns a graph distinct from the raw SDK gate body.
        return remember(gate, signature, name, adjoint_source=False)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if set(arguments) != {"subbloq"}:
            raise SerializationError("Adjoint gate requires exactly one subbloq")
        child = arguments["subbloq"]
        if not isinstance(child, FunctionRecord) or child not in recipes:
            if adjoint_factory is not None:
                return adjoint_factory(name, registers, arguments)
            raise SerializationError(
                "Missing external adjoint-gate constructor context"
            )
        if child not in adjoint_sources:
            if adjoint_factory is not None:
                return adjoint_factory(name, registers, arguments)
            raise SerializationError(
                "Missing exact adjoint owner for external control-wrapper"
            )
        assert child.registers is not None
        signature = Signature(child.registers).adjoint()
        if registers is not None and tuple(signature) != registers:
            raise SerializationError("External adjoint-gate signature differs")
        definitions = _NamedGateDefinitions()
        program = (
            named_inverse_program(
                recipes[child],
                definitions,
                definitions._exact_pairs,
                lambda: name,
                signature=Signature(child.registers),
            )
            if child.blocks
            else None
        )
        # The explicit source Adjoint requires a child decomposition even
        # when the ordinary Cirq gate has an atomic unitary tensor.
        result: FunctionRecord | ProgramRecord = (
            FunctionRecord(name, (), registers=tuple(signature))
            if program is None
            else program
        )
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        recipes[root] = _cirq().ops.raw_types._InverseCompositeGate(recipes[child])
        adjoint_sources.add(root)
        return result

    def positive_control(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes:
            return _fixed_positive_control(child)
        assert child.registers is not None
        signature = quantum_control_signature(
            Signature(child.registers), ((quantum_bit_type(), ()),)
        )
        if not child.blocks:
            program = ProgramRecord(
                (
                    FunctionRecord(
                        child.name + "_positive", (), registers=tuple(signature)
                    ),
                )
            )
        else:
            resolved = _gate_program(
                recipes[child].controlled(), signature, child.name + "_positive"
            )
            program = (
                resolved
                if isinstance(resolved, ProgramRecord)
                else ProgramRecord((resolved,))
            )
        return program, _stored_control_adder(program.functions[0])

    return {
        **controlled_via_and_factories(control_system=positive_control),
        "qualtran.cirq_interop._cirq_to_bloq.CirqGateAsBloq": gate_factory,
        "qualtran._infra.controlled.Controlled": controlled,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
