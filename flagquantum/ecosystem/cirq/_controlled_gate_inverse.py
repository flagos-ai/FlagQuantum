# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Controlled/Adjoint call order follows Qualtran v0.7.0 controlled.py.

"""Retain external control child boundaries before expanding SDK execution."""

from typing import Any

from ...compiler.formats.legacy_component_arguments import ControlSpecification
from ...compiler.formats.legacy_control_specs import control_registers_from_spec
from ...compiler.formats.legacy_word_factories import _rename_program
from ...compiler.registers.component_control import quantum_control_signature
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.signatures import Signature
from ._decompose import decompose_cirq_once
from ._feedback_import import (
    _has_explicit_matrix,
    _lower_feedback_gate,
    _scalar_control_operations,
)
from ._gate_metadata import cirq_gate_signature
from ._named_gate_definitions import _NamedGateDefinitions
from ._register_import import _operation_ports, import_cirq_register_optree
from .conversion import _cirq, _eigen_phase_operations


def controlled_inverse_program(
    original: Any,
    spec: ControlSpecification,
    signature: Signature,
    name: str,
    *,
    inverse: bool = True,
) -> ProgramRecord | None:
    """Keep the directed source child graph under each explicit outer adjoint.

    Raw SDK children, CNOT and atomic power rotations have explicit execution
    views here. Positive ZPow uses the native measured CZPow owner. Other named
    components require their own exact control/adjoint owner. No decomposition
    work may escape its child port.
    """
    cirq = _cirq()
    children = decompose_cirq_once(original)
    if children is None:
        return None
    data = tuple(cirq.LineQid.for_gate(original))
    declared = _NamedGateDefinitions()
    for child in children:
        if (
            not isinstance(child, cirq.GateOperation)
            or child.tags
            or not set(child.qubits) <= set(data)
            or isinstance(
                child.gate,
                (cirq.ControlledGate, cirq.ops.raw_types._InverseCompositeGate),
            )
        ):
            return None
        selected = declared._select(child.gate)
        if selected is not None and selected[0] not in {
            "cnot.CNOT",
            "rotation.XPowGate",
            "rotation.YPowGate",
            "rotation.ZPowGate",
            "global_phase.GlobalPhase",
        }:
            return None
    groups = control_registers_from_spec(spec.clauses)
    bits = tuple(bit for group in groups for bit in group.bits())
    controls = tuple(cirq.NamedQubit(f"control_{i}") for i in range(len(bits)))
    default_control = (
        len(spec.clauses) == 1
        and spec.clauses[0][0] == quantum_bit_type()
        and spec.clauses[0][1].shape == ()
        and bits == (1,)
    )
    definitions: dict[Any, FunctionRecord] = {}
    operations = []

    def admitted(op: Any) -> bool:
        return bool(
            isinstance(op.gate, cirq.GlobalPhaseGate)
            or _has_explicit_matrix(cirq, op.gate)
            or _lower_feedback_gate(cirq, op.gate, [], 0) is not None
        )

    def fallback(op: Any) -> Any:
        result = tuple(
            _scalar_control_operations(cirq, _eigen_phase_operations(cirq, (op,)))
        )
        return None if result == (op,) else result

    def eigen_atoms(op: Any) -> Any:
        eigen = tuple(_eigen_phase_operations(cirq, (op,)))
        normalized = tuple(_scalar_control_operations(cirq, eigen))
        if eigen != (op,) and normalized == eigen and all(map(admitted, normalized)):
            return normalized
        return NotImplemented

    ordered = tuple(reversed(children)) if inverse else tuple(children)
    for index, child in enumerate(ordered):
        directed = cirq.inverse(child) if inverse else child
        operation = directed.controlled_by(*controls, control_values=bits)
        definition = declared.get(child.gate)
        child_signature = (
            cirq_gate_signature(child.gate)
            if definition is None
            else Signature(definition.registers or ())
        )
        ports = quantum_control_signature(
            child_signature,
            tuple((dtype, values.shape) for dtype, values in spec.clauses),
        )
        selected = declared._select(child.gate)
        native_control = (
            selected is not None
            and default_control
            and selected[0] in {"rotation.ZPowGate", "global_phase.GlobalPhase"}
        )
        if (
            definition is not None
            and default_control
            and (native_control or selected is not None and selected[0] == "cnot.CNOT")
        ):
            # Specialized positive controls own their port grouping: Toffoli
            # packs two controls, CZPow packs both bits, and scalar phase
            # becomes a ZPow on the control itself.
            controlled = declared[operation.gate]
            assert controlled.registers is not None
            ports = Signature(controlled.registers)
        if native_control:
            program = declared._module(declared[operation.gate])
        else:
            leaves = tuple(
                cirq.decompose(
                    operation,
                    keep=admitted,
                    intercepting_decomposer=eigen_atoms,
                    fallback_decomposer=fallback,
                )
            )
            if any(not set(leaf.qubits) <= set(operation.qubits) for leaf in leaves):
                return None
            incoming, outgoing = _operation_ports(ports, operation.qubits)
            program = import_cirq_register_optree(
                cirq.Circuit(leaves, strategy=cirq.InsertStrategy.NEW),
                signature=ports,
                in_quregs=incoming,
                out_quregs=outgoing,
            )
        program = _rename_program(program, f"{name}_child_{index}")
        definitions[operation.gate] = program.functions[0]
        definitions.update((function.name, function) for function in program.functions)
        operations.append(operation)
    incoming, outgoing = _operation_ports(signature, (*controls, *data))
    return _rename_program(
        import_cirq_register_optree(
            cirq.Circuit(operations, strategy=cirq.InsertStrategy.NEW),
            signature=signature,
            in_quregs=incoming,
            out_quregs=outgoing,
            gate_definitions=definitions,
        ),
        name,
    )
