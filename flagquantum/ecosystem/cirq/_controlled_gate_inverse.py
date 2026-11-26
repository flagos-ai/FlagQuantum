# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Controlled/Adjoint call order follows Qualtran v0.7.0 controlled.py.

"""Retain external control child boundaries before expanding SDK execution."""

from typing import Any

from ...compiler.formats.legacy_component_arguments import ControlSpecification
from ...compiler.formats.legacy_control_specs import control_registers_from_spec
from ...compiler.formats.legacy_word_factories import _rename_program
from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_control import quantum_control_signature
from ...compiler.registers.component_gate_function import bit_gate_function
from ...compiler.registers.control.specialized_component_control import (
    multi_controlled_component_pair,
)
from ...compiler.registers.primitives.controlled_rotation_component import (
    controlled_rz_program,
)
from ...compiler.registers.primitives.global_phase import scalar_phase_ir
from ...core.ir import CircuitIR
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ._control_values import import_control_values
from ._decompose import decompose_cirq_once
from ._feedback_import import (
    _has_explicit_matrix,
    _lower_feedback_gate,
    _scalar_control_operations,
)
from ._gate_metadata import cirq_gate_signature
from ._named_gate_definitions import _NamedGateDefinitions
from ._register_gate import RegisterCirqGate
from ._register_import import _operation_ports, import_cirq_register_optree
from .conversion import (
    _cirq,
    _cirq_parameter_value,
    _eigen_phase_operations,
    _global_phase_angle,
)


def _raw_controlled_signature(
    gate: Any, declared: _NamedGateDefinitions
) -> Signature | None:
    """Retain nested SDK controls only above an unmapped raw gate owner."""
    cirq = _cirq()
    if declared._select(gate) is not None or isinstance(
        gate, cirq.ops.raw_types._InverseCompositeGate
    ):
        return None
    if not isinstance(gate, cirq.ControlledGate):
        return cirq_gate_signature(gate)
    child = _raw_controlled_signature(gate.sub_gate, declared)
    if child is None:
        return None
    registers, _ = import_control_values(gate.control_values)
    return quantum_control_signature(
        child, tuple((register.dtype, register.shape) for register in registers)
    )


def controlled_inverse_program(
    original: Any,
    spec: ControlSpecification,
    signature: Signature,
    name: str,
    *,
    inverse: bool = True,
) -> ProgramRecord | None:
    """Keep the directed source child graph under each explicit outer adjoint.

    Raw SDK children, CNOT, fixed phases, Hadamard and atomic power rotations
    have explicit execution views here. Positive ZPow uses the native measured
    CZPow owner; positive Hadamard retains the source CHadamard atom. Other
    named components require their own exact control/adjoint owner. No
    decomposition work may escape its child port.
    """
    cirq = _cirq()
    children = decompose_cirq_once(original)
    if children is None:
        return None
    data = tuple(cirq.LineQid.for_gate(original))
    declared = _NamedGateDefinitions()
    nested_signatures = {}
    for child in children:
        if (
            not isinstance(
                child,
                (
                    cirq.GateOperation,
                    cirq.ControlledOperation,
                    cirq.SingleQubitPauliStringGateOperation,
                ),
            )
            or not isinstance(child.gate, cirq.Gate)
            or child.tags
            or not set(child.qubits) <= set(data)
            or isinstance(child.gate, cirq.ops.raw_types._InverseCompositeGate)
        ):
            return None
        selected = declared._select(child.gate)
        if selected is None and isinstance(child.gate, cirq.ControlledGate):
            nested = _raw_controlled_signature(child.gate, declared)
            if nested is None:
                return None
            nested_signatures[child.gate] = nested
        if selected is not None and selected[0] not in {
            "cnot.CNOT",
            "z_basis.CZ",
            "rotation.XPowGate",
            "rotation.YPowGate",
            "rotation.ZPowGate",
            "rotation.Rz",
            "hadamard.Hadamard",
            "s_gate.SGate",
            "t_gate.TGate",
            "swap.TwoBitSwap",
            "x_basis.XGate",
            "global_phase.GlobalPhase",
        }:
            return None
    groups = control_registers_from_spec(spec.clauses)
    bits = tuple(bit for group in groups for bit in group.bits())
    # SDK decompositions can order interchangeable controls by Qid. Keep the
    # declared control-before-data order used by source physical capture.
    controls = tuple(cirq.LineQid(i - len(bits), dimension=2) for i in range(len(bits)))
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

    # The source graph inserts connected nodes from Soquet edges before its
    # isolated zero-port nodes. Control later connects those scalar nodes to
    # the control wire, so their original graph position must survive reversal.
    source_order = tuple(child for child in children if child.qubits) + tuple(
        child for child in children if not child.qubits
    )
    ordered = tuple(reversed(source_order)) if inverse else source_order
    for index, child in enumerate(ordered):
        selected = declared._select(child.gate)
        # Named self-adjoint atoms return themselves. SDK negative powers
        # can have different decompositions despite the same unitary operator.
        self_adjoint = {
            "cnot.CNOT": cirq.CNOT,
            "z_basis.CZ": cirq.CZ,
            "hadamard.Hadamard": cirq.H,
            "swap.TwoBitSwap": cirq.SWAP,
            "x_basis.XGate": cirq.X,
        }
        directed = (
            self_adjoint[selected[0]].on(*child.qubits)
            if selected is not None and selected[0] in self_adjoint
            else (cirq.inverse(child) if inverse else child)
        )
        operation = directed.controlled_by(*controls, control_values=bits)
        definition = declared.get(child.gate)
        child_signature = (
            nested_signatures.get(child.gate, cirq_gate_signature(child.gate))
            if definition is None
            else Signature(definition.registers or ())
        )
        ports = quantum_control_signature(
            child_signature,
            tuple((dtype, values.shape) for dtype, values in spec.clauses),
        )
        combined_cz = (
            selected is not None and selected[0] == "z_basis.CZ" and len(bits) == 1
        )
        if combined_cz:
            # CZ.get_ctrl_system combines the new bit with its existing q1
            # through _MultiControlledFromSinglyControlled, in this order.
            ports = Signature(
                (
                    Register("q1", quantum_bit_type(), (2,)),
                    Register("q2", quantum_bit_type()),
                )
            )
        positive_x_pair = (
            selected is not None
            and selected[0] == "x_basis.XGate"
            and len(spec.clauses) == 1
            and spec.clauses[0][0] == quantum_bit_type()
            and spec.clauses[0][1].shape == (2,)
            and bits == (1, 1)
        )
        native_control = positive_x_pair or (
            selected is not None
            and default_control
            and selected[0]
            in {
                "rotation.ZPowGate",
                "rotation.Rz",
                "global_phase.GlobalPhase",
                "hadamard.Hadamard",
                "swap.TwoBitSwap",
                "x_basis.XGate",
            }
        )
        if definition is not None and (
            native_control
            or default_control
            and selected is not None
            and selected[0] == "cnot.CNOT"
        ):
            # Specialized positive controls own their port grouping: Toffoli
            # packs two controls, CZPow packs both bits, and scalar phase
            # becomes a ZPow on the control itself.
            controlled = declared[operation.gate]
            assert controlled.registers is not None
            ports = Signature(controlled.registers)
        if combined_cz:
            assert definition is not None
            builder = ComponentProgramBuilder(
                child_signature, declared._module(definition).functions, name="main"
            )
            positive = builder.finalize(
                builder.call_dict(definition.name, builder.inputs)
            )
            program = multi_controlled_component_pair(
                positive, positive, "q1", (bits[0], 1)
            )[int(inverse)]
        elif native_control and selected is not None and selected[0] == "rotation.Rz":
            # Adjoint(CRz) reverses the four source calls; CRz(-angle)
            # has the same operator but a different literal call order.
            program = controlled_rz_program(
                _cirq_parameter_value(child.gate._rads, cache={}), adjoint=inverse
            )
        elif native_control:
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
            # Source capture anchors wire-free SDK phases to its first
            # declared input, before the typed importer rotates live words.
            # Keep the owned scalar recipe directly, without a numeric SDK
            # export/import round trip or a new public scalar instruction.
            anchored = []
            scalar_definitions: dict[Any, FunctionRecord] = {}
            scalar_ports = Signature.build(bit0=1)
            for leaf in leaves:
                if not isinstance(leaf.gate, cirq.GlobalPhaseGate):
                    anchored.append(leaf)
                    continue
                scalar = CircuitIR(
                    1,
                    scalar_phase_ir(
                        0, _global_phase_angle(cirq, leaf.gate.coefficient)
                    ),
                )
                carrier = RegisterCirqGate(scalar, scalar_ports)
                scalar_definitions[carrier] = bit_gate_function(
                    f"scalar_{len(scalar_definitions)}", tuple(scalar_ports), scalar
                )
                anchored.append(carrier.on(controls[0]))
            leaves = tuple(anchored)
            incoming, outgoing = _operation_ports(ports, operation.qubits)
            program = import_cirq_register_optree(
                cirq.Circuit(leaves, strategy=cirq.InsertStrategy.NEW),
                signature=ports,
                in_quregs=incoming,
                out_quregs=outgoing,
                gate_definitions=scalar_definitions,
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
