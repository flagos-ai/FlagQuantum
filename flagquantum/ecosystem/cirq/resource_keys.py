# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cirq_gate_to_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_cirq_to_bloq.py

"""Map supported Cirq gates to FlagQuantum's declared resource identities."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, cast

from ...compiler._component_trace.compound_control_resource import (
    CONTROLLED_HADAMARD_COST,
    compound_control_key,
    global_phase_cost_key,
)
from ...compiler._component_trace.resource_classification import classify_resource_gate
from ...compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledRotationCall,
    ControlledSwapCall,
    ControlledZPowerCall,
    ControlledZRotationCall,
    FixedTCall,
    PartitionedCall,
    PhaseCall,
    PredicateControlledCall,
    ResourceCall,
    ResourceSequenceCall,
    RotationCall,
    ZeroCostCall,
    identity_cost_key,
)
from ...compiler._component_trace.resource_model import ComponentCost, GateCost
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...core.value_math import SymbolicComplex, pi, sarg
from ._decompose import decompose_cirq_once
from .conversion import _cirq


@dataclass(frozen=True)
class CirqResourceCall:
    """An SDK gate awaiting explicit recognition at the Ecosystem boundary."""

    gate: Hashable
    # Outer-to-inner partition layouts around the wrapped gate.
    layouts: tuple[Hashable, ...] = ()


def generalize_cirq_resource_call(
    call: ResourceCall | CirqResourceCall,
) -> ResourceCall | CirqResourceCall:
    """Resolve a wrapped SDK gate in place in an ordered resource rewrite chain.

    Unsupported gates stay opaque within Ecosystem. Known gates enter the
    native resource model with their surrounding partition layouts intact.
    """
    if not isinstance(call, CirqResourceCall):
        return call
    cirq = _cirq()
    gate = call.gate
    if (
        isinstance(gate, cirq.ControlledGate)
        and _single_control_clause(gate.control_values) is None
    ):
        raise ValueError("Cirq resource generalization requires one AND control clause")
    key = cirq_gate_resource_key(gate)
    if key is None and isinstance(gate, cirq.ops.raw_types._InverseCompositeGate):
        leaves = cirq_gate_decomposition_resource_keys(gate)
        if leaves is not None:
            key = ResourceSequenceCall(leaves)
    if key is None:
        return call
    for layout in reversed(call.layouts):
        key = PartitionedCall(key, layout)
    return key


def cirq_gate_resource_key(gate: object) -> ResourceCall | None:
    """Return a recognized resource identity or leave the gate at the SDK boundary.

    This follows the source's exact-instance lookup before inspecting power
    parameters. Unknown gates and unsupported control wrappers return None.
    It does not import or lower a circuit.
    """
    cirq = _cirq()
    if not isinstance(gate, cirq.Gate) or any(d != 2 for d in cirq.qid_shape(gate)):
        return None

    fixed: dict[object, ResourceCall] = {
        cirq.T: PhaseCall("t"),
        cirq.T**-1: PhaseCall("t", adjoint=True),
        cirq.S: PhaseCall("s"),
        cirq.S**-1: PhaseCall("s", adjoint=True),
        cirq.H: CliffordCall("h", 1),
        cirq.ControlledGate(cirq.H): CONTROLLED_HADAMARD_COST,
        cirq.CNOT: CliffordCall("cx", 2),
        cirq.TOFFOLI: FixedTCall("toffoli"),
        cirq.X: CliffordCall("x", 1),
        cirq.Y: CliffordCall("y", 1),
        cirq.ControlledGate(cirq.Y): CliffordCall("cy", 2),
        # Recent Cirq versions specialize the positive control as a CY constant.
        cirq.Y.controlled(): CliffordCall("cy", 2),
        cirq.Z: CliffordCall("z", 1),
        cirq.CZ: CliffordCall("cz", 2),
        cirq.SWAP: CliffordCall("swap_bit", 2),
        cirq.CSWAP: ControlledSwapCall(1),
        cirq.I: identity_cost_key(1),
    }
    try:
        key = fixed.get(gate)
    except TypeError:
        return None
    if key is not None:
        return key

    if isinstance(gate, cirq.ControlledGate):
        controls = _single_control_clause(gate.control_values)
        if controls is None:
            return None
        return _controlled_key(cirq_gate_resource_key(gate.sub_gate), controls)

    rotations: dict[type, Literal["rx", "ry", "rz"]] = {
        cirq.Rx: "rx",
        cirq.Ry: "ry",
        cirq.Rz: "rz",
    }
    gate_type = type(gate)
    if gate_type in rotations:
        return RotationCall(rotations[gate_type], cast(SymbolicReal, gate._rads))

    powers: dict[type, Literal["xpow", "ypow", "zpow"]] = {
        cirq.XPowGate: "xpow",
        cirq.YPowGate: "ypow",
        cirq.ZPowGate: "zpow",
    }
    if gate_type in powers and gate.global_shift == 0:
        return RotationCall(powers[gate_type], cast(SymbolicReal, gate.exponent))
    if gate_type is cirq.CZPowGate and gate.global_shift == 0:
        return ControlledZPowerCall(cast(SymbolicReal, gate.exponent))
    if isinstance(gate, cirq.GlobalPhaseGate):
        return ZeroCostCall("global_phase", (cast(Hashable, gate.coefficient),))
    return None


def _single_control_clause(control_values: object) -> tuple[int, ...] | None:
    cirq = _cirq()
    if isinstance(control_values, cirq.ProductOfSums):
        if all(len(allowed) == 1 for allowed in control_values):
            return tuple(allowed[0] for allowed in control_values)
    elif isinstance(control_values, cirq.SumOfProducts):
        clauses = iter(control_values)
        first = next(clauses, None)
        if first is not None and next(clauses, None) is None:
            return tuple(first)
    return None


def _controlled_key(
    key: ResourceCall | None, controls: tuple[int, ...]
) -> ResourceCall | None:
    if (
        isinstance(key, ZeroCostCall)
        and key.opcode == "global_phase"
        and len(key.parameters) == 1
    ):
        # This SDK key retains a coefficient, not the native exponent/precision
        # pair. Preserve the source principal argument before adding controls.
        coefficient = cast(SymbolicComplex, key.parameters[0])
        return global_phase_cost_key(sarg(coefficient) / pi(coefficient), controls)
    if isinstance(key, ZeroCostCall) and key.opcode == "identity":
        return identity_cost_key(1, control_width=len(controls))
    if key is not None:
        compound = compound_control_key(key, controls)
        if compound is not None:
            return compound
    positive = _single_positive_key(key)
    if positive is None or controls == (1,):
        return positive
    return PredicateControlledCall(positive, controls)


def _single_positive_key(key: ResourceCall | None) -> ResourceCall | None:
    if isinstance(key, RotationCall) and key.opcode in {"zpow", "rz"}:
        assert key.parameter is not None
        if key.opcode == "rz":
            return ControlledZRotationCall(key.parameter, key.precision)
        return ControlledZPowerCall(key.parameter, key.precision)
    if isinstance(key, RotationCall):
        return ControlledRotationCall(key, (1,))
    if not isinstance(key, CliffordCall):
        return None
    if key.opcode == "swap_bit":
        return FixedTCall("fredkin")
    return None


def cirq_gate_decomposition_resource_keys(
    gate: object,
) -> tuple[ResourceCall, ...] | None:
    """Resolve known leaves without retaining Cirq objects in the resource graph."""
    cirq = _cirq()

    def visit(
        value: object, ancestors: tuple[object, ...]
    ) -> tuple[ResourceCall, ...] | None:
        key = cirq_gate_resource_key(value)
        if key is not None:
            return (key,)
        if not isinstance(value, cirq.Gate) or value in ancestors:
            return None
        if any(dimension != 2 for dimension in cirq.qid_shape(value)):
            return None
        operations = decompose_cirq_once(value)
        if operations is None:
            return None
        leaves: list[ResourceCall] = []
        path = (*ancestors, value)
        for operation in operations:
            if not isinstance(operation, cirq.GateOperation) or operation.tags:
                return None
            child = visit(operation.gate, path)
            if child is None:
                return None
            leaves.extend(child)
        return tuple(leaves)

    return visit(gate, ())


def cirq_mapped_gate_resource_key(gate: object) -> ResourceCall | None:
    """Price the source named-gate mapping with its opaque-wrapper atoms.

    Named gate mapping takes precedence. Otherwise the source wrapper prices a
    one/two-qubit Clifford as one Clifford, or an arbitrary single-qubit unitary
    as one rotation, before decomposing. This is an analytical declaration,
    independent of the number of SDK-imported primitives. The source also
    classifies stabilizer channels at this cost boundary; this grants no
    executable body. Unknown decompositions return None, and no SDK object
    enters the returned cost graph.
    """
    cirq = _cirq()

    def visit(value: object, ancestors: tuple[object, ...]) -> ResourceCall | None:
        key = cirq_gate_resource_key(value)
        if key is not None:
            return key
        if not isinstance(value, cirq.Gate) or value in ancestors:
            return None
        if any(dimension != 2 for dimension in cirq.qid_shape(value)):
            return None
        if isinstance(value, cirq.ops.raw_types._InverseCompositeGate):
            original = value._original
            original_key = cirq_gate_resource_key(original)
            if original_key is not None:
                if classify_resource_gate(original_key) == "clifford":
                    return cirq_gate_resource_key(cirq.inverse(original, default=None))
                # Generic source Adjoint is not the named leaf's specialized
                # inverse. Its unavailable or separately declared model must
                # not be inferred from equal unitary cost categories.
                return None
            operations = decompose_cirq_once(original)
            if operations is not None:
                operations = cirq.inverse(operations, default=None)
        else:
            # A generic source control owns its child call graph; a raw SDK
            # controlled-matrix decomposition is a different pricing boundary.
            if isinstance(value, cirq.ControlledGate):
                return None
            width = cirq.num_qubits(value)
            if width <= 2 and cirq.has_stabilizer_effect(value):
                return ComponentCost("stabilizer_atom", width, GateCost(clifford=1))
            if width == 1 and cirq.has_unitary(value):
                return ComponentCost("unitary_atom", 1, GateCost(rotation=1))
            operations = decompose_cirq_once(value)
        if operations is None:
            return None
        children = []
        for operation in operations:
            if not isinstance(operation, cirq.GateOperation) or operation.tags:
                return None
            child = visit(operation.gate, (*ancestors, value))
            if child is None:
                return None
            children.append(child)
        return ResourceSequenceCall(tuple(children))

    return visit(gate, ())
