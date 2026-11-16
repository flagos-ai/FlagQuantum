# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cirq_interop/t_complexity_protocol.py.

"""Keep legacy Cirq protocol estimates separate from compiler declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...compiler._component_trace.gate_budget import legacy_gate_budget
from ...compiler._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ...compiler._component_trace.resource_generalization import ResourceCall
from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal, ceil
from ...compiler._component_trace.terminal_t_cost import rotation_t_cost
from ._decompose import decompose_cirq_once
from .conversion import _cirq


@dataclass(frozen=True)
class LegacyTComplexity:
    t: SymbolicInt = 0
    clifford: SymbolicInt = 0
    rotations: SymbolicInt = 0

    def __add__(self, other: LegacyTComplexity) -> LegacyTComplexity:
        return LegacyTComplexity(
            self.t + other.t,
            self.clifford + other.clifford,
            self.rotations + other.rotations,
        )

    def __mul__(self, count: int) -> LegacyTComplexity:
        return LegacyTComplexity(
            count * self.t, count * self.clifford, count * self.rotations
        )

    __rmul__ = __mul__

    def asdict(self) -> dict[str, SymbolicInt]:
        return {"t": self.t, "rotations": self.rotations, "clifford": self.clifford}

    rotation_cost = staticmethod(rotation_t_cost)

    def __str__(self) -> str:
        return (
            f"T-count:   {self.t:g}\n"
            f"Rotations: {self.rotations:g}\n"
            f"Cliffords: {self.clifford:g}\n"
        )

    def t_incl_rotations(self, precision: SymbolicReal = 1e-11) -> SymbolicInt:
        return ceil(self.t + rotation_t_cost(precision) * self.rotations)


def legacy_compiler_t_complexity(
    call: ResourceCall, *, annotations: Mapping[ResourceCall, QECGateCost] | None = None
) -> LegacyTComplexity:
    """Fold shared compiler categories using the original legacy factors."""
    gates = query_qec_gate_cost(call, legacy_shims=True, legacy_annotations=annotations)
    return LegacyTComplexity(**legacy_gate_budget(gates))


def cirq_t_complexity(value: Any) -> LegacyTComplexity:
    """Estimate the SDK object using its annotation or known decomposition."""
    cirq = _cirq()
    t_gates = cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False)
    rotations = cirq.Gateset(
        cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZPowGate
    )

    def estimate(item: Any, path: tuple[Any, ...]) -> LegacyTComplexity:
        if isinstance(item, (cirq.AbstractCircuit, cirq.Moment, list)):
            result = LegacyTComplexity()
            for child in item:
                result += estimate(child, path)
            return result
        if not isinstance(item, (cirq.Gate, cirq.Operation)):
            raise TypeError(
                f"Cannot estimate legacy Cirq cost for {type(item).__name__}"
            )
        if isinstance(item, cirq.Operation) and item.gate is not None:
            item = item.gate
        estimator = getattr(item, "_t_complexity_", None)
        if estimator is not None:
            result = estimator()
            if result is not None and result is not NotImplemented:
                return LegacyTComplexity(result.t, result.clifford, result.rotations)
        if isinstance(item, cirq.ClassicallyControlledOperation):
            item = item.without_classical_controls()
        gate = item.gate if isinstance(item, cirq.Operation) else item
        if isinstance(gate, cirq.GlobalPhaseGate):
            return LegacyTComplexity()
        width = cirq.num_qubits(item)
        if width <= 2 and cirq.has_stabilizer_effect(item):
            return LegacyTComplexity(clifford=1)
        if item in t_gates:
            return LegacyTComplexity(t=1)
        if item in rotations or (width == 1 and cirq.has_unitary(item)):
            return LegacyTComplexity(rotations=1)
        if item in path:
            raise TypeError("Cirq cost decomposition contains a cycle")
        children = decompose_cirq_once(item)
        if children is None:
            raise TypeError(
                f"Cirq operation has no legacy cost or decomposition: {item!r}"
            )
        return estimate(children, (*path, item))

    return estimate(value, ())
