"""Qubitize a double-factorization block in its zero-signal basis."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QubitizationWalkOperator.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qubitization/qubitization_walk_operator.py

from collections.abc import Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
    source_uncontrolled_prepare_reflection_cost,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR
from ...dynamic import DynamicCircuit
from .double_factorization_block_encoding import DoubleFactorizationBlockEncoding


@dataclass(frozen=True)
class DoubleFactorizationWalk:
    """Apply the block followed by reflection about its entire zero signal."""

    encoding: DoubleFactorizationBlockEncoding

    @property
    def layout(self) -> RegisterLayout:
        return self.encoding.layout

    @property
    def alpha(self) -> float:
        return self.encoding.outer.normalization

    def requirements(self) -> tuple[int, int]:
        """Reuse clean work and feedback slots across the block and reflection."""
        block_work, block_slots = self.encoding.requirements()
        signal_width = sum(width for _, width in self.encoding.signal_layout)
        reflect_work, reflect_slots = prepare_reflection_requirements(
            signal_width, controlled=False
        )
        return max(block_work, reflect_work), max(block_slots, reflect_slots)

    def source_gate_cost(self) -> GateCost:
        """Count the locked call graph using its four-register signal declaration."""
        encoding = self.encoding
        block = estimate_component_resources(encoding.source_resource_model())
        signal_width = (
            (len(encoding.inner.ranks) - 1).bit_length()
            + (encoding.select.orbitals - 1).bit_length()
            + 2
        )
        identity = ComponentCost("PrepareIdentity", signal_width)
        reflection = estimate_component_resources(
            source_uncontrolled_prepare_reflection_cost(
                signal_width, 0, identity, identity
            )
        )
        return block.gates.plus(reflection.gates, 1)

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundDoubleFactorizationWalk":
        """Bind a walk step or its inverse for circuit.append."""
        logical = unpartition_register(ports, self.layout)
        _validate_wires((*logical, *workspace))
        if type(adjoint) is not bool or len(workspace) != self.requirements()[0]:
            raise ValueError("DF walk requires matching workspace and direction")
        return _BoundDoubleFactorizationWalk(
            self.encoding,
            dict(ports),
            workspace,
            adjoint,
            max((*logical, *workspace)) + 1,
        )


@dataclass(frozen=True)
class _BoundDoubleFactorizationWalk:
    encoding: DoubleFactorizationBlockEncoding
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    adjoint: bool
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError("DF walk needs a DynamicCircuit with sufficient width")
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        selection = unpartition_register(
            {name: self.ports[name] for name, _ in self.encoding.signal_layout},
            self.encoding.signal_layout,
        )
        reflect_work, reflect_slots = prepare_reflection_requirements(
            len(selection), controlled=False
        )
        reflection = prepare_reflection_ir(
            selection,
            self.workspace[:reflect_work],
            tuple(range(start, start + reflect_slots)),
            global_phase=-1,
        )
        block = self.encoding._emit(
            self.ports,
            self.workspace[: self.encoding.requirements()[0]],
            start,
            adjoint=self.adjoint,
        )
        gates = (*reflection, *block) if self.adjoint else (*block, *reflection)
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
