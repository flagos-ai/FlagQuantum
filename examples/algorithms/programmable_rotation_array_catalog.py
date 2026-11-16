# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 programmable_rotation_gate_array.ipynb.

"""Native bindings for the locked four-entry programmable rotation array."""

from dataclasses import dataclass, fields
from math import pi
from typing import cast

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledZPowerCall,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler.registers.lookup.qrom_resource_model import source_qrom_cost
from flagquantum.compiler.registers.lookup.source_qrom import source_qrom_workspace_size
from flagquantum.compiler.registers.phase.programmable_rotation_array import (
    _rotation_batches,
    programmable_rotation_array_ir,
)
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

_ANGLES = ((1, 2, 3, 4),)
_SELECTION, _LOAD, _TARGET = (0, 1), (2, 3), (4,)
_NAME = "ProgrammableRotationGateArray"


@dataclass(frozen=True)
class ProgrammableRotationArrayExample:
    measured_rotations: bool = True

    @property
    def signature(self) -> Signature:
        return Signature(
            (
                Register("selection", bounded_unsigned_register_type(2, 4)),
                Register("kappa_load_target", opaque_register_type(2)),
                Register("rotations_target", opaque_register_type(1)),
            )
        )

    @property
    def workspace_size(self) -> int:
        return max(
            source_qrom_workspace_size(0, (2,), (delta,))
            for delta, _ in _rotation_batches(_ANGLES, 2)
        )

    @property
    def qubit_count(self) -> int:
        return 5 + self.workspace_size

    def resource_model(self) -> ComponentCost:
        calls: list[CostCall] = []
        for delta, columns in _rotation_batches(_ANGLES, 2):
            qrom = source_qrom_cost(0, (2,), (4,), ((1, 2),), (delta,))
            calls.append(CostCall(qrom, bystander_qubits=1))
            for _, power in columns:
                exponent = 2.0 ** (-1 - power)
                gate = query_qec_gate_cost(ControlledZPowerCall(exponent))
                cost = ComponentCost(
                    f"CZPowGate({exponent})",
                    3,
                    GateCost(
                        **{
                            f.name: cast(int, getattr(gate, f.name))
                            for f in fields(gate)
                        }
                    ),
                )
                calls.append(CostCall(cost, bystander_qubits=3))
        return ComponentCost(_NAME, 5, calls=tuple(calls))

    def call_graph(self) -> SymbolicCallGraph[str]:
        calls = {f"QROM {delta}": 1 for delta, _ in _rotation_batches(_ANGLES, 2)}
        for _, columns in _rotation_batches(_ANGLES, 2):
            for _, power in columns:
                calls[f"CZPowGate({2.0 ** (-1 - power)})"] = 1
        return symbolic_call_graph(_NAME, {_NAME: calls}, max_depth=1)

    def append_to(self, circuit: Circuit) -> Circuit:
        work = tuple(range(5, self.qubit_count))

        def rotation(
            control: int, target: tuple[int, ...], power: int
        ) -> tuple[Instruction, ...]:
            exponent = 2.0 ** (-1 - power)
            if self.measured_rotations:
                return controlled_z_power_ir(control, target[0], exponent, work[0], 0)
            return (
                Instruction("cphase", (control, *target), {"theta": pi * exponent}),
            )

        gates = programmable_rotation_array_ir(
            _SELECTION,
            _LOAD,
            _TARGET,
            _ANGLES,
            work,
            tuple(range(len(work))),
            rotation=rotation,
            interleaved=(),
            rotation_workspace=work[:1] if self.measured_rotations else (),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))
