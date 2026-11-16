# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 sparse_state_preparation_via_rotations.py.

"""Append the original sparse rotation preparation examples."""

from dataclasses import dataclass

from flagquantum.algorithms.primitives.rotation_resource_model import (
    sparse_rotation_preparation_cost,
)
from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.core.ir import CircuitIR


@dataclass(frozen=True)
class SparseRotationExample:
    """Bind target then gradient in big-endian order, followed by clean scratch."""

    preparation: SparseRotationPreparation
    target_width: int

    @property
    def data_width(self) -> int:
        return self.target_width + self.preparation.phase_width

    @property
    def qubit_count(self) -> int:
        work, _ = self.preparation.requirements(self.target_width)
        return self.data_width + work

    def resource_model(self) -> ComponentCost:
        return sparse_rotation_preparation_cost(self.preparation, self.target_width)

    def append_to(self, circuit: Circuit) -> Circuit:
        work, slots = self.preparation.requirements(self.target_width)
        gates = self.preparation.emit(
            tuple(range(self.target_width)),
            tuple(range(self.target_width, self.data_width)),
            tuple(range(self.data_width, self.data_width + work)),
            tuple(range(slots)),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def sparse_rotation_example(*, wide: bool = False) -> SparseRotationExample:
    """Return the original five-coefficient fixture with four or six target wires."""
    preparation = SparseRotationPreparation(
        (0, 4, 6, 9, 12),
        (0.70914953, 0.46943701, 0.2297245, 0.32960471, 0.33959273),
        15,
        2,
    )
    return SparseRotationExample(preparation, 6 if wide else 4)


if __name__ == "__main__":
    for wide in (False, True):
        example = sparse_rotation_example(wide=wide)
        circuit = Circuit(example.qubit_count)
        circuit.append(example)
        print(example.target_width, example.qubit_count, example.resource_model())
