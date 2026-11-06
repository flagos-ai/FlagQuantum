# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CCZ2TFactory, SimpleDataBlock, and PhysicalCostModel.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code

"""Physical costs for a simple data block and a catalyzed CCZ-to-T factory."""

from dataclasses import dataclass
from math import ceil, isfinite, pow

from ...compiler._component_trace.gate_budget import t_and_ccz_budget
from ...compiler._component_trace.resource_model import ModeledResources


@dataclass(frozen=True)
class GidneyFowlerEstimate:
    """Modeled costs under the specified distance and hardware assumptions."""

    cycles: int
    duration_hours: float
    data_block_physical_qubits: int
    factory_physical_qubits: int
    data_error: float
    factory_error: float

    @property
    def physical_qubits(self) -> int:
        return self.data_block_physical_qubits + self.factory_physical_qubits

    @property
    def failure_probability(self) -> float:
        """Return the model's first-order sum of the two error contributions."""
        return self.data_error + self.factory_error

    @property
    def qubit_hours(self) -> float:
        return self.physical_qubits * self.duration_hours


def estimate_gidney_fowler_resources(
    logical: ModeledResources,
    *,
    data_distance: int,
    physical_error: float = 1e-3,
    cycle_time_us: float = 1.0,
    distillation_l1_distance: int = 15,
    distillation_l2_distance: int = 31,
) -> GidneyFowlerEstimate:
    """Estimate the locked factory-limited scenario from declared logical costs.

    The simple data block uses 50% routing overhead and unlimited magic-state
    consumption. Distillation uses the source's eleven-T proxy for each rotation.
    Data and factory distances are supplied as scenario inputs.
    """
    if any(
        type(distance) is not int or distance < minimum
        for distance, minimum in (
            (data_distance, 1),
            (distillation_l1_distance, 2),
            (distillation_l2_distance, 1),
        )
    ):
        raise ValueError("Use positive code distances and at least two for level one")
    if not 0 < physical_error < 0.01:
        raise ValueError("Physical error must lie in (0, 0.01)")
    if not isfinite(cycle_time_us) or cycle_time_us <= 0:
        raise ValueError("Cycle time must be positive and finite")

    gates = logical.gates
    if (
        min(
            logical.peak_logical_qubits,
            gates.t,
            gates.toffoli,
            gates.cswap,
            gates.and_bloq,
            gates.rotation,
        )
        < 0
    ):
        raise ValueError(
            "Logical width and magic-state gate counts must be nonnegative"
        )
    budget = t_and_ccz_budget(gates)
    pairs = ceil(int(budget["n_t"]) / 2)
    ccz_states = int(budget["n_ccz"]) + pairs

    def logical_error(distance: int) -> float:
        return 0.1 * pow(physical_error / 0.01, (distance + 1) / 2)

    l1, l2 = distillation_l1_distance, distillation_l2_distance
    l0_error = physical_error + 100 * logical_error(l1 // 2)
    l1_error = 1000 * logical_error(l1) + 100 * logical_error(l1) + 35 * l0_error**3
    l2_error = 1000 * logical_error(l2) + 28 * l1_error**2
    cycles = ceil((ccz_states * 5.5 + pairs) * max(2 * l1 + 1, l2))

    data_tiles = ceil(1.5 * logical.peak_logical_qubits)
    return GidneyFowlerEstimate(
        cycles=cycles,
        duration_hours=cycle_time_us * cycles / (1_000_000 * 60 * 60),
        data_block_physical_qubits=2 * data_distance**2 * data_tiles,
        factory_physical_qubits=6 * (4 * 8 * 2 * l1**2) + 4 * 8 * 2 * l2**2,
        data_error=data_tiles * cycles * logical_error(data_distance),
        factory_error=l2_error * ccz_states,
    )
