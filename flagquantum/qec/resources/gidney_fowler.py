# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CCZ2TFactory, MultiFactory, and PhysicalCostModel.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code

"""Physical costs for catalyzed CCZ-to-T factories and a surface-code data block."""

from dataclasses import dataclass
from math import ceil, pow

from ...compiler._component_trace.gate_budget import t_and_ccz_budget
from ...compiler._component_trace.resource_model import ModeledResources
from .data_block import DataBlock, DataLayout
from .physical_cost import PhysicalResourceEstimate, validate_logical_counts
from .physical_parameters import PhysicalParameters


@dataclass(frozen=True)
class GidneyFowlerEstimate(PhysicalResourceEstimate):
    """Physical resources for the catalyzed CCZ-to-T architecture."""


def estimate_gidney_fowler_resources(
    logical: ModeledResources,
    *,
    data_distance: int,
    physical_error: float = 1e-3,
    cycle_time_us: float = 1.0,
    distillation_l1_distance: int = 15,
    distillation_l2_distance: int = 31,
    data_layout: DataLayout = "simple",
    factory_count: int = 1,
) -> GidneyFowlerEstimate:
    """Estimate physical costs from declared logical costs.

    The simple layout uses 50% routing overhead and unlimited magic-state
    consumption. The other layouts can limit throughput. Identical parallel
    factories divide production cycles, rounded up. Both stages use the
    source's eleven-T proxy for each rotation.
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
    hardware = PhysicalParameters(physical_error, cycle_time_us)
    data_block = DataBlock(data_layout, data_distance)
    validate_logical_counts(logical)
    budget = t_and_ccz_budget(logical.gates)
    pairs = ceil(int(budget["n_t"]) / 2)
    ccz_states = int(budget["n_ccz"]) + pairs

    def logical_error(distance: int) -> float:
        return 0.1 * pow(physical_error / 0.01, (distance + 1) / 2)

    l1, l2 = distillation_l1_distance, distillation_l2_distance
    l0_error = physical_error + 100 * logical_error(l1 // 2)
    l1_error = 1000 * logical_error(l1) + 100 * logical_error(l1) + 35 * l0_error**3
    l2_error = 1000 * logical_error(l2) + 28 * l1_error**2
    generation_cycles = ceil((ccz_states * 5.5 + pairs) * max(2 * l1 + 1, l2))
    factory_qubits = 6 * (4 * 8 * 2 * l1**2) + 4 * 8 * 2 * l2**2
    return GidneyFowlerEstimate._from_components(
        logical,
        data_block,
        hardware,
        error_rate_scaler=0.1,
        factory_cycles=generation_cycles,
        factory_qubits=factory_qubits,
        factory_error=l2_error * ccz_states,
        factory_count=factory_count,
    )
