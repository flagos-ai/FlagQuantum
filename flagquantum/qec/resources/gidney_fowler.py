# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CCZ2TFactory, MultiFactory, and PhysicalCostModel.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code

"""Physical costs for catalyzed CCZ-to-T factories and a surface-code data block."""

from dataclasses import asdict, dataclass

from ...compiler._component_trace.resource_model import ModeledResources
from .ccz_to_t_factory import CCZToTFactory
from .data_block import DataBlock, DataLayout
from .error_model import SurfaceCodeScheme
from .magic_state_factory import MagicStateFactory
from .parallel_factory import ParallelFactory
from .physical_cost import (
    PhysicalCostModel,
    PhysicalResourceEstimate,
    validate_logical_counts,
)
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
    if type(factory_count) is not int or factory_count < 1:
        raise ValueError("Factory count must be a positive integer")
    factory: MagicStateFactory = CCZToTFactory(
        distillation_l1_distance, distillation_l2_distance
    )
    if factory_count > 1:
        factory = ParallelFactory(factory, factory_count)
    model = PhysicalCostModel(
        hardware,
        data_block._data_layout(),
        factory,
        SurfaceCodeScheme.gidney_fowler(),
    )
    return GidneyFowlerEstimate(**asdict(model.estimate(logical)))
