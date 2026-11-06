# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, commit bcb17bb25360ed68ce3bc084645db3e929799389.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code

"""Beverland physical architecture with noisy 15-to-1 distillation."""

from ...compiler._component_trace.resource_model import ModeledResources
from .data_block import DataBlock, DataLayout
from .fifteen_to_one import FifteenToOneFactory
from .physical_cost import PhysicalResourceEstimate, validate_logical_counts
from .physical_parameters import PhysicalParameters

_BEVERLAND_ERROR_SCALER = 0.03
_DEFAULT_FACTORY = FifteenToOneFactory(9, 3, 3)
_DEFAULT_HARDWARE = PhysicalParameters.beverland()


def estimate_beverland_physical_resources(
    logical: ModeledResources,
    *,
    data_distance: int,
    data_layout: DataLayout = "compact",
    factory: FifteenToOneFactory = _DEFAULT_FACTORY,
    hardware: PhysicalParameters = _DEFAULT_HARDWARE,
    factory_count: int = 1,
) -> PhysicalResourceEstimate:
    """Apply the source's data layout and 15-to-1 factory architecture.

    The source converts all logical gates in the factory budget to T equivalents.
    The data block consumes T and CCZ states using its own source budget.
    """
    block = DataBlock(data_layout, data_distance)
    validate_logical_counts(logical)
    estimate = factory.estimate(
        logical.gates,
        physical_error=hardware.physical_error,
        error_rate_scaler=_BEVERLAND_ERROR_SCALER,
    )
    return PhysicalResourceEstimate._from_components(
        logical,
        block,
        hardware,
        error_rate_scaler=_BEVERLAND_ERROR_SCALER,
        factory_cycles=estimate.cycles,
        factory_qubits=estimate.physical_qubits,
        factory_error=estimate.factory_error,
        factory_count=factory_count,
    )
