# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PhysicalCostModel, PhysicalCostsSummary, and MultiFactory.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code

"""Combine a data layout and magic-state production into physical costs."""

from dataclasses import dataclass
from functools import cached_property, lru_cache

from ...compiler._component_trace.resource_model import ModeledResources
from .ccz_to_t_factory import CCZToTFactory
from .data_block import DataBlock, DataLayout
from .data_layout import RoutingDataLayout, SurfaceCodeDataLayout
from .error_model import LogicalErrorModel, SurfaceCodeScheme
from .fifteen_to_one import FifteenToOneFactory
from .magic_state_factory import MagicStateFactory
from .physical_parameters import PhysicalParameters


@dataclass(frozen=True)
class PhysicalResourceEstimate:
    """Time, footprint and first-order errors under a physical scenario."""

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


@dataclass(frozen=True)
class PhysicalCostModel:
    """Compose hardware, data layout, factory and code-suppression assumptions."""

    physical_params: PhysicalParameters
    data_block: SurfaceCodeDataLayout
    factory: MagicStateFactory
    qec_scheme: SurfaceCodeScheme

    @cached_property
    def logical_error_model(self) -> LogicalErrorModel:
        """Bind the declared physical error to the selected QEC scheme."""
        return LogicalErrorModel(self.physical_params.physical_error, self.qec_scheme)

    @lru_cache
    def estimate(self, logical: ModeledResources) -> PhysicalResourceEstimate:
        """Aggregate complete physical costs for the declared logical workload."""
        generation_cycles = self.factory.n_cycles(
            logical.gates, self.logical_error_model
        )
        consumption_cycles = self.data_block.consumption_cycles(logical.gates)
        cycles = int(max(generation_cycles, consumption_cycles))
        factory_qubits = self.factory.n_physical_qubits()
        data_qubits = self.data_block.physical_qubits(logical.peak_logical_qubits)
        factory_error = float(
            self.factory.factory_error(logical.gates, self.logical_error_model)
        )
        data_error = self.data_block.storage_error(
            logical.peak_logical_qubits, cycles, self.logical_error_model
        )
        return PhysicalResourceEstimate(
            cycles=cycles,
            duration_hours=self.physical_params.cycle_time_us
            * cycles
            / (1_000_000 * 60 * 60),
            data_block_physical_qubits=data_qubits,
            factory_physical_qubits=factory_qubits,
            data_error=data_error,
            factory_error=factory_error,
        )

    def n_cycles(self, logical: ModeledResources) -> int:
        return self.estimate(logical).cycles

    def duration_hours(self, logical: ModeledResources) -> float:
        return self.estimate(logical).duration_hours

    def n_physical_qubits(self, logical: ModeledResources) -> int:
        return self.estimate(logical).physical_qubits

    def error(self, logical: ModeledResources) -> float:
        return self.estimate(logical).failure_probability

    @classmethod
    def gidney_fowler(cls, data_distance: int) -> "PhysicalCostModel":
        """Construct the complete source Gidney-Fowler defaults."""
        return cls(
            PhysicalParameters.gidney_fowler(),
            RoutingDataLayout(data_distance),
            CCZToTFactory(),
            SurfaceCodeScheme.gidney_fowler(),
        )

    @classmethod
    def beverland(
        cls,
        data_distance: int,
        data_layout: DataLayout = "compact",
        factory_distances: tuple[int, int, int] = (9, 3, 3),
    ) -> "PhysicalCostModel":
        """Construct the three supported source data layouts and 15-to-1 factory."""
        if data_layout not in ("fast", "compact", "intermediate"):
            raise ValueError(f"Unknown data block {data_layout!r}")
        return cls(
            PhysicalParameters.beverland(),
            DataBlock(data_layout, data_distance)._data_layout(),
            FifteenToOneFactory(*factory_distances),
            SurfaceCodeScheme.beverland(),
        )


def validate_logical_counts(logical: ModeledResources) -> None:
    """Reject negative inputs to the physical magic-state budget."""
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
