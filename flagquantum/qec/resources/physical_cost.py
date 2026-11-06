# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PhysicalCostModel, PhysicalCostsSummary, and MultiFactory.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code

"""Combine a data layout and magic-state production into physical costs."""

from dataclasses import dataclass
from math import ceil, pow
from typing import TypeVar

from ...compiler._component_trace.resource_model import ModeledResources
from .data_block import DataBlock
from .physical_parameters import PhysicalParameters

_Estimate = TypeVar("_Estimate", bound="PhysicalResourceEstimate")


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

    @classmethod
    def _from_components(
        cls: type[_Estimate],
        logical: ModeledResources,
        block: DataBlock,
        hardware: PhysicalParameters,
        *,
        error_rate_scaler: float,
        factory_cycles: int,
        factory_qubits: int,
        factory_error: float,
        factory_count: int,
    ) -> _Estimate:
        if type(factory_count) is not int or factory_count < 1:
            raise ValueError("Factory count must be a positive integer")
        if factory_count > 1:
            factory_cycles = ceil(factory_cycles / factory_count)
        cycles = max(factory_cycles, block.consumption_cycles(logical.gates))
        width = logical.peak_logical_qubits
        logical_error = error_rate_scaler * pow(
            hardware.physical_error / 0.01, (block.distance + 1) / 2
        )
        return cls(
            cycles=cycles,
            duration_hours=hardware.cycle_time_us * cycles / (1_000_000 * 60 * 60),
            data_block_physical_qubits=block.physical_qubits(width),
            factory_physical_qubits=factory_count * factory_qubits,
            data_error=block.storage_error(width, cycles, logical_error),
            factory_error=factory_error,
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
