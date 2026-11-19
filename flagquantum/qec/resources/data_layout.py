# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 surface_code/data_block.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Surface-code layout geometry, state consumption and storage-error models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil, sqrt

from ...compiler._component_trace.gate_budget import GateCategories, t_and_ccz_budget
from ...core.value_math import SymbolicInt
from .error_model import LogicalErrorModel, SurfaceCodeScheme


class SurfaceCodeDataLayout(ABC):
    """Model data tiles separately from magic-state factory resources."""

    @property
    @abstractmethod
    def distance(self) -> int:
        """Return the code distance protecting the stored data."""

    @abstractmethod
    def tiles(self, logical_qubits: int) -> int:
        """Return storage and routing tiles for the algorithm width."""

    @property
    @abstractmethod
    def steps_per_magic_state(self) -> int:
        """Return distance-sized surface-code steps per consumed state."""

    def consumption_cycles(self, gates: GateCategories) -> SymbolicInt:
        """Preserve the source T-plus-CCZ demand, including symbolic counts."""
        counts = t_and_ccz_budget(gates)
        steps = (counts["n_t"] + counts["n_ccz"]) * self.steps_per_magic_state
        return self.distance * steps

    def physical_qubits(self, logical_qubits: int) -> int:
        """Return two distance-squared physical qubits per tile."""
        return 2 * self.distance**2 * self.tiles(logical_qubits)

    def storage_error(
        self, logical_qubits: int, cycles: int, logical_error_model: LogicalErrorModel
    ) -> float:
        """Apply the shared error model to the data spacetime volume."""
        spacetime_volume = self.tiles(logical_qubits) * cycles
        return spacetime_volume * logical_error_model(self.distance)


@dataclass(frozen=True, slots=True)
class RoutingDataLayout(SurfaceCodeDataLayout):
    """Reserve a proportional routing area with unrestricted consumption."""

    distance: int
    routing_overhead: float = 0.5

    def tiles(self, logical_qubits: int) -> int:
        return ceil((1 + self.routing_overhead) * logical_qubits)

    @property
    def steps_per_magic_state(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class SingleHallwayDataLayout(SurfaceCodeDataLayout):
    """Place data on both sides of one hallway, consuming in nine steps."""

    distance: int

    def tiles(self, logical_qubits: int) -> int:
        return ceil(1.5 * logical_qubits)

    @property
    def steps_per_magic_state(self) -> int:
        return 9


@dataclass(frozen=True, slots=True)
class TwoRowDataLayout(SurfaceCodeDataLayout):
    """Use one data row and one routing row, consuming in five steps."""

    distance: int

    def tiles(self, logical_qubits: int) -> int:
        return ceil(2 * logical_qubits)

    @property
    def steps_per_magic_state(self) -> int:
        return 5


@dataclass(frozen=True, slots=True)
class SquareDataLayout(SurfaceCodeDataLayout):
    """Alternate square data and routing columns for one-step consumption."""

    distance: int

    @staticmethod
    def tile_count(logical_qubits: int) -> int:
        """Evaluate geometry independently of the selected code distance."""
        return ceil(2 * logical_qubits + sqrt(8 * logical_qubits) + 1)

    def tiles(self, logical_qubits: int) -> int:
        return self.tile_count(logical_qubits)

    @property
    def steps_per_magic_state(self) -> int:
        return 1

    @classmethod
    def from_error_budget(
        cls,
        error_budget: float,
        logical_qubits: int,
        qec_scheme: SurfaceCodeScheme,
        physical_error: float,
    ) -> "SquareDataLayout":
        """Select the original odd distance using the per-tile error budget."""
        tiles = SquareDataLayout.tile_count(logical_qubits)
        distance = qec_scheme.code_distance_from_budget(
            physical_error, error_budget / tiles
        )
        return cls(distance=distance)
