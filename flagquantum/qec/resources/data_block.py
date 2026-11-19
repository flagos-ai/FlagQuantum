# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 DataBlock and its four layouts.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code/data_block.py

"""Space and magic-state consumption for surface-code data layouts."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ...compiler._component_trace.gate_budget import t_and_ccz_budget
from ...compiler._component_trace.resource_model import GateCost
from .data_layout import (
    RoutingDataLayout,
    SingleHallwayDataLayout,
    SquareDataLayout,
    SurfaceCodeDataLayout,
    TwoRowDataLayout,
)

DataLayout = Literal["simple", "compact", "intermediate", "fast"]

_LAYOUTS: dict[DataLayout, Callable[[int], SurfaceCodeDataLayout]] = {
    "simple": RoutingDataLayout,
    "compact": SingleHallwayDataLayout,
    "intermediate": TwoRowDataLayout,
    "fast": SquareDataLayout,
}


def fast_data_block_tiles(logical_qubits: int) -> int:
    return SquareDataLayout.tile_count(logical_qubits)


@dataclass(frozen=True)
class DataBlock:
    """A fixed-distance data block with the source layout and consumption model."""

    layout: DataLayout
    distance: int

    def __post_init__(self) -> None:
        if self.layout not in _LAYOUTS:
            raise ValueError(f"Unknown data layout {self.layout!r}")
        if type(self.distance) is not int or self.distance < 1:
            raise ValueError("Data code distance must be a positive integer")

    def _data_layout(self) -> SurfaceCodeDataLayout:
        return _LAYOUTS[self.layout](self.distance)

    def tiles(self, logical_qubits: int) -> int:
        if type(logical_qubits) is not int or logical_qubits < 0:
            raise ValueError("Logical width must be a nonnegative integer")
        return self._data_layout().tiles(logical_qubits)

    def physical_qubits(self, logical_qubits: int) -> int:
        return 2 * self.distance**2 * self.tiles(logical_qubits)

    def consumption_cycles(self, gates: GateCost) -> int:
        counts = t_and_ccz_budget(gates)
        states = int(counts["n_t"] + counts["n_ccz"])
        return int(self._data_layout().consumption_cycles(GateCost(t=states)))

    def storage_error(
        self, logical_qubits: int, cycles: int, logical_error_per_tile_cycle: float
    ) -> float:
        return self.tiles(logical_qubits) * cycles * logical_error_per_tile_cycle
