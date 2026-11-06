# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, commit bcb17bb25360ed68ce3bc084645db3e929799389.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code

"""Beverland resource estimates from declared logical component costs."""

from dataclasses import dataclass, replace
from math import ceil, isfinite, log, log2, sqrt

from ...compiler._component_trace.gate_budget import beverland_gate_budget
from ...compiler._component_trace.resource_model import GateCost, ModeledResources

_ERROR_RATE_SCALER = 0.03
_ERROR_RATE_THRESHOLD = 0.01


@dataclass(frozen=True)
class BeverlandEstimate:
    """Time-step lower bound, magic-state demand and fast data block footprint."""

    minimum_time_steps: int
    t_states: int
    discrete_logical_gates: GateCost
    data_block_tiles: int
    distance_budget_time_steps: float
    code_distance: int | None
    data_block_physical_qubits: int | None


def estimate_beverland_resources(
    logical: ModeledResources,
    *,
    error_budget: float,
    physical_error: float,
    rotation_layers: int | None = None,
    time_steps: float | None = None,
) -> BeverlandEstimate:
    """Apply the source's logarithmic rotation and fast data block models.

    The code distance uses the minimum time unless a time-step scenario is
    supplied. An empty zero-step scenario has no code distance. The data block
    uses two physical qubits per squared distance per tile. Factory footprint,
    factory throughput and hardware cycle duration are outside this report.
    """
    if not 0 < error_budget <= 1:
        raise ValueError("Error budget must lie in (0, 1]")
    if not 0 < physical_error < _ERROR_RATE_THRESHOLD:
        raise ValueError("Physical error must lie in (0, 0.01)")
    if rotation_layers is not None and (
        type(rotation_layers) is not int or rotation_layers < 0
    ):
        raise ValueError("Rotation layers must be a nonnegative integer")
    if time_steps is not None and (not isfinite(time_steps) or time_steps <= 0):
        raise ValueError("Time steps must be positive and finite")
    if logical.peak_logical_qubits < 0:
        raise ValueError("Peak logical qubits must be nonnegative")

    gates = logical.gates
    counts = beverland_gate_budget(gates)
    rotations, measurements = gates.rotation, gates.measurement
    toffoli_like = int(counts["Tof"])
    if min(measurements, rotations, gates.t, toffoli_like) < 0:
        raise ValueError("Beverland gate budget inputs must be nonnegative")
    minimum = measurements + rotations + gates.t + 3 * toffoli_like
    discrete = replace(gates, rotation=0)
    synthesis_budget = error_budget / 3
    if rotations:
        per_rotation_error = synthesis_budget / rotations
        if per_rotation_error == 0:
            raise ValueError("Per-rotation error budget underflowed")
        per_rotation = ceil(-0.53 * log2(per_rotation_error) + 5.3)
        layers = rotations if rotation_layers is None else rotation_layers
        minimum += layers * per_rotation
        discrete = discrete.plus(GateCost(t=per_rotation), rotations)

    width = logical.peak_logical_qubits
    tiles = ceil(2 * width + sqrt(8 * width) + 1)
    budget_steps = minimum if time_steps is None else time_steps
    distance: int | None = None
    physical_qubits: int | None = None
    if budget_steps:
        error_per_tile_step = error_budget / (3 * tiles * budget_steps)
        if error_per_tile_step / _ERROR_RATE_SCALER == 0:
            raise ValueError("Error budget per tile and time step underflowed")
        rounds = log(error_per_tile_step / _ERROR_RATE_SCALER) / log(
            physical_error / _ERROR_RATE_THRESHOLD
        )
        distance = max(3, 2 * ceil(rounds) - 1)
        physical_qubits = tiles * 2 * distance**2

    return BeverlandEstimate(
        minimum,
        discrete.total_t_count(),
        discrete,
        tiles,
        budget_steps,
        distance,
        physical_qubits,
    )
