# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, commit bcb17bb25360ed68ce3bc084645db3e929799389.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code

"""Beverland resource estimates from declared logical component costs."""

from dataclasses import dataclass, replace
from math import ceil, isfinite

from ...compiler._component_trace.gate_budget import beverland_gate_budget
from ...compiler._component_trace.resource_model import GateCost, ModeledResources
from .data_block import fast_data_block_tiles
from .error_model import SurfaceCodeScheme
from .rotation_synthesis import BEVERLAND_ROTATION_COST, RotationCostModel

_SCHEME = SurfaceCodeScheme.beverland()


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


def minimum_time_steps(
    logical: ModeledResources,
    *,
    error_budget: float,
    rotation_model: RotationCostModel,
    rotation_layers: int | None = None,
) -> int:
    """Evaluate the source measurement and non-Clifford time-step lower bound."""
    counts = beverland_gate_budget(logical.gates)
    minimum = int(counts["meas"] + counts["R"] + counts["T"] + 3 * counts["Tof"])
    rotations = logical.gates.rotation
    if rotations > 0:
        per_rotation_budget = (error_budget / 3) / rotations
        cost = rotation_model.rotation_cost(per_rotation_budget).plus(
            rotation_model.preparation_overhead(per_rotation_budget), 1
        )
        layers = rotations if rotation_layers is None else rotation_layers
        minimum += ceil(layers * (cost.t + 4 * cost.toffoli))
    return minimum


def code_distance(
    logical: ModeledResources,
    *,
    error_budget: float,
    time_steps: float,
    qec_scheme: SurfaceCodeScheme,
    physical_error: float,
) -> int:
    """Invert the source one-third storage budget over tiles and time steps."""
    tiles = fast_data_block_tiles(logical.peak_logical_qubits)
    return qec_scheme.code_distance_from_budget(
        physical_error, error_budget / (3 * tiles * time_steps)
    )


def discrete_logical_gates(
    logical: ModeledResources,
    *,
    synthesis_budget: float,
    rotation_model: RotationCostModel,
) -> GateCost:
    """Replace rotations with synthesized gates plus one preparation overhead."""
    rotations = logical.gates.rotation
    result = replace(logical.gates, rotation=0)
    if rotations > 0:
        result = result.plus(rotation_model.preparation_overhead(synthesis_budget), 1)
        result = result.plus(
            rotation_model.rotation_cost(synthesis_budget / rotations), rotations
        )
    return result


def t_states(
    logical: ModeledResources,
    *,
    error_budget: float,
    rotation_model: RotationCostModel,
) -> int:
    """Convert the complete source discrete gate demand to T equivalents."""
    return discrete_logical_gates(
        logical, synthesis_budget=error_budget / 3, rotation_model=rotation_model
    ).total_t_count()


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
    if not 0 < physical_error < _SCHEME.error_rate_threshold:
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
    if rotations and (error_budget / 3) / rotations == 0:
        raise ValueError("Per-rotation error budget underflowed")
    minimum = minimum_time_steps(
        logical,
        error_budget=error_budget,
        rotation_model=BEVERLAND_ROTATION_COST,
        rotation_layers=rotation_layers,
    )
    discrete = discrete_logical_gates(
        logical,
        synthesis_budget=error_budget / 3,
        rotation_model=BEVERLAND_ROTATION_COST,
    )

    width = logical.peak_logical_qubits
    tiles = fast_data_block_tiles(width)
    budget_steps = minimum if time_steps is None else time_steps
    distance: int | None = None
    physical_qubits: int | None = None
    if budget_steps:
        error_per_tile_step = error_budget / (3 * tiles * budget_steps)
        if error_per_tile_step / _SCHEME.error_rate_scaler == 0:
            raise ValueError("Error budget per tile and time step underflowed")
        distance = code_distance(
            logical,
            error_budget=error_budget,
            time_steps=budget_steps,
            qec_scheme=_SCHEME,
            physical_error=physical_error,
        )
        physical_qubits = tiles * _SCHEME.physical_qubits(distance)

    return BeverlandEstimate(
        minimum,
        discrete.total_t_count(),
        discrete,
        tiles,
        budget_steps,
        distance,
        physical_qubits,
    )
