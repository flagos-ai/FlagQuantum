# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 surface_code/rotation_cost_model.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Analytical logical costs for rotation synthesis and reusable preparation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil, log2

from ...compiler._component_trace.resource_model import GateCost


class RotationCostModel(ABC):
    """Report per-rotation costs separately from preparation overhead."""

    @abstractmethod
    def rotation_cost(self, error_budget: float) -> GateCost:
        """Return the source gate categories for approximating one rotation."""

    @abstractmethod
    def preparation_overhead(self, error_budget: float) -> GateCost:
        """Return the preparation cost at the supplied error budget."""


@dataclass(frozen=True)
class LogarithmicRotationCost(RotationCostModel):
    """Ceil the affine negative-logarithm T cost with zero preparation."""

    slope: float
    overhead: float
    gateset: str | None = None

    def rotation_cost(self, error_budget: float) -> GateCost:
        return GateCost(t=ceil(-self.slope * log2(error_budget) + self.overhead))

    def preparation_overhead(self, error_budget: float) -> GateCost:
        return GateCost()


@dataclass(frozen=True)
class PhaseGradientRotationCost(RotationCostModel):
    """Charge fixed-width Toffolis and a separately synthesized gradient state."""

    bitsize: int
    overhead_rotation_cost: RotationCostModel

    def rotation_cost(self, error_budget: float) -> GateCost:
        return GateCost(toffoli=max(self.bitsize - 2, 0))

    def preparation_overhead(self, error_budget: float) -> GateCost:
        return GateCost().plus(
            self.overhead_rotation_cost.rotation_cost(error_budget / self.bitsize),
            self.bitsize,
        )


BEVERLAND_ROTATION_COST = LogarithmicRotationCost(0.53, 5.3, "Clifford+T")
SEVEN_BIT_PHASE_GRADIENT_COST = PhaseGradientRotationCost(7, BEVERLAND_ROTATION_COST)
