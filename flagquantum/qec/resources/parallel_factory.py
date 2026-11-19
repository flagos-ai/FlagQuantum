# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 surface_code/multi_factory.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Parallel replicas of one distillation factory with shared total demand."""

from dataclasses import dataclass
from math import ceil

from ...compiler._component_trace.gate_budget import GateCategories
from ...core.value_math import SymbolicReal
from .error_model import LogicalErrorModel
from .magic_state_factory import MagicStateFactory


@dataclass(frozen=True)
class ParallelFactory(MagicStateFactory):
    """Scale footprint and ceil-divide cycles, retaining cumulative error."""

    base_factory: MagicStateFactory
    n_factories: int

    def n_physical_qubits(self) -> int:
        return self.base_factory.n_physical_qubits() * self.n_factories

    def n_cycles(
        self, n_logical_gates: GateCategories, logical_error_model: LogicalErrorModel
    ) -> int:
        return ceil(
            self.base_factory.n_cycles(n_logical_gates, logical_error_model)
            / self.n_factories
        )

    def factory_error(
        self, n_logical_gates: GateCategories, logical_error_model: LogicalErrorModel
    ) -> SymbolicReal:
        return self.base_factory.factory_error(n_logical_gates, logical_error_model)
