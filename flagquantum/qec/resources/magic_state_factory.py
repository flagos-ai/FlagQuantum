# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 surface_code/magic_state_factory.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Physical-cost interface shared by single and parallel distillation factories."""

from abc import ABC, abstractmethod

from ...compiler._component_trace.gate_budget import GateCategories
from ...core.value_math import SymbolicReal
from .error_model import LogicalErrorModel


class MagicStateFactory(ABC):
    """Report footprint, production cycles and cumulative distillation error."""

    @abstractmethod
    def n_physical_qubits(self) -> int:
        """Return physical qubits occupied by the factory."""

    @abstractmethod
    def n_cycles(
        self, n_logical_gates: GateCategories, logical_error_model: LogicalErrorModel
    ) -> int:
        """Return cycles to satisfy the declared magic-state demand."""

    @abstractmethod
    def factory_error(
        self, n_logical_gates: GateCategories, logical_error_model: LogicalErrorModel
    ) -> SymbolicReal:
        """Return the source first-order cumulative distillation error."""
