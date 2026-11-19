# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 surface_code/ccz2t_factory.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Three-level catalyzed CCZ-to-two-T distillation cost model."""

from dataclasses import dataclass
from math import ceil

from ...compiler._component_trace.gate_budget import GateCategories, t_and_ccz_budget
from ...core.value_math import SymbolicReal
from .error_model import LogicalErrorModel
from .magic_state_factory import MagicStateFactory


@dataclass(frozen=True)
class CCZToTFactory(MagicStateFactory):
    """Use the original level-one and level-two code distances and schedules."""

    level_one_distance: int = 15
    level_two_distance: int = 31

    def l0_state_injection_error(self, error_model: LogicalErrorModel) -> float:
        return error_model.physical_error

    def l0_topo_error_t_gate(self, error_model: LogicalErrorModel) -> float:
        return 100 * error_model(code_distance=self.level_one_distance // 2)

    def l0_error(self, error_model: LogicalErrorModel) -> float:
        return self.l0_state_injection_error(error_model) + self.l0_topo_error_t_gate(
            error_model
        )

    def l1_topo_error_factory(self, error_model: LogicalErrorModel) -> float:
        return 1000 * error_model(code_distance=self.level_one_distance)

    def l1_topo_error_t_gate(self, error_model: LogicalErrorModel) -> float:
        return 100 * error_model(code_distance=self.level_one_distance)

    def l1_distillation_error(self, error_model: LogicalErrorModel) -> float:
        return 35 * self.l0_error(error_model) ** 3

    def l1_error(self, error_model: LogicalErrorModel) -> float:
        return (
            self.l1_topo_error_factory(error_model)
            + self.l1_topo_error_t_gate(error_model)
            + self.l1_distillation_error(error_model)
        )

    def l2_error(self, error_model: LogicalErrorModel) -> float:
        return (
            1000 * error_model(self.level_two_distance)
            + 28 * self.l1_error(error_model) ** 2
        )

    def n_physical_qubits(self) -> int:
        level_one = 4 * 8 * 2 * self.level_one_distance**2
        level_two = 4 * 8 * 2 * self.level_two_distance**2
        return 6 * level_one + level_two

    def factory_error(
        self, n_logical_gates: GateCategories, logical_error_model: LogicalErrorModel
    ) -> SymbolicReal:
        counts = t_and_ccz_budget(n_logical_gates)
        total_ccz_states = counts["n_ccz"] + ceil(counts["n_t"] / 2)
        return self.l2_error(logical_error_model) * total_ccz_states

    def n_cycles(
        self, n_logical_gates: GateCategories, logical_error_model: LogicalErrorModel
    ) -> int:
        distance = max(2 * self.level_one_distance + 1, self.level_two_distance)
        counts = t_and_ccz_budget(n_logical_gates)
        ccz_states = counts["n_ccz"] + ceil(counts["n_t"] / 2)
        catalyzations = ceil(counts["n_t"] / 2)
        return ceil((ccz_states * 5.5 + catalyzations) * distance)
