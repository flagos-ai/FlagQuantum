"""Reproduce the grid notebook's resource queries with native models.

Run with python -m examples.algorithms.grid_hamiltonian_resources.
"""

from flagquantum.algorithms.primitives.grid_inverse_sqrt import (
    source_grid_inverse_sqrt_newton_cost,
    source_grid_inverse_sqrt_polynomial_cost,
)
from flagquantum.algorithms.primitives.grid_kinetic import source_grid_kinetic_cost
from flagquantum.algorithms.primitives.grid_potential_resource_model import (
    source_grid_pair_potential_cost,
    source_grid_potential_cost,
)
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    estimate_component_resources,
)
from flagquantum.compiler._component_trace.terminal_t_cost import rotation_t_cost
from flagquantum.qec.resources.estimation import estimate_ftqc_resources


def _print_resources(model: ComponentCost, legacy_model: ComponentCost) -> None:
    report = estimate_ftqc_resources(model)
    counts = legacy_gate_budget(estimate_component_resources(legacy_model).gates)
    synthesized_t = counts["t"] + counts["rotations"] * rotation_t_cost(1e-11)
    print(report.component)
    print("  QEC T equivalents", report.logical.gates.total_t_count())
    print("  Peak logical qubits", report.logical.peak_logical_qubits)
    print("  Legacy gates", counts)
    print("  Legacy T including synthesized rotations", synthesized_t)


def main() -> None:
    kinetic = source_grid_kinetic_cost(2, 21)
    _print_resources(kinetic, kinetic)
    _print_resources(
        source_grid_potential_cost(2, 21),
        source_grid_potential_cost(2, 21, legacy_shims=True),
    )
    _print_resources(
        source_grid_pair_potential_cost(6),
        source_grid_pair_potential_cost(6, legacy_shims=True),
    )
    _print_resources(
        source_grid_inverse_sqrt_polynomial_cost(14, 15, 24),
        source_grid_inverse_sqrt_polynomial_cost(14, 15, 24, legacy_shims=True),
    )
    _print_resources(
        source_grid_inverse_sqrt_newton_cost(14, 15, 24),
        source_grid_inverse_sqrt_newton_cost(14, 15, 24, legacy_shims=True),
    )


if __name__ == "__main__":
    main()
