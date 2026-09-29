"""Driven one-qubit Lindblad evolution with physical amplitude damping."""

from flagquantum.simulation import (
    evolve_density_matrix,
    plan_density_matrix_evolution,
)

# H = (X + 0.2 Z) / 2 in one shared inverse-time unit.
hamiltonian = [
    {"pauli": "X", "coefficient": 0.5, "wires": [0]},
    {"pauli": "Z", "coefficient": 0.1, "wires": [0]},
]
times = [index * 0.05 for index in range(161)]  # 0, ..., 8
collapse_operators = [
    {
        "operator": "amplitude_damping",
        "rate": 0.1,  # A physical inverse-time rate, not a channel probability.
        "wire": 0,
    }
]
observables = [{"name": "z", "pauli": "Z", "wires": [0]}]

request = {
    "hamiltonian": hamiltonian,
    "initial_state": "1",
    "n_wires": 1,
    "times": times,
    "collapse_operators": collapse_operators,
    "observables": observables,
}

# Planning performs the same validation without running the evolution.
plan = plan_density_matrix_evolution(**request)
result = evolve_density_matrix(**request)

print("time points:", plan.n_times)
print("final [P(0), P(1)]:", result.populations[-1].tolist())
print("final <Z>:", result.observables["z"][-1].item())
print("maximum trace drift:", result.maximum_trace_drift)
print("populations bounded:", result.population_bounded)
print(
    "numerics:",
    result.method,
    f"order={result.order}",
    f"step_size={result.step_size}",
    result.precision,
)
