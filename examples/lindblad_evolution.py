"""Driven one-qubit Lindblad evolution with physical amplitude damping."""

import torch

import flagquantum as fq
import flagquantum.lindblad as lindblad

# H = (X + 0.2 Z) / 2 in one shared inverse-time unit.
hamiltonian = 0.5 * fq.X(0) + 0.1 * fq.Z(0)
times = torch.linspace(0.0, 8.0, 161, dtype=torch.float64)

request = {
    # The rate is physical inverse time, not a channel probability.
    "collapse_operators": [lindblad.amplitude_damping(rate=0.1, wire=0)],
    "observables": {"z": fq.Z(0)},
}

# Planning performs the same validation without running the evolution.
plan = lindblad.plan(hamiltonian, "1", times, **request)
result = lindblad.solve(hamiltonian, "1", times, **request)

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
