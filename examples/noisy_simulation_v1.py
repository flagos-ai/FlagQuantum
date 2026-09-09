"""Exact and MPS-trajectory noisy simulation with one NoiseModel."""

import flagquantum as fq
import flagquantum.noise as fqn
import flagquantum.runtime as fqr  # noqa: E402

circuit = fq.Circuit(2).h(0).cx(0, 1)
noise = (
    fqn.NoiseModel()
    .add("h", fqn.thermal_relaxation_channel(t1=50_000, t2=70_000, duration=35))
    .add("cx", fqn.depolarizing_channel(0.01))
    .add_readout(0, fqn.ReadoutError(((0.98, 0.02), (0.07, 0.93))))
)

# Small-system correctness oracle: exact channel evolution.
exact = fq.run(
    circuit,
    noise_model=noise,
    options=fq.ExecutionOptions(mode="density_matrix"),
    measurements=(fq.MeasurementNode("expectation_z", (0, 1)),),
)
exact_z = exact.measurements[0].value

# Low-entanglement scale-out path: sampled MPS quantum trajectories.
sampled = fqr.run_noisy_mps(
    circuit,
    noise,
    trajectories=4096,
    min_trajectories=128,
    target_standard_error=1e-3,
    seed=42,
    retain_trajectories=False,
)

print("noise_model_identity:", noise.identity)
print("exact Z:", exact_z)
print("trajectory Z mean:", sampled.expectation_z_mean)
print("trajectory Z standard error:", sampled.statistics.standard_error)
print("converged / stopped early:", sampled.converged, sampled.stopped_early)
