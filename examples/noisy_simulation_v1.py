"""Exact and MPS-trajectory noisy simulation with one NoiseModel."""

import flagquantum as fq
import flagquantum.noise as fqn
import flagquantum.runtime as fqr

circuit = fq.Circuit(2).h(0).cx(0, 1)
noise = (
    fqn.NoiseModel()
    .add("h", fqn.thermal_relaxation_channel(t1=50_000, t2=70_000, duration=35))
    .add("cx", fqn.depolarizing_channel(0.01))
    .add_readout(0, fqn.ReadoutError(((0.98, 0.02), (0.07, 0.93))))
)

# A channel can also be written into the program where it happens. The four
# single-scalar channel opcodes declare the parameter their factory takes, so
# this lowers to the same instruction a matched NoiseModel rule produces.
inline = fq.Circuit(2).h(0).cx(0, 1).depolarizing(0, 0.01).depolarizing(1, 0.01)

# Small-system correctness oracle: exact channel evolution.
exact = fq.run(
    circuit,
    noise_model=noise,
    options=fq.ExecutionOptions(mode="density_matrix"),
    outputs=fq.expectation(fq.Z(0) + fq.Z(1)),
)
exact_z = exact.expectation()

# The same program without the model: `Z(0) + Z(1)` is zero on every Bell state,
# so the population of |00> is the number that shows the channel acting. Each
# wire's depolarizing channel keeps 1 - 2p/3 of its population, so the analytic
# value is 0.5 * (1 - 2p/3)**2 = 0.5 * 0.99333**2 = 0.49336.
inline_p00 = float(
    fq.run(inline, outputs=fq.probabilities()).measurements[0].value.reshape(-1)[0]
)

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
print("inline-channel P(00):", inline_p00)
print("trajectory Z mean:", sampled.expectation_z_mean)
print("trajectory Z standard error:", sampled.statistics.standard_error)
print("converged / stopped early:", sampled.converged, sampled.stopped_early)
