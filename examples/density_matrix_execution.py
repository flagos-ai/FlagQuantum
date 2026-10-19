"""Exact density-matrix execution: the oracle, its invariants, and its gradient.

Run with ``python -m examples.density_matrix_execution``.
"""

import math

import torch

import flagquantum as fq

# One option selects the exact route. It stores a dense 2**n by 2**n matrix, so
# it is the small-system correctness oracle rather than the scale-out path, and
# `mode="auto"` selects it exactly when the program carries a channel.
bell = fq.Circuit(2).h(0).cx(0, 1)
exact = fq.run(
    bell,
    options=fq.ExecutionOptions(mode="density_matrix", shots=64, seed=7),
    outputs=[
        fq.expectation(fq.Z(0) @ fq.Z(1), name="zz"),
        fq.probabilities(),
        fq.counts(),
    ],
)
print("state_mode:", exact.plan.state_mode)
print("<Z(0) Z(1)>:", float(exact.expectations[0][0]))
print("probabilities:", exact.probabilities.tolist())
print("counts at 64 shots, seed 7:", exact.counts)

# The matrix itself is inspectable and it is a physical state: the trace is one,
# and the purity is one because a Bell pair is pure. Wire 0 is the highest-order
# index, so the |00> population is element [0, 0].
rho = bell.density_matrix()[0]
print("density matrix:", tuple(rho.shape), rho.dtype)
print("trace:", complex(torch.trace(rho)))
print("purity:", float(torch.trace(rho @ rho).real))

# A channel makes the same matrix mixed. Depolarizing with probability p keeps
# (1 - p) of the state and applies X, Y or Z with probability p/3 each. On a
# Bell pair, X|Phi+> and Y|Phi+> are the two |Psi>+/- states, which have no |00>
# population, and Z|Phi+> is |Phi->, which keeps it, so the surviving |00>
# population is exactly 0.5 * (1 - p + p/3) = 0.5 * (1 - 2p/3).
p = 0.2
noisy = fq.Circuit(2).h(0).cx(0, 1).depolarizing(0, p)
noisy_rho = noisy.density_matrix()[0]
mixed_p00 = float(noisy_rho[0, 0].real)
expected_p00 = 0.5 * (1 - 2 * p / 3)
print("P(00) without a channel:", float(exact.probabilities[0, 0]))
print(f"P(00) with depolarizing {p} on wire 0:", mixed_p00)
print("analytic 0.5 * (1 - 2p/3):", expected_p00)
print("difference:", abs(mixed_p00 - expected_p00))
print("purity with the channel:", float(torch.trace(noisy_rho @ noisy_rho).real))
print("trace with the channel:", complex(torch.trace(noisy_rho)))

# The route is differentiable in ordinary torch. d<Z(0)>/dtheta for RY(theta) on
# |0> is -sin(theta); complex128 reproduces it bit for bit, while the default
# complex64 lands about 1e-8 away.
for precision in ("complex64", "complex128"):
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    run = fq.run(
        fq.Circuit(1).ry(0, theta),
        options=fq.ExecutionOptions(mode="density_matrix", precision=precision),
        outputs=fq.expectation(fq.Z(0)),
    )
    (gradient,) = torch.autograd.grad(run.expectations[0].sum(), theta)
    print(
        f"d<Z(0)>/dtheta in {precision}:",
        float(gradient),
        "analytic:",
        -math.sin(0.7),
    )
