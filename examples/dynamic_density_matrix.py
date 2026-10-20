"""Mid-circuit measurement and feedback, read off an exact density matrix.

Run with ``python -m examples.dynamic_density_matrix``.

``fq.experimental.dynamic.run_dynamic`` answers these programs by drawing
trajectories, so its numbers carry a sampling error and its ``final_states`` are
per-trajectory states rather than the state a reader who ignores the classical
register holds. ``DynamicCircuit.density_matrix()`` answers the same programs
exactly: it keeps one dense density matrix per classical record, applies the
outcome projector at each measurement, honours the conditions on the recorded
bit, and sums the branches at the end.
"""

import math

import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.experimental.dynamic import run_dynamic
from flagquantum.simulation.density_matrix import (
    expectation_z_density,
    reduced_density_matrix,
)


def diagonal(rho):
    """Return the computational-basis populations of a one-row density matrix."""

    return [float(value) for value in rho[0].diagonal().real]


# A measurement nobody reads is a dephasing, not a collapse: the reader loses the
# coherence and keeps neither outcome. A statevector cannot say this, because the
# outer product of one state is rank one and this state has rank two.
superposition = DynamicCircuit(1, dtype=torch.complex128).h(0).measure(0)
rho = superposition.density_matrix()
print("measured |+> populations:", diagonal(rho))
print("measured |+> trace:", float(rho[0].diagonal().sum().real))
print("measured |+> purity:", float((rho[0] @ rho[0]).diagonal().sum().real))

# Feedback puts a pure state back. On a Bell pair wire 1 always agrees with wire
# 0, so flipping wire 1 when bit 0 reads one leaves it clear on both branches:
# the mixture is concentrated on the two states with wire 1 clear, and <Z(1)> is
# exactly one rather than one plus sampling error.
feedback = (
    DynamicCircuit(2, dtype=torch.complex128)
    .h(0)
    .cx(0, 1)
    .measure(0)
    .conditional("x", 1, classical_bit=0)
)
rho = feedback.density_matrix()
print("bell with feedback populations:", diagonal(rho))
print("bell with feedback <Z(1)>:", float(expectation_z_density(rho, 1)[0, 0]))

# Reset is a channel rather than a branch: returning a wire to |0> without
# recording which outcome it was is P0 . P0 + X P1 . P1 X, so it needs one
# density matrix and no record. Applied to a superposition it leaves |0> with
# probability one, and the coherence is gone.
reset = DynamicCircuit(1, dtype=torch.complex128).h(0).reset(0)
print("reset populations:", diagonal(reset.density_matrix()))

# A conditional is only meaningful with an outcome to read, so the route refuses
# a program that reads a bit before measuring it instead of guessing.
try:
    DynamicCircuit(1).x(0).conditional("x", 0, classical_bit=4).density_matrix()
except ValueError as error:
    print("read before measurement:", error)

# Both answers describe the same program, and only one of them is exact. The
# trajectory route is what reaches a backend and what reports shots; the density
# route is the oracle that route is checked against.
shots = 200_000
sampled_program = (
    DynamicCircuit(3, dtype=torch.complex128)
    .ry(0, theta=0.7)
    .ry(1, theta=1.1)
    .measure(0)
    .measure(1)
    .conditional("x", 2, conditions={0: 1, 1: 0})
    .x(2)
    .measure(2)
)
exact = diagonal(sampled_program.density_matrix())
sampled = run_dynamic(sampled_program, shots=shots, seed=3)
index = (
    sampled.final_samples[:, 0].long() * 4
    + sampled.final_samples[:, 1].long() * 2
    + sampled.final_samples[:, 2].long()
)
frequencies = [
    float(count) / shots for count in torch.bincount(index, minlength=8).tolist()
]
gap = max(abs(left - right) for left, right in zip(exact, frequencies, strict=True))
print("exact populations:", [round(value, 6) for value in exact])
print("sampled within three standard errors:", gap < 3 / math.sqrt(shots))

# The analytic value agrees with the exact route rather than with the sample.
# Wire 2 enters |0>, the unconditional X sets it to one, and the conditional X
# clears it again exactly when bit 0 is one and bit 1 is zero, which happens with
# probability sin(0.35)**2 * cos(0.55)**2.
flipped = math.sin(0.35) ** 2 * math.cos(0.55) ** 2
print("analytic P(bit 0 = 1, bit 1 = 0):", round(flipped, 12))
print("exact population of wire 2 clear:", round(exact[4], 12))

# A sum over branches is still a density matrix, so the existing readout works on
# it: tracing out the measured wire leaves the marginal a reader can act on.
marginal = reduced_density_matrix(rho, 1)
print("feedback marginal on wire 1:", diagonal(marginal))
