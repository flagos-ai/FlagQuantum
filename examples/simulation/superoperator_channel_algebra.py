"""Read one amplitude-damping channel as a map, a matrix, and a Kraus list."""

import torch

from flagquantum.operators import SuperOperator
from flagquantum.simulation.pauli import normalized_pauli_basis

dtype = torch.complex128

# Amplitude damping at probability gamma: E(rho) = K0 rho K0^dag + K1 rho K1^dag.
gamma = 0.4
k0 = torch.tensor([[1.0, 0.0], [0.0, (1.0 - gamma) ** 0.5]], dtype=dtype)
k1 = torch.tensor([[0.0, gamma**0.5], [0.0, 0.0]], dtype=dtype)
channel = SuperOperator.from_kraus([k0, k1])

rho = torch.tensor([[0.7, 0.1], [0.1, 0.3]], dtype=dtype)

# The action, the dense generator, and the Choi matrix are three readings of
# the same map rather than three implementations of it.
print("E(rho) diagonal:", channel.apply(rho).diagonal().real.tolist())
print("dense shape:", tuple(channel.dense().shape))
choi = channel.choi()
print("choi eigenvalue count:", choi.shape[0])
print("completely positive:", channel.is_completely_positive())

# A trace-preserving map leaves the output register of the Choi matrix equal to
# the identity and carries no such claim on the input register.
output = channel.partial_trace("output")
print(
    "output register is the identity:",
    torch.allclose(output, torch.eye(2, dtype=dtype)),
)

# The Pauli transfer matrix is the map read in the orthonormal Pauli basis.
pauli = normalized_pauli_basis(1, dtype=dtype, device="cpu")
transfer = channel.matrix_in_basis(pauli)
print("Pauli transfer matrix (real part):")
for row in transfer.real.round(decimals=6).tolist():
    print("  ", row)
print(
    "diagonal is [1, sqrt(1-gamma), sqrt(1-gamma), 1-gamma]:",
    float(transfer[0, 0].real),
)

# The same map placed on one qubit of a two-qubit register leaves the other
# qubit alone, so the caller names the subsystem instead of unpacking the map.
register = torch.eye(4, dtype=dtype) / 4.0
local = channel.apply_to_subsystem(register, [0], 2)
print("local application keeps the trace:", round(float(local.trace().real), 12))

# A Kraus list comes back out, so a channel that arrived as a matrix or a term
# list leaves as operators; the count is the contract, not the phases.
print("Kraus operators recovered:", len(channel.kraus()))
