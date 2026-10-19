"""Prepare a QUBO problem at the local Kaiwu ecosystem boundary.

This example performs no remote submission, reads no credentials, and consumes
no provider quota. Run it from the repository root with::

    python -m examples.kaiwu_matrix_boundary
"""

from __future__ import annotations

import itertools

import torch

from flagquantum.ecosystem.kaiwu import (
    decode_qubo_spins,
    encode_qubo_as_ising,
    ising_energy,
    prepare_integer_precision,
)


def main() -> None:
    """Convert, inspect, and solve one tiny QUBO problem locally."""
    qubo = torch.tensor(
        [[-1.0, 0.75], [0.75, -1.5]],
        dtype=torch.float64,
    )
    offset = 0.25
    encoding = encode_qubo_as_ising(qubo, offset=offset)

    spins = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=encoding.matrix.shape[0])),
        dtype=torch.float64,
    )
    energies = ising_energy(encoding.matrix, spins, bias=encoding.bias)
    best_spins = spins[int(torch.argmin(energies).item())]
    binary = decode_qubo_spins(best_spins)

    binary_float = binary.to(dtype=torch.float64)
    qubo_energy = binary_float @ qubo @ binary_float + offset
    ising_result = ising_energy(
        encoding.matrix,
        best_spins,
        bias=encoding.bias,
    )
    torch.testing.assert_close(ising_result, qubo_energy, rtol=0.0, atol=1.0e-12)

    precision = prepare_integer_precision(
        encoding.matrix,
        target_min=-127,
        target_max=127,
    )

    print("FlagQuantum Kaiwu matrix boundary check passed")
    print(f"  binary solution: {binary.tolist()}")
    print(f"  QUBO energy: {qubo_energy.item():.6f}")
    print(f"  encoded Ising energy: {ising_result.item():.6f}")
    print(f"  integer scale factor: {precision.scale_factor:.6f}")
    print(f"  maximum dequantization error: {precision.max_abs_error:.6g}")
    print("  remote submission: not performed")


if __name__ == "__main__":
    main()
