"""Estimate a data matrix's eigenvalues with quantum PCA, at demonstration scale.

`flagquantum.algorithms.pca` is reachable through the subpackage surface only --
`from flagquantum.algorithms.pca import principal_components` -- because the
algorithms package adds no root-level `fq.` name.

The premise the unit rests on, and does not meet: the unit forms the density
matrix `rho = A A^T / tr(A A^T)` classically, builds its exponential as a dense
matrix, and takes the purification's amplitudes from the caller, so quantum
PCA's input model -- copies of `rho`, never formed -- is not present here, and
nothing in the example is faster than diagonalising `rho` directly.
`docs/guides/ALGORITHMS.md` carries the per-unit boundary in its "Quantum PCA"
section.

Sizes, and why: the data matrix is 2x2 so the data register and the purification
register carry one whole wire each, and the counting register is six wires, the
width the guide measures its own examples at. The default sample is 4096 shots
rather than the guide's 20000: the readout is a sample of the same experiment
either way, and the smaller one keeps the script short.

Run it with:

    python -m examples.algorithms.pca
"""

from __future__ import annotations

import argparse
import math

import torch

from flagquantum.algorithms.pca import principal_components

N_COUNTING_WIRES = 6
# A diagonal data matrix, so its exact eigenvalues can be read off by eye next to
# the readout: the density matrix of a diagonal data matrix is diagonal too.
DATA = torch.diag(
    torch.tensor([math.sqrt(0.9962), math.sqrt(0.0038)], dtype=torch.float64)
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum PCA demonstration")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    data = DATA
    rho = (data @ data.T) / torch.trace(data @ data.T)
    exact = [float(value) for value in torch.linalg.eigvalsh(rho)]

    result = principal_components(
        data, n_counting_wires=N_COUNTING_WIRES, shots=args.shots, seed=args.seed
    )

    print("=" * 72)
    print("quantum PCA -- flagquantum.algorithms.pca")
    print("=" * 72)
    print(f"  {'task':<18}: estimate a density matrix's eigenvalues by phase estimation")
    print(f"  {'execution':<18}: local statevector sampling, CPU")
    print(f"  {'premise':<18}: the density matrix, its exponential and the")
    print(f"  {'':<18}  purification's amplitudes are all built classically")
    print()
    print("inputs")
    rounded = [[round(value, 4) for value in row] for row in data.tolist()]
    print(f"  {'data matrix':<18}: {rounded}")
    print(f"  {'shots':<18}: {args.shots}")
    print(f"  {'seed':<18}: {args.seed}")
    print(f"  {'counting wires':<18}: {result.n_counting_wires}")
    print()
    print("result")
    print(f"  {'exact eigenvalues':<18}: {[round(value, 4) for value in exact]}")
    print(f"  {'readout':<18}: {round(result.dominant_eigenvalue, 4)}")
    print(f"  {'readout share':<18}: {round(result.dominant_probability, 4)}")
    print(f"  {'resolution':<18}: {round(result.resolution, 6)}")
    largest = max(exact)
    print(f"  {'within(largest)':<18}: {result.within(largest)}")
    print()
    print("take away")
    print("  the mode is the counter value carrying the largest share of the sample, and")
    print("  the readout is that counter value's eigenvalue. within() answers one question")
    print("  about the readout: whether a named eigenvalue lies within half a counter step")
    print("  of it. The unit does not claim the readout is the largest eigenvalue, and it")
    print("  does not predict which eigenvalue it will be.")


if __name__ == "__main__":
    main()
