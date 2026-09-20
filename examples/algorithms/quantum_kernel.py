"""Estimate a kernel matrix by swap test, then classify with kernel ridge.

`flagquantum.algorithms.quantum_kernel` is reachable through the subpackage
surface only -- `from flagquantum.algorithms.quantum_kernel import ...` -- because
the algorithms package adds no root-level `fq.` name.

The premise the unit rests on, and does not meet: the kernel-matrix circuit's
cost counts the swap tests and not the data access. The two feature states are
assumed to be available, reached through a qRAM or an amplitude-encoding unitary
whose cost the count does not include, and here each one is built gate by gate
from the classical feature vector on every run, so that cost is paid rather than
assumed away. Every entry is a sample rather than a reading, and the classifier
fitted on the entries is classical kernel ridge regression.
`docs/guides/ALGORITHMS.md` carries the per-unit boundary in its "Quantum kernel
estimation and kernel ridge classification" section.

Sizes, and why: four feature vectors of two features each, which is the instance
the guide measures, and it stays inside the unit's three-feature bound. The
default sample is 4096 shots per entry, the guide's own.

Run it with:

    python -m examples.algorithms.quantum_kernel
"""

from __future__ import annotations

import argparse

import torch

from flagquantum.algorithms.quantum_kernel import (
    kernel_ridge_classifier,
    quantum_kernel_matrix,
)

TRAIN = torch.tensor(
    [[0.4, 0.5], [1.2, 1.7], [4.6, 1.1], [5.4, 1.9]], dtype=torch.float64
)
LABELS = (1, 1, -1, -1)
HELD_OUT = torch.tensor([[1.3, 1.8], [4.5, 1.2], [0.5, 0.6]], dtype=torch.float64)
REGULARIZATION = 1e-2


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum kernel demonstration")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    matrix = quantum_kernel_matrix(TRAIN, shots=args.shots, seed=args.seed)
    classifier = kernel_ridge_classifier(
        TRAIN, LABELS, regularization=REGULARIZATION, shots=args.shots, seed=args.seed
    )
    predictions = classifier.predict(TRAIN, shots=args.shots, seed=args.seed + 1)
    decisions = classifier.decision_function(
        HELD_OUT, shots=args.shots, seed=args.seed + 2
    )
    diagonal_is_one = all(
        row[index] == 1.0 for index, row in enumerate(matrix.matrix)
    )
    symmetric = all(
        matrix.matrix[i][j] == matrix.matrix[j][i]
        for i in range(len(matrix.matrix))
        for j in range(len(matrix.matrix))
    )

    print("=" * 72)
    print("quantum kernel estimation -- flagquantum.algorithms.quantum_kernel")
    print("=" * 72)
    print(f"  {'task':<18}: estimate a kernel matrix by swap test, then classify with it")
    print(f"  {'execution':<18}: local statevector sampling, CPU")
    print(f"  {'premise':<18}: each feature state is built gate by gate from its")
    print(f"  {'':<18}  classical feature vector, and every entry is a sample")
    print()
    print("inputs")
    print(f"  {'training rows':<18}: {TRAIN.tolist()}")
    print(f"  {'labels':<18}: {LABELS}")
    print(f"  {'regularization':<18}: {REGULARIZATION}")
    print(f"  {'shots':<18}: {args.shots} per entry")
    print(f"  {'seed':<18}: {args.seed}")
    print()
    print("sampled kernel matrix")
    for row in matrix.matrix:
        print(f"  {[round(value, 3) for value in row]}")
    print(f"  {'diagonal is 1':<18}: {diagonal_is_one}")
    print(f"  {'symmetric':<18}: {symmetric}")
    print()
    print("classifier")
    print(
        f"  {'dual coefficients':<18}: "
        f"{[round(value, 3) for value in classifier.coefficients]}"
    )
    print(f"  {'training predict':<18}: {tuple(predictions)}")
    print(f"  {'recovers labels':<18}: {tuple(predictions) == LABELS}")
    print()
    print("held-out decisions (a value at or above zero reads +1, below zero -1)")
    for row, value in zip(HELD_OUT.tolist(), decisions):
        print(f"  {row} -> {round(value, 3)}")
    print()
    print("take away")
    print("  the quantum part is the kernel: one swap test per pair of feature vectors,")
    print("  each result mirrored across the diagonal, and the entry is the squared")
    print("  overlap read back out of the sample. The classifier over those entries is")
    print("  classical, its fit inherits the sampler's error, and no error bound,")
    print("  confidence interval or repetition scheme is computed or reported.")


if __name__ == "__main__":
    main()
