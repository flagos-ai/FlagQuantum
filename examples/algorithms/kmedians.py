"""Assign points to centroids with quantum k-medians, at demonstration scale.

`flagquantum.algorithms.kmedians` is reachable through the subpackage surface
only -- `from flagquantum.algorithms.kmedians import kmedians` -- because the
algorithms package adds no root-level `fq.` name.

The premise the unit rests on, and does not meet: the paper counts oracle calls
that evaluate a distance in one step, while here the oracle is synthesized from
the predicate's truth table, at `O(2**n)` cost over the at most eight register
values the three-wire search can carry, and the distance table it compares is
computed classically, one point at a time, before any circuit is built. Nothing
here reads a qRAM, and the assignment is sampled rather than read out, so a small
sample can stop a point's search short.
`docs/guides/ALGORITHMS.md` carries the per-unit boundary in its "Quantum
k-medians" section.

Sizes, and why: six points and three centroids, which is the instance the guide
measures, and it stays inside the unit's three-wire bound of eight centroids.
The default sample is the unit's own default, 1024 shots per Grover search; the
script prints how many searches that took.

Run it with:

    python -m examples.algorithms.kmedians
"""

from __future__ import annotations

import argparse

import torch

from flagquantum.algorithms.kmedians import kmedians

POINTS = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.2, 0.1], [0.2, 1.5], [4.9, -0.4]],
    dtype=torch.float64,
)
CENTROIDS = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 2.0]], dtype=torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum k-medians demonstration")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = kmedians(POINTS, CENTROIDS, shots=args.shots, seed=args.seed)
    classical = torch.cdist(POINTS, CENTROIDS).argmin(dim=1).tolist()
    labels = list(result.labels)

    print("=" * 72)
    print("quantum k-medians -- flagquantum.algorithms.kmedians")
    print("=" * 72)
    print(f"  {'task':<18}: assign each point its nearest centroid, one k-medians step")
    print(f"  {'execution':<18}: local statevector sampling, CPU")
    print(f"  {'premise':<18}: the distances are a classical table and the search's")
    print(f"  {'':<18}  oracle is synthesized from a truth table, so it is not free")
    print()
    print("inputs")
    print(f"  {'points':<18}: {POINTS.tolist()}")
    print(f"  {'centroids':<18}: {CENTROIDS.tolist()}")
    print(f"  {'shots':<18}: {args.shots} per Grover search")
    print(f"  {'seed':<18}: {args.seed}")
    print()
    print("result")
    print(f"  {'assignment':<18}: {tuple(labels)}")
    print(f"  {'medians':<18}: {result.medians}")
    print(f"  {'searches':<18}: {result.searches}")
    print()
    print("reference")
    print(f"  {'classical labels':<18}: {tuple(classical)}")
    print(f"  {'agrees':<18}: {labels == classical}")
    print()
    print("take away")
    print("  the assignment is sampled, one Grover-style minimum search at a time, and")
    print("  the centroid update is classical arithmetic. Six points cost one round each")
    print("  that finds nothing and ends that point's loop, and four further rounds moved")
    print("  a point's threshold: ten searches in all. A sample small enough to miss the")
    print("  indices a round marked ends a search early, at an index it has not finished")
    print("  improving.")


if __name__ == "__main__":
    main()
