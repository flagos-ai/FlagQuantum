"""Build a feature-selection objective and evaluate it, at demonstration scale.

`flagquantum.algorithms.feature_selection` is reachable through the subpackage
surface only -- `from flagquantum.algorithms.feature_selection import
feature_selection_qubo` -- because the algorithms package adds no root-level
`fq.` name.

The premise, and it is an absence: this unit does not solve. It builds the binary
objective of one instance -- a subset's relevance and pairwise redundancy, scored
with a penalty on the subset's size -- and evaluates it at an assignment. The
repository has no annealer, so which subset comes back, and at what cost, belongs
to whatever solver the problem is handed to. The scores and the penalty weight
are the caller's data, and no weight at which the target size starts to bind is
computed or predicted.
`docs/guides/ALGORITHMS.md` carries the per-unit boundary in its "Feature
selection as a QUBO" section.

Sizes, and why: five features and a target of two, which is the instance the
guide measures. It runs no circuit and takes no sampler argument: the whole unit
has no quantum part. The cheapest-subset lines below are this script's own
enumeration of all 32 subsets, not the unit's.

Run it with:

    python -m examples.algorithms.feature_selection
"""

from __future__ import annotations

import itertools

import torch

from flagquantum.algorithms.feature_selection import (
    FeatureSelectionProblem,
    feature_selection_qubo,
)

RELEVANCE = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6], dtype=torch.float64)
N_SELECTED = 2
TIGHT_PENALTY = 5.0
LOOSE_PENALTY = 0.05
SUBSETS = tuple(itertools.product((0, 1), repeat=len(RELEVANCE)))


def _cheapest(problem: FeatureSelectionProblem) -> tuple[tuple[int, ...], float]:
    """Return the lowest-scoring subset of ``problem`` and the score it carries."""
    return min(
        ((subset, problem.energy(subset)) for subset in SUBSETS),
        key=lambda item: item[1],
    )


def main() -> None:
    tight = feature_selection_qubo(
        RELEVANCE, n_selected=N_SELECTED, penalty=TIGHT_PENALTY
    )
    loose = feature_selection_qubo(
        RELEVANCE, n_selected=N_SELECTED, penalty=LOOSE_PENALTY
    )

    print("=" * 72)
    print("feature selection as a QUBO -- flagquantum.algorithms.feature_selection")
    print("=" * 72)
    print(f"  {'task':<18}: build one selection objective and evaluate it at an")
    print(f"  {'':<18}  assignment")
    print(f"  {'execution':<18}: none -- the unit builds an objective and runs no circuit")
    print(f"  {'premise':<18}: the unit solves nothing, and the repository has no annealer")
    print()
    print("inputs")
    print(f"  {'features':<18}: {len(RELEVANCE)}")
    print(f"  {'relevance':<18}: {RELEVANCE.tolist()}")
    print(f"  {'target size':<18}: {N_SELECTED}")
    print(f"  {'penalty':<18}: {TIGHT_PENALTY} (tight) and {LOOSE_PENALTY} (loose)")
    print()
    print(f"objective at penalty {TIGHT_PENALTY}")
    print(f"  {'linear':<18}: {tight.qubo.linear}")
    print(f"  {'quadratic terms':<18}: {len(tight.qubo.quadratic)}")
    print(f"  {'offset':<18}: {tight.qubo.offset}")
    pauli = [term.pauli for term in tight.to_ising().terms]
    print(f"  {'ising ZZ terms':<18}: {pauli.count('ZZ')}")
    for subset in ((1, 1, 0, 0, 0), (1, 1, 1, 1, 1)):
        print(f"  {'energy ' + str(subset):<18}: {round(tight.energy(subset), 3)}")
    print()
    print("the objective's own cheapest subset")
    for penalty, problem in ((TIGHT_PENALTY, tight), (LOOSE_PENALTY, loose)):
        subset, score = _cheapest(problem)
        print(f"  penalty {penalty:<6}: {subset} at {round(score, 3)}")
    print()
    print("take away")
    print("  the unit builds the objective and evaluates it where the caller points; the")
    print("  subset selection above is the script's enumeration of all 32 subsets, not a")
    print("  result of the unit. At the tight weight the cheapest subset has the target")
    print("  size and at the loose one it does not, which is one instance at two weights")
    print("  and not a weight at which the size term starts to bind.")


if __name__ == "__main__":
    main()
