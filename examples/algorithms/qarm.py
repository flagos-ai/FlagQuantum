"""Estimate the frequent-item fraction of a small database by amplitude estimation.

`flagquantum.algorithms.qarm` is reachable through the subpackage surface only --
`from flagquantum.algorithms.qarm import frequent_itemset_operator,
run_frequent_itemset` -- because the algorithms package adds no root-level `fq.`
name.

The premise the unit rests on, and does not meet: the cited paper's speed-up
counts calls to an oracle that returns one database entry per call, and it
reaches the candidate itemset superpositions it prepares through a qRAM. Neither
is present here. The transactions are iterated classically, one controlled
increment of the support register per transaction-item membership, emitted from
the incidence matrix as Python reads it, so the access cost is paid rather than
assumed away and no end-to-end advantage follows.
`docs/guides/ALGORITHMS.md` carries the per-unit boundary in its "Frequent-item
fractions by amplitude estimation" section.

Sizes, and why: two transactions over two items, which is the instance the guide
measures, and it is inside the unit's bounds -- the item count is a power of two
and the database is at most eight items and seven transactions. The amplitude
estimation runs at four counting wires and 8000 shots, the guide's own, which
sets the resolution the estimate is read at. The script ends by showing the
refusal the unit makes for a threshold above this database's transaction count.

Run it with:

    python -m examples.algorithms.qarm
"""

from __future__ import annotations

import argparse

import torch

from flagquantum.algorithms.qarm import frequent_itemset_operator, run_frequent_itemset

DATABASE = torch.tensor([[1, 0], [1, 1]])
THRESHOLD = 2
N_COUNTING_WIRES = 4
REFUSED_THRESHOLD = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequent-item fraction demonstration")
    parser.add_argument("--shots", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    operator = frequent_itemset_operator(DATABASE, threshold=THRESHOLD)
    result = run_frequent_itemset(
        DATABASE,
        threshold=THRESHOLD,
        n_counting_wires=N_COUNTING_WIRES,
        shots=args.shots,
        seed=args.seed,
    )
    supports = operator.support
    exact = sum(1 for count in supports if count >= THRESHOLD) / len(supports)
    transactions, items = (int(size) for size in DATABASE.shape)

    print("=" * 72)
    print("frequent-item fractions -- flagquantum.algorithms.qarm")
    print("=" * 72)
    print(f"  {'task':<18}: estimate the share of items whose support meets a")
    print(f"  {'':<18}  threshold")
    print(f"  {'execution':<18}: local statevector sampling, CPU")
    print(f"  {'premise':<18}: the transactions are iterated classically, in place of")
    print(f"  {'':<18}  the coherent database access the cited paper's count assumes")
    print()
    print("inputs")
    print(f"  {'database':<18}: {DATABASE.tolist()} ({transactions} transactions, {items} items)")
    print(f"  {'threshold':<18}: {THRESHOLD} (inclusive)")
    print(f"  {'shots':<18}: {args.shots}")
    print(f"  {'seed':<18}: {args.seed}")
    print()
    print("circuit")
    print(f"  {'supports':<18}: {supports}")
    print(f"  {'support wires':<18}: {operator.n_support_wires}")
    print(f"  {'evaluation wires':<18}: {operator.n_wires}")
    print()
    print("result")
    print(f"  {'estimate':<18}: {round(result.estimate, 6)}")
    print(f"  {'resolution':<18}: {round(result.resolution, 6)}")
    print(f"  {'exact fraction':<18}: {exact}")
    print(f"  {'within(exact)':<18}: {result.within(exact)}")
    print()
    print("a threshold the unit refuses")
    try:
        frequent_itemset_operator(DATABASE, threshold=REFUSED_THRESHOLD)
    except ValueError as error:
        print(f"  {error}")
    else:
        raise SystemExit(
            f"a threshold of {REFUSED_THRESHOLD} was not refused, and the example "
            "exists to show the refusal"
        )
    print()
    print("take away")
    print("  the estimate is read off the counting register as the fraction the marking")
    print("  operator selects, and the support it marks is filled from the database entry")
    print("  by entry in Python. The comparison at the threshold is inclusive, so the")
    print("  item whose support is exactly 2 counts here. A threshold below one, or")
    print("  above the transaction count, is refused rather than run: every item meets")
    print("  it, or none does, and neither needs a circuit.")


if __name__ == "__main__":
    main()
