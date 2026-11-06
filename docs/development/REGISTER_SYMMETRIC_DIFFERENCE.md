# Reversible symmetric difference

`symmetric_difference_ir` takes two ascending sets of unique big-endian
words. The left set has at least as many entries as the right. The
number of destination words is the requested cardinality. On a match,
the circuit XORs the ascending symmetric difference into the destination
and toggles the flag. On a mismatch both destinations are preserved.
Both input sets and all borrowed quantum work are restored. The
operation is its own inverse on the valid input subspace.

The work layout holds merged keys, comparison histories, adjacency
bits, Hamming count and junk, a size-match bit, and one shared comparison
and controlled-copy pool. Classical slots are reused in disjoint
stages. A shape with impossible cardinality is the identity. The
empty-right case needs only the applicable copies and flag flip.

## Merge, count and compact

1. Copy the sets into equal power-of-two halves. When padding is
   required, a separate leading bit orders sentinel keys after every
   valid word. Merge using the original BitonicMerge.
2. Compare adjacent live words. An overlap produces exactly one equal
   adjacency since each input set is unique. Count these adjacencies
   and test whether the count equals (left length + right length −
   requested length) / 2. Reject impossible sizes before building the
   circuit, including parity mismatches.
3. When the requested result needs duplicate removal, mark both
   endpoints of each equal adjacency. Sort keys by sentinel, duplicate
   bit, then word. This places live unique words first in ascending order.
   Copy the first requested number of words under the clean match bit.
   The caller's flag may already be occupied and is never used as control.
4. Reverse compaction, size comparison, Hamming count, tags, adjacency
   comparison and merge. Erase the input copies and sentinel preparation.

## Source and resource comparison

The locked [Qualtran SymmetricDifference](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/symmetric_difference.py)
provides a signature and abstract graph but no gate decomposition.
Its class description specifies a matching-size flag, while the
register description labels it a duplicate flag. The size predicate
in the class description and model guides this implementation.
The model's floor division lacks a parity guard, and copying the
first words of the merged list requires a preceding compaction step.
For example, merging [0, 2] with [0, 3] gives [0, 0, 2, 3], whose
first two words differ from the required result [2, 3].
The [cited paper](https://arxiv.org/pdf/2406.19378v1#page=38) uses
symmetric difference for a set oracle without a gate-level decomposition.

The native tests compare the unchanged merge, sort, equality, count,
and temporary-AND leaf streams, including operands and feed-forward.
The model uses the same seven-T controlled-swap convention as the
emitter in the table. Its estimates cover its original call graph.

| Left / right / word bits / output length | Emitted T + T† | Emitted CX | Emitted measurements | Clean work | Classical slots | Qualtran model T | Qualtran model measurements |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 / 1 / 0 | 22 | 61 | 2 | 8 | 1 | 22 | 2 |
| 1 / 1 / 1 / 2 | 30 | 75 | 4 | 8 | 1 | 22 | 2 |
| 2 / 2 / 2 / 2 | 752 | 1,833 | 97 | 37 | 5 | 244 | 33 |
| 3 / 2 / 2 / 3 | 3,756 | 9,195 | 477 | 90 | 7 | 528 | 69 |
| 4 / 2 / 3 / 4 | 4,908 | 12,097 | 639 | 102 | 9 | 1,092 | 147 |

Tests exhaust small sorted-set pairs and occupied destinations.
They check branch-independent restoration through the channel
certificate, enumerate raw measurement histories for the smallest
circuits, and check full complex amplitudes with a spectator on
compacted paths.
