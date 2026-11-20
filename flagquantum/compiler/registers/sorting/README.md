# Sorting and permutations

This package owns compare/exchange networks, permutations, duplicate detection and set combinations.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [bitonic_sort.py](bitonic_sort.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_bitonic_sort.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`sorted_duplicates_component.py` stores the original HasDuplicates call graph
using shared half-LT and MCX owners. It propagates the source inverse's returned
physical aliases. Source lists of three or more can violate later erasure
preconditions, and the uncontrolled singleton has no source decomposition.
`sorted_duplicates_ir` retains its separately accepted physical-order-preserving
execution route. See
[the complete source audit](../../../../docs/development/SORTED_DUPLICATES_FILE_AUDIT.md).
Run `python -m pytest tests/unit/test_sorted_duplicates_component.py -q` for
source graph/branch conformance, valid two-word clean channels and defect witnesses.

`contiguous_index_component.py` composes native square, two additions, XOR,
two exact subtraction calls and square erasure. Its three unsigned words are
THRU and MSB first. The square numerator and scratch are owned internally.
Bind with the work count from `contiguous_index_requirements`, then append the
bound operation as in the register package guide. Run
`python -m pytest tests/unit/test_contiguous_index_component.py -q` for the
seven stored arithmetic calls, operands/feedback, occupied-target amplitudes,
spectator coherence and work restoration. The original root provides no
decomposition, and its classical through-target mismatch remains recorded in
[the complete source audit](../../../../docs/development/CONTIGUOUS_INDEX_FILE_AUDIT.md).
