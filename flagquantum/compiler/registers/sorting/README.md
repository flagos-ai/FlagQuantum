# Sorting and permutations

This package owns compare/exchange networks, permutations, duplicate detection and set combinations.

The in-place-sort and sorted-duplicate resource modules also expose direct
source-model call declarations for bounded dependency diagrams. Their counts
come from the same existing model owners. These diagrams preserve original
adjoint identities and do not supply missing physical sorting bodies.

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

`permutation_index_component.py` stores occupied-target rank XOR and indexed
word XOR through the shared equality and controlled-copy components.
`in_place_sort_component.py` composes them with a padded bitonic sorted copy.
Its concrete unsigned xs/pi arrays retain the frozen SortInPlace signature,
including empty singleton rank words. Forward requires distinct words and clean
pi. Inverse restores the original order and clears pi. Internal tables, keys,
histories and predicates are owned by the components and return clean.
The frozen root has a resource model without executable decomposition, so root
primitive parity uses the existing native paper construction while its child
primitive evidence remains tied to the frozen source. Bind using
`in_place_sort_requirements` and append the bound operation through the register
package guide. The complete complex/branch consumer is in
`tests/unit/test_in_place_sort_component.py`, and the frozen object-library
consumer is in `tests/unit/test_in_place_sort_foreign_file.py`.

`symmetric_difference_component.py` composes the finite unsigned sorted-set
oracle through the same merge, equality, Hamming, compaction and exact-adjoint
owners. All four ports are THRU and occupied diff/flag targets are XOR updated.
Run `tests/unit/test_symmetric_difference_component.py` and
`tests/unit/test_symmetric_difference_foreign_file.py` for its shortest checks.
The root source supplies no gate decomposition. See the local audit for original
model omissions, signed/symbolic execution limits and temporary-wire scheduling.

`permutation_declaration.py` owns generic cycle/Permutation metadata and factories,
including complete unknown cycle shapes. Dense and partial conversion reuse
`cycle_decomposition.py`; concrete `program()` delegates to
`permutation_component.py`. The example catalogue consumes these same declarations.
Run `tests/unit/test_permutation_declaration.py` and
`tests/unit/test_permutation_foreign_file.py` for factory, source serialization,
typed SDK, complete complex amplitude and clean-work evidence.

`bitonic_declaration.py` owns generic merge/sort signatures, symbolic histories
and source direct models. It delegates finite forward and inverse conversion to
the existing merge/sort component owners. The source catalogue consumes these
same declarations. Merge uses LEFT input arrays and RIGHT result/history, while
sort uses THRU words and RIGHT history. Run the sorting declaration and foreign
consumer tests for the shortest checks, and see `SORTING_FILE_AUDIT.md` for the
original direct SDK merge wrapper failure and its composite conversion evidence.

`network_graph.py` owns finite array wiring for parallel comparison, bitonic merge
and recursive sort. Its supplied comparator entry has `a`/`b` THRU words and one
RIGHT `out` bit. The comparator owns its temporary workspace. Concrete component
factories supply the existing executable comparator. Symbolic example graphs
supply the same typed interface with unresolved word widths. Both consumers use
the same pair order, left/right/merge call order and retained logical histories.
Physical-width validation and exact inverse construction remain with component
factories and binding. Run `tests/unit/test_sorting_graph.py` for a replaceable
comparator example and `tests/unit/test_sorting_component.py` for normal append,
source streams, measured branches and inverse cleanup.
