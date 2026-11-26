# First-quantization tutorial and particle-pair preparation audit

## Standalone source constructor binding

The complete `prepare.py` class now resolves under its original exact module
identifier, using the already owned particle-pair declaration factory. The
separate copy in `select_and_prepare.py` keeps its own source identity; their
field sets, register declarations and source models are compared rather than
assuming the two classes are aliases. No duplicate preparation implementation
or quantum body was added.

Actual source libraries at depths zero, one and two round-trip and resolve
without Qualtran, Cirq or Qiskit imports. The one-particle declaration omits
both zero-width ports. All declarations refuse physical binding, matching the
source's absent decomposition; primitive counts, complex amplitudes and work
cleanup are unavailable, not zero. Existing whole-file and original-test
acceptance Q-0364/Q-0369 is reused, not counted again.

| Particles, rotation precision | Source and native Toffoli model | Source and native peak model | Emitted primitives |
|---|---:|---:|---|
| 1, 8 | 14 | 3 | Unavailable |
| 2, 1 | -7 | 3 | Unavailable |
| 10, 8 | 42 | 8 | Unavailable |

The negative source resource formula is retained, not repaired or interpreted
as an executable negative gate count. This adds only the missing exact source
constructor key; the existing field admission and all other ordinary root
constructors remain with their original owners.

The frozen Qualtran v0.7.0 reference is at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/pbc/first_quantization`.

| Inventory | Source | Git blob |
| --- | --- | --- |
| Q-0363 | `first_quantization.ipynb` | `b1de1dbf26756d68a56b4d4290dd4905ea45e343` |
| Q-0364 | `prepare.py` | `aba6fd476007748acf99947caba2dcc21e7843be` |
| Q-0369 | `prepare_test.py` | `f07ea4087daeb8dae351ed6e8061345b549d0ee6` |

The native 58-cell tutorial preserves all seven examples. It presents 14
diagrams spanning full and leaf PREPARE/SELECT signatures and depth-one
graphs. Port comparisons cover names, directions, dtypes, iteration
bounds, and array shapes. The SELECT graph includes the allocation and
release vertices without treating them as emitted gates. Each graph is
checked for direct multiplicities, recursive gate categories, and peak
logical qubits. Run the native tutorial with
`python -m pytest tests/integration/test_register_graph_notebook.py -k register_first_quantization`.

## Particle-pair declaration

`source_prepare_ij_signature` declares the `i` and `j` through words. The
locked source uses `(eta - 1).bit_length()` per word. The source prose
describes unary indices. At one particle the signature omits both
zero-width words. The logical gate formula and the single Toffoli leaf
remain independently queryable. Both copies of the source class lack an
executable decomposition. The locked formula can give a negative
resource count at small rotation precision, which is retained as a
model result. The original test constructs the `eta=10`, precision-8
example. Native tests check both source definitions, their adjoints,
and the additional width and precision cases.

## Accounting

| Fixture | Source declaration | Native emitted primitives |
| --- | --- | --- |
| Pair PREPARE, eta 10 and precision 8 | 42 Toffoli, eight peak logical qubits | Source supplies only a model |
| Full PREPARE tutorial | 485 Toffoli, 21 And, peak 81 | Source contains declaration-only leaves |
| Full SELECT tutorial | 576 Toffoli, 720 CSwap, 68 And, peak 298 | Source contains declaration-only leaves |
| Three-value kinetic-coordinate child | Shared uniform-preparation model | 20 T/T† and five measurements across six wires |

The kinetic-coordinate child has independent complex-amplitude and
work-cleanup evidence across all 32 measurement histories. Its
decomposition and emitted statistics are separate from the kinetic
PREPARE estimate. The gate assembly and `circuit.append(op)` bindings for
executable dependencies remain in their existing owners. The rest of
the PREPARE/SELECT implementation boundaries remain in their own source
entries, including the unimplemented physical pair and power-of-two leaves.

The original tutorial and original PREPARE test passed. The native
tutorial passed in an independent kernel. Q-0363, Q-0364 and Q-0369
close the tutorial, source declaration and original test respectively.
