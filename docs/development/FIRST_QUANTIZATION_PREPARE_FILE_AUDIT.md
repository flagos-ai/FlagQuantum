# First-quantization tutorial and particle-pair preparation audit

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
