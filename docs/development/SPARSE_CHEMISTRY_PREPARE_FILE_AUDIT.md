# Sparse chemistry PREPARE audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/sparse`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0407 | `__init__.py` | `6b0c0373aa2cdbfb24bc314c6eb4b47d4aadb674` |
| Q-0408 | `prepare.py` | `02e4b381091a6677d68b299305a9b32c0bf5e04f` |
| Q-0409 | `prepare_test.py` | `4d3fa0ec18ef9abf0aef050231f3147b27bb078f` |

The PREPARE signature uses the exact bounded selection index and
through flags. The selected QROAM cells are right-only outputs with
unsigned orbital words, typed bits, and opaque alternate words.
Retained cells are shaped by dataset, with the output dtype unchanged.
The inverse consumes those output banks. The comparator flag is a
through bit. Both the physical layout and the model width derive from
the declared registers. The tests check typed ports and output
direction in both directions, including empty words and three retained
cells per datum.

## Gate path and coverage

For two and three spatial orbitals the unchanged source forward graph
provides the original register mappings, gate operands, and feedback
schedule. The tests compare the native coherent reverse by replacing
the opaque source erase leaf with the inverse of the source-loaded
bank program. The original abstract erasure costs and the native
measured erasure counts remain separate. The small nonblocked
four-row fixture emits 75 T/T† and three measurements.

For one spatial orbital, the original full decomposition raises
`KeyError: target0_` because its selected address words have zero
width. Native executable lookup filters those fixed-zero datasets
after validating the original named layout. The symmetry and alias
swaps on empty words emit no gates. The source signature continues
to contain the empty ports. The native test exhausts all forward,
coherent inverse, and measurement inverse histories for one and two
data rows. Every history has the expected unnormalized complex
amplitudes and clean work.

| Unblocked one-orbital path | Emitted T/T† | Emitted measurements | Reusable work wires |
| --- | ---: | ---: | ---: |
| Forward | 11 | 1 | 2 |
| Coherent inverse | 11 | 1 | 2 |
| Measurement inverse | 11 | 6 | 2 |

The two-row one-orbital table can also occupy one complete two-cell
block. Its high address has zero width, so the shared QROAM loads
constant banks before the low-address routing. The compiler checks
that stream against locked CSwapApprox gates and the explicit constant
load, including the relative phases and coherent reversal. The full
locked QROAM graph still fails when it tries to bind an omitted selector.
The locked one-orbital PREPARE still fails at `target0_`.

The native sparse binding retains the zero-width orbital metadata and
loads its five nonempty datasets into ten bank wires. All two forward
histories have identical unnormalized complex amplitudes and clean
scratch. Each is composed with both coherent inverse histories and all
2,048 measurement inverse histories, recovering the zero input with
the expected positive branch amplitude. The forward and both reverse
paths each emit 31 T/T† using two work wires. Their measurement counts
are 1, 1, and 11, respectively. The original abstract erasure resource
declaration remains independent of these executed counts.

The original suite returned 13 passes, one skip, and one expected
failure. The native concentrated regression passed 123 cases. Strict
typing, format, ownership, architecture, IR, and public API checks
passed.

Q-0409 closes the runnable original tests. Q-0408 stays open for the
source full one-orbital failure and
the opaque original measurement erase leaf. Q-0407 also covers the
SELECT export and the sparse Hamiltonian convention. That package
boundary remains for the SELECT batch.
