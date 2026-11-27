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

Q-0409 closes the runnable original tests. The earlier Q-0408 hold for the
source one-orbital failure and opaque erase leaf is superseded by the complete
source review below; those upstream limitations remain explicit.


## Complete source-file closure, 2026-11-28

Q-0408 includes the integral representative enumeration and nested insertion,
all stored fields, register properties, integral factory, QROM construction and
renaming, ordered eighteen-call body, resource call graph, source example and
documentation declaration. Algorithms owns classical integral data; Compiler
owns typed composition, alias lookup, swap routing and erasure. Ports are
MSB-first. LEFT inputs and THRU flags remain caller-owned; RIGHT selected and
retained lookup banks become caller-owned outputs. Only private predicates are
allocated and cleared internally. Inverse measured erasure consumes outputs of
the corresponding forward isometry; it is not an arbitrary-bank unitary inverse.

The integral owner now preserves NumPy input precision through vector filtering,
concatenation, alias cumulative rounding and normalization. The source accepts
negative and IEEE nonfinite cutoffs as Boolean filters; finite real integral
validation remains in force. NaN is not serialized as a circuit parameter.
Previously, float32 input was promoted prematurely: an eight-bit witness changed
source keep values `(65, 102, 205, 0)` to `(64, 103, 205, 0)`. The corrected
owner matches the untouched source. Ten dtype/cutoff profiles match all thirteen
lookup datasets and exact normalization. Forty complete previously accepted
float64 data records remain identical to published `d4802f79b`.

The new executable-input regression constructs native programs from native
integral records, compares recursive forward and inverse source calls/operands,
and exercises ordinary append and IR round-trip. A nine-row negative-cutoff
case compares full complex forward amplitudes with coherent dirty THRU input,
checks private work zero, and restores that input through measured erasure.
Both zero/one representative histories are tested here; this does not claim
exhaustion of all histories of the larger root. Existing small all-history and
bank-unitary evidence is reused without rerunning a different reference.

Receipts use `g2-sparse-chemistry-full-closure-*20261128` and
`g2-sparse-preprocessing-differential-20261128.json`. Actual emitted primitives
and source model estimates remain separate. The eight-bit four-row forward
emits 203 T-family gates and fifteen measurements; the nine-row forward emits
259 T-family gates and twenty-nine measurements. Their matching-image erasures
emit respectively 55 and 69 measurements. All private leases return; retained
forward banks are outputs, not uncleared private work.

Q-0408 closes under the approved faithful-migration rule. Source singleton
`target0_` decomposition failure and atomic model-only erase remain source
limitations, separately from the native executable extensions. The arbitrary-bank
owner used by source Walk is documented in
[the bank-unitary audit](SPARSE_WALK_BANK_UNITARY_AUDIT.md).
