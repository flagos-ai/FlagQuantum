# Symmetric difference complete source responsibility

## Signed native execution interface

The stored native oracle accepts an internal `signed` boolean, default false.
Signed S and T are ascending distinct two's-complement QInt arrays; diff keeps
the same nominal type and may contain arbitrary occupied raw bits. Flag is an
occupied QBit. On a cardinality match, the sorted symmetric difference is XORed
into diff and flag is toggled. Otherwise diff and flag are preserved under the
existing native oracle contract. Every input is THRU and MSB first. Source
default constructor declarations remain bodyless; the frozen root has no gate
decomposition or numerical callback from which to claim root-stream parity.

Private copied keys receive a sign-bit XOR before merge and its inverse before
copy cleanup. Shared unsigned merge, equality, Hamming count, compaction and
their exact adjoints retain scratch and feedback ownership. While the private
success predicate is live, one controlled X per output corrects the biased
payload sign. This correction must be conditional: conjugating an occupied
output by an unconditional sign flip would change mismatch behavior. Nominal
Split/Join views connect signed inputs to the existing raw-word XOR owner.
No new primitive, protected API, numerical kernel or measurement is introduced.

Q-0205 and Q-0206 cover the exact frozen implementation and original tests.
The new file audit checks both blobs, all three actual factories, the concrete
DocSpec subset and every ordered signature port, dtype, shape and side.

Compiler `sorting/symmetric_difference_declaration.py` owns standalone metadata
and the original ten-edge concrete call graph. `symmetric_difference.py` owns
the executable set oracle and its clean-work contract. Shared model dependencies
stay with comparison, control, bitonic merge and Hamming components. In-place
sort now reuses the same comparator model. There is no alternate graph registry.

The root declaration retains the source's merge using left length for both
halves, equality on every adjacent, Hamming count and its adjoint, floored
cardinality predicate, word XOR copies and final CNOT. Both directions and
source insertion order remain visible. Individual child QEC queries and root
peak/category queries match source across unequal/equal sets and even/odd
requested cardinalities. Signed/opaque metadata are carried by RegisterType,
while the actual emitter keeps its existing unsigned sorted-set domain.

The actual 4+2, four-bit, four-output factory declares 96 CSwap, 205 logical
And, 2276 Clifford and 205 measurements, with peak 55 logical wires. The native
executable network separately emits 6044 T/TDG, 14953 CX and 797 measurements,
with 113 clean work wires. The smaller 2+2, two-bit, two-output emitter retains
752 T/TDG, 1833 CX, 97 measurements and 37 clean work. These measured accounts
include compaction and parity checking absent from the original root model.

Existing tests retain exact comparable dependency streams, operands and
feedback, exhaustive small sorted-set pairs and occupied outputs, complete
complex amplitudes with spectators, inverse recovery, all applicable small raw
histories and work cleanup. No emitter changed in this batch.

The original non-notebook test file ran unchanged with 6 passed, 9 skipped,
2 xfailed and 1 notebook deselected. Its existing unknown-notebook-marker warning
is retained. Source autotester skips and xfails do not constitute native
executable acceptance.

The source has no root decomposition. Its register prose calls flag a duplicate
flag, while its class and cost predicate specify cardinality matching. Its model
has no impossible-parity rejection or compaction. Source degenerate/invalid diagnostic shapes,
foreign/static execution admission and original test fixture gaps remain open
under Q-0205/Q-0206. The native symbolic model now covers both original factories and mixed-bound
length/width/output fields. Its direct dependencies preserve concrete versus
symbolic comparison rules and source nested peak schedules. Positive non-power-of-two
merge lengths retain the original declaration fallback while executable merging
keeps its power-of-two contract.

For 1+1 one-bit inputs and zero output words, the original root's QubitCount
fails through its one-bit Hamming child. Its QEC query still matches the native
model. Native explicit-storage accounting returns five peak wires, separately
from that failed source query. This boundary is checked directly.

The final concentrated batch passed 820 cases without skips. One existing
IPython temporary-directory fallback warning remains. Changed production/tests
and all extracted notebook code passed strict typing, Black, Ruff, Compiler
scope, architecture, multilevel IR and protected API checks. Four notebook
figures were rendered and visually inspected.


## Symbolic declaration closure

`source_symmetric_difference_resource_formula` composes shared comparator,
bitonic merge, equality, constant equality, controlled-X and Hamming formulas.
Both original symbolic factories match every QEC category and full peak expression.
Mixed symbolic lengths, widths and output sizes are checked independently. Source
nested Max schedules are retained for concrete merge lengths with symbolic widths,
including first-round and parallel-stage history lifetimes. Symbolic list lengths
retain the original individual-child fallback. Unknown control polarities remain
separate from concrete all-zero controls. Symbolic expressions are not replaced
by numerically bound circuit costs.

Original source files remain blob-identical. Existing concrete gate, operand,
feedback, complex-state, measurement-history and work-cleanup evidence is reused.
No execution path changed. Source compaction/parity omissions, raw callback and
protected physical admission gaps remain separately registered.

The symbolic follow-up passed 822 concentrated cases with no skips and one
existing IPython fallback warning. Changed Python and extracted notebook cells
passed strict typing, format, lint, scope, architecture, multilevel IR and API
checks. Source model-only positive non-power-of-two merge fallback is now covered.

## Stored execution interface and ownership

The finite stored owner will retain S, T, diff and flag as THRU ports, using
MSB-first unsigned words. Inputs are distinct ascending sets with left length
at least right length. Diff and flag accept occupied XOR targets. A cardinality
mismatch leaves both targets unchanged. The oracle is self-inverse.

The root owns padded key copies, adjacent-equality flags and the success bit.
Merge and compaction expose histories consumed by their exact adjoints. Hamming
exposes its count and junk, consumed before duplicate tags are erased. Predicate
work belongs to shared children. All owned quantum storage returns clean.
The implementation composes existing typed owners and adds no public export.
The source root provides a declaration/model only, so dependency primitive parity
and native paper-construction execution will be certified separately.

## Stored root delivery and comparison boundary

`symmetric_difference_component.py` now composes the complete finite unsigned
sorted-set oracle using the existing merge, equality, Hamming, compaction, XOR
and controlled-X owners. Empty sets, impossible cardinality/parity, occupied
diff/flag, padded unequal halves and exact reverse cleanup retain the previous
native construction. The normal `circuit.append(op)` path is exercised.

All three actual original factories preserve exact library bytes at depths zero,
one and two. Concrete source reconstruction is equal. Both symbolic original
reconstructions lose positive/integer symbol assumptions themselves, including
the dtype width expression. The native transport preserves those original wire
bytes. Concrete unsigned roots resolve through an explicitly supplied native
factory. At that publication, signed/opaque variants retained metadata only.
The signed continuation below adds explicit QInt execution; approved symbolic
logical declarations are covered in the later constructor section.

Thirteen finite shapes have exhaustive valid sorted-set basis-channel cleanup
certificates, including occupied outputs and self-inverse recovery. One-bit
pairs with zero/two outputs have full complex spectator amplitudes in every raw
measurement history. The compacted 2+2 and padded 3+2 two-bit cases have full
complex input superpositions on three seeded trajectories, separately from the
exhaustive basis certificates. Their exponentially many raw histories were not
enumerated.

Every primitive category count matches the previous native flat construction.
An attempted whole-network comparison using a single fixed permutation of clean
wire names failed at three shapes because child lifetime reuse differs from the
flat shared pool. That stronger wire/order comparison remains unaccepted.
The diagnostic is preserved in `g2-symmetric-difference-static-wire-diagnostic.log`.
Frozen child source primitive/operand/feedback evidence is reused unchanged.
The absent source root decomposition prevents a whole-root upstream gate claim.
Expanded root scheduling remains unqualified.

The completed batch passed 722 concentrated cases, 17 local integration cases,
and 34 focused cases on each of Cirq 1.7 and 1.6.1, without skips or warnings.
Four touched Python files pass strict typing, Black and Ruff. Scope,
architecture, multilevel IR and protected API checks pass. Six frozen source
blobs and eleven unchanged production owners are verified. The whole dependency
scan retains 685 findings with none in this batch. The ten-file pre-existing
default docstring failure baseline remains byte-identical. These results do not
certify the whole repository.

The actual 4+2 four-bit/four-output factory emits 14,953 CX, 6,044 T/TDG and
797 measurement/reset pairs, with peak 113 clean work wires. The original model
separately declares 96 CSwap, 205 And, 2,276 Clifford, 205 measurements and peak
55 logical wires. Both accounts retain their distinct meanings.
Q-0205/Q-0206 remain open for the registered execution and source boundaries.


## Approved symbolic constructor continuation

Proposal 120 supersedes historical symbolic-array declaration rejection.
Existing foreign tests now accept bodyless logical signatures and require
physical binding refusal. The dedicated constructor adapter preserves the
original root declaration and missing decomposition. Previously accepted native
physical synthesis and independent source models retain their separate scopes.
See RESOLVER_LIST_AUDIT.md for the complete constructor dependency closure.

## Complete original test-file acceptance (2026-11-25)

Q-0206 now has a complete mapping for its three original functions. The
unchanged original selection of six passes, nine skips and two xfails is reused;
its notebook test is covered by the accepted original/native lists kernels in
LISTS_NOTEBOOK_AUDIT.md. Both exact source and test blobs remain verified by the
existing file fixture. Implementation parent Q-0205 retains its separate gaps.

| Original function | Complete native assertion mapping |
| --- | --- |
| `test_examples` | All three actual factories and fifteen raw protocol outcomes. Existing ordered signature, complete ten-edge source model graph, foreign library and approved symbolic logical declaration owners retain the absent root decomposition. |
| `test_cost` | Both original symbolic factories match every original QEC category through the native formula, stronger than the source's unspecified Clifford/measurement fields. The native shared T/CCZ conversion matches original T equivalents exactly, and native asymptotic_order verifies the original l*word_width*ceil(log2(l))**2 bound for both factories. |
| `test_notebook` | The complete unchanged original lists notebook and reference-blocked native notebook execute under their accepted kernel evidence, reused here. |

The three factories produce four PASS, five NA, three UNVERIFIED, two FAIL and
one MISSING across fifteen modes. Both symbolic serialization failures remain;
the concrete serializer passes. There is no successful original root physical
decomposition to compare against. The independent native set oracle retains its
existing sorted-set input contract, complete small complex histories, arbitrary
occupied outputs, inverse and zero-work evidence. Its compaction/parity gates and
physical inventories are not substituted for the original analytical model.
Root prose/cardinality, degenerate Hamming peak, general dtype and source
constructor/physical boundaries remain explicitly open under Q-0205. The signed
integer execution gap is closed by the continuation below.

The combined list assertion/component/model/constructor regression passes 673
cases; the independent Cirq 1.6 file/factory selection passes 55. Both complete
changed test files pass strict typing, Black and Ruff. Integration scope,
language, links, architecture, IR and protected API checks pass. Four frozen
source/test blobs and native Kickmix 0.1 Simulator are verified. Receipts use
`g2-list-source-closure-*20261125`; no source body was modified.

## Signed stored-oracle acceptance

All thirteen finite unsigned Programs remain exactly equal to published
f873ea242. Signed execution reuses the unchanged shared child implementations;
source-comparable merge, equality, Hamming and MCX primitive/operand/feedback
certificates remain at those owners. The root source has no physical stream.
No whole-network flat temporary-wire equivalence is inferred from gate counts.

Actual QInt source libraries retain exact bytes at depths 0/1/2 and resolve
through the existing explicit native factory with the same bound circuit as
direct construction. The default constructor still supplies only a declaration.
Independently generated signed sets cover every small valid pair, cardinality
matches and mismatches, occupied outputs, both initial flag values and erasure
by reapplication. The smallest complete complex vectors include every occupied
target and flag encoding, an entangled spectator and every raw measurement
history. Padded/compacted coherent larger profiles retain three seeded
trajectories and exhaustive valid-basis cleanup certificates, rather than an
unfounded claim of enumerating all exponential histories.

For a feasible nonempty two-set shape, the extension adds two X per input word
and one CX per output word. It adds no T-family gates, measurements or private
work. Empty-second-set and impossible-cardinality paths retain the original
native primitive inventory. Thirteen measured resource rows compare the exact
unsigned baseline and independent original QInt models. For 2+2 two-bit inputs
and two outputs, signed emission is 288 X, 1835 CX, 752 T/T-dagger and 97
measurement/reset pairs, with 37 peak clean bits. The independent original
model remains 16 CSwap, 33 And, 348 Clifford, 33 measurements and seventeen
logical bits. Original one-adjacency Hamming peak queries still fail explicitly;
they are not replaced by the measured native peak.

The concentrated set/list/Hamming/sorting regression passes 286 cases, the full
source/interface lane passes 73 cases under Cirq 1.6, and local integration
passes 145. All three changed Python modules pass strict typing, Black and Ruff;
scope, language, links, architecture, IR and protected API checks pass. Frozen
source/test blobs and the real Kickmix 0.1 Simulator are verified. Receipts use
`g2-signed-set-*-20261125`. Q-0205 retains its separately recorded source/model,
general nominal-type, symbolic physical and whole-network comparison limits.
