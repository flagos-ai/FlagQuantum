# In-place sorting source-file responsibility

Q-0204 tracks the complete locked `lists/sort_in_place.py` blob
`d4da3727ad3ce11294c140870f2dbf4e1cb0f9fd`. Its single class owns two
fields, signature, index dtype and a direct call graph. The documentation
specification has no examples. The source has no root decomposition or
classical implementation.

## Native ownership and interface

`sorting/in_place_sort_declaration.py` owns the full original metadata and
resource query. `in_place_sort_signature(length, dtype)` retains both `xs` and
`pi` as THRU arrays and a bounded unsigned index of ceil(log2(length)) bits,
exclusive bound length. This follows the actual signature, despite the source
prose describing a RIGHT ancilla. Native dtype metadata preserves unsigned,
signed and opaque words, including symbolic widths and list lengths.

`source_in_place_sort_cost(length, width)` composes the original Comparator,
Comparator adjoint and Xor calls, each repeated 3 length squared. Comparator
reuses the existing GreaterThan dependency and declares width CSwap calls.
QubitCount's declaration-only fallback takes the maximum root/child width.
`source_in_place_sort_resource_formula` composes the existing symbolic comparison
fallback, whose counts differ from concretely bound widths in the original.
Both models have no primitive stream or measurement execution.

The existing `in_place_sort.py` owns actual distinct unsigned-word sorting and
its exact inverse. Callers bind big-endian words, clean rank words, clean work
and reusable feedback slots. Its existing append scenario verifies coherent
amplitudes with a spectator, forward permutation output, backward recovery and
work cleanup. The dependency gate comparisons retain exact operands and feedback.
No emitter or reference implementation was changed in this batch.

| Three words of two bits | Independent account |
| --- | --- |
| Existing native emitter, per direction | 684 T/TDG, 1608 CX, 108 measurements, 32 clean work |
| Original declaration | 108 CSwap, 162 logical And, 1998 Clifford, 162 measurements, 12 model peak wires |

The source root supplies no gate-level stream against which to assert whole-root
primitive equality. Dependency comparisons remain the applicable evidence.

## Verification and remaining boundaries

The new audit checks the exact source blob, empty DocSpec, all three direct
callees and multiplicities, native signature/type/shape/side metadata, concrete
QEC and peak queries across positive lengths and widths, and symbolic formulas
for separately symbolic length/width and signed/opaque metadata. Existing complete
permutation, sort, branch, coherent append and inverse suites are reused in the
concentrated regression.

The declared model accepts every positive length, independently of the physical
unsigned distinct-word domain. It does not certify signed executable sorting,
occupied rank words, duplicates, generic foreign construction, serialization or
protected static/symbolic execution admission. Zero-length source QubitCount
raises a wrapped math-domain error. These unadmitted execution and diagnostic
boundaries remain open under Q-0204, even after the full declaration is covered.

Batch verification passed 718 concentrated native/reference cases with no skips or warnings.
Changed Python passed strict mypy, Black, Ruff, Compiler scope, architecture,
multilevel IR and protected API checks. Supplemental compare/exchange and
shared-control results are recorded in `g2-lists-file-control-dependencies.log`.

## Stored execution batch interface (2026-11-20, in progress)

Compiler sorting owns two missing typed composition layers. The shared
permutation_index_component owner will expose rank-XOR and indexed-word-XOR
Programs over THRU arrays table, words and ranks. Each word is MSB first. Count
and word width remain concrete and follow permutation_index_requirements.
Rank-XOR accepts occupied rank words and preserves table/source words. Its
logical permutation meaning requires a distinct table containing every source.
Indexed-word-XOR accepts occupied words and any rank bit pattern, with no entry
selected outside the table range. Both return every owned predicate/carry clean.
Their primitive sequence remains composed from the existing equality, encoded
controlled XOR and measured multi-control-X owners.

The in_place_sort_component owner will retain the source xs/pi THRU array
signature for concrete unsigned words. Forward requires distinct input words
and clean pi. Inverse requires sorted words with matching permutation pi and
clears pi. A private sorted copy composes the existing bitonic hierarchy with
padded sentinel keys, followed by the shared rank and word reconstruction
components. Every allocation belongs to this hierarchy and returns clean.
Signed, duplicate and occupied-pi root execution remain separate boundaries.
No new Core admission or user entry is introduced. Users retain circuit.append.

The frozen source SortInPlace has no executable body. Its exact signature,
three direct declared cost edges and lack of root decomposition are independent
from the paper construction. Dependency gates and measured feedback have their
existing frozen-source comparisons. Whole-root primitive counts, complete
complex amplitudes, applicable measurement histories and work cleanup will be
verified against the existing native construction and independent logical
permutation, without inventing an original root gate stream.

### Stored implementation and measured closure

`permutation_index_program(count, width, operation)` now retains both indexing
operations through shared typed equality, encoded controlled XOR and measured
MCX. `rank` XOR accepts occupied ranks. `words` XOR accepts occupied outputs and
all rank bit patterns. All three-entry rank patterns, including the unused
encoding three, preserve the independent indexed-word contract and clean work.
The singleton retains explicit empty rank words. No physical emitter was copied.

`in_place_sort_program(count, width, adjoint=False)` now composes a private sorted
copy with those indexing children. The sorted copy owns padded key words and
calls the existing bitonic Program and its exact adjoint. Sentinel high bits
are prepared and unprepared explicitly. The root preserves the original bounded
pi descriptor via Split/Join views, and both arrays remain THRU. The inverse
reuses the same component stages in the existing paper construction's reverse
order. The normal DynamicCircuit.append path consumes the bound operation.

Five count/width pairs, including singleton, power-of-two and padded three-word
cases, preserve the existing emitter's complete primitive counts in both
directions. Every basis input in their distinct-word subspaces has a valid
measured-And cleanup certificate. Both directions of the two-one-bit-word case
preserve complete complex spectator amplitudes through every raw measurement
history. Padded three-word/two-bit execution additionally preserves the complete
complex distinct-word superposition and clean work under the existing sparse
trajectory executor. That latter channel is sampled, with exhaustive valid-basis
cleanup separately certified. No claim of enumerating all 2**108 histories is
made for it.

The indexing retained-call paths additionally match the existing emitter's
every-wire gate order, all operands and feedback after full temporary-wire
history matching and fresh measurement-slot normalization. Only temporary
physical labels are renamed. The expanded small indexing paths pass complete
complex/branch checks. For four three-bit words the shared expansion scheduler
uses six clean bits versus two with retained calls. Primitive counts are equal.
The expanded larger path is recorded as an allocation difference, with no
larger-width physical placement or gate-order equivalence claimed. The root
retains component calls and its measured peaks fit the existing work budget.

Three two-bit words emit 1608 CX, 338 X, 396 H, 360 T, 324 T-dagger, 108 S,
108 measurements, 108 feedback CZ and 108 resets per direction, with peak
32 owned clean work bits. Independent frozen root estimates are 108 CSwap,
162 And, 1998 Clifford, 162 measurements and peak twelve logical bits. These
models cannot supply an executable root stream. Other measured pairs and
retained/expanded indexing allocations are recorded in
`g2-in-place-sort-resources.json` in the external evidence directory.

### Actual source libraries and retained boundaries

Eighteen unsigned/signed/opaque and concrete/symbolic root variants retain exact
library bytes at depths zero, one and two. Original source reconstruction drops
assumption-bearing symbol identity itself, while plain symbols and concrete
objects roundtrip equal. Native codecs preserve the original wire text and
bytes. Concrete unsigned one- and three-word factories resolve to the typed
sort owner with exact bound circuits and aliases. Other concrete descriptors
and symbolic word widths retain declaration-only resolution. Symbolic list
extents still reject Program construction under Proposal 103 prerequisite 1.20.
The source supplies no factories and its DocSpec has no examples.

At this publication, signed root execution, duplicate inputs and occupied-pi
sorting were outside the unsigned distinct-word contract. The signed extension
below closes the first boundary. Shared indexing operations' occupied XOR
targets still do not extend the sorting root contract. Q-0204 and Q-0204.05
remain open for duplicates, occupied pi, symbolic physical array execution and
the original root's missing executable reference. The source zero-length
QubitCount failure retains its existing audit. Existing Comparator/Xor frozen
primitive and resource evidence is reused without changing reference code.

Final batch checks pass 390 concentrated cases, 17 local integration cases and
55 focused cases in each Cirq 1.7/1.6.1 lane, with no skips or warnings. All six
changed Python files pass strict mypy, Black and Ruff. Existing sort test
fixtures now carry complete annotations so typed consumers reuse the same
independent permutations and resource oracle. Compiler scope, architecture,
multilevel IR and protected API checks pass. Five frozen source blobs, all ten
unchanged shared implementation owners, Qualtran 0.7.0 and Kickmix 0.1 with its
Simulator are verified. The dependency scan retains 685 whole-repository
findings and zero changed-file findings. The previous default docstring failure
baseline's ten files are byte-identical to the batch base. This is concentrated
batch validation, with that existing repository-wide baseline retained.
See g2-in-place-sort-evidence.json and named logs in the external evidence
folder. No parent completion is inferred from these accepted capabilities.


## Approved symbolic constructor continuation

Proposal 120 supersedes historical symbolic-array declaration rejection.
Existing foreign tests now accept bodyless logical signatures and require
physical binding refusal. The dedicated constructor adapter preserves the
original root declaration and missing decomposition. Previously accepted native
physical synthesis and independent source models retain their separate scopes.
See RESOLVER_LIST_AUDIT.md for the complete constructor dependency closure.

## Signed distinct-word execution interface

The native stored synthesis extends `in_place_sort_program` with an internal
`signed` boolean, default false. Its signed xs port uses the original QInt
nominal descriptor and MSB-first two's-complement words. Pi retains the original
bounded THRU array. Forward still requires distinct words and clean pi; inverse
requires ascending signed words with their matching ranks and clears pi.
Duplicates, arbitrary occupied pi and symbolic physical extents remain outside
this executable contract.

XOR of each sign bit maps signed order to unsigned order. Nominal Split/Join
views pass those biased words through the existing unsigned stored sort in
either direction; a second sign-bit XOR restores the original representation.
The unsigned child retains all private sorted-copy, padded-key and measured
predicate ownership. The wrapper owns no new scratch or measurement results.
This is native synthesis of the documented sorting map. The frozen source root
still has no decomposition or classical callback, so no root gate-stream parity
is claimed. Its Comparator/Xor count model remains independent.

The complete stored unsigned child is unchanged at all five existing finite
profiles, both directions, compared to published 7637a662e. The signed wrapper's
primitive operands, per-wire order and measurement feedback match that actual
child with the two sign-bias layers. The older flat synthesis has different
temporary-wire lifetimes, so its physical labels are not used as the oracle for
this wrapper; its primitive inventory remains separately equal apart from the
explicit sign X gates. Full signed basis maps are generated independently from
Python integer sorting and original-to-sorted indices, not by copying the bias
implementation.

Actual QInt source libraries at depths 0/1/2 retain their typed ports and bind
through the same explicit native replacement used for unsigned execution.
Default source constructor factories stay bodyless. Both directions cover all
small distinct signed inputs, including minimum negative, minus one, zero and
maximum positive values. Two one-bit words exhaust every raw measurement
history with complete complex spectator amplitudes and zero private work.
Padded three-word/two-bit channels also preserve complete coherent valid-input
states and clean scratch under the existing sparse trajectory executor; their
exponential raw histories are certified per valid basis rather than enumerated.
Singleton signed sorting is a true identity with zero emitted bias gates.

The extension adds exactly two X gates per word for non-singletons, no extra
measurements, no T-family gates and no private workspace beyond the unsigned
child. Independent original QInt root models remain separate from native
emission. The ten directional resource rows and unsigned compatibility proof
are recorded in `g2-signed-sort-resources-20261125.json`.

For three two-bit signed words, each direction emits 344 X, 1608 CX, 396 H,
360 T, 324 T-dagger, 108 S, 108 measurements, 108 feedback CZ and 108 resets,
with 32 peak owned clean bits. The original model remains 108 CSwap, 162 And,
1998 Cliffords, 162 measurements and twelve logical bits. These different
levels are not combined or relabeled as equal gate counts.

The concentrated stored-sort/index/list regression passes 281 cases. The full
sort/source-interface lane passes 76 cases under Cirq 1.6, and local integration
passes 145. All three changed Python files pass strict typing, Black and Ruff;
scope, language, links, architecture, IR and protected API checks pass. Frozen
source identity and real Kickmix 0.1 Simulator are verified. The first literal
comparison used the unrelated flat temporary labels and failed; the final
comparison above uses the unchanged stored child actually invoked, preserving
all operand, feedback, semantic and workspace assertions at their stated scopes.


## Whole-file acceptance under the approved closure rule, 2026-11-28

Q-0204 closes the entire 99-line source: both fields, THRU xs/pi signature,
bounded index dtype, exact Comparator/adjoint/Xor multiplicities and empty
DocSpec. There are no original factory functions, classical callback or root
physical decomposition to omit. Default source factories retain bodyless typed
declarations; concrete/symbolic and signed/unsigned/opaque metadata, models and
foreign transports are already owned and verified. Proposal120 covers logical
symbolic array declarations; unknown-size physical execution still refuses.

The extra native paper construction is not needed to pretend that the source
contains a gate body. Its existing distinct signed/unsigned sorting, clean pi,
matching-rank inverse, complete small complex vectors/history/work-zero and
source-comparable child streams retain their accepted scope. Duplicate inputs
and arbitrary occupied pi remain outside that independent synthesis contract.
Original no-body, zero-length peak and symbol-assumption-loss boundaries stay
explicit; larger expanded-index allocation/order differences remain native
optimization follow-ups. They do not block migration of a complete model-only
source. No root primitive equality or duplicate-capable physical sorter is
claimed. Earlier open conclusions are superseded for this source-file parent.

Current acceptance receipts use `g2-list-primitives-full-closure-*20261128`:
842 concentrated cases and 99 independent Cirq1.6 cases pass; 109 frozen blobs
are unchanged. Existing original full tests/notebook, exact primitive/model
resource rows and strict owner checks are reused. This batch changes only
these complete source dispositions. The immediately preceding 32 local
integration checks remain valid for the unchanged list owners. Compiler scope,
English/links, architecture/IR/API/required checks and whitespace pass; reference
policy retains 570 existing findings with no new consumer or waiver. No source,
algorithm, public contract, checker or dependency exclusion is changed.
