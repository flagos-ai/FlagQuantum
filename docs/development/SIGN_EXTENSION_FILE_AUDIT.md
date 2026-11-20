# Sign extension and truncation whole-file audit

Q-0195 and Q-0196 cover the complete
`arithmetic/conversions/sign_extension.py` and its original test file at
Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`.
Both Git blobs are checked. The source defines two classes with dtype validation,
sided signatures, width properties, adjoints, decompositions, cost graphs and
classical callbacks, plus two factories and their documentation specifications.

## Native ownership and full method disposition

| Source responsibility | Native owner and evidence |
| --- | --- |
| SignExtend dtype validation and signature | Existing Core signed/one's-complement types and register sides retain width and encoding metadata. Existing source/native width and value tests retain constructor failure ordering. The physical emitter admits concrete nonempty disjoint word/prefix wires. |
| SignExtend extend_bitsize, adjoint, decomposition | Compiler `registers/bits/sign_extension.sign_extend_ir` binds the shared fanout into a caller-owned zero prefix. Logical output is prefix+source. The same physical operation clears a correct sign prefix for truncation. |
| SignTruncate dtype validation, signature and truncate_bitsize | Existing signed truncation value tests compare eighteen width pairs and both dtypes, including NumPy, symbolic widths, failure ordering and negative-zero encodings. |
| SignTruncate adjoint and decomposition | Shared fanout clears redundant prefix before release. Exact source/native gate streams, complete coherent values and zero prefix are already checked. The source Join dtype loss is explicitly retained below. |
| SignExtend classical callback | The source returns the same input value. Encoding and caller-owned arrays retain their existing value ownership, without inventing a scalar transformation. |
| SignTruncate classical callback | Simulation `classical_signed_truncation.signed_truncate_classical_value` preserves metadata construction before integer conversion, original prefix slicing, redundant-bit validation and output decoding. Existing replacement tests swap codecs without changing consumers. |
| Both cost graphs | The source declares one MultiTargetCNOT over the width difference. Existing dependency models and primitive inventories remain separate. |
| Both factories and DocSpecs | Actual eight-to-sixteen-bit extension and sixteen-to-eight-bit truncation identities, width properties and adjoint equality are checked. The factories compose into a real through-word round trip after explicit original decomposition. |
| Default wire symbols | Existing native score defaults already match both complete sided source scores, including directional port glyphs and source title truncation. No new drawing callback is needed. |

The direct dependencies remain Core types, Compiler shared fanout and
Simulation signed codecs. The ordinary physical workflow continues to use
`circuit.append(op)`. No compiler or simulation behavior moves into the drawer.

## Actual capture and exact inverse ordering

The original undecomposed factory composite fails its generic Cirq wrapper
with a wrong-number-of-qubits error on the sided SignExtend gate. The test
retains that error, then calls the original `flatten_once` before capture.
No source file or decomposition is rewritten. This supports the existing
explicitly expanded capture route, not undecomposed sided admission.

Both directions of the expanded composite bind a reversed physical source word
and eight clean prefix wires. All 256 word values and a coherent spectator are
present simultaneously. The intermediate prefix carries the sign, and the
second half restores every prefix wire to zero and the complete complex input.
IR JSON and append preserve the captured stream. The forward stream is two
native fanouts. The generic source composite adjoint reverses every primitive
within each fanout, including the order of commuting gates in a layer. That
reverse stream is asserted explicitly and is distinct from calling the semantic
self-inverse fanout in its forward order.

Each eight-wire fanout emits fifteen CX, so the round trip emits thirty CX,
zero T and zero measurements. The source coarse model reports two Clifford
calls and peak sixteen. A coherent spectator makes seventeen physical wires in
the verification circuit. No equality between coarse Clifford calls and the
thirty emitted CX is claimed.

## Original tests and retained boundaries

The untouched original file has nine test functions and one unsigned encoding
helper. Tests cover examples, adjoint identities, incompatible dtypes, signed
extension/truncation tensors and classical actions, and invalid prefix refusal.
It reports sixteen passes, one skip and one xfail. The skipped strict dtype
check reports QAny(8) feeding the declared QInt(8) output after SignTruncate's
final Join omits its dtype. The new source witness confirms both Join children
are QAny while the output signature remains QInt. The original counts fixture
is explicitly unenforced and xfails. These outcomes remain separate from native
regression results.

Q-0195 and Q-0196 retain that source dtype defect, undecomposed sided capture,
symbolic typed Program admission and original skipped/unenforced modes. Existing
one's-complement minimum width and invalid-prefix checks remain unchanged.
Batch logs are `ftqc-reference-env/g2-sign-extension-file-*.log`.

The concentrated batch passed 902 cases without warnings or skips. The new
source-capture/score test passed strict typing, formatting and lint. Compiler
scope, architecture, multilevel IR and protected API checks passed. Existing
subtraction consumers and all signed truncation value tests were included.

## Complete native interchange and sided SDK closure

The final closure maps all eighteen recursive class/function definitions, two
factories, two DocSpecs, nine original test functions and the original unsigned
encoding helper to the owners above. Q-0195/Q-0196 close with the original dtype
and wrapper failures retained as source witnesses. The preceding open-item
paragraphs describe earlier checkpoints.

Both actual original factory objects pass original serialization and exact
native library-byte roundtrips at depths zero, one and two. Existing Compiler
records retain constructor dtypes, signatures, decomposition edges and counts.
Signed and one's-complement symbolic extension/truncation declarations resolve
through the existing explicit native factory boundary. Unresolved physical
widths retain their existing refusal. General arbitrary source-type coverage
remains tracked by Q-0937.

Existing native `sign_extension_program`, Program SDK export and optree import
provide executable sided transport. Both encodings, both directions and width/
prefix pairs 2/1, 2/3, 3/2 and the actual 8/8 factory preserve logical ports,
returned aliases, source integer metadata, full gate dependencies and resource
counts. SDK input Qids are reversed. Inputs are leased by the supplied manager,
so the same manager can release a consumed prefix. Every surviving output lease
returns to that manager and can be reallocated, demonstrating lifecycle closure.
The bound Program uses the export contract's consecutive input wires.

Fresh output storage follows first gate use on import. The independently
observed complete input/output port map identifies each physical wire. Every
wire's complete ordered operation sequence, opcode, operands and total gate
count agree under that bijection. Prefixes up to three also retain the full
mapped global instruction sequence. For the eight-bit prefix, SDK moment
scheduling reorders independent fanout gates. No gate is removed or sorted in
these comparisons. The physical port mapping and independent scheduling are
explicit representation adaptations. Original scalar source/native primitive
order and exact adjoint evidence remain in the accepted component tests.

All tested small widths verify full complex amplitudes with an entangled
spectator under the same physical mapping. Extension copies the raw sign,
including one's-complement negative zero. Truncation consumes a correctly
redundant prefix and leaves every consumed bit at zero. There are zero
measurement branches. The existing actual eight-bit factory roundtrip test
revalidates all 256 raw words and a coherent spectator through ordinary
`circuit.append(operation)`. Source classical behavior, eighteen-width-pair
validation, shared fanout replacement, source scores and resource models are
reused in the concentrated regression.

The unchanged original generic sided wrapper still fails before explicit
source flattening. Its final truncation Join retains QAny rather than the
declared QInt output. Original strict qtyping therefore remains skipped and its
unenforced count fixture remains xfailed. Native typed SDK transport and those
source outcomes have separate evidence. The reference code is unchanged.

The final concentrated run passes 967 cases without warnings or skips, plus
17 integration cases. The focused and Cirq 1.6.1 lanes each pass 22 cases.
The untouched original test file retains sixteen passes, one source skip and
one source xfail. Strict typing, formatting, lint, team scope, architecture,
multilevel IR and protected API checks pass. Reused production/test owners and
the ten implicated default-run docstring-policy files are byte-identical to
the batch baseline. Whole-repository status retains that documented policy
failure. Dependency policy has 687 existing findings and zero in changed paths.
Receipts and logs use `ftqc-reference-env/g2-sign-extension-foreign-*`.

An e-bit prefix emits exactly 2e-1 CX in each direction. Each actual eight-bit
factory emits fifteen CX, while the combined roundtrip emits thirty. Quantum
storage is the visible wider word, with zero hidden scratch, T and measurement.
The independent original model charges one coarse Clifford per fanout, or two
for the combined roundtrip with peak sixteen. SDK transport adds zero gates or
storage and retains these separate accounts.
