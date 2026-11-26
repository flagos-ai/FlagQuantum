# Complex phase constructor and exact encoder ownership

Reference: complete unchanged Qualtran v0.7.0
`mean_estimation/complex_phase_oracle.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Interface recorded before implementation

The constructor consumes the actual encoder and arctan_bitsize (default32).
An explicit native encoder pair supplies exact forward/inverse definitions,
including private workspace and measurement feedback. Target names identify
the actual source target registers, not a guessed suffix of the signature.
All remaining encoder ports are caller-owned through controls and selection.
Encoder targets are privately zero-initialized, retained through phase
evaluation, then erased and freed. Packed and shaped targets concatenate in
declared target order, row-major by element, MSB-first by bit.

The shared component follows encoder, ArcTan, alternating zero/one-controlled
Z powers from most to least significant magnitude bit, exact ArcTan erasure,
and exact encoder erasure. Phase sign/magnitude and encoded targets are released
only after cleanup. The source's ArcTan has no primitive decomposition. This
adapter explicitly selects the already accepted bounded native XOR lookup,
including its exact measurement cleanup; it does not claim an upstream ArcTan
gate recipe. Widths outside its admitted domain remain unsupported.

Cirq owns translation of the literal controlled phase gates to existing native
definitions, preserving raw S/T specializations rather than substituting an
equal-unitary CZPow constructor. External objects stop there. Compiler receives
complete Programs only; normal binding, execution and circuit.append(op) do not
need source SDKs. The old public flat encoder contract remains unchanged.

## Closed constructor and direct dependency path

The complete source class, properties, signature, generator and diagram contract
are mapped to the existing native declaration and the new typed constructor.
Original serializer depths 0/1/2 retain encoder and arctan_bitsize metadata. The
factory validates the actual encoder ports, output signature and exact requested
precision, including default32. It never infers target names or an inverse from
port positions. The Ecosystem phase gate owner stops external objects at its
existing register importer; Compiler composes owned Programs only.

Encoder, ArcTan and phase-stage private lifetimes remain separate. The component
joins explicitly declared target-order bits for the lookup and restores their original
typed grouping before exact encoder erasure. A source encoder whose full
signature reverses its declared target list verifies that these orders are
independent; no positional-signature inference is used. It preserves the source generator's
final release of sign, magnitude and targets. The public control and selection
words survive. No protected flat make_on signature or encoder allowance changes.

Complete complex operators include coherent spectators, reversed placement,
zero precision, one/two-bit encodings and an explicit measured encoder with two
separate target words. Both encoder directions independently match unchanged
source snapshots, including their And feedback and nontrivial S/S-adjoint phase.
All applicable small histories retain raw branch weights and return every work
bit to zero. A source-success controlled encoder retains its public control and
has identity action on the inactive branch. Its metadata and typed dependency
path are exercised separately from the original broken controlled ExampleSelect.

Every actual phase gate agrees with the original literal generator. Independent
source SDK snapshots compare both control polarities at bit powers0/1/2/7/31:
operands, classical feedback, angles, instruction counts and per-wire order match.
The follow-up mean-estimation batch checks every matrix payload exactly as well.
Single S/T phase aliases are normalized for comparison; no gate is removed,
fused or resynthesized. Raw controlled phases retain their source-selected SDK
recipe rather than an equal-unitary named CZPow substitute.

Construction runs with Qualtran blocked. A pre-resolved replacement context
produces an identical complete Program with both Qualtran and Cirq blocked.
Normal append, CircuitIR JSON roundtrips and dynamic trajectories then execute
without either SDK. Existing bounded ArcTan lookup tests independently compare
its tables to original apply, its QROM operands and feedback to original QROM,
and its every-history channel. The maximum12/32 table comparison is reused.

## Original failures and reference protocols

Original ArcTan has no tensor/decomposition implementation. For ordinary copy
encoders the independent full operator comes from the original literal Cirq
operation sequence, where the ArithmeticGate unitary protocol is available.
The original whole-circuit unitary request refuses a measured private encoder's
classical feedback. That exact failure is retained; each independently available
original encoder tensor supplies its logical operator inside the otherwise
unchanged literal sequence. This evaluates existing source protocols rather
than editing or repairing source code. Native raw measurement branches remain
independently checked, including their absolute phases and normalization.

The original generator allocates shaped target words as flat lists, then rejects
shape(2,1) versus shape(2,). The SDK adapter retains that physical refusal instead
of silently reshaping it. The zero-width ExampleSelect source graph also fails
with an unavailable Soquet; no executable claim is inferred for that fixture.
The old controlled ExampleSelect omits its control parameter and remains a
separate source failure. No universal encoder reflection is implied by an
explicit exact child owner.

The ArcTan lookup domain remains source_width0..12 and target_width0..32. Wider
exact encodings, unavailable codec/graph contexts and generic outer wrappers
are not accepted by this batch. The source's symbolic opaque arithmetic model
remains distinct from finite native lookup execution. Q-0564 and whole serializer
parents Q-0866/Q-0876 remain open; no new upstream-defect waiver is inferred.

## Actual primitives and independent declared models

Six rows captured in the actual Cirq1.7.0 profile in
g2-complex-phase-constructor-resources-20261126.json separate actual
IR inventories and event-derived live work from declared source costs. Actual
instruction totals below include the listed explicit quantum matrix primitives. Counts
are neither T-equivalent estimates nor a synthesis model for opaque ArcTan.

| Fixture | Encoding / precision | Actual instructions | Matrix primitives | Actual measurements | Actual private peak | Source model peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Copy | 1 / 0 | 8 | 0 | 0 | 2 | Existing model rejects zero precision |
| Copy | 1 / 1 | 36 | 3 | 1 | 4 | 5 |
| Copy | 1 / 3 | 88 | 9 | 3 | 6 | 13 |
| Copy | 2 / 1 | 82 | 3 | 3 | 5 | 7 |
| Private measured encoder | 2 / 1 | 118 | 3 | 5 | 5 | 7 |
| Controlled private encoder | 2 / 1 | 118 | 3 | 5 | 5 | 8 |

All leases return. Five available seven-category native declaration models match
the source, including symbolic _c0 multiplication counts and both independently
priced encoder directions. The zero-precision model guard remains unchanged.
The source model's generic controlled phase price can differ from its SDK
execution: precision3 declares four measurements while its six literal SDK
phase stages execute three. Both inventories are retained. Native QROM
implementation of opaque ArcTan likewise adds real feedback absent from that
original symbolic product estimate. Neither difference is hidden by adjusting
measured gates to the model.

## Batch validation

168 concentrated cases pass without skips in each actual Cirq1.7.0/Qiskit2.5.0
and Cirq1.6.1/Qiskit2.0.0 profile;145local integration cases pass. Four files pass
strict typing, Black and Ruff. Both scopes, English/local links, architecture,
internal IR and protected API checks pass. Five frozen source/test/dependency
blobs match the locked commit. Qualtran0.7.0 and actual Kickmix0.1 Simulator are
asserted. Existing original13-case and full-file numerical/model receipts are
reused after identity checks. Registration union204/303 is not an acceptance
counter. Pending118/119/124/126/127 and the unrelated127 file are preserved.
No scientific installation, ReadTheDocs, CI wait, G3 transition or merge occurs.

2026-11-26 resource-label correction: any is an explicit quantum matrix
primitive, not a classical OR. Original total inventories are unchanged; the
table heading and prose have been corrected. No raw numerical receipt changed.

The subsequent [mean-estimation dependency audit](MEAN_ESTIMATION_COMPONENT_AUDIT.md)
independently checks the default-32 literal SDK stream against exact gate
matrices. Original tiny zero-control phase decompositions lose phase: a retained
Bernoulli witness differs from the ideal complex operator by approximately
`1.1572070829341236e-5`. Per-wire source stream equality through bit 31 therefore
does not establish exact default-precision mathematics. This source accuracy
gap remains open; no source change, global-phase alignment or defect waiver is
part of the constructor delivery.
