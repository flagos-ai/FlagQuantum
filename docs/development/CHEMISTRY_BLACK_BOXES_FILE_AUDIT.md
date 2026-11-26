# Chemistry black-box source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen chemistry files
are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0335 | `black_boxes.py` | `f9670a2ebaf2fe0973323cf7b532c88142970902` | Accepted complete definition/direct-dependency closure. Actual foreign model declarations and stored phase graphs are covered in the November 21 section. |
| Q-0336 | `black_boxes_test.py` | `23cfe1e1bae3f437837e031348a0d09bfe201bf0` | Closed for all four original OpenFermion comparisons and the corresponding native resource queries. |

## QROAM estimates

The Algorithms resource owner keeps the unrounded clean and dirty block
scores and the one- and two-axis integer Toffoli estimates. It reuses the
Compiler's clean forward block-exponent selection. The two directions
select block size independently, and explicit block size one keeps the
source's rows-minus-one rule. The two-axis placeholder uses both axes
in the cost but only the first axis in the signature. Its selection
wire width is `(first - 1).bit_length()`; a zero-width selection is
omitted. The target is a through word in both directions.

The four original OpenFermion comparisons ran unchanged in an isolated
OpenFermion 1.6.1 environment. The native tests check the same four
data-size and target-width pairs in both directions against the locked
Qualtran formulas. They also check explicit blocks, concrete port order,
single-row selection omission, two-axis cost and the dirty score.
The source QROAM types are resource placeholders. Their estimates supply
no physical gate stream or numerical lookup evidence. Executable lookup
stays with the existing Compiler table and erasure components. The
declared dimensions are concrete; no symbolic extension is inferred.

## Controlled number phase

The Algorithms binding keeps the source control polarity, control-array
shape and through system register. It acts on the most significant system
wire and leaves the remaining system wires untouched. The caller owns
the predicate work wires and classical result slots. The Compiler
MultiControlZ emitter supplies gates and feedback. DF and THC use the
same resource declaration owner.

| Controls | Emitted predicate plus phase | Original wrapper estimate |
| --- | --- | --- |
| One | CZ with any negative-control X pair, zero T or measurement | Zero Toffoli |
| Two | Four T/T-dagger, one measurement, one work wire | One Toffoli |
| Three | Eight T/T-dagger, two measurements, two work wires | Two Toffoli |

The estimates above retain the source wrapper's explicit Toffoli-only
call graph. All matched primitive streams, including measurement
feedback, are checked independently. Complete complex states spanning
all control and system values agree with the expected sign pattern in
every applicable history. Work returns to zero.

Drawer reads the declared control array index for each filled or empty
circle. The system port uses a Z box. The connector retains every C
followed by Z, even for ten controls. Six score fixtures match the
unchanged source geometry and glyphs. The empty control array draws Z
on the system. Its frozen decomposition passes an invalid controls
keyword to ZGate and its declared Toffoli count is minus one. Native
execution continues to admit nonempty systems with at least one binary
control. Actual external object handoff is completed in the November 21 section.

## Complete source and external consumers, November 21

Q-0335 closes the complete 19-definition source file and its direct dependencies.
The original four tests in Q-0336 also pass unchanged in OpenFermion 1.6.1 during
this batch. The source stays byte-identical to the frozen inventory. The earlier
control-geometry, numerical recipe, all-column and full-branch evidence is reused.

### Ownership of every definition

| Source definition | Native owner and complete disposition |
| --- | --- |
| `qroam_cost` | `chemistry_qroam_model.chemistry_qroam_block_score`, clean scalar score |
| `qroam_cost_dirty` | The same owner with `dirty=True`, original inverse and forward formulas |
| `get_optimal_log_block_size_clean_ancilla` | `chemistry_qroam_log_block_size`, independently callable forward/inverse query, explicit blocks and floor-first ties |
| `get_qroam_cost_clean_ancilla` | `source_chemistry_qroam_cost`, including original explicit-block-one rows-minus-one rule |
| `QROAM` | Immutable native dimensions, signature, directional resource model and owned external declaration |
| `QROAM.signature` | `chemistry_qroam_signature`, first selector and target word |
| `QROAM.build_call_graph` | Native model Toffoli count, actual original serialized direct counts and source QEC query |
| `QROAM.adjoint` | Existing model's `adjoint` argument, with actual foreign direction flag verified |
| `QROAM.__str__` | Source presentation title retained as metadata; native model carries its domain name |
| `QROAMTwoRegs` | Existing two-axis model retains both axes and both block dimensions |
| `QROAMTwoRegs.signature` | Original first-axis-only selector and target signature retained |
| `QROAMTwoRegs.build_call_graph` | `source_chemistry_qroam_two_axis_cost` and actual foreign direct counts |
| `QROAMTwoRegs.adjoint` | Existing two-axis model direction, actual foreign flag and double adjoint |
| `QROAMTwoRegs.__str__` | Original QROAM/dagger title verified independently of native two-axis model identity |
| `ApplyControlledZs` | `ChemistryNumberPhase` plus existing Compiler MultiControlZ and Ecosystem named capture |
| `ApplyControlledZs.signature` | Original shaped QBit controls and packed through-system word |
| `ApplyControlledZs.wire_symbol` | Existing `number_phase_wire_symbol`, including empty controls and ten-control labels |
| `ApplyControlledZs.build_composite_bloq` | Existing split, most-significant target, measured conjunction/erase and join; actual stored source graph capture verified |
| `ApplyControlledZs.build_call_graph` | Original wrapper Toffoli-only declaration retained separately from measured recipe costs |

The standalone exponent query was the remaining numerical interface gap. It now
uses the existing forward selector and the existing chemistry scalar-score
owner for inverse and explicit-block selection. Existing cost calculation calls
this shared query. It retains the source floor-before-ceiling tie ordering and
the existing native validation. The original forward query for fewer rows than
target bits raises AssertionError; native raises its existing domain ValueError.
The concrete positive-dimension model domain remains explicit.

### Actual foreign declarations and executable handoff

Actual one- and two-axis source placeholders serialize at depths zero, one and
two, roundtrip exact bytes through the owned library codec, reconstruct in the
unchanged source resolver, and reproduce native signatures, direct counts,
QEC Toffoli counts and logical peaks from their constructor arguments. Original
placeholder decomposition raises DecomposeNotImplementedError in both
directions. Their original scope is an analytical declaration. Physical lookup
continues to use the existing executable table owners.

The actual `ApplyControlledZs` decomposition is flattened by unchanged source
code and passed to the existing `import_register_emitter`. That named boundary
captures the stored graph and its actual adjoint without depending on the source
serializer. Literal primitive operands, per-wire ordering and feedback match
independent expansion of the actual source root and its actual root adjoint.
Fresh classical slot names are compared after checking each feedback read's
latest measurement producer. The executed streams retain their original slot
names. Every small measurement history matches complete unnormalized complex
states entangled with an external spectator, returns clean work to zero and
passes ordinary append and CircuitIR roundtrip with SDK imports blocked after
capture. The source graph path retains measured AND cleanup. Direct SDK synthesis
of the outer root has its own primitive recipe.

The flattened source graph's original serializer fails on numpy.uint8 control
arguments. The original outer component serializes successfully before recursive
flattening. This source interchange defect is retained as an explicit test and
separate evidence. The original source code receives zero edits. The already
recorded zero-control construction error and minus-one wrapper count remain
separate failure/model dispositions.

| Control values, both actual directions | Measured T-family / CX / CZ / measurements / clean work | Source wrapper Toffoli / logical peak |
| --- | --- | --- |
| `0` | 0 / 0 / 1 / 0 / 0 | 0 / 3 |
| `1` | 0 / 0 / 1 / 0 / 0 | 0 / 3 |
| `01` | 4 / 6 / 2 / 1 / 1 | 1 / 5 |
| `101` | 8 / 12 / 3 / 2 / 2 | 2 / 7 |

CZ counts include measurement-feedback CZ instructions. System width is two in
these external captures. The external spectator is excluded from source peaks.
All eight directional resource receipts are retained independently.

### Verification and evidence provenance

Concentrated regression passes 262 cases, local integration passes 17 and the
Cirq 1.6.1 lane passes 86 with zero skips or warnings. Both changed Python files
pass strict typing, Black and Ruff. Algorithms scope, architecture, multi-level
IR and protected API checks pass. The external evidence prefix for this closure
is `g2-chemistry-black-boxes-closure-`.

The initial verification reused the November 16 batch's external log names and
overwrote those historical raw outputs. Those old raw logs are unavailable.
The original commits, historical audit and historical PR verification receipt
remain intact. Current outputs are independently retained under the closure
prefix, and all four original source tests have been rerun successfully. The
provenance receipt records the exact affected log roles. Historical test counts
continue to refer to the historical acceptance, and current checks refer to the
current commit's closure logs.

Five frozen source blobs and six unchanged native owners are verified. A separate
480-input comparison against the exact pre-batch native cost implementation
retains every supported result and domain-error message. Dependency policy keeps
666 existing findings and zero changed-path findings. Evidence and eight separate
measured/model rows are retained under the closure prefix.

## Exact placeholder constructors: interface recorded before implementation

The remaining exact library provider gap is QROAM/QROAMTwoRegs. Ecosystem owns
translation into bodyless native FunctionRecord declarations and reuses the
existing Algorithms signature/model owners. Existing ApplyControlledZs binding
is reused from its current provider; no duplicate phase emitter is added.
Original dimensions, direction, optional None/explicit blocking metadata and
source counts remain in the owned library. The typed Program is deliberately
nonexecutable because the original placeholders supply no physical bodies.

Both placeholders use MSB-first caller-owned sel/trg words. A single row omits
the selector. Two-axis signatures use the first axis only; the second axis and
both blocks affect the model, not extra quantum ports. Existing positive,
concrete dimension/model admission is preserved, with explicit refusal of
zero-target/symbolic/invalid metadata. No physical lookup, measurements,
complex-state execution or clean workspace is invented for these declarations.
The earlier complete source acceptance is reused, not counted again.


### Verified exact constructor and narrow model closure, November 27

Both placeholder keys now resolve actual source libraries at levels zero, one
and two through `ecosystem/_chemistry_black_boxes_library.py`. Wire round trips
retain dimensions, is_adjoint, absent/None/explicit-one/nonpower block metadata
and source counts. Resolution succeeds with SDK imports blocked. One-row
selectors are omitted; two-axis costs still depend on both axes while the
signature uses only the first. All resulting placeholder definitions have no
body and explicitly refuse physical binding. The same provider reuses the
existing ApplyControlledZs factory; exact definition equality verifies reuse.
No execution, work, measurement or emitted gate count is claimed for QROAM.

New one- and two-bit target cases exposed an existing native model gap. The
original QubitCount fallback takes the maximum of signature width and its
three-qubit Toffoli callee, even when multiplicity is zero. The native models
now preserve that local model peak without changing logical ports or creating
work. For QROAM(1,1,block=1), the original model has zero Toffoli and peak three
although the signature is one bit. QROAMTwoRegs(1,7,1,3,2) has seven forward/six
inverse Toffoli and peak three with two signature bits. These are analytical
source semantics, not a physical implementation or a source defect waiver.

An independent grid compares 330 valid inputs against unchanged source peaks
and exact published20a08c460 native gate counts. All Toffoli values are
unchanged; 71 narrow-width peak corrections now match original QubitCount.
Six prior domain refusals retain their exact native messages. Automatic forward
rows<target-width still has the original AssertionError and the existing native
model ValueError. Positive concrete constructor declarations can exist even
when that independent cost query refuses. Zero targets, symbolic dimensions,
invalid blocks and nonbinary direction remain outside this native admission.

The two actual Cirq1.7.0/Qiskit2.5.0 and Cirq1.6.1/Qiskit2.0.0 profiles each
pass 266 concentrated cases without skips; local integration passes145.
Three Python files pass strict typing, Black/Ruff, Algorithms/Ecosystem scopes,
English/links, architecture, IR and public API baselines. Two frozen source/test
blobs remain unchanged. Existing complete source and original four OpenFermion
comparisons are reused, not represented as new parent acceptance.
External evidence uses `g2-chemistry-placeholder-*-20261127`; resource receipt
`g2-chemistry-placeholder-resources-20261127.json` separates source peak,
signature width and unavailable primitive inventory. Q-0335/Q-0336 retain their
previous acceptance. Exact registrations246/303 are separate from that status.
