# Allocate and Free source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/bookkeeping/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0310 | `allocate.ipynb` | `8d91513e62d1015fcbd97c0601442916e3ba6796` | Closed by the complete native Allocate and Free tutorial. |
| Q-0311 | `allocate.py` | `e459095285de98c1468b661afdbcbe58238ff1ad` | Approved 116/125 storage, direct SDK lifetime hooks and symbolic drawing verified. Open for signed-empty 124 and symbolic field declaration 126; original field codec and empty callback failures retained. |
| Q-0312 | `allocate_test.py` | `96f411955b10a466d37e8ba3b0698a0823d42ba9` | Closed for the runnable source scenarios and notebook. |
| Q-0323 | `free.py` | `8a0c73db71e1d8ffa31b2c7a15d246f02f8f4f63` | Approved 116/125 storage, direct SDK lifetime hooks and symbolic drawing verified. Open for signed-empty 124 and symbolic field declaration 126; original field codec failures retained. |
| Q-0324 | `free_test.py` | `63ffa89314718cfaa94bfb0104953e52fd1c518e` | Closed for the runnable symbolic fixture and ten-wire utility round trip. |
| Q-0309 | `_bookkeeping_bloq.py` | `32b9b7498417c20ee8ca86c8e53f16b2d63bada4` | Complete base control protocol accepted in 6d6460528; see BOOKKEEPING_CONTROL_FILE_AUDIT.md. |

## Lifetime, bit order and ownership

Compiler Program records own the typed operations and physical binding. Allocate
creates a right-side `reg` word and Free consumes a left-side `reg` word.
The adjoint exchanges their roles while preserving the dirty flag. Packed words
retain big-endian order and physical bindings use the caller's pool order.
Clean work comes from the clean pool; borrowed work comes from the separate
borrowed pool. The allocator tracks provenance, so a mismatched declaration
at release does not transfer ownership to the other pool. Numerical callers
must restore clean wires to zero and restore borrowed wires with their joint
state intact. Control passthrough does not guard either lifetime operation.

The classical source allocation decodes an all-zero word under both flags.
Classical Free rejects nonzero clean words and accepts nonzero dirty words.
The source tensor views contract a zero ket or bra even with the dirty flag.
Native numerical pool binding preserves the live borrowed state according to
the caller contract. The tensor zero projection is a separate diagnostic;
its amplitude is intentionally unnormalized. Existing Cirq probes compare
ordered qalloc, qborrow and qfree events with native leases. The source
PennyLane hooks return no operation. Native emitted circuits likewise have
no lifetime gate. Actual direct hooks are now compared with the existing native
explicit factory and pool owners across all eight supported foreign dtype
families, including both dirty flags.

Concrete typed component scores identify lifetime calls by their primitive
bodies. Allocate and Free reuse the existing default arrows and empty
connectors. The glyphs and full layout match the frozen scores for one
and three bits under both dirty flags. The tested source `qalloc` or
`qborrow` and `qfree` calls match the native lease events on the same
reversed pool. Both bindings produce zero gates. The physical circuit
reserves the pool and one spectator. The source qubit model visits its
disconnected spectator separately and reports the pool width. The
locked symbolic glyph has the same labels. Internal native typed component
scores now retain symbolic width metadata through geometry and export.

## Shared bookkeeping control

The unchanged base callback returns the original control tuple and invokes
one child with only its data bindings. The native test family covers
Allocate, Free, ordinary Split and Join, Partition in both directions,
Split2, Join2, and Cast. Positive, negative, shaped and grouped controls
exercise one actual native call and one actual source call per case.
New wire-map checks compare the source Cirq output maps for pair rebinding
and shaped Cast. Complete complex states agree on the bound circuits,
including their idle controls. These callbacks emit zero gates and require
no predicate workspace. Allocate and Free retain their existing pool events.

The shaped CBit Cast keeps its two original classical slots. The only
emitted instructions initialize the caller's two input bits. The locked
graph accepts the shaped Cast call, and its numerical callback raises
on the array. Scalar source callbacks provide the value oracle for each
element. The two finite-field polynomial descendants live under
`gf_poly_arithmetic`. Their binary coefficient views also use the shared typed control callback.
Generic descriptor admission remains open in Q-0533.

## Original cases and resource accounting

| Fixture | Native evidence |
| --- | --- |
| Symbolic Allocate and Free | Directional typed signatures, atomic leaf multiplicity one, zero QEC gates, independent source peak `Max(0, n)` under both flags. |
| Allocate, ten H, Free | Ten emitted H on the allocated word. Full contracted scalar is `1/32` and squared norm is `1/1024`, matching the locked reference. |
| Allocate, Split, Join, Free | Original ten-wire identity flow, allocation and release in original order, no output word or emitted gates, full zero state after append. |
| Previous controlled and mixed-flag paths | Reuse the complete gate, allocator ordering, measurement feedback and complex branch evidence in the corresponding workspace lifetime tests. |

The two ten-wire fixtures reserve ten physical wires. The first fixture's
H layer and tensor contraction are distinct from the zero-gate allocation
and free model. The second fixture has no primitive gates or measurements.
The native runner retains both symbolic leaf graphs and both directional
register diagrams. The unchanged 18-cell original and 20-cell native asset
pass independent kernels. The unchanged original tests have six passes and
six mode skips with the notebook executed separately. The mode skips cover
the original atomic decomposition, call-graph comparison and type checking.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_allocate_free.ipynb
```

## Complete foreign factory and lifetime-body consumers

The actual `_alloc` and `_free` examples serialize and deserialize unchanged at
library depths zero, one and two. Their symbolic signatures resolve through the
existing native library resolver and existing `allocate_word_function` and
`free_word_function` bodies. Symbolic lifetime operations are valid logical IR;
the physical binder rejects unresolved executable widths. No surrogate
reference implementation or additional declaration type is required.

Concrete Allocate/Free roots round-trip the original wire format and resolve to
those same owners for QBit, QAny, QUInt, QInt, QFxp, QIntOnesComp, BQUInt and
QMontgomeryUInt. Both clean and dirty flags preserve their serialized value and
native exact boolean. Composition calls each leaf once, allocates and releases
the same reversed pool wires, and emits zero primitives or measurements. Clean
storage starts at zero and dirty storage retains an arbitrary complete complex
state entangled with a spectator through `circuit.append(op)`. The original
zero-initialized classical callback and original zero-gate/qubit-count models
are queried separately. The spectator increases caller capacity by one wire.

The original serializer rejects QGF and QGFPoly before producing a wire-format
library. These exact source TypeErrors are asserted independently. Approved
native binary-field lifetime behavior continues to use its earlier typed
execution evidence; this batch makes no successful original-wire claim for those
dtypes. It adds no new production owner or third-party adapter.

At this historical publication, Q-0311 and Q-0323 still awaited complete dtype
admission, direct SDK hooks and symbolic drawing. The later sections verify the
hooks and drawing; the approved 116/125 storage closure below supersedes the
old fixed-point and nonbinary storage blockers. Original codec and callback
failures remain separate from native physical storage.

## Regression disposition

The focused foreign profile passes 26 cases. An expanded consumer regression
reports 1653 passes and ten failures. Eight compare mixed-lifetime event order
and two expect an omitted-selector refusal. All ten reproduce with the
pre-batch `9f03b3b11` bookkeeping control owner loaded before test collection;
every other production file is unchanged between that baseline and this batch.
The baseline run is retained in
`g2-workspace-foreign-baseline-failures.log`. This historical expanded run remains failing evidence. The reconciliation below
records the subsequent repair. The bounded directly owned lifetime regression
from that publication is reported separately. No failing test or production
behavior is weakened to close these parents.

## Expanded lifetime and blocked-selector reconciliation

The eight mixed-flag failures came from executable call expansion. Extra native
bit/word adapter nodes delayed a ready Free until after predicate erasure. The
resulting release order changed LIFO reuse on the next child invocation. This
was a physical scheduling defect despite the existing primitive dependency
comparison succeeding.

`component_schedule.py` now gives directly action-free register views earlier
insertion indices when building the physical scheduling graph. The existing
source-derived greedy priorities and SSA edges still determine readiness. A
view consists only of Split/Join operations, or a call to a single block with
only those operations. Computation, measurement, allocation and release bodies
retain their existing classification. The logical graph continues to use source
connection insertion order, so independent source resource estimation and
logical call ordering keep their accepted evidence.

The unchanged `test_mixed_workspace_flags.py` verifies the exact original event
sequence, per-wire flags, pool provenance and subsequent reuse for both expansion
modes. It also verifies source primitive dependencies/operands/feedback, complete
complex measured branches, zero work population, all forward/inverse history
pairs, ordinary append execution and independent model estimates. The focused
mixed-flag and SelectSwap profile passes all 111 cases with the original lifetime
assertions retained.

The other two failures were stale expectations that a whole-table block must
be refused by `QroamCleanTable`. The existing owner deliberately supports a
constant bank with no high address, covered by the accepted whole-table loading
and exact-phase routing scenario in `test_source_qroam_clean.py`. The SelectSwap
boundary test now asserts that admitted case has one batch and zero temporary
And workspace. Mixed omitted-axis cases still raise. The original unmodified
SelectSwap decomposition failure is asserted in every case and is kept distinct
from the native supported storage/routing behavior.

No upstream source file is changed. No failing lifetime equality, amplitude,
workspace or resource assertion is removed. Q-0311/Q-0323 retain their independent
storage, direct SDK and symbolic drawing obligations.

The expanded scheduler/expansion/lifetime consumer profile now reports 3253
passes and one optional PennyLane skip. The existing PennyLane dependency
directory resolves that skip, and the exact skipped case passes together with
the final mixed-lifetime/SelectSwap profile of 112 cases. Thus all 3254 selected
expanded cases have passing evidence across the recorded runs. Local integration
passes 17 cases and the second Cirq version profile passes 118. Strict typing,
Black, Ruff, scope, architecture, IR and protected API checks pass. Four inherited
strict errors in the touched SelectSwap test reproduce on baseline and are
resolved by removing variable shadowing and typing the deliberately invalid
boolean fixture explicitly. The original frozen reference is untouched.

## Symbolic presentation and complete direct lifetime hooks

The internal Drawer projection now retains the Core symbolic width expression
instead of coercing it to int. RegisterLine and RegisterConnection share the
existing SymbolicInt representation. Layout uses the existing symbolic metadata
predicate and the original concrete nonnegative-integer validation. A logical
word occupies one display lane, independently of its physical bit count. Finite
array shapes, original operand dependencies and lane ownership retain their
existing traversal rules.

Actual original _alloc and _free factories now match complete open-boundary
geometry, directional glyphs and width labels. Closed allocation/free graphs
also match for n, 2*n+1 and symbolic Integer(3), under both dirty flags. The exact
width object survives through layout and connection export. JSON, HTML and DOT
presentation succeeds. Physical binding continues to reject these unresolved
symbolic storage descriptors before allocating wires. Invalid negative integer,
boolean, float and string widths still fail the concrete display guard.

Across the eight supported foreign wire dtype families and both dirty flags,
actual source qalloc/qborrow/qfree calls produce the same reversed pool events
as native resolved lifetime bodies. The source and native calls return the same
logical ownership transitions, and both original PennyLane hooks return None.
Native emission has zero gates. Previous coherent borrowed-state restoration,
clean-work zero, spectator phase and independent zero-QEC/word-width resource
evidence remain applicable because all Compiler and numerical owners are
unchanged.

Two old mixed-score tests still expected rejection of classical unsigned words
admitted by approved Proposal 108. Both fail identically with all three Drawer
owners restored temporarily to the pre-batch HEAD. Their assertions now check
the approved exact Core port metadata and valid Program identity. No production
Core behavior changes. The baseline failing run is retained separately.

At this historical publication, Q-0311 and Q-0323 still awaited Proposal 116
storage admission. Their direct SDK lifetime hooks and symbolic presentation
were verified. The approved storage closure below supersedes that blocker.
Static controlled-Hadamard admission was independent and subsequently approved
and implemented under Proposal 122; it is not a remaining lifetime dependency.

The final batch passes 172 concentrated cases and 17 local integration cases.
All 89 direct-lifetime, symbolic and mixed-score cases also pass under Cirq
1.6.1, while the primary environment uses 1.7.0. The eight existing source
Graphviz/parser deprecation warnings remain recorded. Strict typing passes all
six changed Python files. Black, Ruff, Compiler scope, architecture, multilevel
IR and protected API checks pass. Three frozen source blobs and six unchanged
Core/Compiler/numerical/lane owners are verified. Dependency findings total 681
repository-wide and zero on the batch surface after optional dependency guards
are added to the touched mixed-score tests. Evidence is recorded under
ftqc-reference-env/g2-symbolic-lifetime-*.

## Approved storage dependency reconciliation

The complete original Allocate and Free files and their direct bookkeeping base
were reviewed again after Proposals 116 and 125 were implemented. Typed creation
and consumption already use the shared lifetime owners; no new implementation
or API is needed. Actual source-library consumers now exercise signed fixed
point at positive and negative fractions and unsigned empty words under both
clean and dirty flags. Original wire round trips, exact typed interfaces,
ownership events, SDK hooks, ordinary append and complete complex restoration
pass. Empty words retain both logical events and borrow a real spectator for
the physical plan. The actual Cirq manager accepts empty qalloc/qborrow/qfree;
an empty qfree contains no wire from which a tracking oracle can recover a flag.

The original empty fixed-point Allocate classical callback fails while decoding
an empty binary string. Its Free callback accepts zero. Both facts are asserted
without changing the source callback or manufacturing a successful numeric
decode. They do not invalidate the successful source SDK storage lifetime.

All six nonbinary scalar/polynomial profiles now cover dirty as well as clean
storage. Literal original qborrow/qfree events match native ownership and wire
order. Complete complex states include entanglement between borrowed words and
the spectator, including raw codes outside the numerical field range. Borrowed
words are restored, rather than falsely required to be zero. Clean words return
to zero. Original dirty tensor hooks still contract zero bras/kets: their
spectator-only H/T matrix is distinct from the full borrowed-state SDK identity.

| Scope | Actual emission and ownership | Independent original model |
|---|---|---|
| Fixed-point Allocate/Free, both flags | No gates or measurements; same word allocated and released; empty word has no physical wires | Zero QEC gates; peak word width, excluding the caller spectator |
| Raw field lifetime plus spectator H/T | One H and one T on spectator; word has no gates or feedback; clean/borrowed pool unchanged after release | One Clifford and one T; DAG peak word width, physical caller capacity is width plus one |

Q-0311 and Q-0323 remain open for the separately pending signed zero-width
storage (124) and symbolic field declarations (126). Ordinary symbolic physical
widths remain explicitly unexecutable. Classical descriptor interpretation is
not expanded by this quantum storage evidence. Original QGF/QGFPoly wire codec
failures remain unchanged. Neither approved 116 nor approved 125 remains a
blocker. Receipts use `g2-lifetime-approved-storage-*-20261125`.

The reconciled lifetime, storage, pool and controlled-workspace regression
passes 278 cases; the direct source/SDK lane passes 87 cases with Cirq 1.6.
All three frozen source blobs and the real Kickmix 0.1 Simulator are verified.
Both changed Python modules pass strict typing, Black and Ruff; Compiler scope,
language, links, architecture, multilevel IR and protected API checks pass.
The preceding 145-case local integration result is reused from the identical
production tree: this reconciliation changes only tests and documentation.
