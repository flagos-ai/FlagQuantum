# Allocate and Free source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/bookkeeping/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0310 | `allocate.ipynb` | `8d91513e62d1015fcbd97c0601442916e3ba6796` | Closed by the complete native Allocate and Free tutorial. |
| Q-0311 | `allocate.py` | `e459095285de98c1468b661afdbcbe58238ff1ad` | Open for external object hooks, full executable dtype and symbolic component drawing. |
| Q-0312 | `allocate_test.py` | `96f411955b10a466d37e8ba3b0698a0823d42ba9` | Closed for the runnable source scenarios and notebook. |
| Q-0323 | `free.py` | `8a0c73db71e1d8ffa31b2c7a15d246f02f8f4f63` | Open for external object hooks, full executable dtype and symbolic component drawing. |
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
no lifetime gate. Direct PennyLane and Cirq object-level bookkeeping adapters remain open.

Concrete typed component scores identify lifetime calls by their primitive
bodies. Allocate and Free reuse the existing default arrows and empty
connectors. The glyphs and full layout match the frozen scores for one
and three bits under both dirty flags. The tested source `qalloc` or
`qborrow` and `qfree` calls match the native lease events on the same
reversed pool. Both bindings produce zero gates. The physical circuit
reserves the pool and one spectator. The source qubit model visits its
disconnected spectator separately and reports the pool width. The
locked symbolic glyph has the same labels, while native typed component
scores require concrete widths.

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

Q-0311 and Q-0323 remain open for complete executable dtype admission, direct SDK
lifetime hooks and symbolic component drawing. Signed and empty fixed-point
storage still awaits Proposal 116. Generic classical and nonbinary physical
lifetime semantics remain with their existing descriptor/execution owners.
Source PennyLane `None` hooks and qubit-manager events retain their earlier
observations, with direct foreign-object adapters still explicitly open.

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
