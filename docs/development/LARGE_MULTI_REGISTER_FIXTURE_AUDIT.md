# Large and shaped register fixtures

Q-0495 through Q-0497 match the inventory blobs at Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. The original
many-register test passed unchanged. Native fixture code lives under
tests and uses the existing primitive and typed Program owners.

Current status, 2026-11-25: approved Proposal 116 closes the typed companion
storage dependency. The final section records the complete Q-0496 acceptance;
earlier pending-116 statements are historical.

## Repeated AND ladder

The literal first control value of each source AND is applied to the
incoming prefix. The new selection bit is positive. The inverse ladder
uses the same control values in reverse order. The source graph reader
compares each physical operation, operand and classical feedback slot.
The controlled atomic leaf is identified as CX by its complete source
tensor. This supplies an explicit native lowering for the fixture.

At two and three selection bits, every measurement history was projected
directly on a coherent input over all logical basis states. The larger
four-selection-bit and five-repetition fixture uses endpoint histories
and an AND channel certificate over the full logical basis. The
certificate checks the necessary work state at every AND erasure and
establishes the same map for all 32,768 histories. The complete vector
agrees with the untouched source tensor, with the unnormalized branch
factor and zero work. The zero-repetition fixture allocates no work.

The large fixture emits 60 T-family instructions, five atomic action
CX gates within the complete ladder stream, and 15 measurements across
eight wires. Its source qubit estimate is eight. The full source QEC
query fails at the opaque `C[TestAtom]` leaf, which cannot decompose.
The executable gate count and tensor evidence retain this source resource
query failure separately. Q-0495 is closed by the complete-source follow-up.

## Shaped words and typed companion signatures

The shaped fixture applies X, then four two-bit actions in row-major
order. Each pair has its own split and join. The three-bit spectator
also splits and rejoins without a physical operation. The direct
source calls, nonconsecutive physical port map, tensor axes and full
4,096-entry complex output match. The physical stream is one X and
four CX, with zero work or measurements and 12 wires. The upstream
`TestTwoBitOp` supplies a tensor but no QEC decomposition.

The three companion signatures retain the unsigned bound, fractional
width and signedness. Two source companions declare no decomposition.
The composite companion fails when wiring an unsigned `d` word into a
fixed-point input. The native executable binder admits bounded unsigned and unsigned fixed-point
storage. Signed fixed-point admission remains pending Proposal 116. Q-0496
stays open for that dependency. Q-0497 is covered by the signature,
binding, action and tensor comparisons.

The concentrated native suite passed 209 tests. Strict typing,
formatting, team scope, architecture, IR and public API checks passed.

## Complete-source follow-up

The previous Q-0495 disposition treated a reproduced upstream QEC failure as
missing executable behavior. It is now retained as a source defect, separately
from the demonstrated native primitive stream. The original controlled TestAtom
has no decomposable QEC model. No numerical estimate is substituted for that
failed source query. Complete `large_bloq.py` Q-0495 is accepted with this explicit
source limitation and the transport/invalid-input dispositions below.

A real omitted valid case was identified. One selection bit produces repeated
controlled TestAtom actions without ANDs or work, despite the original empty
control-table modulo issuing a divide-by-zero RuntimeWarning. The native fixture
now follows this exact path. One and two repetitions match source operations,
nonconsecutive operands and the full complex tensor action through append. The
measured stream is respectively one and two CX, with zero T-family operations,
measurements and work; the independent source qubit estimate is two. The source
warning is asserted explicitly during decomposition, tensor and qubit queries.

Actual LargeBloq depth-zero foreign bytes and nominal signatures round-trip
through the existing native interchange. Original reconstruction fails because
its class is absent from the upstream deserializer registry. Depths one and two
for the measured ladder fail while serializing NumPy uint8 control values.
Neither reference is patched. Original zero-selection decomposition fails with
a missing select argument. Negative repetitions fail an upstream assertion. The
native fixture rejects these invalid inputs at its own boundary.

All four `many_registers.py` classes now have actual foreign consumers. The
runnable shaped fixture preserves its exact library bytes at depths zero, one
and two and resolves through an explicit native factory to the existing complete
Program. Its physical stream is one X and four CX on the original row-major,
nonconsecutive operands. The full 4,096-component complex vector matches the
original tensor. There are no measurements or work events. The two atomic typed
companions retain canonical declarations, and the typed composite retains its
original invalid dtype wiring at depths one and two. Bodyless declarations have
no physical executable claim. Their original nominal resource and decomposition
failures remain recorded separately.

The inherited companion test incorrectly expected bounded and unsigned fixed
storage to be rejected. That failure was reproduced on the unchanged batch
base. Its assertions now follow already-approved admission, including Proposal
115. Signed fixed-point storage still refuses under pending Proposal 116.
Q-0496 remains open for this concrete storage dependency. The source companion
atoms have no decomposition and the source typed composite has invalid wiring.
These source failures are preserved rather than supplied with invented bodies.

The follow-up `g2-large-multi-*` evidence reuses the earlier full measured-channel
certificate, branch, work-zero and exact primitive comparisons. It records 268
concentrated and 17 local integration passes, strict checks for three changed
files, formatting, team scope, architecture, IR and public API checks. The
independent resource evidence remains separate from primitive measurements.


## Complete many-register source closure under approved storage, 2026-11-25

Q-0496's outstanding signed-storage dependency is now closed under the already
approved and implemented Proposal 116. Its inherited metadata test still
expected signed QFxp(8,6) storage to refuse. That failure was reproduced before
editing the test. All declared source descriptors now execute the same exact
width/admission assertion; the earlier signed-only refusal is removed without
changing any source or production implementation.

The complete 114-line source has four classes and eight methods, all reviewed:

| Definition | Complete native disposition |
| --- | --- |
| TestMultiRegister signature and composite | Existing row-major shaped Program, all original Split/Join and ordered X/CX calls, exact physical aliases, full complex tensor and ordinary append |
| TestBoundedQUInt signature | Actual source library bytes and canonical bounded/signed-fixed descriptors; original no-decomposition failure retained as a bodyless terminal |
| TestQFxp signature | Actual source library bytes and canonical signed/unsigned fixed descriptors; original no-decomposition failure retained as a bodyless terminal |
| TestMultiTypedRegister signature and composite | All four nominal descriptors admitted as storage; original invalid QUInt-to-QFxp child wiring and depth-one/two serialization failure retained, with no invented executable body |
| Source presentation labels | Foreign roundtrip objects retain the original source labels; native fixture function naming remains owned by FlagQuantum |

Actual source libraries cover all four classes at depths zero, one and two,
with the two original typed-composite failures asserted separately. The runnable
shaped Program emits one X and four CX in the exact source order on 12 physical
wires, with no workspace or measurements. Its nonconsecutive row-major bindings
and all 4,096 complex output amplitudes match the original tensor. Source atomic
QEC-model failure remains an unavailable source estimate, not a fabricated gate
count. Three typed companion declarations retain the same bounded, fractional
and signed metadata; no numerical behavior is asserted for undefined terminals.

The unchanged original many_registers_test.py passes. Native fixture and actual
foreign-consumer tests pass, and the earlier complete-source/primitive/channel
receipts remain reusable because frozen blobs and production owners are
unchanged. Seven relevant source/dependency blobs and real Kickmix 0.1 Simulator
are reverified. This closes the sole remaining admission requirement recorded
at d9c469b78; original source resource, terminal and invalid-wiring dispositions
are preserved under the same prior full-file criteria. Q-0496 is accepted.

The same approved-storage batch removes two obsolete SignExtend refusal branches.
Its actual fanout stream is five CX for three live prefix output bits, in each
direction, while the independent source model reports one Clifford and peak
five. Caller width six includes the external spectator. No measurements or
feedback occur, and the exact inverse clears/releases the prefix. These
resources remain separate from the shaped fixture's one-X/four-CX stream.


Current closure verification: 17 native fixture/actual-foreign-consumer cases,
17 independent Cirq 1.6 cases and the unchanged original source test pass.
The shared signed-storage/prefix regression passes 135 cases plus 20 complete
component/source cases, with 62 Cirq 1.6 prefix/foreign cases. Strict typing
covers both changed test files; Black/Ruff, Compiler scope, language, links,
architecture and IR/API checks pass. Seven frozen blobs and the real Kickmix
Simulator are checked. Prior 145 local integration cases are reused from the
identical production tree. No production owner was edited in this batch.
Receipts use `g2-sign-fixed-storage-*20261125` and
`g2-multi-register-fixed-storage-*20261125` in the reference workspace.
