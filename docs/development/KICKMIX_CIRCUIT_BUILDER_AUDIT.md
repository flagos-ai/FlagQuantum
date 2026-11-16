# Kickmix circuit builder source audit

The reference is the frozen Kickmix subtree at
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| File | Git blob |
| --- | --- |
| `circuit_builder.cc` | `cb165727a016fde525a4708233c985c52614ff64` |
| `circuit_builder.h` | `fd05956a28c443f414310781f2926a2ef1f09208` |
| `circuit_builder.test.cc` | `d01e4c1aa3ad397312e52229cc1883c4a15debd5` |

Compiler owns gate decomposition and register lifetimes. The Core classical
condition and assignment records remain the boundary consumed by Runtime and
Simulation. Existing positive, negative and disjunctive condition lowering is
recorded in REVERSIBLE_CLASSICAL_LOWERING.md and
REVERSIBLE_DISJUNCTIVE_PREDICATES.md. KMX preserves effective classical guards.

## Three-control X and Z

An isolated C++ harness compiled the unchanged `CircuitBuilder::cccx` and
`CircuitBuilder::cccz` implementations. It used controls 5, 1, 3, target 4,
and clean helper 7 when supplied. Both dirty cases selected wire 0 as the
smallest available helper. The native comparator and masked phase now share
the same helper selection, control/target order, compute, use, measurement and
phase correction. Their existing admission and signatures remain intact.

Native KMX matches the recorded C++ instructions and operands. The only stream
adaptation is the initial `R q7` in the clean source case: the native callers
supply a wire already in zero. Native physical measurement clearing expands
HMR to H, measurement and conditional X. A subsequent conditional CZ restores
both branches' phases. Full 256-amplitude comparisons retain dirty helper
entanglement, preserve spectators and return the clean helper to zero.

| Schedule | Native emitted primitives | C++ builder instructions |
| --- | --- | --- |
| Dirty CCCX | CCX4 | CCX4 |
| Dirty CCCZ | CCX2, CCZ2 | CCX2, CCZ2 |
| Clean CCCX | CCX2, H1, measure1, conditional X1, conditional CZ1 | R1, CCX2, HMR1, conditional CZ1 |
| Clean CCCZ | CCX1, CCZ1, H1, measure1, conditional X1, conditional CZ1 | R1, CCX1, CCZ1, HMR1, conditional CZ1 |

These are emitted counts. The builder does not declare an analytical QEC model
for either routine. Both clean streams use one result slot and one caller-owned
work wire. Dirty schedules use one borrowed helper and no result slot.

## Two-term parity with one additional control

`parity_ccx` and `parity_ccz` cancel equal tagged parity operands before
emission. If either parity operand repeats the additional control, the source
emits the original pair-controlled gate followed by the single-controlled gate.
The native helper now follows that path. Quantum wire, classical slot and
Boolean tags remain distinct even when their numeric spellings coincide.

The unchanged builder was compiled for every original two-term parity type
combination: 64 parity-CCX calls and 256 parity-CCZ calls. Native KMX agreed in
all 320 cases after expressing condition scopes as effective guards. A second
oracle ran both alias positions and equal-parity cancellation for both
operations. The regression checks each of those six streams and all 64 complex
amplitudes, with zero added work or measurement. Alias-CCX emits CCX1/CX1,
alias-CCZ emits CCZ1/CZ1, and equal parity emits zero instructions.

## Two-term parity with two additional controls

The three-control parity calls use the first quantum parity operand as their
temporary XOR target. Sharing that pivot with either additional quantum
control is refused before emission, matching both original CCCX and CCCZ
failure paths. When both parity operands are classical, the source emits the
two corresponding products in order. The native helper retains that order.

An unchanged builder oracle covers all 256 parity-CCCX and 1,024 parity-CCCZ
tag combinations with each of the dirty and clean policies. All 2,560 native
streams agree after scope expansion and accounting for the 14 initial source
resets under the native zero-entry contract. The reset is only needed in the
source's clean, all-quantum central schedule. Both measured branches match the
full 256-amplitude result and clear the designated work wire. The dirty test
borrows a wire which is still the second parity operand and verifies that its
coherence and the final parity uncompute are preserved.

For the full-quantum fixtures the parity compute and uncompute add CX2 to
the central three-control stream above. The clean path retains one result
slot. The dirty path adds zero measurement, reset or result slots. Emitted
gate counts remain independent of a source analytical model.

## Quantum-target broadcast

The shared CCX and CCZ dispatcher preserves lane order and tag-specific
length handling. A false shared control emits nothing. A true or classical
shared control delegates to the equal-length broadcast CX or CZ path. A
quantum shared control uses the common length. The homogeneous quantum loop
stores the lane control before the shared control in `CircuitPattern`; the
mixed lane's scalar path stores the shared control first. The native caller
infers homogeneous dispatch from the lane tuple.

The unchanged C++ builder oracle compares both gates over four shared control
tags, four lane layouts and five lengths. All 96 emitted streams match the
native KMX operands and effective classical guards, and all 64 invalid
lengths are refused. The mixed layout is also checked over all four
classical assignments and all 512 complex amplitudes. Neither broadcast
allocates work wires or performs measurements.

For the mixed lane example with a quantum shared control, X dispatch emits
one CX, one guarded CX and one CCX. The Z dispatch emits one CZ, one guarded
CZ and H–CCX–H in native circuit IR. KMX encodes the last three as CCZ1.
These are emitted counts. The source builder declares no QEC model for
these operations.

## Deferred block marks and register declarations

The locked `RaiiDeferredMark` emits an entry immediately and emits one exit
on release. Moving the guard transfers exit ownership. Named empty blocks
emit adjacent entry and exit lines. An empty name suppresses both marks.
The original nested fixture and a C++ probe with a moved guard and an
unnamed block produce the same eight-line stream.

The optional KMX block-mark codec maps those events to the compiler's
existing `ComponentCall` ranges, indexed by lowered operations. It preserves
nested ordering, zero-width named ranges and the source's five escapes for
quoted text. The standard reversible-phase codec still owns executable
operand and scope validation. Exports with marks require explicitly bound
KMX operation ranges. Translating ranges across lowering and encoding
trace parameters are outside the current format adapter.

The original native executor reports identical basis outcomes and zero
phase with the eight-line fixture. Its two executable operations are
X1 and CX1, with six debug marks, no measurements and no allocation.
There is no separate resource model for debug marks.

### Physical register declaration stream

The KMX declaration adapter keeps a register's name and ordered quantum or
classical members at its numeric ID. It emits memberships before the name,
uses explicit lines for empty IDs and accepts the last declaration of a
name for the same ID. Repeated members are kept. A final nonempty name
shared across two IDs is refused, as in the locked source reader. Names
reuse the block-mark codec's quoted text grammar.

The original writer and native adapter agree byte for byte on sparse mixed
members and on a circuit with both declarations and nested block marks.
Members refer to declared executable widths and can include unused wires
and classical slots. These declarations add no instructions and do not
initialize classical slots. The existing typed binding layer still owns
logical ports and workspace lifetimes. The tests verify the resulting
operation stream, input slots and execution against the locked simulator.
The adapter requires the versioned widths header.

Dense source declarations keep names and membership. The unchanged C++
probe reserves r0 before appending a two-wire register named `quantum` and
one classical bit named `classical`. It reports the name `quantum` on r0 and
the two wires on r1. The dense control probe appends a zero-width named r0,
then assigns both the name and the wires to r1. The reserved case follows
from appending the name to the end of `register_data` before the membership
routine grows it to the reserved ID. The native adapter round-trips the
wire format by ID. The source allocation-time name displacement remains
an upstream defect.

## Broadcast CX and CZ with classical and X-basis operands

The direct source dependencies are `py/circuit/ops/cx.cc`
(blob `b83879b628ef730c0ca6c499868d0eb467e7d2a4`) and `cz.cc`
(blob `e418d2ebe013f419f2ac5a8128b5431f3e198dd7`). They validate
equal lengths and check mixed operand tags before emitting an operation.
CX rejects quantum control of a classical target. With a classical target,
each lane reads the previous target and the current control and writes the
XOR in lane order. A bit controlling its own inversion clears the bit
and now emits the same KMX `BIT_INVERT bN if bN` form as the source.

A known X eigenvalue provides a constant phase control. A measured X
eigenvalue reads a classical slot. Both use the existing controlled sign
emitter. The locked source matrix covers 24 CX and 16 CZ requests, including
38 operation streams and two atomic refusals. Classical guard sets are
compared after scope expansion. The ordered classical alias chain matches
the source bytes and final values for all eight input assignments. The
X-basis probe compares all 16 complex amplitudes for each of eight
classical assignments, including scalar phase.

The source alias chain emits three conditional bit inversions, with no
measurement or workspace. The selected X-basis example emits Z1, NEG2,
and X1 in KMX with the corresponding classical guards. In circuit IR each
NEG is represented by four scalar-sign primitives. The source defines
no separate analytical model for these dispatch rules.

## Classical result lifetimes

The source allocators share one LIFO pool. Dirty reuse preserves the slot
value. Clean reuse emits a zero store, and freshly reserved storage relies on
the source's initial classical state. Moving a lease transfers its release;
disarming a lease keeps the result out of the pool. The C++ intake exercises
all of these transitions. The guarded HMR lease pops its condition before
returning its slot, so the next HMR can reuse that slot without retaining
the old predicate.

The format and execution binder replay both source streams with the existing
classical store and X-measurement owners. KMX now retains the source
`BIT_STORE1` opcode on import and export and binds it to one constant XOR
assignment. The native compiler's ordinary XOR emission remains canonical.
The local binder uses explicit width, initialization and scope definitions
and allocates its guard snapshot above caller-owned result slots. Four
unnormalized complex HMR branches agree with the analytical projection and
clear both measured wires. The native executor also verifies the dirty-read
and following clean-clear results on the unchanged store fixture.

The source lease fixture emits two classical stores and two guarded X gates.
The measurement fixture emits HMR2, one condition pair, one Z and one NEG,
using one result slot. Local execution binds the condition through one
temporary classical snapshot. There is no source analytical cost model for
the lease operations. Source allocation and disarm are compile-time policies;
this replay uses caller-owned physical slots. Implicit initial-bit admission
and general source allocator parity remain open.

## Forward and reversed strided comparison loops

The locked `for_each_reversed` iterates source lanes in reverse and keeps the
original operation order within each lane. A standalone locked builder probe
uses a nonzero lower bound, stride-two controls, a negative target stride
and two operations per lane. It emits six operations in each direction.
Empty and inverted intervals emit nothing and never invoke the recipe.

The low-space comparison now binds its self-inverse prefix CCX sequence
once per pass and uses the same instructions in reverse lane order during
cleanup. Forward and reverse prefix blocks therefore share the same physical
operands. The C++ `gen_cmp_low_space.cc` reference was recompiled with the
locked builder and `CircuitPattern`. Whole comparison streams match for
widths 5, 8, 13, 19 and 24 after expanding effective classical guards,
omitting debug boundaries and accounting for explicit source resets on
fresh or already-cleared work wires. The tested comparison uses three
caller-owned clean wires. The source compiler emits 3, 5, 7, 7 and 10
resets respectively, which native entry/clear contracts supply. The
individual width cases have 54, 95, 164, 224 and 312 remaining operations
in both streams. The half-open range and physical operand schedule do not
add an analytical cost model.

## Guarded classical stores

The locked `bit_store_and` sequence first defines the destination
with `BIT_STORE0`, then enters a scope for the first input and emits
`BIT_STORE1` conditioned on the second input. The KMX target preserves
both stores and binds the guarded write through the existing
`ClassicalXor` assignment. A guarded store may overwrite a defined slot.
Introducing the first definition under a guard is still refused.

The two reference streams cover the AND sequence and a guarded zero store
whose inline predicate aliases the destination. The latter also overwrites
the outer guard's original slot inside the scope. A snapshot retains the
outer guard's entry value. The locked native simulator and the local
binder agree for all four input assignments per stream, including
both complex amplitudes of a downstream controlled X.

The source AND emits two classical stores and one scope pair, using one
caller-owned result slot. Local execution uses one classical snapshot
for the scope. The comparison fixture adds one guarded X to observe the
result. No measurement, quantum work wire or independent source model is
required by the stores.

## Physical debug records

The locked builder prints bare register vectors in reverse and registered
members in declaration order. Conditional register printing ends in a
conditional empty record. Constant printing reuses one dirty result slot.
A missing register ID emits nothing, and an X-basis result prints through
its classical slot. The original native executor confirms the prefix,
optional register line and newline placement for both condition values.

The opt-in format adapter round-trips the six physical `DEBUG_PRINT`
forms, preserving quantum or classical operands and the inline predicate.
Positions are indexed across the serialized body so adjacent marks around
a named empty block remain distinguishable. The locked debug fixture
round-trips byte for byte. Ten debug lines stay outside the reversible
operation list. The fixture still contains six effective store/scope
records and the explicit constant stores remain executable. Debug records
do not allocate workspace or contribute to the source gate model.

Rendering debug text remains the consumer's responsibility. The adapter
accepts the physical records emitted by the source writer. Parsing raw
register shorthand and rendering source stdout in the local runtime
remain open.

## Mixed SWAP and controlled SWAP

The locked `swap.cc` (blob `9021c1181802fe771e352e88f5f6dfa51b7d7e92`)
exchanges classical bits and measured X-basis slots through three ordered
conditional inversions. Each lane reads the values produced by preceding
lanes. When both operands alias the same classical slot, the source
clears that slot. The native emitter preserves its three source operations
and the final value for all four assignments. It retains mixed lane order
and uses its existing classical XOR owner for the writes.

A false controlled SWAP emits nothing. True emits SWAP, a classical slot
emits guarded SWAP, and a quantum control emits CSWAP. Two lanes match the
locked operands and all 32 complex amplitudes for each of those four
controls. The ordered physical swap primitive now also serves rotations.
The source emits a single physical SWAP per quantum lane, three
conditional inversions per classical or X-basis lane, and two corresponding
physical operations for the tested controlled two-lane broadcast. No
additional work wire or measurement is required. These are emitted counts. The separate Qualtran SWAP model is unchanged.

On a mismatched mixed lane the source builder may retain operations emitted
by earlier lanes. The native immutable emitter rejects the call before
publishing any stream. Raw quantum self-alias behavior remains outside the
supported gate contract. K-0208 remains open for both source boundaries.

## Ordered HMR and reset broadcasts

The locked builder's HMR broadcast requires equally long target and result
sequences. It pairs physical lanes in their supplied order. It accepts a
repeated result slot and a repeated measured wire. The later HMR overwrites
that result slot or measures the physical zero left by the earlier HMR
respectively. The native register primitive uses the same lane order and
existing single-wire clearing operation. Paired-table erasure uses this
shared path.

The locked source and native KMX agree on all three two-lane examples,
including the read of the final result. All four unnormalized complex
branches agree per example and every measured wire clears. Each example
emits HMR2 in KMX, or H2, measure2 and conditional X2 in physical IR,
plus one external feedback X in the test. It requires no extra quantum
work or analytical source model.

The source simulator implements R by X-basis dephasing followed by reset,
as recorded in `sim/sim.h`. The three-lane reset stream retains a repeated
wire, matches R3 in KMX and leaves the exact spectator density matrix
after tracing out the reset wires. Its native IR uses H3 and reset3, with
no result slots. Guarded reset uses the same native emitter.

K-0005, K-0006 and K-0008 remain open for the remaining whole-file families.
C++ allocation scopes, general loop patterns and debug runtime output have
separate source assertions. The full public mixed
vector admission and degenerate quantum wire aliases remain outside the
checked dispatch fixtures.
The original `CircuitBuilder.*` suite passed all 27 tests with GoogleTest
1.14.0 from the locked Bazel manifest. The build compiled the unchanged
builder, original tests and test helpers, and linked the remaining native
objects from the existing reference extension build. The native predicate
intake passed 631 cases and the focused gate and arithmetic regression passed
185 cases. The C++ baseline still needs per-family native replacement before
the original-test parent closes.
