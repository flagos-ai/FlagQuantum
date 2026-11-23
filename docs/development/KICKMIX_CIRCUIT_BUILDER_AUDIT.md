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

## Mixed operand inversion and condition guard movement

The locked `inplace_invert` flips a literal in its operand binding, emits
X for a quantum operand and emits BIT_INVERT for a classical slot. Reusing
one slot in two lanes emits two inversions in order. The native primitive
returns the replacement binding and any emitted gates. Mixed addition,
modular doubling and greater-than comparison now share the same primitive.
The unchanged C++ fixture and native KMX agree on X1 and BIT_INVERT2.
Both input values of the repeated slot preserve the expected full complex
state. The literal replacements emit no operation, and there is no work,
measurement or separate source model.

The locked move-assignment operator for a live condition guard overwrites
the pointer to its existing builder scope. Two pushes then produce a single
pop. The source accepts this stream and its leaked guard conditions the
following X. All four guard assignments demonstrate the leak. The native
KMX target refuses the unbalanced stream at validation. The scope leak remains a source issue.

## Mixed control resolution at full arity

The locked resolver preserves quantum and classical operand order while
combining classical predicates. The native emitter now keeps those guards
when it selects the maximum-arity CCCX and CCCZ schedules. It reuses the
guarded X-measurement clear primitive for feedback. A guarded result slot
must be defined by the caller and cannot alias a predicate that controls
its own clearing. The disabled path retains the old result. The H pair
around a guarded CCX in a CCZ cancels on the inactive path.

Four locked C++ streams cover quantum X and phase output with a borrowed
or clean helper under two classical guards. Effective guard sets and
ordered operands agree after accounting for the two explicit source resets
through the supplied-zero contract. Every classical assignment and both
clean measurement branches preserve all 256 complex amplitudes. The
borrowed helper remains coherent, the clean helper clears and inactive
measurement leaves its result intact. The primitive counts remain those
of the three-control schedules above, with one caller-owned slot on the
clean path and the source condition records counted separately.

The native resolver applies X-basis target phase kickback through the same
dispatch and has no mutable pending-control queue. The locked positive-X
target path leaves a false flag which suppresses the next use of the
builder. The source also leaves a pushed condition when rejecting a
mixed request beyond its supported quantum arity. Native emission returns
empty for the positive-X target and refuses unsupported arity before
returning a gate stream. The immutable source fixtures retain both defects.

## Mixed AND reset and erasure

The locked `reset_and` emits R and then the mixed controlled flip.
The locked `del_and` always measures and clears the target, then uses the
two original controls and the X outcome for phase kickback. Even when a
literal control is false, the source emits HMR. The native erasure owner
uses the shared mixed dispatcher. The measurement result must be distinct
from the classical inputs used in its feedback. Rotation, two-bit mixed
addition and modular doubling reuse the same erasure sequence.

The unchanged builder produced 16 control-tag pairs for each of reset and
erasure. Both KMX streams match in all pairs after expanding condition
scopes. With the AND already computed into the target, the native erasure
matches all 128 complex branches across tag combinations and classical
inputs, restores controls and spectators, and clears the target. The
broadcast zero erasure uses three HMR in supplied lane order and reuses
one result slot, including a repeated target. Its eight branches clear
the measured lanes. The field inverse chain now shares that emitter.

The source reset path emits R1 plus at most one controlled flip per
lane. Erasure emits HMR1 plus at most one effective phase gate or scalar
sign. Native HMR lowers from H, measure and conditional X. The existing
Qualtran four-T AND compute and its resource model own a separate
decomposition. The source reset is supplied by the zero-entry contract
where an existing caller already owns clean work.

## Original test file acceptance

All 27 named tests from `circuit_builder.test.cc` pass under GoogleTest
1.14.0. The fixture build uses the unchanged source file and builder.
A capture hook calls the original diagram assertion and records the
circuit that reached it. All 21 assertions across the 15 diagram tests
are retained and checked against native ordered operations, wire IDs and
effective guards. The original renderer continues to check its own text
spacing and scope labels. The FlagQuantum artifact represents the same
connections as ordered instructions with conditions.

| Original tests | Native evidence |
| --- | --- |
| `basic` | Exact `CCX q0 q1 q3` opcode and operands in the scalar replacement. |
| `store_and` | Locked scoped store sequence and all four input assignments in `test_kmx_result_lifetimes.py`. |
| `cccz_qqqqa`, `cccz_qbqqa`, `cccz_bqcqa`, `cccz_bqfqa`, `cccz_qqqq` | Five captured diagram streams in `test_kickmix_builder_file_audit.py`, plus branch and work restoration tests for the measured and borrowed helper paths. |
| `broadcast_ccx_aqq_1` through `broadcast_ccx_aqq_4` | Four ordered diagram streams captured from the unchanged source, including empty and guarded cases. |
| `broadcast_ccx_acq_1` through `broadcast_ccx_acq_4` | Four ordered diagram streams captured from the unchanged source, with classical and quantum shared controls. |
| `broadcast_ccx_abq`, `broadcast_ccx_amq` | Eight captured diagrams, including constant pruning and repeated bit guards. |
| `cx`, `cz`, `ccx`, `ccz` | All 160 source tag streams in the independent scalar fingerprint. Gate order, condition semantics and phase feedback are also covered by the mixed broadcast and controlled resolution tests. |
| `parity_ccx`, `parity_ccz` | 320 original source/native streams, plus complex-state checks for equal tags and quantum aliases. |
| `parity_cccx`, `parity_cccz` | 2,560 original source/native streams under both helper policies, including exact measurand, cleanup, and rejected quantum aliases. |
| `cswap` | All four source control tag streams and complete complex amplitudes in `test_kickmix_mixed_swap.py`. |
| `raii_mark_block_entry` | Original inclusive sequence of outer, inner and empty named blocks in `test_kmx_block_marks.py`. |

The source test consumer K-0008 has a native equivalent for every
assertion family. K-0005 and K-0006 remain open. Production compatibility
still has gaps for allocation scopes, general loops, debug output, mutable
builder failure behavior and degenerate quantum-wire aliases.


## Generator context ownership

K-0009 covers locked `circuit_gen_ctx.h` at blob
`ab6cc825738f96922e1a447c0ea7693aaa861ed8`. The native
`GenerationWorkspace` selects ordered storage for nested generators.
The existing `RegisterWorkspace` continues to issue and release live
leases. No gate, measurement or result slot is emitted by a generator
view.

| Source behavior | Native owner and verification |
| --- | --- |
| Empty context and optional clean storage when minimizing | View state and `preferred_clean` select the same paths in in-place addition and mixed addition. Explicit clean requests can still use the supplied work. |
| Child clean subspan and prefix take | A validated prefix-binding returns the chosen prefix and a sliced child view. The parent's tuple is retained. An out-of-range optional subview is represented by an ordinary empty tuple slice. |
| Nested dirty insertion and count | New segments are prepended, preserving direction within a segment. The source span with stride -1 becomes the physical order 10, 9, 8. A child sees 5, 6, 10, 9, 8 while its parent sees 10, 9, 8. |
| Borrowed prefix and back | A prefix walks the newest group first. Back selects the last lane of that group, q6 in the fixture. Mixed addition consumes the prefix, and in-place addition selects the carry through back. |
| Insufficient storage | The locked borrowed copy appends the available prefix and then throws. Native selection validates the whole request before returning a tuple. Both reject over-requesting, with the native caller's lease state preserved. |
| Local assertion helper | Existing validation raises `ValueError` at the owning call site. |

The locked source oracle and the native context test agree on the selected
lanes and widths. The unchanged addition fixtures also verify ordered
physical gates, clean return and borrowed state restoration. The source
context declares no analytical costs, and the view changes no resource
counts.

## Reversible pattern binding and affine indices

The locked `CircuitPattern` collects operand columns and emits repeated
rows. The native binder reuses `ReversiblePhaseOperation` for the local
pattern and binds quantum and classical columns separately. A constant
column repeats one operand. Strided columns and lane ranges express forward
and reversed traversal, including affine offsets. Each lane retains the
original operation order and in-row operand order. The native unary-to-binary
erasure binds its three-operation cleanup row through this shared path.

The immutable C++ fixture and the native binder emit the same 37 ordered
records across a forward range, a reversed range and a final shorter loop.
The pattern carries X, reset, HMR, conditions, physical swap, CCZ, classical
inversion and scalar sign. The source pattern-specific CCZ packing puts its
second input first, which is represented directly in the native row. Resets
are preserved here. The 37 records include four HMR, four resets and four
scope pairs. Binding creates no allocation or analytical cost model.

K-0013 uses Python's bounded integer indexing for the source's affine
index. The locked `iota.h` fixture verifies all seven addition overload
paths exercised by signed and unsigned widths, subtraction and unary
negation, yielding the same 27 wire positions. Native admission checks the
bound columns and refuses out-of-range access. The difference for
overflowing C++ integer offsets or invalid pointer arithmetic is outside
the supported physical indexing contract.


### Typed pattern dispatch

K-0010 and K-0011 cover the locked implementation and declarations at
blobs `46a69f4144b1cb5c41d8800a31175b8d83ea10fa` and
`aaae344e260bef14a204fc34ec2183b4e4e2a5e7`. The native pattern layer
reuses the existing operation representation and selects the source
typed opcodes before binding rows.

| Locked source family | Native replacement |
| --- | --- |
| `spanish`, `ptr`, operand arenas and `gather_and_clear` | Integer constants or physical sequence columns. Binding rows consumes neither input columns nor interpreter state. Forward, reversed and subsequent shorter patterns reproduce source row and operand order. |
| Reset, HMR, bit inversion, push/pop | The existing reversible-phase operation carries result and condition columns separately. The mixed fixture contains all five and retains both reset and result order. |
| X, Z, NEG, CX, CZ, and their guarded forms | Direct operation rows use the same bit and wire positions. A homogeneous quantum control column provides CX/CZ/Z. A homogeneous classical column supplies X/Z/NEG guards. Mixed controls are refused before emitting. |
| SWAP and controlled SWAP | The optional enable retains its wire or selects SWAP. Guarded variants preserve the classical condition. |
| CCX and CCZ | The quantum-enable path packs the varying operand before the fixed control, matching the source pattern. The true enable produces CX or CZ. The guarded CCX retains its condition. |
| `LoopBuilder` and `dump_into` | Emit one typed row recipe over a bounded range. Reverse traversal changes lane order, while each row remains ordered. Physical columns replace source-arena storage and lifetime management. |

The locked C++ typed fixture and native implementation agree on all 30
physical records and the three heterogeneous refusal cases. The first 37
mixed-lane records and the 27 affine wire selections also remain covered.
Binary/unary conversion uses native typed CSWAP and CX row binding in both
traversal directions. No additional quantum resources or result slots
are allocated by dispatch. K-0012 covers the separate benchmark file.

### Physical column workloads

K-0012 covers the five workloads in locked `circuit_pattern.perf.cc`
(blob `fcf2d76936178fd3fb503e544e5eb33e89ff7995`). Both paths use 256 lanes
and the original unsigned input sequence `k*k+3`. The original C++ oracle
uses the locked gather implementation and the original column definitions.
Native binding checks the SHA-256 of the full lane-ordered operand stream
against that oracle before timing.

| Columns | Operands checked | Source gather | Native binding median |
| --- | ---: | ---: | ---: |
| 1 | 256 | 260 ns | 181 µs |
| 2 | 512 | 360 ns | 370 µs |
| 3 | 768 | 410 ns | 549 µs |
| 4 | 1024 | 880 ns | 744 µs |
| 7 | 1792 | 1.3 µs | 1293 µs |

The five shapes cover offset, contiguous, positive and negative stride, and
broadcast columns. The source figures time raw 32-bit arena gathering
with a 0.01-second target. Native figures time binding preselected physical
columns and allocating Python operation objects. The two timed contracts
differ. Neither workload executes a circuit or has primitive or modeled
quantum resource counts.

Run the native verification and timing with
`python -m benchmarks.reversible_pattern_binding --iterations 3`.
The separate circuit-builder benchmark is K-0007.

### Builder workloads and admission

K-0007 covers all three workloads of locked `circuit_builder.perf.cc`
(blob `a5de60e3ab96a1afbdfb55e4a9d2274bfd85db9b`). The original benchmarks
run unchanged. A separate observer captures each original workload before
its arena is cleared. The native benchmark verifies the ordered stream
before timing the existing generators and binder.

| Workload | Locked source records | Native verification |
| --- | --- | --- |
| Controlled left rotation by one on 256 wires | 255 CSWAP and two debug marks | All 255 CSWAP agree in order, including the fixed enable and both swap operands. |
| Repeated CCX with overlapping lanes | 255 CCX | All 255 CCX agree in order, including the shifted target and the source's lane-first control packing. |
| Mixed tagged scalar CCX | 293 raw builder records across 256 calls, including scopes | Both admissions agree on 56 wire collisions. All 200 admitted calls agree on their 111 ordered KMX operations and effective guards. |

The source mixed workload reuses the numeric lane in each operand family.
When a generated operation duplicates a quantum wire, source
`finish_circuit` rejects the circuit. The source benchmark measures raw
builder writes, including those operations. Native admission refuses each
collision during construction. Its 231 admitted IR instructions include
four X/Z leaves for each of 40 scalar signs. The existing KMX lowerer
emits one NEG for each sign. Scopes are compared by effective guards,
and debug marks are excluded from primitive counts.

With a 0.01-second source target, the three C++ arena timings were 120 ns,
210 ns and 1.0 µs. The independent native medians were 399 µs, 225 µs and
495 µs, respectively. The native measurement includes Python object
construction or binding and its applicable admission checks. It excludes
the verification-only KMX lowering and fixture setup. These are distinct
construction measurements. No circuit is executed and no analytical
resource model is inferred.

Run `python -m benchmarks.reversible_builder_dispatch --iterations 3`.

## Complete raw builder ownership follow-up

The full circuit_builder.cc and circuit_builder.h are being reviewed together
with their direct dependencies. The existing sections above retain their
accepted streams and numerical evidence. K-0005 and K-0006 remain open during
this batch. The explicit source defects in sparse register naming, positive-X
pending controls and live condition move assignment remain recorded.

The remaining construction-time result ownership uses the existing
KmxCircuitBuilder and WireAllocation. The allocator retains one physical cursor
and authoritative RegisterWorkspace leases. Its clean bank serves explicit
allocations, while its dirty bank holds raw builder helper results. Clean and
dirty result requests share the helper bank. A fresh clean result emits no
store. A reused clean result emits BIT_STORE0 before use. Explicit bit frees
cannot feed the helper bank. Quantum allocation is independent.

result_bit returns an immediately allocated lease with a classical OperandIdentity.
Its context closes on successful and exceptional exits. measure_result first
emits HMR and can then push the result as a condition. Closing pops that owned
condition before LIFO release. disarm transfers both the live bit and pending
condition cleanup to its caller. C++ move-only storage is represented by an
explicit Python owner and deterministic context release. Python garbage
collection does not emit circuit instructions. Clearing a builder invalidates
outstanding contexts, which fail explicitly on later release.

Physical wire IDs and classical slots keep their existing independent numeric
spaces. The ordinary typed component and circuit.append(op) entry points remain
the consumers. Numerical execution stays with existing lowering, Runtime and
Simulation. Compile-time lease operations add no quantum primitives or model
estimates. HMR, feedback, clean reuse stores and conditions are counted from
the complete emitted stream.

### Complete source and header ownership map

Every definition in the 1171-line implementation and every declaration or inline
body in the 453-line header belongs to the following native owners. Overloads
share a row only when their tag and operand dispatch were checked together.
The header includes additional CX, CZ, SWAP and bit-store definitions residing
in separate source files, whose accepted binding audits are reused here.

| Complete source responsibility | Native owner and retained evidence |
| --- | --- |
| append_register, its mixed/array returns, reserve_qubits | KmxCircuitBuilder quantum registration and WireAllocation permanent reservation. PHYSICAL_BUILDER_BINDING_AUDIT.md retains independent ID spaces and lifetime transactions. |
| append_classical_register overloads and mixed/array returns, reserve_bits | Classical registration, OperandArray and the same scalar allocation owner. No second register-value representation. |
| reserve_register, append_bit_to_register, append_qubit_to_register | KmxRegister, replace_registers and physical declaration codec. Sparse ID membership and repeated members retain existing byte-exact fixtures. The source allocation-time sparse name displacement remains open on K-0005. |
| alloc_clean_raii_bit, alloc_dirty_raii_bit, result lease constructors/destructors, disarm and move ownership | Shared WireAllocation banks and deterministic result_bit context. Clean/dirty reuse, nested LIFO, disarm, copy refusal and failed-body release are covered by the raw original helper fixture. Native explicit release replaces C++ destructor timing. |
| hmr_raii_xbit, hmr_raii_push_condition, XBit conversion | measure_result and OperandIdentity.conjugated_by_h. Shared HMR lowering retains all four complex branches, exact scalar phase and measured-wire zero. |
| RaiiCircuitBuilderPopPushedCondition constructors, destructor and move assignment | Existing condition codec plus owned measurement condition exit. Source live move-assignment scope leak remains an implementation defect. A pending disarmed condition is explicitly caller-owned. |
| ControlLike, QuantumTargetLike and ClassicalControlLike | Existing typed operand binding, OperandKind, _Control and _XBasisTarget. Source concepts become tag admission at the existing binding boundary. |
| c_push overloads, c_resolve overloads, c_resolve_exact and failure explanation | Existing immutable _flip and mixed control resolution. Full supported arities and classical guards retain locked streams. Source false-flag retention on a positive-X target and failure-state pending conditions remain source defects. Native diagnostics retain native ownership. |
| x overloads, x_if, broadcast_x | Existing unary physical binding, _flip and ordered broadcast_x_ir. Direct physical records retain their opcodes in KmxCircuitBuilder. |
| z overloads, z_if, z_pow, z_pow_if, neg and neg_if | Existing mixed phase dispatch, PhaseAngle128 and reversible-phase codec. Full exact angle words, phase and actual counters reuse accepted phase and operation binding receipts. |
| ccx/ccz overloads, conditioned variants, cz_if and cx_if | Existing _mixed_gates, scalar/typed binding and KMX columns. All 160 source scalar streams and unchanged diagram assertions retained. |
| cccx/cccz concrete and header templates | Existing maximum-arity _flip schedules with explicit clean/borrowed workspace. Templates become one typed dispatch. Full dirty coherence and every small clean branch retain source controls and feedback. |
| parity_ccx, parity_ccz, parity_cccx and parity_cccz | Existing parity-controlled emitter and accepted 2880 source streams, alias cancellation and clean/borrowed numerical evidence. |
| broadcast_ccx, broadcast_ccz, broadcast_cswap and cswap/swap_if variants | Existing broadcast and mixed swap owners. Shared-control-dependent lengths, lane operand order and effective guards retain accepted fixtures. |
| cx/cz/swap scalar, mux and broadcast declarations defined outside this implementation | Existing broadcast_binding, broadcast_swap and typed mux owners. Direct CX/CZ/SWAP source file audits and scalar/array fixtures retained. Source destructive classical self-alias and immutable native partial-failure policy stay documented. |
| broadcast_hmr, hmr, hmr_if, reset, reset_if and broadcast_reset | Existing ordered measurement/reset primitives and KMX operation columns. Repeated wire/result fixtures and full small reset/HMR branch evidence retained. |
| reset_and, del_and, del_zero and broadcast_del_zero | Existing broadcast_init_and_ir, broadcast_del_and_ir and X-measurement clear. All 16 source tag pairs and 128 erasure branches retained. |
| inplace_invert scalar/span/array overloads | Existing operand inversion and shared mixed arithmetic consumers. Literal rebinding and repeated classical slot order retained. |
| bit_invert, bit_store0/1, guarded variants and bit_store_and | Existing classical assignment binder and Core ClassicalXor. Atomic read-before-write, scope snapshots and alias feedback retain accepted fixtures. |
| broadcast_bit_store header overloads defined in ops/bit_store.cc | Existing physical_classical_store with complete source broadcast, refused type and sequential feedback cases. |
| debug_print and debug_print_if overloads | Existing KMX debug records, raw register projection and physical-session output owner. Bare vectors reverse lane order, registered members keep declaration order and constant printing uses the helper bank. Actual debug observer fixtures retain the source instruction sequence. |
| info_tag, RaiiDeferredMark construction/move/destruction and raii_mark_block_entry | Existing component trace and block-mark codec. The raw unnamed guard emits no marks, while the independently accepted Python mark context emits empty names. These retain their separately recorded source contracts. |
| for_each and for_each_reversed | Existing reversible pattern owner, affine physical columns and typed row binding. Recipe is bound once and rows expand in forward/reverse lane order. Empty/inverted intervals retain no callback. |
| finish_circuit, MutableCircuit storage and CircuitPattern field | KmxCircuitBuilder.freeze, KmxCircuit and existing pattern owner. Validation stays at the native boundary. Source arena/pointer layout is replaced by owned columns. |
| next_qubit_id, next_bit_id, next_register_id, raii_bits and pending-control fields | Existing allocation, register columns and result banks. Immutable gate dispatch has no pending-control queue. No foreign builder state is added to Core or Runtime. |
| skip_validation | The locked field has no reads in the source subtree. Native finalization retains validation. No inert execution-policy switch is introduced. |

K-0006 declaration and ownership scope can close with this map and successful
batch acceptance. K-0005 remains open for its recorded source implementation
boundaries. This does not close K-0180 automatic wrapper failure-state mutation,
K-0232 lookup defects, numerical factory rounding or source performance gaps.

### Follow-up batch acceptance

K-0006 is accepted for its complete declaration and ownership scope. The full
114-definition implementation map retains K-0005's recorded implementation
boundaries. All 53 frozen source/dependency blobs pass verification. The new
standalone C++ probe compiles three unchanged source files with their 36-header
closure and regenerates the seven lifecycle observations byte for byte. Each
observation also executes three original and three native lanes, preserving
canonical records, outputs, exact phases and separately recorded static/actual
counts. The accepted original 27 builder tests are reused with their unchanged
source blob. Existing complete scalar, parity, broadcast and work-zero evidence
is retained.

Local checks pass 42 focused checks, 4655 concentrated checks, 121 integration
checks and strict typing for six files. The focused coverage selection including
existing mutable-circuit tests passes 151 cases. WireAllocation has complete
statement and branch coverage. The full builder has 209/216 statements covered,
with 96% combined statement/branch coverage. Black, Ruff, Compiler scope,
architecture, multilevel IR and API checks pass. The dependency finding set
remains the identical 720-entry baseline. The replacement HMR construction runs
every branch of the accepted complex spectator consumer with measured wires
zero. Receipt g2-raw-builder-acceptance-20261123.json records this scope.
