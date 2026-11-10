# Physical register Program binding

The internal Compiler entry `register_binding.bind_register_program` binds actual
finite Program definitions to existing CircuitIR. Proposal 103 defines the entry
and ownership profile. Users continue to consume emitted operations through
`circuit.append(op)`.

## Ownership and interfaces

- Core verifies typed SSA ownership for the complete module before binding.
- `component_schedule.local_operation_order` derives each function's local
  visitation from operation operands, scalar parameters, feedback and returns.
- Existing global greedy traversal schedules that graph. Existing finite call
  expansion keeps each invocation's already-scheduled body together.
- Existing `layout.workspace.RegisterWorkspace` owns physical pool allocation
  and ordered partial release. No second wire pool or executable graph is added.
- `register_binding` maps words, scalar values and boolean handles to existing
  Instructions. Simulation remains responsible for numerical execution.

`inputs` is an immutable tuple containing one physical-wire tuple per declared
left element, in port and row-major order. Word wires are MSB first. Unpacked
quantum gate inputs contribute one wire each. `clean` and `borrowed` are disjoint
immutable wire tuples, disjoint from inputs. All wires are concrete nonnegative
built-in integers. Supplied unused pool wires remain in the declared circuit
width. At least one physical wire is required by CircuitIR.

The frozen result contains the actual CircuitIR, ordered right-element wire
tuples and immutable workspace events. Events retain the flattened operation
position, allocation/free kind, current wire order and declared dirty flag.
Free returns each wire to its allocation pool even when its source Free flag
differs. Partial releases and cross-allocation joins use the same pool method.
Core proves zero-width logical ownership before empty physical tuples disappear.

## Source ordering and classical slots

The connection adapter follows the locked `_create_binst_graph` edge traversal.
Each edge introduces its source then its target. Parameters and feedback provide
zero-width dependencies. Real single-operation allocation/free helper chains
receive the corresponding atomic source priority. The shared greedy queue uses
that node insertion order. Each function is scheduled before call expansion.

Six actual source builder graphs compare the whole local operation order and
Cirq manager events, including split-to-bit release, cross-allocation join and
release-order reuse, for both clean and dirty work. Source isolated nodes enter
from a frozenset, whose tied ordering is not portable. Native isolated nodes
follow declaration order. Isolated source tie parity remains open.

Classical slots are fresh in actual physical visitation order. Each invocation
gets independent slots. Existing emitter replacement tests explicitly bind those
local handles back to the consumer's original wire-associated slot numbers.
This alpha-renaming preserves the measured wire and every feedback operand.
Dependency-equivalent ordering of independent primitive gates is admitted in
those emitter comparisons. Primitive counts and per-wire order remain exact.

Binding derives instructions from actual Program operations. An optional
`canonical_payload` does not replace or attest that graph. Existing Core
payload restoration remains a separate entry with its own attestation checks.

## Local evidence

Run the focused contributor path with the locked reference environment and
preimport torch before invoking pytest.

```bash
python -m pytest tests/unit/test_register_physical_binding.py -q
```

The focused batch has 33 passing scenarios in 5.40 seconds. Both new production modules have
100% statement and branch coverage in that set. Verification includes real
Add/QROM append consumers with private allocated work, original primitive
operands and feedback, all small measurement histories, unnormalized complex
amplitudes and zero clean workspace. Dirty double-CNOT lifetimes preserve the
entire state jointly entangled with an external spectator. Repeated nested
private-work calls reuse one physical wire, with allocation and release events
remaining within each invocation. A 1300-level call chain, output permutations,
zero-width split/join lifetimes, reset, rotations, boolean parity, disjunctive
feedback and entry failures are covered.

Physical double-CNOT execution emits two CX instructions, zero T and zero
measurements. Its source QEC model independently reports two Cliffords, zero T
and zero measurements. Allocate, Free, Split and Join emit zero primitives.
Add/QROM primitive counters and original source QEC/width models are checked
independently by the existing consumer.

| Path | Emitted primitive counts | Source QEC model | Declared width / work | Source modeled peak |
| --- | --- | --- | --- | --- |
| Add uint2 into uint3 | 20 CX, 6 H, 4 T, 4 T-dagger, 2 S, 2 measurements, 2 CZ, 2 resets | 2 And, 11 Clifford, 2 measurements, 8 total T-equivalent | 7 / 2 | 7 |
| QROM table [1,3,0,2], target2 | 18 CX, 6 X, 6 H, 4 T, 4 T-dagger, 2 S, 2 measurements, 2 CZ, 2 resets | 2 And, 10 Clifford, 2 measurements, 8 total T-equivalent | 5 / 1 reused | 5 |

Each of these two paths has four exhaustive unnormalized measurement histories.
The model Clifford category differs from expanded primitive Clifford counts.
Declared circuit width includes supplied pool wires, while the source peak query
describes modeled live width.

Acceptance logs are stored in the external reference environment under
`g2-physical-binding-{native,regression,checks,examples,resources,default}.log`
and `g2-physical-binding-source.txt`. Source blobs are checked against the locked
inventory. The 12 source blobs match the locked inventory. The concentrated regression
passes 4861 tests in 34.52 seconds without skips or warnings. Strict Python 3.12
checks pass for 17 production modules. Formatting, ownership scopes,
architecture, Program/public API gates and four existing examples pass.

The default smoke/unit tier stops at four failures after 5516 passing tests,
166 skips and 30 warnings in 287.14 seconds. The same two QFT fusion dtype cases,
product-state QFT fusion and terminal fused-rotation test failed before this
batch. Earlier independent snapshot reproduction at `4d57e638` is retained in
`g2-workspace-construction-baseline-failures.log`. The default tier remains open.

## Remaining boundaries

The admitted entry has only quantum input/output ports. Its default requires
balanced nonempty workspace. Explicit allocated-output transfer is available
through the policy below. Releasing caller inputs requires separately declared
pool ownership and receives an explicit diagnostic.
Internal boolean and scalar calls are supported. General classical boundary
ports, symbolic widths and physical control-flow blocks remain outside this
profile. Conditional measurement/reset follow the existing CircuitIR rejection
rules. Zero-qubit physical circuits remain outside the existing CircuitIR schema.

Clean-zero and borrowed-joint-state restoration are component obligations.
Binding does not numerically prove arbitrary emitters. The focused consumers
supply that evidence for their actual paths. Whole G2 and full source-file
acceptance remain open.


## Allocated output ownership transfer

Proposal 103 prerequisite 1.7 adds keyword-only
`allow_workspace_outputs: bool = False`. A built-in true value transfers live
allocated wires through the existing ordered `outputs` result. Allocation events
retain clean/borrowed origin. The caller owns the subsequent lifetime, cleanup
and eventual restoration of borrowed state. Returned work remains unavailable to
later allocations within the same binding. Partial output words may combine
input wires with allocated wires, or retain part of a lease after other bits
are freed and reused. Zero-width right ports retain their Core logical identity
and bind to an empty tuple.

Run `pytest tests/unit/test_register_output_transfer_contract.py
tests/unit/test_register_output_transfer.py tests/unit/test_register_physical_binding.py`.
Eight source contract witnesses and 21 native transfer tests join the existing
33 binding tests. The 62 focused tests pass in 6.23 seconds, with both physical
binding modules retaining 100% statement/branch coverage. Source witnesses
confirm allocation without a free event and exact reversed right-port order
for clean/borrowed widths 0, 1 and 3. Default source managers reject unmanaged
caller-input release for both Free flags.

Real H/CX/T component bodies return allocated work through actual Program calls.
An unchanged `fq.Circuit.append` consumer compares complete complex states
against original source Cirq conversion, including a joint external spectator
and reversed right-port wires. The path emits one H, one CX and one T. Its source
model independently reports two Cliffords, one T and zero measurements, with
three modeled qubits. The four-wire numerical test includes one spectator.
Borrowed outputs preserve the source's transformed joint state. Final borrowed
restoration belongs to the enclosing caller lifetime.

A measured Add returns both private work bits and verifies all four exhaustive
unnormalized branches against original source primitives. Both returned bits
are zero in every branch. Emitted and modeled resources are the Add row above.
Nested calls retain live output leases until their outer return. Insufficient
remaining pool capacity is diagnosed, and a later allocation uses a separate
wire. Partial output transfer frees/reuses only the released bit. Tests also
retain default balanced-profile rejection and reject non-boolean policy values.

Contract commit `44c6300b` precedes Compiler implementation `021c0548`. External
logs use `g2-output-transfer-{contract,native,regression,checks,examples,default}.log`
and `g2-output-transfer-source.txt`. General classical boundary ports remain
separate acceptance work.


The output-transfer batch passes 4890 concentrated tests in 43.59 seconds,
without skips or warnings. All 15 reference blobs match the inventory, including
H, CX and T source definitions. Strict checking of 17 production files,
Black/Ruff, both ownership scopes, architecture, Program/public API gates and
four existing examples pass. The default smoke/unit tier again stops at the
same four previously reproduced failures, with 5516 passes, 166 skips and
30 warnings in 443.03 seconds. Its log is `g2-output-transfer-default.log`.

## Pre-owned input pool adoption

The internal keyword-only `input_ownership=((wire, dirty), ...)` declares which
entry wires already belong to caller-owned clean or borrowed pools. Each wire
is a distinct concrete input wire and each origin flag is a built-in boolean.
An empty declaration keeps caller inputs unmanaged. The physical pool registers
these leases atomically without emitting an allocation event or consuming
available workspace. A partial word release must contain only declared owned
wires. Free records its source dirty flag, while each wire returns to its
original pool regardless of that flag.

An unreleased adopted input may return through the ordinary right port. Once
released, the same wire may be allocated as new Program work. Such an output
requires `allow_workspace_outputs=True` and transfers its subsequent lifetime to
the caller. The existing pool reuse order is preserved. The original Greedy
manager witness has no spare available wires, so it establishes only this
specific reuse case, not arbitrary manager-policy equivalence.

Eight unmodified-source manager scenarios cover widths one and three, both
input origins and both Free flags. The native binding matches their free-before-
allocation events, reversed wire order, zero emitted primitives and source
modeled peak equal to the width. A separate actual Program runs two CX gates
on a joint complex state with a spectator. Both clean-zero and borrowed
restoration match the original Cirq conversion, which models two Cliffords,
zero T and zero measurements with peak width two. That numerical test has
three physical wires including the spectator.

A measured And compute/erase Program releases the adopted inputs after its
private clean work returns to zero. Both unnormalized measurement branches
match the unchanged-source decomposition, including all complex amplitudes,
one spectator and clean work zero. Emitted primitives are 6 CX, 3 H, 2 T,
2 T-dagger, 1 S, 1 measurement, 1 CZ and 1 reset. The independent source
model reports one And, one Clifford, one measurement and four total
T-equivalent gates, with peak width three. The numerical comparison has four
physical wires including the spectator. Output allocation after input release
uses the declared origin pool and has no quantum primitive of its own.

The focused 130 tests cover the prior binding/output paths, pool collision
atomicity, invalid ownership, mixed-origin joins, released-wire reuse and
default policy rejection. Both modified production modules have 100% statement
and branch coverage. The concentrated regression passes 4946 tests without
skips or warnings. All 16 reference blobs match the locked inventory. Strict
Python 3.12 checks of 17 production modules, Black/Ruff, both ownership scopes,
architecture, Program/public API gates and four existing append/resource
examples pass. Logs are stored under `g2-input-pool-{regression,checks,examples}.log`
and `g2-input-pool-source.txt`. The default smoke/unit tier was not rerun in
this batch. Its preceding run stopped at the four independently reproduced
QFT/rotation fusion failures described above.

## Measurement-produced owned classical outputs

Proposal 103 prerequisite 1.10 adds destructive `measure_register` for an exact
owned QBit. Binding emits one existing measurement instruction and records its
fresh slot in the owned CBit result. Existing nested calls can forward that
handle. CBit dependencies retain graph edges with zero quantum width.

Quantum output tuples keep one position per returned element. Classical elements
have empty quantum tuples. When classical results exist, `classical_outputs`
contains parallel slot tuples with empty tuples for quantum positions. Pure
quantum results keep `classical_outputs == ()`. External classical inputs use explicit initialization under prerequisite 1.17. Pool-owned measured inputs and
allocated work require a retirement contract and remain rejected, including
borrowed wires and explicit workspace-output transfer.

Run `tests/unit/test_register_owned_measurement.py` and
`tests/unit/test_program_owned_measurement.py`. The two-measurement array passes
through two CBit identity calls per element and reverses returned element order.
All four unnormalized complex branches match the original MeasureZ copy tensor
with its DiscardInd leg, including spectator phase and unmeasured work at zero.
A consumed physical qubit retains its projected outcome and has no returned
quantum handle. Existing nondestructive measurement and bool feedback remain
separate operations, sharing the fresh-slot allocator without collisions.

| Witness | Emitted primitives | Original source model | Native physical width |
| --- | --- | --- | --- |
| Two destructive MeasureZ operations and four CBit identities | 2 measurements, 0 resets, 0 T | 2 measurements, 0 T-equivalent | 4 wires including one spectator and one untouched clean work wire |

This witness compares the original instrument because classical-port Cirq export
is unsupported upstream. It does not certify a flat original primitive stream.

## Owned classical parity and feedback

Proposal 103 prerequisite 1.11 adds `parity_register` and
`read_classical_register` using the existing Program representation. Parity
returns an owned successor per input and a fresh owned XOR result. Successors
retain their predecessor slots. A read returns one owned successor and one
shareable bool snapshot of the same immutable slot. It emits zero primitives
and allocates zero slots. Both operations retain zero-width dependency edges.

`component_classical.parity_word_function(name, count)` constructs the original
CBit[count] THRU and parity RIGHT signature. Counts must be concrete built-in
nonnegative integers. Zero inputs emit one constant-zero classical XOR. Two
nested parity calls emit two classical XOR assignments with distinct targets.
Classical entry initialization follows prerequisite 1.17. Workspace retirement
follows prerequisite 1.16.

The conformance scenario measures zero through three QBits, computes nested
parity, reads the result and uses one bool for conditional Z and X on distinct
targets. Both condition polarities are checked on every small unnormalized
measurement branch against original MeasureZ tensors, Parity callbacks and ClassicallyControlled
Z/X tensors, including joint complex spectator states and unmeasured work at zero. Two
runtime strategies each check 32 normalized conditional states per polarity.

| Three-input nested-parity scenario | Emitted primitives | Original source model |
| --- | --- | --- |
| Destructive measurement subpath | 3 measurements, 0 resets | 3 measurements, 0 T-equivalent |
| Two Parity calls | 2 classical XOR assignments | No QEC implementation or source decomposition |
| Feedback snapshot | 0 instructions and 0 new slots | Native ownership-to-feedback adaptation |
| Conditional correction | 1 conditional Z and 1 conditional X | Quantum primitives counted separately |

The original Parity QubitCount query gives zero quantum width. Its QEC query
raises DecomposeNotImplementedError. Neither fact supplies a complete quantum
resource model for this combined source graph. Original classical-port Cirq
export remains unsupported. The reference is retained unchanged.

## Explicit measured workspace retirement

Proposal 103 prerequisite 1.16 adds the internal measured_workspace policy.
Its default reject retains the existing refusal for pool-owned destructive
measurement. With retire, clean measured owners leave their leases permanently
and retain their projected physical outcomes. With reset, each measurement is
followed by an existing reset instruction and its clean wire returns to the
pool for later allocations. Events distinguish retire from free. Adopted clean
inputs follow the same rules. Both policies refuse borrowed owners atomically.

This is a FlagQuantum physical adaptation. Original Allocate/Free delegate to
the source qubit manager and emit zero primitives. Original Free assumes clean
zero or original borrowed-state restoration. Measurement alone supplies neither
condition. Explicit reset overhead is counted separately from the unchanged
logical component graph and source QEC estimates.

Run tests/unit/test_register_workspace_retirement.py and
tests/unit/test_workspace_measurement_retirement.py for partial leases, pool
exhaustion, reuse, invalid policies, source instruments, discarded-owner bool
snapshots, actual dynamic execution and owned GF2MulMBUC branches. Small fields
exhaust all measurement histories. The eight-bit witness checks instruction
stream and resources without a full statevector branch enumeration.


## External CBit entry initialization

The internal classical_inputs tuple aligns with flattened LEFT elements in
declared row-major order. Quantum positions use None. Owned CBit positions
require built-in booleans or integers zero/one and empty physical input tuples.
Each gets one existing constant classical XOR write into a fresh immutable slot.
Later measurements and Parity outputs allocate slots after this input prefix.
Nested calls and reordered outputs preserve slot ownership. Discard preserves
previously shared feedback snapshots. These use the existing Core operations.

Run tests/unit/test_register_classical_inputs.py for source entry assignment,
all small classical inputs, measurement branches, both feedback polarities,
complex spectators, untouched work zero and actual append execution. Input
initialization costs one classical write per CBit. Parity costs remain separate
from that prefix and source model queries.

This is concrete binding-time specialization. Runtime-injected classical slots
and a zero-wire physical CircuitIR retain separate acceptance. Purely classical
logical graphs can use an explicitly supplied existing carrier through the
current workspace contract. Source logical quantum width remains zero. Original
empty-array identity entry assignment succeeds, but its call_classically output
finalization raises KeyError(RightDangle). Native empty arrays execute with zero
initialization writes. That source failure is asserted and recorded separately.
