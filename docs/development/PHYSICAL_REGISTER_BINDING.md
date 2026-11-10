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

## Call and return replacement inventory

GRAPH33 inventories all 39 pinned cases in
`tests/unit/test_register_component_calls.py`. A source observer runs the
unmodified BloqBuilder and exposes construction observations. Executable native
witnesses use actual Program function bodies and Core verification. The table
records scenario families, including differences in their supported boundaries.

| Original scenario family | Original cases | Executable native witness | Boundary or remaining work |
| --- | ---: | --- | --- |
| Through port order and row-major successor identity | 12 | `test_component_program_builder.py`, real identity bodies | Includes zero-width scalar and empty arrays |
| Dict, tuple and automatic output arity with LEFT consumption | 9 | `test_component_call_boundaries.py`, real Free/Allocate bodies | Each body is independently verified before calling |
| Opposite sides sharing a name with different widths | 1 | `test_component_call_boundaries.py`, uint2 Free and uint5 Allocate | Input and output retain distinct SSA values |
| Missing, surplus, dtype, duplicate and short call failures | 5 | `test_component_program_builder.py` | Reserved instance and consumed prefix match |
| Reusing an old output after another call | 1 | `test_component_program_builder.py` | Failure reserves its call, Core rejects its incomplete body at finalization |
| Failed return prefix and retry | 5 | `test_component_program_builder.py` | Existing original consumer runs against native return observations |
| Explicit empty return with empty or consumed inputs | 1 | `test_component_call_boundaries.py`, two native cases | LEFT-only consumption uses a real Free body |
| Empty shaped right port with no dangle connection | 1 | `test_component_call_boundaries.py` | Native retains its declaration, original final_soqs omits the key |
| Classical binding single consumption | 1 | `test_component_call_boundaries.py` | Uses canonical CBit descriptor, retains stale-call failure prefix |
| Actual Add signatures for widths (2, 3) and (0, 2) | 2 | `test_component_add_calls.py`, real repeated Add bodies and source failure witness | Nonempty Add matches source gates and all 16 histories, zero-left original constructs but fails decomposition |
| Shaped array alias at the second row-major use | 1 | `test_component_call_boundaries.py`, real identity body | First two elements consumed, last two remain available, no successors |

The original classical scenario passes a nominal `classical_bit` descriptor
through a source adapter that selects CBit by classical width. Native execution
requires the existing canonical `classical_bit_type()` descriptor. The new
witness uses that descriptor on both sides and records this admission distinction.
A stale CBit call keeps an empty reserved operation in both builders. Original
finalize can retain that incomplete signature-only instance. Core rejects the
incomplete executable native module. Consumed return prefixes remain observable
in both cases. These tests preserve the validation boundary.

Directional uint2 input lifetimes also compare actual source Free/Allocate tensor
contractions for output widths zero, one and five. Both native physical binding
and original tensors preserve the complex spectator, with all supplied clean
wires zero. The caller explicitly grants input ownership and accepts allocated
outputs. This is tested on the legal zero-input subspace required by clean Free.
Allocate/Free emit zero primitive gates and there are no measurement histories.
Existing Add/QROM consumer tests independently retain exhaustive measured
branches, exact complex amplitudes, zero work and separate model counts.

The exact Add witness wraps existing measured addition with its a/b typed ports,
private clean work and existing split/join functions. Two actual calls retain
the original operands, measurement feedback and per-wire primitive order. The
original generator reverses qalloc results before assigning carry significance.
Its supplied physical allocation order is explicitly reversed to bind the same
carry roles. No reference code is modified. All 16 unnormalized histories retain
the full complex spectator state and zero work.

Two Add calls emit 40 CX, 12 H, 8 T, 8 T-dagger, 4 S, 4 measurements, 4 CZ and
4 resets. The source QEC model independently reports 16 total T-equivalent and
4 measurements, with modeled peak width 8. Actual declared width is 8 with two
reused clean work wires. Split/join and lifetime binding add zero primitives.

Original Add(QUInt(0), QUInt(2)) permits two calls to be constructed and finalized
but decompose_from_registers raises IndexError while reading input_bits[0].
Native add_ir rejects the empty input with ValueError before emission. The
zero-width ownership token remains supported by the existing real identity
body. This case is recorded as an original executable-source limitation.

Q-0157.03 closes the finite call/return replacement inventory with the stated
descriptor admission, failed-instance verification, empty-array metadata and
zero-input Add exceptions. The complete composite builder source file and G2
remain open.

### GRAPH33 acceptance

Directional call witnesses are committed as 82b61d5a. Repeated Add and zero-left
source witnesses are committed as cb6ee8e8. Twenty added tests pass. The final
concentrated regression passes 5,235 tests in 67.23 seconds with no skips or
warnings. Focused coverage passes 154 tests in 15.81 seconds, with 100% statement
and branch coverage of component_builder and layout.component_workspace.
Twenty-nine integration tests pass in 2.79 seconds. Strict Python 3.12 typing
passes 38 production/test files. Black/Ruff, architecture, Program/API checks,
both ownership scopes and four append/resource examples pass. All 28 pinned
source blobs match the locked inventory.

The default smoke/unit run reports 22,286 passed, 300 skipped, 1,065 warnings
and seven failures in 304.86 seconds. The seven failure identifiers match the
recorded pre-batch baseline. The run preceded the two additional Add witnesses,
which are included in the final concentrated regression and focused checks.
The default tier remains failing. Its evidence is
`ftqc-reference-env/g2-directional-calls-default-parallel.log`, with comparison
in `g2-call-inventory-default-comparison.txt`. Final concentrated, coverage,
checks, examples and source evidence use `g2-call-inventory-`. Integration
evidence uses `g2-directional-calls-integration.log`.

## Named component traversal and rebuilding

The finite typed call profile admits a single returned block whose operations
are calls to declared typed functions. Compiler traversal yields the original
operation position, named predecessor bindings in LEFT signature order, and
output collections in RIGHT signature order. Arrays retain row-major indexing,
including empty arrays. Scalar zero-width words retain ownership. Existing
`local_operation_order` determines visitation. Positions identify the input
Program operation, while rebuilding assigns positions in visitation order.

Rebuilding replays calls through `ComponentProgramBuilder`, mapping each old SSA
value to its fresh binding. Callee definitions remain shared immutable records.
Entry function/block metadata and module metadata are retained. Returned values
are remapped explicitly, preserving permutations and directional ports. Core
verifies all definitions before traversal, including unreachable definitions.
This slice covers pinned `CompositeBloq.iter_bloqsoqs` and `copy`. Selective
`flatten_once`, predicate behavior and bounded recursive `flatten` have separate
acceptance. Isolated source frozenset ties and omitted empty output metadata
retain the previously documented native policies.

Logical graph visitation counts CBit edges as one bit, matching the original
source greedy sort. Physical visitation counts them as zero quantum width.
`local_operation_order(logical_graph=True)` selects the former for named
traversal/rebuilding only. The original GF2MulMBUC graph and its copy at widths
1, 2, 3 and 8 exercise the ordering distinction, including Join versus Parity.

GRAPH34 acceptance adds 24 traversal/rebuilding cases and four copied execution
variants. The unchanged source erasure graph and copy match every predecessor,
port shape, output and visited instance at widths 1/2/3/8. Copied repeated Add
retains all 16 unnormalized complex histories, clean workspace and exact source
primitive dependencies. Measured primitives remain 40 CX, 12 H, 8 T, 8 Tdag,
4 S, 4 measurements, 4 CZ and 4 resets. The independent source model remains
16 T-equivalent, 4 measurements and peak width 8.
Copied field erasure retains all 14 histories at widths 1/2/3 and exact bound IR.
Unmeasured work is zero. Destroyed result wires retain their measured projection.
At width 8 the existing separate execution witness remains 64 conditional CZ,
8 H, 8 measurements and 8 classical XOR assignments, while the source model
reports 64 CZ only. Rebuilding introduces zero primitives or workspace.

Pinned source copy still fails for an empty-shaped output omitted by final_soqs.
The native copy retains the declared empty collection. Pinned zero-width
split/join tensor contraction raises ValueError for an empty tensor network,
while its graph traversal and copy succeed. The tests assert both facts without
altering reference code. GRAPH34 concentrated regression passed 5263 tests,
focused coverage passed 56 at 96% combined statement/branch coverage for the two
compiler modules, and strict typing passed 40 files. Default-tier baseline
failures remain recorded from GRAPH33. Selective flattening remains open.

## Selective typed call expansion

`component_expansion.expand_component_calls_once` replaces selected typed call
occurrences with one level of the callee's existing typed call body. A verified
single-block callee with explicit returns and exclusively typed calls provides
that decomposition, including an empty identity body. A callee containing
primitive/register operations remains an atomic leaf. Other errors, including
predicate failures and invalid unreachable definitions, propagate.

The input and output pair the existing ProgramRecord with an operation-aligned
tuple of integer occurrence labels. Initially labels default to declaration
positions. Labels are compiler transformation provenance for correspondence with
pinned BloqInstance.i, without becoming another graph or changing Core schemas.
Retained calls keep their labels and output SSA definitions. New calls receive
labels above the previous maximum, in source local visitation order. The
predicate receives the label and the existing Operation. Returned words are
mapped element by element, including empty shapes, zero-width owned words,
classical owners, directional ports and permutations. All callee definitions
and module metadata remain available for independent verification and lowering.

A pass visits only the original entry calls. Inserted calls are considered on
the next pass. A pass without an accepted decomposition raises
NoComponentExpansionError.
`expand_component_calls` repeats passes until that condition, returning the
last graph and labels. Its depth bound includes the final no-change probe,
matching pinned flatten's range/else behavior. Reaching the bound raises
ValueError, including max_depth zero or negative. Immutable input records and
labels remain intact on every failure. The user execution entry continues
through the existing binding and circuit.append(op) workflow.

Logical current-layer classification also distinguishes a composite wrapper
around Allocate/Free from the atomic lifetime helper itself. The wrapper keeps
ordinary signature-delta priority until expanded. The logical_graph option now
selects both source classical widths and current-layer classification. Physical
binding retains its existing nested helper classification. Exact source
allocation-wrapper and free-wrapper witnesses cover traversal before and after
expansion.

GRAPH35 implements the finite typed-call form of pinned flatten_once/flatten.
Occurrence labels remain an operation-aligned compiler provenance tuple. They
preserve gapped retained IDs and assign fresh IDs above the input maximum across
repeated passes, without adding Core fields. Primitive bodies provide the native
atomic fallback. Arbitrary Python decomposition callbacks and their dynamically
thrown exceptions are outside this immutable-definition profile. Source atomic
Hadamard and undecomposed classical leaves exercise both source fallback error
families. Predicate exceptions propagate and invalid unreachable definitions are
verified before the predicate runs. Empty identity bodies, arrays and return
permutations exercise element-wise output substitution. The source empty-array
identity still raises KeyError during add_from; native metadata retains the
empty return, following its existing declared-empty policy.

The implementation commit is fdcc313f. There are 54 new expansion cases and two
expanded Add variants. Concentrated regression passed 5319 tests with no skips
or warnings in 26.25 seconds. Focused coverage passed 112 tests in 12.34 seconds,
with 97% combined statement/branch coverage across expansion, named graph and
local schedule modules. Strict typing passed 42 production/test files. Both
existing integration suites passed 29 tests in 1.25 seconds. Formatting, scope,
architecture, Program/API checks and four existing append/resource examples
passed. Twenty-nine pinned source blobs, including the original composite tests,
retain their inventory hashes.

Expanded repeated Add preserves the exact source wire dependencies and all 16
unnormalized complex measurement histories, spectator state and zero clean work.
Measured primitive counts remain 40 CX, 12 H, 8 T, 8 Tdag, 4 S, 4 measurements,
4 CZ and 4 resets. Source QEC independently reports 16 T-equivalent and four
measurements, with peak width eight. Expanded field erasure retains all 14
histories at widths 1/2/3 and exact bound IR. Destroyed result wires retain their
projections, while unmeasured work is zero. At width eight the same bound IR has
64 conditional CZ, eight H, eight measurements and eight classical XOR writes.
The original GF2MulMBUC model reports only 64 CZ, as recorded in GRAPH30.
Source and native wrapped erasure graphs match every occurrence label, ordered
predecessor, shaped output and final graph through one and two wrapper levels
at widths 1/2/3/8. Transformation adds zero physical primitives or workspace.

The default smoke/unit run finished after fail-fast with 22361 passed, 300 skipped,
1065 warnings and six failures in 280.09 seconds. All six failure identifiers
match the independently recorded baseline. The seventh baseline ledger case
was absent from this stopped run and retains its unresolved status. No default
pass is claimed. The final focused run includes the refined invalid-unreachable
body fixture, with the same 112 passing tests. Evidence logs are stored under
ftqc-reference-env/g2-component-expansion-{source.txt,regression.log,coverage.log,
checks.log,integration.log,examples.log,default-parallel.log,default-comparison.txt}.
The next inventory is CompositeBloq.adjoint against existing precise component
adjoints and their measurement/lifetime boundaries. Full parent-file acceptance
and G2 remain open.

## Component graph adjoint admission

The compiler-owned `component_adjoint.adjoint_component_program(record,
adjoints, entry="main")` rebuilds one verified finite typed-call entry using
pinned `_adjoint_cbloq` ordering. `adjoints` maps each used forward definition
name to an existing specialized adjoint definition name in the same module.
Each target must have the exact ordered LEFT and RIGHT ports of
`Signature.adjoint()`. Specialized Split/Join may group opposite-side
declarations differently, matching the pinned source. Definitions
remain immutable and shared. This mapping is an explicit semantic obligation of
the caller, with source and numerical witnesses required for acceptance.

The new entry flips LEFT/RIGHT ports, retains declaration and row-major element
order, starts from the original explicit returns, reverses logical call
visitation, and reconnects original inputs as outputs. Program SSA remains the
authority. New occurrence positions follow reversed insertion order. A second
application uses the reverse definition mapping. Entry and block names are retained. The forward canonical CircuitIR payload
is cleared because it describes different semantics after taking the adjoint. The selected entry cannot also be a target definition.

No automatic reversal of measured primitives is admitted. Exact measured
component adjoints must be supplied before decomposition. Allocate/Free and
Split/Join use their actual opposite definitions. Quantum and canonical CBit
metadata can be rewired, but an executable classical or measured adjoint requires
an independently supplied body. Arbitrary abstract source Adjoint wrappers and
symbolic structures remain separate inventory. Native declared empty-array
returns retain their metadata even where the pinned builder omits empty keys.

GRAPH36 acceptance at 70970978 adds 41 scenarios. Original Split/Join and
clean/dirty Allocate/Free graphs, direction-flipped entries, row-major elements,
empty returns, output permutations and double adjoints match the pinned source.
The source empty-array identity adjoint raises a missing-output ValueError;
native preserves declared empty metadata. H/T/T reverses to Tdag/Tdag/H with
exact complex spectator amplitudes. Actual a2/b3 Add definitions map to their
existing exact Add-adjoint bodies. One/two calls cover 20 inverse measurement
histories and 20 forward restoration histories, with full unnormalized complex
states, spectator entanglement and zero clean workspace through circuit.append.

Per call, measured primitives are 20 CX, 6 H, 4 T, 4 Tdag, 2 S, 2 measurements,
2 CZ and 2 resets. The independent source QEC model gives 8 T-equivalent and
2 measurements, with source peak width 8 including the spectator. The graph
transformation adds zero primitives and workspace. Concentrated regression
passed 5360 tests in 26.92 seconds. Focused coverage passed 119 in 5.86 seconds,
with 97% combined statement/branch coverage of component_adjoint. Integration
passed 29 in 1.30 seconds. Strict typing covered 44 files. Formatting, scope,
architecture, Program/API checks and four existing examples passed. Thirty-one
pinned source blobs retain their inventory hashes.

Default smoke/unit stopped after 22264 passes, 300 skips, 1065 warnings and six
failures in 173.67 seconds. Each failure identifier matches the existing baseline.
The seventh baseline ledger case was absent from the stopped run. Two final guard
cases were added after that run began and passed the final concentrated/focused
runs. The default layer remains open. Logs are stored under
ftqc-reference-env/g2-component-adjoint-{source.txt,regression.log,coverage.log,
checks.log,integration.log,examples.log,default-parallel.log,default-comparison.txt}.
Generic abstract adjoint wrappers, caller-supplied semantic validation, symbolic
structures and dynamic decomposition remain separate inventory. The next slice
is hierarchical adjoint decomposition and resource-call record correspondence.

## Hierarchical adjoint and Program call-count admission

The compiler-owned `adjoint_component_hierarchy(record, adjoints, entry="main")`
replaces one typed-call entry and derives missing adjoints of reachable stored
call-only component definitions. Existing exact specialized definitions in
`adjoints` take precedence for child components. The root is always derived and
must not occur in that mapping. Primitive or mixed bodies require an explicit
specialized target. Each generated definition is shared by every occurrence of
its forward component, under a collision-free internal function name. Original
child definitions stay available to forward callers. The entry and its generated
hierarchy use the existing graph reversal, per-side signature validation and
canonical-payload invalidation. Finite traversal uses a work stack.

`_component_trace.program_call_graph.program_component_call_graph` derives direct
multiplicities from the same verified typed-call Program, then delegates weighted
terminal counting, keep policy and depth bounds to existing symbolic_call_graph.
Nodes use existing function names. Primitive/mixed bodies remain terminal at this
component boundary. These are actual stored component occurrences. Source model
estimates and emitted primitive instruction counts retain separate accounting.
Declared symbolic call graphs without stored bodies remain in the existing model
path. No automatic numerical inversion or new Core graph is introduced.

The source default Adjoint protocol is exercised on a real Bloq whose stored
body is the original executable CompositeBloq. Directly wrapping CompositeBloq
in Adjoint retains the original decomposition ValueError, since CompositeBloq
intentionally forbids decompose_bloq. Source files remain unchanged. Component
resource node identity uses Program function names. Automatic source structural
aliasing remains outside this profile. Specialized mappings that collapse two
forward names onto one target are aggregated normally. Bounded first-visit
counts follow stored declaration order. The source counts CompositeBloq instances
from a frozenset, so cross-depth shared-node cutoff ties remain an explicit
portability boundary. Full unbounded counts and levelled depth witnesses are
compared independently to the source below.

GRAPH37 implements hierarchical derivation in 68ea3256 and stored-call counting
in 3f5bb84a. Seventy-five new scenarios cover one to five hierarchy levels,
shared definitions, specialized overrides, name collisions, directional clean
and borrowed lifetime wrappers, zero-width words, shaped returns and custom entry
names. Arrays preserve source connections and original empty-array failures.
A raw CompositeBloq wrapped in source Adjoint keeps its original decomposition
refusal. The source-backed executable stored-component adapter uses the original
default Adjoint implementation without changing its body.

Two/three-level measured Add graphs are checked before and after expansion.
All four scenarios preserve each of 16 inverse and 16 forward restoration
histories, with exact unnormalized complex spectator state and clean work zero.
For two Add-adjoint calls, measured primitives remain 40 CX, 12 H, 8 T, 8 Tdag,
4 S, 4 measurements, 4 CZ and 4 resets. The independent source QEC model reports
16 T-equivalent, four measurements and peak width eight. Actual stored call
weights independently yield two Add-adjoint leaves. Generation adds zero gates
or workspace. Direct/weighted source counts, keep, levelled depth bounds and
many-to-one specialized adjoint aggregation pass.

Concentrated regression passed 5435 tests in 42.89 seconds without skips or
warnings. Focused coverage passed 116 in 6.69 seconds, with 97% combined statement
and branch coverage across component_adjoint and program_call_graph. The latter
adapter has 100% coverage. Integration passed 29 in 1.41 seconds. Strict typing
covered 47 files, and formatting, scope, architecture, Program/API gates and four
existing examples passed. Thirty-three pinned source blobs remain unchanged.
Default smoke/unit stopped with 22286 passes, 300 skips, 1065 warnings and seven
failures in 135.56 seconds. All seven identifiers match the independently recorded
baseline exactly. The ledger failure absent from GRAPH35/36 was exercised and
continues failing. The default layer remains open. Evidence is under
ftqc-reference-env/g2-hierarchy-adjoint-{source.txt,regression.log,coverage.log,
checks.log,integration.log,examples.log,default-parallel.log,default-comparison.txt}.
The next inventory is generic Controlled wrapper control propagation and counts
at the typed-call boundary. Parent files and G2 remain open.

## Positive controlled calls over the stored Program

GRAPH38 implements `component_control.control_component_program` in b3fbdd3c.
The internal Compiler interface accepts an existing typed-call Program and
explicit per-child controlled adders. It follows the original v0.7.0
`Controlled.build_composite_bloq` call chain. The all-THRU root gains one positive
scalar QBit, using the source control-name collision rule. Adders consume the
current control and child inputs, call already-defined specialized components,
and return the control plus data successors in the original output order.
Port renaming for CHadamard and array packing for CZPowGate are verified against
the original specialized get_ctrl_system callbacks. Core Program SSA remains
the authority. Forward definitions stay shared and the forward canonical payload
is cleared. Missing adders, incompatible returns, invalid unused definitions
and retained callers with obsolete entry arity are rejected. Adder exceptions
propagate. Specialized bodies retain ownership of work and measured cleanup.

Twenty-five new tests cover mixed H/Z-power sequences, one to three repetitions,
control-name collisions, empty and shaped ports, custom entries, invalid inputs
and adder replacement through the existing consumer boundary. Actual connections,
ports, ordering and stored call multiplicities match the original controlled
source graph. Every small forward measurement history preserves unnormalized
complex amplitudes, spectator entanglement and clean work zero. Two repetitions
also cover all 16 forward/inverse history pairs using existing exact adjoints.
Expansion preserves the emitted circuit. The original empty-array decomposition
failure is asserted separately while native declarations remain intact.

For two H/Z-power repetitions, measured primitives are 2 CH, 12 CX, 6 H, 4 T,
4 Tdag, 2 S, 2 phase, 2 measurements, 2 CZ and 2 resets, using one reusable clean
work qubit. Actual stored leaves are 2 CHadamard and 2 CZPowGate calls. The
independent source QEC model reports t=2, and_bloq=2, clifford=10, rotation=4 and
measurement=2, with peak width four including the spectator. Weighted actual
call counts close separately against every source QEC category. The wrapper
adds zero physical gates and workspace. Model categories retain source synthesis
assumptions and are recorded independently of primitive counts.

Concentrated regression passed 5460 tests in 29.81 seconds without skips or
warnings. Focused coverage passed 100 in 6.29 seconds, with 97% statement/branch
coverage for component_control. Its uncovered guard verifies the rebuilt module
after the existing builder has already verified it. Integration passed 64 in
5.30 seconds. Strict Python 3.12 typing passed 49 production/test files. Format,
architecture, team scope, Program/API checks and four append/resource examples
passed. Thirty-five pinned source blobs match the inventory Git objects.

The default smoke/unit fail-fast run recorded 22264 passed, 300 skipped,
1065 warnings and six failures in 114.43 seconds. All six identifiers match the
independent seven-failure baseline. The ledger baseline assertion did not run
before termination and remains open. Single negative/multiple controls,
automatic hierarchical specialization and arbitrary dynamic Python decomposition
remain separate acceptance work. G2 remains open.

Persistent evidence is under
`/Users/baai/Documents/liuwei/ftqc-reference-env/g2-component-control-` with
`source.txt`, `regression.log`, `coverage.log`, `checks.log`, `integration.log`,
`examples.log`, `default-parallel.log` and `default-comparison.txt` suffixes.

## Shared positive controls through stored hierarchies

GRAPH39 implements `control_component_hierarchy` in 99af2d3b. The Compiler walks
stored call-only definitions bottom-up and derives one shared controlled function
per unmapped child. Explicit specialized adders take precedence. Primitive and
mixed bodies require an explicit adder, and each derived body requires THRU
ports. The root is always derived. Generated names avoid all existing functions.
Forward child definitions remain shared. Control/data SSA, the source's control
naming and custom entry/block names use the same one-level implementation.
The original positive-control factory and generic _make_ctrl_system adder supply
the source comparison at every derived level.

This batch also fixes the previous one-level implementation dropping helpers
created inside an explicit adder. The same split/join/allocate/free regression
fails against the e67edadb implementation with unknown helper callees and passes
with retained definitions. The helpers are appended with their derived body and
shared across repeated calls. Existing immutable functions remain unchanged.
The original failure is retained in g2-hierarchy-control-helper-baseline.log.

Forty-three new scenarios cover one to five levels, repeated/shared definitions,
a diamond with direct calls across depths, explicit overrides, generated-name
collisions, zero-width words, scalar/matrix/empty ports, custom entries and
failure propagation. Original ordered signatures and named operand identities
are compared separately. The source generic adder inserts control last in its
input dictionary, while native traversal lists inputs in declaration order.
Both preserve the same named connections and ordered outputs. Native empty-array
declarations survive, while the source's empty final-soquet failure is asserted.

Every measurement history is exhausted for graphs with at most four measurements.
The sixteen-measurement graph checks three explicit histories, its complete
primitive stream and every component connection. It does not claim exhaustive
verification of all 65536 histories. Four expanded/unexpanded exact-adjoint
scenarios exhaust all forward/inverse history pairs. All checked histories retain
unnormalized complex amplitudes, spectator entanglement and clean workspace zero.

Four H/Z-power leaf pairs emit 4 CH, 24 CX, 12 H, 8 T, 8 Tdag, 4 S, 4 phase,
4 measurements, 4 CZ and 4 resets, reusing one clean work qubit. Actual stored
leaves are four CHadamard and four CZPowGate calls. The independent source model
reports t=4, and_bloq=4, clifford=20, rotation=8 and measurement=4 with peak width
four including the spectator. Every modeled category closes against weighted
actual stored calls. Structural hierarchy generation adds zero primitive gates
or workspace. Adder-created explicit lifetimes remain part of that adder's own
resource obligations.

Concentrated regression passed 5503 in 69.96 seconds without skips or warnings.
Focused coverage passed 68 in 9.09 seconds with 97% statement/branch coverage for
component_control. The two uncovered guards verify completed modules after the
builder has already validated them. Integration passed 64 in 9.48 seconds.
Strict Python 3.12 typing passed 50 production/test files. Format, team scope,
architecture, Program/API checks and four existing append/resource examples pass.
Thirty-five pinned source blobs remain equal to their inventory Git objects.

The default smoke/unit run recorded 22286 passed, 300 skipped, 1065 warnings and
seven failures in 242.83 seconds. All seven failure identifiers exactly match
the independently recorded baseline, including the ledger assertion. Negative
and multiple controls, their typed activation metadata and predicate cleanup,
and arbitrary symbolic/dynamic decompositions remain separate work. G2 stays
open. Evidence uses the `g2-hierarchy-control-` prefix under the existing
ftqc-reference-env directory, with source, regression, coverage, checks,
integration, examples, helper-baseline, default-parallel and default-comparison
files. The user append workflow is retained through the existing physical binder.


## GRAPH40 structured controls and directional predicates

Commits `34e5d4f4` and `fb2b256d` bind the existing And emitter and positive
hierarchy through typed Program ownership. Explicit outer control names precede
THRU data ports. Scalar and shape-(1,) negative QBits receive X before and after
the positive call. Multi-bit controls call the forward predicate, positive body
once and measured erasure. Forward allocates n-1 clean logical outputs. Opaque
junk and QBit target remain live during the child, whose work stays separate.
Erasure measures in reverse ladder order, applies feedback, resets and frees.
Control words retain row-major element order and MSB-first bit order.

Core currently admits unparameterized QBit, QAny and QUInt on this path. Other
integer, symbolic, parameterized and classical descriptors are rejected. Existing
raw-wire emitters retain their accepted wider profile. Automatic outer naming
and source Partition/And internal typed-node identity need separate acceptance.

The 63 new scenarios match source upper-level named connections and output
order, primitive operands, per-wire dependencies, parameters and feedback. The
existing literal ZPow boundary retains a phase parameter without Cirq's extra
T canonicalization. Histories through five measurements and forward/inverse
history pairs through three measurements are exhaustive. Complex amplitudes,
entangled spectators, clean work, expansion, callback replacement and existing
circuit.append(op) dynamic execution pass. Seven source QEC categories close
against actual native component-call multiplicities.

Five activation bits 10110 around H/ZPow(0.25) emit 8 X, 15 H, 10 T, 10 Tdag,
30 CX, 5 S, 1 CH, 1 phase, 5 measurements, 5 CZ and 5 resets. Private clean peak
is five and total width twelve. Independently, source QEC gives t=1, and_bloq=5,
clifford=17, rotation=2, measurement=5, other categories zero, total T-equivalent
43 and qubit peak twelve. The twenty emitted T/Tdag gates stay a separate count.

Local acceptance includes 5566 concentrated, 131 focused and 64 integration
cases, plus a 63-case exact-parameter rerun, with no skips or warnings. Strict
mypy checks 52 files. Formatting, scope, architecture, Program/API checks and
four append/resource examples pass. Predicate statement/branch coverage is 100%.
The control module reports 98%, with two final guards after builder verification
uncovered. 39 pinned source Git blobs match the inventory. All seven default
failure IDs equal the independent directional-call baseline.

Default raw summary. 7 failed, 22286 passed, 300 skipped, 1065 warnings in 449.87s (0:07:29).


## GRAPH41 nested control extension

Commits `390c2422` and `567ad61c` share the original control-name rule and
implement ControlledViaAnd.get_ctrl_system using the existing forward Program.
Additional clauses precede existing clauses in one combined predicate. Names
are regenerated against forward data ports. The returned adder pops old control
inputs, invokes the combined definition once and separates new controls from
old controls and data outputs. Tuple positions, array shapes, zip truncation and
partial-pop failure behavior match the original callback. The existing builder,
positive hierarchy, predicate ownership and measured erasure retain authority.

The 60 new scenarios compare parent calls and combined upper-level graphs,
signatures, outputs, primitive operands, exact parameters, per-wire dependencies
and feedback. Histories through five measurements and forward/inverse history
pairs through three measurements are exhaustive. Complex amplitudes, entangled
spectators, clean work, repeated extension and circuit.append(op) dynamic
execution pass. Seven source QEC categories close against actual stored calls.
The existing stored-body source-adjoint adapter is reused. The direct raw
CompositeBloq adjoint decomposition limitation remains open.

For five bits 10010 around H/ZPow(0.25), emitted primitives are 12 X, 15 H,
10 T, 10 Tdag, 30 CX, 5 S, 1 CH, 1 phase, 5 measurements, 5 CZ and 5 resets.
Clean peak is five and total width twelve. Source QEC independently reports
 t=1, and_bloq=5, clifford=21, rotation=2, measurement=5, other categories zero,
T-equivalent=43 and qubit peak twelve. Twenty emitted T/Tdag gates remain a
separate count. The existing literal phase boundary retains the ZPow parameter.

Acceptance includes 5626 concentrated, 191 focused and 64 integration cases,
strict mypy on 54 files, formatting, scope, architecture, Program/API checks
and four append/resource examples. The extension has 100% statement/branch
coverage. The control module reports 98%, with two final guards after builder
validation uncovered. 39 pinned source blobs remain unchanged. Default results
are 22264 passed, 300 skipped, 1065 warnings and six failures in 115.01 seconds.
Maxfail stopped that run. All six IDs belong to the independent seven-failure
baseline. The remaining baseline ledger test did not run in this selection.

Current typed admission remains unparameterized bit, opaque and uint. Other
Core descriptors, specialized single-control extension protocols and source
internal predicate-node identity need their separate acceptance slices.


## GRAPH42 packed specialized controls

Commits `9c7442b3` and `16adb7a3` reuse local helper-name isolation and implement
_MultiControlledFromSinglyControlled around an existing positive single-control
Program. The root replaces its leading scalar THRU QBit with a packed QBit
array in activation order. Three calls compute the predicate, invoke the positive
child and erase the predicate. Source junk is a QBit array for more than two
controls. Existing split/join adapters retain shared opaque-word ownership
without extra primitive gates. The valid two-control factory packs new then old
control in the original input dictionary and returns their separate successors.
Compiler owns binding and transforms. Existing circuit.append(op) execution
continues through the shared Runtime path.

The 91 new cases compare root and three child signatures, named graph connections,
primitive operands, exact parameters, per-wire dependencies and measurement
feedback. All control polarities at widths two and three and two patterns each
at widths four and five pass. Every measurement history for each tested pattern
is exhaustive, at most 32 branches. Two patterns each at widths two and three
also exhaust all forward/inverse history pairs, at most 64 pairs. Full complex
amplitudes, entangled spectators, zero work, expanded calls and eight dynamic
trajectories pass. Larger widths and other four/five-bit patterns are untested.
The existing stored-body source-adjoint adapter retains the original decomposition.

The five-bit 10101 example emits 8 X, 15 H, 10 T, 10 Tdag, 30 CX, 5 S, 1 CH,
1 phase, 5 measurements, 5 CZ and 5 resets. Five clean work wires yield total
width twelve. Independent source QEC reports t=1, and_bloq=5, clifford=17,
rotation=2, measurement=5, other categories zero, T-equivalent=43 and peak twelve.
Twenty emitted T/Tdag gates remain separately counted. Seven logical categories
close against actual stored child calls. Literal ZPow phase parameters remain
at the previously verified primitive boundary.

Acceptance includes 5717 concentrated cases in 39.25 seconds, 282 focused cases
in 42.84 seconds and 64 integration cases in 4.20 seconds. Strict mypy checks
56 files. Formatting, team scope, architecture, Program/API checks and four
append/resource examples pass. The specialized module and shared predicate have
100% statement/branch coverage. Control retains two uncovered final guards and
98% coverage. All 41 original source blobs match the pinned inventory.
Default results are 22179 passed, 300 skipped, 1065 warnings and five failures
in 106.35 seconds. Maxfail stopped the run. All five IDs are a subset of the
independent seven-failure baseline. The other two baseline IDs were unobserved.

The previously recorded further-single-control callback TypeError from the
original np.concatenate call remains reproducible for both added polarities.
Direct three-control helper execution does not certify that failing callback.
Source files remain unchanged. Generic multi-register fallback, direct positive
and negative specialization selection, AdjointWithSpecializedCtrl, internal And
node-by-node graphs and complete parent-file acceptance remain open. Core
admission continues to cover unparameterized bit, opaque and uint registers.
Evidence is retained under ftqc-reference-env/g2-specialized-control-*.


## GRAPH43 specialization selection and controlled adjoints

`446798f2` implements single-control specialization selection. Multi-bit and
multi-register requests delegate unchanged to the existing fallback owner.
Single-bit requests query their activation, with unavailable negative control
falling back and missing positive control rejected. Existing control queries
only the positive specialization and reuses the packed new/old helper. The
direct adder inserts the control into the original input dictionary and returns
control/data successors in declaration order. Missing positive specialization
raises native ValueError, where the source raises AssertionError. Invalid input
exception types are an explicit admission difference.

`0d025cb1` implements the admitted AdjointWithSpecializedCtrl protocol. A
frozenset of activations expresses NONE, ZERO, ONE and BOTH. Selected single-bit
requests use the actual forward control factory. Other requests delegate to the
inverse fallback owner. Existing exact adjoints reverse both the selected body
and the graph built by its real returned adder. Data ports precede the temporary
control in that graph, matching the original builder. Its inverse remains a
stored local adapter. One caller expansion reproduces source add_from inlining.
Nested returned-adder calls preserve generated inverse helpers and retarget them
to the same selected inverse root. Compiler retains ownership of transforms,
while the existing Program remains the only graph representation.

Acceptance covers 68 new cases, 5785 concentrated cases in 42.56 seconds,
350 focused cases in 44.09 seconds and 64 integration cases in 4.74 seconds.
Strict mypy checks 60 files. Formatting, team scope, architecture, Program/API
checks and four existing append/resource examples pass. Both new modules have
100% statement/branch coverage. Shared predicate, extension and packed helpers
retain 100%. Control remains at 98% with two existing uncovered final guards.
All 41 original source blobs match the pinned inventory.

Direct positive/negative control, missing negative specialization, all four
old/new polarities, multi-bit dispatch at widths two, three and five, mixed
QBit/QUInt groups, all four adjoint masks, control-name collisions, adapter-name
collisions, nested adders and wrong callback arity pass. All two single-control
measurement histories, all four forward/inverse history pairs and all eight
mixed three-bit histories preserve full complex amplitudes, entangled spectators
and zero work. Positive and negative inverse append paths each pass eight
dynamic trajectories. The mixed fallback's actual native root calls close
against seven source QEC categories and source QubitCount.

A positive single-control H/ZPow(0.25) emits 1 CH, 3 H, 2 T, 2 Tdag, 6 CX,
1 S, 1 phase, 1 measurement, 1 CZ and 1 reset. Negative control adds 2 X.
The inverse has phase parameter -0.25 and places CH after that operation,
with the same counts. Clean peak is one and total width four. Independent source
QEC reports t=1, and_bloq=1, clifford=5 for positive or 7 for negative,
rotation=2, measurement=1, other categories zero, T-equivalent=27 and peak four.
Four emitted T/Tdag gates and 27 modeled T-equivalents remain separately listed.

AutoPartition input unwrapping, shaped single-bit specialization, directional
and custom-entry scenarios, complete internal node graphs and complete parent
file acceptance remain open. Core admission remains unparameterized bit, opaque
and uint. The original further-single-control callback TypeError remains open.
Source algorithms remain unchanged. G2 continues in the same integration PR.
Detailed evidence is retained under ftqc-reference-env/g2-control-dispatch-*.

The final default smoke/unit run reports 22286 passed, 300 skipped,
1065 warnings and seven failures in 108.44 seconds. Maxfail stopped that run.
All seven failure IDs exactly match the independently recorded baseline.
Those baseline failures remain unresolved by this batch.


## GRAPH44 typed automatic partitions and specialization boundaries

`b8898c01` moves existing local helper isolation into component_graph for its
predicate, packed-control and layout consumers. `6c670622` appends a typed packed
entry around an existing child entry, retaining the child as a separate callable.
The existing AutoPartition layout owns group names, widths, unused slices and
left_only rules. Calls follow the original Partition, child, inverse-Partition
sequence. Internal rebindings use existing builder split/join functions. Child
array elements are row-major and words are MSB-first. Unused portions stay live
across the child. left_only exposes child right ports and consumes packed inputs.
Physical frees require explicit input_ownership transfer. No partition adds a
primitive instruction or workspace.

`58add03c` verifies the distinction between the original specialization factory's
unwrapped returned child and its wrapped callback target. Native retains the
child root and selects the explicit packed entry for the actual callback.
One and two wrapper layers preserve source parent connections, mutation, output
order and primitive execution. Custom entry names and a nonleading control port
execute correctly. Scalar opaque packed external ports are the current admission.
Other external dtypes, shaped outer words, empty axes and symbolic layouts are
not covered by this slice.

Acceptance includes 25 new cases, 5810 concentrated cases in 40.68 seconds,
375 focused cases in 54.31 seconds and 64 integration cases in 4.29 seconds.
Strict mypy checks 63 files and formatting checks 89. Team scope, architecture,
Program/API checks and four existing append/resource examples pass. All 45
original source blobs match the pinned inventory. Initial concentrated coverage
was 98% for the new module. `ad02b6ad` removes a redundant direction guard already
implied by layout and child-output checks. Final 25-case verification passes in
5.52 seconds with 100% statement/branch coverage, followed by strict typing and
format checks on the changed module.

Three packed layouts exhaust both child measurement histories and preserve full
complex amplitudes, entangled unused/spectator bits and zero child work.
Two bit-array shapes with repack or left_only preserve the source's complete
tensor and output-wire permutation. A consuming LEFT port passes source tensor
comparison with transferred input ownership. Existing circuit.append(op)
execution preserves eight dynamic trajectories. Source root signatures, named
connections, primitive parameters, per-wire order and feedback agree. Native
root call multiplicities close against source bloq_counts, seven QEC categories
and QubitCount.

A packed ctrl/unused/q/spectator example emits 1 CH, 3 H, 2 T, 2 Tdag, 6 CX,
1 S, 1 phase, 1 measurement, 1 CZ and 1 reset. Partitions add zero gates and
zero work. Child clean peak is one and total width five. Independent source QEC
reports t=1, and_bloq=1, clifford=5, rotation=2, measurement=1, other categories
zero, T-equivalent=27 and peak five. Four emitted T/Tdag gates remain separately
reported. The unused bit contributes to the actual peak.

The original shape-(1,) single-control array passed to a scalar specialization
asserts for both polarities. Native retains that collection and the same
AssertionError, with no scalar coercion or successful-execution claim. Original
RIGHT-only children fail construction. LEFT-only children repacked on both sides
fail decomposition with KeyError. Native rejects those layouts earlier with
ValueError, an explicit invalid-input timing/type difference. Valid left_only
consumption is separately accepted. Source files remain unchanged.

Default results are 22179 passed, 300 skipped, 1065 warnings and five failures
in 114.89 seconds. Maxfail stopped that run. All five IDs belong to the independent
seven-failure baseline. The other two baseline IDs were unobserved. Packed
wrapper adjoints and existing-control combinations, broader Core admission,
complete internal graphs and parent-file audits remain open. G2 continues.
Evidence is retained under ftqc-reference-env/g2-typed-partition-*.

## GRAPH45 packed existing controls and specialized-adjoint callbacks

`369a6f85` removes an extra native restriction from the stored specialized-control
helper. The original source selects its scalar control by name, removes that
port from its child signature and prepends the QBit control array. Native now
follows that order for nonleading controls and admitted one-qubit dtypes,
including the opaque port exposed by AutoPartition. Other data ports retain
their declaration order. Array order is new control, old control. Existing type
compatibility, partition ownership, predicate cleanup and exact-adjoint owners
remain authoritative; no Core admission, execution graph or user API is added.

Thirty-two scenarios cover one/two packed layers, leading/nonleading control,
all four old/new polarities and retained/expanded calls. Each checks the original
root signature, named connections, callback mutation/output order, complete
primitive parameters, dependencies and measurement feedback. Actual native root
call counts close against source bloq_counts, seven QEC categories and peak
width. Forward and exact-inverse children exhaust four measurement histories
per direction and all sixteen history pairs. Full complex amplitudes, entangled
unused/spectator bits and zero workspace agree. Eight dynamic trajectories pass
through the existing circuit.append(op) path.

Four additional scenarios exercise the original AdjointWithSpecializedCtrl's
actual packed callback for a positive single-control specialization. The returned
unwrapped child and callback's packed entry remain distinct. Explicit inverse
Partition pairs and exact child leaves let the existing adjoint owner reverse
the actual adder graph. After one adapter expansion, its connections agree with
the original add_from callback. Primitive streams and all four forward/inverse
measurement pairs agree. This is separate from taking the adjoint of an entire
existing-control returned-adder; that latter path for all four old/new polarities
remains the next slice. An inverse child inside a forward predicate does not
certify it.

Acceptance: 38 added cases; 5848 concentrated in 63.21 seconds, 413 focused in
69.97 seconds, and 64 integration in 3.58 seconds. Final 129-case verification in
26.85 seconds measures all 65 statements and 12 branches of the modified helper
at 100%. Strict mypy checks 64 files; formatting checks 90. Lint, scope,
architecture, Program/API checks and four existing append/resource examples
pass. All 45 pinned source blobs are unchanged.

For old=0/new=1, both child directions emit 4 X, 6 H, 4 T, 4 Tdag, 12 CX,
2 S, 1 CH, 1 phase, 2 measurements, 2 CZ and 2 resets. Inverse phase sign and
CH placement follow the original inverse child. Two clean work wires and seven
physical wires include the unused bit. Partitions add no primitive or work.
The separate source model reports t=1, and_bloq=2, clifford=6, rotation=2,
measurement=2, other categories zero, T-equivalent=31 and peak seven. Eight
emitted T/Tdag gates are not labeled as 31 primitive gates. The single-control
specialized-adjoint callback emits four T/Tdag and one measurement; its model
reports T-equivalent=27 and peak five including the unused bit.

Default smoke/unit results: 22434 passed, 300 skipped, 1065 warnings and nine
failures in 126.32 seconds, stopped by maxfail. Six failure IDs belong to the
independent seven-failure baseline; one baseline ID was unobserved. The other
three failures were unchanged multinode tests denied local socket binding by
the sandbox. Those three pass separately with socket access in 0.10 seconds.
This does not certify the entire default suite or repair its baseline failures.
Source further-single-control callback defects, broader layout boundaries,
complete internal graph audits and whole-file acceptance remain open. Evidence
is retained under ftqc-reference-env/g2-packed-control-*.

## GRAPH46 whole packed existing-control specialized adjoints

`d084244d` verifies the complete returned-adder path through the original
AdjointWithSpecializedCtrl, existing-control specialization, multi-control
helper, AutoPartition and positive child. The existing native adjoint owners
already express that chain. This batch adds composition acceptance without
another production adjoint implementation. Both the current packed operation
and selected helper use the explicit packed entry. Exact mappings pair And
compute/erase, Partition/inverse-Partition and child leaves. Predicate and child
workspace retain their separate lifetimes. Controls remain new then old; data
is MSB-first q/unused/spectator. Users still append the bound operation.

Sixteen valid scenarios cover all four old/new polarities, one/two packed layers
and expanded/retained calls. Returned root signatures and named connections
match the original. Expanding one native adapter matches the real source
add_from callback, including dictionary mutation and output order. Complete
primitive operands, parameters, per-wire dependencies and feedback agree.
Native caller leaf counts close against source composite.bloq_counts, seven QEC
categories and peak width. Every inverse measurement history agrees with an
independent complex-amplitude formula; all sixteen forward/inverse history pairs
restore the entangled input and zero workspace. Returned roots and actual
callbacks bind to the same CircuitIR. Four append scenarios each verify eight
dynamic trajectories.

Eight diagnostic scenarios separately retain a source failure. For a nonleading
existing control and a three-qubit packed data port, the existing-control adder
returns old-control/data while the specialized-adjoint builder zips outputs in
original data/control declaration order. Its finalize therefore assigns QBit to
QAny(3), raising original BloqError. Native rejects the same connection with
RegisterBindingError. Both derive from ValueError, but exact exception classes
are not claimed equal. Source declarations are not reordered to repair the
reference, and these cases are not counted as successful execution. Equal-width
nonleading ports remain unaudited; the dtype failure does not establish their
behavior. The original further-control concatenate defect also remains open.

All 28 new cases pass in 7.32 seconds; 5876 concentrated in 64.26 seconds,
441 focused in 81.85 seconds and 64 integration in 3.60 seconds pass without
skips or warnings. Strict typing checks 65 files; formatting checks 91. Changed
tests receive a final strict/lint recheck after narrowing the native diagnostic
exception. Scope, architecture, Program/API checks and four existing examples
pass. The adjoint-control, selector, extension, predicate, packed-helper and
typed-partition owners all measure 100% statement/branch coverage. The older
component_control owner remains 98%, with two existing final guards uncovered.
All 45 pinned source blobs are unchanged; the native Kickmix Simulator and
versions are checked before the reference suites run.

All four whole adjoints emit 6 H, 4 T, 4 Tdag, 12 CX, 2 S, 1 phase,
2 measurements, 2 CZ, 2 resets and 1 CH. Phase is negative and CH follows it.
Additional X count is zero for old=new=1, four for one zero, and eight for two
zeros. Clean peak is two and total width seven including the unused bit.
Packing adds no gates or work. Separately, the original model for each polarity
reports t=1, and_bloq=2, clifford=6, rotation=2, measurement=2, other categories
zero, T-equivalent=31 and peak seven. Its clifford=6 is not a count of all literal
Clifford gates; emitted X and eight T/Tdag remain explicitly reported.

Default smoke/unit results are 22286 passed, 300 skipped, 1065 warnings and
seven failures in 124.96 seconds, stopped by maxfail. All seven failure IDs
exactly match the independent baseline. No added or missing failure IDs are
observed. The run allows local sockets and does not reproduce the previous
sandbox socket failures. The baseline is not repaired and whole-suite success
is not claimed. Equal-width nonleading output permutations, left_only control
wrappers, broader type/shape admission and complete parent-file audits remain
open. Evidence is retained under ftqc-reference-env/g2-packed-adjoint-*.

## GRAPH47 equal-width defect audit and paired directional controls

`54232a82` diagnoses an additional original specialized-adjoint defect. Equal
one-qubit control/data ports let a nonleading existing control pass dtype
checks. The actual returned-adder callback is the intended inverse preceded by
an input SWAP of old-control/data; the returned root remains the intended inverse.
Sixteen scenarios cover leading/nonleading order, one/two packed layers and all
four polarities. Eight nonleading scenarios are defect evidence rather than
acceptance of the intended operation. Native reproduces the original graph and
output-binding permutation. Complete original tensor matrices and all four
measurement histories agree with an independent formula for the faulty path,
and differ from the intended inverse. Work still clears. Root and callback
have the same primitive gate counts, QEC model and peak, demonstrating why
those metrics alone cannot certify semantics. The matrix Frobenius distance
from the intended inverse is 2.8284271247461903. With both activations one,
input |001> in added-control/old-control/data order remains |001> under the
intended inverse but becomes |010> through the callback. No literal SWAP gate
is needed: returned register bindings encode the permutation. Source files are
unchanged; the faulty branch is not certified.

`3ece32ae` fills a separate native admission gap. Original left_only wrappers
can expose a same-named LEFT opaque one-qubit control and RIGHT QBit control.
The original multi-control helper removes both declarations by name and prepends
one THRU QBit array. Native now requires exactly one scalar one-qubit input and
output for that name, accepting either one THRU declaration or a LEFT/RIGHT pair.
Missing sides, arrays and wider controls remain rejected through existing
executable type admission. Other port order, new/old activation order, predicate
compute/erase and child work ownership are preserved. There is no new Core
admission, execution graph or numerical kernel.

Fourteen directional scenarios include four polarities with expanded/retained
calls, four append paths with eight trajectories each, and two source failure
boundaries. Forward LEFT packed data exposes the child's separate RIGHT
q/spectator ports; the whole adjoint consumes those ports and returns packed
data. Existing layout functions supply explicit inverse Partition leaves to the
existing adjoint owner. Source returned-root signatures/connections, actual
callback graph after one adapter expansion, dictionary mutation, output order,
primitive parameters, per-wire order and feedback agree. Returned root and
callback bind to the same CircuitIR. Every inverse history matches an independent
complex-amplitude formula; all sixteen forward/inverse history pairs restore
the entangled input. Two clean work wires clear, and caller control/data wires
are never freed. This validates shared controlled H/phase components, not a
complete THC lookup.

Nesting a directional wrapper as an AutoPartition child fails the original
all-declarations-on-left requirement and native admission. Renaming its external
control without a same-named child output makes the original multi-control
helper fail later with KeyError. Native rejects the missing output earlier with
ValueError; exception type/timing equivalence is not claimed. Neither case is
counted as successful execution. Freeing unknown inputs and RIGHT-only children
are not newly supported.

All 30 new tests pass in 5.96 seconds; 5906 concentrated in 62.60 seconds,
471 focused in 93.44 seconds and 64 integration in 3.17 seconds pass without
skips or warnings. Strict typing checks 67 files; formatting checks 93. Lint,
scope, architecture, Program/API checks and four existing examples pass. The
modified helper's 65 statements and 12 branches have 100% coverage, as do the
adjoint-control, selector, extension, predicate and typed-partition owners.
The older component_control owner remains 98% with two existing final guards
uncovered. All 45 original blobs match the pinned inventory and native reference
runtime/version assertions run before the suites.

Directional inverse counts are 6 H, 4 T, 4 Tdag, 12 CX, 2 S, 1 phase,
2 measurements, 2 CZ, 2 resets and 1 CH, plus zero/four/eight X according to
activation polarity. Phase is negative and CH follows it. Two clean work wires
and six physical wires have no unused slice. Source QEC separately reports
t=1, and_bloq=2, clifford=6, rotation=2, measurement=2, other categories zero,
T-equivalent=31 and peak six for all polarities. Eight emitted T/Tdag and literal
X counts remain separate. Partitioning adds no primitive gate or work.

Default smoke/unit reports 22264 passed, 300 skipped, 1065 warnings and six
failures in 122.18 seconds, stopped by maxfail. All six IDs are members of the
independent seven-failure baseline. The ledger-scanner baseline case was not
observed. There are no newly observed failure IDs; whole-suite success is not
claimed. Next are left_only direct single-control callbacks and multi-bit
CtrlSpec fallback. The nonleading source callback defect, further-control
concatenate defect and complete parent-file audits remain open. Evidence is
retained under ftqc-reference-env/g2-directional-control-*.
