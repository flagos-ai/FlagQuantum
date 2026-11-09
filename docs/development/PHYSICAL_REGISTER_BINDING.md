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

The admitted entry has only quantum input/output ports and balanced nonempty
workspace. Releasing caller inputs or returning nonempty live workspace requires
caller ownership-transfer semantics and receives an explicit diagnostic here.
Internal boolean and scalar calls are supported. General classical boundary
ports, symbolic widths and physical control-flow blocks remain outside this
profile. Conditional measurement/reset follow the existing CircuitIR rejection
rules. Zero-qubit physical circuits remain outside the existing CircuitIR schema.

Clean-zero and borrowed-joint-state restoration are component obligations.
Binding does not numerically prove arbitrary emitters. The focused consumers
supply that evidence for their actual paths. Whole G2 and full source-file
acceptance remain open.
