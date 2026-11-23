# Physical circuit fuzzing source closure

## Ownership and interface

The locked source is Kickmix `sim/fuzzer.cc`, `sim/fuzzer.h` and
`sim/fuzzer.test.cc` at `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.
Testing owns the private sampler orchestration and failure diagnostics.
Compiler owns the existing `KmxCircuitBuilder`, immutable circuit, register
metadata and phase values. Core owns `FixedWidthInteger`. Simulation owns
fixed-width sampling, ChaCha measurement draws, register transposition and
packed basis execution. Drawer owns circuit text. Production execution does
not import Testing.

Callbacks use the existing builder and caller-owned uint64 iterator. Input and
output samples expose existing immutable fixed-width values by register name.
A write replaces a cell through `sample[name] = successor`. Register members
and scalar bits are LSB-first. Output `old(name)` retains the input snapshot.
Quantum workspace belongs to the circuit and its explicit special registers.
No new stable namespace export or execution entry point is introduced.

The sampler uses 64 packed lanes, including inactive tail lanes during numerical
execution. Only requested lanes invoke callbacks and contribute to fuzzer error
statistics. Primitive counters describe actual executed lanes separately.

## Intake

All 86 original source and dependency blobs match the locked inventory. All
8 unchanged original GoogleTests pass. The checked-in oracle captures 36 source
scenarios and 1,100 trace lines, including unused warnings. Regeneration is
byte-exact. The same original increment, four incorrect increments, phase and
XOR scenarios pass through native callbacks.

## Native ownership adaptations and source boundaries

Immutable values replace mutable C++ integer references. Explicit `close()` and
context management own the unused-fuzzer warning. Python garbage collection
timing is outside the lifecycle contract. Source copy construction and assignment
are implicitly deleted. The native helper refuses copying.

Output-touch flags persist across batches exactly as observed in the original.
Input flags reset for each active lane. Special initialization precedes callback
conflict checks. Names determine lexical random-draw order. Configuration counts
advance before a failing builder. Duplicate names are rejected by both builders,
with the existing native Compiler exception wording retained. No map-overwrite
behavior is claimed for rejected duplicate names. Original snapshots remain
unchanged across numerical execution and repeated batches. Error counts persist
across calls, with only the first error receiving detailed register/phase grids.
The exact source diagnostic bytes match for admitted names and registers.
Six SIMD architecture/runtime parents remain separately open.
Pending Proposals 116–119 remain unimplemented.


## Complete definition disposition

| Source responsibility | Native owner and evidence |
| --- | --- |
| TouchedRegister, InputSample, OutputSample, old input references | Testing named cell views over existing immutable Core values. Access, reassignment and original snapshot checks pass. |
| Constructor, warning, ownership | Existing Simulation MT/ChaCha draws seed immediately. Context/close warning matches four source scenarios. Copy refusal matches deleted C++ ownership. |
| Four callback setters | Testing stores callbacks. Separate/combined modes, all conflicts and missing touches match source traces. |
| switch_to_new_circuit | Existing Compiler builder validates and freezes. Unassigned quantum wires receive a zero-valued special register. No automatic classical register is added. |
| fuzz_single_batch | Testing composes 64-lane scalar buffers, Simulation transposition and the existing packed executor. Active callback/tail order, special values, measurement draws, phase and input snapshots match source. |
| process_shot_data and grid_string | Testing owns exact UTF-8 byte widths, register/member order, first-error grid, diagram thresholds and accumulated counts. Existing Drawer renders circuits. |
| fuzz_current_circuit and fuzz | Testing preserves zero shots/variations, repeated batches, configuration changes and errors. Native counts additionally reject values outside source size_t. |
| Complete original tests | All eight original C++ tests pass unchanged. Native scenarios retain successful XOR/increment/phase and all original incorrect increment/phase failures. |

## Verification and resources

The batch passes 1,071 concentrated checks, 121 local integration checks and
strict typing for five Python files. Both new modules have 100% statement and
branch coverage. Black, Ruff, team scope, architecture, multi-level IR and API
checks pass. The existing 720 dependency findings are unchanged as a complete
line set. Locked Kickmix 0.1, Simulator availability and extension SHA-256 are
verified before acceptance.

Replacing the packed execution function with existing one-lane execution leaves
all 32 numerical/callback source traces unchanged. Three forced measurement masks
exercise HMR, feedback Z and a final X across two configurations and 129 requested
shots per configuration. Each branch preserves exact phase, original input
snapshots and zero quantum outputs. Primitive counts reflect all 384 executed
packed lanes, including inactive tail lanes. They are 384 HMR, 384 X and
`6 * popcount(outcome_mask)` conditional Z. Fuzzer statistics count 258 requested
shots. The orchestration and transposition themselves emit zero primitives,
measurements or quantum workspace. The analytical resource model is inapplicable
and recorded separately as `None`.

Existing 16 normal `circuit.append(op)` source cases are rerun with complete
complex state, workspace zeroing and 512 exact phase/actual-counter lanes.
Existing complete measurement-branch and QROAM erasure checks are also rerun.
This batch introduces no replacement circuit entry or simulation arithmetic.

External receipts are `ftqc-reference-env/g2-fuzzer-*-20261122.*`.
