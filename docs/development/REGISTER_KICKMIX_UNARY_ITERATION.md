# Unary address traversal contract

G2-KCUR01 ports the unchanged locked Kickmix `UnaryIterationCursor` and
`gen_unary_iteration` schedule from commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`,
`src/kickmix/gen/iteration/gen_unary_iteration.{cc,h}`.

The compiler owns traversal, not numerical execution. Address ports are
big-endian at the FlagQuantum boundary; the source level loop uses their
reverse. A caller supplies `max(1, address_width)` clean workspace wires,
reserved through `RegisterWorkspace`, in source low-level order. The first
workspace wire is the match wire. Construction emits nothing. Values are
unsigned 64-bit integers, addresses have at most 63 bits, and every value
outside the address space parks the cursor. Duplicate visits emit nothing.
An explicit `close()` parks it and emits measured cleanup; destruction emits
nothing. Callbacks must preserve address, control and match bit values while
live. Callback operations and result slots remain caller-owned.

The returned reversible-phase operations preserve resets, negative-control
X pairs, boundary CX, HMR and feedback operand ordering. Erasures reuse one
caller-owned result slot because each feedback consumes it before overwrite.
The full operation binds through the existing reversible-phase execution
adapter and remains usable through `circuit.append(op)`. No Stable Core schema
or public exports change. Existing Qualtran SELECT schedules remain unchanged.

Acceptance requires native source gate streams (including repeated result-slot
reuse), coherent measurement branches with entangled spectators, work cleanup,
resource counts and a runnable append example. Source primitive counts and
expanded Core IR counts are separate; neither is a model T estimate.

## Local acceptance

The source `.cc` blob is `ec97900dbce6ac31ca66a33eb63e95ea7073fb1d`,
verified against the pinned upstream Git object. An independent C++ harness
links the unchanged native objects and runs 90 complete streams at widths
0, 1, 2, 3, 4, 8, 16, 32 and 63, controlled and uncontrolled. Cases include
singletons, boundary jumps, reverse visits, duplicates and parked callbacks.
Only register declarations and informational tags are excluded. Resets,
classical slot reuse and every feedback operand remain in the fingerprints
committed in `tests/fixtures/kickmix_unary_iteration.json`.

Every unnormalized complex branch is checked through two address bits,
including jumps, reentry after parking, duplicate visits, coherent quantum
control and entangled target/spectator inputs. Both reset Kraus outcomes and
HMR outcomes are enumerated, not sampled or coherently summed. All branches
implement the selected-address XOR with all work zero. Larger widths have
source-stream evidence, not dense statevector evidence.

The controlled two-bit traversal of 0, 1, 2, 3 emits 25 source primitives:
6 X, 3 R, 3 CCX, 7 CX, 3 HMR and 3 CZ. Its bound Core IR instead contains
one classical initialization, 9 X, 6 H, 3 reset, 3 CCX, 7 CX, 3 measure and
3 CZ. These are syntactic counts, including feedback; no T synthesis or
model T conversion is inferred. Logical work is two qubits and one reused
classical result. Extra caller-supplied work is untouched.

Run `python -m examples.algorithms.unary_address_traversal` for the coherent
append example. No upstream package is required by the example. The body
must not change live address/control/match values; arbitrary user callbacks
cannot be statically proven to satisfy that precondition. Native fuzz/performance
coverage and classical outer condition scopes remain separate ledger items.

Local acceptance: 28 new tests; 299 focused regression tests passed with no
skips. Strict mypy (two production/example files), Black, Ruff, compiler scope,
architecture and diff checks passed. Existing Fourier, exact-phase, arithmetic
measurement and resource-model consumers were included in regression.
