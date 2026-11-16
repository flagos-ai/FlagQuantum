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

Dense declarations keep names and membership. The locked C++ probe also
reserves r0 before appending a two-wire register named `quantum` and one
classical bit named `classical`. It reports the name `quantum` on r0 and
the two wires on r1. The dense control probe appends a zero-width named r0,
then assigns both the name and the wires to r1. The reserved case follows
from appending the name to the end of `register_data` before the membership
routine grows it to the reserved ID. Declaration lowering and source
compatibility for sparse IDs remain open.

K-0005, K-0006 and K-0008 remain open for the remaining whole-file families.
Broadcast paths with classical or X-basis targets, owning C++ scope guards,
register declarations, general loop patterns and debug metadata beyond
named block ranges have separate source assertions.
The original `CircuitBuilder.*` suite passed all 27 tests with GoogleTest
1.14.0 from the locked Bazel manifest. The build compiled the unchanged
builder, original tests and test helpers, and linked the remaining native
objects from the existing reference extension build. The native predicate
intake passed 631 cases and the focused gate and arithmetic regression passed
185 cases. The C++ baseline still needs per-family native replacement before
the original-test parent closes.
