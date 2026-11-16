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

K-0005, K-0006 and K-0008 remain open for the remaining whole-file families.
Broadcast and parity controls, owning C++ scope guards, register declaration
bookkeeping, loop patterns and debug metadata have separate source assertions.
The original `CircuitBuilder.*` suite passed all 27 tests with GoogleTest
1.14.0 from the locked Bazel manifest. The build compiled the unchanged
builder, original tests and test helpers, and linked the remaining native
objects from the existing reference extension build. The native predicate
intake passed 631 cases and the focused gate and arithmetic regression passed
163 cases. The C++ baseline still needs per-family native replacement before
the original-test parent closes.
