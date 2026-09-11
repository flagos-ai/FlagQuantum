# Phase 26 evidenced native-gate legalization evidence

Date: 2026-09-10

## Accepted profile

Compiler consumes the selected target snapshot's verified `gates.native` fact.
It canonicalizes opcode strings and descriptor objects while preserving each
parameter-name variant independently. A gate is native only when one complete
descriptor covers the canonical parameters required by its operator schema.

The first exact decomposition set is X to H-Z-H, RX to H-RZ-H, and RY to
Sdg-H-RZ-H-S. Generated instructions reuse the original parameter objects and
copy condition metadata. Circuit observables, measurements, dtype, shape, and
metadata remain under the existing Core `CircuitIR` authority. Expansion is
bounded to 256 added operations by default.

Target legalization now applies native-gate legalization first. Its resource
and backend checks therefore observe the actual post-decomposition CircuitIR.

## Verification summary

| Gate | Result |
| --- | --- |
| Current native-gate evidence is required | pass |
| Opcode and parameter-name descriptors normalize deterministically | pass |
| Separate descriptor variants remain separate | pass |
| Native circuits preserve the exact CircuitIR instance | pass |
| RX and RY decompositions match source statevectors | pass |
| Trainable parameter objects survive decomposition by reference | pass |
| Dynamic condition metadata is copied to every replacement | pass |
| Missing basis gates fail before target emission | pass |
| Expansion overflow fails closed | pass |
| Phase 25 operation limits observe expanded size | pass |
| Phase 26 native-gate tests | 7 passed |
| Hybrid compiler and private-contract focused suite | 181 passed |

The wider unit run excluding the already identified strong-scaling registry
failure completed 1,335 tests successfully with 14 skips. Its only failure was
an accumulator rollback assertion in `test_trajectory_runtime.py`; Phase 26
does not modify Runtime trajectory statistics or that test.

## Claim boundary

Descriptor matching covers opcode and parameter names, not parameter ranges,
periodicity, calibration, fidelity, or duration. The closed decomposition set
is exact and does not authorize arbitrary unitary synthesis or approximation.
Topology routing, scheduling, timing, pulse lowering, artifact negotiation,
target-format emission, provider execution, TargetIR, MLIR/LLVM/QIR, public
APIs, default-path changes, and performance claims remain outside this phase.
