# Phase 25 capability-driven target-legality evidence

Date: 2026-09-10

## Accepted profile

The Compiler derives mandatory target requirements from a validated Core
`CircuitIR`, validates each distinct opcode against one explicitly selected
backend in the existing operator-lowering registry, and matches the result
against one caller-supplied Core Target Capabilities v1 snapshot.

The requirement set covers logical-qubit capacity, maximum program operations,
effective precision, semantic measurement-result kinds, and maximum finite
shots when applicable. Every fallback authorization remains false. Effective
precision requires observed evidence; static target limits may use declared or
observed evidence under the existing Core contract.

Successful legalization retains the exact `CircuitIR` instance and its content
hash. The audit identity binds the circuit, backend, requirement set, target
snapshot, and resolved lowering strategies. It neither selects nor probes a
target and does not execute the circuit.

## Verification summary

| Gate | Result |
| --- | --- |
| Hybrid lowering enters target legality as the existing `CircuitIR` | pass |
| Circuit content and live trainable parameter reference are preserved | pass |
| Requirement and legalization identities are deterministic | pass |
| Fallback authorizations remain false | pass |
| Logical-width mismatch fails before emission | pass |
| Program-operation limit mismatch fails before emission | pass |
| Effective-precision mismatch fails before emission | pass |
| Measurement-result mismatch fails before emission | pass |
| Finite-shot limit mismatch fails before emission | pass |
| Missing backend opcode lowering fails before capability acceptance | pass |
| Phase 25 target-legality tests | 10 passed |
| Hybrid compiler and private-contract focused suite | 172 passed |
| Repository pre-commit checks on the Phase 25 file set | pass |

The current whole unit suite completed 1,329 tests successfully with 14 skips.
Its only failure is the pre-existing strong-scaling runner module-ownership
assertion in `test_statevector_strong_scaling_runner.py`; Phase 25 does not
modify the benchmarking registry or that test.

## Claim boundary

This phase establishes a capability-driven legality seam, not a new IR layer.
The existing `CircuitIR` remains authoritative. Native-gate descriptor matching,
decomposition, coupling-map legalization, scheduling, target-format emission,
artifact negotiation, target selection, device probing, execution, automatic
default-path integration, TargetIR, MLIR/LLVM/QIR, public APIs, and performance
claims remain outside the accepted profile.
