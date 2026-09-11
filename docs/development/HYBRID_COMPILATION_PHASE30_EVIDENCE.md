# Phase 30 strict target-text-conformance evidence

Date: 2026-09-10

## Accepted profile

Compiler reproduces the expected Phase 29 result from the supplied target
legalization and requires exact equality before independently parsing emitted
text. OpenQASM 2, OpenQASM 3, and QCIS 1 each have a strict parser for only the
canonical subset produced by the existing emitters.

Parsing reconstructs the existing Core `CircuitIR`. Terminal full-register
samples are retained with unspecified shots because the target text does not
contain execution shot count. Compiler does not execute the reconstructed
program. Tests use the Simulation-owned statevector path as a bounded semantic
oracle.

## Verification summary

| Gate | Result |
| --- | --- |
| OpenQASM 2 strict reconstruction and statevector equivalence | pass |
| OpenQASM 3 strict reconstruction and statevector equivalence | pass |
| QCIS reconstruction and global-phase-insensitive equivalence | pass |
| Payload text and digest tampering fail closed | pass |
| Target snapshot and emission identity tampering fail closed | pass |
| Emission paired with another legalization fails closed | pass |
| Reconstruction and conformance identities are deterministic | pass |
| Phase 30 direct tests | 8 passed |
| Hybrid compiler and private-contract focused suite | 214 passed |

## Claim boundary

Statevector comparison is bounded test evidence, not production Compiler work
or hardware certification. No second IR, executable-artifact envelope, Runtime
adapter, deployment package, provider submission, TargetIR, public API,
default-path change, or performance claim is made.
