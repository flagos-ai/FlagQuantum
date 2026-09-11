# Phase 29 verified deterministic target-text-emission evidence

Date: 2026-09-10

## Accepted profile

Compiler accepts only a completed `TargetLegalizationResult` and invokes one of
the existing OpenQASM 2, OpenQASM 3, or QCIS deterministic text emitters. A
closed profile table validates the exact format version and legalized backend.
The result binds payload, final circuit, target snapshot, legalization, and
logical schedule identities.

Before emission, logical scheduling is recomputed and compared with stored
evidence. The lossless static profile requires one terminal full-register
samples request and rejects dynamic control, channels, arbitrary matrices,
observable expectations, partial results, and unbound parameters.

## Verification summary

| Gate | Result |
| --- | --- |
| OpenQASM 2 and 3 use the existing versioned emitter | pass |
| QCIS uses the existing native-text emitter | pass |
| Repeated payload and audit identities are deterministic | pass |
| Profile version changes payload and emission identity | pass |
| Backend/profile mismatch and unknown profile fail closed | pass |
| Unsupported result semantics fail before emission | pass |
| Unbound parameters fail before a result is returned | pass |
| Post-legalization mutation invalidates schedule evidence | pass |
| Phase 29 direct tests | 6 passed |
| Hybrid compiler and private-contract focused suite | 205 passed |

## Claim boundary

The result is an in-process Compiler record, not a new artifact envelope,
serialized cross-layer schema, signature, deployment package, or trust boundary.
Core `ProgramArtifact` v1 is unchanged. No Runtime adapter, provider submission,
credentials, TargetIR, public API, default-path change, or performance claim is
made.
