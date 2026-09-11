# IR Phase 2 Batch C implementation review

Status: private placement/routing implementation and baseline complete; performance-budget approval pending.

## Delivered

- Added a deterministic directed coupling graph with canonical edge ordering and lexicographically stable shortest paths.
- Added a private, opt-in placement/routing pass for the Batch B `rx`/`ry`/`rz`/`cx` target profile.
- Added explicit logical-to-physical initial layouts and deterministic final restoration to identity layout.
- Added persistent-layout SWAP routing, expressed entirely in the target gate set.
- Added direction-aware CX synthesis. A reverse-only physical edge preserves control/target semantics rather than silently exchanging operands.
- Added fail-closed diagnostics for invalid layouts, unsupported operations, disconnected topology, and physical/logical capacity mismatch.
- Exposed deterministic pass statistics for initial/final layouts, directed edges, physical SWAP count, reversed-CX count, and emitted operations.

## Correctness evidence

- State differential checks cover non-adjacent routing, reverse-only edges, and explicit non-identity placement.
- Trainable parameter expectation values and gradients match the source circuit.
- All 36 combinations of three-qubit initial layouts and ordered CX operands pass against an asymmetric directed line.
- Re-routing an already routed module preserves unique SSA value identities.
- The complete internal-IR suite passes, including the isolated historical timing gate.
- Stable public APIs and the default execution path are unchanged.

## Performance baseline

The baseline isolates Batch C rather than including upstream canonicalization. On an 8-qubit bidirectional line with a long-range CX-heavy workload:

| Source gates | Emitted gates | Physical SWAPs | p95 | Peak traced memory |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 142 | 44 | 3.025 ms | 321,484 B |
| 100 | 652 | 184 | 17.568 ms | 1,382,292 B |
| 1,000 | 5,872 | 1,624 | 381.174 ms | 12,551,456 B |
| 10,000 | 58,072 | 16,024 | 2,212.930 ms | 123,544,108 B |

The measurements are an internal CPU regression baseline, not a public SLA. A candidate budget with at least 25% measured headroom is prepared but remains unapproved.

## Current capability boundary

Batch C v1 intentionally requires one static block and the same number of logical and physical qubits. It does not route through unused ancilla qubits, optimize for calibration/error rates, persist a provider payload, emit provider code, submit remote jobs, or alter the public/default execution path.

Supporting extra physical ancillas requires a later IR decision about allocation, lifetime, output layout, and observable remapping; it must not be hidden inside this pass.

## Requested decision

Approve the internal performance budget and machine gate with:

`approve IR-PHASE2-BATCH-C-PERFORMANCE-BUDGET`

This approval would not authorize Batch C exit, Batch D, emitter/provider work, public API changes, default-path changes, or legacy retirement.
