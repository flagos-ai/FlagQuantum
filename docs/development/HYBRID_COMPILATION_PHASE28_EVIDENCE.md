# Phase 28 dependency-preserving logical-scheduling evidence

Date: 2026-09-10

## Accepted profile

Compiler constructs immutable logical scheduling evidence over the final
Core-owned `CircuitIR`. Deterministic ASAP layers preserve per-wire source order
and record explicit measurement-producer dependencies for conditioned
operations on any wire. Dynamic operations, conditions, and channels are
conservative global barriers.

The schedule stores source instruction indices, depth, maximum parallel width,
dependency audit records, and a deterministic identity bound to final circuit
content and the selected target snapshot. An optional logical-depth limit is
checked after topology routing, native decomposition, backend lowering
validation, and target resource matching.

## Verification summary

| Gate | Result |
| --- | --- |
| Independent operations share an ASAP layer | pass |
| Same-wire operations retain source order | pass |
| Cross-wire measurement feedback retains causal order | pass |
| DNF conditions depend on all referenced producers | pass |
| Read-before-measurement and malformed conditions fail closed | pass |
| Maximum logical depth fails closed | pass |
| Identity binds final circuit and target snapshot | pass |
| Target legalization schedules the final transformed program | pass |
| Phase 28 direct and integration tests | 21 passed |
| Hybrid compiler and private-contract focused suite | 198 passed |

## Claim boundary

Logical layers have unit weight and are not hardware time. No gate duration,
measurement or feedback latency, concurrent-operation support, crosstalk,
fidelity, calibration, pulse scheduling, schedule execution, provider emission,
TargetIR, public API, default-path change, or performance claim is made.
