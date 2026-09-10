# FlagQuantum IR Phase 2 Batch A Completion Review

Status: **Technically complete — awaiting explicit Batch A exit approval**

Date: 2026-09-02

## 1. Technical Conclusion

Phase 2 Batch A private static canonicalization has completed implementation,
semantic and gradient differential tests, determinism verification, performance
remediation, and validation against approved budgets. This evidence supports
Batch A exit review. This document itself approves neither the exit nor Batch B.

## 2. Completed Scope

- Identity and numerical zero-rotation cleanup.
- Adjacent self-inverse gate cancellation.
- Adjacent rotation merging, preserving trainable-tensor gradients.
- Linear qubit value reconnection and revision/identity contracts.
- Exact equivalence of the fused private production pipeline and separate passes.
- Fail-closed entry/exit verification.
- Caching of immutable identities, pipeline digests, and schema registries.
- Differential checks across CircuitIR import, private passes, and restricted lowering.
- Successor performance-budget gates for 10/100/1K/10K gates.

## 3. Approved Performance Gates

| Gates | p95 | Limit | Peak bytes | Limit | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 0.629 ms | 1.25 ms | 44,480 | 1,310,720 | Pass |
| 100 | 4.470 ms | 7 ms | 257,326 | 1,310,720 | Pass |
| 1,000 | 38.603 ms | 65 ms | 2,505,077 | 3,145,728 | Pass |
| 10,000 | 431.662 ms | 650 ms | 25,075,043 | 31,457,280 | Pass |

Identity determinism and growth-rate checks pass at all four sizes. These are
internal regression budgets, not public SLAs.

## 4. Preserved Boundaries

- Stable Core, CircuitIR 1.0, and public exports are unchanged.
- `fq.run`, `fq.plan`, and default compiler/runtime paths do not use the new IR.
- The legacy compiler has not been removed.
- Routing, emitters, provider codegen, TargetIR, and ProgramIR have not started.
- Original Phase 2 budget snapshots are unchanged.
- Batch B is not yet authorized.

## 5. Next Decision Gate

Batch A exit and authorization to start only Batch B require separate approval
through the next review candidate binding all evidence. Until then, technical
completion must not be described as formal phase exit.
