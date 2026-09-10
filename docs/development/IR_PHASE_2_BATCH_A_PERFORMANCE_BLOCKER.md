# FlagQuantum IR Phase 2 Batch A Performance Blocker

Status: **Blocked — Batch A exit and Batch B entry are not allowed**
Date: 2026-09-02
Blocker: `IR2A-PERF-001`

## 1. Conclusion

Batch A authorization, four canonicalization categories, linear value reconnection,
structural/state/expectation differentials, and two-parameter trainable-tensor
gradient differentials pass. However, the approved Phase 2 full-pipeline budget
cannot be met.

The budget remains unchanged. Batch A must not be declared complete.

## 2. Machine Facts

Minimum valid 1K-gate sampling in the Docker CPU environment:

| Metric | Measured | Approved limit | Result |
| --- | ---: | ---: | --- |
| Full-pipeline p95 | 67.629 ms | 15.0 ms | Fail |
| Peak host memory | 2,726,417 bytes | 2,097,152 bytes | Fail |
| identity determinism | true | true | Pass |

Cold-start stage diagnostics:

| Stage | Single 1K-gate duration |
| --- | ---: |
| seal/import/verify | 17.971 ms |
| Batch A pass pipeline | 52.180 ms |
| optimized lowering/verify | 16.778 ms |

Two consecutive full-gate runs failed to produce acceptable complete 10K evidence,
so the same command was not repeated further. The 1K case already exceeds both
time and memory limits, which is sufficient to trigger a fail-closed blocker.

## 3. Root Cause

Budget derivation used approximately `2 × legacy_plan p95`, but the comparison covers:

```text
CircuitIR import + verify
  + Phase 2 passes
  + restricted lowering
  + lowered-result verification
```

The approved Phase 1 1K import/verify limit is already 12.5 ms. A 15 ms Phase 2
total therefore leaves only 2.5 ms for passes, lowering, and repeated verification.
Minor implementation tuning cannot reliably meet that boundary.

The first implementation also revealed and fixed quadratic scans of the output
list during cancellation/rotation handling, replacing last-touch queries with
per-wire history stacks. The optimized 1K result remains 67.629 ms. The remaining
problem primarily concerns budget scope and repeated identity/verification/
serialization costs, rather than one more local loop.

## 4. Passing Nonperformance Evidence

- Exact authorization records bind the original review candidate.
- Identity, zero rotation, self-inverse, adjacent rotation, and disjoint-wire
  behavior match legacy behavior.
- Each independent pass is idempotent.
- Trainable rotation merging preserves forward results and both original gradients.
- complex64/complex128 state and expectation parity.
- Malformed multi-block input and missing source provenance fail closed.
- The default path does not use `_compiler`.
- Original budget files remain immutable.

This evidence does not cover or replace the performance exit gate.

## 5. Proposed Remediation

The following work requires separate authorization:

1. Measure import, identity, each pass, verification, lowering, and serialization separately.
2. Cache immutable module identities to avoid repeated JSON/hash work for one revision.
3. Replace full verification after every by-construction pass with full entry/exit
   verification, retaining independent verifier tests for each pass.
4. Evaluate equivalence of a fused production pipeline and separate pass descriptors.
5. Collect stable 10/100/1K/10K results.
6. Propose a successor budget from component costs for separate owner approval.
7. Preserve Phase 0, Phase 1, and original Phase 2 candidate budget snapshots.

## 6. Pause Boundaries

Until remediation and a successor budget are approved and pass:

- Do not enter Batch B.
- Do not connect default compiler/runtime/deployment paths.
- Do not change the original budget to make gates pass.
- Do not claim Phase 2 performance or Batch A completion.
- Do not remove the legacy compiler.

Proposed exact authorization command:

```text
approve IR-PHASE2-BATCH-A-PERFORMANCE-REMEDIATION
```

This authorizes only performance diagnosis, internal optimization, and a successor
budget candidate. It does not approve the budget, Batch A exit, or Batch B.
