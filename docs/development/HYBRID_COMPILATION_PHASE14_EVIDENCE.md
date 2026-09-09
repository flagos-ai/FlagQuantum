# Phase 14 measurement-dependent SSA merge evidence

Date: 2026-09-09

## Accepted profile

The private dynamic lowerer now carries scalar, index, and bool results out of
measurement-dependent branches as bounded condition-partitioned values.
Downstream classical operations preserve those partitions, and quantum
consumers lower to mutually exclusive conditioned Core instructions.

## Evidence

| Gate | Result |
| --- | --- |
| Capture emits explicit scalar, index, bool, and effect branch results | pass |
| Dynamic lowering merges both classical branch result tuples | pass |
| Post-branch scalar addition remains partitioned by the measured bit | pass |
| A conditional scalar produces the matching RX parameter per branch | pass |
| A conditional index selects wires 1 and 2 under opposite predicates | pass |
| A carried bool controls a later X gate | pass |
| Reference and batched trajectories agree shot by shot | pass |
| Equal primitive or shared-value cases can be coalesced | pass |
| Conditional gates retain canonical Phase 13 metadata | pass |
| Existing compile-time branch carrying remains unchanged | pass |

## Claim boundary

This evidence covers bounded scalar, index, and bool SSA values consumed by
classical expressions, quantum gates, later branches, and statically bounded
loops. It does not cover conditional measurement, measurement-dependent loop
bounds, returning arbitrary dynamic classical values, finite-shot gradients,
provider execution, accelerators, distributed execution, capacity, or
performance.
