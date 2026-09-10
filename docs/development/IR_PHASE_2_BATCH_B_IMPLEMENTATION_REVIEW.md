# FlagQuantum IR Phase 2 Batch B Implementation Review

Status: **Implementation complete — awaiting performance-budget approval**

Date: 2026-09-02

## 1. Implementation Conclusion

Batch B establishes the private, versioned `universal_rx_ry_rz_cx` target profile.
After Batch A identity cleanup, all 30 static unitary opcodes lower deterministically
to `rx`, `ry`, `rz`, and `cx`. Channels, custom unitaries, and inputs failing
preconditions fail closed, without retaining illegal gates or falling back.

## 2. Contracts and Evidence

- Target gate sets and certified source sets appear in pass descriptors.
- Every rewrite explicitly reconnects linear qubit values.
- Intermediate ValueIds use a separate deterministic scope; final results reuse
  original operation result identities.
- Symbolic expressions scale parameters while preserving trainable bindings and autograd.
- Per-gate state differential tests cover one-, two-, and three-qubit gates,
  allowing and handling only global phase.
- Trainable forward/gradient differential tests cover phase/u1/u2/u3,
  controlled rotations/phases, and RXX/RYY/RZZ separately.
- Target legality, idempotency, program identity determinism, and unsupported
  fail-closed behavior are verified.
- Only explicit private PassManager calls use the implementation; default public paths remain unchanged.

## 3. Independent Performance Baseline

The rewrite-heavy workload has an emitted/source gate ratio of 6.0 at 10K gates:

| Source gates | Emitted gates | p95 | Peak bytes |
| ---: | ---: | ---: | ---: |
| 10 | 48 | 4.098 ms | 121,676 |
| 100 | 589 | 31.978 ms | 1,206,565 |
| 1,000 | 5,984 | 179.677 ms | 12,091,980 |
| 10,000 | 60,000 | 2,059.455 ms | 123,473,111 |

Candidate budgets use at least `1.25 × measured`, rounded upward to reviewable
limits. They are not yet approved, so Batch B does not meet exit conditions.

## 4. Explicitly Unauthorized

- Batch B exit and Batch C routing.
- QASM/QCIS emitters or provider codegen.
- Public API, default compiler/runtime, or deployment-path changes.
- Legacy compiler retirement.
- Public performance or hardware throughput claims based on this CPU baseline.

## 5. Next Gate

The next step is limited to approving Batch B's internal performance budget and
enabling its machine gate. This neither approves Batch B exit nor authorizes routing.
