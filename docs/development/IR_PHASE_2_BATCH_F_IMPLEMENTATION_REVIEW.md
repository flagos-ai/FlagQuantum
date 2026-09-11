# IR Phase 2 Batch F Implementation Review

## Conclusion

Batch F provides an end-to-end static compilation path within private, offline,
explicit invocation boundaries:

```text
CircuitIR import/seal
  -> static canonicalization
  -> RX/RY/RZ/CX target decomposition
  -> directed placement/routing
  -> post-routing canonicalization
  -> deterministic identity/cache
  -> OpenQASM 2 / OpenQASM 3 / QCIS text emission
```

It does not discover cloud resources, accept real backend IDs, read credentials,
call provider SDKs, submit jobs, or enter public APIs/default execution paths.

## Established Evidence

- A versioned offline corpus covers legacy-native cases, synthetic Quafu-static
  linear routing, and directed-edge reversal.
- Tests fix output gate structure, directed coupling legality, and output hashes.
- OpenQASM 2, OpenQASM 3, and QCIS outputs are parsed back to verify states and qubit order.
- Parsed semantic differentials compare against legacy QASM/QCIS emission.
- The mode without text emission verifies expectations and autograd gradients.
- Machine evidence covers compilation identity, calibration snapshot changes,
  cache misses/hits, and deterministic text output.
- Measurements, observables, dynamic circuits, unbound parameters, noise channels,
  and qubit-count mismatches fail closed to prevent silent loss of deployment semantics.

## Performance Baseline

Under `flagquantum-dev:pr-check`, single-threaded CPU execution, and 5 samples,
the complete 10K-gate path has cold/cached p95 of 574.794476 ms / 440.719608 ms
and cold peak host memory of 26,196,356 bytes. This is a private engineering
baseline, not a public SLA.

A candidate performance budget exists but awaits owner approval and is not yet a
regression gate.

## Boundaries That Remain Closed

- Adapter source, provider SDKs, and remote submission.
- Tokens, credentials, real backend IDs, queues, jobs, and pricing.
- Public APIs, default-path switches, and legacy retirement.
- Automatic Phase 2 exit.

## Next Authorization Point

If the baseline and candidate thresholds are accepted, the exact approval command is:

`approve IR-PHASE2-BATCH-F-PERFORMANCE-BUDGET`

Only after approval may the candidate become an active gate and preparation of
the Phase 2 exit review candidate continue.
