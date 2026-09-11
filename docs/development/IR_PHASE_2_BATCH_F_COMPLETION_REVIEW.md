# IR Phase 2 Batch F completion review

Status: implementation and approved performance gate complete; Phase 2 owner exit approval pending.

## Delivered

- A private provider-free transaction from sealed `CircuitIR` through canonicalization,
  target decomposition, directed placement/routing, compilation identity/cache, and
  OpenQASM 2/OpenQASM 3/QCIS emission.
- A versioned offline corpus for legacy-native and synthetic Quafu-static compilation;
  it contains no provider SDK source, credentials, remote endpoint, or real backend ID.
- Structural, parser, state, expectation, gradient, qubit-order, legacy-text semantic,
  content-hash, compilation-identity, and cache hit/miss evidence.
- Fail-closed rejection for measurements, observables, dynamic circuits, unresolved
  parameters, noise channels, and logical/physical qubit-count mismatches.
- An approved, executable private CPU regression gate covering the full pipeline.

## Approved performance evidence

The five-iteration, two-warmup validation passed every approved threshold.

| Gates | Cold p95 / budget | Cached p95 / budget | Peak memory / budget |
| ---: | ---: | ---: | ---: |
| 10 | 0.930 / 2 ms | 0.446 / 1 ms | 55,466 / 1,048,576 B |
| 100 | 4.414 / 10 ms | 2.601 / 6 ms | 264,350 / 1,048,576 B |
| 1,000 | 54.994 / 100 ms | 43.203 / 80 ms | 2,626,298 / 8,388,608 B |
| 10,000 | 569.190 / 1,000 ms | 356.807 / 800 ms | 26,196,356 / 67,108,864 B |

Identity and emitted text size were deterministic at every size. These limits are private
CPU regression budgets, not public SLAs.

One additional five-sample run observed a transient 1,000-gate cold p95 of 105.464 ms
against the 100 ms limit; an unchanged-method bounded rerun measured 62.664 ms. The
approved threshold was not relaxed. CI flake rate should be monitored before making any
public performance claim.

## Phase 2 aggregate result

Phase 2 now has technical evidence for all six planned private batches:

1. static canonicalization;
2. deterministic lowering to the RX/RY/RZ/CX target profile;
3. directed placement and routing with logical output order restored;
4. restricted deterministic OpenQASM 2, OpenQASM 3, and QCIS emitters;
5. complete compilation identity and bounded in-memory pipeline cache;
6. provider-free end-to-end compilation and differential validation.

The implementation remains private under `flagquantum._compiler`. Existing `fq.run`,
`fq.plan`, `compile_for_backend`, `DeploymentPackage`, public exports, and default runtime
selection are unchanged.

## Remaining boundary after technical completion

Phase 2 completion does not mean production migration is complete. The following remain
outside the accepted scope:

- public IR/pass/compiler APIs;
- default compiler or deployment path switching;
- legacy compiler retirement;
- provider discovery, live calibration ingestion, credentials, real backend IDs, remote
  submission, queue/job lifecycle, and result retrieval;
- dynamic-circuit, measurement/classical-register, timing, pulse, noise-channel, or
  unrestricted OpenQASM 3 lowering;
- persistent/distributed compilation cache and auxiliary-qubit allocation.
- one five-sample local CPU run recorded a host-scheduler jitter excursion.

## Migration recommendation

After Phase 2 exit, prepare a separately reviewed integration phase. It should start with
shadow-mode comparison behind explicit internal configuration, then canary one local
deployment path, and only later propose a public/default migration. Each step needs its own
rollback, compatibility, performance, and provider-contract evidence.

## Requested decision

Accept the private static compiler evidence and mark only IR Phase 2 technically complete:

`approve IR-PHASE2-EXIT`

This approval must not authorize public APIs, default-path switching, legacy retirement,
provider/adapter implementation, remote submission, or the next integration phase.
