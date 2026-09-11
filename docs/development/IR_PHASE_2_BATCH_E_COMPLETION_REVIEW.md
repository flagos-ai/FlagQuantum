# IR Phase 2 Batch E completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- Deterministic compilation identity covering source, input QuantumIR, pass pipeline,
  target, topology, calibration, compile options, and identity-schema version.
- Thread-safe, entry-bounded in-memory LRU cache with exact canonical-payload collision
  checks and concurrent single-flight execution.
- Explicit hit, miss, bypass, invalidation, eviction, collision, computation, failure, and
  wait observability.
- Fail-closed exclusion of incomplete identities, unresolved trainable runtime bindings,
  and non-deterministic options.
- Failed pipelines and raised exceptions are never cached; structured diagnostics retain
  stage, pass name, source identity, error type, and causal message.
- Approved machine gate for cold miss, cache hit, identity bypass, minimum hit speedup,
  peak memory, and deterministic identity.

## Approved performance evidence

The formal five-iteration, two-warmup validation passed all approved thresholds.

| Gates | Cold miss p95 / budget | Hit p95 / budget | Bypass p95 / budget | Hit speedup / minimum | Memory / budget |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.192 / 0.5 ms | 0.017 / 0.05 ms | 0.142 / 0.4 ms | 11.35× / 4× | 22,669 / 65,536 B |
| 100 | 1.046 / 2 ms | 0.066 / 0.15 ms | 0.942 / 2 ms | 15.88× / 5× | 89,319 / 262,144 B |
| 1,000 | 9.416 / 20 ms | 0.451 / 1.5 ms | 8.989 / 20 ms | 20.89× / 5× | 576,484 / 1,048,576 B |
| 10,000 | 110.056 / 200 ms | 4.375 / 10 ms | 147.157 / 200 ms | 25.16× / 8× | 4,438,231 / 8,388,608 B |

These are private CPU regression limits, not public SLAs. Cache hits still execute the
runtime-binding safety scan and exact identity construction.

## Remaining support boundary

The cache remains private, process-local, in-memory, and bounded by entry count. It is not
connected to the default compiler or public API. It does not persist artifacts, coordinate
across ranks or processes, cache unresolved trainable bindings, select a cloud backend, or
submit jobs.

## Batch F proposed boundary

Batch F should be the final Phase 2 evidence batch, limited to offline private validation:

- a versioned corpus representing existing legacy deployment cases and Quafu static
  compilation cases already expressible without credentials or backend IDs;
- an explicit private end-to-end pipeline from imported static `CircuitIR` through the
  approved canonicalization, target decomposition, routing, compilation identity/cache,
  and OpenQASM/QCIS emitters;
- legacy/new structural comparison where formats are directly comparable, plus parser,
  state, expectation, gradient, wire-order, and emitted-text semantic comparison where
  decomposition changes structure;
- strict failure fixtures for dynamic circuits, unresolved trainable bindings, unsupported
  operations, invalid topology/calibration identity, and unsupported deployment features;
- full-pipeline deterministic identity, output-hash, cache behavior, and independent
  performance evidence;
- proof that `fq.run`, `fq.plan`, `compile_for_backend`, `DeploymentPackage`, public
  exports, and legacy behavior remain unchanged;
- a Phase 2 exit review and migration recommendation after all Batch F gates pass.

Batch F must not contain adapter repository source, provider SDK calls, remote submission,
credentials, tokens, queue/job handling, prices, or real backend IDs. It must not change
public/default paths, retire the legacy compiler, or declare Phase 2 complete without a
separate owner exit approval.

## Requested decision

Approve Batch E exit and only the private Batch F offline deployment/Quafu-static corpus,
end-to-end differential validation, and independent performance baseline with:

`approve IR-PHASE2-BATCH-E-EXIT-BATCH-F`
