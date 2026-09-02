# IR Phase 2 Batch E implementation review

Status: private pipeline infrastructure complete; performance budget approval pending.

## Delivered

Batch E adds a private compilation identity and bounded in-memory pipeline cache without
changing `fq.run`, `fq.plan`, public exports, or the default compiler path.

The compilation identity includes:

- source identity and exact input `QuantumModule` identity;
- deterministic pass-pipeline digest;
- target-profile identity;
- topology and calibration identities;
- canonically encoded compile options;
- an explicit compilation-identity schema version.

Unknown identity fields, trainable runtime bindings, and non-deterministic compile options
cause an observable cache bypass. Compilation may still proceed, but unsafe input can
never create or hit a cache entry.

## Cache and diagnostic contract

The cache is a thread-safe, entry-bounded LRU with:

- exact canonical-payload comparison in addition to the SHA-256 lookup key;
- collision detection that bypasses rather than returning the wrong artifact;
- single-flight execution for concurrent identical requests;
- no caching of failed pass pipelines or raised exceptions;
- exact-entry invalidation, full clear, and deterministic LRU eviction;
- immutable snapshots for hits, misses, bypasses, invalidations, evictions, collisions,
  computations, failures, waits, size, and capacity.

Pipeline failures retain the source identity, stage, exact pass name, structured compiler
diagnostic, and root cause. Raised exceptions retain their type and message. Cached
pipeline results remain immutable and isolated from callers.

The bound is currently an entry-count limit rather than a byte-size admission policy.
Persistent, filesystem, distributed, and cross-process caching remain explicitly outside
Batch E.

## Correctness evidence

The focused suite covers:

- every compilation-identity input changing the digest independently;
- deterministic identity reconstruction;
- runtime-binding, missing-identity, and unordered-option bypass;
- hit, miss, bypass, exact invalidation, clear, and LRU eviction counters;
- forced digest collision without wrong-result reuse;
- failed result and exception non-caching with causal diagnostics;
- concurrent single-flight behavior;
- immutable cached results and zero public/default-path exposure.

## Performance baseline

Two five-iteration, two-warmup runs measured a one-pass private canonicalization pipeline.
The envelope retains the slower observation from either run.

| Gates | Cold miss p95 | Cache hit p95 | Bypass p95 | Minimum observed hit speedup | Cold peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.231 ms | 0.017 ms | 0.151 ms | 13.24× | 22,669 B |
| 100 | 1.031 ms | 0.057 ms | 0.938 ms | 17.94× | 89,319 B |
| 1,000 | 9.282 ms | 0.681 ms | 9.041 ms | 13.63× | 576,484 B |
| 10,000 | 108.522 ms | 4.374 ms | 115.314 ms | 24.81× | 4,438,231 B |

A cache hit still performs the runtime-binding safety scan and reconstructs the exact
compilation identity. The measured speedup therefore does not come from skipping identity
validation.

The proposed budget preserves at least 25% headroom over retained latency and memory
maxima and applies a conservative minimum hit-speedup threshold. It remains unapproved
and is not a public SLA.

## Explicitly not delivered

- Provider SDK integration, remote submission, credentials, or backend IDs;
- public compiler/cache APIs or default-path integration;
- persistent, on-disk, cross-process, or distributed cache;
- cache reuse for unresolved trainable runtime bindings;
- legacy compiler retirement, Batch F deployment migration, or Phase 2 exit.

## Requested decision

Approve only the proposed private Batch E performance budget and machine gate with:

`approve IR-PHASE2-BATCH-E-PERFORMANCE-BUDGET`

This approval must keep Batch E exit, Batch F, provider work, public/default-path changes,
persistent/distributed cache, legacy retirement, and Phase 2 exit closed.
