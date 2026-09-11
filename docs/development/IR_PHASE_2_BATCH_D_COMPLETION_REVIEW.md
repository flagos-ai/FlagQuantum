# IR Phase 2 Batch D completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- Private deterministic OpenQASM 2 static emitter.
- Private restricted-static OpenQASM 3 emitter.
- Private versioned QCIS emitter using native `X2P/X2M/Y2P/Y2M/RZ/CZ` instructions.
- Immutable result contract binding profile, source program identity, exact text, and
  content SHA-256.
- Golden, parse, semantic, asymmetric-qubit-order, deterministic-format, malformed-input,
  symbolic/runtime-binding rejection, and default-path-zero-impact evidence.
- Approved and fail-closed CPU regression gate at 10, 100, 1,000, and 10,000 target gates.

## Approved performance evidence

The formal five-iteration, two-warmup validation passed every approved latency, memory,
and deterministic-content threshold.

| Gates | OpenQASM 2 p95 / budget | OpenQASM 3 p95 / budget | QCIS p95 / budget |
| ---: | ---: | ---: | ---: |
| 10 | 0.120 / 0.2 ms | 0.081 / 0.2 ms | 0.074 / 0.2 ms |
| 100 | 0.544 / 0.75 ms | 0.371 / 0.75 ms | 0.388 / 1.25 ms |
| 1,000 | 3.559 / 20 ms | 4.275 / 10 ms | 3.574 / 7 ms |
| 10,000 | 43.445 / 125 ms | 35.442 / 150 ms | 55.424 / 100 ms |

At 10,000 gates, peak traced memory was 3,316,309 bytes for OpenQASM 2, 3,323,817
bytes for OpenQASM 3, and 4,236,926 bytes for QCIS, against approved limits of
5,242,880, 5,242,880, and 6,291,456 bytes respectively. These are private regression
budgets, not public SLAs.

## Remaining support boundary

Batch D emits formats only. It does not submit jobs, choose a provider backend, carry
credentials, query calibration, bind trainable parameters, or replace the existing public
deployment path. Static binding must happen before emission. Dynamic control flow,
measurement/classical registers, timing, pulse programs, and unrestricted OpenQASM 3
remain unsupported by this profile.

## Batch E proposed boundary

Batch E should add only private pipeline infrastructure:

- deterministic compilation identity over source identity, pipeline/pass descriptors,
  target profile, topology/calibration identity, and relevant compile options;
- bounded in-memory pipeline cache with exact hit/miss/invalidation semantics;
- fail-closed cache exclusion for runtime object identity, trainable bindings, unknown
  identity inputs, or non-deterministic options;
- structured diagnostics preserving pass name, stage, source identity, and causal errors;
- observable private counters for cache hit, miss, bypass, invalidation, and eviction;
- cache-key, mutation/isolation, collision, failure-path, concurrency, determinism,
  zero-default-path-impact, and separate performance evidence.

Batch E must not add provider SDK calls, remote jobs, credentials, backend IDs, public
cache/compiler APIs, default-path integration, persistent/on-disk cache, distributed
cache, legacy retirement, Batch F, or Phase 2 exit.

## Requested decision

Approve Batch D exit and only the private Batch E pipeline cache, diagnostics, and
compilation-identity work with:

`approve IR-PHASE2-BATCH-D-EXIT-BATCH-E`
