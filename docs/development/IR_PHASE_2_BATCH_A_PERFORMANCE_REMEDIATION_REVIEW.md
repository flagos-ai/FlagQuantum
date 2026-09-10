# FlagQuantum IR Phase 2 Batch A Performance Remediation Review

Status: **Awaiting successor-budget approval — Batch A exit and Batch B entry are not allowed**

Date: 2026-09-02

Original blocker: `IR2A-PERF-001`

## 1. Conclusion

Authorized implementation fixes, equivalence proofs, and 10/100/1K/10K
measurements are complete. Full-pipeline 1K p95 fell from 67.629 ms to 35.769 ms
(a 47.11% reduction). Peak memory fell from 2,726,417 bytes to 2,505,077 bytes
(an 8.12% reduction).

The fixes help, but the original 15 ms / 2 MiB budget for 1K still fails. Therefore:

- Original budget files remain unchanged.
- The original blocker remains active.
- The successor budget is only a candidate and is not approved.
- Batch A exit, Batch B entry, default-path integration, and legacy compiler
  removal remain prohibited.

## 2. Fixes

1. Cache program identities for immutable `QuantumModule` and import artifacts.
2. Cache pass pipeline digests.
3. Share an immutable opcode registry and replace linear opcode lookup with O(1) lookup.
4. Use a fused private pipeline in the production benchmark, with full verification
   only at entry and exit.
5. Retain per-pass verification for independent passes and tests.
6. Prove exact equivalence between fused and separate pipelines for modules,
   canonical encoding, program identities, and lowering results.
7. Add import/pass/lower/identity stage measurements without changing full-pipeline
   timing boundaries.

All changes are private to `_compiler`. Public APIs, default compiler/runtime
paths, and failure semantics remain unchanged.

## 3. Measurements

Environment: `flagquantum-dev:pr-check`, Python 3.12.13, Torch 2.13.0+cpu,
single-threaded; 3 warmups and 7 measurements per size.

| Gates | Full p95 | Instrumented stage total p95 | Peak bytes | Original budget result |
| ---: | ---: | ---: | ---: | --- |
| 10 | 0.554 ms | 0.493 ms | 44,480 | Pass |
| 100 | 5.366 ms | 3.893 ms | 257,326 | Latency fails |
| 1,000 | 35.769 ms | 50.915 ms | 2,505,077 | Latency and memory fail |
| 10,000 | 434.161 ms | 517.262 ms | 25,075,043 | Latency and memory fail |

Stage measurements include Python instrumentation overhead and serve diagnosis
and budget robustness analysis only. Full-pipeline results establish end-to-end
facts. Output identities and pipeline digests remain deterministic at every size.

## 4. Successor Budget Candidates

Candidate derivation:

```text
latency = max(original limit, 1.25 × max(full p95, instrumented total p95)), rounded up to reviewable limits
memory  = max(original limit, 1.25 × tracemalloc peak), rounded up to binary limits
```

| Gates | latency candidate | memory candidate |
| ---: | ---: | ---: |
| 10 | 1.25 ms | 1.25 MiB |
| 100 | 7 ms | 1.25 MiB |
| 1,000 | 65 ms | 3 MiB |
| 10,000 | 650 ms | 30 MiB |

These are internal regression budgets, not public SLAs or an end to optimization.

## 5. Decision Requiring Separate Approval

If the owner accepts the measurement scope and candidates, the exact next
authorization command is:

```text
approve IR-PHASE2-BATCH-A-SUCCESSOR-BUDGET
```

This permits freezing and enabling the successor budget gate only. It does not
automatically authorize Batch A exit, Batch B, default-path changes, public API
changes, or legacy retirement.
