# FlagQuantum IR Phase 0 Performance Baseline and Phase 1 Budget Candidates

Status: Phase 0 local CPU baseline measured; Phase 1 internal budget approved by the API owner.
Benchmark: `benchmarks/internal/ir_phase0_baseline.py`
Machine results: `tests/fixtures/internal_ir/phase0_performance_baseline.json`
Active regression budget: `tests/fixtures/internal_ir/phase1_performance_budget.json`

## 1. Measurement Scope

- Docker CPU profile.
- Linux aarch64, Python 3.12.13, Torch 2.13.0+cpu.
- 4 wires, complex64.
- 10, 100, 1K, and 10K gates.
- 3 warmups and 15 measurements.
- Explicit Python garbage collection before timing; GC disabled during measurement.
- Single-threaded Torch.
- Local development baseline only, not GPU, distributed, or QPU performance.

An initial trial observed one-time import overhead contaminating the 10-gate plan
measurement, so it was not retained as the baseline. Final results use explicit
warmup and are reproducible with the standalone script.

## 2. Key Results

| Gates | Build-to-IR p95 | JSON p95 | simple compile p95 | plan p95 | run p95 | plan peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.043 ms | 0.067 ms | 0.034 ms | 0.190 ms | 1.229 ms | 11,511 B |
| 100 | 0.327 ms | 0.189 ms | 0.163 ms | 0.850 ms | 7.257 ms | 70,456 B |
| 1,000 | 3.284 ms | 1.846 ms | 1.473 ms | 6.876 ms | 64.468 ms | 628,493 B |
| 10,000 | 36.265 ms | 32.251 ms | 17.378 ms | 67.788 ms | Not run | 6,288,342 B |

The 10K execution was not run, to avoid conflating long-circuit execution
throughput with the IR compiler baseline.

## 3. Phase 1 Budget Candidates

The candidate importer-plus-verifier p95 latency uses:

```text
max(1.0 ms, 1.75 × legacy plan p95)
```

Round upward to reviewable limits. The peak-memory candidate uses:

```text
max(1 MiB, 2 × legacy plan peak memory)
```

| Gates | Import+verify p95 limit | Peak memory limit |
| ---: | ---: | ---: |
| 10 | 1.0 ms | 1 MiB |
| 100 | 1.5 ms | 1 MiB |
| 1,000 | 12.5 ms | 1.5 MiB |
| 10,000 | 120 ms | 12 MiB |

These budgets allow validation and typed-node construction for the initial Python
internal IR while bounding general compiler overhead. Default legacy
`fq.plan/fq.run` must not import `_compiler` or perform work on the new path.

## 4. Usage Rules

- Current budget status: `active_private_regression_budget`.
- Budgets cover internal import+verify regression, not a public SLA.
- Tests compare implementation results; they cannot raise budgets automatically.
- Preserve old and new results when environments change; do not overwrite the raw baseline.
- Exceeding a budget requires optimization, reduced scope, or a recorded blocker.
- One fast result does not prove performance; CPU results do not establish scalability.
- Add benchmark output-schema and budget-gate tests before Phase 1 implementation.

## 5. P0-005/P0-006 Acceptance

- [x] Reproducible benchmark script.
- [x] 10/100/1K/10K gate baselines.
- [x] Complete warmup, iteration, environment, dtype, and thread-count records.
- [x] Timing, serialization size, and peak host memory recorded.
- [x] Candidate budgets derived from measurements.
- [x] Tests protect budget derivation.
- [x] API owner approved the Phase 1 internal performance budget.
- [ ] Phase 1 implementation passes the budget gate in the same environment.
