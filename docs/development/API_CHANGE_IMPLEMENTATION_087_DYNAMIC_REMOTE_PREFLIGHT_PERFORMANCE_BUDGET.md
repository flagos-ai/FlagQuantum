# API Change Implementation 087: Dynamic remote preflight performance budget

## Status

**Performance budget authorized and measured on 2026-10-24.**

The API owner authorized this work with:

```text
approve API-CHANGE-087-DYNAMIC-REMOTE-PREFLIGHT-PERFORMANCE-BUDGET
```

This stage measures the fail-closed local decision installed by API Change 087.
It does not acquire credentials, import a provider SDK, access a network, submit
a task, or make a remote-QPU performance claim.

## User workflow

Run the maintained benchmark through the public benchmark CLI:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run dynamic_remote_preflight \
  --feedback-pairs 1 16 128 --warmup 5 --iterations 101 \
  --static-iterations 1001 --json-output \
  benchmarks/results/comparison/dynamic_remote_preflight_cpu_arm64_20261024.json
```

The command exits successfully only when every measured case passes its budget.
The JSON retains raw samples, p50, p95, relative median absolute deviation,
requirement and snapshot identities, ordered blockers, explicit no-remote-activity
metadata, budget inputs, and the derived verdict.

## Measurement design

The runner constructs each circuit before timing and evaluates one fixed target
at one fixed timezone-aware instant. This keeps the requirement and missing-fact
snapshot identities reproducible. The cases contain 1, 16, and 128
measurement/feedback pairs, or 2, 32, and 256 instructions.

The static sentinel measures the exact lazy-import plus `isinstance` classifier
shape used to bypass DynamicCircuit preflight. It deliberately does not call a
static provider route, because doing so would mix provider, compiler, credential,
and network behavior into the classifier budget.

## Budget

An initial fixed 50 ms candidate did not describe the measured range: repeated
256-instruction observations had 23–29 ms medians but 93–119 ms p95 tail latency,
while 32 instructions remained at 17–20 ms p95. Raising one constant would hide
small-program regressions, so the accepted budget is a linear envelope:

```text
preflight p95 limit = 10 ms + instruction_count × 0.55 ms
static classifier p95 limit = 1 ms
```

This yields preflight limits of 11.1, 27.6, and 150.8 ms. The budget is generous
to scheduler noise but still fails large fixed overhead and superlinear growth.
CI checks the payload contract and arithmetic with short runs; it does not turn
shared-runner timing into a flaky required check.

## Retained observation

The retained macOS arm64 observation used Python 3.12.14, PyTorch 2.13.0,
FlagQuantum 0.4.0, one process, and one PyTorch thread.

| Instructions | p50 | p95 | p95 budget | Result |
| ---: | ---: | ---: | ---: | :---: |
| 2 | 0.198 ms | 0.423 ms | 11.1 ms | pass |
| 32 | 0.970 ms | 1.640 ms | 27.6 ms | pass |
| 256 | 6.960 ms | 25.207 ms | 150.8 ms | pass |

Static classification measured 0.209 microseconds p50 and 0.250 microseconds
p95 against the 1 ms limit. All observations characterize this machine and run
only; the versioned runner and budget logic are the durable regression mechanism.

## Compatibility boundary

- No stable API, root export, serialized schema, capability fact, or provider
  behavior changes.
- The benchmark imports an internal evaluator but does not expose it publicly.
- No positive remote DynamicCircuit path is admitted.
- The retained artifact is comparison-only, non-release local evidence.
- Provider-owned snapshot acquisition, post-routing rematch, task semantics,
  live evidence, and provider-slice promotion remain future gates.

## Verification

- benchmark and CLI contract tests;
- registry discovery and lazy resolution;
- fail-closed blockers and reproducible identities;
- derived budget verdicts rather than hard-coded pass values;
- invalid count and iteration refusal;
- Black, Ruff, strict mypy, documentation, and team-scope checks; and
- the full benchmark-contract tier: 846 passed and 79 skipped.
