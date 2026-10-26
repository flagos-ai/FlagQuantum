# Module derivative-demand execution: CPU development evidence

## Outcome and limits

The existing JAX executor now dispatches by Torch grad mode and trainable
arguments. Validation/inference computes values only; frozen-parameter training
can compute only input derivatives. No public signature, default, checkpoint
schema, device or precision choice changed. Module.eval() alone does not disable
autograd. No input-batch or automatic-selection capability is added here.

All ten shapes in each phase completed five derivative contexts through three
routes: 300 numerical route/context checks in total. Widths 2/4/6, depths 1/3
and chain/ring connections use deterministic affine feature binding, RY/RZ gates
and entangling CX. The redundant two-wire ring is omitted. Values and requested
VJPs agree with Torch execution to 1e-10 absolute/relative tolerance. Analytic
VJP tests separately check derivative-demand correctness, including changing
values, non-unit cotangents and grad-mode transitions.

## Controlled replacement

Candidate producer: `3b255c383e56f1cf08f1e692c5306e286c66ad85`, clean source.
The baseline uses that same harness and all other source, replacing only the
compile_quantum_kernel factory with the frozen kernel module from
`fb072a42fc931d478dd12e443cd00c28efa81c0f` in each fresh process. This is a
controlled executor replacement, **not a complete old-revision installation**.
The baseline source, worker and driver are retained as text beside the reports;
original/published hashes are in publication.json. No raw timing is rewritten.

[Baseline summary](baseline/summary.json), [candidate summary](candidate/summary.json).
Original per-case JSON and logs are retained in the local experiment archive.

## Timing

Five blocks of twenty completed calls per context, rotating route order, CPU,
complex128, Torch/BLAS configured to one thread. XLA thread count is not claimed
to be one. Each shape has its own process; the five blocks are within-process
samples, not independent process trials. First call in each context is separate
and reuses earlier contexts' caches; this is not a cold-import ranking. Extra
value-only/input-only variants can add a first compilation cost.

The table is the median across ten shapes of baseline/candidate median call
cost. Values above one indicate lower candidate time. This is a development
signal, not a statistically established speedup or a framework ranking.

| Derivative demand | Untouched default Module control | Public JAX Module | Internal JAX diagnostic |
| --- | ---: | ---: | ---: |
| parameters | 1.053x | 1.022x | 1.047x |
| inputs | 1.020x | 1.172x | 1.156x |
| both | 1.039x | 0.982x | 1.048x |
| no_grad | 1.005x | 1.492x | 1.650x |
| inference | 1.037x | 1.507x | 1.658x |

Untouched control timings drifted: parameter-only default ratios ranged from
0.900x to 2.466x across shapes. Public no_grad ratios ranged from 1.274x to
22.610x; the extreme ratio is not representative. The median public validation
and inference signal is about 1.5x, but repeated independent paired process
trials are still needed. Parameter-only/both-gradient timing does not establish
a reliable speedup. No optimizer/time-to-target or memory claim follows from
this call-level matrix, and no external framework is measured here.

## Remaining failures

Only **19/50** public-JAX overhead budgets passed; **0/50** default-accessibility
budgets passed. Across shapes, the median public/kernel ratio by context was
1.24-1.34. These are proposed development targets, not approved release gates:
public <= 1.15 * kernel + 5 us; default <= 1.25 * best measured public + 5 us.
The matrix does not close #918 or establish ordinary default competitiveness.

Run `python -m benchmarks.module_execution_matrix --output /tmp/fq-module-matrix`
from the candidate source. Add --enforce to fail unmet proposed budgets.
Timeout/worker/numerical failures fail regardless of that flag. Budget unit
checks prove invalid timings cannot pass and failed cases remain in the report.

See [the implementation plan](../../../docs/development/MODULE_TRAINING_EXECUTION_PLAN.md)
and #928/#929/#930 for canonical compiled execution, input batching and execution
selection. #925 remains the separate finite-angle validation gap. All evidence
is single_device_fast_path, non-release, and authorizes no scalability claim.

## Local verification

171 focused benchmark/runtime/API/snapshot/dependency-lane/JAX tests passed.
Analytic derivative-demand cases cover statevector, MPS and tensor-network
executors, with JIT enabled and disabled. Black, Ruff, strict changed-file mypy,
architecture, dependency policy, public API baseline, public evidence privacy and
repository hygiene checks passed. This is not a complete-suite or remote-CI
success claim. The PR remains draft; full-suite limitations recorded in the
previous evidence still apply.

Prior remote quality/dependency jobs exposed two Module Any-return errors; the
private kernel cache now has its concrete type and the strict three-file check
passes. The prior supply-chain job failed on a PyPI read timeout. New remote
checks must complete before merge. This annotation-only follow-up does not
change the measured numerical implementation.

Full local strict package mypy passed for all 840 selected source files.
