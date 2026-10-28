# Bound scalar rotation packing: public CPU Module development evidence

## Outcome

The ordinary PyTorch statevector route now packs compiled scalar angle slots
once per rotation region, preserving shared parameters and scalar broadcasting.
It avoids resolving and stacking each gate column separately before stacking
regions. No circuit width, depth, task name or data value is special-cased.
Constants, vectors, dtype/device conversions and unsupported slot layouts retain
the existing parameter path, including shape errors. No backend/default,
precision, public signature or checkpoint schema changes.

Candidate: `7cb725d727f9cfa0465152c1c94a8ececc79fd0a`, clean source in every
measurement. Baseline replaces only _rotation_region_angles with the frozen
function from `7dfe445e12984624e74e69315493540c662dff6e`, using the candidate
harness and all other source unchanged. This is controlled numerical-helper
replacement, not a full old-installation comparison. Frozen source and worker
are retained with original/published hashes.

## Repeated process measurements

Three fresh processes per shape and phase; ten width/depth/chain/ring shapes,
five derivative contexts and three routes. Phase order was candidate/baseline,
baseline/candidate, candidate/baseline. Each process rotates route order across
five blocks of twenty completed calls. All 60 processes completed, with 900
route/context numerical checks against Torch values/VJPs at 1e-10 tolerance.
First calls in each context are separate, reuse earlier contexts' caches and
are not cold-import rankings. CPU complex128; Torch/BLAS configured to one
thread; XLA thread count is not constrained to one.

For each shape/context/route, take the median of its three process medians per
phase; divide baseline by candidate. The table reports median and range of
those ratios across all ten shapes. A ratio above one means lower candidate
cost. The JAX Module is an untouched control. No case or failed sample is
removed from the aggregation.

| Demand | Default Module median speed ratio | Range across shapes | JAX control median ratio | Remaining default/JAX median cost |
| --- | ---: | ---: | ---: | ---: |
| parameters | 1.298x | 1.136-1.496x | 1.032x | 6.25x |
| inputs | 1.222x | 1.100-1.301x | 1.060x | 6.50x |
| both | 1.322x | 1.112-1.470x | 1.066x | 6.10x |
| no_grad | 1.253x | 1.149-1.387x | 1.057x | 6.45x |
| inference | 1.194x | 1.143-1.352x | 1.000x | 6.51x |

The aggregate ratios improve in all 50 default shape/context combinations;
this is a bounded CPU development result, not a universal performance claim.
The initial six-wire/depth-one/ring inference trial regressed from about 298 to
562 microseconds. Its other candidate trials were about 253 and 251 us; the
baseline trials were about 298/306/322 us. The three-trial medians yield a 1.207x
ratio, so the initial regression did not persist, but remains in the raw data.
Control drift remains visible. Three processes do not certify a reliable ranking
on arbitrary hardware, workloads or dependencies. No GPU, memory, time-to-target,
external-framework or scalability claim is made.

The default still costs about six times the public JAX route. There is no
claim that this closes #918, #928, #929 or #930. Packing reduces eager tensor
bookkeeping; whole-program compiled execution and input batches remain needed.
Do not advertise this as an automatic backend-selection capability.

## Evidence and reproduction

- [All aggregate ratios and three-process medians](aggregate.json)
- Baseline process summaries: [1](baseline/trial-1.json), [2](baseline/trial-2.json), [3](baseline/trial-3.json)
- Candidate process summaries: [1](candidate/trial-1.json), [2](candidate/trial-2.json), [3](candidate/trial-3.json)

Each process summary retains every block sample, context-first time and MAD.
The current candidate's single-example matrix is runnable through the existing
benchmark, without a circuit-specific performance adapter:

```bash
python -m benchmarks.module_execution_matrix --output /tmp/fq-rotation-matrix
```

For controlled baseline execution, use baseline_worker.py.txt as Python source,
materialize baseline_batching.py.txt at the path named there, and pass source
root, output JSON, width, depth and ring flag (0/1). It calls the same measure()
function. Original per-case JSON/logs, frozen source and phase drivers are
preserved outside source control; publication.json binds originals to public
copies. The artifact does not alter historical training measurements.

## Verification

440 focused statevector numerical, compiled-Module, batch, gradient, Adam,
shape-refusal and parameter-binding tests passed. New workflows cover shared
nonlinear parameters and trainable features, widths 2/4, depths 1/3, float32/64,
batches 1/4, no_grad/inference and constants/vector/cast parameter paths.
Reference and optimized training own independent initial parameter storage.
The independent statevector kernel equivalence suite remains passing.

Evidence is single_device_fast_path, non-release; required CI and the recorded
full-suite limitations remain separate from these focused local checks.

Strict package mypy passed for all 840 selected source files. Black, Ruff,
architecture, dependency policy, public API baseline, public evidence privacy
and repository hygiene checks passed. The aggregated proposed budgets passed
10/50 public-JAX overhead cases and 0/50 default-accessibility cases; JAX was
not changed by this optimization, so changes from prior single-campaign budget
counts are not an implementation-speed claim.
