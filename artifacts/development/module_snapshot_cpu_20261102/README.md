# Default Module diagnostic snapshot: CPU development comparison

The default Torch Module path now defers diagnostic IR copies for eligible static
binding graphs, using the existing JAX snapshot mechanism. Current bound angles
are still validated before numerical execution. Diagnostic/checkpoint reads
materialize consumed values even after in-place updates. Unsupported graphs and
large unused feature buffers retain eager snapshots. No public signature,
backend/default, precision, derivative support or checkpoint schema changes.
This is a `single_device_fast_path` development result, not release certification.

## Frozen comparison

Before: `6649717cd9859ceb64eaa525e07ec32932b70485`.
After: `3b414763bdaa23aa4d93a2f4f28da5227c6e120f`.
Both clean source checkouts use the same Python 3.12.14 environment, Torch 2.14.1,
JAX/jaxlib 0.10.2 and identical loadable native CPU extension binary. This is a
source-checkout comparison, not a wheel installation comparison. The native
extension was built in place; availability does not imply Module uses a
whole-circuit native adjoint. Apple M5 CPU, complex128, Torch/BLAS one thread;
XLA thread count is not constrained. CUDA was unavailable on this machine.

Three fresh shape processes per phase, ten width/depth/chain/ring cases, five
derivative demands and three routes: 60 processes and 900 numerical checks all
completed. Each process rotates route order across five blocks of 30 completed
calls. Phase orders are before/after, after/before, before/after, executed serially
without concurrent test/build jobs. Context-first timing retains compilation,
but reuses earlier contexts' caches: it is not cold-import or time-to-quality.
The earlier exploratory campaigns are not used in this report.

For each shape/demand/route take the median of three process medians, then divide
before by after. The table reports ratios across all ten shapes. Values above
one mean lower candidate cost. All samples, process MAD/min/max and first-call
costs remain in [raw trials](publication.json) and [aggregate.json](aggregate.json).
No confidence interval is asserted with three trials.

| Demand | Default median ratio | Range | Untouched public JAX median ratio |
| --- | ---: | ---: | ---: |
| parameters | 1.149 | 1.104–1.307 | 1.028 |
| inputs | 1.147 | 0.967–1.337 | 1.059 |
| both | 1.182 | 1.082–1.235 | 1.054 |
| no_grad | 1.310 | 1.230–1.453 | 1.050 |
| inference | 1.351 | 1.231–1.444 | 1.066 |

One aggregate input-only case regresses (ratio 0.967). Untouched JAX controls
also drift; these bounded development signals do not establish a universal
speedup. The proposed public-overhead and default-accessibility budgets both
pass **0/50** aggregate cases before and after. Their thresholds are unchanged.
No external-framework, GPU, memory, whole-step or time-to-target claim is made.

## User reproduction

From a checkout with the optional Torch/JAX benchmark dependencies and native
CPU extension built, the existing public-consumer benchmark is:

```bash
python -m benchmarks.module_execution_matrix --output /tmp/module-matrix \
  --repeats 5 --iterations 30 --timeout 120
# Fails on currently unmet proposed budgets:
python -m benchmarks.module_execution_matrix --output /tmp/module-budget --enforce
```

It runs ordinary `fq.Module` calls, backward, no_grad and inference; the internal
kernel is an explicitly labelled diagnostic. For the complete paired protocol,
check out the two source commits above in independent directories, make the
same loadable native extension available in both, and run with the same Python:

```bash
python artifacts/development/module_snapshot_cpu_20261102/validation/reproduce.py \
  --before /tmp/module-before --after /tmp/module-after --output /tmp/module-paired
python artifacts/development/module_snapshot_cpu_20261102/validation/verify_sources.py
```

The verifier checks retained-file hashes, source producer hashes, all trial
shapes/contexts and byte-equivalent aggregate reconstruction. Published JSON and
compressed logs use the canonical evidence sanitizer; gzip timestamps are zero.
`publication.json` records original and published SHA256 digests. Original local
campaign directories are retained separately. Validation logs accompany this
packet, including the initial trajectory-test failure caused by an unprimed
restore target and its corrected run. No production fix was made for that test.

## Closure assessment

| Requirement | Result and owner |
| --- | --- |
| Snapshot overhead, gradients, consumed hashes | Runtime compatible optimization; independent analytic VJPs, Torch Hessian and Adam recovery tests |
| Repeated fresh-process samples | This finite single-example CPU matrix complete; all raw results retained |
| Reusable canonical program | Compiler/Runtime/Simulation #928; user-approved contract and replacement fake, production implementation pending |
| Realistic batch/external/GPU acceptance | #929 and parent verification; not supplied by this scalar CPU matrix |
| Visible effective route and adoption | Runtime/API #930; no automatic selection claimed |
| Complete training-step compilation | #936; separate boundary review remains required |
| Public accessibility budgets, cold result and memory | Parent #918 remains open; budgets fail, broader acceptance still required |

The default Torch route's higher-order derivatives are retained; the explicit
JAX bridge remains first-order. A silent switch to JAX would be incompatible.
This packet cannot close #918 or its remaining implementation issues.
