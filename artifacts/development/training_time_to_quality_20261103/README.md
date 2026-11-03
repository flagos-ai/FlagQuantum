# Cold CPU training time to a matched objective

This campaign repeats the six previously reserved width/depth/batch/layout
cases and the SGD and Adam settings from the earlier
[training candidate campaign](../torch_training_cpu_20261103/README.md).
The three routes use the same batch, data schedule, objective and 60-update
limit within each case. Three fresh serial processes repeat the fixed seed
20261102 for every route/case/optimizer, with route order rotated by trial.
Each process gets an empty Inductor cache, and JAX persistent caching is
disabled. The recorded Torch candidate uses the private test gate only for
measurement. Adam updates stay eager on that route.

The coordinator writes all 12 thresholds before launching workers. Each is
90% of the first-to-final loss improvement in the previously archived eager
run, using the historical medians of the first and final observations. The
first crossing is based on the pre-update training objective for the current
features. This measures matched progress on frozen benchmark workloads and
does not establish a held-out data quality result. The workers still complete
all 60 updates and replay the step-30 checkpoint. Each receipt includes
framework import, setup, time to each update, steady intervals, checkpoint
replay and peak worker RSS. Cold time to target adds import and setup to
elapsed training. It excludes interpreter launch and evaluation after the
crossing. Public callbacks observe the same loss and retain consumed IR
hashes. Whole JAX uses a separate optimizer and an in-memory checkpoint.
These interfaces have different receipt costs.

The shape probe runs after the measured trajectory and its replay. It takes
one step on the existing batch, one after changing batch size and one more on
the new shape. Cases with batch one grow to two, and other cases shrink to
one. Target and weight tensors retain their original shape and broadcast.
This measures end-to-end first-use cost for a changed shape, including
binding and execution, rather than timing just the compiler. Torch diagnostics
are retained for each probe. The timings are at a new public train call
boundary. Existing tests separately cover the in-call eager invalidation
behavior. The [native adjoint comparison](../native_adjoint_training_cpu_20261103/README.md)
uses the original classifier and Heisenberg quality protocols and discloses
the native batching refusal.

## Reproduction

Run from the repository root with Torch, JAX and TensorCircuit available.

```bash
D=artifacts/development/training_time_to_quality_20261103
python "$D/run.py" --output /tmp/fq936-time-quality-rerun
python "$D/analyze.py" /tmp/fq936-time-quality-rerun
python "$D/analyze.py"
```

The compressed receipt keeps the protocol, each full worker record and any
failure or timeout. Producer hashes cover the two shared workers, the
coordinator and the historical calibration archives. The runtime source
commit is recorded in each worker receipt. Runtime source files were
unmodified. The benchmark changes add timing receipts and shape probes.

## Results and acceptance

All 108 workers passed and reached the target. Every matched group crossed
on the same observation, at step 45–56. Maximum paired loss error was
2.39e-15. The historical archive verifier also passed.

Ranges below span the 12 case/optimizer medians, each from three fresh
processes. The steady column compares each route with eager in the same
case. Peak RSS is per worker and excludes compiler subprocess memory.

| Route | Cold target time | Cold target / eager | Steady step / eager | First changed batch | Peak worker RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public fq.train eager | 0.935–1.274 s | 1.00× | 1.00× | 0.017–0.088 s | 293–309 MiB |
| Torch candidate | 7.829–15.501 s | 8.37–12.49× | 0.81–1.14× | 4.65–9.15 s | 435–572 MiB |
| TensorCircuit whole JAX | 1.841–3.601 s | 1.97–2.99× | 0.044–0.382× | 0.275–1.283 s | 547–760 MiB |

The Torch candidate reduced the steady median in six of twelve pairs. A
linear extrapolation from first-update cost and steady median puts break-even
at about 17,805–291,821 updates in those pairs and predicts no crossing in
the other six. For whole JAX the same calculation gives 368–1,358 updates.
These are extrapolations under a fixed shape and rate, outside the 60-step
measurement window. They do not establish a promotion threshold.

A separate instrumented eager run on case 3 with Adam identified cold
builder tracing and Torch backward as substantial costs. Across the
training, replay and shape probe, four builder invocations accumulated
0.441 s, and 93 backward engine calls accumulated 0.244 s. The statevector
program accounted for 0.125 s across 94 executions. These cumulative
cProfile durations include instrumentation overhead and nested calls. They
are diagnostic rather than the uninstrumented campaign times. Reproduce
the profile separately with:

```bash
PYTHONPATH=. python -m cProfile -o /tmp/fq936-eager.pstats "$D/run.py" \
  --worker --case 3 --optimizer Adam --route default_eager \
  --output /tmp/fq936-eager-profile.json
```

The results support retaining the existing eager default. **#936** now has
matched objective and shape-change timings, while default CPU compiled
benefit and certified compiled Adam remain outstanding. A production
optimization can target cold builder capture or backward work and must
preserve changing-input gradients, checkpoint identity, and validation.
**#918** still needs comparable strong public routes on realistic tasks and
a demonstrated default performance benefit. The six-shape objective and
the separate native task results answer complementary parts of that
acceptance.
