# QEC preparation: avoid leaf-operation environment snapshots

## Measured problem and ownership

The [bounded compatibility matrix](QEC_COMPATIBILITY_PERFORMANCE_BASELINE.md)
is delivered by #825. Its original 36 completed cells and 12 process timeouts
remain unchanged. Phase timing selected preparation for investigation, not a GPU
kernel. Compiler owns this change; QEC and Simulation keep their current APIs,
noise law, annotations, layouts, sampler ownership and SDK boundary.

On clean main `0a9a97126864b3cd3fe274c98212100ff0812d1b`, predeclared profiling
used Surface d=9 X, three rounds, the existing all-data X product and noise
0.005/0.007/0.009. Each of `memory_to_stim_text` and `compile_memory_sampler`
ran once for warmup, then once under cProfile with a newly constructed memory.
Each fresh process was bounded to 60 seconds, total budget 180 seconds; threads
were fixed to one. The sampler additionally completed a 17-shot shape/lifecycle
check outside the instrumented call. These are attribution observations, not
uninstrumented end-to-end timings or speedup measurements.

The instrumented calls took 1.842813 and 1.792365 seconds respectively.
`_Verifier._verify_block` accumulated approximately 1.043 and 1.047 seconds,
through eight `verify_program` calls per component. The verifier built a filtered
snapshot of the entire growing lexical environment after every operation,
including leaf operations with no region to consume that snapshot. This makes
flat-block work quadratic in its number of definitions.

## Small private change and correctness

Construct that snapshot only when `operation.regions` is nonempty. All operation
schemas, SSA uses/definitions, type checks, linear-effect consumption, nested
region verification and deterministic diagnostics still execute. The snapshot
is still taken before the parent operation's results are defined, and shared
between its regions exactly as before. There is no cache, skipped verifier,
second IR, numerical kernel, public API or serialized-result change.

The new deterministic complexity test counts `IRType.linear` inspections rather
than wall time. The original implementation fails its growth bound with
2598 inspections at 64 constants and 9286 at 128. Functional cases independently
check that nested regions can capture prior non-linear values, and still refuse
capture of the parent result not yet in scope, preserving both diagnostics.
The full existing hybrid compiler suite remains the semantics authority.

## Predeclared paired measurement

Compare exact baseline main above and a clean candidate producer using the same
unmodified `_observe` workload from the compatibility matrix: d=9 X/native,
three rounds, 1025 shots, batch128, seed53 and the same three noise probabilities.
Use three fresh-process pairs in order baseline/candidate, candidate/baseline,
baseline/candidate. Every process has one warmup and three measured repetitions;
per-process timeout is 30 seconds and total budget 180 seconds. Retain all rows,
including warmups and failures. No concurrent local test/measurement job is run.
No post-hoc speedup gate or extra samples are enabled.

The baseline is an exact Git archive of the recorded main, containing the
original package and observation helper. The candidate is its clean source
checkout. The controller verifies identities rather than fabricating a Git
checkout inside the archive. Initial imports are outside the helper's pipeline
wall; each repetition constructs fresh memory/DEM/decoder and closes its sampler.
After timing, compare exact source-text, detector, observable and prediction
SHA256 values for a separate 1025-shot deterministic diagnostic run. Require
nonzero detector events so an all-zero witness cannot hide a broken sampler.
Those diagnostics and their memory belong outside the timed observations.

After the paired study, rerun the original 48-cell matrix with its unchanged
30-second/600-second budget. Keep the twelve historical timeouts. If newly
completed large cells have no original accepted timing, do not invent a speedup
for them. Count reduction includes visible-byte bookkeeping; process RSS includes
imports/warmups/SDK scratch and is not a phase-exclusive peak or capacity claim.

## Runnable user workflow

Install `python -m pip install -e '.[stim,pymatching]'`, then run the public Python
workflow in the [compatibility baseline](QEC_COMPATIBILITY_PERFORMANCE_BASELINE.md#reproduction-and-user-workflow).
It uses both approved experiment entrances, strict source-derived DEMs and
same-sample PyMatching prediction. Expected invariant: both complete 17 shots.
The baseline's scoped observed stdout was `native 17 0` and
`exported_stim 17 0`; the candidate is rerun and recorded with its evidence.
Normal users do not call private compiler verification or manage a cache.

The change is accepted only with preserved compiler/QEC semantics, recorded
same-workload observations, necessary quality checks and current-head full
CI/GPU plus normal merge. Local CPU observations do not certify A800, multi-card,
threshold, capacity or release-grade performance. Further optimization or an
internal kernel replacement remains a separate decision based on residual cost.
