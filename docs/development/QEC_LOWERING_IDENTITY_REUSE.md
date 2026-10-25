# QEC preparation: reuse verified identities in lowering results

## Ownership and observed repeated work

Compiler owns this small follow-up to [pipeline-local identity reuse](QEC_SEMANTIC_PREPARATION.md).
Latest baseline main is `2cde712535badbe1ef8e2a0e56dcde74911d4e7d`.
The predeclared d9 X component profiles each record nine `semantic_identity`
calls after #840: seven from the six-pass pipeline, then two duplicate
serializations during dynamic lowering result assembly. Instrumented cumulative
identity time was approximately .153/.154 seconds for export/compilation.
Those are attribution measurements, not uninstrumented pipeline wall times.

Both dynamic lowering and static specialization already hold the authoritative
`HybridOptimizationResult`. Assemble their source/optimized identity fields
from its `source_identity`/`optimized_identity` rather than recomputing them.
The original immutable program, canonical JSON/SHA256 algorithm, every pass and
complete stage validation remain unchanged. No cache, global state, alternate
IR, SDK adapter, public API/default/schema or numerical/seed/noise change is
introduced. Other main QEC/Ising work is not reimplemented here.

Four regression cases cover static/dynamic paths with optimization enabled
and disabled. Each compares every optimization record and independently
canonicalizes source/final programs; full verification counts remain seven/one.
On unchanged baseline source, all four reach the cost assertion and fail with
nine-versus-seven or three-versus-one identity calls. The candidate passes,
as does the full 390-test hybrid Compiler suite. Existing tests remain the
semantics authority for lowered instructions, bindings, diagnostics and limits.
Separate attribution confirms seven identity calls in each QEC component.

## Fixed same-workload comparison

Frozen candidate source: `bd5b8170a47ee6968bc2b5984eeef1ad10d771fd`.
[Protocols, all raw observations and profiles](evidence/qec_lowering_identity_cpu_20261025.json)
retain every warmup and measured row. Surface d9 X/native, three rounds,
all-data X product, noise .005/.007/.009,1025shots,batch128,seed53 and one thread
match the previous narrow protocol. AB/BA/AB six fresh processes, each one warmup
and three measured repetitions;30-second process/180-second study budget.
Initial imports and separate diagnostics are outside timed pipeline rows.
No concurrent local tests ran during collection, and no samples were added.

All six processes completed. Nine measured rows per implementation give median
pipeline wall 1.262200s and 1.217204s, ratio 1.0370.
This is a descriptive local observation without a confidence interval or speed
promise. Repeated seeds are not independent scientific trials. Do not combine
these absolute times with previous host runs or multiply reported ratios.
All source-text, detector, observable and prediction SHA256 witnesses are
identical across the six diagnostic runs, with nonzero detector events.
Diagnostics and their memory are excluded from timing; whole-process RSS also
includes imports, warmups and SDK scratch and is not a phase-exclusive peak.
No new full-distance matrix, threshold, distance trend, GPU or capacity claim
is made. The prior matrix and timeout history remain untouched.

## User workflow and acceptance

Install `python -m pip install -e '.[stim,pymatching]'`, then use the unchanged
[public two-entrance Python example](QEC_COMPATIBILITY_PERFORMANCE_BASELINE.md#reproduction-and-user-workflow).
PR evidence records its actual candidate output. Users do not manage identity
state or call private compiler functions. The source sampler, strict actual-law
DEM and same-sample decoder remain authoritative; the existing nominal-decimal
precision limitation is not repaired or hidden by this work.

Required local checks and current-head full CI/GPU, reviews and branch rules
must pass before normal fixed-head merge; actual merge/main verification closes
delivery. No admin/bypass/force or weakened scientific assertion is permitted.
Further preparation work or a single A800 experiment is a separate decision;
actual access, driver/CUDA/Torch, free VRAM and a bounded workload are still
prerequisites. Sixteen A800s have not been accessed by this slice.
