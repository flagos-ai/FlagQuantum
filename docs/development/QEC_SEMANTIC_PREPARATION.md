# QEC preparation: reuse immutable stage identities

Compiler owns this private pass-pipeline optimization. The previous
[leaf-snapshot optimization](QEC_PREPARATION_OPTIMIZATION.md) is already merged.
This slice does not replace Stim/PyMatching or introduce a numerical kernel,
global cache, IR representation, API, result schema or identity algorithm.

## Repeated work and preserved semantics

The earlier attribution profile recorded sixteen `semantic_identity` calls per
QEC preparation component. The six-pass pipeline alone computed fourteen:
input/output for every pass, followed by source/final identities. Adjacent
records serialize the same immutable stage again. `HybridProgram` and its
attributes are recursively frozen by the existing private model.

Compute the initial identity after initial analysis, and each transformed
identity after its complete analysis. Carry those strings only within the
current pipeline invocation. The next record consumes the preceding output
identity, and final evidence uses the initial and final strings. Every stage
still calls `analyze_program`, including its full verification; canonical JSON,
SHA256, attributes, diagnostics and pass transformations remain unchanged.
No cross-invocation state is retained. The immutable-program contract remains
required; bypassing frozen models with reflective mutation is unsupported.

The regression uses empty and three-renaming-pass pipelines. Independent
canonical JSON/SHA256 checks every record and result; verification counts stay
one per stage. The original implementation makes two/eight identity calls and
fails the one/four bound. The candidate passes, including a subsequent fresh
invocation. The full hybrid compiler suite passes 386 tests. Initial fixture
capture mistakenly returned an unsupported scalar; that failure is retained,
and the corrected fixture returns a supported expectation before the original
cost regression is measured.

## Predeclared same-workload observations

Baseline: `677c5d47d67f1fc71274be672548d6a626334852`.
Frozen candidate: `3a10eb650f35994f9a2c5dfafcb7cb5507f42f95`.
[All paired raw rows and protocol](evidence/qec_semantic_preparation_cpu_20261025.json)
are retained, including warmups. Surface d9 X/native, three rounds, all-data X
product,1025shots,batch128,seed53 and X/Z/measurement noise .005/.007/.009.
Three fresh-process pairs in AB/BA/AB order; each process has one warmup and
three measured repeats,30-second process/180-second study budget, one thread.
No concurrent local tests ran during collection. All six processes succeeded.
The baseline is a verified exact source archive; the candidate is a clean
checkout. Initial imports and separate diagnostics are outside timed pipeline
rows. There is no post-hoc performance gate or additional sample collection.

Across nine measured rows per implementation, pipeline medians are
1.944327s and 1.692940s (local ratio 1.1485).
All six source, detector, observable and prediction SHA256 sets are identical;
the diagnostic detector event counts are nonzero. This is one local CPU workload,
not a confidence interval, universal speed guarantee, threshold, distance trend,
GPU benefit or capacity result. Repeated seeds are not independent trials.
Do not combine these absolute times with earlier studies from other runs.
No new full-distance matrix claim is made by this narrower slice.

## User workflow and limits

Install `python -m pip install -e '.[stim,pymatching]'` and run the existing
[public two-entrance Python example](QEC_COMPATIBILITY_PERFORMANCE_BASELINE.md#reproduction-and-user-workflow).
Users keep their memory declaration, strict source-derived DEM and same-sample
decoder calls; they do not configure identity caching. Scope-specific example
output is recorded in the PR. No nominal probability or actual positioned
Kraus law is changed, and the existing nominal-decimal precision limit remains.
Current-head complete CI/GPU and normal branch rules are required for delivery.
An A800 investigation still needs actual device access, driver/CUDA/Torch,
free VRAM and a bounded workload; sixteen cards have not been accessed.
