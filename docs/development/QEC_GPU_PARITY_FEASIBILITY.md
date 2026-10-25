# QEC parity on one GPU: feasibility boundary

## Question and ownership

The existing compiled memory sampler returns caller-owned CPU int8 tensors,
and PyMatching consumes CPU detection events. Moving parity extraction alone
to CUDA therefore requires a CPU engine record transfer to the device and
detector/observable transfers back before decoding. A fast device-only kernel
does not establish a faster user workflow.

This study evaluates that boundary without adding a GPU backend or changing the
public sampler. Core owns the existing QEC workflow; the observer is private
research code, not a registered implementation or a supported GPU API. The
source sampler, strict actual-law DEM and CPU decoder remain authoritative.
There is no replacement noise model, IR, Surface class, registry or statistic.
The existing nominal-decimal versus positioned-Kraus precision limitation is
unchanged.

## Fixed experiment

The source is the actual merged CPU-parity revision
`87751dbe694269d8801d0c9af11c96e749357207`. Each of two shared A800 hosts
runs one mapped GPU, with an independently checked clean source tree, a frozen
container and installed Stim/PyMatching versions. GPU0 on a800-171 and GPU2 on
a800-172 avoid the other observed compute processes. A zero-utilization and
no-compute-process check precedes every owned container; other work is never
cancelled. This is an availability check, not an exclusive reservation or proof
that the shared host remained free of interference throughout a process.

The predeclared workload is Surface d9, all-data X product, three rounds,
65,537 shots, seed 53 and X/Z/measurement probabilities .005/.007/.009.
Batches are 128, 4,096 and 65,536. For each batch the process order is CPU,
GPU, GPU, CPU, CPU, GPU; each fresh process retains one warmup and three
measured repetitions. Limits are 60 seconds per process and 600 seconds per
host, one numerical-library thread, a two-CPU quota and 4 GiB host-memory cap.
Same-seed repetitions and host replay are not independent scientific trials.

The GPU prototype prepares immutable parity index tensors once. Within each
sample loop it copies the actual CPU engine record to a byte CUDA tensor,
computes grouped GF(2) reductions and copies both outputs back to CPU. Transfers
are blocking and use pageable memory. It then runs the unchanged PyMatching
batch decoder and logical reduction. The timed loop includes all those steps;
preparation is recorded separately and also added to the loop time. CUDA is
synchronized at timing boundaries. Initial imports and CUDA-context setup are
outside the timings in both modes. There is no pinned-buffer, overlap,
custom-kernel, GPU-sampling or GPU-decoding claim.

A separate full-shot diagnostic compares every GPU batch against the existing
CPU parity implementation on the same engine record, then hashes source text,
strict DEM, detectors, observables and predictions. An independent XOR witness
includes empty, duplicate and 513-reference products. Diagnostic work is
excluded from timing. Peak allocated CUDA tensor bytes exclude the CUDA
context, driver and untracked allocations; RSS covers the whole process,
including imports, warmup and diagnostics. Neither is a total-memory or
capacity certificate.

## User workflow

Install `python -m pip install -e '.[stim,pymatching]'`. The supported
[public Python workflow](QEC_COMPATIBILITY_PERFORMANCE_BASELINE.md#reproduction-and-user-workflow)
continues to use `compile_memory_sampler` or `compile_stim_sampler`, followed
by `decoder.decode_batch(sample.detectors)`. Users do not manage device parity
groups or access a prepared sampler's private engine. The observer is not a
new user entry point and does not make GPU QEC a supported capability.

The unchanged example was executed on macOS arm64 with Python 3.12.14,
Torch 2.13, Stim 1.16 and PyMatching 2.4: `native 17 0` and
`exported_stim 17 0`. These small CPU workflow observations are distinct from
the Linux GPU research timings below.

## Observations and decision

[Protocol, observer/controllers, all raw observations and interruptions](evidence/qec_gpu_parity_feasibility_20261025.json)
retain 33 completed processes out of 36 planned. Both hosts completed all six
processes for batches 128 and 4,096. The a800-172 host completed all 18 processes
in 358.127 seconds. On a800-171, a foreign compute process appeared on the
selected GPU after 15 completed processes; the controller stopped before the
remaining three. No remap, extra sample or budget extension was used. The
two-host matrix is incomplete. The partial a800-171 batch-65,536 rows are
retained but have no summary timing or speed ratio.

The original controller also stopped between completed processes when a GPU
without a compute process still reported 1% or 9% utilization. A bounded idle
wait was added without changing the zero-utilization requirement. Resumption
kept every completed cell, the original observer, process order and absolute
600-second budget; no measured cell was repeated. Original failure logs and
both controller versions are retained. All 33 owned containers were checked
absent after collection; no other container or process was removed.

Each complete pair has nine measured rows per mode. The following medians are
descriptive observations, not confidence intervals or significance results.
The last two columns include preparation, sampling, both transfer directions,
CPU decoding and reduction. They exclude imports and CUDA-context setup.

| Host | Batch | CPU loop (s) | GPU-prototype loop (s) | CPU prepare + loop (s) | GPU prototype prepare + loop (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| a800-171 | 128 | 0.747250 | 0.616162 | 2.853211 | 2.662139 |
| a800-172 | 128 | 0.762779 | 0.625908 | 2.861176 | 2.683018 |
| a800-171 | 4096 | 0.600283 | 0.430463 | 2.635429 | 2.473706 |
| a800-172 | 4096 | 0.582163 | 0.402147 | 2.595327 | 2.380057 |
| a800-171 | 65536 | unavailable | unavailable | unavailable | unavailable |
| a800-172 | 65536 | 0.875315 | 0.667161 | 2.875016 | 2.641773 |

The prototype's prepare-plus-loop medians are approximately 6%–8% lower in
the complete pairs. This does **not** isolate a GPU acceleration benefit:
the prototype prepares immutable indices once while the current CPU sampler
rebuilds those indices on every call. A similarly prepared CPU control is
needed before attributing the difference to CUDA or choosing a production
implementation. This study therefore leaves the supported CPU path unchanged
and does not approve a GPU API, automatic device selection or custom kernel.
The next decision is that matched control, under a separately declared budget,
rather than promoting this research prototype.

The subsequent [matched prepared CPU control](QEC_PREPARED_PARITY_CONTROL.md)
records a separate three-mode study. Its new observations do not replace or
pool with this study's raw measurements or incomplete coverage.

Every completed process passed the independent XOR witness and full-shot
CPU/GPU equivalence diagnostic. Source, strict DEM, detector, observable and
prediction hashes agree across all completed processes within each host/batch.
The device operations execute on an A800-SXM4-80GB with driver 580.126.20,
Torch 2.13.0+cu130, CUDA 13.0, Python 3.12.13, Stim 1.16 and PyMatching 2.4.
There is no GPU stabilizer sampler or GPU matching decoder in this experiment.
No result establishes a logical threshold, distance trend, multi-GPU throughput,
state capacity, general speedup or release-certified GPU QEC capability.

The real-SDK sampler/bridge/Surface/fixed-shot checks passed 149 tests. The
default suite was also run against this exact main's unchanged production
code: 11,348 passed and 85 skipped, with four native-fusion expectation failures
and five sandbox Gloo setup errors at its failure cutoff. These raw failures
are retained; local default verification is not claimed green. Documentation,
evidence provenance, architecture and read-only public API checks accompany
the record, with complete current-head CI/GPU and normal branch-rule-compliant
merge still required for delivery.
