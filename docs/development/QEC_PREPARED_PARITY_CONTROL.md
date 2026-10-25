# QEC parity: matched prepared CPU control

## Decision under test

The [single-GPU feasibility study](QEC_GPU_PARITY_FEASIBILITY.md) compared the
current CPU sampler with a GPU prototype that also prepared its parity indices
once. Its transfer-inclusive improvement could therefore not be attributed to
CUDA alone. This new, independently predeclared study adds the missing prepared
CPU control and does not pool observations with that earlier study.

Core owns the existing QEC workflow. This record evaluates private research
observers and does not add a production backend, cache, public API, device
selector or dependency. The original CPU stabilizer stream, strict actual-law
DEM, PyMatching decoder and logical reduction remain authoritative. The known
nominal-decimal versus actual positioned-Kraus precision limitation is unchanged.

## Three matched modes

| Mode | Index construction | Record conversion | Parity and decoding |
| --- | --- | --- | --- |
| Current CPU | Existing public sampler builds indices every call | Existing per-layout conversion | Existing CPU parity and CPU PyMatching |
| Prepared CPU | CPU indices built once per sampler | One CPU byte conversion shared by both layouts | Same prepared gather/reduce algorithm as GPU, executed on CPU; CPU PyMatching |
| Prepared GPU | CUDA indices built once per sampler | One byte record transfer to GPU shared by both layouts | Same prepared gather/reduce algorithm on CUDA, both outputs returned to CPU; CPU PyMatching |

Both prepared modes group the same immutable columns by width and use unsigned
byte sums followed by the low bit. They differ in device placement and the
associated transfers. Current-CPU versus prepared-CPU differences include both
index reuse and sharing a record conversion; this is not an isolated timing of
index construction. No global or cross-sampler cache is introduced.

Each fresh process draws Surface d9 all-data X memory, three rounds, 65,537
shots, seed 53, with X/Z/measurement probabilities .005/.007/.009. The fixed
batches are 128, 4,096 and 65,536, including a one-shot tail. Per batch the
process order is current CPU / prepared CPU / GPU, prepared CPU / GPU /
current CPU, GPU / current CPU / prepared CPU. Each process retains one warmup
and three measured repetitions. A complete host/batch/mode thus has nine
measured rows. Repeated seeds and replay on a second host are not independent
scientific trials.

The frozen source is actual main
`29605a979d66d95809eca785bb81bdd5b57bdc20`. Two shared A800 hosts independently
replay the protocol, with one mapped GPU, one numerical-library thread, a
two-CPU quota and a 4 GiB host-memory cap. Limits are 60 seconds per process
and 900 seconds per host for the new three-mode study. These are new declared
budgets, not an extension or retry of the earlier experiment. GPU availability
is checked before every container: no compute process, more than 70,000 MiB
free, and zero utilization after at most ten seconds of waiting for residual
utilization. Foreign work is never cancelled, and an unavailable device stops
the study rather than triggering remapping or extra samples. Availability is
not an exclusive reservation.

The loop timer includes the original CPU engine sampling, conversion/parity,
both CUDA transfer directions where applicable, CPU decoding and reduction.
Preparation, including static indices, is separately timed and added to the
loop time for the total. Imports, initial CUDA-context setup, metadata and
independent diagnostics are excluded equally across modes; CUDA is synchronized
at timing boundaries. Transfers are blocking and pageable. No pinned-memory,
overlap, fused-kernel, GPU stabilizer or GPU matching decoder is tested.

Independent CPU and CUDA XOR witnesses include empty and duplicate products
and widths 255, 256, 257 and 513. A separate full-shot diagnostic compares every
prepared batch to the unchanged CPU function on the identical engine record,
then hashes source, strict DEM, detectors, observables and predictions.
Diagnostics and warmups remain in raw evidence, but diagnostics and warmups do
not contribute to measured medians. CUDA tensor allocation excludes driver and
context memory; RSS is a whole-process high-water mark including imports,
warmup and diagnostics. Neither proves a phase memory bound or state capacity.

## Supported user workflow

Install `python -m pip install -e '.[stim,pymatching]'`. The existing
[public two-entrance Python example](QEC_COMPATIBILITY_PERFORMANCE_BASELINE.md#reproduction-and-user-workflow)
continues to use `compile_memory_sampler` or `compile_stim_sampler` and
`decoder.decode_batch(sample.detectors)`. It returns caller-owned CPU int8
samples. Research-mode names and private prepared state are not user APIs.

No production implementation or performance promotion is selected before the
fixed collection and correctness review. Incomplete cells must remain explicit;
no seed selection, extra samples or post-hoc speed threshold is permitted.

## Collected evidence

[Protocol, exact observer/controllers and review](evidence/qec_prepared_parity_control_20261025.json)
link the complete raw observations for
[host A](evidence/qec_prepared_parity_control_20261025_host_a.json) and
[host B](evidence/qec_prepared_parity_control_20261025_host_b.json).
All warmups and diagnostics are retained. Public records use anonymous host
labels and omit GPU UUIDs, SSH aliases and container identifiers; original
private archives retain those identifiers and their transfer SHA256 digests.
No timing, event count, diagnostic hash or numerical observation is redacted.

The study completed 41 of 54 planned processes. Host B completed all 27 in
506.930 seconds. Host A completed 14; a foreign compute process appeared on
the selected device before the next process, so its remaining 13 were not
started. No device remapping, budget increase or measured-cell retry was used.
The two-host matrix is incomplete. Host A has a complete batch-128 comparison;
its batch-4,096 counts are three current-CPU, six prepared-CPU and six GPU
measured rows, and batch 65,536 has no observations. Neither incomplete pair
has a summary timing or ratio. All 41 exact owned containers were checked
absent after collection; no foreign process was removed.

Each complete three-mode pair has nine measured rows per mode. The following
medians are descriptive and do not imply significance, independent trials or
confidence intervals. Preparation-plus-loop totals are medians of each row's
sum, not sums of separately minimized phases.

| Host | Batch | Current CPU loop (s) | Prepared CPU loop (s) | Prepared GPU loop (s) |
| --- | ---: | ---: | ---: | ---: |
| A | 128 | 0.745995 | 0.651660 | 0.627802 |
| B | 128 | 0.752526 | 0.662165 | 0.624691 |
| A | 4096 | unavailable | unavailable | unavailable |
| B | 4096 | 0.657713 | 0.642288 | 0.466106 |
| A | 65536 | unavailable | unavailable | unavailable |
| B | 65536 | 0.840896 | 0.806588 | 0.632542 |

| Host | Batch | Current CPU prepare + loop (s) | Prepared CPU prepare + loop (s) | Prepared GPU prepare + loop (s) |
| --- | ---: | ---: | ---: | ---: |
| A | 128 | 2.797113 | 2.718792 | 2.702432 |
| B | 128 | 2.859815 | 2.810284 | 2.776785 |
| A | 4096 | unavailable | unavailable | unavailable |
| B | 4096 | 2.778320 | 2.817318 | 2.570530 |
| A | 65536 | unavailable | unavailable | unavailable |
| B | 65536 | 2.952560 | 2.951853 | 2.811226 |

At batch 128, preparation reuse and sharing a conversion reduce the CPU loop
median by approximately 12%–13% relative to current CPU. Relative to that
prepared CPU control, the GPU loop medians are another approximately 4%–6%
lower, but preparation-plus-loop totals differ by only approximately
0.6%–1.2%. The current-CPU versus GPU gap therefore cannot be read as the
benefit of device placement alone. The earlier study explicitly left this
attribution unresolved; its numbers remain scoped to its own experiment.

On the single host with complete larger-batch pairs, the prepared GPU loop
medians are approximately 22%–27% lower than prepared CPU, and the corresponding
totals are approximately 4.8%–8.8% lower. This is evidence about the declared
workload and implementations, not a general accelerator or production claim.
Prepared CPU is not uniformly better end to end: at batch 4,096 on host B its
total median is approximately 1.4% higher than current CPU, while batch 65,536
totals are nearly equal. Preparation-time variation and shared-host effects
remain in the raw rows; none is discarded or reassigned to a faster phase.

All completed processes passed both independent XOR witnesses and full-shot
same-record parity comparisons. Source, strict DEM, detector, observable and
prediction hashes agree within each observed host/batch across all modes.
The host CPUs are Xeon Platinum 8358, with one configured numerical-library
thread; the frozen environment is Python 3.12.13, Torch 2.13.0+cu130,
CUDA 13.0, Stim 1.16 and PyMatching 2.4, with A800-SXM4-80GB devices and
driver 580.126.20. These are CPU/GPU parity observations, not a GPU stabilizer
sampler or matching-decoder implementation.

## Disposition and verification

Keep current public behavior and CPU defaults unchanged. This control answers
the attribution question but does not authorize a device-selection API or a
new global cache. A production CPU preparation change would need its own
bounded lifecycle, caller-ownership and replacement tests. A GPU proposal
would need an explicit opt-in boundary and supported workload, while retaining
the CPU fast path and including transfer and preparation costs. Neither
implementation is activated by this record. Do not extrapolate to more
distances, million-shot runs, multi-GPU throughput, state capacity, thresholds
or logical suppression.

The unchanged public example was executed on macOS arm64 with Python 3.12.14,
Torch 2.13, Stim 1.16 and PyMatching 2.4, reporting `native 17 0` and
`exported_stim 17 0`. Those small CPU checks are distinct from the Linux timing
experiment. The related real-SDK sampler/bridge/Surface/fixed-shot suite passed
149 tests. The exact main's unchanged production code was also run through
the default suite: 11,347 passed, 85 skipped, five failed and five sandbox Gloo
setup errors at the failure cutoff. Four failures are native-fusion expectations;
the evidence-history failure was caused by missing real commits cited by newly
merged FTQC evidence. Importing SHA-verified ancestry headers restored the
evidence gate without modifying another team's evidence, policy or assertions.
Original failures are retained; the local default suite is not claimed green.
Complete current-head CI/GPU, applicable review and branch rules, normal merge
and actual merge/main verification remain delivery requirements.
