# QEC sampling: group binary measurement parities

## Observed work and implementation

The native compiled sampler already lowers and prepares once. The remaining
small-batch cost was classical parity extraction: `_parity_bits` launched
conversion and XOR work separately for every measurement reference. On the
predeclared d9 X sample-only profile, 65,537 shots at batch 128 made 513 sample
calls. Parity extraction accumulated 4.201 instrumented seconds; the prepared
engine accumulated .182 seconds. These are attribution timings, not speed
baselines. The equal-width regression independently counts 2,305 tensor
operations for 128 three-reference outputs on unchanged main.

Core owns the change in `qec/execution/sampling.py`. Group output columns by
parity width and gather/reduce each group, retaining the original column order.
The already validated binary engine record is converted to bytes. Unsigned byte
sums wrap modulo 256; since 256 is even, taking the low bit preserves GF(2)
parity even for long products. Empty parities remain zero and duplicate
references cancel. Grouping avoids padding a long logical product across every
detector. Tests cover noncontiguous input, empty batches/layouts, duplicates,
caller ownership, and 255/256/257/513-reference products; the same tensor-work
regression now counts nine operations, with a ceiling of 64 tensor operations.

No cache, SDK import, sampler/decoder/IR authority, API/default/schema, seed,
noise placement or probability change is introduced. The original native source
sampler advances exactly the same stream. Engine binary/shape validation,
process/thread ownership, failure poisoning and handle close remain in place.
CPU int8 outputs stay caller-owned. The actual positioned Kraus law and the
existing nominal-decimal precision limitation remain unchanged.

## Fixed comparison and retained counterexample

[Protocols, profiles, all warmups and raw paired observations](evidence/qec_sampling_parity_cpu_20261025.json)
record baseline `7bd873856c13275c140e4be622560ab008bea60f` and frozen byte
producer `e132e74ab0d7f978b240e780686e9d38d47fad54`.
Two shared Xeon Platinum 8358 hosts, labelled a800-171/172, independently replay
Surface d9 X/native, three rounds, all-data X product, 65,537 shots, seed 53 and
X/Z/measurement probabilities .005/.007/.009 at batches 128/4,096. Each batch
uses AB/BA/AB six fresh processes, one warmup and three measured repetitions;
one Torch/OMP/MKL/OpenBLAS thread, 60 seconds per process, 600 seconds per host. The image
and SDKs are frozen; each container has a two-CPU quota, a 4 GiB cap and no mapped GPU.
Actual host Git HEAD/tree/clean checks bracket every read-only-source container;
the process asserts its imported FlagQuantum source path.

All 24 final processes completed within the original budgets. Nine measured
rows per implementation/host/batch give these descriptive pipeline medians:

| Host | Batch | Main seconds | Byte producer seconds | Main / producer |
| --- | ---: | ---: | ---: | ---: |
| a800-171 |128|6.576842|2.801477|2.3476|
| a800-172 |128|6.623834|2.797646|2.3676|
| a800-171 |4096|2.701684|2.667576|1.0128|
| a800-172 |4096|2.706793|2.621041|1.0327|

The large-batch differences are small; no significance or general speed promise
is inferred. Source, strict DEM, detector, observable and prediction SHA256
witnesses match across all six processes within each host/batch. Separate
full-shot diagnostics are outside timing. Repeated seeds and the second host
are not independent statistical trials. Pipeline time includes construction,
export, DEM, decoder/sampler preparation, sampling, decoding, reduction and
close; initial imports, metadata and report IO are excluded.

The first grouped implementation, producer
`550553114c224b7a4091e4db2a760ffb2861cd4d`, gathered int64 intermediates.
Its complete separate study is retained: batch 128 improved, but batch 4,096
pipeline medians increased by approximately 2–6%, and whole-process RSS rose
approximately 7.6–7.7 MB. The byte implementation reduces this traffic; its batch 4,096
whole-process high-water marks were approximately 0.9–1.0 MB above their paired
baselines. RSS includes imports, SDK scratch, warmups and diagnostics; neither
RSS nor visible output tensor bytes is a parity-exclusive memory peak or a
bounded-memory certificate. Do not pool the two studies, select favorable
repetitions, reuse their baselines, or multiply previous preparation ratios.

The original 48-cell matrix and its timeout history are unchanged. No new
full-distance or million-shot candidate matrix, threshold, distance trend,
A800 QEC acceleration or state-capacity result is claimed. PyMatching remains
a CPU reference; matching graph conversion never replaces source sampling.
Before a GPU replacement, re-profile the remaining engine/conversion work and
include transfer costs; the CPU phase share alone does not establish payoff.

## User workflow and validation

Install `python -m pip install -e '.[stim,pymatching]'` and use the unchanged
[public two-entrance Python workflow](QEC_COMPATIBILITY_PERFORMANCE_BASELINE.md#reproduction-and-user-workflow).
The executed example reports `native 17 0` and `exported_stim 17 0` on the local
SDK environment. These are small path checks, not logical-suppression evidence.
Users do not construct parity groups or manage private state. The research
observer/controller sources in the evidence reproduce the fixed study; they
reuse the existing observation function without changing the original CLI
budget and are not installed APIs or a second benchmark platform.

Relevant real-SDK tests, the independent GF(2)/byte-wrap/cost regressions and
read-only API/architecture/format/ownership checks accompany this change.
Local default and full-package type checks are not claimed green: exact main
reproduces four native-fusion failures and the same four unrelated type errors under the same local SDK/stub environment;
sandbox Gloo setup errors are also retained. The final default run additionally
hit an advisor calibration assertion, which its full module of 28 tests did not
reproduce on either baseline or candidate; this remains a recorded default-only
failure. A local missing-history evidence failure was resolved by importing
real SHA-verified commits, without changing the provenance policy or evidence.
Current-head complete CI/GPU, applicable checks, reviews and branch rules are
still required before normal fixed-head merge, followed by actual merge/main
verification. Detailed local raw failures are retained with PR evidence.
