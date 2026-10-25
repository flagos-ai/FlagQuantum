# Prepared CPU parity for compiled memory sampling

## Scope and decision

This slice follows the [matched prepared-CPU control](QEC_PREPARED_PARITY_CONTROL.md).
That study does not establish a universal end-to-end benefit for prepared CPU,
and its incomplete grid remains unchanged. The current production candidate
removes repeated index construction inside the existing CPU `MemorySampler`.
It does not introduce a GPU backend, public option or new sampling model.

The source baseline is `5f377224db39bc5a51796f723364e9de9d2d73bd`.
Changes since the matched-control delivery concern QEC AI examples, FTQC
continuation evidence and Quafu provider paths; the measured memory sampler,
strict Stim bridge, DEM and decoder paths are unchanged. This is a
`single_device_fast_path` change owned by Core QEC execution.

## Ownership and failure boundaries

The existing immutable detector/observable column snapshots define the maps.
Each live handle now prepares grouped CPU index tensors once from those
snapshots. Their size depends on the fixed maps, not on the number of sample
calls or sampled shots. The handle retains no engine record or output batch.
There is no global or cross-handle cache.

Each draw converts its validated binary CPU record to bytes once for both maps.
Equal-width gathers use byte sums modulo 256 and retain the low bit, preserving
GF(2) parity even for products wider than 255. Empty products remain zero;
duplicate references cancel by parity. The returned `DemSample` still contains
caller-owned CPU int8 tensors. Altering a returned batch or the caller's old
layout cannot alter future draws.

Close and execution failure release both index groups with the existing
prepared stream. Creator-process/thread checks, invalid-shot validation,
continuous RNG consumption, copy/pickle refusal and exception propagation are
unchanged. Native handle-assembly failure explicitly closes the already prepared
engine, matching the external bridge's existing cleanup boundary. Both native
memory and external reference-normalized Stim use the same private handle
factory. The standalone one-shot parity helper still builds its own temporary
indices; no shared mutable state is added there.

## Acceptance sequence

1. Demonstrate that the unmodified baseline rebuilds index tensors after compile
   and fails the new no-rebuild regression.
2. Check independent XOR references for ragged/empty/duplicate/noncontiguous
   records and widths 255, 256, 257 and 513, one byte allocation per draw, output
   ownership, close/poison release and partial-assembly cleanup.
3. Run existing replacement, native/external stream, lifecycle, Surface and
   fixed-shot tests with real optional SDKs, plus the core-only profile.
4. Freeze a bounded local CPU comparison before collecting new timing data.
   Keep preparation in total latency, all warmups/raw rows and any regressions.
   Earlier private research timings are not production speedup evidence.
5. Complete type/format/architecture/read-only API and required default checks;
   retain and compare real baseline failures. Deliver only after exact-head
   complete CI/GPU, review and branch rules pass, then verify actual merge/main.

The unmodified baseline failed the no-rebuild regression at its per-draw
`torch.tensor` construction. The candidate passed 156 related real-SDK checks
(including seven new checks), the final eight-test cost/lifecycle group and a
22-test core-only profile with Stim/PyMatching actually absent. Strict typing
of the modified production module and format, architecture, read-only API,
dependency, retirement, vocabulary, language and documentation checks passed.

The required local default run completed with 11,681 passed, 91 skipped, four
native-fusion expectation failures and four sandbox Gloo setup errors in
183.26 seconds. An exact unmodified-main archive, with the imported package path
asserted, reproduced the same four failures and six setup errors in the full
focused Gloo module. Raw logs are retained; local default is not called green.
Complete final-head CI/GPU and normal PR delivery are still required. Whole-package strict mypy
reported four errors in three unrelated Simulation/JAX files; the identical
errors were reproduced on the exact unmodified main archive with the same SDK
and stub environment. Whole-package typing is not claimed green.

## Bounded local comparison

[Complete protocol, observer source, raw observations and hashes](evidence/qec_prepared_cpu_parity_20261025.json)
pin the actual main and frozen implementation commit. The producer's sampling
source and tests remain unchanged in the later documentation/evidence commit.
A single local macOS arm64 CPU host used Python 3.12.14, Torch 2.13, Stim 1.16
and PyMatching 2.4. Workload: d9 X memory, three rounds, 65,537 shots, batches
128/4,096, seed 53, noise .005/.007/.009 and one numerical-library thread.

Each batch ran six fresh processes in baseline/candidate, candidate/baseline,
baseline/candidate order. Every process retained one warmup and three measured
repetitions. The predeclared limits were 30 seconds per process and 360 seconds
for the complete study, with no concurrent local tests. All 12 processes
completed in 83.925118667 seconds. No retries, extra samples or seed changes
were used. Warmups are retained but excluded from medians; repeated same-seed
rows are not independent statistical trials.

| Batch | Baseline pipeline median (s) | Candidate pipeline median (s) | Baseline loop median (s) | Candidate loop median (s) |
| --- | --- | --- | --- | --- |
| 128 | 1.282442625 | 1.258787167 | .270640290 | .243457467 |
| 4,096 | 1.219854334 | 1.212773000 | .191597997 | .189111372 |

There are nine measured rows per implementation per batch. Pipeline ratios are
1.0188 and 1.0058, small descriptive differences of about 1.8% and 0.6% less
wall time; they do not establish statistical significance or general throughput
benefit. Loop ratios are 1.1117 and 1.0131. Pipeline includes memory construction,
export, strict DEM, decoder setup, sampler compilation, sampling/decoding/count
reduction and close. Imports, metadata and independent full-shot diagnostics
are excluded equally. The change is justified by eliminating verified repeated
allocation while preserving semantics, not by a claimed general speed threshold.

All six processes for each batch have identical source/strict-DEM/detector/
observable/prediction SHA256 and detector/logical counts. Diagnostics process
all 65,537 shots, including the one-shot tail. Batch128 has 512,157 detector
events and two logical mismatches; batch4096 has 513,809 events and zero
mismatches. Cross-batch bit identity is not a seed contract, and these counts
are not threshold or suppression evidence. RSS is the entire process high-water
mark including imports, warmup and diagnostics: roughly 267–269 MB at batch128
and 284–286 MB at batch4096, with overlapping baseline/candidate ranges.
It is neither a phase-specific peak nor a capacity certificate. Public resource
identifiers are anonymous; original raw-file SHA256s and all numerical rows are
preserved. Earlier CPU/GPU study data and its incomplete grid remain untouched.

## Existing user workflow

Install `python -m pip install -e '.[stim]'`. This workflow continues to use the
existing public sampler and its CPU result contract:

```python
from flagquantum.qec import (
    PhenomenologicalNoise, RepetitionCode, build_memory_circuit,
    compile_memory_sampler,
)

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
with compile_memory_sampler(memory, noise=PhenomenologicalNoise(), seed=53) as sampler:
    for shots in (8, 8, 1):
        batch = sampler.sample(shots=shots)
        assert batch.detectors.shape == (shots, 8)
        assert batch.observables.shape == (shots, 1)
        assert not batch.detectors.any() and not batch.observables.any()
print("17 shots, no noiseless flips")
```

The expected output is `17 shots, no noiseless flips`. Noisy source sampling,
strict actual-law DEM construction and CPU PyMatching retain their existing
ownership boundaries. Nominal decimal versus positioned-Kraus precision limits
are unchanged. This change makes no threshold, distance-trend, GPU, statistical
significance or state-capacity claim.
