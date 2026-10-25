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

At candidate-review time, the original no-rebuild regression failed at the
baseline's `torch.tensor` index construction. Candidate parity/lifecycle/bridge
checks passed. Production timing, complete local verification and PR delivery
remain pending; no speedup or completed delivery is claimed here.

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
