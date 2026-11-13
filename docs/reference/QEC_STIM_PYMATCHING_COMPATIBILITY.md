# Stim and PyMatching integration compatibility

This table describes the supported FlagQuantum integration. It does not claim a
complete native replacement for either SDK. User entry points are in the
[quickstart](../guides/QEC_INTEGRATION_QUICKSTART.md).

## Dependency and platform evidence

| Component | Existing declared requirement | Local candidate validation |
| --- | --- | --- |
| Python | Existing project Python requirement; CI keeps its current version matrix | CPython 3.12.14 source checks; other required versions await final-head CI |
| Torch | Existing project requirement | 2.13.0 with SDKs; separate core-only source check uses 2.10.0 |
| Stim | >=1.16,<2 | 1.16.0 |
| PyMatching | >=2.3,<3 | 2.4.0 |
| NumPy | Existing SDK dependency resolution | 2.5.4 in the SDK target environment |

No dependency requirement is changed by this integration. A declared range is
not certification of every version pair. The exact candidate wheel/sdist reports
record what was actually installed. Local source SDK targets are not fresh
installed profiles. Linux package CI retains the four real wheel profiles and
sdist-both; existing macOS/Windows wheel checks remain mandatory. Optional-SDK
qualification on one OS does not certify every OS/SDK combination.

## Entry points and supported domain

| Operation | Owner and route | Preserved boundary |
| --- | --- | --- |
| Compiled measurement file writing | Simulation SDK/embedded boundary | `01`, `b8`, `r8`, `hits`, `dets`, `ptb64`; explicit measurement columns; 1 GiB encoded bound; ptb64 shot alignment 64; new-head delivery pending |
| Code/memory/noise declaration | FlagQuantum QEC | Only explicitly executed code/basis scenarios are accepted as workflows |
| Native-memory fixed-shot report | `execution.experiments.run_memory_experiment` | Original MemoryCircuit, required noise declaration, default decoder settings; source `memory_circuit` |
| External-text fixed-shot report | `execution.experiments.run_stim_experiment` | Text accepted by existing Stim bridge, strict DEM, at least one detector/observable; source `stim_circuit` |
| Stim parsing/compilation/sampling | Simulation SDK boundary | Existing instruction, size, seed, reference-normalization and ownership limits |
| DEM text and probability model | FlagQuantum canonical DEM | Existing repeat bounds, detector/logical indexing, exact law and explicit refusal rules |
| Complete independent-shot batches | `decoders.batch.decode_batch` | Sparse indices; five certified concrete types; complete original record fields |
| PyMatching logical-only batches | `adapters.pymatching.PyMatchingDecoder.decode_batch` | Dense binary CPU Tensor; logical predictions only |
| Decoder factory options | Existing `get_decoder` | Constructor options belong to the selected decoder; report entries intentionally use defaults only |
| Chunk/window operations | Existing DEM chunks and decoder owners | Stable shared-fault columns/layer ownership; finite geometry/refusal contracts |

## Explicit limits

- The two report entries use existing strict actual-law analysis. They do not
  silently decompose composite faults or substitute declared-DEM sampling for
  original-circuit sampling.
- The PyMatching adapter keeps its faithful-translation limits for graphlike
  models, parallel mechanisms and detectors not represented by edges. Advanced
  upstream correlated/packed/native APIs are not implied by this adapter.
- Complete records and logical-only Tensor batches are separate interfaces.
  Independent batching refuses stateful streams, subclasses and unknown plugins.
- Native matching, BP/BPOSD and PyMatching have different algorithmic objectives,
  precision/tie and convergence semantics. Equality between every decoder's
  predictions is not a requirement.
- Adaptive stopping, threshold fitting, persistence, arbitrary experiment decoder
  options, GPU decoding and physical realtime feedback are not added here.
- A zero observed failure count in a small exact example proves no threshold,
  suppression, speedup, large-scale capacity or hardware capability.

## Mandatory installed-candidate acceptance

The existing package job installs core, Stim-only, PyMatching-only and both
wheel profiles, plus sdist-both. The original four sample cells remain unchanged.
Additional report fields must verify:

1. Complete per-shot/batch records and swapped logical mapping for every
   supported native decoder, plus PyMatching when present; its dense batch path
   is checked independently.
2. Missing-Stim report refusal in core and PyMatching-only profiles.
3. External two-observable reports using native matching in Stim-only and
   PyMatching in both profiles, including nonzero ideal-reference normalization.
4. Repetition Z and Surface X/Z fixed-shot public reports using the same
   profile-appropriate decoder, preserving counts, intervals and source labels.

Actual failed/cancelled jobs and their logs remain part of delivery history.
Final-head CI/GPU, reviews and actual-main artifact acceptance must finish before
this candidate is called delivered. Historical receipts validate their original
SHA and cannot certify a newly changed package automatically.

## Noiseless stabilizer execution candidate

The recovered internal replacement uses the existing embedded Stim source
closure for noiseless `sample_stabilizer` execution. It adds no Python SDK
requirement to embedded installations and preserves the SDK implementation in
marker-free installations. The candidate adds no QEC API, decoder behavior,
format, general-feedback support or full Stim compatibility claim.

Fresh installed execution passed on macOS arm64: no-SDK embedded 82 tests,
embedded with SDK 246, and marker-free SDK 246. Actual embedded preparation and
sampling calls, owned CPU integer results, requested wire ordering, seeded
repeatability and the 16,384-wire regression were checked independently from
replacement fakes. Complete smoke/unit passed 20,567 tests (296 selected-node
skips and 36 collection skips); disjoint integration groups completed all
12,274 selected nodes (12,028 passed, 246 skipped). Failed attempts and approved
recoveries remain retained. These results qualify this bounded local candidate;
final-head CI and actual-main acceptance remain separate delivery gates.
