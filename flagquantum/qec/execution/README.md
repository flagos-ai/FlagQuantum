# QEC execution composition

Core owns the composition of existing QEC declarations, noise, DEMs and decoders
with compiler/runtime/stabilizer execution. This package must not define another
code or decoder registry, numerical kernel, code family or serialized schema.
Its initializer imports no experiment machinery. Allowed dependencies are the
existing QEC records and algorithms, compiler lowering, runtime feedback/session
execution, core IR, noise channels and the optional stabilizer sampler.

- `repetition.py`: the original repetition-memory source and feedback controller.
- `sampling.py`: lowering plans, noise placement, measurement records and sampled
  detection/observable parity. It shares lowering with `signatures.py` and keeps
  existing one-shot draws, location order and seed handling unchanged.
  `compile_memory_sampler` prepares a separate reusable stream: lowering,
  placement and Simulation compilation happen once; immutable parity-column
  tuples are retained instead of the caller's memory/noise records. Each draw
  returns the existing `DemSample` and retains no sample buffers.
- `context.py`: declaration-to-DEM and per-basis measurement maps, without shot
  execution. It consumes declarations and DEMs; declarations never import it.
- `signatures.py`: source injection, exact lowering, forced reference execution
  and the existing shared fault vocabulary. Gate-bound signatures now compose
  a private Simulation-owned Clifford parity primitive with the detector and
  observable layouts, without executing an amplitude store. The original forced
  execution remains the independent small-system test reference. DEM construction consumes these
  functions/constants without a reverse import, avoiding a module cycle.
- `experiments.py`: the approved fixed-shot composition of original-source
  export, strict actual-law DEM, complete-record batch decoding and a single
  owned sampler. Canonical imports only; no root/parent re-export or persistence
  schema. Run `examples/qec/memory_experiment.py` with the Stim/PyMatching extras
  for all five default decoders; unsupported model/decoder intersections refuse.
- `measurement_conversion.py`: reference-normalized detector and logical flips
  from existing records. Use `convert_measurement_records` with
  `circuit_format="stim"` and run
  `python -m examples.qec.stim_measurement_conversion` with the Stim extra.
  Simulation selects and owns the current execution implementation.
  For a bound native `fq.Circuit`, pass logical terminal Z readout order and
  detector/observable maps to `compile_m2d_converter`. Reuse the returned
  reference for successive CPU bool record batches. Run
  `python -m examples.qec.terminal_measurement_conversion` to compare original,
  compiled and placed circuits.
- `noise_sweep.py`: the existing finite-shot loop over repetition execution.

The approved 0.4.0 retirement removes flat execution imports and their pickle
lookups. Use the root exports or the owning modules above. Definitions and own
methods now use canonical metadata; `run_repetition_memory_noise_sweep` keeps
its surviving `qec.noise` identity. Caller-owned `weights_only=True` allowlists
must use current objects when loading new canonical checkpoints. No library
allowlist or custom reducer restores retired lookups. The eager root is unchanged.

The shortest complete user workflow is:

```bash
pip install -e '.[stim]'
PYTHONPATH=. python examples/qec/memory_layout_and_sampling.py
```

It constructs an existing repetition memory, reads decoder maps, samples zero-noise
measurements and detection events, and executes all three feedback modes. It prints
`9 8 1`, map shape `(8, 9)`, measurement shape `(4, 9)`, no detection/observable
flips, and `0.0 (0, 0, 0)` for every feedback mode. With only core dependencies,
`PYTHONPATH=. python examples/qec/restructure_compatibility.py` remains runnable.

A typical sampling/noise-placement change starts in `sampling.py` or
`signatures.py`. Preserve instruction/measurement order, source anchors and RNG
consumption; then run:

```bash
python -m pytest tests/qec/test_restructure_compatibility.py tests/qec/test_restructure_boundaries.py tests/qec/test_restructure_execution.py tests/qec/test_sampling_memory_circuit.py tests/qec/test_decoder_context.py tests/qec/test_repetition_memory.py -q
```

The trusted pre-004 fixture protects exact source/layout/map results, matrix-vs-
circuit DEM semantics, forced signatures, raw samples, detected parities, feedback
results and all surviving Python identifiers; retired identifiers are refused. Raw seeded sample equality is scoped
to its recorded Stim version and machine architecture; the unchanged independent
statistical/reference tests cover other supported sampling environments.

## Repeated batches

```python
from flagquantum.qec import (
    PhenomenologicalNoise, RepetitionCode, build_memory_circuit,
    compile_memory_sampler,
)

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
with compile_memory_sampler(memory, noise=PhenomenologicalNoise(), seed=53) as sampler:
    for _ in range(2):
        batch = sampler.sample(shots=32)
        assert batch.detectors.shape == (32, 8)
        assert not batch.detectors.any() and not batch.observables.any()
```

The handle belongs to its creator process and thread. Context exit releases the
prepared numerical stream even on an exception; `close()` is owner-idempotent.
Sampling after close or from another owner raises `RuntimeError`. Invalid shot
counts do not advance the stream. Execution/parity failure closes it before
propagating the original error. Live handles cannot be copied or pickled.
Create separate handles and assign seeds explicitly in separate workers.

A fixed seed reproduces an identical sequence of sample calls within one engine
version and machine instruction set. Different batching is not a promise of
identical bits. `PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py`
uses this stream with the optional PyMatching batch adapter, including a final
partial chunk. Configuration changes require constructing a new handle.

Run `python -m pytest tests/qec/test_memory_sampler.py tests/team/simulation/test_prepared_stabilizer_sampling.py -q`
for lifecycle, numerical seam and replacement checks. The approved additive
surface is recorded in `contracts/qec-memory-sampler-v1.json`; historical
restructuring fixtures remain unchanged.

Gate-bound one-/two-qubit Pauli mixtures are placed as complete channels using
the locations shared with DEM assembly. Branches at one location are exclusive;
different locations remain independent. The prior X/Z channel reconstruction
and seeded instruction stream are preserved. See
[QEC-FUNCTIONAL-004](../../../docs/development/QEC_FUNCTIONAL_004.md) and run
`PYTHONPATH=. python examples/qec/correlated_circuit_noise.py` for the core-only
declared-model example; physical circuit sampling uses the existing Stim extra.

## Native detector sampling and exact DEM

`Circuit.compile_detector_sampler(detector_map=..., observable_map=..., seed=...)`
uses explicit measurement instruction order, including repeated classical slots.
`sample(shots)` returns reference-normalized CPU bool detector flips; use
`separate_observables=True` for a pair, `append_observables=True` for one array,
and `bit_packed=True` for LSB-first uint8. Appended packing crosses the parity
boundary without intermediate byte padding. Zero shots do not advance RNG.
`Circuit.detector_error_model` returns the existing FlagQuantum DEM record.

Run `python -m examples.qec.native_detector_sampling` with PyMatching and either
the embedded wheel or the Stim 1.16.0 extra. This bounded three-data-qubit
code-capacity example is not a threshold study. The default source must have
explicit Z records and a default CPU PyTorch initial state. Fixed Clifford
gates, resets, single-slot X feedback and one-/two-qubit Pauli channels are
supported. T, conditional H, general noise and unsupported exact DEM analysis
refuse; there is no approximation or SV fallback. A channel may be sampleable
but not convertible to an exact independent-mechanism DEM.

The approved contract is [QEC native detector execution](../../../docs/development/QEC_NATIVE_DETECTOR_EXECUTION_PROPOSAL.md).
Output copies and conservatively padded tables are individually bounded at
64 MiB, exact-analysis support at one million cells and DEM text at 16 MiB.
These are allocation/work guards, not an RSS or performance claim. Compiled
sampling reuses lowered source and RNG, not frame scratch. No detector file
writer, MemoryCircuit backend replacement, new builder, SDK source integration
for PyMatching, or broad parity certification is implied.


### Detector-file workflow

`Circuit.compile_detector_sampler` owns maps and a reusable execution handle;
Simulation owns the source lowering, reference normalization and numerical
writer. The approved `sample_write` candidate streams six formats through the
same private stream boundary as `sample`, with no new registry or backend.
QEC validates layout and lifecycle; the shared private file validator checks
framing before no-replace publication. Same-call observable output preserves
correlation. Upstream files remain unchanged.

The contributor golden path is `examples/qec/native_detector_sampling.py`;
its deterministic file witness prints `detector_bytes 64`. Start with
`tests/qec/test_native_detector_execution.py`, then numerical replacement tests
in `tests/team/simulation/test_native_detector_execution_conformance.py`.
See `docs/development/QEC_DETECTOR_FILE_OUTPUT_PROPOSAL.md` for capacities,
publication limits and installed-wheel acceptance. No public file reader,
research execution, universal Clifford-control support or RSS cap is added.

## Convert captured measurement files

The existing compiled converter also exposes `convert_file()`. Both terminal
and explicit factories reuse their prepared source/reference. QEC owns maps and
the public request; Simulation owns snapshot validation, numerical conversion
and file handles. No public file-reader or execution result type is added.

Run `python -m examples.qec.measurement_file_conversion` with the embedded wheel
or Stim 1.16.0. It prints `shots 64`, `detectors 10; observables 1` and
`file_bytes 64 64`. Start contributor changes with
`python -m pytest tests/qec/test_measurement_file_conversion.py -m unit -q`.
See the approved [file conversion proposal](../../../docs/development/QEC_MEASUREMENT_FILE_CONVERSION_PROPOSAL.md)
for snapshot, allocation, no-replace publication and refusal contracts.

## Shot-file reading

The public record I/O composition lives in
`flagquantum.qec.execution.shot_data`; Simulation's private `_shot_data.py` owns
snapshot admission, count/allocation checks and codec access. The embedded
CPython cursor reuses the locked Stim reader; it introduces no numerical
algorithm, decoder, scheduler or public SDK object. The standard build supports
eager reading through pinned Stim; iteration requires the embedded build.
Run `python -m pytest tests/qec/test_shot_data_reading.py -q` for fake-contract
checks and `python -m pytest tests/team/simulation/test_shot_data_reading_conformance.py -q`
for byte and numerical conformance. The ten-minute user path is
`python examples/qec/shot_file_decoding.py` in an embedded + PyMatching profile.

## Write caller-owned shot data

The approved writing candidate adds `write_shot_data_file` beside the two
readers. QEC owns the public signature; Simulation validates one owned bounded
copy, chooses the pinned SDK or embedded codec, validates staged output and
publishes without replacement. Run `python -m examples.qec.shot_file_encoding`
with Stim 1.16.0 or an embedded build. Fake, fixed-byte and installed
acceptance are tracked in the proposal; no memory-mapped, append or streaming
writer is included.

## Reusable independent DEM sampling

`model.compile_sampler(seed=7)` returns a CPU `CompiledDemSampler` from
`flagquantum.qec.execution.dem_sampling`. `sample(shots, bit_packed=True,
return_errors=True)` returns detector, observable and error tables. Error column
`i` follows canonical `model.errors[i]`, including repeated signatures; it does
not follow the insertion order of constructor arguments. Replay accepts bool or
packed uint8 independently of output packing and ignores unused input padding.
Returned padding is zero and arrays own storage.

`sample_write` streams optional detector, observable and error outputs in `01`,
`b8`, `r8`, `ptb64`, `hits` or `dets`. Error files use M labels, detector files D
and observable files L. PTB64 requires a multiple of 64 shots. Zero shots produce
empty results without advancing RNG. Files never overwrite destinations; three
outputs publish sequentially, with owned-inode rollback on failure rather than
cross-file atomic visibility. Replay is validated on a bounded immutable snapshot.

The 64 MiB principal-data guard includes all native tables, result and replay
copies; it is not process RSS or model/parser heap. Each dimension is at most
65,536, targets at most 262,144, canonical text at most 1 MiB, aggregate final
outputs at most 1 GiB, and replay snapshots separately at most 1 GiB. Native
stripes have at most 1024 shots. Handles belong to the creating thread/process,
cannot be serialized, and invalidate after execution/publication failure.
Validation failures preserve the handle. No backend fallback is performed.

Every stated `error_id` is refused before preparation, preserving the existing
independence boundary. Use unchanged `dem_sampling()` for exclusive groups.
Seed reproducibility requires the same backend/build/SIMD and call partition;
SDK/native/platform byte equality is not promised for random draws. Exact replay
and deterministic mechanisms are the cross-implementation witnesses.

Run `python -m examples.qec.compiled_dem_sampler` with the existing PyMatching
extra and a Stim 1.16.0 SDK or embedded build. Expected output is
`shots 64 batches 4 logical_failures 0`. macOS arm64 offline wheel/sdist
builds and installed native acceptance passed, including an environment without
the Stim Python package. See the implementation acceptance record in
`docs/development/QEC_COMPILED_DEM_SAMPLING_PROPOSAL.md`. This bounded
capability makes no performance, threshold, full-parity or scalability claim.
