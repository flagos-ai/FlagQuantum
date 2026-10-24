# Optional QEC adapters

Core owns the existing PyMatching cross-check boundary. `pymatching.py` translates
the QEC-owned decoding graph, invokes the optional SDK and translates its answer
back into the existing result record. It does not own matching algorithms, code
families, provider tasks, circuit execution or the decoder registry.

Allowed dependencies are the standard library, QEC model/graph/result records and
the SDK imported only when the adapter is constructed. Importing QEC or listing
registered decoders must work without PyMatching. Selecting the adapter without
the SDK raises the existing `MatchingDependencyError` with installation guidance;
there is no fallback or automatic dependency installation.

Public entry points remain `flagquantum.qec.PyMatchingDecoder` and
`flagquantum.qec.adapters.{PyMatchingDecoder, MatchingDependencyError}`. The package
initializer is an explicit alias boundary. Canonical and historical imports use
the same classes and helpers, with historical class/method/function pickle paths.
The SDK's native graph remains non-pickleable in the verified PyMatching 2.4.0
profile; relocation adds no reducer or new serialized representation.

Core retains these surviving `qec.adapters` identifiers under the approved
0.4.0 contract; this package is outside the 23 retired flat module paths. Source-inspection tools
should follow `pymatching.py`, not infer a source filename from retained metadata.

Proposal 087 adds `PyMatchingDecoder.decode_batch(events)` for logical-only
prediction: a dense CPU binary bool/integer tensor `(B, D)` yields a caller-owned
CPU bool tensor `(B, L)`. Noncontiguous input is copied. Empty batches retain L;
zero-observable models still undergo reachability checks. An undecodable batch
raises `CapabilityError` atomically. The method reuses the existing SDK graph,
calls the SDK once per nonempty batch, and does not recover edge paths.
The optional SDK is still imported only at construction. Existing single-shot
decoding and registry protocols are unchanged. The approved additive contract is
`contracts/qec-batch-prediction-v1.json`; restructuring fixtures remain immutable.

For a ten-minute change, install the declared `pymatching` extra, then run
`python -m pytest tests/qec/test_adapters_pymatching.py
tests/qec/test_restructure_decoders.py -q`. Build a small `DetectorErrorModel` and
call `get_decoder("pymatching", model).decode((0,))` through the existing root API.
The example in `examples/qec/decoder_registry_and_pickle.py` provides the core
model. `PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py` uses both
extras to sample and batch-decode 128 repetition-memory shots in chunks. It
reports 8 detector columns, 1 observable and the observed logical failure count;
the random count is version/machine scoped. It compiles the existing sampler
once and advances its stream across chunks; this is not a performance claim.
Keep graph sizing, refusals, weights and observable labels aligned with those
existing independent cross-check tests.

## Annotated Stim experiments

The approved opt-in `flagquantum.qec.adapters.stim` module provides
`memory_to_stim_text`, `compile_stim_sampler` and `stim_circuit_to_dem_text`.
QEC owns native placement and parity maps. Simulation alone imports Stim and
owns parsing, actual Kraus translation, reference-normalized sampling and exact
DEM analysis. No SDK objects, new result classes or root exports are added.
The returned `MemorySampler` retains the existing process/thread ownership,
bounded draws, failure poisoning and close contract.

Install `pip install -e '.[stim,pymatching]'` and run
`PYTHONPATH=. python examples/qec/stim_experiment_bridge.py`. Both native memory
export and external annotated text feed the same logical-only decoder workflow.
Always sample the original experiment text; DEM decomposition suggestions are
for decoding and cannot substitute for the original circuit distribution.

The initial support limits are 1 MiB UTF-8 input, 4,096 qubits, 1,000,000 expanded
instructions and 1,000,000 combined measurement/detector/observable columns.
Noiseless detector/logical parities must be deterministic; nonzero ideal raw
measurements are normalized to flips. Exact DEM refusal does not prevent
otherwise supported circuit sampling. Missing shape declarations are supplied
without changing the existing DEM reader; source/canonical text preserves
coordinates while the owned numerical DEM gains no geometry fields.

Run `python -m pytest tests/qec/test_stim_experiment_bridge.py
tests/team/simulation/test_stim_annotated_engine.py tests/qec/test_memory_sampler.py
tests/team/simulation/test_prepared_stabilizer_sampling.py -q` after a typical
change. The approved surface is in
`contracts/qec-stim-experiment-bridge-v1.json`; the complete contract and numerical
precision limitations are in
`docs/development/QEC_STIM_EXPERIMENT_BRIDGE_PROPOSAL.md`.
