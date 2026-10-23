# Optional decoder adapters

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
the random count is version/machine scoped. Sampler compilation still occurs
per chunk, so this is not a persistent-sampler or performance claim.
Keep graph sizing, refusals, weights and observable labels aligned with those
existing independent cross-check tests.
