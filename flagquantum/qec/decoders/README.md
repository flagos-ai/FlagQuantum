# Decoder algorithms and registration

Core owns the existing repetition-history protocols, in-tree decoder algorithms
and the single detector-model decoder registry. This package does not own code
construction, DEM text parsing, circuit execution, SDK imports or provider tasks.

| Module | Responsibility |
| --- | --- |
| `repetition.py` | Existing `Decoder`/`StreamingDecoder` protocols and lookup/temporal repetition decoders. |
| `matching.py` | Exact in-tree matching and its result record. |
| `belief_propagation.py` | Existing factor-graph decoder and result records. |
| `bposd.py` | Ordered-statistics decoder over the original model. |
| `sliding_window.py` | Existing windowed matching and chunk-stream decoder/state. |
| `registry.py` | Existing DEM decoder protocols and sole name table; duplicate/replacement and graph-carrier policy. |
| [`_ai/`](_ai/README.md) | Private offline neural predecoding, training and paired assessment; no public exports or registry entries. |

Allowed dependencies are standard-library utilities, PyTorch, QEC algebra,
model/graph/chunk records and the optional adapter's class. The registry imports
that class without importing its SDK. No new protocol or second registry is
introduced. Code-independent and repetition-history protocols retain their
distinct established meanings.

Public users can keep importing from `flagquantum.qec`; the canonical algorithm
modules are available for expert imports. `decoders/__init__.py` retains the
surviving former `qec.decoders` exports and pickle identifiers. Registration
through root or `decoders.registry` reaches the same table. Windowed defaults
remain lazy and never overwrite an explicit plugin replacement.

The approved 0.4.0 retirement removes flat decoder/registry imports and their
pickle lookups. Moved algorithms, methods, the registry TypeVar and returned
decorator now use canonical module metadata. New in-progress stream checkpoints
resume unchanged; surviving repetition-decoder pickle identifiers remain valid.
No unpickling bridge, custom reducer or second registry is introduced.

For a ten-minute change, run `PYTHONPATH=. python
examples/qec/decoder_registry_and_pickle.py`, edit the owning algorithm, and run
its existing test file plus `tests/qec/test_restructure_decoders.py`. The example
needs no SDK: three decoders predict observable index `(0,)`, the restored matcher
class is `flagquantum.qec.decoders.matching`, and five default names remain registered.
Never refresh the frozen 001–006 fixtures to accommodate a relocation.
