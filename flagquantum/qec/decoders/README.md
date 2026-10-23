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

Allowed dependencies are standard-library utilities, PyTorch, QEC algebra,
model/graph/chunk records and the optional adapter's class. The registry imports
that class without importing its SDK. No new protocol or second registry is
introduced. Code-independent and repetition-history protocols retain their
distinct established meanings.

Public users can keep importing from `flagquantum.qec` and the historical flat
modules. `decoders/__init__.py` preserves the former `qec.decoders` exports.
The old `registry.py` aliases the canonical table and functions; registration
through either historical or root imports reaches that one table. The windowed
default remains lazy and never overwrites an explicit plugin replacement.

Classes, methods, functions, the registry TypeVar and returned decorator retain
their historical module/qualified names. Old instance pickles, including an
in-progress stream, keep loading without wrappers or new reducers. Core owns
these aliases; no removal release is approved. Removal needs an approved
deprecation and artifact migration plan. Source tools and private global
replacement must use the physical owner, rather than the retained pickle module.

For a ten-minute change, run `PYTHONPATH=. python
examples/qec/decoder_registry_and_pickle.py`, edit the owning algorithm, and run
its existing test file plus `tests/qec/test_restructure_decoders.py`. The example
needs no SDK: three decoders predict observable index `(0,)`, the restored matcher
class remains `flagquantum.qec.matching`, and five default names remain registered.
Never refresh the frozen 001–006 fixtures to accommodate a relocation.
