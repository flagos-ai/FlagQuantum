# Zero-state register tensor nodes (GRAPH61)

`simulation/tensor_network/register_zero_state.py` supplies the existing
TensorNetworkNode representation with one `[1, 0]` vector per output bit.
It follows unchanged Qualtran v0.7.0 ZeroState tensor data, pinned at
bcb17bb25360ed68ce3bc084645db3e929799389, and connects to the
[zero-input wrapper](COMPONENT_ZERO_INITIALIZATION.md) through GRAPH60 wiring.

`register_zero_state_nodes(signature, outgoing, like)` accepts pure-output
quantum ports and existing row-major/MSB-first bit-label maps. It reuses the
existing register axis binder. Dtype and device come from `like`; allocation is
two scalar entries per bit, with no exponential dense word vector. Repeated
labels, input/THRU ports and classical ports are rejected. Missing keys and
shape errors retain binder validation. Zero-width/empty ports return no nodes;
no scalar identity is inferred for an empty graph.

The numerical supplier does not allocate physical wires, reset data, select
components or choose an execution backend. Compiler owns graph initialization;
the caller explicitly identifies its zero-state leaves. Other leaf suppliers
remain replaceable without changing the graph-to-tensor consumer.

Tests compare individual ket data against actual unchanged ZeroState tensors,
check four real/complex PyTorch dtypes, preserve device and differentiate
attached covectors. Six positive-width/nonempty phase-array layouts compare
complete complex states against the original initialized composite contraction.
Across all twelve layouts, the initialized `circuit.append(op)` state agrees
with native contraction after explicit output-wire ordering, with an extra
unused clean wire remaining zero. Empty-array and zero-width source failures
are tested explicitly; analytical phase states verify the corresponding native
behavior separately.

A replacement phase supplier preserves consumer code and differentiable complex
amplitudes. Initialized measured two-Add execution checks sixteen forward
measurement branches, all 256 inverse history pairs and eight append trajectories
with exact complex amplitudes and clean work. The existing arbitrary joint-input
source/gate/feedback test is also retained so zero-only execution is not the sole
measurement check.

Numerical ket-node counts, actual emitted primitives and original QEC model
categories are separate. Clean preparation emits no primitive gates; original
IntState has a coarse ArbitraryClifford model, which must not be equated to that
empty preparation stream. Full resource evidence records both original
initialized and uninitialized model categories and source peak width.

Run in the locked reference environment:

```bash
python -m pytest tests/unit/test_register_zero_state.py -q
python -m examples.algorithms.binary_field_arithmetic
```

The existing example still uses ordinary appended operations and prints GF(8)
multiplication plus clean inverse and measured-erasure results. This internal
numerical inspection capability does not replace the circuit execution backend.

## Final verification

Simulation implementation `05c6455b` has thirteen statements and four branches
at 100% coverage. Its separate Compiler wrapper is `c55f7b12`. All 33 new tests,
916 related tests, 221 focused tests and 64 integration tests passed. Sixteen
files passed strict typing, formatting and linting; architecture, both ownership
scopes, Program and protected API checks passed. Twelve source blobs match the
locked inventory, and seven unchanged upstream tests passed. Full numerical,
resource and environment evidence is summarized in
[the initialization verification](COMPONENT_ZERO_INITIALIZATION.md#final-verification).

The default suite has seven failures matching the existing seven-case baseline,
with 21,507 passed and 300 skipped. No new failure ID was observed; maxfail means
this is not a complete or green default run. Original empty-array initialization,
zero-bit IntState decomposition and zero-bit peak-estimate failures remain
negative source evidence, not rewritten references or accepted source parity.
