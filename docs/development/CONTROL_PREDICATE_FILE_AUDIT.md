# Grouped equality predicate source audit

The reference is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native audit hashes these
unchanged files before loading the source factories.

| File | Frozen Git blob | Scope |
| --- | --- | --- |
| `ctrl_spec_and.ipynb` | `e96d6fa66990fc52bb679a7892edef94d6fcd3c9` | Q-0552, all original tutorial sections |
| `ctrl_spec_and.py` | `3fa7edd03c6f488edc5b11358c5aa22819ccef5b` | Q-0553, signature, grouped encoding, predicate and resource declarations |
| `ctrl_spec_and_test.py` | `f0f8638db9882c8c75f598effba197f80b927e29` | Q-0554, seven factories, three refusals and both truth tables |

## Signature and execution

Each control group passes through without a value change. The shared
signature retains its ordered integer type and array shape. Flattening uses
row-major group order and big-endian word order. Compute owns `n-1` clean
prefix wires and returns them as `junk` and `target`. Binary compute returns
only `target`. The measured adjoint consumes the returned outputs and clears
all prefix wires in reverse order. The flags for each measurement live in
caller-owned classical slots.

The [native notebook](../../examples/notebooks/control_predicate.ipynb)
retains all six tutorial fixtures, the grouped signature figure and the
depth-one call graph. It also shows the seventh, symbolic-width source
factory and a compute/erase example through `circuit.append(op)`.

The existing predicate emitter and resource model remain the shared execution
and cost owners. The new signature helper also serves the typed Program
builder. Gate tests compare primitive identities, operands, per-wire order,
measurement conditions and returned bindings for all four concrete source
factories. The three four-control factories check every computational input
in a coherent vector and all eight erasure histories. The sixteen-control
mixed-integer factory checks sparse complex amplitudes on four inputs and
three erasure histories. Both original truth tables compare all 16 and 256
input rows, including the retained junk output.

## Emission and declared resources

| Concrete tutorial group | Emitted T/T† in compute | Emitted erasure measurements | Modeled And calls in compute | Modeled peak qubits |
| --- | ---: | ---: | ---: | ---: |
| Unsigned word, flat bits or 2 × 2 bits | 12 | 3 | 3 | 7 |
| Mixed unsigned and signed words | 60 | 15 | 15 | 31 |

All seven examples match source Clifford, And, measurement and peak width
queries. The symbolic binary child retains the explicit `cv1` and `cv2`
symbols. Symbolic widths and polarities have metadata and resource formulas.

## Complete definition and dependency map

| Source definitions | Native ownership and evidence |
| --- | --- |
| `CtrlSpecAnd`, `__attrs_post_init__` | Existing integer control validation and canonical concrete storage admission. |
| `signature`, `control_registers`, `junk_registers`, `n_ctrl_qubits` | Grouped signature owner and symbolic width/resource metadata. |
| `_ctrl_partition_bloq` | Shared `partition_component_program` for both directions. |
| `_flat_cvs` | Existing `ControlRegister` row-major MSB-first codecs and symbolic ControlPattern metadata. |
| `build_composite_bloq` | Typed Partition inverse, Split, conjunction, Join, Partition and junk Join calls match original connections. |
| `wire_symbol` | `control_predicate_wire_symbol` delegates grouped controls and the existing adjoint symbol protocol. |
| `build_call_graph` | Existing ControlPredicateCall/QEC authority selects binary And or MultiAnd. |
| Seven `_ctrl_on_*` factories | Original tutorial and source test fixtures, including symbolic metadata models. |

All eighteen definitions in the frozen file are mapped. Q-0553 remains open
for symbolic activation graph construction.
The shared typed conjunction owner retains each And call and directional target
ownership. Its primitive leaves reuse the existing physical emitter. The
previous tutorial-local graph construction is removed in favor of this owner.
Concrete signed, ones'-complement, bounded unsigned and Montgomery controls now
cross the same typed Program path and preserve their original descriptors.
No codec or physical gate sequence is duplicated.

The unchanged upstream suites complete with 45 passes, three source skips and
three expected source failures. Their notebook runs in the existing reference
kernel environment. Its unregistered notebook-marker warning is retained.
Symbolic counts/activations have source-matching metadata and QEC models.
For concrete widths with unknown activations the original additionally builds
a symbolic And/MultiAnd graph. The native typed predicate still rejects those
values before graph construction. `g2-integer-storage-symbolic-boundary.json`
records this remaining implementation gap. No finite physical stream is
claimed for unresolved activation values. The original symbolic serialization failures remain
visible. Physical counts and model counts are separate in
`g2-integer-storage-resources.json`.

Original and native root connections match for every supported encoding in
both directions. Full primitive ordering and feedback, small complete complex
states, every measurement history, exact inverse history pairs and clean work
are covered by the concentrated regression. Full six-control state comparison
covers 64 single-pass histories; larger source tutorial fixtures reuse sparse
amplitude and exact stream evidence. Predicate glyphs match every original
port and direction, including daggered text boxes under the generic adjoint.
