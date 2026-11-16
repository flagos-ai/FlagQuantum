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

The locked original notebook runs in an independent kernel. The remaining
original suite has 34 passes, three symbolic decomposition skips and three
symbolic serialization expected failures. Q-0553 remains open for typed
Program binding of signed, bounded and Montgomery encodings, explicit
Partition/And nodes in the typed Program, and per-control source wire symbols
in the native drawer. Signed integer register groups already have exact
physical wiring and symbolic layouts are admitted for metadata.
