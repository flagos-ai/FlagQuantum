# Cirq register export source audit

The locked reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The Cirq package, exporter,
exporter tests, and tutorial blobs are pinned in
`tests/test_cirq_register_export.py`.

## Native ownership

The optional Cirq boundary places a validated physical circuit onto explicit
external qubits. `export_register_cirq` takes the native signature and the
physical wires of each port. It retains each register's shape with the bit
axis last. Supplied left and through ports retain their qubits. Right-only
ports receive fresh names, and the result reports right and through ports in
signature order. Unbound temporary wires remain separate from the ports.
The exporter remaps the result of the existing static converter, so operation
order, measurement key grouping and conversion diagnostics use the same
implementation. No external qubit object enters native IR.

The static exporter explicitly rejects instructions with classical conditions.
This also covers conditions referring to inputs with no preceding measurement.
An explicit lossy export reports and drops such an operation instead of
turning it into an unconditional gate.

| Source responsibility | Native support |
| --- | --- |
| Source port order, qubit arrays and named bits | Concrete quantum register layouts, row-major words and explicit physical wire binding. Qid dimensions, aliases, shape and ownership are checked. |
| Through, left and right port placement | Through ports preserve identity, left-only ports are excluded from output, and fresh right names cannot capture input or scratch. |
| Component conversion and graph order | The adapter consumes native physical IR. Native append and Compiler own decomposition and allocation before conversion. |
| Source Cirq gate wrappers, tensor fallback and allocation | Arbitrary external gate objects, tensor-only gates and source clean/dirty reuse are separate from native physical placement and remain open. |
| Diagram symbols | The exported primitive circuits use Cirq's native diagram rendering for exchange and Toffoli. |

## Verification

The locked original suite including its tutorial passed 16 cases. Native tests
compare original one- through five-bit word swaps, the named two-bit diagram,
a square of the wrapper's swap, and both Toffoli diagram modes. The full
unitaries use a caller-selected qubit order, including original disjoint
external line ranges. A separate shaped register checks the ordered
terminal measurement operands and exact key.

The original four-control MultiAnd produces seven through/result wires. Native
execution matches the full original Cirq result, including the two logical
junk values and target. Its literal inverse phase primitives still trigger
the existing adapter diagnostic. The native measured compute/uncompute
restores a coherent input and zero scratch. Its feedback remains explicitly
outside static export. These tests preserve the distinction between source
operation support and adapter admission.

The [native tutorial](../../examples/notebooks/cirq_register_interop.ipynb)
uses the original seeded Cirq circuit, rewrites its iSWAP into exact S, CZ and
SWAP operations, and checks the complete matrix through import and export.
It demonstrates named placement, shaped registers, fresh output placement and
append. It keeps the original modular exponentiation dimensions and call
declaration, and runs the adapter diagnostic on a smaller native compiled
instance. The original five-qubit composite QFT, exact MultiAnd primitive
export and modular exponentiation roundtrip remain outside the current static
contract.

## Remaining files

- Q-0714 awaits the rest of the source package export map, including
  import, decomposition, and named-wire management.
- Q-0715 and Q-0716 retain source graph wrapper, tensor fallback, allocation
  lifetime reuse and dynamic arithmetic export coverage. Current named
  placement assigns the wires the native circuit already owns.
- Q-0721 retains the source composite QFT, MultiAnd export and ModExp
  roundtrip as open tutorial sections. The adaptation shows their diagnostics.
