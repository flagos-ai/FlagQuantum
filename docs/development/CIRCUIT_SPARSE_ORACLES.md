# Sparse oracle snapshots and the optional source boundary

The Hermitian sparse matrix operator accepts the existing RowColumnOracle and
SqrtEntryOracle structural interfaces. Algorithms now provides CircuitColumnOracle
and CircuitSqrtEntryOracle for user-owned CircuitIR snapshots. They carry separate
forward and adjoint circuits and independent ComponentCost declarations. Neither
provider retains an external dependency object or reverses measured instructions
to guess an inverse.

Snapshot wire order is ordinal, index, clean work for a column, and flag, row,
column, clean work for an entry. Each direction's work and classical prefix are
validated independently. The public requirements reserve the larger direction;
emission uses only its own prefix through the shared primitive binder. Caller
work is disjoint from logical operands, returns to zero under the oracle's clean
work contract, and uses explicit caller result slots. The same Hermitian
consumer, resource query and `circuit.append(op)` workflow use either built-in
or snapshot providers.

The optional `ecosystem/cirq/_sparse_oracles.py` entry points convert actual
concrete source objects immediately into these native providers. They accept
the original scalar l/i or q/i/j ports, require both supplied source cost models,
read the actual source adjoint and preserve entry precision. They use the
external builder and its decomposition context with a tracked clean allocator.
No gate table is selected by a reference class name. Unsupported primitives,
dirty borrows, symbolic widths, extra or shaped ports, undeclared external wires
and aliased outputs are rejected rather than guessed.

The pinned generic gate wrapper rejects a same-name LEFT/RIGHT l register. The
column importer uses the unchanged source Composite exporter for that signature;
other signatures use as_cirq_op. This is a source API limitation, not a patched
reference. The resulting primitive circuit passes through the already verified
feedback importer. It retains measurement histories and native Boolean dataflow.

Returned output wires are checked explicitly. A pure port permutation is
canonicalized by Compiler's restore_output_order_ir using the shared atomic
SWAP owner. A newly returned, explicitly allocated output is similarly moved
into the caller's original logical port, and the remaining clean storage stays
in workspace. Unknown outputs are never assumed zero. Both directional binding
swap counts are exposed separately; the provider composes their exact Clifford
cost with the supplied source body model. No source model is inferred from
primitive T counts. A physical binding SWAP is separate from a source's free
logical wire rename.

```python
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._sparse_oracles import (
    import_column_oracle, import_sqrt_entry_oracle,
)
from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
    HermitianSparseMatrixBlockEncoding,
)

# source_column/source_entry are concrete dependency objects; these four models
# are independently declared native ComponentCost values, not primitive counts.
column = import_column_oracle(source_column, models=(column_cost, column_inverse_cost))
entry = import_sqrt_entry_oracle(source_entry, models=(entry_cost, entry_inverse_cost))
block = HermitianSparseMatrixBlockEncoding(column, entry)
# Bind caller storage as with the built-in block, then append the bound operation.
```

For a runnable native example and the complete optional conversion proof:

```bash
python -m examples.ecosystem.sparse_oracle_import
pytest -q tests/unit/test_circuit_sparse_oracles.py tests/unit/test_foreign_sparse_oracles.py
```

The original uniform objects reproduce the full complex block matrix and source
cost model. Independently defined nonuniform objects reproduce the existing
native per-wire primitive stream, independent source estimates and the complete
source tensor action at widths one and two, with and without positive control.
All four single-step and sixteen forward/adjoint histories match, with clean
work, spectator entanglement and an occupied prior result slot. Pure returned
aliases and newly allocated outputs have separate full complex-state and binding
cost tests. Dirty, unknown and overlapping storage has rejection evidence.

The snapshot contract cannot prove an arbitrary user's oracle is reversible or
cleans its declared workspace. Those are the same semantic obligations as a
hand-written native emitter; the reference fixtures verify them explicitly.
Symbolic source declarations remain resource-only, as in the unchanged source.
