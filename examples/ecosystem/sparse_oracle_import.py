"""Convert optional source objects once, then append a native sparse block."""

import torch

from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
    HermitianSparseMatrixBlockEncoding,
    UniformSqrtEntryOracle,
)
from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
    TopLeftRowColumnOracle,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._sparse_oracles import (
    import_column_oracle,
    import_sqrt_entry_oracle,
)


def build_circuit() -> DynamicCircuit:
    """Import a full-support two-by-two example with independent source models."""
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceColumn,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        UniformSqrtEntryOracle as SourceEntry,
    )

    column_model, entry_model = (
        TopLeftRowColumnOracle(1),
        UniformSqrtEntryOracle(1, 0.3),
    )
    column = import_column_oracle(
        SourceColumn(1),
        models=(
            column_model.resource_model(),
            column_model.resource_model(adjoint=True),
        ),
    )
    entry = import_sqrt_entry_oracle(
        SourceEntry(1, 0.3),
        models=(entry_model.resource_model(), entry_model.resource_model(adjoint=True)),
    )
    block = HermitianSparseMatrixBlockEncoding(column, entry)
    assert block.requirements() == (0, 0)
    operation = block.make_on(system=(0,), ancilla=(1, 2, 3))
    circuit = DynamicCircuit(4, dtype=torch.complex128)
    circuit.append(operation)
    return circuit


if __name__ == "__main__":
    circuit = build_circuit()
    assert circuit.n_qubits == 4
    print("Converted source oracles and appended a native four-qubit block.")
