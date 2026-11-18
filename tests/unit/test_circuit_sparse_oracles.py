"""Owned snapshots replace oracle callbacks without changing the consumer."""

import pytest

from examples.algorithms.custom_sparse_oracles import custom_hermitian_block
from flagquantum.algorithms.primitives.circuit_sparse_oracles import (
    CircuitColumnOracle,
    CircuitSqrtEntryOracle,
)
from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
    HermitianSparseMatrixBlockEncoding,
)
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def snapshot_block(width: int) -> HermitianSparseMatrixBlockEncoding:
    original = custom_hermitian_block(width)
    column, entry = original.column_oracle, original.entry_oracle
    columns = tuple(
        CircuitIR(
            2 * width,
            column.emit(
                tuple(range(width)),
                tuple(range(width, 2 * width)),
                (),
                (),
                adjoint=direction,
            ),
        )
        for direction in (False, True)
    )
    entries = tuple(
        CircuitIR(
            2 * width + 2,
            entry.emit(
                0,
                tuple(range(1, width + 1)),
                tuple(range(width + 1, 2 * width + 1)),
                (2 * width + 1,),
                (0,),
                adjoint=direction,
            ),
        )
        for direction in (False, True)
    )
    return HermitianSparseMatrixBlockEncoding(
        CircuitColumnOracle(
            width,
            1 << width,
            columns[0],
            columns[1],
            (column.resource_model(), column.resource_model(adjoint=True)),
        ),
        CircuitSqrtEntryOracle(
            width,
            entries[0],
            entries[1],
            (entry.resource_model(), entry.resource_model(adjoint=True)),
        ),
    )


@pytest.mark.parametrize("width", (1, 2))
@pytest.mark.parametrize("controlled", (False, True))
def test_snapshots_preserve_complete_callback_stream_and_independent_model(
    width: int, controlled: bool
) -> None:
    original, imported = custom_hermitian_block(width), snapshot_block(width)
    shift = int(controlled)
    through = 2 * width + 2 + shift
    args = dict(
        system=tuple(range(shift, shift + width)),
        ancilla=tuple(range(shift + width, through)),
        control=0 if controlled else None,
        workspace=(through,),
        classical_bits=(7,),
    )
    assert imported.requirements() == original.requirements()
    assert (
        imported.make_on(**args).instructions == original.make_on(**args).instructions
    )
    assert imported.resource_model(controlled=controlled) == original.resource_model(
        controlled=controlled
    )


def test_adjoint_snapshot_has_its_own_storage_instead_of_reversed_measurements() -> (
    None
):
    original = snapshot_block(1).entry_oracle
    backward = CircuitIR(3, (Instruction("ry", (0,), {"theta": -0.37}),))
    entry = CircuitSqrtEntryOracle(1, original.forward, backward, original.models)
    assert entry.requirements() == (1, 1)
    assert entry.emit(3, (4,), (5,), (6,), (7,), adjoint=True) == (
        Instruction("ry", (3,), {"theta": -0.37}),
    )
    with pytest.raises(ValueError, match="declared work"):
        entry.emit(3, (4,), (5,), (), (), adjoint=True)
    with pytest.raises(ValueError):
        entry.emit(3, (4,), (5,), (3,), (7,), adjoint=True)
