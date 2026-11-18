"""General amplitude oracles retain their source convention and quantization."""

from hashlib import sha1
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from flagquantum.algorithms.primitives.sparse_matrix_block_encoding import (
    SparseMatrixBlockEncoding,
)
from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
    ExplicitEntryOracle,
    TopLeftRowColumnOracle,
    UniformEntryOracle,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._sparse_oracles import (
    import_column_oracle,
    import_entry_oracle,
)
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.block_encoding import sparse_matrix

    raw = Path(sparse_matrix.__file__).read_bytes()
    assert (
        sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        == "cacbdab733f0be4f0f28bf2a749e1bad79987bfd"
    )


@pytest.mark.parametrize("width", (1, 2))
@pytest.mark.parametrize("adjoint", (False, True))
def test_uniform_foreign_amplitude_full_matrix_projection_and_cost(
    width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import SparseMatrix
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceColumn,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        UniformEntryOracle as SourceEntry,
    )

    index, entry = TopLeftRowColumnOracle(width), UniformEntryOracle(width, 0.3)
    sc, se = SourceColumn(width), SourceEntry(width, 0.3)
    column = import_column_oracle(
        sc, models=(index.resource_model(), index.resource_model(adjoint=True))
    )
    imported = import_entry_oracle(
        se, models=(entry.resource_model(), entry.resource_model(adjoint=True))
    )
    assert not hasattr(se, "epsilon")
    assert imported.requirements() == (0, 0)
    native = SparseMatrixBlockEncoding(column, column, imported)
    source = SparseMatrix(sc, sc, se, eps=0)
    assert_source_model(
        native.resource_model(adjoint=adjoint), source.adjoint() if adjoint else source
    )
    op = native.make_on(
        system=tuple(range(width)),
        ancilla=tuple(range(width, 2 * width + 1)),
        adjoint=adjoint,
    )
    matrix = get_unitary(
        CircuitIR(2 * width + 1, op.instructions), dtype=torch.complex128
    )
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        matrix, expected.mH if adjoint else expected, atol=5e-12, rtol=0
    )
    dimension = 1 << width
    projection = matrix.reshape(dimension, 2 * dimension, dimension, 2 * dimension)[
        :, 0, :, 0
    ]
    torch.testing.assert_close(
        projection, torch.full_like(projection, 0.3 / dimension), atol=5e-12, rtol=0
    )


@pytest.mark.parametrize("adjoint", (False, True))
def test_explicit_foreign_entry_retains_original_quantized_complex_action(
    adjoint: bool,
) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        ExplicitEntryOracle as SourceEntry,
    )

    data = ((0.0, 0.25), (1 / 3, 0.467))
    original = ExplicitEntryOracle(1, data, 2)
    source = SourceEntry(1, np.asarray(data), 2)
    imported = import_entry_oracle(
        source,
        models=(original.resource_model(), original.resource_model(adjoint=True)),
    )
    assert_source_model(
        imported.resource_model(adjoint=adjoint),
        source.adjoint() if adjoint else source,
    )
    work, slots = imported.requirements()
    gates = imported.emit(
        0,
        (1,),
        (2,),
        tuple(range(3, 3 + work)),
        tuple(range(4, 4 + slots)),
        adjoint=adjoint,
    )
    values = torch.arange(16, dtype=torch.float64)
    state = torch.complex(values + 1, values % 5 - 2).reshape(8, 2)
    state /= state.norm()
    result = torch.zeros_like(state)
    words = (np.arccos(np.asarray(data)) / np.pi * 4).astype(int)
    for flag, row, column in product((0, 1), repeat=3):
        theta = 2 * np.pi * int(words[row, column]) / 4 * (-1 if adjoint else 1)
        for target in (0, 1):
            coefficient = (
                np.cos(theta / 2)
                if target == flag
                else np.sin(theta / 2) * (1 if flag == 0 else -1)
            )
            result[target * 4 + row * 2 + column] += (
                coefficient * state[flag * 4 + row * 2 + column]
            )

    def extend(v: torch.Tensor) -> torch.Tensor:
        a = torch.zeros((8, 1 << work, 2), dtype=torch.complex128)
        a[:, 0, :] = v
        return a.flatten()

    count = sum(g.name == "measure" for g in gates)
    assert count <= 10
    for outcomes in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(extend(state), gates, outcomes),
            extend(result) / (2 ** (count / 2)),
            atol=6e-12,
            rtol=0,
        )
