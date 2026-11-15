"""Compare the frozen sparse-access block and oracle costs and amplitudes."""

from itertools import product
from math import sqrt
from typing import Literal

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.sparse_matrix_block_encoding import (
    SparseMatrixBlockEncoding,
)
from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
    ExplicitEntryOracle,
    SymmetricBandedRowColumnOracle,
    TopLeftRowColumnOracle,
    UniformEntryOracle,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ["uniform", "explicit", "banded"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_sparse_block_models_and_source_amplitudes(
    kind: Literal["uniform", "explicit", "banded"],
    adjoint: bool,
) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        _explicit_matrix_block_encoding,
        _sparse_matrix_block_encoding,
        _symmetric_banded_matrix_block_encoding,
    )

    if kind == "uniform":
        frozen = _sparse_matrix_block_encoding()
        oracle = TopLeftRowColumnOracle(2)
        native = SparseMatrixBlockEncoding(oracle, oracle, UniformEntryOracle(2, 0.3))
    elif kind == "explicit":
        frozen = _explicit_matrix_block_encoding()
        oracle = TopLeftRowColumnOracle(1)
        native = SparseMatrixBlockEncoding(
            oracle, oracle, ExplicitEntryOracle(1, ((0.0, 0.25), (1 / 3, 0.467)), 10)
        )
    else:
        frozen = _symmetric_banded_matrix_block_encoding()
        band = SymmetricBandedRowColumnOracle(3, 1)
        native = SparseMatrixBlockEncoding(band, band, UniformEntryOracle(3, 0.3))
    source = frozen.adjoint() if adjoint else frozen
    shape = (native.system_bitsize, native.ancilla_bitsize)
    assert shape == tuple(int(register.total_bits()) for register in source.signature)
    assert (native.alpha, native.epsilon, native.resource_bitsize) == (
        frozen.alpha,
        frozen.epsilon,
        frozen.resource_bitsize,
    )
    assert (
        native.signal_state.emit(tuple(range(native.ancilla_bitsize)), (), (), ()) == ()
    )
    model = native.resource_model(adjoint=adjoint)
    from qualtran.bloqs.basic_gates import Swap
    from qualtran.resource_counting.generalizers import ignore_split_join

    source_top = source.call_graph(max_depth=1, generalizer=ignore_split_join)[1]
    assert source_top[Swap(shape[0])] == 1
    assert sum(source_top.values()) == len(model.calls) == 6
    assert_source_model(model, source)

    logical = sum(shape)
    work, slots = native.requirements()
    system = tuple(reversed(range(shape[0])))
    ancillary = tuple(range(shape[0], logical))
    workspace = tuple(range(logical, logical + work))
    operation = native.make_on(
        system=system,
        ancilla=ancillary,
        workspace=workspace,
        classical_bits=tuple(range(4, 4 + slots)),
        adjoint=adjoint,
    )
    circuit = Circuit(logical + work, dtype=torch.complex128).append(operation)
    gates = circuit.to_ir().instructions
    assert gates == operation.instructions
    if kind == "uniform":
        assert [gate.name for gate in gates] == (
            ["h"] * shape[0]
            + (
                ["swap"] * shape[0] + ["ry"]
                if adjoint
                else ["ry"] + ["swap"] * shape[0]
            )
            + ["h"] * shape[0]
        )
        swap = tuple(
            Instruction("swap", (ordinal_wire, system_wire))
            for ordinal_wire, system_wire in zip(ancillary[1:], system, strict=True)
        )
        offset = shape[0] if adjoint else shape[0] + 1
        assert gates[offset : offset + shape[0]] == swap

    order = (*system, *ancillary)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    values = torch.arange(1 << logical, dtype=torch.float64)
    data = torch.complex(values + 1, values % 7 - 3)
    data /= data.norm()
    basis = (2,) * logical
    reference = (
        (matrix @ data.reshape(basis).permute(order).reshape(-1))
        .reshape(basis)
        .permute(order)
        .reshape(-1)
    )
    clean = torch.zeros(1 << work, dtype=torch.complex128)
    clean[0] = 1
    start, expected = torch.kron(data, clean), torch.kron(reference, clean)
    events = sum(gate.name == "measure" for gate in gates)
    histories = (
        ((0,) * events, (1,) * events, tuple(bit % 2 for bit in range(events)))
        if kind == "banded"
        else tuple(product((0, 1), repeat=events))
    )
    for history in histories:
        torch.testing.assert_close(
            forced_branch(start, gates, history),
            expected / sqrt(2**events),
            atol=2e-11,
            rtol=0,
        )
