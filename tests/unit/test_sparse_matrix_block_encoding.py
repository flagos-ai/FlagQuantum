"""Compare the frozen sparse-access block and oracle costs and amplitudes."""

from collections import Counter
from itertools import product
from math import sqrt
from typing import Literal

import numpy as np
import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.sparse_matrix_block_encoding import (
    SparseMatrixBlockEncoding,
    source_sparse_matrix_signature,
)
from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
    ExplicitEntryOracle,
    RowColumnOracle,
    SymmetricBandedRowColumnOracle,
    TopLeftRowColumnOracle,
    UniformEntryOracle,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ["uniform", "explicit", "banded", "top_left"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_sparse_block_models_and_source_amplitudes(
    kind: Literal["uniform", "explicit", "banded", "top_left"],
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
    elif kind == "top_left":
        from qualtran.bloqs.block_encoding.sparse_matrix import SparseMatrix
        from qualtran.bloqs.block_encoding.sparse_matrix import (
            TopLeftRowColumnOracle as SourceTop,
        )
        from qualtran.bloqs.block_encoding.sparse_matrix import (
            UniformEntryOracle as SourceEntry,
        )

        source_top = SourceTop(3, num_nonzero=3)
        frozen = SparseMatrix(source_top, source_top, SourceEntry(3, 0.3), eps=0)
        top = TopLeftRowColumnOracle(3, 3)
        native = SparseMatrixBlockEncoding(top, top, UniformEntryOracle(3, 0.3))
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
    if kind == "top_left":
        selected = torch.arange(8) << native.ancilla_bitsize
        expected_top = torch.zeros((8, 8), dtype=torch.complex128)
        expected_top[:3, :3] = 0.3
        torch.testing.assert_close(
            matrix[selected][:, selected] * native.alpha,
            expected_top,
            rtol=0,
            atol=3e-12,
        )
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


@pytest.mark.parametrize("case", [*range(11), "band2", "band3", "vlasov"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_locked_explicit_stress_and_vlasov(
    case: int | str, adjoint: bool, request: pytest.FixtureRequest
) -> None:
    request.config.addinivalue_line("markers", "notebook: upstream notebook execution")
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        ExplicitEntryOracle as SourceEntry,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import SparseMatrix
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SymmetricBandedRowColumnOracle as SourceBand,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceTop,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_test import (
        gen_vlasov_hamiltonian,
        random_banded_matrix,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from tests.unit.test_sparse_chemistry_prepare import _branch

    rng = np.random.RandomState(1234)
    if isinstance(case, int):
        n = 1
        data = np.array([[0.0, 0.25], [1 / 3, 0.467]])
        for _ in range(case):
            n = rng.randint(1, 3)
            data = rng.rand(1 << n, 1 << n)
        precision, tolerance = 10, 0.003
        oracle: RowColumnOracle = TopLeftRowColumnOracle(n)
        source_oracle = SourceTop(n)
    else:
        n = 2 if case == "band2" else 3
        data = (
            gen_vlasov_hamiltonian(n, 0.5, (1 << n) - 1)
            if case == "vlasov"
            else random_banded_matrix(n, 1, rs=rng)
        )
        precision, tolerance = 7, 0.02 if case == "vlasov" else 0.03
        oracle = SymmetricBandedRowColumnOracle(n, 1)
        source_oracle = SourceBand(n, 1)
    native = SparseMatrixBlockEncoding(
        oracle,
        oracle,
        ExplicitEntryOracle(n, tuple(map(tuple, data.tolist())), precision),
    )
    source = SparseMatrix(
        source_oracle, source_oracle, SourceEntry(n, data, precision), eps=0
    )
    locked = source.adjoint() if adjoint else source
    assert_source_model(native.resource_model(adjoint=adjoint), locked)
    assert [(r.name, r.bitsize) for r in native.signature] == [
        (r.name, r.bitsize) for r in source.signature
    ]

    logical = 2 * n + 1
    work, slots = native.requirements()
    width = logical + work
    circuit = Circuit(width).append(
        native.make_on(
            system=tuple(range(n)),
            ancilla=tuple(range(n, logical)),
            workspace=tuple(range(logical, width)),
            classical_bits=tuple(range(3, 3 + slots)),
            adjoint=adjoint,
        )
    )
    instructions = circuit.to_ir().instructions
    count = Counter(g.name for g in instructions)
    gates: list[Instruction] = []
    for gate in instructions:
        if gate.name == "swap":
            # The reusable sparse evaluator supports CNOT as the exact SWAP expansion.
            left, right = gate.wires
            gates.extend(
                Instruction("cx", pair)
                for pair in ((left, right), (right, left), (left, right))
            )
        else:
            gates.append(gate)
    source_cost = get_cost_value(locked, QECGatesCost())
    assert count["measure"] == source_cost.measurement
    assert count["t"] + count["tdg"] == 4 * source_cost.and_bloq
    assert count["cry"] == precision

    matrix = locked.tensor_contract()
    projected_indices = np.arange(1 << n) << (n + 1)
    projected = matrix[np.ix_(projected_indices, projected_indices)] * float(
        native.alpha
    )
    np.testing.assert_allclose(
        projected, data.T if adjoint else data, atol=tolerance, rtol=0
    )
    grid = np.arange(1 << logical)
    coherent = grid + 1 + 1j * (grid % 11 - 4)
    coherent /= np.linalg.norm(coherent)
    expected = {
        index << work: amp
        for index, amp in enumerate(matrix @ coherent)
        if abs(amp) > 1e-12
    }
    initial = {index << work: amp for index, amp in enumerate(coherent)}
    for history in (
        (0,) * count["measure"],
        tuple(index % 2 for index in range(count["measure"])),
    ):
        actual = _branch(gates, width, history, initial, normalize_measurements=True)
        # Exact equality of support also proves no populated private-work wires.
        assert actual == pytest.approx(expected, abs=3e-10, rel=0)


def test_sparse_symbolic_signature_matches_locked_shape() -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        _sparse_matrix_symb_block_encoding,
    )

    original = _sparse_matrix_symb_block_encoding()
    signature = source_sparse_matrix_signature(original.system_bitsize)
    assert [(reg.name, reg.bitsize, reg.dtype.encoding) for reg in signature] == [
        (reg.name, reg.bitsize, "opaque") for reg in original.signature
    ]
    assert signature.n_qubits() == original.signature.n_qubits()


@pytest.mark.parametrize("adjoint", [False, True])
def test_original_banded_index_oracle_full_table_and_branches(adjoint: bool) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SymmetricBandedRowColumnOracle as SourceBand,
    )

    from tests.unit.test_sparse_chemistry_prepare import _branch

    oracle = SymmetricBandedRowColumnOracle(3, 1)
    source = SourceBand(3, 1)
    locked = source.adjoint() if adjoint else source
    assert_source_model(oracle.resource_model(adjoint=adjoint), locked)
    work, slots = oracle.requirements()
    width = 6 + work
    gates = oracle.emit(
        (0, 1, 2),
        (3, 4, 5),
        tuple(range(6, width)),
        tuple(range(slots)),
        adjoint=adjoint,
    )
    expected = locked.tensor_contract()
    basis = np.arange(64)
    initial = (basis + 1 + 1j * (basis % 3 - 1)).astype(np.complex128)
    initial /= np.linalg.norm(initial)
    inputs = {i << work: amp for i, amp in enumerate(initial)}
    outputs = {i << work: amp for i, amp in enumerate(expected @ initial)}
    for history in product((0, 1), repeat=sum(g.name == "measure" for g in gates)):
        assert _branch(
            gates, width, history, inputs, normalize_measurements=True
        ) == pytest.approx(outputs, abs=2e-12, rel=0)
    if not adjoint:
        for index, ordinal in product(range(8), range(3)):
            mapped, untouched = source.call_classically(l=ordinal, i=index)
            assert untouched == index and mapped == (ordinal + index - 1) % 8
            assert expected[(mapped << 3) | index, (ordinal << 3) | index] == 1
