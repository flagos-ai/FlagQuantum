"""Compare the frozen sparse-access block and oracle costs and amplitudes."""

from collections import Counter
from itertools import product
from math import sqrt
from typing import Any, Literal

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
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    assert_source_wire_streams,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _assert_locked_sparse_primitive_flow(
    source: Any,
    native: SparseMatrixBlockEncoding,
    ports: SourcePorts,
    actual: tuple[Instruction, ...],
    work: tuple[int, ...],
    slots: tuple[int, ...],
) -> None:
    """Bind the source stages to the same ordered work and feedback storage."""
    from qualtran import Adjoint, Controlled
    from qualtran.bloqs.arithmetic import Add, AddK
    from qualtran.bloqs.basic_gates import Ry, Swap
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_source_qrom import source_qrom_gates
    from tests.unit.test_uniform_preparation import reference_uniform_preparation_gates

    entry = native.entry_oracle
    word: tuple[int, ...] = ()
    qrom_work: tuple[int, ...] = ()
    qrom_slots: tuple[int, ...] = ()
    if isinstance(entry, ExplicitEntryOracle):
        precision = entry.entry_bitsize
        entry_work, entry_slots = entry.requirements()
        word, qrom_work = work[:precision], work[precision:entry_work]
        qrom_slots = slots[:entry_slots]
    width = native.system_bitsize
    carries = work[: width - 1]
    constant = work[width - 1 : 2 * width - 1]

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        if isinstance(bloq, Allocate):
            assert len(word) == bloq.signature.n_qubits()
            return (), {next(iter(bloq.signature)).name: word}
        if isinstance(bloq, Free):
            assert tuple(inputs.values()) == (word,)
            return (), {}
        inverse = isinstance(bloq, Adjoint)
        base = bloq.subbloq if inverse else bloq
        if isinstance(base, PrepareUniformSuperposition):
            assert base.cvs == ()
            return (
                reference_uniform_preparation_gates(
                    base.n, (), inputs["target"], work, slots, (), adjoint=inverse
                ),
                inputs,
            )
        if isinstance(base, (Add, AddK)):
            stage_work = carries if isinstance(base, Add) else (*constant, *carries)
            feedback = dict(zip(carries, slots[: len(carries)], strict=True))
            return reference_arithmetic_gates(
                bloq, inputs, stage_work, feedback
            ), inputs
        if isinstance(bloq, Swap):
            return tuple(
                Instruction("swap", pair)
                for pair in zip(inputs["x"], inputs["y"], strict=True)
            ), inputs
        if isinstance(bloq, QROM):
            return source_qrom_gates(bloq, inputs, qrom_work, qrom_slots), inputs
        sign = -1 if inverse else 1
        if isinstance(base, Ry):
            return (
                Instruction("ry", inputs["q"], {"theta": sign * float(base.angle)}),
            ), inputs
        if isinstance(base, Controlled) and isinstance(base.subbloq, Ry):
            assert base.ctrl_spec.get_single_ctrl_val() == 1
            (control,) = (wires for name, wires in inputs.items() if name != "q")
            theta = sign * float(base.subbloq.angle)
            return (
                Instruction("cry", (*control, *inputs["q"]), {"theta": theta}),
            ), inputs
        return None

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source_wire_streams(actual, expected)


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
    assert str(native) == str(frozen)
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
    feedback = tuple(range(4, 4 + slots))
    operation = native.make_on(
        system=system,
        ancilla=ancillary,
        workspace=workspace,
        classical_bits=feedback,
        adjoint=adjoint,
    )
    circuit = Circuit(logical + work, dtype=torch.complex128).append(operation)
    gates = circuit.to_ir().instructions
    assert gates == operation.instructions
    _assert_locked_sparse_primitive_flow(
        source,
        native,
        {"system": system, "ancilla": ancillary},
        gates,
        workspace,
        feedback,
    )
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
    system, ancilla = tuple(range(n)), tuple(range(n, logical))
    workspace, feedback = tuple(range(logical, width)), tuple(range(3, 3 + slots))
    circuit = Circuit(width).append(
        native.make_on(
            system=system,
            ancilla=ancilla,
            workspace=workspace,
            classical_bits=feedback,
            adjoint=adjoint,
        )
    )
    instructions = circuit.to_ir().instructions
    _assert_locked_sparse_primitive_flow(
        locked,
        native,
        {"system": system, "ancilla": ancilla},
        instructions,
        workspace,
        feedback,
    )
    from tests.unit.measurement_certificate import certify_erasure_histories

    certified = certify_erasure_histories(
        instructions, n_wires=width, zero_wires=workspace
    )
    assert certified == sum(g.name == "measure" for g in instructions)
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


@pytest.mark.parametrize(
    "configuration",
    (
        "original",
        "independent-count",
        "concrete-width",
        "two-entries",
        "band-one",
        "symbolic-band",
    ),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_symbolic_sparse_oracle_calls_gate_categories_and_peak(
    configuration: str, adjoint: bool
) -> None:
    from dataclasses import asdict

    from attrs import evolve
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SparseMatrix,
        _sparse_matrix_symb_block_encoding,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SymmetricBandedRowColumnOracle as SourceBanded,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceTop,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        UniformEntryOracle as SourceEntry,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join
    from sympy import Symbol, simplify

    from flagquantum.algorithms.primitives.sparse_matrix_block_encoding import (
        source_symbolic_sparse_matrix_resource_formula,
    )
    from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
        source_symbolic_banded_oracle_resource_formula,
        source_top_left_oracle_resource_formula,
        source_uniform_entry_oracle_resource_formula,
    )

    source = _sparse_matrix_symb_block_encoding()
    width = source.system_bitsize
    band = None
    count = Symbol("m", positive=True, integer=True)
    if configuration == "original":
        count = source.alpha
    elif configuration in ("band-one", "symbolic-band"):
        band = (
            1
            if configuration == "band-one"
            else Symbol("b", positive=True, integer=True)
        )
        count = 2 * band + 1
        index = SourceBanded(width, band)
        source = evolve(source, row_oracle=index, col_oracle=index)
    else:
        if configuration == "concrete-width":
            width = 4
        if configuration == "two-entries":
            width, count = Symbol("w", positive=True, integer=True), 2
        index = SourceTop(width, num_nonzero=count)
        source = SparseMatrix(index, index, SourceEntry(width, 0.3), eps=0)
    assert source.alpha == count
    directional = source.adjoint() if adjoint else source
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        directional.decompose_bloq()
    assert (
        sum(
            directional.call_graph(max_depth=1, generalizer=ignore_split_join)[
                1
            ].values()
        )
        == 6
    )

    def index_formula(reverse: bool, legacy: bool):
        if band is None:
            return source_top_left_oracle_resource_formula(width)
        return source_symbolic_banded_oracle_resource_formula(
            width, band, adjoint=reverse, legacy_shims=legacy
        )

    entry = source_uniform_entry_oracle_resource_formula(width)
    for legacy in (False, True):
        profile = QECGatesCost(legacy_shims=legacy)
        row = index_formula(not adjoint, legacy)
        column = index_formula(adjoint, legacy)
        children = (
            (source.row_oracle if adjoint else source.row_oracle.adjoint(), row),
            (source.col_oracle.adjoint() if adjoint else source.col_oracle, column),
            (source.entry_oracle.adjoint() if adjoint else source.entry_oracle, entry),
        )
        model = source_symbolic_sparse_matrix_resource_formula(
            width, count, row=row, column=column, entry=entry, legacy_shims=legacy
        )
        for original, native in (*children, (directional, model)):
            expected = get_cost_value(original, profile)
            for name, value in asdict(native).items():
                if name != "logical_qubits":
                    assert simplify(value - getattr(expected, name)) == 0
            assert (
                simplify(native.logical_qubits - get_cost_value(original, QubitCount()))
                == 0
            )


def test_symbolic_singleton_diffusion_retains_source_admission_boundary() -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import SparseMatrix
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceTop,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        UniformEntryOracle as SourceEntry,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from sympy import Symbol

    from flagquantum.algorithms.primitives.sparse_matrix_block_encoding import (
        source_symbolic_sparse_matrix_resource_formula,
    )
    from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
        source_top_left_oracle_resource_formula,
        source_uniform_entry_oracle_resource_formula,
    )

    width = Symbol("n", positive=True, integer=True)
    index = SourceTop(width, num_nonzero=1)
    source = SparseMatrix(index, index, SourceEntry(width, 0.3), eps=0)
    row = source_top_left_oracle_resource_formula(width)
    entry = source_uniform_entry_oracle_resource_formula(width)
    with pytest.raises(ValueError, match="Bloq registers do not match"):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(ValueError, match="singleton diffusion lacks its target port"):
        source_symbolic_sparse_matrix_resource_formula(
            width, 1, row=row, column=row, entry=entry
        )


@pytest.mark.parametrize(
    "configuration,width,count,band",
    (
        ("original", 2, 4, None),
        ("independent-count", 2, 3, None),
        ("concrete-width", 3, 3, None),
        ("two-entries", 2, 2, None),
        ("band-one", 2, 3, 1),
        ("symbolic-band", 3, 3, 1),
    ),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_resolved_symbolic_sparse_declarations_execute_original_call_chain(
    configuration: str, width: int, count: int, band: int | None, adjoint: bool
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding.sparse_matrix import SparseMatrix
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SymmetricBandedRowColumnOracle as SourceBand,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceTop,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        UniformEntryOracle as SourceEntry,
    )
    from sympy import Symbol, sympify

    from tests.unit.measurement_certificate import certify_erasure_histories

    n, s, b = (Symbol(name, positive=True, integer=True) for name in ("n", "s", "b"))
    symbolic_width = 3 if configuration == "concrete-width" else n
    symbolic_count = (
        2**n
        if configuration == "original"
        else 2
        if configuration == "two-entries"
        else s
    )
    symbolic_band = b if configuration == "symbolic-band" else 1
    symbolic_index = (
        SourceTop(symbolic_width, num_nonzero=symbolic_count)
        if band is None
        else SourceBand(symbolic_width, symbolic_band)
    )
    symbolic = SparseMatrix(
        symbolic_index, symbolic_index, SourceEntry(symbolic_width, 0.3), eps=0
    )
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        symbolic.decompose_bloq()
    substitutions = {n: width, s: count, b: band or 1}
    resolved_width = int(sympify(symbolic.system_bitsize).subs(substitutions))
    resolved_count = int(sympify(symbolic.alpha).subs(substitutions))
    assert (resolved_width, resolved_count) == (width, count)
    oracle: RowColumnOracle
    if band is None:
        source_index = SourceTop(resolved_width, num_nonzero=resolved_count)
        oracle = TopLeftRowColumnOracle(resolved_width, resolved_count)
    else:
        resolved_band = int(sympify(symbolic_band).subs(substitutions))
        source_index = SourceBand(resolved_width, resolved_band)
        oracle = SymmetricBandedRowColumnOracle(resolved_width, resolved_band)
    original = SparseMatrix(source_index, source_index, SourceEntry(width, 0.3), eps=0)
    source = original.adjoint() if adjoint else original
    native = SparseMatrixBlockEncoding(oracle, oracle, UniformEntryOracle(width, 0.3))
    assert [(reg.name, reg.bitsize) for reg in native.signature] == [
        (reg.name, int(sympify(reg.bitsize).subs(substitutions)))
        for reg in symbolic.signature
    ]
    assert_source_model(native.resource_model(adjoint=adjoint), source)
    work, slots = native.requirements()
    logical = 2 * width + 1
    workspace = tuple(range(logical, logical + work))
    feedback = tuple(range(slots))
    ports = {"system": tuple(range(width)), "ancilla": tuple(range(width, logical))}
    operation = native.make_on(
        **ports, workspace=workspace, classical_bits=feedback, adjoint=adjoint
    )
    gates = Circuit(logical + work).append(operation).to_ir().instructions
    _assert_locked_sparse_primitive_flow(
        source, native, ports, gates, workspace, feedback
    )
    events = certify_erasure_histories(
        gates, n_wires=logical + work, zero_wires=workspace
    )
    assert events == sum(g.name == "measure" for g in gates)
    values = np.arange(1 << logical)
    coherent = (values + 1 + 1j * (values % 5 - 2)).astype(np.complex128)
    coherent /= np.linalg.norm(coherent)
    expected = source.tensor_contract() @ coherent
    state = torch.as_tensor(coherent, dtype=torch.complex128)
    reference = torch.as_tensor(expected, dtype=torch.complex128)
    clean = torch.zeros(1 << work, dtype=torch.complex128)
    clean[0] = 1
    state, reference = torch.kron(state, clean), torch.kron(reference, clean)
    for history in {tuple(0 for _ in range(events)), tuple(1 for _ in range(events))}:
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            reference / (2 ** (events / 2)),
            atol=3e-11,
            rtol=0,
        )


@pytest.mark.parametrize("adjoint", (False, True))
def test_resolved_explicit_entry_precision_preserves_quantization_and_every_branch(
    adjoint: bool,
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        ExplicitEntryOracle as SourceEntry,
    )
    from sympy import Symbol

    precision = Symbol("k", positive=True, integer=True)
    data = ((0.0, 0.25), (1 / 3, 0.467))
    symbolic = SourceEntry(1, data, precision)
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        symbolic.decompose_bloq()
    resolved = int(precision.subs({precision: 2}))
    native = ExplicitEntryOracle(1, data, resolved)
    concrete = SourceEntry(1, data, resolved)
    locked = concrete.adjoint() if adjoint else concrete
    assert_source_model(native.resource_model(adjoint=adjoint), locked)
    work, slots = native.requirements()
    workspace = tuple(range(3, 3 + work))
    feedback = tuple(range(slots))
    actual = native.emit(0, (1,), (2,), workspace, feedback, adjoint=adjoint)
    from qualtran.bloqs.block_encoding.sparse_matrix import SparseMatrix
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceTop,
    )

    # The whole source walker binds both QROM passes to the same caller-owned
    # event slot; an external snapshot instead retains two historical keys.
    index = TopLeftRowColumnOracle(1)
    block = SparseMatrixBlockEncoding(index, index, native)
    frozen_index = SourceTop(1)
    original = SparseMatrix(frozen_index, frozen_index, concrete, eps=0)
    operation = block.make_on(
        system=(0,),
        ancilla=(1, 2),
        workspace=workspace,
        classical_bits=feedback,
        adjoint=adjoint,
    )
    _assert_locked_sparse_primitive_flow(
        original.adjoint() if adjoint else original,
        block,
        {"system": (0,), "ancilla": (1, 2)},
        operation.instructions,
        workspace,
        feedback,
    )
    grid = torch.arange(8, dtype=torch.float64)
    state = torch.complex(grid + 1, grid % 3 - 1)
    state /= state.norm()
    expected = torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128) @ state
    clean = torch.zeros(1 << work, dtype=torch.complex128)
    clean[0] = 1
    events = sum(g.name == "measure" for g in actual)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(torch.kron(state, clean), actual, history),
            torch.kron(expected, clean) / (2 ** (events / 2)),
            atol=3e-12,
            rtol=0,
        )
