"""Check the locked Hermitian sparse fixtures, controlled swaps and symbolic costs."""

from dataclasses import asdict
from itertools import product
from math import acos, sqrt

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
    HermitianSparseMatrixBlockEncoding,
    UniformSqrtEntryOracle,
    source_uniform_hermitian_sparse_hadamards,
    source_uniform_hermitian_sparse_resource_formula,
)
from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
    TopLeftRowColumnOracle,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.primitives.controlled_swap import controlled_swap_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("controlled", [False, True])
def test_original_hermitian_sparse_full_matrix_and_ordered_swaps(
    controlled: bool,
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        _sparse_matrix_hermitian_block_encoding,
    )
    from qualtran.resource_counting.generalizers import ignore_split_join

    reference = _sparse_matrix_hermitian_block_encoding()
    source = reference.controlled() if controlled else reference
    assert reference.adjoint() is reference
    native = HermitianSparseMatrixBlockEncoding(
        TopLeftRowColumnOracle(2), UniformSqrtEntryOracle(2, 0.3)
    )
    assert (native.alpha, native.epsilon, native.resource_bitsize) == (
        reference.alpha,
        reference.epsilon,
        reference.resource_bitsize,
    )
    assert (native.system_bitsize, native.ancilla_bitsize) == tuple(
        r.total_bits() for r in reference.signature
    )
    assert native.signal_state.emit((0, 1, 2, 3), (), (), ()) == ()
    model = native.resource_model(controlled=controlled)
    _, calls = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    assert sum(calls.values()) == len(model.calls) == 8
    exchange_name = "CSwap" if controlled else "Swap"
    source_stages = [
        type(instance.bloq).__name__
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if type(instance.bloq).__name__ != "Partition"
    ]
    assert source_stages == ["Always"] * 3 + [exchange_name] * 2 + ["Always"] * 3
    assert_source_model(model, source)

    shift = int(controlled)
    ctrl = 0 if controlled else None
    system = (shift, shift + 1)
    ancilla = tuple(range(shift + 2, shift + 6))
    first, low, high, second = ancilla
    expected_middle = (
        (
            Instruction("swap", (low, system[0])),
            Instruction("swap", (high, system[1])),
            Instruction("swap", (first, second)),
        )
        if ctrl is None
        else (
            *controlled_swap_ir(ctrl, (low, high), system),
            *controlled_swap_ir(ctrl, (first,), (second,)),
        )
    )
    op = native.make_on(system=system, ancilla=ancilla, control=ctrl)
    angle = 2 * acos(sqrt(0.3))
    prepare = (Instruction("h", (low,)), Instruction("h", (high,)))
    assert op.instructions == (
        *prepare,
        Instruction("ry", (second,), {"theta": angle}),
        *expected_middle,
        Instruction("ry", (second,), {"theta": -angle}),
        *reversed(prepare),
    )
    assert op == native.make_on(
        system=system, ancilla=ancilla, control=ctrl, adjoint=True
    )
    circuit = Circuit(6 + shift, dtype=torch.complex128).append(op)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(get_unitary(circuit), matrix, atol=2e-12, rtol=0)
    ir = circuit.to_ir()
    assert CircuitIR.from_json(ir.to_json()).validate().instructions == ir.instructions
    physical = count_emitted_instructions(ir)
    assert physical.t_count == (21 if controlled else 0)
    assert physical.unconditional.get("measure", 0) == 0
    if not controlled:
        _, leaves = source.call_graph(generalizer=ignore_split_join)
        assert (
            physical.unconditional["h"]
            == (source_uniform_hermitian_sparse_hadamards(2))
            == leaves[Hadamard()]
        )
    circuit.append(op)
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.eye(1 << (6 + shift), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize("controlled", [False, True])
def test_original_top_left_matrix_and_clean_measured_work(controlled: bool) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceColumn,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        SparseMatrixHermitian,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        UniformSqrtEntryOracle as SourceEntry,
    )

    source = SparseMatrixHermitian(SourceColumn(3, 3), SourceEntry(3, 0.3), 0)
    native = HermitianSparseMatrixBlockEncoding(
        TopLeftRowColumnOracle(3, 3), UniformSqrtEntryOracle(3, 0.3)
    )
    locked = source.controlled() if controlled else source
    assert_source_model(native.resource_model(controlled=controlled), locked)
    through = 8 + int(controlled)
    work, slots = native.requirements()
    controls = (0,) if controlled else ()
    control = controls[0] if controls else None
    op = native.make_on(
        system=tuple(range(len(controls), len(controls) + 3)),
        ancilla=tuple(range(len(controls) + 3, through)),
        control=control,
        workspace=tuple(range(through, through + work)),
        classical_bits=tuple(range(7, 7 + slots)),
    )
    ir = Circuit(through + work, dtype=torch.complex128).append(op).to_ir()
    gates = ir.instructions
    physical = count_emitted_instructions(ir)
    assert work == 4
    assert physical.t_count == (68 if controlled else 40)
    values = torch.arange(1 << through, dtype=torch.float64)
    state = torch.complex(values + 1, values % 5 - 2)
    state /= state.norm()
    clean = torch.zeros(1 << work, dtype=torch.complex128)
    clean[0] = 1
    source_matrix = torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128)
    expected = torch.kron(source_matrix @ state, clean)
    initial = torch.kron(state, clean)
    count = sum(gate.name == "measure" for gate in gates)
    assert count == 10
    for outcomes in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / sqrt(2**count),
            atol=6e-12,
            rtol=0,
        )
    stride = 1 << (native.ancilla_bitsize + work)
    selected = torch.arange(8) * stride + (
        (1 << (through + work - 1)) if controlled else 0
    )
    analytical = torch.zeros((8, 8), dtype=torch.complex128)
    analytical[:3, :3] = 0.3
    for i, basis_index in enumerate(selected):
        basis = torch.zeros_like(initial)
        basis[basis_index] = 1
        branch = forced_branch(basis, gates, (0,) * count)
        torch.testing.assert_close(
            branch[selected] * (native.alpha * sqrt(2**count)),
            analytical[:, i],
            atol=6e-12,
            rtol=0,
        )


@pytest.mark.parametrize("controlled", [False, True])
def test_original_symbolic_hermitian_resource_formula(controlled: bool) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        _sparse_matrix_symb_hermitian_block_encoding,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join
    from sympy import Symbol, simplify

    source = _sparse_matrix_symb_hermitian_block_encoding()
    original = source.controlled() if controlled else source
    width = Symbol("n", positive=True, integer=True)
    assert tuple(register.total_bits() for register in original.signature) == (
        *((1,) if controlled else ()),
        width,
        width + 2,
    )
    assert (source.alpha, source.epsilon, source.resource_bitsize) == (2**width, 0, 0)
    if not controlled:
        _, leaves = source.call_graph(generalizer=ignore_split_join)
        assert source_uniform_hermitian_sparse_hadamards(width) == leaves[Hadamard()]
    actual = asdict(
        source_uniform_hermitian_sparse_resource_formula(width, controlled=controlled)
    )
    peak = actual.pop("logical_qubits")
    assert simplify(peak - get_cost_value(original, QubitCount())) == 0
    expected = get_cost_value(original, QECGatesCost())
    assert all(
        simplify(count - getattr(expected, name)) == 0 for name, count in actual.items()
    )
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        source.decompose_bloq()
