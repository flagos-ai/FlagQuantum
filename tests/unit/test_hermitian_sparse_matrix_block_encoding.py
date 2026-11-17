"""Check the locked Hermitian sparse fixtures, controlled swaps and symbolic costs."""

from dataclasses import asdict
from itertools import product
from math import acos, sqrt
from typing import Any

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
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    assert_source_wire_streams,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_control_predicate import ControlCase

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


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize(
    "configuration",
    ("full", "independent-count", "band-one", "symbolic-band", "concrete-width"),
)
def test_hermitian_symbolic_oracle_costs_and_fallback_peak(
    controlled: bool, configuration: str
) -> None:
    from attrs import evolve
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SymmetricBandedRowColumnOracle as SourceBanded,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceTop,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        _sparse_matrix_symb_hermitian_block_encoding,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join
    from sympy import Symbol, simplify

    from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
        source_hermitian_sparse_fallback_formula,
    )
    from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
        source_symbolic_banded_oracle_resource_formula,
        source_top_left_oracle_resource_formula,
        source_uniform_entry_oracle_resource_formula,
    )

    source = _sparse_matrix_symb_hermitian_block_encoding()
    width = 4 if configuration == "concrete-width" else source.system_bitsize
    band = None
    if configuration == "full":
        count = source.alpha
    elif configuration in ("band-one", "symbolic-band"):
        band = (
            1
            if configuration == "band-one"
            else Symbol("b", positive=True, integer=True)
        )
        count = 2 * band + 1
        source = evolve(source, col_oracle=SourceBanded(width, band))
    else:
        count = Symbol("m", positive=True, integer=True)
        source = evolve(
            source,
            col_oracle=SourceTop(width, num_nonzero=count),
            entry_oracle=evolve(source.entry_oracle, system_bitsize=width),
        )

    original = source.controlled() if controlled else source
    assert (
        sum(original.call_graph(max_depth=1, generalizer=ignore_split_join)[1].values())
        == 8
    )
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        source.decompose_bloq()

    entry = source_uniform_entry_oracle_resource_formula(width)
    for legacy in (False, True):
        if band is None:
            top = source_top_left_oracle_resource_formula(width)
            column = top, top
        else:
            column = (
                source_symbolic_banded_oracle_resource_formula(
                    width, band, legacy_shims=legacy
                ),
                source_symbolic_banded_oracle_resource_formula(
                    width, band, adjoint=True, legacy_shims=legacy
                ),
            )
        model = source_hermitian_sparse_fallback_formula(
            width,
            count,
            column=column,
            entry=(entry, entry),
            controlled=controlled,
            legacy_shims=legacy,
        )
        gates = get_cost_value(original, QECGatesCost(legacy_shims=legacy))
        assert all(
            simplify(count - getattr(gates, name)) == 0
            for name, count in asdict(model).items()
            if name != "logical_qubits"
        )
        assert (
            simplify(model.logical_qubits - get_cost_value(original, QubitCount())) == 0
        )


def _assert_locked_hermitian_stream(
    source: Any,
    ports: SourcePorts,
    actual: tuple[Instruction, ...],
    work: tuple[int, ...],
    slots: tuple[int, ...],
    predicate_work: tuple[int, ...] = (),
    predicate_slots: tuple[int, ...] = (),
) -> None:
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import Ry, Swap, XGate
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    from tests.unit.register_source_graph import (
        source_control_predicate_leaf,
        source_partition_leaf,
        walk_source_graph,
    )
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_uniform_preparation import reference_uniform_preparation_gates

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        predicate = source_control_predicate_leaf(
            bloq, inputs, predicate_work, predicate_slots
        )
        if predicate is not None:
            return predicate
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
        if isinstance(bloq, Ry):
            return (
                Instruction("ry", inputs["q"], {"theta": float(bloq.angle)}),
            ), inputs
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, CSwap):
            return (
                reference_arithmetic_gates(
                    bloq, inputs, (), {}, preserve_components=True
                ),
                inputs,
            )
        if isinstance(bloq, Swap):
            return tuple(
                Instruction("swap", (left, right))
                for left, right in zip(inputs["x"], inputs["y"], strict=True)
            ), inputs
        return None

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert ports == outputs
    assert_source_wire_streams(actual, expected)


@pytest.mark.parametrize("controlled", (False, True))
def test_padded_hermitian_source_primitive_flow(controlled: bool) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceColumn,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        SparseMatrixHermitian,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        UniformSqrtEntryOracle as SourceEntry,
    )

    source = SparseMatrixHermitian(SourceColumn(3, 3), SourceEntry(3, 0.3), eps=0)
    native = HermitianSparseMatrixBlockEncoding(
        TopLeftRowColumnOracle(3, 3), UniformSqrtEntryOracle(3, 0.3)
    )
    start = int(controlled)
    ports = {
        "system": tuple(range(start, start + 3)),
        "ancilla": tuple(range(start + 3, start + 8)),
    }
    work_count, slot_count = native.requirements()
    work = tuple(range(start + 8, start + 8 + work_count))
    slots = tuple(range(17, 17 + slot_count))
    op = native.make_on(
        system=ports["system"],
        ancilla=ports["ancilla"],
        control=0 if controlled else None,
        workspace=work,
        classical_bits=slots,
    )
    if controlled:
        source = source.controlled()
        ports["ctrl"] = (0,)
    _assert_locked_hermitian_stream(source, ports, op.instructions, work, slots)


@pytest.mark.parametrize(
    "case",
    (
        (("QBit", 1, (), (0,), "unsigned"),),
        (("QBit", 1, (2,), (0, 1), "unsigned"),),
        (("QUInt", 2, (), (2,), "unsigned"),),
        (("QUInt", 2, (), (2,), "unsigned"), ("QBit", 1, (), (1,), "unsigned")),
    ),
)
def test_hermitian_structured_controls_against_locked_circuit(
    case: ControlCase,
) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        _sparse_matrix_hermitian_block_encoding,
        _sparse_matrix_symb_hermitian_block_encoding,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import simplify

    from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
        source_hermitian_sparse_fallback_formula,
    )
    from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
        source_top_left_oracle_resource_formula,
        source_uniform_entry_oracle_resource_formula,
    )
    from tests.unit.test_register_control_predicate import control_fixture

    spec, registers, groups = control_fixture(case)
    source = _sparse_matrix_hermitian_block_encoding().controlled(spec)
    native = HermitianSparseMatrixBlockEncoding(
        TopLeftRowColumnOracle(2), UniformSqrtEntryOracle(2, 0.3)
    )
    start = sum(map(len, groups))
    through = start + native.system_bitsize + native.ancilla_bitsize
    work = tuple(range(through, through + start - 1))
    slots = tuple(range(19, 19 + len(work)))
    system, ancilla = tuple(range(start, start + 2)), tuple(range(start + 2, through))
    bind = dict(
        controls=groups,
        control_registers=registers,
        system=system,
        ancilla=ancilla,
        control_workspace=work,
        control_classical_bits=slots,
    )
    operation = native.make_controlled_on(**bind)
    assert operation == native.make_controlled_on(**bind, adjoint=True)
    ir = Circuit(through + len(work)).append(operation).to_ir()
    counters = count_emitted_instructions(ir)
    assert counters.t_count == 21 + 4 * len(work)
    assert counters.unconditional.get("measure", 0) == len(work)
    ports = {
        **{
            register.name: group
            for register, group in zip(
                source.signature[: len(groups)], groups, strict=True
            )
        },
        "system": system,
        "ancilla": ancilla,
    }
    _assert_locked_hermitian_stream(source, ports, ir.instructions, (), (), work, slots)
    assert_source_model(native.controlled_resource_model(registers), source)

    index = torch.arange(1 << through, dtype=torch.float64)
    coherent = torch.complex(index % 7 - 3, index % 11 - 5)
    coherent /= coherent.norm()
    clean = torch.zeros(1 << len(work), dtype=torch.complex128)
    clean[0] = 1
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected = torch.kron(matrix @ coherent, clean)
    for history in product((0, 1), repeat=len(work)):
        torch.testing.assert_close(
            forced_branch(torch.kron(coherent, clean), ir.instructions, history),
            expected / sqrt(2 ** len(work)),
            atol=6e-12,
            rtol=0,
        )

    symbolic = _sparse_matrix_symb_hermitian_block_encoding()
    width = symbolic.system_bitsize
    source_symbolic = symbolic.controlled(spec)
    column = source_top_left_oracle_resource_formula(width)
    entry = source_uniform_entry_oracle_resource_formula(width)
    for legacy in (False, True):
        formula = source_hermitian_sparse_fallback_formula(
            width,
            symbolic.alpha,
            column=(column, column),
            entry=(entry, entry),
            control_registers=registers,
            legacy_shims=legacy,
        )
        gates = get_cost_value(source_symbolic, QECGatesCost(legacy_shims=legacy))
        assert all(
            simplify(count - getattr(gates, name)) == 0
            for name, count in asdict(formula).items()
            if name != "logical_qubits"
        )
        assert (
            simplify(
                formula.logical_qubits - get_cost_value(source_symbolic, QubitCount())
            )
            == 0
        )


def test_padded_hermitian_compound_control_cleans_disjoint_work() -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle as SourceColumn,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        SparseMatrixHermitian,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        UniformSqrtEntryOracle as SourceEntry,
    )

    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from tests.unit.test_sparse_chemistry_prepare import _branch

    source = SparseMatrixHermitian(SourceColumn(3, 3), SourceEntry(3, 0.3), eps=0)
    source = source.controlled(CtrlSpec(qdtypes=QUInt(2), cvs=2))
    native = HermitianSparseMatrixBlockEncoding(
        TopLeftRowColumnOracle(3, 3), UniformSqrtEntryOracle(3, 0.3)
    )
    registers = (ControlRegister(2, (2,)),)
    controls, system, ancilla = (0, 1), (2, 3, 4), (5, 6, 7, 8, 9)
    work, control_work = (10, 11, 12, 13), (14,)
    slots, control_slots = (17, 18), (33,)
    options = dict(
        system=system,
        ancilla=ancilla,
        controls=(controls,),
        control_registers=registers,
        workspace=work,
        classical_bits=slots,
        control_workspace=control_work,
        control_classical_bits=control_slots,
    )
    operation = native.make_controlled_on(**options)
    ir = Circuit(15).append(operation).to_ir()
    _assert_locked_hermitian_stream(
        source,
        {"ctrl": controls, "system": system, "ancilla": ancilla},
        ir.instructions,
        work,
        slots,
        control_work,
        control_slots,
    )
    assert_source_model(native.controlled_resource_model(registers), source)
    counters = count_emitted_instructions(ir)
    assert counters.t_count == 72
    assert counters.unconditional["measure"] == 11
    with pytest.raises(ValueError):
        native.make_controlled_on(**(options | {"control_workspace": work[-1:]}))
    with pytest.raises(ValueError, match="distinct"):
        native.make_controlled_on(**(options | {"control_classical_bits": slots[:1]}))

    data = {0: 0.5, 1 << 8: 0.5j, (2 << 8) | (1 << 5): -0.5, (2 << 8) | 2: -0.5j}
    initial = {index << 5: value for index, value in data.items()}
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    target = sum(value * matrix[:, index] for index, value in data.items())
    expected = {
        index << 5: value
        for index, value in enumerate(target.tolist())
        if abs(value) > 1e-12
    }
    for history in product((0, 1), repeat=11):
        assert _branch(
            ir.instructions, ir.n_wires, history, initial, normalize_measurements=True
        ) == pytest.approx(expected, abs=6e-12)
