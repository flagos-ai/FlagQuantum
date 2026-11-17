"""Reconcile the four original Hubbard oracle wrappers and controlled cost."""

from dataclasses import asdict
from math import sqrt

import pytest

from examples.algorithms.hubbard_lcu_catalog import (
    HubbardLCUExample,
    hubbard_lcu_circuit,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_hubbard_walk import _projected_select
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("packed", [False, True])
def test_hubbard_lcu_original_wrapper_metadata_and_models(packed: bool) -> None:
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        _black_box_lcu_block,
        _black_box_select_block,
        _lcu_block,
        _select_block,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import (
        ignore_alloc_free,
        ignore_split_join,
    )

    example = HubbardLCUExample(packed=packed)
    select, lcu = example.select_block, example.lcu_block()
    select_layout = (*select.selection_layout, *select.target_layout)
    lcu_layout = (*lcu.selection_layout, *lcu.junk_layout, *lcu.target_layout)
    source_select = (_black_box_select_block if packed else _select_block)()
    source_lcu = (_black_box_lcu_block if packed else _lcu_block)()
    for block, shape, model, source in (
        (
            select,
            select_layout,
            select.resource_model(example.select_model()),
            source_select,
        ),
        (lcu, lcu_layout, example.lcu_model(), source_lcu),
    ):
        assert shape == tuple(
            (reg.name, int(reg.total_bits())) for reg in source.signature
        )
        assert (
            block.system_bitsize,
            block.ancilla_bitsize,
            block.resource_bitsize,
        ) == (
            source.system_bitsize,
            source.ancilla_bitsize,
            source.resource_bitsize,
        )
        assert (block.alpha, block.epsilon) == (source.alpha, source.epsilon)
        result = estimate_component_resources(model)
        assert {k: v for k, v in asdict(result.gates).items() if v} == get_cost_value(
            source, QECGatesCost()
        ).asdict()
        assert result.peak_logical_qubits == get_cost_value(source, QubitCount())
        _, calls = source.call_graph(
            max_depth=1, generalizer=[ignore_split_join, ignore_alloc_free]
        )
        assert tuple(call.repetitions for call in model.calls) == tuple(calls.values())


@pytest.mark.parametrize("control", [None, 0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_named_and_packed_hubbard_lcu_emit_identical_instruction_streams(
    control: int | None,
    adjoint: bool,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.lcu_block_encoding import _lcu_block
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    named = HubbardLCUExample()
    packed = HubbardLCUExample(packed=True)
    named_ir = hubbard_lcu_circuit(named, control=control, adjoint=adjoint).to_ir()
    packed_ir = hubbard_lcu_circuit(packed, control=control, adjoint=adjoint).to_ir()
    assert named_ir.instructions == packed_ir.instructions
    for ir in (named_ir, packed_ir):
        restored = CircuitIR.from_json(ir.to_json()).validate()
        assert restored.instructions == ir.instructions
    counts = count_emitted_instructions(named_ir)
    source = _lcu_block()
    if control is not None:
        source = source.controlled(CtrlSpec(cvs=control))
    modeled = estimate_component_resources(named.lcu_model(control))
    assert {k: v for k, v in asdict(modeled.gates).items() if v} == get_cost_value(
        source, QECGatesCost()
    ).asdict()
    assert modeled.peak_logical_qubits == get_cost_value(source, QubitCount())
    expected = (540, 86) if control is None else (552, 89)
    assert (counts.t_count, counts.unconditional["measure"]) == expected


@pytest.mark.parametrize("adjoint", [False, True])
def test_hubbard_select_named_and_packed_execution(adjoint: bool) -> None:
    outputs = []
    for packed in (False, True):
        example = HubbardLCUExample(packed=packed)
        block = example.select_block
        ancilla = block.ancilla_bitsize
        through = ancilla + block.system_bitsize
        work, slots = example.selection.requirements()
        operation = block.make_on(
            selection=tuple(range(ancilla)),
            target=tuple(range(ancilla, through)),
            workspace=tuple(range(through, through + work)),
            classical_bits=tuple(range(slots)),
            adjoint=adjoint,
        )
        ir = DynamicCircuit(through + work).append(operation).to_ir()
        assert (
            CircuitIR.from_json(ir.to_json()).validate().instructions == ir.instructions
        )
        outputs.append(ir.instructions)
    assert outputs[0] == outputs[1]


def test_hubbard_lcu_projected_complex_outputs_and_workspace() -> None:
    example = HubbardLCUExample(side=2, packed=True)
    circuit = hubbard_lcu_circuit(example)
    block = example.lcu_block()
    width = circuit.n_qubits
    reserved = (
        width - block.system_bitsize - block.ancilla_bitsize - block.resource_bitsize
    )
    instructions = circuit.to_ir().instructions
    events = sum(gate.name == "measure" for gate in instructions)
    data = {0: 1 / sqrt(3), 129: 1j / sqrt(3), 36: -1 / sqrt(3)}
    initial = {word << reserved: amplitude for word, amplitude in data.items()}
    expected = {
        index: amp for index, amp in _projected_select(data).items() if abs(amp) > 1e-12
    }
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        result = _branch(
            instructions,
            width,
            (pattern * events)[:events],
            initial,
            normalize_measurements=True,
        )
        assert all(index % (1 << reserved) == 0 for index in result)
        observed = {
            index >> reserved: amp
            for index, amp in result.items()
            if index < 1 << (block.system_bitsize + reserved)
        }
        assert observed == pytest.approx(expected, abs=8e-12)


@pytest.mark.parametrize("control", [None, 0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_hubbard_lcu_three_by_three_locked_full_output(
    control: int | None, adjoint: bool
) -> None:
    import cirq
    from qualtran import CtrlSpec
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.block_encoding.lcu_block_encoding import _lcu_block
    from qualtran.cirq_interop import BloqAsCirqGate

    from tests.unit.test_hubbard_hamiltonian_gqsp import (
        _modadd_leaf,
        _source_sparse_branch,
    )

    source = _lcu_block()
    if control is not None:
        source = source.controlled(CtrlSpec(cvs=control))
    if adjoint:
        source = source.adjoint()
    named = get_named_qubits(source.signature)
    visible = tuple(
        q for register in source.signature for q in named[register.name].flat
    )
    context = cirq.DecompositionContext(cirq.SimpleQubitManager(prefix="hubbard_lcu"))
    source_gate = source if isinstance(source, cirq.Gate) else BloqAsCirqGate(source)
    source_ops = cirq.decompose(
        source_gate.on(*visible),
        context=context,
        keep=lambda op: (
            _modadd_leaf(op) is not None
            or (
                len(op.qubits) <= 3
                and (
                    cirq.is_measurement(op)
                    or cirq.has_unitary(op)
                    or isinstance(op, cirq.ClassicallyControlledOperation)
                    or isinstance(op.gate, cirq.ResetChannel)
                )
            )
        ),
    )
    source_events = sum(map(cirq.is_measurement, source_ops))
    assert source_events > 0

    native = HubbardLCUExample(side=3, packed=True)
    ir = hubbard_lcu_circuit(native, control=control, adjoint=adjoint).to_ir()
    work = ir.n_wires - len(visible)
    assert work == 6
    events = sum(instruction.name == "measure" for instruction in ir.instructions)
    assert events == (86 if control is None else 89)
    data = {0: 1 / sqrt(3), 1 << 17: 1j / sqrt(3), (1 << 13) + (1 << 4): -1 / sqrt(3)}
    if control is not None:
        control_mask = 1 << (len(visible) - 1)
        data = {
            index | bit: phase * amplitude
            for index, amplitude in data.items()
            for bit, phase in ((0, 1 / sqrt(2)), (control_mask, 1j / sqrt(2)))
        }
    initial = {basis << work: amp for basis, amp in data.items()}
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        expected = _source_sparse_branch(
            source_ops, visible, data, (pattern * source_events)[:source_events]
        )
        result = _branch(
            ir.instructions,
            ir.n_wires,
            (pattern * events)[:events],
            initial,
            normalize_measurements=True,
        )
        assert all(index % (1 << work) == 0 for index in result)
        assert {index >> work: amp for index, amp in result.items()} == pytest.approx(
            expected, abs=8e-12
        )


@pytest.mark.parametrize("packed", (False, True))
def test_lcu_declarations_preserve_original_register_types(packed: bool) -> None:
    from qualtran import BQUInt, CtrlSpec, QAny, QBit
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        _black_box_lcu_block,
        _black_box_select_block,
        _lcu_block,
        _select_block,
    )

    example = HubbardLCUExample(packed=packed)
    select = (_black_box_select_block if packed else _select_block)()
    lcu = (_black_box_lcu_block if packed else _lcu_block)()
    pairings = [
        (example.select_signature(), select.signature),
        (example.lcu_signature(), lcu.signature),
        (
            example.lcu_signature(controlled=True),
            lcu.controlled(CtrlSpec(cvs=0)).signature,
        ),
        (
            example.lcu_signature(controlled=True),
            lcu.controlled(CtrlSpec(cvs=1)).signature,
        ),
    ]
    for native, original in pairings:
        assert len(native) == len(original)
        for actual, expected in zip(native, original, strict=True):
            assert (actual.name, actual.bitsize, actual.shape, actual.side.name) == (
                expected.name,
                expected.bitsize,
                expected.shape,
                expected.side.name,
            )
            if isinstance(expected.dtype, BQUInt):
                assert actual.dtype.encoding == "uint"
                assert actual.dtype.parameters == (
                    ("bound", expected.dtype.iteration_length),
                )
            elif isinstance(expected.dtype, QBit):
                assert actual.dtype.encoding == "bit"
            else:
                assert isinstance(expected.dtype, QAny)
                assert actual.dtype.encoding == "opaque"


def test_lcu_declared_signature_exposes_only_through_junk() -> None:
    from attrs import frozen
    from qualtran import BQUInt, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        LCUBlockEncoding as SourceLCU,
    )
    from qualtran.bloqs.block_encoding.lcu_block_encoding import _select_block
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    from flagquantum.algorithms.primitives.lcu_block_encoding import (
        source_lcu_block_signature,
    )
    from flagquantum.core.ir.program.register_types import (
        bounded_unsigned_register_type,
        opaque_register_type,
    )
    from flagquantum.core.ir.program.registers import Register, RegisterSide

    @frozen
    class DirectedPrepare(PrepareOracle):
        @property
        def selection_registers(self):
            return (SourceRegister("selection", BQUInt(2, 3), shape=(2,)),)

        @property
        def junk_registers(self):
            return (
                SourceRegister("consumed", QAny(2), side=Side.LEFT),
                SourceRegister("retained", QAny(3), shape=(2,)),
                SourceRegister("created", QAny(4), side=Side.RIGHT),
            )

    source = SourceLCU(_select_block().select, DirectedPrepare())
    selection = Register("selection", bounded_unsigned_register_type(2, 3), shape=(2,))
    retained = Register("retained", opaque_register_type(3), shape=(2,))
    target = Register("target", opaque_register_type(18))
    native = source_lcu_block_signature(
        (selection,),
        (
            Register("consumed", opaque_register_type(2), side=RegisterSide.LEFT),
            retained,
            Register("created", opaque_register_type(4), side=RegisterSide.RIGHT),
        ),
        (target,),
    )
    assert tuple(native) == (selection, retained, target)
    assert [(reg.name, reg.total_bits(), reg.side.name) for reg in native] == [
        (reg.name, reg.total_bits(), reg.side.name) for reg in source.signature
    ]
    assert native.n_qubits() == source.signature.n_qubits() == 28
