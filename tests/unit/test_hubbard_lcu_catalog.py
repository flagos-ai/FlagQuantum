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
