"""THC walk source costs, complete-signal phases, and feedback ownership."""

from dataclasses import asdict, replace
from math import sqrt, tau

import pytest

from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.algorithms.primitives.thc_chemistry_prepare import PrepareTHCChemistry
from flagquantum.algorithms.primitives.thc_chemistry_walk import THCChemistryWalk
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.core.classical_control import validate_classical_control
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_sparse_chemistry_prepare import _branch
from tests.unit.test_sparse_chemistry_walk import _reflect
from tests.unit.test_thc_chemistry_prepare import _tensors

pytestmark = pytest.mark.unit

NETWORKS = (((0, 1, tau / 8),), ((0, 1, -tau / 16),))
ONE_BODY = ((), ((0, 1, tau / 32),))


def _walk(log, work=0, measured=False):
    data = thc_chemistry_data(*_tensors(2), probability_bits=1, log_block_size=log)
    return THCChemistryWalk(
        PrepareTHCChemistry(data), NETWORKS, ONE_BODY, work, measured
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "log,angle_bits,kr1,kr2", [(0, 8, 1, 1), (1, 8, 1, 1), (1, 5, 2, 4)]
)
def test_source_thc_walk_costs(log, angle_bits, kr1, kr2):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.chemistry.thc.prepare import PrepareTHC
    from qualtran.bloqs.chemistry.thc.select_bloq import SelectTHC
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    prepare = PrepareTHC.from_hamiltonian_coeffs(
        *_tensors(2), num_bits_state_prep=1, log_block_size=log
    )
    select = SelectTHC(2, 4, angle_bits, 1, kr1=kr1, kr2=kr2)
    source = QubitizationWalkOperator(SelectBlockEncoding(select, prepare))
    assert [
        type(child.bloq) for child, *_ in source.decompose_bloq().iter_bloqsoqs()
    ] == [SelectBlockEncoding, ReflectionUsingPrepare]
    native = _walk(log)
    assert native.alpha == source.sum_of_lcu_coefficients == 32
    options = dict(num_bits_theta=angle_bits, kr1=kr1, kr2=kr2)
    assert {
        key: value
        for key, value in asdict(native.source_gate_cost(**options)).items()
        if value
    } == (get_cost_value(source, QECGatesCost()).asdict())
    if not log:
        with pytest.raises(RuntimeError, match="iteration dimensions are 0"):
            get_cost_value(source, QubitCount())
        with pytest.raises(ValueError, match="junk_target0_"):
            native.source_resource_model(**options)
    else:
        model = native.source_resource_model(**options)
        assert_source_model(model.calls[1].component, source.reflect)
        assert_source_model(model, source)


def _bind(walk):
    width = sum(size for _, size in walk.layout)
    work, slots = walk.requirements()
    ports = partition_register(tuple(range(width)), walk.layout)
    scratch = tuple(range(width, width + work))
    bits = tuple(range(20, 20 + slots))
    return width + work + 1, ports, scratch, bits


@pytest.mark.parametrize("log,select_work,measured", [(0, 0, False), (1, 26, True)])
def test_thc_walk_complex_signal_and_inverse(log, select_work, measured):
    walk = _walk(log, select_work, measured)
    width, ports, scratch, bits = _bind(walk)
    bound = walk.make_on(ports, workspace=scratch, classical_bits=bits)
    forward = DynamicCircuit(width).append(bound).to_ir()
    inverse = (
        DynamicCircuit(width)
        .append(
            walk.make_on(ports, workspace=scratch, classical_bits=bits, adjoint=True)
        )
        .to_ir()
    )
    counts = count_emitted_instructions(forward)
    assert (
        counts.unconditional["measure"]
        == count_emitted_instructions(inverse).unconditional["measure"]
    )
    assert counts.declared_qubits == width

    prep_work, prep_slots = walk.prepare.requirements()
    prep = walk.prepare.make_on(
        {name: ports[name] for name, _ in walk.layout[:-2]},
        workspace=scratch[1 : 1 + prep_work],
        classical_bits=bits[:prep_slots],
    ).instructions
    events = sum(gate.name == "measure" for gate in prep)
    signal = _branch(prep, width, (0,) * events, normalize_measurements=True)
    flag_mask = 1 << (width - 1 - ports["nu_eq_mp1"][0])
    one = next(index for index in signal if index & flag_mask)
    two = next(index for index in signal if not index & flag_mask and index)
    system_mask = (1 << (width - 1 - ports["sys_a"][1])) | (
        1 << (width - 1 - ports["sys_b"][1])
    )
    initial = {
        one | system_mask | 1: (1 + 2j) / sqrt(7),
        two | system_mask: (1 - 1j) / sqrt(7),
    }
    select = replace(bound.select, workspace=(), measurement_erasure=False)
    exact = DynamicCircuit(width).append(select).to_ir().instructions
    selected = _branch(exact, width, (), initial)
    # Exact-angle networks use multiples of the 8-bit rotation grid.
    assert _branch(exact, width, (), selected) == pytest.approx(initial, abs=2e-12)
    low_mask = (1 << (len(scratch) + 5)) - 1
    expected = _reflect(selected, signal, low_mask)
    events = counts.unconditional["measure"]
    for pattern in ((0, 1), (1, 0)):
        history = (pattern * events)[:events]
        actual = _branch(
            forward.instructions, width, history, initial, normalize_measurements=True
        )
        assert actual == pytest.approx(expected, abs=2e-12)
        restored = _branch(
            inverse.instructions, width, history, expected, normalize_measurements=True
        )
        assert restored == pytest.approx(initial, abs=2e-12)


@pytest.mark.parametrize("adjoint", [False, True])
def test_measured_thc_walk_reserves_selector_slots_after_existing_and_reflection_slots(
    adjoint,
):
    walk = _walk(1, 26, True)
    width, ports, scratch, bits = _bind(walk)
    circuit = DynamicCircuit(width)
    circuit.measure(width - 1, classical_bit=200)
    bound = walk.make_on(ports, workspace=scratch, classical_bits=bits, adjoint=adjoint)
    circuit.append(bound)
    ir = circuit.to_ir()
    all_slots = {
        gate.metadata["classical_bit"]
        for gate in ir.instructions
        if gate.name == "measure"
    }
    assert set(bits) <= all_slots
    assert all_slots - set(bits) - {200} == set(range(201, 216))
    assert validate_classical_control(ir.instructions, n_qubits=width) == 216


def test_thc_walk_binding_failure_leaves_circuit_unchanged():
    walk = replace(_walk(1), networks=(((0, 0, 0.2),), ()))
    width, ports, scratch, bits = _bind(walk)
    circuit = DynamicCircuit(width)
    circuit.measure(width - 1, classical_bit=200)
    before = circuit.to_ir().instructions
    inverse = walk.make_on(ports, workspace=scratch, classical_bits=bits, adjoint=True)
    with pytest.raises(ValueError, match="adjacent orbital offsets"):
        circuit.append(inverse)
    assert circuit.to_ir().instructions == before
