"""Sparse chemistry walk signal semantics and locked resource accounting."""

from itertools import product

import numpy as np
import pytest

from flagquantum.algorithms.primitives.sparse_chemistry_data import (
    sparse_chemistry_data,
)
from flagquantum.algorithms.primitives.sparse_chemistry_prepare import (
    PrepareSparseChemistry,
)
from flagquantum.algorithms.primitives.sparse_chemistry_walk import SparseChemistryWalk
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.compiler.registers.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _integrals(spatial):
    one = np.eye(spatial)
    one[0, 1] = one[1, 0] = -0.75
    two = np.zeros((spatial,) * 4)
    two[0, 0, 0, 0] = -0.5
    return one, two


def _prepare(spatial, log):
    return PrepareSparseChemistry(
        sparse_chemistry_data(
            *_integrals(spatial), probability_bits=1, log_block_size=log
        )
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("spatial,log", tuple(product((2, 3), (0, 1))))
def test_sparse_walk_locked_aggregate_model(spatial, log):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.chemistry.sparse import PrepareSparse, SelectSparse
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    one, two = _integrals(spatial)
    prepare = PrepareSparse.from_hamiltonian_coeffs(
        2 * spatial, one, two, num_bits_state_prep=1, log_block_size=log
    )
    block = SelectBlockEncoding(select=SelectSparse(2 * spatial), prepare=prepare)
    source = QubitizationWalkOperator(block)
    assert [
        type(child.bloq) for child, *_ in source.decompose_bloq().iter_bloqsoqs()
    ] == [
        SelectBlockEncoding,
        ReflectionUsingPrepare,
    ]
    walk = SparseChemistryWalk(_prepare(spatial, log))
    assert walk.alpha == source.sum_of_lcu_coefficients
    model = walk.source_resource_model()
    assert_source_model(model.calls[1].component, source.reflect)
    assert_source_model(model, source)


@pytest.mark.parametrize("log,leak", [(0, 3 / 64), (1, 1 / 32)])
def test_source_mark_with_fresh_junk_has_leakage_under_coherent_prepare(log, leak):
    prepare = _prepare(2, log)
    layout = (*prepare.selection_layout, *prepare.junk_layout)
    signal = sum(bits for _, bits in prepare.selection_layout)
    width = sum(bits for _, bits in layout)
    ports = partition_register(tuple(range(width)), layout)
    prep_work, prep_slots = prepare.requirements()
    mark_work, mark_slots = prepare_reflection_requirements(signal, controlled=False)
    scratch = tuple(range(width + 1, width + 1 + prep_work))
    slots = tuple(range(prep_slots))
    forward = prepare.make_on(ports, workspace=scratch, classical_bits=slots)
    inverse = prepare.make_coherent_adjoint_on(
        ports, workspace=scratch, classical_bits=slots
    )
    gates = prepare_reflection_ir(
        tuple(range(signal)),
        tuple(range(width, width + mark_work)),
        tuple(range(mark_slots)),
        prepare=forward.instructions,
        unprepare=inverse.instructions,
        global_phase=-1,
    )
    events = sum(gate.name == "measure" for gate in gates)
    state = _branch(gates, width + mark_work, (0,) * events)
    junk_mask = ((1 << (width - signal)) - 1) << mark_work
    dirty_probability = sum(
        abs(amp) ** 2 for index, amp in state.items() if index & junk_mask
    )
    assert dirty_probability * 2**events == pytest.approx(leak)
    assert all(not index & ((1 << mark_work) - 1) for index in state)


def _select(state, ports, width):
    result = {}

    def bit(index, wire):
        return (index >> (width - wire - 1)) & 1

    for index, amplitude in state.items():
        flag = bit(index, ports["flag_1b"][0])
        if flag:
            amplitude *= 1j
        for orbital, spin, is_y in (
            ("p", "alpha", True),
            ("q", "alpha", False),
            ("r", "beta", True),
            ("s", "beta", False),
        ):
            if orbital in ("p", "q") and not flag:
                continue
            address = 2 * bit(index, ports[orbital][0]) + bit(index, ports[spin][0])
            targets = ports["sys"]
            amplitude *= (-1) ** sum(bit(index, wire) for wire in targets[:address])
            if is_y:
                amplitude *= -1j if bit(index, targets[address]) else 1j
            index ^= 1 << (width - targets[address] - 1)
        result[index] = amplitude
    return result


def _reflect(state, signal, low_mask):
    overlaps = {}
    for index, amplitude in state.items():
        key = index & low_mask
        overlaps[key] = (
            overlaps.get(key, 0)
            + signal.get(index & ~low_mask, 0).conjugate() * amplitude
        )
    result = {index: -amplitude for index, amplitude in state.items()}
    for key, overlap in overlaps.items():
        for high, coefficient in signal.items():
            index = high | key
            result[index] = result.get(index, 0) + 2 * coefficient * overlap
    return {index: amp for index, amp in result.items() if abs(amp) > 1e-12}


@pytest.mark.parametrize(
    "log,source_t,source_peak,emitted_t,emitted_measurements,total_wires",
    [(0, 152, 34, 278, 38, 50), (1, 196, 47, 418, 47, 76)],
)
def test_complete_sparse_walk_complex_branches_and_inverse(
    log, source_t, source_peak, emitted_t, emitted_measurements, total_wires
):
    walk = SparseChemistryWalk(_prepare(2, log))
    logical = sum(bits for _, bits in walk.layout)
    work, slots = walk.requirements()
    n = logical + work + 1  # Untouched spectator at the least-significant bit.
    ports = partition_register(tuple(range(logical)), walk.layout)
    scratch = tuple(range(logical, logical + work))
    bits = tuple(range(slots))
    forward = walk.make_on(ports, workspace=scratch, classical_bits=bits)
    inverse = walk.make_on(ports, workspace=scratch, classical_bits=bits, adjoint=True)
    circuit = DynamicCircuit(n).append(forward)
    inverse_ir = DynamicCircuit(n).append(inverse).to_ir()
    report = estimate_ftqc_resources(
        walk.source_resource_model(), emitted_ir=circuit.to_ir()
    )
    assert report.emitted is not None
    assert report.emitted.declared_qubits == n == total_wires + 1
    assert report.emitted.instruction_qubits == total_wires
    assert report.emitted.t_count == emitted_t
    assert report.emitted.unconditional["measure"] == emitted_measurements
    assert report.logical.peak_logical_qubits == source_peak
    assert report.logical.gates.total_t_count() == source_t

    prep_ports = {name: ports[name] for name, _ in walk.layout[:-1]}
    prep_work, prep_slots = walk.prepare.requirements()
    prep = walk.prepare.make_on(
        prep_ports,
        workspace=scratch[1 : 1 + prep_work],
        classical_bits=bits[:prep_slots],
    ).instructions
    prep_events = sum(gate.name == "measure" for gate in prep)
    signal = {
        index: amplitude * 2 ** (prep_events / 2)
        for index, amplitude in _branch(prep, n, (0,) * prep_events).items()
    }
    state = {
        index
        | (system << (work + 1))
        | spectator: amplitude * (1 + 1j * (2 * spectator - 1)) / (2 * np.sqrt(2))
        for index, amplitude in signal.items()
        for system in (3, 12)
        for spectator in (0, 1)
    }
    expected = _reflect(_select(state, ports, n), signal, (1 << (work + 5)) - 1)
    gates = circuit.to_ir().instructions
    events = sum(gate.name == "measure" for gate in gates)
    assert events == sum(gate.name == "measure" for gate in inverse_ir.instructions)
    norm = 2 ** (-events / 2)
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        actual = _branch(gates, n, history, state)
        assert actual == pytest.approx(
            {i: a * norm for i, a in expected.items()}, abs=2e-12
        )
        # Feed the independent expected state to the adjoint to check its phases.
        restored = _branch(inverse_ir.instructions, n, history, expected)
        assert restored == pytest.approx(
            {i: a * norm for i, a in state.items()}, abs=2e-12
        )
