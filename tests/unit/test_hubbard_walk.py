"""Check the complete-signal Hubbard walk and the locked analytical model."""

from collections import defaultdict
from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.algorithms.primitives.hubbard_walk import HubbardWalk
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.compiler.registers.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("side", [2, 3, 5])
def test_hubbard_walk_locked_schedule_and_resources(side):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.walk_operator import (
        get_walk_operator_for_hubbard_model,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = get_walk_operator_for_hubbard_model(side, side, 2, 8)
    walk = HubbardWalk(PrepareHubbard(side, 2, 8))
    model = walk.source_resource_model()
    stages = tuple(original.decompose_bloq().iter_bloqsoqs())
    assert [call.component.name for call in model.calls] == [
        type(instance.bloq).__name__ for instance, _, _ in stages
    ]
    assert set(stages[1][1]) == dict(walk.prepare.selection_layout).keys()
    assert walk.alpha == original.sum_of_lcu_coefficients
    for component, source in (
        (model, original),
        *(
            (edge.component, instance.bloq)
            for edge, (instance, _, _) in zip(model.calls, stages, strict=True)
        ),
    ):
        result = estimate_component_resources(component)
        assert {k: v for k, v in asdict(result.gates).items() if v} == (
            get_cost_value(source, QECGatesCost()).asdict()
        )
        assert result.peak_logical_qubits == get_cost_value(source, QubitCount())


def _projected_select(vector):
    """Average signed Pauli actions using the original 2×2 PREPARE probabilities."""
    result = defaultdict(complex)
    for x, y, spin, axis, direction in product((0, 1), repeat=5):
        p = 2 * (2 * x + y) + spin
        nx, ny = (x ^ 1, y) if axis == 0 else (x, y ^ 1)
        q = 2 * (2 * nx + ny) + spin
        if direction:
            p, q = q, p
        for source, amp in vector.items():
            # The two Majoranas act as Y on p, then X on q, with
            # separate Jordan–Wigner prefixes and scalar -i.
            p_bit, q_bit = 1 << (7 - p), 1 << (7 - q)
            middle = source ^ p_bit
            signs = (source >> (8 - p)).bit_count() + (middle >> (8 - q)).bit_count()
            signs += int(bool(source & p_bit))
            result[middle ^ q_bit] += (-1) ** signs * amp * (0.25 / 32)

            # U=1, V=0 gives -Z on the selected site and spin.
            onsite = 2 * (2 * x + y) + spin
            z = -1 if source & (1 << (7 - onsite)) else 1
            result[source] -= z * amp * (0.60 / 32)

            # U=0, V=1 fixes the Majorana spin to zero and uses the
            # source's independent y/x address order for the spin-up Z.
            down, up = 2 * (2 * x + y), 4 + 2 * y + x
            parity = bool(source & (1 << (7 - down))) ^ bool(source & (1 << (7 - up)))
            result[source] += (-1) ** parity * amp * (0.15 / 32)
    return result


def test_hubbard_walk_branches_chebyshev_projection_and_adjoint():
    prepare = PrepareHubbard(2, 2, 8)
    walk = HubbardWalk(prepare)
    logical = sum(width for _, width in walk.layout)
    work, _ = walk.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), walk.layout)
    workspace = tuple(range(logical, width))
    prep_ports = {name: ports[name] for name, _ in prepare.layout}
    prep_work, _ = prepare.requirements()
    prep = prepare.make_on(prep_ports, workspace=workspace[:prep_work])
    unprep = prepare.make_on(prep_ports, workspace=workspace[:prep_work], adjoint=True)
    forward = walk.make_on(ports, workspace=workspace)
    backward = walk.make_on(ports, workspace=workspace, adjoint=True)
    counts = count_emitted_instructions(DynamicCircuit(width).append(forward).to_ir())
    assert (counts.t_count, counts.unconditional["measure"], work) == (232, 30, 10)

    vector = {0: 1 / sqrt(3), 129: 1j / sqrt(3), 36: -1 / sqrt(3)}
    first = _projected_select(vector)
    second = _projected_select(first)
    initial = {word << work: amp for word, amp in vector.items()}
    for steps, expected in (
        ((forward,), first),
        (
            (forward, forward),
            {key: 2 * second[key] - vector.get(key, 0) for key in second},
        ),
        ((forward, backward), vector),
    ):
        circuit = DynamicCircuit(width).append(prep)
        for step in steps:
            circuit.append(step)
        ir = circuit.append(unprep).to_ir()
        events = sum(gate.name == "measure" for gate in ir.instructions)
        projection = {word: amp for word, amp in expected.items() if abs(amp) > 1e-12}
        for pattern in ((0,), (1,), (0, 1), (1, 0)):
            state = _branch(
                ir.instructions,
                width,
                (pattern * events)[:events],
                initial,
                normalize_measurements=True,
            )
            assert all(index % (1 << work) == 0 for index in state)
            actual = {
                index >> work: amp
                for index, amp in state.items()
                if index < 1 << (8 + work)
            }
            assert actual == pytest.approx(projection, abs=8e-12)


def test_hubbard_walk_marks_both_direction_bits():
    prepare = PrepareHubbard(2, 2, 8)
    walk = HubbardWalk(prepare)
    logical = sum(width for _, width in walk.layout)
    work, _ = walk.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), walk.layout)
    workspace = tuple(range(logical, width))
    prep_work, prep_slots = prepare.requirements()
    select_work, select_slots = SelectHubbard(2).requirements()
    select = SelectHubbard(2)._emit(
        ports, workspace[:select_work], tuple(range(select_slots))
    )
    full = (
        DynamicCircuit(width).append(walk.make_on(ports, workspace=workspace)).to_ir()
    )
    native = full.instructions[len(select) :]
    args = (ports, workspace[1 : 1 + prep_work], tuple(range(prep_slots)))
    forward = prepare._emit(*args, False)
    backward = prepare._emit(*args, True)
    signal = tuple(wire for name, _ in prepare.selection_layout for wire in ports[name])
    reflection_work, slots = prepare_reflection_requirements(
        len(signal), controlled=False
    )
    narrow = prepare_reflection_ir(
        signal,
        workspace[:reflection_work],
        tuple(range(slots)),
        prepare=forward,
        unprepare=backward,
        global_phase=-1,
    )
    source_walk = HubbardWalk(prepare, reflect_directions=False)
    source_work, _ = source_walk.requirements()
    source_ir = (
        DynamicCircuit(width)
        .append(source_walk.make_on(ports, workspace=workspace[:source_work]))
        .to_ir()
    )
    assert source_ir.instructions == (*select, *narrow)

    def run(gates, state):
        events = sum(gate.name == "measure" for gate in gates)
        return _branch(gates, width, (0,) * events, state, normalize_measurements=True)

    direction_mask = sum(1 << (width - wire - 1) for wire in ports["temp"])
    selection_word = 1 << (width - ports["p_x"][0] - 1)
    state = run(source_ir.instructions, {selection_word: 1 + 0j})
    assert all(word & (direction_mask | ((1 << work) - 1)) == 0 for word in state)
    for direction in ports["temp"]:
        prepared = run(forward, {1 << (width - direction - 1): 1 + 0j})
        assert run(narrow, prepared) == pytest.approx(prepared, abs=5e-12)
        assert run(native, prepared) == pytest.approx(
            {word: -amp for word, amp in prepared.items()}, abs=5e-12
        )
