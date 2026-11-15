"""Check the complete-signal Hubbard walk and the locked analytical model."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from itertools import product
from math import pi, sqrt

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
from flagquantum.compiler.registers.control.multi_control import multi_control_z_ir
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
)
from flagquantum.core.ir import Instruction
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
    walk = HubbardWalk.from_lattice(side, 2, 8)
    assert not walk.reflect_directions
    assert dict(walk.source_layout) == {
        reg.name: reg.total_bits() for reg in original.signature
    }
    assert walk.prepare.layout[-1] == ("temp", 2)
    assert original.block_encoding.resource_bitsize == 2
    assert original.block_encoding.junk_registers == ()
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


@pytest.mark.ftqc_reference
def test_locked_walk_binds_the_original_reflection_operands_and_phase():
    pytest.importorskip("qualtran")
    import cirq
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.walk_operator import (
        get_walk_operator_for_hubbard_model,
    )

    source = get_walk_operator_for_hubbard_model(2, 2, 1, 2)
    reflect = source.reflect
    named = get_named_qubits(reflect.signature)
    backward, flip, mark, scalar, clear, forward = cirq.flatten_op_tree(
        reflect.decompose_from_registers(
            context=cirq.DecompositionContext(cirq.SimpleQubitManager()), **named
        )
    )
    assert backward == cirq.inverse(forward)
    assert flip == clear and flip.gate == cirq.X
    assert scalar.gate.coefficient == -1

    walk = HubbardWalk.from_lattice(2, 1, 2)
    prepare = walk.prepare
    logical = sum(width for _, width in walk.layout)
    work, slots = walk.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), walk.layout)
    workspace = tuple(range(logical, width))
    mapping = {
        soq: wire
        for name, qubits in named.items()
        for soq, wire in zip(qubits.flat, ports[name], strict=True)
    }
    mapping.update(zip(forward.qubits[-2:], ports["temp"], strict=True))
    mapping[flip.qubits[0]] = workspace[0]
    assert tuple(mapping[q] for q in forward.qubits) == tuple(
        wire for name, _ in prepare.layout for wire in ports[name]
    )
    controls = tuple(mapping[q] for q in mark.qubits[:-1])
    values = tuple(mark.gate.cvs)
    assert values == (0,) * len(controls)
    target = mapping[mark.qubits[-1]]
    assert target == workspace[0]

    prep_work, prep_slots = prepare.requirements()
    args = (ports, workspace[1 : 1 + prep_work], tuple(range(prep_slots)))
    phase = Instruction("phase", (target,), params={"theta": pi})
    x = Instruction("x", (target,))
    source_reflection = (
        *prepare._emit(*args, True),
        x,
        *multi_control_z_ir(
            controls, values, target, workspace[1:], tuple(range(slots))
        ),
        phase,
        x,
        *prepare._emit(*args, False),
    )
    select = SelectHubbard(2)
    select_work, select_slots = select.requirements()
    select_args = (ports, workspace[:select_work], tuple(range(select_slots)))
    for adjoint, expected in (
        (False, (*select._emit(*select_args), *source_reflection)),
        (True, (*source_reflection, *select._emit(*select_args, adjoint=True))),
    ):
        ir = (
            DynamicCircuit(width)
            .append(walk.make_on(ports, workspace=workspace, adjoint=adjoint))
            .to_ir()
        )
        assert ir.instructions == expected

    bound_forward = walk.make_on(ports, workspace=workspace)
    bound_backward = walk.make_on(ports, workspace=workspace, adjoint=True)
    circuit = DynamicCircuit(width).append(bound_forward).append(bound_backward)
    stream = circuit.to_ir().instructions
    count = count_emitted_instructions(
        DynamicCircuit(width).append(bound_forward).to_ir()
    )
    assert (count.t_count, count.unconditional["measure"], work) == (224, 28, 8)
    events = sum(gate.name == "measure" for gate in stream)
    selection_bit = 1 << (width - ports["p_x"][0] - 1)
    system_bit = 1 << (width - ports["target"][0] - 1)
    initial = {0: 1 / sqrt(2), selection_bit | system_bit: 1j / sqrt(2)}
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        state = _branch(
            stream,
            width,
            (pattern * events)[:events],
            initial,
            normalize_measurements=True,
        )
        assert state == pytest.approx(initial, abs=5e-12)


def _projected_select(vector: Mapping[int, complex]) -> dict[int, complex]:
    """Average signed Pauli actions using the original 2×2 PREPARE probabilities."""
    result: dict[int, complex] = defaultdict(complex)
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
