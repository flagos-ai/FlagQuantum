"""Verify Hubbard SELECT's source resources and exact complex Pauli action."""

from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("side,controlled", tuple(product((2, 3, 5), (False, True))))
def test_hubbard_select_locked_signature_and_cost(side, controlled):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = Source(side, side, control_val=1 if controlled else None)
    select = SelectHubbard(side, controlled)
    assert select.layout == tuple(
        (reg.name, reg.total_bits()) for reg in original.signature
    )
    model = estimate_component_resources(select.source_resource_model())
    assert {name: value for name, value in asdict(model.gates).items() if value} == (
        get_cost_value(original, QECGatesCost()).asdict()
    )
    assert model.peak_logical_qubits == get_cost_value(original, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("side", [2, 3, 5])
def test_hubbard_select_controlled_gate_dependencies_match_source(side):
    pytest.importorskip("qualtran")
    import cirq
    from qualtran import BQUInt, QAny, Register
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )
    from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit

    select = SelectHubbard(side, controlled=True)
    logical = sum(size for _, size in select.layout)
    work, slots = select.requirements()
    workspace = tuple(range(logical, logical + work))
    ports = partition_register(tuple(range(logical)), select.layout)
    source = Source(side, side, control_val=1)
    source_ports = ports
    if side > 2:
        # The complete DAG export reuses temporaries in a different order.
        # Read the source's final unary consumer directly to retain its traversal.
        bitsize = (side - 1).bit_length()
        source = ApplyGateToLthQubit(
            (
                Register("y", BQUInt(bitsize, side)),
                Register("x", BQUInt(bitsize, side)),
            ),
            nth_gate=lambda *_: cirq.Z,
            control_regs=Register("control", QAny(2)),
        )
        source_ports = {
            "control": (*ports["V"], *ports["control"]),
            "y": ports["q_y"],
            "x": ports["q_x"],
            "target": ports["target"][side**2 :],
        }
    original = reference_arithmetic_gates(
        source,
        source_ports,
        workspace,
        dict(zip(workspace, workspace, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
        preserve_components=True,
    )
    circuit = DynamicCircuit(logical + work).append(
        select.make_on(ports, workspace=workspace)
    )
    actual = circuit.to_ir().instructions
    if side > 2:
        actual = actual[-len(original) :]

    def feedback_by_measured_wire(stream):
        # The native emitter reuses logical slots across the three children.
        active = {}
        result = []
        for gate in stream:
            metadata = dict(gate.metadata)
            if gate.name == "measure":
                wire = gate.wires[0]
                active[metadata["classical_bit"]] = wire
                metadata["classical_bit"] = wire
            if "conditions" in metadata:
                metadata["conditions"] = tuple(
                    (active[bit], value) for bit, value in metadata["conditions"]
                )
            result.append(Instruction(gate.name, gate.wires, gate.params, metadata))
        return tuple(result)

    assert slots == work - 1
    assert_source(
        feedback_by_measured_wire(actual), feedback_by_measured_wire(original)
    )


@pytest.mark.parametrize("side,controlled", tuple(product((2, 3), (False, True))))
def test_hubbard_select_complex_amplitudes_branches_and_work(side, controlled):
    select = SelectHubbard(side, controlled)
    logical = sum(size for _, size in select.layout)
    work, _ = select.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), select.layout)
    operation = select.make_on(ports, workspace=tuple(range(logical, width)))
    ir = DynamicCircuit(width).append(operation).to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert (counts.t_count, events, work) == {
        (2, False): (102, 15, 3),
        (2, True): (114, 18, 4),
        (3, False): (230, 40, 5),
        (3, True): (242, 43, 6),
    }[side, controlled]

    def pack(fields):
        result = 0
        for name, size in select.layout:
            result = (result << size) | fields[name]
        return result << work

    initial, expected = {}, {}
    n = 2 * side**2
    rows = tuple(
        product(range(1 + int(controlled)), (0, 1), (0, 1), range(side), range(side))
    )
    for row, (ctrl, u, v, x, y) in enumerate(rows):
        qx, qy = (x + 1) % side, (y + 1) % side
        alpha, beta = row % 2, (row // 2) % 2
        data = (12345 * (row + 1)) % (1 << n)
        fields = dict(
            control=ctrl,
            U=u,
            V=v,
            p_x=x,
            p_y=y,
            q_x=qx,
            q_y=qy,
            alpha=alpha,
            beta=beta,
            target=data,
        )
        amp = 1j**row / sqrt(len(rows))
        initial[pack(fields)] = amp
        if not controlled or ctrl:
            p = 2 * (x * side + y) + alpha
            q = p if v else 2 * (qx * side + qy) + beta
            p_bit, q_bit = 1 << (n - p - 1), 1 << (n - q - 1)
            amp *= (1j if not data & p_bit else -1j) * (-1) ** (
                (data >> (n - p)).bit_count()
            )
            data ^= p_bit
            amp *= (-1) ** ((data >> (n - q)).bit_count())
            data ^= q_bit
            amp *= -1j * (-1) ** u
            if v:
                spin_up = side**2 + qy * side + qx
                if data & (1 << (n - spin_up - 1)):
                    amp *= -1
        fields["target"] = data
        expected[pack(fields)] = amp

    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        result = _branch(
            ir.instructions, width, history, initial, normalize_measurements=True
        )
        assert result == pytest.approx(expected, abs=5e-12)
        assert _branch(
            ir.instructions,
            width,
            tuple(1 - bit for bit in history),
            result,
            normalize_measurements=True,
        ) == pytest.approx(initial, abs=5e-12)
