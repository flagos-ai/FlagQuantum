"""Verify Hadamard control dispatch with original predicates and complex branches."""

from collections import Counter
from dataclasses import asdict
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.compound_control_resource import (
    compound_control_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.control.multi_control import multi_control_h_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("values", [(), (1,), (0,), (1, 1), (0, 1), (0, 1, 0)])
def test_hadamard_control_matches_source(values: tuple[int, ...]) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = len(values)
    controls = tuple(reversed(range(n)))
    target = n
    work = tuple(range(n + 2, n + 2 + max(0, n - 1)))
    bits = tuple(range(19, 19 + len(work)))
    gate = Hadamard()
    source = (
        gate
        if not values
        else gate.controlled(CtrlSpec(cvs=values[0] if n == 1 else np.asarray(values)))
    )
    gates = multi_control_h_ir(controls, values, target, work, bits)
    if values in ((), (1,)):
        name = "ch" if values else "h"
        expected_qubits = tuple(cirq.LineQubit(wire) for wire in (*controls, target))
        bound = (
            {"ctrl": expected_qubits[:1], "target": expected_qubits[1:]}
            if values
            else {"q": expected_qubits}
        )
        operation, _ = source.as_cirq_op(qubit_manager=None, **bound)
        assert operation.qubits == expected_qubits
        assert operation.gate == (cirq.ControlledGate(cirq.H) if values else cirq.H)
        assert gates == (Instruction(name, (*controls, target)),)
    else:
        ports = {"ctrl": controls, "q": (target,)}
        assert_source(
            gates,
            reference_arithmetic_gates(
                source,
                ports,
                work,
                dict(zip(work, bits, strict=True)),
            ),
        )

    root = CliffordCall("h", 1)
    key = compound_control_key(root, values) if values else root
    assert key is not None
    model = query_qec_gate_cost(key)
    assert {name: count for name, count in asdict(model).items() if count} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == n + 1 + len(work)
    counts = Counter(g.name for g in gates)
    assert counts["measure"] == len(work)
    assert counts["t"] + counts["tdg"] == 4 * len(work)

    # The final data wire is an entangled spectator and clean work follows it.
    width = n + 2 + len(work)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    data = torch.tensor(
        [complex(i + 1, 3 * i - 2) for i in range(1 << (n + 2))],
        dtype=torch.complex128,
    )
    state[:: 1 << len(work)] = data / data.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (*controls, target, n + 1, *work)
    inverse_order = tuple(order.index(wire) for wire in range(width))
    expected = (
        (matrix @ state.reshape((2,) * width).permute(order).reshape(1 << (n + 1), -1))
        .reshape((2,) * width)
        .permute(inverse_order)
        .reshape(-1)
    )
    histories = tuple(product((0, 1), repeat=len(work)))
    scale = 2 ** (-len(work) / 2)
    for first in histories:
        branch = forced_branch(state, gates, first)
        torch.testing.assert_close(branch, expected * scale, rtol=0, atol=2e-12)
        for second in histories:
            torch.testing.assert_close(
                forced_branch(branch, gates, second),
                state * scale**2,
                rtol=0,
                atol=2e-12,
            )


def test_hadamard_rejects_target_aliases() -> None:
    with pytest.raises(ValueError, match="distinct"):
        multi_control_h_ir((0,), (1,), 0, (), ())


def test_measured_hadamard_appends_to_dynamic_circuit() -> None:
    circuit = DynamicCircuit(4, dtype=torch.complex128)
    circuit.h(0).h(1).s(0)
    initial = circuit.state().flatten()
    gates = multi_control_h_ir((0, 1), (0, 1), 2, (3,), (0,))
    circuit.append(FieldOperation(4, gates))
    result = run_dynamic(circuit, shots=8, seed=5, strategy="trajectory")
    expected = initial.clone()
    expected[4] = initial[4] / 2**0.5
    expected[6] = initial[4] / 2**0.5
    torch.testing.assert_close(
        result.final_states.reshape(8, -1),
        expected.expand(8, -1),
        rtol=0,
        atol=2e-12,
    )
