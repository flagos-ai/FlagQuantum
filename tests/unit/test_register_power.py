"""Check source Power repetition, modeled weights and dynamic slot reuse."""

from collections import Counter
from dataclasses import asdict
from itertools import product

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledZPowerCall,
    PhaseCall,
    PowerCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.selected_call_count import (
    count_selected_calls,
)
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.primitives.power import register_power_ir
from flagquantum.core.classical_control import validate_classical_control
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.operator_schema import MAX_POWER_REPEATS
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("exponent", [1, 3, -2])
def test_power_repeats_source_rotation_without_angle_fusion(exponent: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Power, Rz
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    angle = 0.37
    single = Rz(angle)
    source = Power(single, 1) ** exponent
    body = CircuitIR(2, (Instruction("rz", (1,), {"theta": angle}),))
    gates = register_power_ir(body, abs(exponent), adjoint=exponent < 0)
    leaves = tuple(item.bloq for item, _, _ in source.decompose_bloq().iter_bloqsoqs())
    assert len(leaves) == len(gates) == abs(exponent)
    assert all(
        gate == Instruction("rz", (1,), {"theta": leaf.angle})
        for gate, leaf in zip(gates, leaves, strict=True)
    )
    assert get_cost_value(source, QubitCount()) == 1
    key = RotationCall("rz", leaves[0].angle)
    model = query_qec_gate_cost(PowerCall(key, abs(exponent)))
    assert model.rotation == get_cost_value(source, QECGatesCost()).rotation

    state = torch.tensor([1 + 1j, 2 - 1j, -3 + 1j, -2j], dtype=torch.complex128)
    state /= state.norm()
    circuit = fq.Circuit(2, dtype=torch.complex128, inputs=state)
    circuit.append(FieldOperation(2, gates))
    matrix = torch.as_tensor(
        source.tensor_contract(), dtype=torch.complex128
    ).contiguous()
    expected = torch.kron(torch.eye(2, dtype=torch.complex128), matrix) @ state
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("repetitions", [2, 3])
def test_measured_power_reuses_slots_and_cleans_every_branch(repetitions: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import CZPowGate, Power
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    exponent = -0.3
    power = Power(CZPowGate(exponent=exponent), repetitions)
    inverse_power = power.adjoint()
    width, scratch, slot = 4, 3, 2
    ports = {"q": (1, 0)}
    forward, inverse = (
        register_power_ir(
            CircuitIR(width, controlled_z_power_ir(1, 0, phase, scratch, slot)),
            repetitions,
        )
        for phase in (exponent, -exponent)
    )
    for source, gates in ((power, forward), (inverse_power, inverse)):
        leaves = tuple(
            leaf.bloq for leaf, _, _ in source.decompose_bloq().iter_bloqsoqs()
        )
        expected = tuple(
            gate
            for leaf in leaves
            for gate in source_gradient_gates(
                leaf, ports, ports, work=scratch, bit=slot
            )[0]
        )
        assert_source(gates, expected)
        assert [g.params for g in gates if g.name == "phase"] == [
            g.params for g in expected if g.name == "phase"
        ]
        assert validate_classical_control(gates, n_qubits=width) == slot + 1
        assert get_cost_value(source, QubitCount()) == width - 1
    counts = Counter(g.name for g in forward)
    model = query_qec_gate_cost(PowerCall(ControlledZPowerCall(exponent), repetitions))
    assert counts["measure"] == model.measurement == repetitions
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq == 4 * repetitions
    assert {name: count for name, count in asdict(model).items() if count} == (
        get_cost_value(power, QECGatesCost()).asdict()
    )

    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = torch.tensor([complex(i + 1, i - 2) for i in range(8)])
    initial /= initial.norm()
    source_matrix = torch.as_tensor(power.tensor_contract(), dtype=torch.complex128)
    expected = (source_matrix @ initial.reshape(4, 4)).reshape(-1)
    scale = 2 ** (-repetitions / 2)
    histories = tuple(product((0, 1), repeat=repetitions))
    for first, second in product(histories, repeat=2):
        branch = forced_branch(initial, forward, first)
        torch.testing.assert_close(branch, expected * scale, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            forced_branch(branch, inverse, second),
            initial * scale**2,
            rtol=0,
            atol=2e-12,
        )
    circuit = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(width, forward))
    run = run_dynamic(circuit, shots=8, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        run.final_states.reshape(8, -1), expected.expand(8, -1), rtol=0, atol=2e-12
    )


@pytest.mark.ftqc_reference
def test_symbolic_power_repeats_one_source_call() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Power, TGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    count = Symbol("count", positive=True, integer=True)
    source = Power(TGate(), count) ** 2
    key = PhaseCall("t")
    call = PowerCall(PowerCall(key, count), 2)
    assert count_selected_calls(call, (key,)) == {key: source.power}
    assert query_qec_gate_cost(call).t == get_cost_value(source, QECGatesCost()).t
    with pytest.raises(RuntimeError, match="Symbolic power"):
        get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="positive integer"):
        register_power_ir(CircuitIR(1, (Instruction("t", (0,)),)), count)


def test_power_validates_iteration_local_contract() -> None:
    gate = Instruction("x", (0,))
    body = CircuitIR(1, (gate,))
    for invalid in (False, 0, -1, 2.0):
        with pytest.raises(ValueError, match="positive integer"):
            register_power_ir(body, invalid)
    with pytest.raises(ValueError, match="positive integer"):
        PowerCall(PhaseCall("t"), 0)
    with pytest.raises(ValueError, match="exceeds"):
        register_power_ir(body, MAX_POWER_REPEATS + 1)
    with pytest.raises(ValueError, match="result requests"):
        register_power_ir(
            CircuitIR(1, (gate,), measurements=(MeasurementNode("sample", (0,)),)), 2
        )
    measured = Instruction(
        "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}
    )
    reads_first = Instruction(
        "x", (0,), metadata={"is_dynamic": True, "conditions": ((0, 1),)}
    )
    with pytest.raises(ValidationError, match="read before measurement"):
        register_power_ir(CircuitIR(1, (reads_first, measured)), 2)
    with pytest.raises(CapabilityError, match="dynamic"):
        register_power_ir(CircuitIR(1, (measured,)), 2, adjoint=True)
    first, second = register_power_ir(body, 2)
    assert first == second == gate and first is not second and first is not gate
