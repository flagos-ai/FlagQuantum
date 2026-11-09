"""Compare source gate flow, exact phases, and measured cleanup for controlled Z."""

from cmath import exp
from collections import Counter
from itertools import product
from math import pi, sqrt

import pytest
import torch

from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledZPowerCall,
    ControlledZRotationCall,
)
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_rz_ir,
    controlled_z_power_ir,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("angle", [0.0, 0.3, -0.7, pi / 2, 2 * pi])
def test_controlled_rz_exact_phase_and_source_gate_order(angle: float) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates.rotation import CRz
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    ports = {"ctrl": (2,), "q": (0,)}
    source = CRz(angle=angle, eps=1e-9)
    gates = controlled_rz_ir(2, 0, angle)
    reference, phases = source_gradient_gates(source, ports, ports)
    assert gates == reference
    assert tuple(leaf.eps for leaf in phases) == (5e-10, 5e-10)
    assert Counter(gate.name for gate in gates) == {"phase": 2, "cx": 2}
    expected_model = get_cost_value(source, QECGatesCost())
    actual_model = query_qec_gate_cost(ControlledZRotationCall(angle, 1e-9))
    assert actual_model.rotation == expected_model.rotation
    assert actual_model.t == expected_model.t
    assert actual_model.clifford == expected_model.clifford
    assert get_cost_value(source, QubitCount()) == 2

    state = torch.tensor(
        [complex(i + 1, 3 * i - 7) for i in range(8)], dtype=torch.complex128
    )
    state /= state.norm()
    expected = state.clone()
    for i in range(8):
        if i & 1:
            expected[i] *= exp(1j * angle * (0.5 if i & 4 else -0.5))
    actual = forced_branch(state, gates, ())
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    torch.testing.assert_close(
        forced_branch(actual, controlled_rz_ir(2, 0, -angle), ()),
        state,
        rtol=0,
        atol=2e-12,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("exponent", [0.0, 0.3, -0.25, 0.5, 1.0])
def test_cz_power_source_and_all_cleanup_branches(exponent: float) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates.rotation import CZPowGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    ports = {"q": (2, 0)}
    source = CZPowGate(exponent=exponent, eps=1e-9)
    gates = controlled_z_power_ir(2, 0, exponent, 3, 7)
    reference, phases = source_gradient_gates(source, ports, ports, work=3, bit=7)
    assert_source(gates, reference)
    assert [gate for gate in gates if gate.name == "phase"] == [
        gate for gate in reference if gate.name == "phase"
    ]
    assert len(phases) == 1 and phases[0].eps == 1e-9
    counts = Counter(gate.name for gate in gates)
    assert counts["phase"] == counts["measure"] == counts["reset"] == 1
    assert counts["t"] + counts["tdg"] == 4
    expected_model = get_cost_value(source, QECGatesCost())
    actual_model = query_qec_gate_cost(ControlledZPowerCall(exponent, 1e-9))
    assert (
        (actual_model.and_bloq, actual_model.measurement)
        == (expected_model.and_bloq, expected_model.measurement)
        == (1, 1)
    )
    assert (actual_model.t, actual_model.rotation, actual_model.clifford) == (
        expected_model.t,
        expected_model.rotation,
        expected_model.clifford,
    )
    assert get_cost_value(source, QubitCount()) == 3

    state = torch.zeros(16, dtype=torch.complex128)
    state[::2] = torch.tensor(
        [complex(i + 1, 2 * i - 3) for i in range(8)], dtype=torch.complex128
    )
    state /= state.norm()
    expected = state.clone()
    expected[[10, 14]] *= exp(1j * pi * exponent)
    inverse = controlled_z_power_ir(2, 0, -exponent, 3, 7)
    for first, second in product((0, 1), repeat=2):
        branch = forced_branch(state, gates, (first,))
        torch.testing.assert_close(branch, expected / sqrt(2), rtol=0, atol=2e-12)
        torch.testing.assert_close(
            forced_branch(branch, inverse, (second,)),
            state / 2,
            rtol=0,
            atol=2e-12,
        )


@pytest.mark.parametrize("angle", [True, float("nan"), float("inf"), 10**400])
def test_rotation_validates_concrete_angles(angle: float) -> None:
    with pytest.raises(ValueError, match="finite phase angle"):
        controlled_rz_ir(0, 1, angle)
    with pytest.raises(ValueError, match="finite phase angle"):
        controlled_z_power_ir(0, 1, angle, 2, 0)


def test_rotation_validates_disjoint_wires_and_measurement_slot() -> None:
    with pytest.raises(ValueError, match="distinct"):
        controlled_rz_ir(0, 0, 0.3)
    with pytest.raises(ValueError, match="distinct"):
        controlled_z_power_ir(0, 1, 0.3, 1, 0)
    with pytest.raises(ValueError, match="classical slot"):
        controlled_z_power_ir(0, 1, 0.3, 2, -1)
