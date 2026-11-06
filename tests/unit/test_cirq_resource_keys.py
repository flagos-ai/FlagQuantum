"""Check the resource-only Cirq boundary against the locked gate mapper."""

from dataclasses import asdict, fields

import pytest

from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledRotationCall,
    ControlledSwapCall,
    ControlledZPowerCall,
    ControlledZRotationCall,
    FixedTCall,
    PhaseCall,
    RotationCall,
    ZeroCostCall,
)
from flagquantum.ecosystem.cirq.resource_keys import cirq_gate_resource_key

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_source_recognition_order_and_gate_costs() -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from sympy import Symbol

    theta = Symbol("theta", real=True)
    cases = (
        (cirq.T, "TGate", PhaseCall("t")),
        (cirq.T**-1, "TGate", PhaseCall("t", True)),
        (cirq.S**-1, "SGate", PhaseCall("s", True)),
        (cirq.H, "Hadamard", CliffordCall("h", 1)),
        (cirq.ControlledGate(cirq.H), "CHadamard", None),
        (cirq.ControlledGate(cirq.Y), "CYGate", CliffordCall("cy", 2)),
        (cirq.CNOT, "CNOT", CliffordCall("cx", 2)),
        (cirq.CZ, "CZ", CliffordCall("cz", 2)),
        (cirq.TOFFOLI, "Toffoli", FixedTCall("toffoli")),
        (cirq.SWAP, "TwoBitSwap", CliffordCall("swap_bit", 2)),
        (cirq.CSWAP, "CSwap", ControlledSwapCall(1)),
        (cirq.I, "Identity", ZeroCostCall("identity", (1,))),
        (cirq.rx(theta), "Rx", RotationCall("rx", theta)),
        (cirq.ry(-0.2), "Ry", RotationCall("ry", -0.2)),
        (cirq.rz(0.125), "Rz", RotationCall("rz", 0.125)),
        (cirq.XPowGate(exponent=0.3), "XPowGate", RotationCall("xpow", 0.3)),
        (cirq.YPowGate(exponent=theta), "YPowGate", RotationCall("ypow", theta)),
        (cirq.ZPowGate(exponent=0.3), "ZPowGate", RotationCall("zpow", 0.3)),
        (cirq.CZPowGate(exponent=0.3), "CZPowGate", ControlledZPowerCall(0.3)),
        (
            cirq.ControlledGate(cirq.ZPowGate(exponent=0.3)),
            "CZPowGate",
            ControlledZPowerCall(0.3),
        ),
        (
            cirq.ControlledGate(cirq.T),
            "Controlled",
            ControlledRotationCall(PhaseCall("t"), (1,)),
        ),
        (
            cirq.ControlledGate(cirq.rx(0.2)),
            "Controlled",
            ControlledRotationCall(RotationCall("rx", 0.2), (1,)),
        ),
        (
            cirq.global_phase_operation(1j).gate,
            "GlobalPhase",
            ZeroCostCall("global_phase", (1j,)),
        ),
    )
    for gate, source_type, expected_key in cases:
        source = cirq_gate_to_bloq(gate)
        assert type(source).__name__ == source_type
        key = cirq_gate_resource_key(gate)
        assert key is not None
        if expected_key is not None:
            assert key == expected_key
        expected_cost = get_cost_value(source, QECGatesCost())
        assert asdict(query_qec_gate_cost(key)) == {
            field.name: getattr(expected_cost, field.name)
            for field in fields(QECGateCost)
        }


@pytest.mark.ftqc_reference
def test_unsupported_sdk_wrappers_remain_at_the_boundary() -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    for gate in (
        cirq.XPowGate(exponent=0.3, global_shift=0.125),
        cirq.ZZPowGate(exponent=0.25),
        cirq.IdentityGate(qid_shape=(3,)),
    ):
        assert type(cirq_gate_to_bloq(gate)).__name__ == "CirqGateAsBloq"
        assert cirq_gate_resource_key(gate) is None
    assert (
        cirq_gate_resource_key(cirq.ControlledGate(cirq.CNOT, control_values=(0,)))
        is None
    )
    assert cirq_gate_resource_key(cirq.H.on(cirq.LineQubit(0))) is None


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("controls", [(0,), (1, 1), (1, 0), (0, 1, 0)])
def test_predicate_controls_map_to_source_costs(controls: tuple[int, ...]) -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    for base in (
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.H,
        cirq.T,
        cirq.SWAP,
        cirq.S**-1,
        cirq.rx(0.3),
        cirq.rz(-0.7),
        cirq.ZPowGate(exponent=0.3),
    ):
        gate = cirq.ControlledGate(base, control_values=controls)
        key = cirq_gate_resource_key(gate)
        assert key is not None
        source = get_cost_value(cirq_gate_to_bloq(gate), QECGatesCost())
        assert asdict(query_qec_gate_cost(key)) == {
            field.name: getattr(source, field.name) for field in fields(QECGateCost)
        }


def test_positive_rz_uses_dedicated_resource_decomposition() -> None:
    cirq = pytest.importorskip("cirq")
    gate = cirq.ControlledGate(cirq.rz(0.3))
    assert cirq_gate_resource_key(gate) == ControlledZRotationCall(0.3)
    assert query_qec_gate_cost(ControlledZRotationCall(0.3)) == QECGateCost(
        clifford=2, rotation=2
    )
