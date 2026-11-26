"""Check the resource-only Cirq boundary against the locked gate mapper."""

import sys
from dataclasses import asdict, fields

import pytest

from flagquantum.compiler._component_trace.compound_control_resource import (
    CONTROLLED_HADAMARD_COST,
    compound_control_key,
)
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
    identity_cost_key,
)
from flagquantum.ecosystem.cirq.resource_keys import cirq_gate_resource_key
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("controls", ((1,), (0,), (0, 1), (0, 1, 0)))
@pytest.mark.parametrize(
    "phase", ("identity", "minus", "s", "t", "general", "symbolic")
)
def test_controlled_scalar_uses_source_phase_model(
    controls: tuple[int, ...],
    phase: str,
    locked_source: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cmath import exp
    from math import pi

    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from flagquantum.compiler._component_trace.compound_control_resource import (
        global_phase_cost_key,
    )
    from flagquantum.core.value_math import pi as scalar_pi
    from flagquantum.core.value_math import sarg
    from flagquantum.ecosystem.cirq.resource_keys import (
        CirqResourceCall,
        _controlled_key,
        cirq_mapped_gate_resource_key,
        generalize_cirq_resource_call,
    )

    coefficient = {
        "identity": 1,
        "minus": -1,
        "s": 1j,
        "t": exp(1j * pi / 4),
        "general": exp(-0.317j),
        "symbolic": sympy.exp(sympy.I * sympy.pi * sympy.Symbol("theta", real=True)),
    }[phase]
    scalar = cirq.GlobalPhaseGate(coefficient)
    gate = cirq.ControlledGate(scalar, control_values=controls)
    source = cirq_gate_to_bloq(gate)
    key = cirq_gate_resource_key(gate)
    assert key == global_phase_cost_key(
        sarg(coefficient) / scalar_pi(coefficient), controls
    )
    assert cirq_mapped_gate_resource_key(gate) == key
    assert generalize_cirq_resource_call(CirqResourceCall(gate)) == key
    assert key is not None
    expected = get_cost_value(source, QECGatesCost())
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        actual = asdict(query_qec_gate_cost(key))
    assert actual == {
        field.name: getattr(expected, field.name) for field in fields(QECGateCost)
    }
    assert cirq_gate_resource_key(scalar) == ZeroCostCall(
        "global_phase", (coefficient,)
    )
    # Native scalar keys use an exponent/precision pair, not this SDK coefficient.
    assert _controlled_key(global_phase_cost_key(0.317), controls) is None


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("controls", ((1,), (0,), (0, 1), (0, 1, 0)))
@pytest.mark.parametrize("depth", (1, 2, 3))
@pytest.mark.parametrize(
    "kind",
    (
        "matrix",
        "diag1",
        "diag2",
        "diag3",
        "iswap",
        "iswap_t",
        "iswap_s",
        "iswap_zero",
        "ccx",
        "ccx_half",
        "ccx_integer",
        "ccx_zero",
        "ccx_symbolic",
    ),
)
def test_controlled_external_declared_cost_closes_source_child_models(
    controls: tuple[int, ...],
    depth: int,
    kind: str,
    locked_source: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran._infra.adjoint import Adjoint
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.controlled import Controlled, CtrlSpec
    from qualtran.cirq_interop._cirq_to_bloq import CirqGateAsBloq
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from flagquantum.ecosystem.cirq._decompose import decompose_cirq_once
    from flagquantum.ecosystem.cirq.resource_keys import (
        _controlled_key,
        cirq_mapped_gate_resource_key,
    )

    if kind.startswith("diag"):
        gate = cirq.DiagonalGate(
            tuple(
                0.13 + 0.07 * i * i - 0.11 * (i % 3) for i in range(1 << int(kind[-1]))
            )
        )
    elif kind.startswith("iswap"):
        gate = cirq.ISwapPowGate(
            exponent={
                "iswap": 0.375,
                "iswap_t": 0.5,
                "iswap_s": 1.0,
                "iswap_zero": 0.0,
            }[kind]
        )
    elif kind == "ccx_symbolic":
        import sympy

        gate = cirq.CCXPowGate(exponent=sympy.Symbol("theta"))
    elif kind.startswith("ccx"):
        gate = cirq.CCXPowGate(
            exponent={
                "ccx": 0.375,
                "ccx_half": 0.5,
                "ccx_integer": 1.0,
                "ccx_zero": 0.0,
            }[kind]
        )
    else:
        assert kind == "matrix"
        gate = cirq.MatrixGate(cirq.testing.random_unitary(2, random_state=17))
    source: Bloq = Controlled(
        CirqGateAsBloq(gate),
        CtrlSpec(cvs=controls[0] if len(controls) == 1 else controls),
    )
    for _ in range(depth):
        source = Adjoint(source)
    children = decompose_cirq_once(gate)
    assert children is not None and children
    keys = []
    for child in children:
        key = _controlled_key(cirq_mapped_gate_resource_key(child.gate), controls)
        assert key is not None
        keys.append(key)
    expected = get_cost_value(source, QECGatesCost())
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        cost = QECGateCost()
        for key in keys:
            cost = cost.plus(query_qec_gate_cost(key), 1)
    assert asdict(cost) == {
        field.name: getattr(expected, field.name) for field in fields(QECGateCost)
    }


@pytest.mark.ftqc_reference
def test_source_recognition_order_and_gate_costs() -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from sympy import Symbol

    theta = Symbol("theta", real=True)
    cases = (
        (cirq.T, "TGate", PhaseCall("t")),
        (cirq.T**-1, "TGate", PhaseCall("t", True)),
        (cirq.S**-1, "SGate", PhaseCall("s", True)),
        (cirq.H, "Hadamard", CliffordCall("h", 1)),
        (cirq.ControlledGate(cirq.H), "CHadamard", CONTROLLED_HADAMARD_COST),
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
        cirq_gate_resource_key(cirq.ControlledGate(cirq.CNOT, control_values=((0, 1),)))
        is None
    )
    assert cirq_gate_resource_key(cirq.H.on(cirq.LineQubit(0))) is None


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("controls", [(0,), (1, 1), (1, 0), (0, 1, 0)])
def test_predicate_controls_map_to_source_costs(controls: tuple[int, ...]) -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    for base in (
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.H,
        cirq.T,
        cirq.SWAP,
        cirq.CNOT,
        cirq.TOFFOLI,
        cirq.CSWAP,
        cirq.S**-1,
        cirq.rx(0.3),
        cirq.rz(-0.7),
        cirq.ZPowGate(exponent=0.3),
    ):
        gate = cirq.ControlledGate(base, control_values=controls)
        key = cirq_gate_resource_key(gate)
        assert key is not None
        if base in (cirq.X, cirq.Y, cirq.Z, cirq.H):
            assert key == compound_control_key(
                {
                    cirq.X: CliffordCall("x", 1),
                    cirq.Y: CliffordCall("y", 1),
                    cirq.Z: CliffordCall("z", 1),
                    cirq.H: CliffordCall("h", 1),
                }[base],
                controls,
            )
        elif base in (cirq.T, cirq.S**-1):
            phase = PhaseCall("t") if base == cirq.T else PhaseCall("s", adjoint=True)
            assert key == compound_control_key(phase, controls)
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


def test_nested_singleton_controls_preserve_original_activation_order() -> None:
    cirq = pytest.importorskip("cirq")
    inner = cirq.ControlledGate(cirq.T, control_values=(0,))
    nested = cirq.ControlledGate(inner, control_values=(1, 0))
    flat = cirq.ControlledGate(cirq.T, control_values=(1, 0, 0))
    assert cirq_gate_resource_key(nested) == cirq_gate_resource_key(flat)


@pytest.mark.ftqc_reference
def test_identity_resource_key_respects_the_source_cirq_boundary() -> None:
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates.identity import Identity
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    controlled = cirq.ControlledGate(cirq.I, control_values=(0, 1))
    source = cirq_gate_to_bloq(controlled)
    assert source == Identity(3)
    assert cirq_gate_resource_key(controlled) == identity_cost_key(3)

    atomic = cirq.IdentityGate(3)
    assert type(cirq_gate_to_bloq(atomic)).__name__ == "CirqGateAsBloq"
    assert cirq_gate_resource_key(atomic) is None


@pytest.mark.ftqc_reference
def test_external_wrapper_static_atoms_precede_recursive_costs() -> None:
    from importlib.metadata import version

    import cirq
    import kickmix
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from flagquantum.ecosystem.cirq.resource_keys import cirq_mapped_gate_resource_key

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__", None) == "0.1" and hasattr(
        kickmix, "Simulator"
    )
    for gate in (
        cirq.MatrixGate(cirq.unitary(cirq.H)),
        cirq.MatrixGate(cirq.testing.random_unitary(2, random_state=17)),
        cirq.MatrixGate(cirq.testing.random_unitary(4, random_state=23)),
        cirq.CXPowGate(exponent=-0.375),
        cirq.SwapPowGate(exponent=0.375),
        cirq.GlobalPhaseGate(1j),
        cirq.ControlledGate(cirq.H),
        cirq.ResetChannel(),
        cirq.ops.raw_types._InverseCompositeGate(
            cirq.MatrixGate(cirq.testing.random_unitary(2, random_state=17))
        ),
        cirq.ops.raw_types._InverseCompositeGate(
            cirq.MatrixGate(cirq.testing.random_unitary(4, random_state=23))
        ),
        cirq.ops.raw_types._InverseCompositeGate(cirq.S),
        cirq.ops.raw_types._InverseCompositeGate(cirq.XPowGate(exponent=0.5)),
    ):
        key = cirq_mapped_gate_resource_key(gate)
        assert key is not None
        expected = get_cost_value(cirq_gate_to_bloq(gate), QECGatesCost())
        actual = query_qec_gate_cost(key)
        assert asdict(actual) == {
            field.name: getattr(expected, field.name) for field in fields(actual)
        }
    assert cirq_mapped_gate_resource_key(cirq.IdentityGate(qid_shape=(3,))) is None
    assert cirq_mapped_gate_resource_key(object()) is None
    unowned_control = cirq.ControlledGate(
        cirq.MatrixGate(cirq.testing.random_unitary(2, random_state=17))
    )
    assert cirq_mapped_gate_resource_key(unowned_control) is None
    from qualtran._infra.bloq import DecomposeNotImplementedError

    unchecked_t_inverse = cirq.ops.raw_types._InverseCompositeGate(cirq.T)
    with pytest.raises(DecomposeNotImplementedError, match="decomposition"):
        get_cost_value(cirq_gate_to_bloq(unchecked_t_inverse), QECGatesCost())
    assert cirq_mapped_gate_resource_key(unchecked_t_inverse) is None
