"""Retain the locked gradient precision, port identities and model bounds."""

import pytest
from sympy import Symbol, simplify

from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledZPowerCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.phase.phase_addition import (
    source_phase_add_formula,
    source_phase_add_signature,
)
from flagquantum.compiler.registers.phase.phase_gradient import (
    source_phase_gradient_state_signature,
    source_phase_gradient_unitary_calls,
    source_phase_gradient_unitary_signature,
)
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    source_symbolic_scaled_phase_add_formula,
)
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_N, _P = Symbol("n"), Symbol("p")


def _ports(signature: Signature) -> list[tuple[str, SymbolicInt, int]]:
    return [(port.name, port.bitsize, port.side.value) for port in signature]


@pytest.mark.parametrize(
    ("width", "exponent", "controlled", "epsilon"),
    [
        (0, 1.0, False, 1e-10),
        (4, 0.0, True, 0.03),
        (4, -0.5, False, 0.02),
        (9, 0.25, True, 0.05),
        (_N, 0.25, True, 0.02),
        (_N, -0.5, False, 0.02),
    ],
)
def test_unitary_preserves_source_phase_keys(
    width: SymbolicInt, exponent: float, controlled: bool, epsilon: float
) -> None:
    source = pytest.importorskip("qualtran.bloqs.rotations.phase_gradient")
    resource = pytest.importorskip("qualtran.resource_counting")
    op = source.PhaseGradientUnitary(
        width, exponent, is_controlled=controlled, eps=epsilon
    )
    native = source_phase_gradient_unitary_signature(width, controlled=controlled)
    assert _ports(native) == [
        (port.name, port.bitsize, port.side.value) for port in op.signature
    ]
    calls = source_phase_gradient_unitary_calls(
        width, exponent, controlled=controlled, epsilon=epsilon
    )
    expected: dict[RotationCall | ControlledZPowerCall, SymbolicInt] = {}
    for gate, count in op.build_call_graph(resource.SympySymbolAllocator()).items():
        assert type(gate).__name__ == ("CZPowGate" if controlled else "ZPowGate")
        key = (
            ControlledZPowerCall(gate.exponent, gate.eps)
            if controlled
            else RotationCall("zpow", gate.exponent, gate.eps)
        )
        expected[key] = count
    assert calls == expected


@pytest.mark.parametrize("width", [0, 4, _N])
def test_preparation_keeps_the_right_only_port(width: SymbolicInt) -> None:
    source = pytest.importorskip("qualtran.bloqs.rotations.phase_gradient")
    op = source.PhaseGradientState(width)
    assert _ports(source_phase_gradient_state_signature(width)) == [
        (port.name, port.bitsize, port.side.value) for port in op.signature
    ]


@pytest.mark.parametrize(
    ("input_width", "gradient_width", "control"),
    [
        (_N, _P, None),
        (_N, _P, 0),
        (_N, _P, 1),
        (0, 1, 1),
        (4, 1, None),
    ],
)
def test_abstract_phase_addition_counts(
    input_width: SymbolicInt, gradient_width: SymbolicInt, control: int | None
) -> None:
    source = pytest.importorskip("qualtran.bloqs.rotations.phase_gradient")
    resource = pytest.importorskip("qualtran.resource_counting")
    op = source.AddIntoPhaseGrad(input_width, gradient_width, controlled_by=control)
    signature = source_phase_add_signature(
        input_width, gradient_width, controlled=control is not None
    )
    assert _ports(signature) == [
        (port.name, port.bitsize, port.side.value) for port in op.signature
    ]
    model = source_phase_add_formula(
        input_width, gradient_width, controlled=control is not None
    )
    assert (
        simplify(
            model.toffoli - resource.get_cost_value(op, resource.QECGatesCost()).toffoli
        )
        == 0
    )
    if not isinstance(input_width, int) or not isinstance(gradient_width, int):
        assert (
            simplify(
                model.logical_qubits
                - resource.get_cost_value(op, resource.QubitCount())
            )
            == 0
        )


def test_scaled_symbolic_bound_and_mixed_fixed_point_port() -> None:
    qualtran = pytest.importorskip("qualtran")
    source = pytest.importorskip("qualtran.bloqs.rotations.phase_gradient")
    resource = pytest.importorskip("qualtran.resource_counting")
    op = source.AddScaledValIntoPhaseReg(
        qualtran.QFxp(5, 3), _P, Symbol("gamma"), qualtran.QFxp(4, 2)
    )
    signature = source_phase_add_signature(5, _P, input_fraction_bits=3)
    assert _ports(signature) == [
        (port.name, port.bitsize, port.side.value) for port in op.signature
    ]
    assert [dict(port.dtype.parameters)["fractional_width"] for port in signature] == [
        3,
        _P,
    ]
    ((call, multiplicity),) = op.build_call_graph(
        resource.SympySymbolAllocator()
    ).items()
    assert (call.x_bitsize, call.phase_bitsize, multiplicity) == (5, _P, 3)
    model = source_symbolic_scaled_phase_add_formula(5, _P, 4)
    assert (
        simplify(
            model.toffoli - resource.get_cost_value(op, resource.QECGatesCost()).toffoli
        )
        == 0
    )
    assert (
        simplify(
            model.logical_qubits - resource.get_cost_value(op, resource.QubitCount())
        )
        == 0
    )
