"""Source SU(2) call order, amplitudes, and adjoint under FlagQuantum lowering."""

from cmath import exp
from collections import Counter
from math import pi

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.phase.su2_rotation import (
    su2_angles_from_matrix,
    su2_angles_from_zxz,
    su2_rotation_calls,
    su2_rotation_ir,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize(
    "angles",
    [
        (0, 0, 0, 0),
        (pi / 4, pi / 2, pi / 2, 0),
        (0.4, 1.7, -0.5, 0.2),
        (pi / 2, 0.3, -0.9, -0.7),
    ],
)
def test_su2_decomposition_and_exact_adjoint(
    target: int, angles: tuple[float, float, float, float]
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import GlobalPhase, Rx, Rz, SU2RotationGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    theta, phi, lambd, shift = angles
    source = SU2RotationGate(theta, phi, lambd, shift)
    forward = su2_rotation_ir(target, theta, phi, lambd, global_shift=shift)
    reverse = su2_rotation_ir(
        target, theta, phi, lambd, global_shift=shift, adjoint=True
    )
    for bloq, actual in ((source, forward), (source.adjoint(), reverse)):
        children = [inst.bloq for inst, _, _ in bloq.decompose_bloq().iter_bloqsoqs()]
        scalar = next(child for child in children if isinstance(child, GlobalPhase))
        rotations = [child for child in children if isinstance(child, (Rz, Rx))]
        assert [type(child) for child in rotations] == [Rz, Rx, Rz]
        assert [g.name for g in actual[-3:]] == ["rz", "rx", "rz"]
        assert [g.params["theta"] for g in actual[-3:]] == [
            child.angle for child in rotations
        ]
        assert all(g.wires == (target,) for g in actual)
        expected_scalar = float(scalar.exponent) * pi
        assert [g.name for g in actual[:-3]] == (
            ["x", "phase", "x", "phase"] if expected_scalar else []
        )
        assert [g.params["theta"] for g in actual[:-3] if g.name == "phase"] == (
            [expected_scalar, expected_scalar] if expected_scalar else []
        )
        assert get_cost_value(bloq, QubitCount()) == 1
    calls = su2_rotation_calls(theta, phi, lambd, global_shift=shift)
    gates = Counter()
    for call in calls:
        cost = query_qec_gate_cost(call)
        gates.update(t=cost.t, clifford=cost.clifford, rotation=cost.rotation)
    assert +gates == get_cost_value(source, QECGatesCost()).asdict()

    state = torch.tensor([1j, 2 - 1j, 3j, -2], dtype=torch.complex128)
    state /= state.norm()
    circuit = fq.Circuit(2, inputs=state, dtype=torch.complex128)
    matrix = torch.as_tensor(
        source.tensor_contract(), dtype=torch.complex128
    ).contiguous()
    identity = torch.eye(2, dtype=torch.complex128)
    expected = (
        torch.kron(matrix, identity) if target == 0 else torch.kron(identity, matrix)
    ) @ state
    circuit.append(FieldOperation(2, forward))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    circuit.append(FieldOperation(2, reverse))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)


@pytest.mark.ftqc_reference
def test_symbolic_calls_keep_exact_angles_and_precision() -> None:
    pytest.importorskip("qualtran")
    import sympy
    from qualtran.bloqs.basic_gates import GlobalPhase, SU2RotationGate

    theta, phi, lambd, shift, precision = sympy.symbols("theta phi lambda alpha eps")
    source = SU2RotationGate(theta, phi, lambd, shift, precision)
    scalar, *rotations = su2_rotation_calls(
        theta, phi, lambd, global_shift=shift, precision=precision
    )
    upstream, _ = source.call_graph()
    root = next(node for node in upstream if node == source)
    children = tuple(upstream.successors(root))
    expected_phase = next(child for child in children if isinstance(child, GlobalPhase))
    assert scalar.parameters == (expected_phase.exponent, expected_phase.eps)
    assert {(call.opcode, call.parameter, call.precision) for call in rotations} == {
        (type(child).__name__.lower(), child.angle, child.eps)
        for child in children
        if child != expected_phase
    }


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "theta,near_singular",
    [
        (0.0, False),
        (0.5e-8, True),
        (2e-8, False),
        (0.39, False),
        (pi / 2 - 2e-8, False),
        (pi / 2 - 0.5e-8, True),
        (pi / 2, False),
    ],
)
def test_matrix_recovery_preserves_pinned_singular_branches(
    theta: float, near_singular: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import SU2RotationGate

    original = SU2RotationGate(theta, 0.73, -0.82, 0.39).rotation_matrix
    source = SU2RotationGate.from_matrix(original)
    recovered = su2_angles_from_matrix(original.tolist())
    assert recovered == pytest.approx(
        (source.theta, source.phi, source.lambd, source.global_shift),
        rel=0,
        abs=2e-15,
    )
    matrix = torch.as_tensor(source.rotation_matrix, dtype=torch.complex128)
    residual = float((matrix - torch.as_tensor(original)).abs().max())
    if near_singular:
        # The pinned branch drops one phase below the tolerance. Preserve its
        # recovered unitary and keep the input residual separate.
        assert 1e-10 < residual < 2e-8
    else:
        assert residual < 2e-12
    theta_out, phi_out, lambda_out, shift = recovered
    gates = su2_rotation_ir(0, theta_out, phi_out, lambda_out, global_shift=shift)
    for column in range(2):
        initial = torch.eye(2, dtype=torch.complex128)[:, column].contiguous()
        circuit = fq.Circuit(1, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(1, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), matrix[:, column], rtol=0, atol=2e-12
        )
    assert [g.name for g in gates[-3:]] == ["rz", "rx", "rz"]
    assert all(g.wires == (0,) for g in gates)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "z_1,x,z_2,shift",
    [(0.0, 0.0, 0.0, 0.0), (0.38, -0.92, 1.61, -0.21), (pi, pi, -pi / 2, pi)],
)
def test_zxz_preserves_order_and_absolute_phase(
    z_1: float, x: float, z_2: float, shift: float
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import SU2RotationGate

    source = SU2RotationGate.from_euler_zxz_angles(
        z_1, x, z_2, global_shift=shift, eps=None
    )
    angles = su2_angles_from_zxz(z_1, x, z_2, global_shift=shift)
    assert angles == (source.theta, source.phi, source.lambd, source.global_shift)
    theta_out, phi_out, lambda_out, alpha = map(float, angles)
    state = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128)
    state /= state.norm()
    circuit = fq.Circuit(1, inputs=state, dtype=torch.complex128)
    circuit.append(
        FieldOperation(
            1, su2_rotation_ir(0, theta_out, phi_out, lambda_out, global_shift=alpha)
        )
    )
    reference = fq.Circuit(1, inputs=state, dtype=torch.complex128)
    reference.rz(0, theta=z_1).rx(0, theta=x).rz(0, theta=z_2)
    expected = reference.state().flatten() * exp(1j * shift)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    torch.testing.assert_close(
        expected, torch.as_tensor(source.rotation_matrix) @ state, rtol=0, atol=2e-12
    )


@pytest.mark.ftqc_reference
def test_mixed_symbolic_zxz_retains_source_precision_parameters() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import SU2RotationGate
    from sympy import symbols

    z, shift, eps = symbols("z shift eps")
    for z_1, z_2 in ((z, 0.3), (0.3, z)):
        source = SU2RotationGate.from_euler_zxz_angles(
            z_1, 0.21, z_2, global_shift=shift, eps=eps
        )
        angles = su2_angles_from_zxz(z_1, 0.21, z_2, global_shift=shift)
        assert angles == (source.theta, source.phi, source.lambd, source.global_shift)
        theta_out, phi_out, lambda_out, alpha = angles
        _, *rotations = su2_rotation_calls(
            theta_out, phi_out, lambda_out, global_shift=alpha, precision=eps
        )
        assert all(call.precision == source.eps / 3 for call in rotations)


def test_matrix_recovery_validates_shape_and_finite_entries() -> None:
    with pytest.raises(ValueError, match="2x2"):
        su2_angles_from_matrix(((1, 0),))
    with pytest.raises(ValueError, match="finite"):
        su2_angles_from_matrix(((1, 0), (float("nan"), 1)))
