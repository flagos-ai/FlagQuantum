"""Source SU(2) call order, amplitudes, and adjoint under FlagQuantum lowering."""

from collections import Counter
from math import pi

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.su2_rotation import (
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
