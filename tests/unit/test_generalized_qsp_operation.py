"""Named GQSP binding consumes the existing numerical and Compiler paths."""

from collections import Counter
from collections.abc import Mapping
from typing import Literal

import cirq
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from flagquantum.algorithms.primitives.generalized_qsp import GeneralizedQSP
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.phase.generalized_qsp import generalized_qsp_ir
from flagquantum.compiler.registers.primitives.pauli_rotation import pauli_power_ir
from flagquantum.core.ir import Instruction
from flagquantum.simulation.numerics.qsp_complement import qsp_complementary_polynomial
from flagquantum.simulation.numerics.qsp_phases import qsp_phase_factors
from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial
from flagquantum.simulation.numerics.unitary_polynomial import (
    evaluate_polynomial_of_unitary_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _unitary(
    signal: int,
    ports: Mapping[str, tuple[int, ...]],
    workspace: tuple[int, ...],
    value: int | None,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    a, b = ports["left"]
    (c,) = ports["right"]
    (work,) = workspace
    stages: tuple[tuple[Literal["x", "z"], int, float, float], ...] = (
        ("x", a, 0.29, 0.17),
        ("z", b, 0.19, 0.0),
        ("z", c, -0.21, 0.0),
    )
    gates: list[Instruction] = []
    if value == 0:
        gates.append(Instruction("x", (signal,)))
    for axis, target, exponent, shift in reversed(stages) if adjoint else stages:
        sign = -1 if adjoint else 1
        control = None if value is None else signal
        if axis == "z":
            # Clean work computes the target bit, receives its phase, then erases.
            copy = Instruction("cx", (target, work))
            gates.extend(
                (
                    copy,
                    *pauli_power_ir("z", work, sign * exponent, control=control),
                    copy,
                )
            )
        else:
            gates.extend(
                pauli_power_ir(
                    axis, target, sign * exponent, global_shift=shift, control=control
                )
            )
    if value == 0:
        gates.append(Instruction("x", (signal,)))
    return tuple(gates)


@pytest.mark.parametrize("length", [2, 4])
@pytest.mark.parametrize("negative_power", [0, 1, 6])
@pytest.mark.parametrize("real", [False, True])
def test_named_scrambled_binding_full_source_matrix_clean_work_and_counts(
    length: int, negative_power: int, real: bool
) -> None:
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    p = tuple(
        random_qsp_polynomial(
            length, random_state=np.random.RandomState(31), only_real_coeffs=real
        )
    )
    q = tuple(complex(c) for c in qsp_complementary_polynomial(p, verify=True))
    u = np.kron(
        np.kron(
            cirq.unitary(cirq.XPowGate(exponent=0.29, global_shift=0.17)),
            cirq.unitary(cirq.ZPowGate(exponent=0.19)),
        ),
        cirq.unitary(cirq.ZPowGate(exponent=-0.21)),
    )
    assert np.isfinite(u).all()
    source = Source(
        MatrixGate(bitsize=3, matrix=u), p, q, negative_power=negative_power
    )
    actual_calls: list[
        tuple[tuple[tuple[str, tuple[int, ...]], ...], int | None, bool]
    ] = []

    def record(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        value: int | None,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert signal == 2 and workspace == (0,)
        actual_calls.append((tuple(ports.items()), value, adjoint))
        return _unitary(signal, ports, workspace, value, adjoint)

    operation = GeneralizedQSP(
        (("left", 2), ("right", 1)), record, p, q, negative_power
    )
    target = (4, 1, 3)
    gates = operation.emit(2, target, (0,))
    calls = actual_calls.copy()
    assert all(ports == (("left", (4, 1)), ("right", (3,))) for ports, _, _ in calls)
    expected_calls = Counter(
        {
            (0, False): max(0, length - 1 - negative_power),
            (1, True): min(length - 1, negative_power),
            (None, True): max(0, negative_power - length + 1),
        }
    )
    assert Counter((value, adjoint) for _, value, adjoint in calls) == +expected_calls
    # Same unchanged source phases, with the existing Compiler emitter replaced
    # directly at the Algorithms boundary: no extra physical gates in binding.
    t, f, initial_lambda = source._qsp_phases
    assert gates == generalized_qsp_ir(
        2,
        (*target, 0),
        tuple(map(float, t)),
        tuple(map(float, f)),
        float(initial_lambda),
        unitary=lambda v, a: _unitary(2, {"left": (4, 1), "right": (3,)}, (0,), v, a),
        negative_power=negative_power,
    )
    reference = cirq.unitary(source)
    assert np.isfinite(reference).all()
    rng = np.random.RandomState(7)
    coherent = rng.normal(size=16) + 1j * rng.normal(size=16)
    coherent /= np.linalg.norm(coherent)
    # Logical signal,left[0],left[1],right[0],work,idle -> physical 0..5.
    order = (2, 4, 1, 3, 0, 5)
    for initial in (*np.eye(16, dtype=complex), coherent):
        logical = np.kron(initial, [1, 0, 0, 0]).reshape((2,) * 6)
        physical = logical.transpose(tuple(np.argsort(order))).flatten()
        circuit = Circuit(6, inputs=torch.tensor(physical), dtype=torch.complex128)
        circuit.append(operation.make_on(signal=2, target=target, workspace=(0,)))
        assert circuit.to_ir().instructions == gates
        state = (
            circuit.state()
            .detach()
            .numpy()
            .reshape((2,) * 6)
            .transpose(order)
            .reshape(16, 4)
        )
        np.testing.assert_allclose(state[:, 0], reference @ initial, atol=5e-14, rtol=0)
        np.testing.assert_array_equal(state[:, 1:], 0)
    for row, coefficients in ((0, p), (8, q)):
        np.testing.assert_allclose(
            reference[row : row + 8, :8],
            evaluate_polynomial_of_unitary_matrix(
                coefficients, u, offset=-negative_power
            ),
            atol=3e-12,
            rtol=0,
        )
    assert not any(g.name in ("measure", "reset") for g in gates)


@pytest.mark.parametrize(
    "p,q", [((), (0.2, 0.3)), ((0.2, 0.3), (0.2,)), ((0.2,), (0.3,))]
)
def test_source_degree_validation(
    p: tuple[complex, ...], q: tuple[complex, ...]
) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    with pytest.raises(ValueError) as expected:
        Source(XGate(), p, q)
    with pytest.raises(ValueError) as actual:
        GeneralizedQSP((("q", 1),), lambda s, p, w, v, a: (), p, q)
    assert str(actual.value) == str(expected.value)


def test_coefficients_snapshot_and_unequal_lengths_source_failure() -> None:
    p = [0.2, 0.3]
    op = GeneralizedQSP((("q", 1),), lambda s, p, w, v, a: (), p, (0.1, 0.2, 0.3))
    p[0] = 0.8
    assert op.p == (0.2, 0.3)
    with pytest.raises(ValueError, match="same degree"):
        op.emit(0, (1,))


@pytest.mark.parametrize(
    "layout,target,workspace,error",
    [
        ((("signal", 1),), (1,), (), "reserves"),
        ((("q", 2),), (1,), (), "sum of partition"),
        ((("q", 1),), (0,), (), "distinct"),
        ((("q", 1),), (1,), (1,), "distinct"),
        ((("q", 1), ("q", 1)), (1, 2), (), "distinct"),
    ],
)
def test_binding_failures_before_emission(
    layout: tuple[tuple[str, int], ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    error: str,
) -> None:
    def forbidden(
        s: int,
        p: Mapping[str, tuple[int, ...]],
        w: tuple[int, ...],
        v: int | None,
        a: bool,
    ) -> tuple[Instruction, ...]:
        raise AssertionError("No callback allowed on invalid binding")

    op = GeneralizedQSP(layout, forbidden, (0.2, 0.3), (0.1, 0.4))
    with pytest.raises(ValueError, match=error):
        op.emit(0, target, workspace)


def test_callback_escape_and_failed_append_leave_circuit_unchanged() -> None:
    op = GeneralizedQSP(
        (("q", 1),),
        lambda s, p, w, v, a: (Instruction("x", (3,)),),
        (0.2, 0.3),
        (0.1, 0.4),
    )
    circuit = Circuit(4, dtype=torch.complex128)
    circuit.h(0)
    before = circuit.to_ir()
    with pytest.raises(ValueError, match="outside"):
        circuit.append(op.make_on(signal=0, target=(1,)))
    assert circuit.to_ir() == before
    with pytest.raises(ValueError, match="exceeds"):
        circuit.append(op.make_on(signal=0, target=(4,)))
    assert circuit.to_ir() == before


def test_ports_are_fresh_on_each_callback_and_phase_result_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.algorithms.primitives import generalized_qsp as module

    invocations = 0

    def phases(
        p: tuple[complex, ...], q: tuple[complex, ...]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        nonlocal invocations
        invocations += 1
        return qsp_phase_factors(p, q)

    def mutate(
        s: int,
        p: Mapping[str, tuple[int, ...]],
        w: tuple[int, ...],
        v: int | None,
        a: bool,
    ) -> tuple[Instruction, ...]:
        assert p == {"q": (1,)}
        assert isinstance(p, dict)
        p.clear()
        return ()

    monkeypatch.setattr(module, "qsp_phase_factors", phases)
    op = GeneralizedQSP((("q", 1),), mutate, (0.2, 0.3, 0.1), (0.1, 0.4, 0.2))
    assert op.emit(0, (1,)) == op.emit(0, (1,))
    assert invocations == 1


def test_existing_batch_append_metadata_is_preserved() -> None:
    op = GeneralizedQSP((("q", 1),), lambda s, p, w, v, a: (), (0.2, 0.3), (0.1, 0.4))
    circuit = Circuit(2, bsz=2, dtype=torch.complex128)
    circuit.append(op.make_on(signal=0, target=(1,)))
    assert circuit.bsz == 2
    state = circuit.state().detach().numpy().reshape(2, 4)
    np.testing.assert_array_equal(state[0], state[1])
