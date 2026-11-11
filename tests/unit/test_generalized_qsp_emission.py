"""Finite source GQSP stages execute through the native append protocol."""

from collections import Counter
from typing import Literal

import cirq
import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.phase.generalized_qsp import generalized_qsp_ir
from flagquantum.compiler.registers.phase.su2_rotation import su2_rotation_ir
from flagquantum.compiler.registers.primitives.pauli_rotation import pauli_power_ir
from flagquantum.core.ir import Instruction
from flagquantum.simulation.numerics.qsp_complement import qsp_complementary_polynomial
from flagquantum.simulation.numerics.qsp_phases import qsp_phase_factors
from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial
from flagquantum.simulation.numerics.unitary_polynomial import (
    evaluate_polynomial_of_unitary_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("length", [2, 3, 5])
@pytest.mark.parametrize("negative_power", [0, 1, 6])
@pytest.mark.parametrize("axis", ["x", "z"])
def test_source_stages_physical_stream_complex_amplitudes_and_costs(
    length: int, negative_power: int, axis: Literal["x", "z"]
) -> None:
    from qualtran import Controlled
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP
    from qualtran.resource_counting import SympySymbolAllocator

    exponent, shift = 0.29, 0.17 if axis == "x" else 0.0
    original_u = (
        cirq.XPowGate(exponent=exponent, global_shift=shift)
        if axis == "x"
        else cirq.ZPowGate(exponent=exponent)
    )
    u = cirq.unitary(original_u)
    p = tuple(random_qsp_polynomial(length, random_state=np.random.RandomState(12)))
    q = tuple(complex(c) for c in qsp_complementary_polynomial(p, verify=True))
    theta, phi, lambd = qsp_phase_factors(p, q)
    source = GeneralizedQSP(
        MatrixGate(bitsize=1, matrix=u), p, q, negative_power=negative_power
    )
    calls: list[tuple[int | None, bool]] = []

    def emit(value: int | None, adjoint: bool) -> tuple[Instruction, ...]:
        calls.append((value, adjoint))
        sign = -1 if adjoint else 1
        core = pauli_power_ir(
            axis,
            1,
            sign * exponent,
            global_shift=shift,
            control=None if value is None else 0,
        )
        if value == 0:
            flip = Instruction("x", (0,))
            return (flip, *core, flip)
        return core

    gates = generalized_qsp_ir(
        0,
        (1, 2),
        tuple(map(float, theta)),
        tuple(map(float, phi)),
        lambd,
        unitary=emit,
        negative_power=negative_power,
    )
    native_calls = calls.copy()
    source_calls: list[tuple[int | None, bool]] = []
    expected: list[Instruction] = []
    # Derive stage order and angles from the unchanged upstream decomposition.
    for op in cirq.flatten_op_tree(
        cirq.decompose_once(source.on(*cirq.LineQubit.range(2)))
    ):
        assert isinstance(op, cirq.Operation)
        gate = op.gate
        if isinstance(gate, SU2RotationGate):
            expected.extend(
                su2_rotation_ir(
                    0, float(gate.theta), float(gate.phi), float(gate.lambd)
                )
            )
            assert float(gate.eps) == source.precision / (length + 1)
        else:
            if isinstance(gate, (Controlled, ControlledViaAnd)):
                value = int(gate.ctrl_spec.cvs[0])
                actual_u = cirq.unitary(gate.subbloq)
            else:
                value = None
                actual_u = cirq.unitary(gate)
            adjoint = np.allclose(actual_u, u.conj().T, atol=1e-15, rtol=0)
            np.testing.assert_allclose(
                actual_u, u.conj().T if adjoint else u, atol=1e-15
            )
            source_calls.append((value, bool(adjoint)))
            expected.extend(emit(value, bool(adjoint)))
    assert native_calls == source_calls
    assert gates == tuple(expected)
    degree = length - 1
    model = Counter(
        {
            (0, False): max(0, degree - negative_power),
            (1, True): min(degree, negative_power),
            (None, True): max(0, negative_power - degree),
        }
    )
    assert Counter(native_calls) == +model
    assert sum(
        source.build_call_graph(SympySymbolAllocator()).values()
    ) == length + sum(model.values())
    # Physical gate totals are counted independently of source component calls.
    assert sum(Counter(g.name for g in gates).values()) == len(expected)
    assert not any(g.name in ("measure", "reset") for g in gates)
    reference = cirq.unitary(source)
    rng = np.random.RandomState(9)
    coherent = rng.normal(size=4) + 1j * rng.normal(size=4)
    coherent /= np.linalg.norm(coherent)
    for initial in (*np.eye(4, dtype=complex), coherent):
        inputs = np.kron(initial, [1, 0])
        circuit = Circuit(3, inputs=torch.tensor(inputs), dtype=torch.complex128)
        circuit.append(FieldOperation(3, gates))
        assert circuit.to_ir().instructions == gates
        state = circuit.state().detach().numpy().reshape(4, 2)
        np.testing.assert_allclose(state[:, 0], reference @ initial, atol=3e-14, rtol=0)
        np.testing.assert_array_equal(state[:, 1], 0)
    for row, coefficients in ((0, p), (2, q)):
        np.testing.assert_allclose(
            reference[row : row + 2, :2],
            evaluate_polynomial_of_unitary_matrix(
                coefficients, u, offset=-negative_power
            ),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize("negative_power", [-1, True, 1.5])
def test_invalid_negative_power_rejected_before_callback(negative_power: int) -> None:
    def forbidden(value: int | None, adjoint: bool) -> tuple[Instruction, ...]:
        raise AssertionError("Callback must not execute")

    with pytest.raises(ValueError, match="nonnegative integer"):
        generalized_qsp_ir(
            0,
            (1,),
            (0.1, 0.2),
            (0.3, 0.4),
            0.0,
            unitary=forbidden,
            negative_power=negative_power,
        )


@pytest.mark.parametrize(
    "theta,phi", [((), ()), ((0.1,), (0.2,)), ((0.1, 0.2), (0.3,))]
)
def test_invalid_phase_shapes(theta: tuple[float, ...], phi: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="equal phase lengths"):
        generalized_qsp_ir(0, (1,), theta, phi, 0.0, unitary=lambda v, a: ())


def test_wire_binding_validation() -> None:
    with pytest.raises(ValueError, match="distinct"):
        generalized_qsp_ir(
            0, (0,), (0.1, 0.2), (0.3, 0.4), 0.0, unitary=lambda v, a: ()
        )
    with pytest.raises(ValueError, match="outside its binding"):
        generalized_qsp_ir(
            0,
            (1,),
            (0.1, 0.2),
            (0.3, 0.4),
            0.0,
            unitary=lambda v, a: (Instruction("x", (2,)),),
        )


def test_nonfinite_phases_rejected_before_callback() -> None:
    with pytest.raises(ValueError, match="finite"):
        generalized_qsp_ir(
            0, (1,), (0.1, float("nan")), (0.3, 0.4), 0.0, unitary=lambda v, a: ()
        )


def test_replacing_unitary_emitter_preserves_consumer() -> None:
    def empty(value: int | None, adjoint: bool) -> tuple[Instruction, ...]:
        return ()

    def cancelled(value: int | None, adjoint: bool) -> tuple[Instruction, ...]:
        return (Instruction("x", (1,)), Instruction("x", (1,)))

    states = []
    for emit in (empty, cancelled):
        gates = generalized_qsp_ir(
            0, (1,), (0.1, 0.2), (0.3, 0.4), 0.5, unitary=emit, negative_power=3
        )
        circuit = Circuit(2, dtype=torch.complex128)
        circuit.append(FieldOperation(2, gates))
        states.append(circuit.state().detach().numpy())
    np.testing.assert_array_equal(states[0], states[1])
