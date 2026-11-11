"""Source polynomial factories and signal precision feed the existing append path."""

from collections.abc import Mapping

import cirq
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.fft import fft as scipy_fft

import flagquantum as fq
from flagquantum.algorithms.primitives.generalized_qsp import GeneralizedQSP
from flagquantum.compiler._component_trace.resource_generalization import (
    RotationCall,
    ZeroCostCall,
)
from flagquantum.compiler.registers.primitives.pauli_rotation import pauli_power_ir
from flagquantum.core.ir import Instruction
from flagquantum.simulation.numerics.qsp_grid import _NumericArray
from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _fft(data: _NumericArray) -> NDArray[np.complex128]:
    return np.asarray(scipy_fft(data), dtype=np.complex128)


def _unitary(
    signal: int,
    ports: Mapping[str, tuple[int, ...]],
    workspace: tuple[int, ...],
    value: int | None,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    (target,) = ports["q"]
    gates = pauli_power_ir(
        "z",
        target,
        -0.29 if adjoint else 0.29,
        control=None if value is None else signal,
    )
    if value == 0:
        flip = Instruction("x", (signal,))
        return (flip, *gates, flip)
    return gates


@pytest.mark.parametrize("length", [2, 3])
@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("negative_power", [0, 1, 6])
def test_factory_source_coefficients_precision_children_full_amplitudes_and_clean_idle(
    length: int, real: bool, verify: bool, negative_power: int
) -> None:
    from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
    from qualtran.bloqs.basic_gates.rotation import Rx, Rz
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    p = tuple(
        random_qsp_polynomial(
            length, random_state=np.random.RandomState(18), only_real_coeffs=real
        )
    )
    source = Source.from_qsp_polynomial(
        MatrixGate(bitsize=1, matrix=cirq.unitary(cirq.Z**0.29)),
        p,
        negative_power=negative_power,
        precision=2e-8,
        verify=verify,
        verify_precision=2e-7,
    )
    native = GeneralizedQSP.from_qsp_polynomial(
        (("q", 1),),
        _unitary,
        p,
        negative_power=negative_power,
        precision=2e-8,
        verify=verify,
        verify_precision=2e-7,
        fft=_fft,
    )
    np.testing.assert_array_equal(native.p, source.P)
    np.testing.assert_array_equal(native.q, source.Q)
    assert native.precision == source.precision
    for calls, rotation in zip(
        native.signal_rotation_calls, source.signal_rotations, strict=True
    ):
        # Creation IDs retain the source GlobalPhase/Rz/Rx/Rz sequence, even
        # when the independent zero-port phase traverses last topologically.
        original = [
            entry[0].bloq
            for entry in sorted(
                rotation.decompose_bloq().iter_bloqsoqs(), key=lambda entry: entry[0].i
            )
        ]
        assert isinstance(original[0], GlobalPhase)
        assert calls[0] == ZeroCostCall(
            "global_phase", (original[0].exponent, original[0].eps)
        )
        for call, leaf in zip(calls[1:], original[1:], strict=True):
            assert isinstance(leaf, (Rz, Rx)) and isinstance(call, RotationCall)
            assert call.opcode == ("rz" if isinstance(leaf, Rz) else "rx")
            assert call.parameter == leaf.angle
            assert call.precision == leaf.eps == (2e-8 / (length + 1)) / 3
    reference = cirq.unitary(source)
    assert np.isfinite(reference).all()
    rng = np.random.RandomState(8)
    coherent = rng.normal(size=4) + 1j * rng.normal(size=4)
    coherent /= np.linalg.norm(coherent)
    for initial in (*np.eye(4, dtype=complex), coherent):
        circuit = fq.Circuit(
            3, inputs=torch.tensor(np.kron(initial, [1, 0])), dtype=torch.complex128
        )
        circuit.append(native.make_on(signal=0, target=(1,)))
        state = circuit.state().detach().numpy().reshape(4, 2)
        assert np.isfinite(state).all()
        np.testing.assert_allclose(state[:, 0], reference @ initial, atol=3e-14, rtol=0)
        np.testing.assert_array_equal(state[:, 1], 0)
    # Precision is a declared synthesis budget; continuous IR is unchanged.
    other = GeneralizedQSP(
        native.unitary_layout,
        _unitary,
        native.p,
        native.q,
        negative_power,
        precision=1e-4,
    )
    assert native.emit(0, (1,)) == other.emit(0, (1,))


@pytest.mark.parametrize("p", [(), (0.5,), (0.5, 0.0), (2.0, 0.2)])
@pytest.mark.parametrize("verify", [False, True])
def test_source_factory_failures_are_preserved(
    p: tuple[complex, ...], verify: bool
) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    try:
        expected = Source.from_qsp_polynomial(XGate(), p, verify=verify)
    except (ValueError, AssertionError) as error:
        with pytest.raises(type(error)) as actual:
            GeneralizedQSP.from_qsp_polynomial(
                (("q", 1),), _unitary, p, verify=verify, fft=_fft
            )
        assert str(actual.value) == str(error)
    else:
        native = GeneralizedQSP.from_qsp_polynomial(
            (("q", 1),), _unitary, p, verify=verify, fft=_fft
        )
        np.testing.assert_array_equal(native.q, expected.Q)


def test_factory_defaults_match_source_and_do_not_consume_fft_without_verify() -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    def forbidden(data: _NumericArray) -> NDArray[np.complex128]:
        raise AssertionError("FFT must not be called")

    p = (0.25, 0.1, 0.2)
    source = Source.from_qsp_polynomial(XGate(), p)
    native = GeneralizedQSP.from_qsp_polynomial((("q", 1),), _unitary, p, fft=forbidden)
    assert source.precision == native.precision == 0
    explicit = GeneralizedQSP((("q", 1),), _unitary, native.p, native.q)
    assert explicit.precision == Source(XGate(), source.P, source.Q).precision == 1e-11
    assert all(
        call.precision == 0
        for calls in native.signal_rotation_calls
        for call in calls[1:]
    )


def test_verified_factory_requires_provider_and_preserves_backend_failure() -> None:
    with pytest.raises(ValueError, match="FFT provider"):
        GeneralizedQSP.from_qsp_polynomial(
            (("q", 1),), _unitary, (0.25, 0.1), verify=True
        )

    def broken(data: _NumericArray) -> NDArray[np.complex128]:
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        GeneralizedQSP.from_qsp_polynomial(
            (("q", 1),), _unitary, (0.25, 0.1), verify=True, fft=broken
        )


def test_rejected_grid_stops_before_complement(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum.algorithms.primitives import generalized_qsp as module

    def forbidden(
        p: tuple[complex, ...], *, verify: bool, verify_precision: float
    ) -> NDArray[np.float64]:
        raise AssertionError("Complement must not be called")

    monkeypatch.setattr(module, "qsp_complementary_polynomial", forbidden)
    with pytest.raises(AssertionError, match="Not a QSP polynomial"):
        GeneralizedQSP.from_qsp_polynomial(
            (("q", 1),), _unitary, (2.0, 0.2), verify=True, fft=_fft
        )
