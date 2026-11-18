"""Complete flip-convolution optimization and its ordinary GQSP append consumer."""

from collections import Counter
from collections.abc import Callable, Sequence
from hashlib import sha1
from pathlib import Path
from types import SimpleNamespace

import cirq
import numpy as np
import pytest
from numpy.typing import NDArray

from flagquantum.algorithms.primitives.generalized_qsp import GeneralizedQSP
from flagquantum.circuit import Circuit
from flagquantum.simulation.numerics.qsp_optimization import (
    ComplementaryPolynomialObjective,
    fast_complementary_polynomial,
)
from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial
from tests.unit.test_generalized_qsp_factory import _unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("length", [1, 2, 4, 8])
def test_complete_objective_and_rng_optimizer_trace(real: bool, length: int) -> None:
    from qualtran.bloqs.qsp.fast_qsp import (
        FastComplementaryQSPHelper as SourceObjective,
    )
    from qualtran.bloqs.qsp.fast_qsp import fast_complementary_polynomial as source

    p = random_qsp_polynomial(
        length, random_state=np.random.RandomState(42), only_real_coeffs=real
    )
    dtype = np.float64 if real else np.complex128
    poly = np.array(p, dtype=dtype)
    actual = ComplementaryPolynomialObjective(poly, only_reals=real)
    expected = SourceObjective(poly, only_reals=real)
    np.testing.assert_array_equal(actual.conv_p_negative, expected.conv_p_negative)
    x = np.random.RandomState(15).normal(size=length if real else length * 2)
    assert actual.loss_function(x) == expected.loss_function(x)
    np.testing.assert_array_equal(
        actual.array_to_complex(x), expected.array_to_complex(x)
    )
    np.testing.assert_array_equal(
        actual.conv_by_flip_conj(poly), expected.conv_by_flip_conj(poly)
    )
    np.testing.assert_array_equal(
        actual.complex_conv_by_flip_conj(poly.real, poly.imag),
        expected.complex_conv_by_flip_conj(poly.real, poly.imag),
    )
    left_rng, right_rng = np.random.RandomState(91), np.random.RandomState(91)
    q = fast_complementary_polynomial(tuple(p), left_rng, real)
    reference = source(tuple(p), right_rng, real)
    np.testing.assert_array_equal(q, reference)
    np.testing.assert_array_equal(left_rng.randn(20), right_rng.randn(20))
    np.testing.assert_array_equal(poly, p)
    z = np.exp(1j * np.linspace(0, 2 * np.pi, 129))
    residual = np.max(
        abs(
            abs(np.polynomial.polynomial.polyval(z, p)) ** 2
            + abs(np.polynomial.polynomial.polyval(z, q)) ** 2
            - 1
        )
    )
    if real and length in (4, 8):
        # Unchanged source optimizer exits at the same inaccurate real solution.
        assert residual > 1e-3
    else:
        assert residual < 1e-5


@pytest.mark.parametrize("real", [False, True])
def test_optimizer_protocol_and_unsuccessful_iterate(
    real: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scipy.optimize
    from qualtran.bloqs.qsp.fast_qsp import fast_complementary_polynomial as source

    calls: list[tuple[NDArray[np.float64], str, float, float]] = []

    def minimize(
        fun: Callable[[NDArray[np.float64]], float],
        x0: NDArray[np.float64],
        *,
        jac: str,
        tol: float,
    ) -> SimpleNamespace:
        calls.append((x0.copy(), jac, tol, fun(x0)))
        return SimpleNamespace(x=x0 + 0.1, success=False, message="injected stop")

    monkeypatch.setattr(scipy.optimize, "minimize", minimize)
    # The original imported its optimizer at module scope.
    monkeypatch.setattr("qualtran.bloqs.qsp.fast_qsp.minimize", minimize)
    p = (0.2, 0.3)
    actual = fast_complementary_polynomial(p, np.random.RandomState(19), real, 2e-8)
    expected = source(p, np.random.RandomState(19), real, 2e-8)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    assert calls[0][1:] == calls[1][1:] == ("3-point", 2e-8, calls[0][3])


@pytest.mark.parametrize("real", [False, True])
def test_source_empty_and_dtype_failures(real: bool) -> None:
    from qualtran.bloqs.qsp.fast_qsp import (
        FastComplementaryQSPHelper as SourceObjective,
    )
    from qualtran.bloqs.qsp.fast_qsp import fast_complementary_polynomial as source

    for function in (source, fast_complementary_polynomial):
        with pytest.raises(ValueError, match="empty"):
            function((), np.random.RandomState(9), only_reals=real)
    dtype = np.complex128 if real else np.float64
    for helper in (SourceObjective, ComplementaryPolynomialObjective):
        with pytest.raises(AssertionError):
            helper(np.array([0.2, 0.3], dtype=dtype), only_reals=real)


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("negative_power", [0, 1, 5])
def test_optimized_pair_complete_append_operator_cleanup_and_resources(
    real: bool, negative_power: int
) -> None:
    p = tuple(
        random_qsp_polynomial(
            3, random_state=np.random.RandomState(62), only_real_coeffs=real
        )
    )
    q = tuple(
        fast_complementary_polynomial(p, np.random.RandomState(93), only_reals=real)
    )
    assert_gqsp_append_pair(p, q, negative_power)


def assert_gqsp_append_pair(
    p: Sequence[complex], q: Sequence[complex], negative_power: int
) -> None:
    """Verify the shared append consumer for each numerical complement solver."""
    import torch
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    native = GeneralizedQSP((("q", 1),), _unitary, p, q, negative_power)
    source = Source(
        MatrixGate(bitsize=1, matrix=cirq.unitary(cirq.Z**0.29)),
        p,
        q,
        negative_power=negative_power,
    )
    expected = cirq.unitary(source)
    rng = np.random.RandomState(12)
    coherent = rng.randn(4) + 1j * rng.randn(4)
    coherent /= np.linalg.norm(coherent)
    for initial in (*np.eye(4, dtype=complex), coherent):
        circuit = Circuit(
            3,
            inputs=torch.tensor(np.kron(initial, [1, 0])),
            dtype=torch.complex128,
        )
        circuit.append(native.make_on(signal=0, target=(1,)))
        state = circuit.state().detach().numpy().reshape(4, 2)
        np.testing.assert_allclose(state[:, 0], expected @ initial, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(state[:, 1], 0)
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    assert not (counts.keys() & {"measure", "reset"})
    theta, phi, phase = source._qsp_phases
    native_theta, native_phi, native_phase = native._phases
    np.testing.assert_array_equal(native_theta, theta)
    np.testing.assert_array_equal(native_phi, phi)
    assert native_phase == phase
    assert len(native.signal_rotation_calls) == len(p)
    expected_counts = {
        0: {"x": 10, "phase": 6, "rz": 6, "rx": 3, "cphase": 2},
        1: {"x": 8, "phase": 6, "rz": 6, "rx": 3, "cphase": 2},
        5: {"x": 6, "phase": 9, "rz": 6, "rx": 3, "cphase": 2},
    }
    assert counts == expected_counts[negative_power]


def test_frozen_complete_source_and_test_blobs() -> None:
    from qualtran.bloqs.qsp import fast_qsp

    root = Path(fast_qsp.__file__).parent
    for name, expected in (
        ("fast_qsp.py", "eada77ac614c12dc6c40def0e5a034678d05659d"),
        ("fast_qsp_test.py", "e03d0a478f84ae0c3bcb6783b4498b3229592f70"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected
