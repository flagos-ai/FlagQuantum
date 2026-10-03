"""The vectorized Lindblad generator and the matrix-free exponential step.

The generator is held as its pieces and applied through the density form, so the
only way to know that the pieces really are the vectorized Liouvillian is to
compare them against the matrix they claim to represent. These tests do that:
the dense form must reproduce the density-form application entry for entry, must
annihilate the vectorized identity, must keep hermitian states hermitian, and
must have a dissipative spectrum. Each of those is checked against a deliberate
wrong rule, so the assertion is evidence rather than a restatement.

The exponential step is then held to a dense ``torch.linalg.matrix_exp`` of that
same matrix, and the defect it reports is shown to bound the error it actually
made -- which is what makes the defect usable as a convergence criterion.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation import amplitude_damping, evolve_density_matrix
from flagquantum.simulation.lindblad_generator import (
    DEFAULT_DENSE_GENERATOR_BYTES,
    Liouvillian,
)
from flagquantum.simulation.lindblad_integrators import advance_interval
from flagquantum.simulation.matrix_free_exponential import exponential_action

pytestmark = pytest.mark.unit

LOWERING = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
PAULI_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)

_WIRES = 2
_DIMENSION = 2**_WIRES


def _embed(single: torch.Tensor, wire: int, n_wires: int) -> torch.Tensor:
    """Place a single-wire operator on one wire of an ``n_wires`` register."""

    factors = [torch.eye(2, dtype=torch.complex128)] * n_wires
    factors[wire] = single
    embedded = factors[0]
    for factor in factors[1:]:
        embedded = torch.kron(embedded, factor)
    return embedded


def _hamiltonian() -> torch.Tensor:
    """Return a hermitian Hamiltonian whose transpose, conjugate and self differ.

    A complex off-diagonal pair is what makes ``H^T`` distinguishable from
    ``conj(H)``, so the vectorization identity is tested on a case where the
    wrong right factor is actually wrong.
    """

    real = torch.tensor(
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.0, -0.2, 0.3, 0.0],
            [0.0, 0.3, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ],
        dtype=torch.complex128,
    )
    real = (real + real.T) / 2
    imaginary = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.complex128,
    )
    return real + 0.35j * imaginary


def _collapse(n_wires: int) -> list[torch.Tensor]:
    """Return three non-unitary jump operators on an ``n_wires`` register."""

    return [
        0.7 * _embed(LOWERING, 0, n_wires),
        0.5 * _embed(LOWERING, 1, n_wires),
        0.3 * _embed(PAULI_X, 0, n_wires),
    ]


def _generator_and_state() -> tuple[Liouvillian, torch.Tensor, torch.Tensor]:
    """Return a generator, a normalized hermitian state, and its dense matrix."""

    generator = Liouvillian(_hamiltonian(), _collapse(_WIRES), hilbert_dimension=4)
    generator_of = torch.Generator().manual_seed(5)
    raw = torch.randn(
        (_DIMENSION, _DIMENSION), generator=generator_of, dtype=torch.complex128
    )
    state = raw @ raw.conj().T
    state = state / torch.trace(state)
    return generator, state, generator.dense()


def _transpose_involution(dimension: int) -> torch.Tensor:
    """Return the permutation that vectorizes the transpose of a matrix."""

    involution = torch.zeros(
        (dimension * dimension, dimension * dimension), dtype=torch.complex128
    )
    for row in range(dimension):
        for column in range(dimension):
            involution[row * dimension + column, column * dimension + row] = 1.0
    return involution


def _generator_with_wrong_right_factor(untouched: bool) -> torch.Tensor:
    """Build the generator without transposing the right-hand factors.

    This is the mistake the vectorization identity guards against: ``vec(A rho)``
    is ``(A (x) I) vec(rho)``, so a right factor that is left alone is a
    different operator whenever the matrix is not symmetric.
    """

    hamiltonian = _hamiltonian()
    identity = torch.eye(_DIMENSION, dtype=torch.complex128)
    right = hamiltonian if untouched else hamiltonian.T
    generator = -1j * (
        torch.kron(hamiltonian, identity)
        - torch.kron(identity, right.contiguous())
    )
    for operator in _collapse(_WIRES):
        product = torch.conj(operator).T @ operator
        generator = generator + torch.kron(operator, torch.conj(operator))
        generator = generator - 0.5 * (
            torch.kron(product, identity)
            + torch.kron(identity, product.transpose(0, 1).contiguous())
        )
    return generator


def test_the_vectorized_generator_reproduces_the_density_form_application() -> None:
    generator, state, dense = _generator_and_state()
    vectorized = state.reshape(-1)

    applied = generator.derivative(state).reshape(-1)
    expected = dense @ vectorized

    # Measured 2.776e-17 for the largest entry of the difference against a
    # largest entry of 2.556e-01 in the result, so the agreement is at the
    # round-off floor of complex128 rather than at any tolerance of choice.
    assert float(torch.max(torch.abs(applied - expected))) < 1e-14

    # Leaving the right-hand factors untransposed is a different operator:
    # measured 2.216e-01 against the same 2.556e-01 scale, which is 87 percent
    # of the signal and cannot hide under the round-off floor above.
    untransposed = _generator_with_wrong_right_factor(untouched=True)
    assert float(torch.max(torch.abs(untransposed @ vectorized - expected))) > 1e-2


def test_the_vectorized_generator_preserves_the_trace() -> None:
    generator, _, dense = _generator_and_state()
    identity_vector = torch.eye(_DIMENSION, dtype=torch.complex128).reshape(-1)

    # Trace preservation is exactly the statement that the transpose of the
    # generator annihilates the vectorized identity.
    assert float(torch.max(torch.abs(dense.T @ identity_vector))) < 1e-14

    # Dropping the 1/2 on the anticommutator is the classic way to lose trace
    # preservation: measured 9.000e-02 here against a scale of 4.900e-01 for
    # the jump term it comes from, so the assertion above is discriminating.
    identity = torch.eye(_DIMENSION, dtype=torch.complex128)
    halved = -1j * (
        torch.kron(_hamiltonian(), identity)
        - torch.kron(identity, _hamiltonian().T.contiguous())
    )
    for operator in _collapse(_WIRES):
        product = torch.conj(operator).T @ operator
        halved = halved + torch.kron(operator, torch.conj(operator))
        halved = halved - 0.25 * (
            torch.kron(product, identity)
            + torch.kron(identity, product.transpose(0, 1).contiguous())
        )
    assert float(torch.max(torch.abs(halved.T @ identity_vector))) > 1e-2


def test_the_vectorized_generator_preserves_hermiticity() -> None:
    generator, state, dense = _generator_and_state()
    involution = _transpose_involution(_DIMENSION)
    image = dense @ state.reshape(-1)

    # The advance stays hermitian: measured 2.861e-17 of anti-hermitian part
    # against 2.556e-01 of signal.
    defect = float(torch.max(torch.abs(image - involution @ torch.conj(image))))
    assert defect < 1e-14

    # Commuting with the transpose alone would not be the right statement, and
    # is false here: measured 8.000e-01, so the transpose and the conjugation
    # are both load-bearing.
    ones = torch.ones(_DIMENSION * _DIMENSION, dtype=torch.complex128)
    assert (
        float(torch.max(torch.abs((dense @ involution - involution @ dense) @ ones)))
        > 1e-2
    )

    # The antiunitary involution ``v -> T conj(v)`` is the one that commutes,
    # on every vector rather than only on hermitian ones: measured 2.220e-16
    # for a generic complex vector against 2.184e+00 of scale.
    generic = torch.randn(_DIMENSION * _DIMENSION, dtype=torch.complex128)
    generic = generic + 1j * torch.randn(_DIMENSION * _DIMENSION)
    conjugation = involution @ torch.conj(generic)
    assert (
        float(
            torch.max(
                torch.abs(
                    involution @ torch.conj(dense @ generic) - dense @ conjugation
                )
            )
        )
        < 1e-12
    )


def test_the_vectorized_generator_is_dissipative() -> None:
    _, _, dense = _generator_and_state()

    # Every eigenvalue of a Lindblad generator has a non-positive real part.
    # Measured 1.664e-16, which is round-off around zero rather than a positive
    # decay rate.
    assert float(torch.max(torch.linalg.eigvals(dense).real)) < 1e-12

    # Flipping the sign of the jump contribution turns the anticommutator into
    # an amplification: measured 9.200e-01. The generator is therefore not
    # accidentally dissipative for every sign convention.
    identity = torch.eye(_DIMENSION, dtype=torch.complex128)
    amplified = -1j * (
        torch.kron(_hamiltonian(), identity)
        - torch.kron(identity, _hamiltonian().T.contiguous())
    )
    for operator in _collapse(_WIRES):
        product = torch.conj(operator).T @ operator
        amplified = amplified + torch.kron(operator, torch.conj(operator))
        amplified = amplified + 0.5 * (
            torch.kron(product, identity)
            + torch.kron(identity, product.transpose(0, 1).contiguous())
        )
    assert float(torch.max(torch.linalg.eigvals(amplified).real)) > 1e-2


def test_the_dense_generator_is_refused_above_its_byte_ceiling() -> None:
    # The entry count is ``(2**n)**4`` and the default ceiling admits exactly
    # n = 5, where the generator is 1024 x 1024 and fills 16 MiB to the byte.
    assert DEFAULT_DENSE_GENERATOR_BYTES == 16 * 1024 * 1024
    admissible = Liouvillian(
        0.0 * torch.eye(32, dtype=torch.complex128), [], hilbert_dimension=32
    )
    assert admissible.dense().numel() * 16 == DEFAULT_DENSE_GENERATOR_BYTES

    refused = Liouvillian(
        0.0 * torch.eye(64, dtype=torch.complex128), [], hilbert_dimension=64
    )
    with pytest.raises(ValueError) as error:
        refused.dense()

    message = str(error.value)
    assert "4096 x 4096" in message
    assert "268435456 bytes" in message
    assert "apply the generator instead" in message

    # The ceiling is a parameter, so the same refusal is reachable at any size.
    small = Liouvillian(
        0.0 * torch.eye(8, dtype=torch.complex128), [], hilbert_dimension=8
    )
    with pytest.raises(ValueError):
        small.dense(max_bytes=64)
    assert small.dense(max_bytes=65536).numel() == 4096

    # The refusal is priced in the generator's own element width, so the same
    # shape at half the precision is refused with half the byte count: 4096 x
    # 4096 complex64 entries is 134217728 bytes, not 268435456. An estimate
    # that assumed one width would report the complex128 figure for both.
    narrow = Liouvillian(
        0.0 * torch.eye(64, dtype=torch.complex64), [], hilbert_dimension=64
    )
    assert narrow.dtype == torch.complex64
    with pytest.raises(ValueError) as narrow_error:
        narrow.dense()
    assert "134217728 bytes" in str(narrow_error.value)

    # The declared ceiling is this generator's, and it must reach the map that
    # materializes the matrix rather than letting the map apply its own default:
    # at 48 dimensions the generator is 2304 x 2304 complex64 entries, 42467328
    # bytes, which the map's own 16 MiB default would refuse. Measured numel
    # 5308416 once the ceiling is raised above it.
    raised = Liouvillian(
        0.0 * torch.eye(48, dtype=torch.complex64), [], hilbert_dimension=48
    )
    with pytest.raises(ValueError):
        raised.dense()
    assert raised.dense(max_bytes=1 << 26).numel() == 2304 * 2304


def test_the_exponential_action_reproduces_a_dense_matrix_exponential() -> None:
    generator, state, dense = _generator_and_state()
    calls: list[torch.Tensor] = []

    def apply(vector: torch.Tensor) -> torch.Tensor:
        calls.append(vector)
        return generator.derivative(vector.reshape(_DIMENSION, _DIMENSION)).reshape(-1)

    advanced, defect = exponential_action(
        apply, 0.5, state.reshape(-1), tolerance=1e-14, max_iterations=64, restart=30
    )
    exact = torch.linalg.matrix_exp(0.5 * dense) @ state.reshape(-1)

    # Measured 1.178e-16 of absolute deviation from the dense exponential of the
    # same matrix, against 2.093e-01 of norm, at a reported defect of 1.278e-15.
    assert float(torch.linalg.vector_norm(advanced - exact)) < 1e-12
    assert defect < 1e-12
    # The action is matrix-free: the generator was applied, never formed.
    assert len(calls) < 30


def test_the_exponential_action_reports_a_defect_that_bounds_the_error() -> None:
    generator, state, dense = _generator_and_state()
    exact = (torch.linalg.matrix_exp(dense) @ state.reshape(-1)).reshape(
        _DIMENSION, _DIMENSION
    )
    grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    measured: list[tuple[float, float]] = []
    for tolerance in (1e-3, 1e-6, 1e-9, 1e-12, 1e-14):
        advanced = state.clone()
        defect = 0.0
        for start, stop in zip(grid[:-1], grid[1:], strict=True):
            advanced, defect = advance_interval(
                generator.derivative,
                advanced,
                stop - start,
                method="krylov-exponential",
                solve_tolerance=tolerance,
                solve_max_iterations=2 * _DIMENSION * _DIMENSION,
            )
        error = float(torch.linalg.vector_norm(advanced - exact)) / float(
            torch.linalg.vector_norm(exact)
        )
        measured.append((defect, error))

    # Each entry is (reported defect, true relative error). The defects fall by
    # about three orders of magnitude per three orders of tolerance, and in
    # every row the reported defect is at or above the error that actually
    # occurred -- measured ratios 9.987e-01, 1.483e+00, 1.982e+00, 2.232e+00,
    # 2.469e+00. A defect that understated the error would make the criterion
    # unusable even if it looked convergent.
    for defect, error in measured:
        assert defect >= error

    # Measured 6.591e-05, 1.133e-07, 3.873e-11, 7.437e-13, 5.589e-15.
    defects = [defect for defect, _ in measured]
    assert defects[0] == pytest.approx(6.591426e-05, rel=0.05)
    assert defects[-1] < 1e-13
    assert defects[0] > defects[1] > defects[2] > defects[3] > defects[4]


def test_the_exponential_step_stops_at_the_krylov_dimension_it_needs() -> None:
    generator, state, _ = _generator_and_state()
    grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    intervals = grid.numel() - 1

    counts: dict[float, int] = {}
    for tolerance in (1e-3, 1e-6, 1e-9, 1e-12):
        applications = 0

        def counting(state_matrix: torch.Tensor) -> torch.Tensor:
            nonlocal applications
            applications += 1
            return generator.derivative(state_matrix)

        advanced = state.clone()
        for start, stop in zip(grid[:-1], grid[1:], strict=True):
            advanced, _ = advance_interval(
                counting,
                advanced,
                stop - start,
                method="krylov-exponential",
                solve_tolerance=tolerance,
                solve_max_iterations=2 * _DIMENSION * _DIMENSION,
            )
        counts[tolerance] = applications

    # Measured 16, 24, 32 and 36 applications over four intervals, that is 4, 6,
    # 8 and 9 per interval. A step that ran the whole restart window regardless
    # of tolerance would apply 30 or 31 times per interval and show 120 or more
    # here, and an implementation that recomputed the exponential from scratch
    # on every iteration would grow quadratically rather than linearly.
    assert counts == {1e-3: 16, 1e-6: 24, 1e-9: 32, 1e-12: 36}
    for tolerance, applications in counts.items():
        assert applications // intervals < 30

    # An unreachable tolerance is still bounded by the window instead of
    # running away: measured 16 applications at a tolerance below complex128
    # resolution, with the tolerance itself reached to 1.326e-31.
    unreachable = 0

    def bounded(state_matrix: torch.Tensor) -> torch.Tensor:
        nonlocal unreachable
        unreachable += 1
        return generator.derivative(state_matrix)

    _, defect = advance_interval(
        bounded,
        state,
        torch.tensor(1.0, dtype=torch.float64),
        method="krylov-exponential",
        solve_tolerance=1e-30,
        solve_max_iterations=1000,
    )
    assert unreachable <= 31
    assert defect < 1e-30


def test_the_exponential_action_refuses_an_argument_that_is_not_a_flat_vector() -> None:
    matrix = torch.eye(4, dtype=torch.complex128)

    with pytest.raises(ValueError, match="must be a flat vector"):
        exponential_action(
            lambda vector: vector,
            1.0,
            matrix,
            tolerance=1e-9,
            max_iterations=8,
            restart=4,
        )


def test_the_exponential_action_refuses_a_restart_below_one() -> None:
    vector = torch.ones(4, dtype=torch.complex128)

    with pytest.raises(ValueError, match="at least one Krylov vector"):
        exponential_action(
            lambda value: value,
            1.0,
            vector,
            tolerance=1e-9,
            max_iterations=8,
            restart=0,
        )


def test_the_exponential_action_on_a_zero_vector_returns_zero_without_applying() -> None:
    applications = 0

    def counting(vector: torch.Tensor) -> torch.Tensor:
        nonlocal applications
        applications += 1
        return vector

    advanced, defect = exponential_action(
        counting,
        torch.tensor(1.0, dtype=torch.float64),
        torch.zeros(4, dtype=torch.complex128),
        tolerance=1e-9,
        max_iterations=8,
        restart=4,
    )

    assert applications == 0
    assert defect == 0.0
    assert torch.equal(advanced, torch.zeros(4, dtype=torch.complex128))


def test_a_pauli_sum_hamiltonian_drives_the_same_generator() -> None:
    hamiltonian = fq.Z(0) + 0.5 * fq.X(1)
    collapse = [0.4 * _embed(LOWERING, 0, _WIRES)]
    matrix_free = Liouvillian(hamiltonian, collapse, hilbert_dimension=_DIMENSION)
    dense = Liouvillian(hamiltonian.dense(), collapse, hilbert_dimension=_DIMENSION)

    assert matrix_free.dtype == torch.complex128
    _, state, _ = _generator_and_state()
    assert torch.allclose(
        matrix_free.derivative(state), dense.derivative(state), atol=1e-14
    )

    times = torch.linspace(0.0, 0.5, 4, dtype=torch.float64)
    plan = fq.lindblad.plan(
        hamiltonian,
        torch.eye(4, dtype=torch.complex128) / 4,
        times,
        collapse_operators=[amplitude_damping(rate=0.5, wire=0)],
        method="krylov-exponential",
        return_density_matrices=True,
    )
    assert plan.order is None
    result = fq.lindblad.run(plan)
    reference = evolve_density_matrix(
        hamiltonian.dense(),
        torch.eye(4, dtype=torch.complex128) / 4,
        2,
        times,
        [amplitude_damping(rate=0.5, wire=0)],
        method="krylov-exponential",
        return_density_matrices=True,
        solve_tolerance=1e-12,
    )
    assert result.density_matrices is not None
    assert reference.density_matrices is not None
    # The matrix-free Hamiltonian and its dense expansion must produce the same
    # trajectory, since they are the same generator.
    assert torch.allclose(
        result.density_matrices, reference.density_matrices, atol=1e-12
    )


def test_the_exponential_scheme_has_no_step_size_error_on_the_grid() -> None:
    hamiltonian = _hamiltonian()
    _, state, _ = _generator_and_state()
    operators = [amplitude_damping(rate=0.3, wire=0)]
    exact_generator = Liouvillian(
        hamiltonian,
        [
            torch.sqrt(torch.tensor(0.3, dtype=torch.complex128))
            * _embed(LOWERING, 0, _WIRES)
        ],
        hilbert_dimension=_DIMENSION,
    ).dense()
    exact = (torch.linalg.matrix_exp(exact_generator) @ state.reshape(-1)).reshape(
        _DIMENSION, _DIMENSION
    )

    errors: dict[str, list[float]] = {"runge-kutta": [], "crank-nicolson": [], "krylov-exponential": []}
    for points in (3, 5, 9, 17):
        grid = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
        for method, collected in errors.items():
            extra = {} if method == "runge-kutta" else {"solve_tolerance": 1e-12}
            result = evolve_density_matrix(
                hamiltonian,
                state,
                _WIRES,
                grid,
                operators,
                method=method,
                return_density_matrices=True,
                **extra,
            )
            assert result.density_matrices is not None
            collected.append(
                float(torch.max(torch.abs(result.density_matrices[-1] - exact)))
            )

    # Measured worst-entry errors, coarse grid first:
    #   runge-kutta         3.779e-06  2.316e-07  1.433e-08  8.913e-10
    #   crank-nicolson      7.088e-04  1.780e-04  4.454e-05  1.114e-05
    #   krylov-exponential  2.319e-14  6.799e-15  1.527e-14  2.581e-14
    # The two order-p schemes fall with the step; the exponential scheme does
    # not improve with refinement because it is already at the round-off floor
    # of the generator it exponentiates.
    assert errors["runge-kutta"][0] > errors["runge-kutta"][-1] * 1000
    assert errors["crank-nicolson"][0] > errors["crank-nicolson"][-1] * 10
    for value in errors["krylov-exponential"]:
        assert value < 1e-12
    assert min(errors["krylov-exponential"]) < errors["runge-kutta"][-1] / 1e3


def test_the_exponential_scheme_advances_a_stiff_decay_without_a_step_limit() -> None:
    rate = 1000.0
    times = torch.linspace(0.0, 0.1, 11, dtype=torch.float64)
    initial = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    exact_population = float(torch.exp(torch.tensor(-rate * 0.1)))

    result = evolve_density_matrix(
        torch.zeros((2, 2), dtype=torch.complex128),
        initial,
        1,
        times,
        [amplitude_damping(rate=rate, wire=0)],
        method="krylov-exponential",
        return_density_matrices=True,
        solve_tolerance=1e-12,
    )

    # rate * interval is 10 here, five times the explicit stability limit. The
    # exponential scheme is exact for this decay: measured 9.358e-14 in the
    # excited population against the closed form 3.784e-44, with a trace drift
    # of 6.551e-12, where the trapezoidal rule leaves 4.370e-08 and the explicit
    # scheme leaves 3.245e-21 but is only stable because the interval is short.
    assert float(result.populations[-1, 1]) == pytest.approx(exact_population, abs=1e-12)
    assert result.maximum_trace_drift < 1e-9
    assert result.population_bounded
    assert result.order is None
    assert result.method == "krylov-exponential"
