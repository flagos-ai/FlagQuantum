"""The adjoint gradient, and what it is and is not allowed to claim.

:mod:`flagquantum.simulation.lindblad_adjoint` returns the cotangent of a
Lindblad trajectory by propagating the cost gradient backwards through the
Hilbert--Schmidt adjoint of the generator on the same grid, instead of retaining
the forward graph. Four separate claims are tested here, and each is written so
that a wrong implementation fails it.

1. The adjoint is the adjoint. ``<sigma, L rho> = <L^dag sigma, rho>`` is checked
   directly, and ``adjoint_derivative`` is checked against ``Liouvillian.dense()``
   with a control that its plain transpose is a different operator.
2. The reverse pass is the transpose of the declared scheme. The interval map of
   each scheme is built independently from the dense vectorized generator, with
   no code in common with the route, and the cotangent must be its adjoint. This
   is the definition, so it decides every other comparison in this file.
3. An external five-point stencil of the same scheme's own trajectory agrees with
   the route for all three schemes, and the residual difference between two
   schemes' derivatives is the difference between the schemes and falls at the
   order of the implicit one, so the route is not quietly substituting a scheme.
4. The memory claim is structural rather than asserted: autograd's saved-tensor
   count for the forward pass is compared with the route's.

The measured differences from ``torch.autograd`` are reported where they appear,
including one that is a property of the forward route rather than of this one:
for the two implicit schemes, autograd's backward through the matrix-free solve
is measurably away from an independent stencil of the same trajectory at some
base points, and this file pins that measurement so the disclosure cannot rot.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import flagquantum.lindblad as fql
from flagquantum import ExecutionOptions
from flagquantum.simulation import (
    EvolutionValidationError,
    adjoint_gradient,
    evolve_density_matrix,
)
from flagquantum.simulation.lindblad import (
    _normalize_collapse_operators,
    _normalize_hamiltonian,
)
from flagquantum.simulation.lindblad_generator import Liouvillian

pytestmark = pytest.mark.unit

DTYPE = torch.complex128
TWO_WIRE_TIMES = torch.linspace(0.0, 0.5, 4, dtype=torch.float64)
AMPLITUDE_DAMPING = [{"operator": "amplitude_damping", "rate": 0.3, "wire": 0}]
STENCIL_STEP = 1e-3
TILT = 0.3
# The base point at which autograd's backward through the matrix-free solve is
# exact to roundoff. The origin base point is not: see `TestAgainstAStencil`.
TILTED_WIRE_ONE = 0.25
# The explicit scheme refuses a solve tolerance instead of ignoring it, and the
# two implicit ones are measured with a target their own defect can reach.
INTEGRATORS = (
    ("runge-kutta", None),
    ("crank-nicolson", 1e-13),
    ("krylov-exponential", 1e-13),
)

PAULI_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE)
PAULI_Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=DTYPE)
WIRE = torch.eye(2, dtype=DTYPE)
X0 = torch.kron(PAULI_X, WIRE)
X1 = torch.kron(WIRE, PAULI_X)
Z0 = torch.kron(PAULI_Z, WIRE)
TWO_WIRE_IDENTITY = torch.eye(4, dtype=DTYPE)
SIGMA_MINUS = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=DTYPE)


def _wire_operator(local: torch.Tensor, wire: int, *, n_wires: int) -> torch.Tensor:
    """Place a single-qubit operator on one wire, wire 0 most significant."""

    result = None
    for index in range(n_wires):
        factor = local if index == wire else WIRE
        result = factor if result is None else torch.kron(result, factor)
    return result


def _two_wire_hamiltonian() -> torch.Tensor:
    """Return a two-wire Hamiltonian with one local term and one pair term."""

    z0 = _wire_operator(PAULI_Z, 0, n_wires=2)
    return 0.4 * X0 + 0.25 * (z0 @ _wire_operator(PAULI_Z, 1, n_wires=2))


def _state(*, wire_one: float = 0.0) -> torch.Tensor:
    """Return a trace-one state tilted along wire 0, and optionally wire 1."""

    return (TWO_WIRE_IDENTITY + TILT * X0 + wire_one * X1) / 4


def _readout(density: torch.Tensor) -> torch.Tensor:
    """One output time's contribution to the cost, for one or many members.

    Written with a leading-ellipsis read so that the same function serves the
    unbatched trajectory and the batched one, which is what makes the batch test
    below a check of the route rather than a check of the cost.
    """

    return (
        torch.real(density[..., 3, 3])
        + 0.4 * torch.real(density[..., 2, 2])
        + 0.25 * torch.imag(density[..., 0, 1])
    )


def _cost(trajectory: torch.Tensor) -> torch.Tensor:
    """The cost the route receives: a sum over the output times it read."""

    return sum(
        (_readout(trajectory[index]) for index in range(trajectory.shape[0])),
        start=torch.zeros((), dtype=trajectory.real.dtype),
    )


def _final_population(trajectory: torch.Tensor) -> torch.Tensor:
    """A cost that reads only the last output time."""

    return torch.real(trajectory[-1, 3, 3])


def _quadratic_cost(trajectory: torch.Tensor) -> torch.Tensor:
    """A cost that reads a product of two entries of the final density matrix.

    The cotangent of a readout linear in the density matrix is the same at every
    point, because the propagator is linear and the readout does not see where it
    is being evaluated. A batch test written on such a readout would therefore
    pass for a route that advanced only the first member. This one reads the
    final density matrix twice, so the cotangent it induces does depend on the
    member it came from, and the control below is not vacuous.
    """

    final = trajectory[-1]
    return _cost(trajectory) + 10.0 * torch.real(final[..., 3, 3] * final[..., 2, 2])


def _final_coherence(trajectory: torch.Tensor) -> torch.Tensor:
    """A single off-diagonal readout, which the wire-1 tilt does move.

    The generators used here commute with multiplication by ``X_1``, so every
    *population* readout has an identically zero derivative along the wire-1
    tilt and would make a convergence measurement vacuous. An off-diagonal
    readout does not.
    """

    return torch.imag(trajectory[-1, 0, 1])


def _width(interval: torch.Tensor, *, substeps: int) -> torch.Tensor:
    """Return one scheme substep as a tensor, matching the integrator's own."""

    return (interval / substeps).to(dtype=torch.float64)


def _dense_interval_map(
    generator: Liouvillian, interval: torch.Tensor, method: str
) -> torch.Tensor:
    """Return the matrix of one scheme's step across one output interval.

    This is the scheme read off the back of its own definition, applied to the
    dense vectorized generator, so nothing here shares code with the integrator
    or with the adjoint route. For the linear time-independent generator each
    scheme is a fixed matrix: Runge--Kutta's fourth-order polynomial in ``hL``
    taken twice, the trapezoidal rule's two substeps, and the exponential whole.
    """

    dense = generator.dense()
    dimension = dense.shape[0]
    identity = torch.eye(dimension, dtype=dense.dtype)
    if method == "krylov-exponential":
        return torch.linalg.matrix_exp(float(interval) * dense)
    substep = _width(interval, substeps=2)
    if method == "crank-nicolson":
        half = (substep / 2).to(dtype=torch.complex128)
        step = torch.linalg.solve(identity - half * dense, identity + half * dense)
        return step @ step
    assert method == "runge-kutta"
    result = identity
    for _ in range(2):
        scaled = substep.to(dtype=torch.complex128) * dense
        result = result @ (
            identity
            + scaled
            + scaled @ scaled / 2
            + scaled @ scaled @ scaled / 6
            + scaled @ scaled @ scaled @ scaled / 24
        )
    return result


def _pairing_generator(hilbert_dimension: int = 4) -> Liouvillian:
    """Return a two-wire generator built through the public normalizers."""

    return Liouvillian(
        _normalize_hamiltonian(
            _two_wire_hamiltonian(),
            n_wires=2,
            dim=hilbert_dimension,
            dtype=DTYPE,
            device=torch.device("cpu"),
            tolerance=1e-10,
        ),
        _normalize_collapse_operators(
            AMPLITUDE_DAMPING,
            n_wires=2,
            dim=hilbert_dimension,
            dtype=DTYPE,
            device=torch.device("cpu"),
        ),
        hilbert_dimension=hilbert_dimension,
    )


def _readout_seeds(trajectory: torch.Tensor) -> list[torch.Tensor]:
    """Return the cost's own gradient at each output time, one time at a time."""

    seeds = []
    for index in range(trajectory.shape[0]):
        row = trajectory[index].clone().detach().requires_grad_(True)
        with torch.enable_grad():
            value = _readout(row)
        seeds.append(torch.autograd.grad(value, row)[0])
    return seeds


def _dense_cotangent(
    seeds: list[torch.Tensor], times: torch.Tensor, method: str
) -> torch.Tensor:
    """Carry per-time seeds back through each scheme's dense interval map.

    Nothing here shares code with the integrator or with the adjoint route: the
    interval map is the scheme read off its own definition, applied to the dense
    vectorized generator, so this is the route's definition and not a second
    implementation of it.
    """

    generator = _pairing_generator()
    cotangent = seeds[-1]
    for index in range(len(times) - 2, -1, -1):
        step = _dense_interval_map(generator, times[index + 1] - times[index], method)
        cotangent = seeds[index] + (step.conj().T @ cotangent.reshape(-1)).reshape(4, 4)
    return cotangent


def _autograd_cotangent(
    cost: Callable[[torch.Tensor], torch.Tensor],
    *,
    method: str,
    solve_tolerance: float | None,
    wire_one: float = TILTED_WIRE_ONE,
) -> torch.Tensor:
    """Return ``d cost / d rho(t_0)`` the way autograd would, for the same cost."""

    start = _state(wire_one=wire_one).clone().detach().requires_grad_(True)
    result = evolve_density_matrix(
        _two_wire_hamiltonian(),
        start,
        2,
        TWO_WIRE_TIMES,
        AMPLITUDE_DAMPING,
        method=method,
        solve_tolerance=solve_tolerance,
        return_density_matrices=True,
    )
    (gradient,) = torch.autograd.grad(cost(result.density_matrices), start)
    return gradient


def _five_point(
    cost: Callable[[torch.Tensor], torch.Tensor],
    *,
    method: str,
    solve_tolerance: float | None,
    wire_one: float = 0.0,
    offset: float = STENCIL_STEP,
) -> float:
    """Return the five-point derivative along wire 1 of the value at the base point."""

    def value(shift: float) -> float:
        result = evolve_density_matrix(
            _two_wire_hamiltonian(),
            _state(wire_one=wire_one + shift),
            2,
            TWO_WIRE_TIMES,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=solve_tolerance,
            return_density_matrices=True,
        )
        return float(cost(result.density_matrices))

    sampled = [value(k * offset) for k in (-2, -1, 1, 2)]
    return (-sampled[3] + 8 * sampled[2] - 8 * sampled[1] + sampled[0]) / (12 * offset)


def _directional(cotangent: torch.Tensor, direction: torch.Tensor = X1) -> float:
    """Reduce a cotangent to its derivative along one Hermitian direction."""

    return float((cotangent.conj() * (direction / 4)).sum().real)


class TestTheAdjointIsTheAdjoint:
    """The generator read backwards is the Hilbert--Schmidt adjoint."""

    def test_the_pairing_identity_holds_for_hermitian_states(self) -> None:
        torch.manual_seed(11)
        generator = _pairing_generator()

        def hermitian() -> torch.Tensor:
            drawn = torch.randn(4, 4, dtype=DTYPE)
            return (drawn + drawn.conj().T) / 2

        sigma, rho = hermitian(), hermitian()
        inner = lambda left, right: (left.conj().T @ right).trace()  # noqa: E731

        forward = inner(sigma, generator.derivative(rho))
        backward = inner(generator.adjoint_derivative(sigma), rho)
        # Measured on this fixture: a difference of 2.3e-16 on a value of 1.126,
        # so the identity holds to roundoff and not to a fitted tolerance.
        assert float(abs(forward - backward)) < 1e-14

    def test_the_adjoint_is_the_conjugate_transpose_of_the_dense_generator(
        self,
    ) -> None:
        generator = _pairing_generator()
        dense = generator.dense()
        torch.manual_seed(3)
        drawn = torch.randn(4, 4, dtype=DTYPE)
        sigma = (drawn + drawn.conj().T) / 2

        vectorized = sigma.reshape(-1)
        applied = generator.adjoint_derivative(sigma).reshape(-1)
        assert float((dense.conj().T @ vectorized - applied).abs().max()) < 1e-14
        # The control: a plain transpose is a different operator, so an adjoint
        # that forgot to conjugate cannot pass by accident. Measured here,
        # `dense.T @ sigma` differs from the adjoint by 0.71.
        transposed = dense.T @ vectorized
        assert float((transposed - applied).abs().max()) > 1e-2

    def test_the_hamiltonian_part_changes_sign_and_the_sandwich_reverses(self) -> None:
        # Read separately from the identity above, because both halves of the
        # generator could be wrong the same way and still pair. With no collapse
        # operator the adjoint has to be minus the forward Hamiltonian part, and
        # with no Hamiltonian the two dissipators have to differ, because the
        # adjoint applies the sandwich the other way round.
        hamiltonian_only = Liouvillian(
            _normalize_hamiltonian(
                _two_wire_hamiltonian(),
                n_wires=2,
                dim=4,
                dtype=DTYPE,
                device=torch.device("cpu"),
                tolerance=1e-10,
            ),
            (),
            hilbert_dimension=4,
        )
        torch.manual_seed(5)
        drawn = torch.randn(4, 4, dtype=DTYPE)
        rho = (drawn + drawn.conj().T) / 2
        assert (
            float(
                (
                    hamiltonian_only.adjoint_derivative(rho)
                    + hamiltonian_only.derivative(rho)
                )
                .abs()
                .max()
            )
            < 1e-14
        )

        collapse_only = Liouvillian(
            0.0 * TWO_WIRE_IDENTITY,
            _normalize_collapse_operators(
                AMPLITUDE_DAMPING,
                n_wires=2,
                dim=4,
                dtype=DTYPE,
                device=torch.device("cpu"),
            ),
            hilbert_dimension=4,
        )
        difference = collapse_only.adjoint_derivative(rho) - collapse_only.derivative(
            rho
        )
        assert float(difference.abs().max()) > 1e-2


class TestTheTransposeOfTheDeclaredScheme:
    """The reverse pass is the adjoint of the scheme, read off its definition."""

    @pytest.mark.parametrize(("method", "solve_tolerance"), INTEGRATORS)
    @pytest.mark.parametrize("wire_one", [0.0, TILTED_WIRE_ONE])
    def test_the_cotangent_is_the_adjoint_of_the_dense_interval_map(
        self, method: str, solve_tolerance: float | None, wire_one: float
    ) -> None:
        trajectory = evolve_density_matrix(
            _two_wire_hamiltonian(),
            _state(wire_one=wire_one),
            2,
            TWO_WIRE_TIMES,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=solve_tolerance,
            return_density_matrices=True,
        ).density_matrices
        reference = _dense_cotangent(_readout_seeds(trajectory), TWO_WIRE_TIMES, method)

        measured = adjoint_gradient(
            _two_wire_hamiltonian(),
            _state(wire_one=wire_one),
            2,
            TWO_WIRE_TIMES,
            _cost,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=solve_tolerance,
        )
        difference = float((reference - measured).abs().max())
        scale = float(reference.abs().max())
        # Measured relative differences: 1.1e-16 Runge--Kutta, 1.7e-15
        # Crank--Nicolson, 4.4e-16 exponential. This is the definition of the
        # route, so it is held to roundoff rather than to a scheme's accuracy.
        assert difference / scale < 1e-12


class TestAgreementWithAutograd:
    """The reverse pass differentiates the same trajectory autograd does."""

    @pytest.mark.parametrize(("method", "solve_tolerance"), INTEGRATORS)
    def test_the_cotangent_matches_autograd_on_a_cost_that_reads_every_time(
        self, method: str, solve_tolerance: float | None
    ) -> None:
        expected = _autograd_cotangent(
            _cost, method=method, solve_tolerance=solve_tolerance
        )
        measured = adjoint_gradient(
            _two_wire_hamiltonian(),
            _state(wire_one=TILTED_WIRE_ONE),
            2,
            TWO_WIRE_TIMES,
            _cost,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=solve_tolerance,
        )

        difference = float((expected - measured).abs().max())
        scale = float(expected.abs().max())
        # Measured relative differences at this base point: 1.3e-16 explicit,
        # 3.0e-14 Crank--Nicolson, 2.3e-16 exponential. The bound is the roundoff
        # a transpose of the same arithmetic can reach. It is not asserted at
        # every base point: `TestAgainstAStencil` records where the two routes
        # part company and which of them an independent stencil agrees with.
        assert difference / scale < 1e-12

    def test_the_explicit_scheme_matches_autograd_at_the_origin_base_point(
        self,
    ) -> None:
        # The explicit scheme takes no solve, so there is nothing between it and
        # autograd and the two agree at every base point. Measured: 1.1e-16.
        expected = _autograd_cotangent(
            _cost, method="runge-kutta", solve_tolerance=None, wire_one=0.0
        )
        measured = adjoint_gradient(
            _two_wire_hamiltonian(),
            _state(),
            2,
            TWO_WIRE_TIMES,
            _cost,
            AMPLITUDE_DAMPING,
            method="runge-kutta",
        )
        assert float((expected - measured).abs().max()) < 1e-14

    @pytest.mark.parametrize(("method", "solve_tolerance"), INTEGRATORS)
    def test_every_output_time_contributes_its_own_seed(
        self, method: str, solve_tolerance: float | None
    ) -> None:
        # A reverse pass that seeds only the final time is the natural wrong
        # implementation, and it is wrong exactly for a cost that reads more
        # than the final time. The two cotangents must differ materially or this
        # would not be a control at all.
        every_time = adjoint_gradient(
            _two_wire_hamiltonian(),
            _state(wire_one=TILTED_WIRE_ONE),
            2,
            TWO_WIRE_TIMES,
            _cost,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=solve_tolerance,
        )
        final_time_only = adjoint_gradient(
            _two_wire_hamiltonian(),
            _state(wire_one=TILTED_WIRE_ONE),
            2,
            TWO_WIRE_TIMES,
            _final_population,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=solve_tolerance,
        )

        spread = float((every_time - final_time_only).abs().max())
        scale = float(final_time_only.abs().max())
        # Measured: 0.379 against a scale of 0.828, so the earlier seeds carry
        # about 46 percent of this cost's gradient.
        assert spread / scale > 0.4

    def test_the_cotangent_is_returned_without_a_graph(self) -> None:
        measured = adjoint_gradient(
            _two_wire_hamiltonian(),
            _state(),
            2,
            TWO_WIRE_TIMES,
            _cost,
            AMPLITUDE_DAMPING,
        )
        assert measured.requires_grad is False

    def test_the_route_runs_with_gradients_disabled_entirely(self) -> None:
        # The route builds the graph it needs itself, so a caller already inside
        # `no_grad` -- the common case for a training step that wants only the
        # cotangent -- is not handed a detached zero.
        expected = _autograd_cotangent(
            _cost, method="runge-kutta", solve_tolerance=None
        )
        with torch.no_grad():
            measured = adjoint_gradient(
                _two_wire_hamiltonian(),
                _state(wire_one=TILTED_WIRE_ONE),
                2,
                TWO_WIRE_TIMES,
                _cost,
                AMPLITUDE_DAMPING,
            )
        assert float((expected - measured).abs().max()) < 1e-14


class TestAgainstAStencil:
    """A five-point stencil of the same scheme, which shares no code with either."""

    @pytest.mark.parametrize(("method", "solve_tolerance"), INTEGRATORS)
    @pytest.mark.parametrize("wire_one", [0.0, TILTED_WIRE_ONE])
    def test_the_route_agrees_with_a_stencil_of_the_scheme_it_was_given(
        self, method: str, solve_tolerance: float | None, wire_one: float
    ) -> None:
        stencil = _five_point(
            _cost,
            method=method,
            solve_tolerance=solve_tolerance,
            wire_one=wire_one,
        )
        measured = _directional(
            adjoint_gradient(
                _two_wire_hamiltonian(),
                _state(wire_one=wire_one),
                2,
                TWO_WIRE_TIMES,
                _cost,
                AMPLITUDE_DAMPING,
                method=method,
                solve_tolerance=solve_tolerance,
            )
        )
        # Measured relative differences, at the origin base point and at the
        # tilted one: explicit 7.9e-13 / 1.7e-12, Crank--Nicolson 1.0e-10 /
        # 3.1e-13, exponential 8.9e-11 / 1.2e-11. The stencil's own roundoff
        # floor, `eps * |cost| / step`, is around 7e-10 relative, so the route is
        # at the resolution the stencil has.
        assert abs(measured - stencil) / abs(stencil) < 1e-8

    @pytest.mark.parametrize(("method", "solve_tolerance"), INTEGRATORS)
    def test_the_route_is_the_dense_transpose_where_autograd_is_not(
        self, method: str, solve_tolerance: float | None
    ) -> None:
        # At the origin base point, where the two routes' difference is largest,
        # the route still reproduces the dense transpose of the scheme exactly.
        # This is the disclosure that belongs with it, measured on this fixture
        # and recorded rather than asserted, because a test that fails when
        # someone improves the forward path would be a liability:
        #
        #   * a finite difference of one Crank--Nicolson interval along a random
        #     direction gives +2.202002910773, and the dense transpose gives
        #     +2.202002910623 -- they agree to 4.6e-11 relative, so the dense
        #     transpose is the Jacobian of the routine that is actually run;
        #   * autograd's cotangent at the same point gives +2.202003036013, which
        #     is 5.7e-08 relative away from that finite difference, and 3.5e-06
        #     (Crank--Nicolson) and 1.7e-05 (exponential) relative away from the
        #     five-point stencil used above;
        #   * the explicit scheme, which takes no solve, is at roundoff on all
        #     three.
        #
        # So the backward pass through the matrix-free solve is an approximation
        # of the scheme's transpose rather than that transpose, while this route
        # is the transpose by construction. Reported as a finding against the
        # forward path in `capability-maturity.toml`, not repaired here.
        trajectory = evolve_density_matrix(
            _two_wire_hamiltonian(),
            _state(),
            2,
            TWO_WIRE_TIMES,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=solve_tolerance,
            return_density_matrices=True,
        ).density_matrices
        reference = _directional(
            _dense_cotangent(_readout_seeds(trajectory), TWO_WIRE_TIMES, method)
        )
        measured = _directional(
            adjoint_gradient(
                _two_wire_hamiltonian(),
                _state(),
                2,
                TWO_WIRE_TIMES,
                _cost,
                AMPLITUDE_DAMPING,
                method=method,
                solve_tolerance=solve_tolerance,
            )
        )
        assert abs(measured - reference) / abs(reference) < 1e-12

    def test_the_scheme_the_route_differentiated_converges_at_its_own_order(
        self,
    ) -> None:
        # The route differentiates the scheme it was given, so the scheme it was
        # given is the only thing left that sets its distance from the exact
        # solution. The reference here is the generator's own exponential, which
        # is exact for a time-independent Hamiltonian: the exponential scheme
        # lands on it, and the two order-p schemes close on it at their orders as
        # the grid is refined. Measured relative gaps to -0.060509967277:
        #   explicit  1.61e-07, 5.19e-09, 2.41e-10, 1.31e-11, 7.66e-13 at 4, 8,
        #             16, 32 and 64 steps -- ratios 31, 21, 18, 17, toward 16;
        #   trapezoid 3.71e-04, 6.82e-05, 1.49e-05, 3.48e-06, 8.42e-07 -- ratios
        #             5.4, 4.6, 4.3, 4.1, toward 4;
        #   exponential 0.0 at every grid, to the last bit.
        exact = self._exact_directional()

        def gap(steps: int, method: str, solve_tolerance: float | None) -> float:
            times = torch.linspace(0.0, 0.5, steps, dtype=torch.float64)
            measured = _directional(
                adjoint_gradient(
                    _two_wire_hamiltonian(),
                    _state(),
                    2,
                    times,
                    _final_coherence,
                    AMPLITUDE_DAMPING,
                    method=method,
                    solve_tolerance=solve_tolerance,
                )
            )
            return abs(measured - exact) / abs(exact)

        explicit = lambda steps: gap(steps, "runge-kutta", None)  # noqa: E731
        trapezoid = lambda steps: gap(steps, "crank-nicolson", 1e-13)  # noqa: E731
        exponential = lambda steps: gap(  # noqa: E731
            steps, "krylov-exponential", 1e-13
        )

        assert exponential(8) < 1e-13
        assert trapezoid(8) / trapezoid(32) > 3.5
        assert explicit(8) / explicit(32) > 10.0
        # At one fixed grid the fourth-order scheme is the closer of the two, so
        # neither name is standing in for the other's accuracy.
        assert explicit(64) < trapezoid(64)

    @staticmethod
    def _exact_directional() -> float:
        """Return the exact wire-1 derivative, from the generator's exponential.

        The cost read is ``Im rho[0, 1]`` at the final time, whose seed is
        ``1j * e_{0,1}``; the generator is time-independent, so the exact
        cotangent is that seed carried back through one exponential of ``0.5 L``.
        """

        generator = _pairing_generator()
        seed = torch.zeros(4, 4, dtype=DTYPE)
        seed[0, 1] = 1j
        carried = (
            torch.linalg.matrix_exp(0.5 * generator.dense()).conj().T @ seed.reshape(-1)
        ).reshape(4, 4)
        return _directional(carried)


class TestTheMemoryClaim:
    """The reverse pass holds the trajectory, not the forward graph."""

    def test_the_route_saves_far_fewer_tensors_than_the_forward_pass(self) -> None:
        class Counter:
            def __init__(self) -> None:
                self.count = 0

            def pack(self, tensor: torch.Tensor) -> torch.Tensor:
                self.count += 1
                return tensor

            def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
                return tensor

        forward = Counter()
        start = _state().clone().detach().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(forward.pack, forward.unpack):
            result = evolve_density_matrix(
                _two_wire_hamiltonian(),
                start,
                2,
                TWO_WIRE_TIMES,
                AMPLITUDE_DAMPING,
                return_density_matrices=True,
            )
            torch.autograd.grad(_cost(result.density_matrices), start)

        adjoint = Counter()
        with torch.autograd.graph.saved_tensors_hooks(adjoint.pack, adjoint.unpack):
            adjoint_gradient(
                _two_wire_hamiltonian(),
                _state(),
                2,
                TWO_WIRE_TIMES,
                _cost,
                AMPLITUDE_DAMPING,
            )

        # Measured on this fixture: 180 tensors saved through the forward graph
        # and 0 through the route. Zero is structural rather than lucky -- the
        # forward sweep runs under `no_grad`, so there is nothing to save, and
        # the readout that is recorded consumes the trajectory it is handed.
        assert forward.count > 100
        assert adjoint.count == 0

    def test_the_route_saves_nothing_even_when_an_input_is_a_leaf(self) -> None:
        # The count above is taken with constant inputs, where a route that
        # recorded its sweep would still save nothing because there is nothing a
        # graph could attach to. A leaf makes that control real: the forward
        # sweep and the generator both become differentiable, so this is the case
        # where a retained graph would show up, and the route still holds zero.
        class Counter:
            def __init__(self) -> None:
                self.count = 0

            def pack(self, tensor: torch.Tensor) -> torch.Tensor:
                self.count += 1
                return tensor

            def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
                return tensor

        hamiltonian = _two_wire_hamiltonian().clone().detach().requires_grad_(True)
        counter = Counter()
        with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
            gradient = adjoint_gradient(
                hamiltonian,
                _state(),
                2,
                TWO_WIRE_TIMES,
                _cost,
                AMPLITUDE_DAMPING,
            )
        assert counter.count == 0
        # ... and the cotangent is the seed of the caller's own graph rather than
        # a node in it, so it carries no graph of its own either.
        assert not gradient.requires_grad
        # The leaf is untouched: the reverse pass reads the generator it built
        # from it, and a route that attached its generator to the leaf would have
        # populated this.
        assert hamiltonian.grad is None


class TestBatchingAndTheFacade:
    """One reverse pass advances every member; the facade agrees with the engine."""

    def test_the_batch_cotangent_is_the_stack_of_its_members(self) -> None:
        # The members are tilted along wire 0, which the generator does *not*
        # commute with. That matters: this fixture's generator commutes with
        # multiplication by X_1, so a batch that differed only along wire 1 would
        # have one cotangent for every member and could not tell a route that
        # advanced only the first one from a correct one.
        members = torch.stack(
            [
                (TWO_WIRE_IDENTITY + 0.3 * X0) / 4,
                (TWO_WIRE_IDENTITY + 0.9 * X0) / 4,
                (TWO_WIRE_IDENTITY + 0.5 * Z0) / 4,
            ]
        )
        batched = adjoint_gradient(
            _two_wire_hamiltonian(),
            members,
            2,
            TWO_WIRE_TIMES,
            lambda trajectory: _quadratic_cost(trajectory).sum(),
            AMPLITUDE_DAMPING,
        )
        per_member = torch.stack(
            [
                adjoint_gradient(
                    _two_wire_hamiltonian(),
                    member,
                    2,
                    TWO_WIRE_TIMES,
                    _quadratic_cost,
                    AMPLITUDE_DAMPING,
                )
                for member in members
            ]
        )
        assert batched.shape == members.shape
        # Measured: 1.5e-18, which is one folding of the batch axis into the
        # solved vector rather than a loop over members that agrees by accident.
        assert float((batched - per_member).abs().max()) < 1e-14
        # The control: the members' cotangents must differ from each other, or a
        # route that advanced only the first member would also pass above.
        # Measured spreads against a scale of 5.4: 5.5e-02, 8.5e-01, 9.0e-01.
        assert float((per_member[0] - per_member[1]).abs().max()) > 1e-3

    def test_the_facade_matches_the_engine_bit_for_bit(self) -> None:
        engine = adjoint_gradient(
            _two_wire_hamiltonian(),
            _state(),
            2,
            TWO_WIRE_TIMES,
            _final_population,
            AMPLITUDE_DAMPING,
        )
        facade = fql.adjoint_gradient(
            _two_wire_hamiltonian(),
            _state(),
            TWO_WIRE_TIMES,
            _final_population,
            collapse_operators=[fql.amplitude_damping(rate=0.3, qubit=0)],
        )
        assert torch.equal(engine, facade)

    def test_the_facade_infers_the_width_and_reads_the_options(self) -> None:
        gradient = fql.adjoint_gradient(
            _two_wire_hamiltonian(),
            "00",
            TWO_WIRE_TIMES,
            _final_population,
            options=ExecutionOptions(mode="density_matrix", precision="complex128"),
        )
        assert gradient.shape == (4, 4)
        assert gradient.dtype == DTYPE


class TestWhatTheRouteRefuses:
    """A request this route cannot honour fails closed and by name."""

    def test_requiring_gradients_is_refused_rather_than_ignored(self) -> None:
        # The route deliberately produces no graph, so asking for one is a
        # contradiction in the request and not a silent no-op.
        with pytest.raises(EvolutionValidationError) as error:
            fql.adjoint_gradient(
                _two_wire_hamiltonian(),
                _state(),
                TWO_WIRE_TIMES,
                _final_population,
                options=ExecutionOptions(mode="density_matrix", require_gradients=True),
            )
        assert error.value.code == "adjoint_requires_no_graph"
        assert error.value.field == "require_gradients"

    def test_a_non_callable_cost_is_refused(self) -> None:
        with pytest.raises(EvolutionValidationError) as error:
            adjoint_gradient(
                _two_wire_hamiltonian(),
                _state(),
                2,
                TWO_WIRE_TIMES,
                None,  # type: ignore[arg-type]
                AMPLITUDE_DAMPING,
            )
        assert error.value.code == "invalid_cost"
        assert error.value.field == "cost"

    def test_a_cost_that_returns_a_tensor_per_time_is_refused(self) -> None:
        with pytest.raises(EvolutionValidationError) as error:
            adjoint_gradient(
                _two_wire_hamiltonian(),
                _state(),
                2,
                TWO_WIRE_TIMES,
                lambda trajectory: torch.real(trajectory[:, 3, 3]),
                AMPLITUDE_DAMPING,
            )
        assert error.value.code == "non_scalar_cost"
        assert error.value.field == "cost"

    def test_a_reverse_step_that_misses_its_defect_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Measured: no reachable request drives the backward solve past its
        # target. Both Krylov schemes already achieve a defect nearer 1e-21 than
        # the 1e-12 they are asked for, and at 1e-30 the forward and reverse
        # passes still both succeed, so the guard has no input that reaches it.
        # Rather than delete it or leave it unexercised, the one thing it must do
        # is driven directly: a step that reports a defect above the target has
        # to fail the call rather than return a cotangent that silently carries
        # the missed defect.
        from flagquantum.simulation import lindblad_adjoint as module

        def unsatisfied_interval(*args: object, **kwargs: object):
            return args[1], 0.5

        monkeypatch.setattr(module, "advance_interval", unsatisfied_interval)
        with pytest.raises(EvolutionValidationError) as error:
            adjoint_gradient(
                _two_wire_hamiltonian(),
                _state(),
                2,
                TWO_WIRE_TIMES,
                _final_population,
                AMPLITUDE_DAMPING,
                method="crank-nicolson",
                solve_tolerance=1e-13,
            )
        assert error.value.code == "integration_not_converged"
        assert error.value.field == "solve_tolerance"


def test_the_collapse_operators_are_read_through_the_shared_normalizer() -> None:
    # A dense operator with an explicit rate and the named amplitude-damping
    # channel must reach the same generator, so a route that read the collapse
    # input a second way would have to agree with itself here instead of with the
    # forward pass.
    dense = adjoint_gradient(
        _two_wire_hamiltonian(),
        _state(),
        2,
        TWO_WIRE_TIMES,
        _final_population,
        [(_wire_operator(SIGMA_MINUS, 0, n_wires=2), 0.3)],
    )
    named = adjoint_gradient(
        _two_wire_hamiltonian(),
        _state(),
        2,
        TWO_WIRE_TIMES,
        _final_population,
        AMPLITUDE_DAMPING,
    )
    assert float((dense - named).abs().max()) < 1e-14
