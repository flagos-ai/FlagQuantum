"""Batched continuous-time evolution and the gradient path through it.

These tests hold three claims that a per-member loop cannot make for the
evolution as a whole: that one trajectory advances every member of a batch,
that one backward pass returns the gradient of the whole batch, and that the
integrators are differentiable at all. Each numeric assertion states the
fixture, the measured margin, and the divergence the wrong rule produces, so a
tolerance that a wrong implementation would also satisfy is not evidence.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.simulation import (
    EvolutionValidationError,
    evolve_density_matrix,
    plan_density_matrix_evolution,
)
from flagquantum.simulation.lindblad import (
    _normalize_collapse_operators,
    _normalize_hamiltonian,
)
from flagquantum.simulation.lindblad_generator import Liouvillian
from flagquantum.simulation.lindblad_integrators import advance_interval
from flagquantum.simulation.matrix_free_exponential import exponential_action
from flagquantum.simulation.matrix_free_hamiltonian import PauliSum, PauliSumTerm
from flagquantum.simulation.matrix_free_linear_solve import gmres

pytestmark = pytest.mark.unit

DTYPE = torch.complex128
DEVICE = torch.device("cpu")
TOLERANCE = 1e-10
STENCIL_STEP = 1e-4
TWO_WIRE_TIMES = torch.linspace(0.0, 0.5, 4, dtype=torch.float64)
FOUR_WIRE_TIMES = torch.linspace(0.0, 0.4, 3, dtype=torch.float64)
AMPLITUDE_DAMPING = [{"operator": "amplitude_damping", "rate": 0.3, "wire": 0}]

PAULI_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE)
PAULI_Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=DTYPE)
WIRE = torch.eye(2, dtype=DTYPE)
X0 = torch.kron(PAULI_X, WIRE)
Z0 = torch.kron(PAULI_Z, WIRE)
IDENTITY_TWO_WIRE = torch.eye(4, dtype=DTYPE)


def _two_wire_batch(lanes: int = 3) -> torch.Tensor:
    """Return `lanes` normalized statevectors with distinct populations."""

    columns = []
    for index in range(lanes):
        alpha = 0.2 + 0.3 * index
        vector = torch.zeros(4, dtype=DTYPE)
        vector[0] = torch.cos(torch.tensor(alpha, dtype=torch.float64))
        vector[3] = torch.sin(torch.tensor(alpha, dtype=torch.float64))
        columns.append(vector)
    return torch.stack(columns)


def _wide_batch(lanes: int = 3) -> torch.Tensor:
    """Return an unbalanced four-wire batch so member contributions differ."""

    columns = []
    for index in range(lanes):
        vector = torch.zeros(16, dtype=DTYPE)
        vector[0] = 1.0 / math.sqrt(2.0)
        vector[3 + index] = 1.0 / math.sqrt(2.0)
        columns.append(vector)
    return torch.stack(columns)


def _wire_operator(local: torch.Tensor, wire: int, *, n_wires: int) -> torch.Tensor:
    """Place a single-qubit operator on one wire, wire 0 most significant."""

    result = None
    for index in range(n_wires):
        factor = local if index == wire else WIRE
        result = factor if result is None else torch.kron(result, factor)
    return result


def _wide_hamiltonian() -> torch.Tensor:
    """Return a four-wire Hamiltonian with a term on wire 0 and on wires 0, 3."""

    z0 = _wire_operator(PAULI_Z, 0, n_wires=4)
    return 0.4 * z0 + 0.25 * (z0 @ _wire_operator(PAULI_Z, 3, n_wires=4))


def _five_point(values: list[float], step: float = STENCIL_STEP) -> float:
    """Return the fourth-order five-point derivative of a sampled function."""

    return (-values[0] + 8 * values[1] - 8 * values[2] + values[3]) / (12 * step)


def _sampled(values: list[float], base: float) -> list[float]:
    """Order a symmetric five-point stencil around ``base``."""

    offsets = (2 * STENCIL_STEP, STENCIL_STEP, -STENCIL_STEP, -2 * STENCIL_STEP)
    by_offset = dict(zip(offsets, values, strict=True))
    return [by_offset[offset] for offset in offsets]


def _two_wire_run(
    scale: torch.Tensor,
    *,
    method: str,
    solve_tolerance: float | None,
    state: torch.Tensor,
    hamiltonian: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the final populations of one run, one population per member."""

    result = evolve_density_matrix(
        (X0 if hamiltonian is None else hamiltonian) * scale.to(DTYPE),
        state,
        2,
        TWO_WIRE_TIMES,
        AMPLITUDE_DAMPING,
        method=method,
        solve_tolerance=solve_tolerance,
    )
    return result.populations


class TestBatchedEvolution:
    """A leading batch axis advances every member under one Hamiltonian."""

    def test_a_batched_statevector_advances_every_member_identically(self) -> None:
        batch = _two_wire_batch()
        batched = evolve_density_matrix(
            IDENTITY_TWO_WIRE, batch, 2, TWO_WIRE_TIMES, AMPLITUDE_DAMPING
        )
        assert batched.populations.shape == (4, 3, 4)
        per_member = [
            evolve_density_matrix(
                IDENTITY_TWO_WIRE, batch[index], 2, TWO_WIRE_TIMES, AMPLITUDE_DAMPING
            ).populations
            for index in range(3)
        ]
        worst = max(
            float((batched.populations[:, index, :] - per_member[index]).abs().max())
            for index in range(3)
        )
        # The batched route folds the members into one flattened vector, so the
        # arithmetic is identical rather than merely close.
        assert worst == 0.0

    def test_a_batched_density_matrix_advances_every_member_identically(self) -> None:
        matrices = torch.stack(
            [torch.outer(vector, vector.conj()) for vector in _two_wire_batch()]
        )
        batched = evolve_density_matrix(
            IDENTITY_TWO_WIRE, matrices, 2, TWO_WIRE_TIMES, AMPLITUDE_DAMPING
        )
        assert batched.populations.shape == (4, 3, 4)
        worst = max(
            float(
                (
                    batched.populations[:, index, :]
                    - evolve_density_matrix(
                        IDENTITY_TWO_WIRE,
                        matrices[index],
                        2,
                        TWO_WIRE_TIMES,
                        AMPLITUDE_DAMPING,
                    ).populations
                )
                .abs()
                .max()
            )
            for index in range(3)
        )
        assert worst == 0.0

    @pytest.mark.parametrize("method", ["runge-kutta", "crank-nicolson"])
    def test_both_input_forms_of_one_member_agree(self, method: str) -> None:
        vector = _two_wire_batch(1)[0]
        from_vector = evolve_density_matrix(
            IDENTITY_TWO_WIRE,
            vector,
            2,
            TWO_WIRE_TIMES,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=None if method == "runge-kutta" else 1e-12,
        )
        from_matrix = evolve_density_matrix(
            IDENTITY_TWO_WIRE,
            torch.outer(vector, vector.conj()),
            2,
            TWO_WIRE_TIMES,
            AMPLITUDE_DAMPING,
            method=method,
            solve_tolerance=None if method == "runge-kutta" else 1e-12,
        )
        assert from_vector.populations.shape == (4, 4)
        assert from_matrix.populations.shape == (4, 4)
        assert float(
            (from_vector.populations - from_matrix.populations).abs().max()
        ) < 1e-12

    def test_the_batch_axis_is_kept_by_every_integrator(self) -> None:
        batch = _two_wire_batch()
        for method, solve_tolerance in (
            ("runge-kutta", None),
            ("crank-nicolson", 1e-12),
            ("krylov-exponential", 1e-12),
        ):
            result = evolve_density_matrix(
                X0,
                batch,
                2,
                TWO_WIRE_TIMES,
                AMPLITUDE_DAMPING,
                method=method,
                solve_tolerance=solve_tolerance,
            )
            assert result.populations.shape == (4, 3, 4), method
            assert result.method == method
            assert result.maximum_trace_drift < 1e-11, method

    def test_a_named_observable_keeps_the_batch_axis(self) -> None:
        result = fql.run(
            IDENTITY_TWO_WIRE,
            _two_wire_batch(),
            TWO_WIRE_TIMES,
            n_qubits=2,
            outputs=[fq.expectation(fq.Z(0), name="z0")],
            options=fq.ExecutionOptions(batch_size=3),
        )
        assert tuple(result.populations.shape) == (4, 3, 4)
        assert tuple(result.observables["z0"].shape) == (4, 3)

    def test_an_unbatched_request_keeps_the_established_shapes(self) -> None:
        vector = _two_wire_batch(1)[0]
        single = fql.run(IDENTITY_TWO_WIRE, vector, TWO_WIRE_TIMES, n_qubits=2)
        assert tuple(single.populations.shape) == (4, 4)
        square = fql.run(
            IDENTITY_TWO_WIRE,
            torch.outer(vector, vector.conj()),
            TWO_WIRE_TIMES,
            n_qubits=2,
        )
        assert tuple(square.populations.shape) == (4, 4)

    def test_the_wire_count_is_inferred_from_a_batched_state(self) -> None:
        batch = _two_wire_batch()
        matrices = torch.stack(
            [torch.outer(vector, vector.conj()) for vector in batch]
        )
        from_vectors = fql.plan(IDENTITY_TWO_WIRE, batch, TWO_WIRE_TIMES)
        from_matrices = fql.plan(IDENTITY_TWO_WIRE, matrices, TWO_WIRE_TIMES)
        assert from_vectors.n_qubits == 2
        assert from_vectors.batch_size == 3
        assert from_vectors.trajectory_bytes == 3072
        assert from_matrices.n_qubits == 2
        assert from_matrices.batch_size == 3

    def test_a_member_that_is_not_a_physical_state_is_refused(self) -> None:
        matrices = torch.stack(
            [torch.outer(vector, vector.conj()) for vector in _two_wire_batch()]
        )
        negative = matrices.clone()
        negative[1] = torch.diag(torch.tensor([1.2, -0.2, 0.0, 0.0], dtype=DTYPE))
        with pytest.raises(EvolutionValidationError) as negative_error:
            plan_density_matrix_evolution(
                IDENTITY_TWO_WIRE, negative, 2, TWO_WIRE_TIMES
            )
        assert negative_error.value.code == "non_positive_initial_state"

        unnormalized = matrices.clone()
        unnormalized[2] = 0.5 * unnormalized[2]
        with pytest.raises(EvolutionValidationError) as trace_error:
            plan_density_matrix_evolution(
                IDENTITY_TWO_WIRE, unnormalized, 2, TWO_WIRE_TIMES
            )
        assert trace_error.value.code == "unnormalized_initial_state"


class TestBatchDeclaration:
    """A declared batch is cross-checked against the state it must describe."""

    def test_a_declared_batch_reaches_the_plan(self) -> None:
        plan = plan_density_matrix_evolution(
            IDENTITY_TWO_WIRE,
            _two_wire_batch(),
            2,
            TWO_WIRE_TIMES,
            batch_size=3,
        )
        assert plan.batch_size == 3
        payload = plan.to_dict()
        assert payload["batch_size"] == 3
        assert payload["trajectory_bytes"] == 3 * 4 * 4 * 4 * 2
        assert payload["numerics"]["require_gradients"] is False

    def test_an_undeclared_batch_adopts_the_one_the_state_carries(self) -> None:
        plan = plan_density_matrix_evolution(
            IDENTITY_TWO_WIRE, _two_wire_batch(), 2, TWO_WIRE_TIMES
        )
        assert plan.batch_size == 3
        single = plan_density_matrix_evolution(
            IDENTITY_TWO_WIRE, _two_wire_batch(1)[0], 2, TWO_WIRE_TIMES
        )
        assert single.batch_size == 1

    def test_a_declared_batch_that_contradicts_the_state_is_refused(self) -> None:
        with pytest.raises(EvolutionValidationError) as error:
            plan_density_matrix_evolution(
                IDENTITY_TWO_WIRE,
                _two_wire_batch(),
                2,
                TWO_WIRE_TIMES,
                batch_size=2,
            )
        assert error.value.code == "batch_mismatch"
        assert "batch_size=2 but initial_state carries 3" in str(error.value)

    @pytest.mark.parametrize("value", [0, -1, True, 2.0, "3"])
    def test_a_batch_that_is_not_a_positive_integer_is_refused(
        self, value: object
    ) -> None:
        # A boolean is an ``int`` in Python, so it is refused explicitly rather
        # than read as a batch of one or zero members.
        with pytest.raises(EvolutionValidationError) as error:
            plan_density_matrix_evolution(
                IDENTITY_TWO_WIRE,
                _two_wire_batch(),
                2,
                TWO_WIRE_TIMES,
                batch_size=value,  # type: ignore[arg-type]
            )
        assert error.value.code in {"invalid_batch_size", "batch_mismatch"}

    @pytest.mark.parametrize("value", [1, 0, "yes", None])
    def test_a_require_gradients_flag_that_is_not_a_boolean_is_refused(
        self, value: object
    ) -> None:
        with pytest.raises(EvolutionValidationError) as error:
            plan_density_matrix_evolution(
                IDENTITY_TWO_WIRE,
                _two_wire_batch(),
                2,
                TWO_WIRE_TIMES,
                require_gradients=value,  # type: ignore[arg-type]
            )
        assert error.value.code == "invalid_require_gradients"

    def test_the_options_layer_reports_the_same_refusals(self) -> None:
        with pytest.raises(Exception) as batch_error:
            fq.ExecutionOptions(batch_size=0)
        assert "batch_size must be >= 1" in str(batch_error.value)
        with pytest.raises(TypeError) as boolean_error:
            fq.ExecutionOptions(batch_size=True)
        assert "batch_size must be an integer or None" in str(boolean_error.value)
        with pytest.raises(TypeError) as flag_error:
            fq.ExecutionOptions(require_gradients=1)
        assert "require_gradients must be a bool or None" in str(flag_error.value)

    def test_the_options_this_domain_cannot_honour_are_still_refused(self) -> None:
        for options, name in (
            (fq.ExecutionOptions(backend="qpp-cpu"), "backend"),
            (fq.ExecutionOptions(shots=100), "shots"),
            (fq.ExecutionOptions(seed=7), "seed"),
            (fq.ExecutionOptions(allow_approximate=True), "allow_approximate"),
        ):
            with pytest.raises(fq.errors.CapabilityError) as error:
                fql.plan(
                    IDENTITY_TWO_WIRE,
                    _two_wire_batch(),
                    TWO_WIRE_TIMES,
                    n_qubits=2,
                    options=options,
                )
            assert f"does not support options: {name}" in str(error.value)


class TestIntegratorGradients:
    """Every integrator keeps a live autograd graph and a correct derivative."""

    def test_every_integrator_returns_a_trajectory_with_a_graph(self) -> None:
        for method, solve_tolerance in (
            ("runge-kutta", None),
            ("crank-nicolson", 1e-12),
            ("krylov-exponential", 1e-12),
        ):
            scale = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
            result = evolve_density_matrix(
                scale.to(DTYPE) * X0,
                _two_wire_batch(),
                2,
                TWO_WIRE_TIMES,
                AMPLITUDE_DAMPING,
                method=method,
                solve_tolerance=solve_tolerance,
                require_gradients=True,
            )
            assert result.populations.requires_grad, method

    @pytest.mark.parametrize(
        ("method", "solve_tolerance", "measured"),
        [
            ("runge-kutta", None, -5.812525050626545e-01),
            ("crank-nicolson", 1e-12, -5.790840534931964e-01),
            ("krylov-exponential", 1e-12, -5.812565894535635e-01),
        ],
    )
    def test_the_batched_gradient_matches_a_five_point_stencil(
        self, method: str, solve_tolerance: float | None, measured: float
    ) -> None:
        base = 0.6
        scale = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        _two_wire_run(
            scale, method=method, solve_tolerance=solve_tolerance, state=_two_wire_batch()
        )[-1, :, 0].sum().backward()
        autograd = float(scale.grad)
        sampled = [
            float(
                _two_wire_run(
                    torch.tensor(base + delta, dtype=torch.float64),
                    method=method,
                    solve_tolerance=solve_tolerance,
                    state=_two_wire_batch(),
                )[-1, :, 0].sum()
            )
            for delta in (2 * STENCIL_STEP, STENCIL_STEP, -STENCIL_STEP, -2 * STENCIL_STEP)
        ]
        stencil = _five_point(sampled)
        # Measured margins at step 1e-4: 2.5e-12, 4.2e-11, 7.5e-12. A gradient
        # that skipped the second substep, or that advanced only the first
        # member, lands near -2.7e-01 instead of -5.8e-01.
        assert abs(autograd - stencil) / abs(stencil) < 1e-9
        assert autograd == pytest.approx(measured, rel=1e-9)

    @pytest.mark.parametrize(
        ("method", "solve_tolerance"),
        [
            ("runge-kutta", None),
            ("crank-nicolson", 1e-12),
            ("krylov-exponential", 1e-12),
        ],
    )
    def test_the_batched_gradient_is_the_sum_of_its_members(
        self, method: str, solve_tolerance: float | None
    ) -> None:
        batch = _two_wire_batch()
        scale = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
        _two_wire_run(
            scale, method=method, solve_tolerance=solve_tolerance, state=batch
        )[-1, :, 0].sum().backward()
        batched = float(scale.grad)
        members = []
        for index in range(3):
            member_scale = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
            _two_wire_run(
                member_scale,
                method=method,
                solve_tolerance=solve_tolerance,
                state=batch[index],
            )[-1, 0].backward()
            members.append(float(member_scale.grad))
        # The members are not interchangeable: their gradients are -2.7118e-01,
        # -2.1743e-01, and -1.3704e-01, so dropping or repeating one is visible.
        assert len({round(value, 12) for value in members}) == 3
        assert batched == pytest.approx(sum(members), abs=1e-12)
        assert abs(batched - members[0]) > 1e-3

    def test_a_wider_batched_circuit_matches_its_members_and_a_stencil(self) -> None:
        batch = _wide_batch()
        hamiltonian = 0.4 * _wire_operator(PAULI_Z, 0, n_wires=4) + 0.25 * (
            _wire_operator(PAULI_Z, 0, n_wires=4)
            @ _wire_operator(PAULI_Z, 3, n_wires=4)
        )
        x1 = _wire_operator(PAULI_X, 1, n_wires=4)
        base = 0.6

        def advance(scale: float) -> torch.Tensor:
            return evolve_density_matrix(
                torch.tensor(scale, dtype=torch.float64).to(DTYPE) * x1 + hamiltonian,
                batch,
                4,
                FOUR_WIRE_TIMES,
                AMPLITUDE_DAMPING,
            ).populations

        scale = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        evolve_density_matrix(
            scale.to(DTYPE) * x1 + hamiltonian,
            batch,
            4,
            FOUR_WIRE_TIMES,
            AMPLITUDE_DAMPING,
        ).populations[-1, :, 0].sum().backward()
        autograd = float(scale.grad)
        stencil = _five_point(
            [float(advance(base + delta)[-1, :, 0].sum())
             for delta in (2 * STENCIL_STEP, STENCIL_STEP, -STENCIL_STEP, -2 * STENCIL_STEP)]
        )
        members = []
        for index in range(3):
            member_scale = torch.tensor(base, dtype=torch.float64, requires_grad=True)
            evolve_density_matrix(
                member_scale.to(DTYPE) * x1 + hamiltonian,
                batch[index],
                4,
                FOUR_WIRE_TIMES,
                AMPLITUDE_DAMPING,
            ).populations[-1, 0].backward()
            members.append(float(member_scale.grad))
        # Measured: autograd -1.847101446941476e-01, stencil rel 2.7e-12, the
        # sum of the three members within 2.8e-17. Member one of this fixture
        # has an exactly flat population, so a run that repeated a member
        # instead of advancing all three would report -9.2e-02 or -1.8e-01.
        assert abs(autograd - stencil) / abs(stencil) < 1e-9
        assert autograd == pytest.approx(sum(members), abs=1e-12)
        assert abs(autograd - members[0]) > 1e-3

    def test_a_diagonal_hamiltonian_leaves_the_populations_flat(self) -> None:
        # Observable isolation: a population cannot see a phase, so a gradient
        # assembled from populations alone must be exactly zero here. The same
        # run tracked through a density-matrix element is not zero, which is
        # what shows the measurement is capable of seeing this Hamiltonian.
        z3 = _wire_operator(PAULI_Z, 3, n_wires=4)
        fixed = 0.4 * _wire_operator(PAULI_Z, 0, n_wires=4)
        scale = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
        evolve_density_matrix(
            scale.to(DTYPE) * z3 + fixed,
            _wide_batch(),
            4,
            FOUR_WIRE_TIMES,
            AMPLITUDE_DAMPING,
        ).populations[-1, :, 0].sum().backward()
        assert float(scale.grad) == 0.0

        matrix_scale = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
        with_matrices = evolve_density_matrix(
            matrix_scale.to(DTYPE) * z3 + fixed,
            _wide_batch(),
            4,
            FOUR_WIRE_TIMES,
            AMPLITUDE_DAMPING,
            return_density_matrices=True,
        )
        assert with_matrices.density_matrices is not None
        with_matrices.density_matrices[-1, :, 0, 3].imag.sum().backward()
        assert float(matrix_scale.grad) == pytest.approx(-3.1101790907301224e-01)

    def test_require_gradients_is_decided_on_the_returned_trajectory(self) -> None:
        # A caller may parameterize the Hamiltonian and hold a fixed initial
        # state, or the other way round, so neither input can decide this.
        parameterized = torch.tensor(0.5, dtype=DTYPE, requires_grad=True)
        result = evolve_density_matrix(
            parameterized * IDENTITY_TWO_WIRE,
            _two_wire_batch(),
            2,
            TWO_WIRE_TIMES,
            require_gradients=True,
        )
        assert result.populations.requires_grad
        with pytest.raises(EvolutionValidationError) as error:
            evolve_density_matrix(
                IDENTITY_TWO_WIRE,
                _two_wire_batch(),
                2,
                TWO_WIRE_TIMES,
                require_gradients=True,
            )
        assert error.value.code == "gradients_not_available"


class TestBatchedKrylovWork:
    """A batched Krylov solve costs the same number of applies as one member."""

    @pytest.mark.parametrize(
        ("method", "solve_tolerance", "measured"),
        [("crank-nicolson", 1e-12, 12), ("krylov-exponential", 1e-12, 4)],
    )
    def test_three_members_cost_the_same_applies_as_one(
        self, method: str, solve_tolerance: float, measured: int
    ) -> None:
        hamiltonian = _normalize_hamiltonian(
            0.5 * X0 + 0.25 * Z0,
            n_wires=2,
            dim=4,
            dtype=DTYPE,
            device=DEVICE,
            tolerance=TOLERANCE,
        )
        collapse = _normalize_collapse_operators(
            AMPLITUDE_DAMPING, n_wires=2, dim=4, dtype=DTYPE, device=DEVICE
        )
        batch = _two_wire_batch()
        counts = []
        for state in (batch[0], batch):
            generator = Liouvillian(hamiltonian, collapse, hilbert_dimension=4)
            calls = 0

            def counting(vector: torch.Tensor, _derivative=generator.derivative) -> torch.Tensor:
                nonlocal calls
                calls += 1
                return _derivative(vector)

            advanced, residual = advance_interval(
                counting,
                state,
                torch.tensor(0.25, dtype=torch.float64),
                method=method,
                solve_tolerance=solve_tolerance,
                solve_max_iterations=200,
            )
            counts.append(calls)
            assert residual < 1e-12
            assert float(
                (torch.diagonal(advanced, dim1=-2, dim2=-1).sum(dim=-1).real - 1.0)
                .abs()
                .max()
            ) < 1e-12
        # One restart window covers every member, because the flattened
        # generator is block diagonal. A solve per member would report 3x the
        # measured 12 and 4 applies.
        assert counts[1] == counts[0]
        assert counts[0] == measured

    def test_the_linear_solve_gradient_follows_its_right_hand_side(self) -> None:
        matrix = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], dtype=DTYPE
        )
        right_hand_side = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
        # A x = s b has x = s A^-1 b, so the derivative of the real sum is the
        # real sum of the solution itself: a closed form, not a tolerance.
        expected = float(torch.linalg.solve(matrix, right_hand_side).real.sum())
        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        solution = gmres(
            lambda vector: matrix @ vector,
            scale.to(DTYPE) * right_hand_side,
            tolerance=1e-14,
            max_iterations=50,
            restart=10,
        )[0]
        assert float((matrix @ solution - right_hand_side).abs().max()) < 1e-14
        solution.real.sum().backward()
        autograd = float(scale.grad)
        # Measured: autograd +1.777777777777779 against the closed form
        # +1.777777777777778 and a five-point stencil +1.777777777777952. The
        # rule that reads the scaling norm into a Python float, which is what
        # this solve did before batching, reports -2.258420716105410 instead.
        stencil = _five_point(
            [
                float(
                    gmres(
                        lambda vector: matrix @ vector,
                        torch.tensor(1.0 + delta, dtype=torch.float64).to(DTYPE)
                        * right_hand_side,
                        tolerance=1e-14,
                        max_iterations=50,
                        restart=10,
                    )[0].real.sum()
                )
                for delta in (
                    2 * STENCIL_STEP,
                    STENCIL_STEP,
                    -STENCIL_STEP,
                    -2 * STENCIL_STEP,
                )
            ]
        )
        assert autograd == pytest.approx(expected, rel=1e-12)
        assert abs(autograd - stencil) < 1e-9

    def test_the_exponential_action_argument_carries_its_gradient(self) -> None:
        right_hand_side = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
        apply = lambda vector: -0.5 * vector  # noqa: E731
        expected = math.exp(-0.2) * 6.0
        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        advanced, defect = exponential_action(
            apply,
            0.4,
            scale.to(DTYPE) * right_hand_side,
            tolerance=1e-14,
            max_iterations=50,
            restart=10,
        )
        advanced.real.sum().backward()
        autograd = float(scale.grad)
        # Measured: autograd +4.91238451846789 against the closed form
        # +4.912384518467891, defect 3.2e-17. The rule that reads the opening
        # norm into a Python float reports 4.912358918711636, which is 2.6e-05
        # low: small, deterministic, and exactly the wrong answer.
        assert defect < 1e-15
        assert autograd == pytest.approx(expected, rel=1e-12)
        assert abs(autograd - expected) < 1e-3

    def test_a_batched_commutator_matches_the_dense_form(self) -> None:
        terms = [
            PauliSumTerm(coefficient=0.5, mask=1 << 1, signs=()),
            PauliSumTerm(coefficient=0.25, mask=0, signs=(1,)),
            PauliSumTerm(coefficient=0.1, mask=0b11, signs=(0,)),
        ]
        pauli = PauliSum(
            terms, n_wires=2, dimension=4, dtype=DTYPE, device=DEVICE
        )
        batch = torch.stack(
            [torch.outer(vector, vector.conj()) for vector in _two_wire_batch()]
        )
        dense = pauli.dense()
        expected = torch.stack([dense @ item - item @ dense for item in batch])
        commuted = pauli.commutator(batch)
        assert commuted.shape == (3, 4, 4)
        assert float((commuted - expected).abs().max()) < 1e-15
        single = pauli.commutator(batch[0])
        assert single.shape == (4, 4)
        assert float((single - (dense @ batch[0] - batch[0] @ dense)).abs().max()) < 1e-15


class TestSealedPlanBatchAndGradients:
    """A sealed plan records the batch it counts and the graph it must keep."""

    def test_the_sealed_decision_records_both_new_fields(self) -> None:
        plan = fql.plan(
            IDENTITY_TWO_WIRE,
            _two_wire_batch(),
            TWO_WIRE_TIMES,
            n_qubits=2,
            options=fq.ExecutionOptions(batch_size=3, require_gradients=True),
            method="crank-nicolson",
        )
        payload = plan.to_dict()
        assert payload["version"] == "1.3"
        assert payload["batch_size"] == 3
        assert payload["numerics"]["require_gradients"] is True
        assert plan.batch_size == 3
        assert plan.require_gradients is True
        assert plan.trajectory_bytes == 3 * 4 * 4 * 4 * 2

    def test_a_live_plan_keeps_the_matrix_free_graph(self) -> None:
        coefficient = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        live = fql.plan(
            coefficient.to(DTYPE) * X0,
            _two_wire_batch(),
            TWO_WIRE_TIMES,
            n_qubits=2,
            options=fq.ExecutionOptions(batch_size=3, require_gradients=True),
        )
        result = fql.run(live)
        assert result.populations.requires_grad
        result.populations[-1, :, 0].sum().backward()
        # Measured -4.3149110071529656e-01. A plan that stored its request as
        # JSON and read it back would fail the assertion rather than return a
        # detached number, because require_gradients was sealed as true.
        assert float(coefficient.grad) == pytest.approx(-4.3149110071529656e-01)

    def test_a_restored_plan_refuses_a_gradient_request_it_cannot_meet(self) -> None:
        coefficient = torch.tensor(0.4, dtype=torch.float64)
        live = fql.plan(
            coefficient.to(DTYPE) * X0,
            _two_wire_batch(),
            TWO_WIRE_TIMES,
            n_qubits=2,
            options=fq.ExecutionOptions(batch_size=3, require_gradients=True),
        )
        restored = fql.LindbladPlan.from_json(live.to_json())
        assert restored.require_gradients is True
        assert restored.identity == live.identity
        assert restored.batch_size == 3
        with pytest.raises(EvolutionValidationError) as error:
            fql.run(restored)
        assert error.value.code == "gradients_not_available"
        assert error.value.field == "require_gradients"

    def test_a_restored_plan_without_the_request_still_runs(self) -> None:
        restored = fql.LindbladPlan.from_json(
            fql.plan(
                IDENTITY_TWO_WIRE,
                _two_wire_batch(),
                TWO_WIRE_TIMES,
                n_qubits=2,
                options=fq.ExecutionOptions(batch_size=3),
            ).to_json()
        )
        result = fql.run(restored)
        assert result.populations.shape == (4, 3, 4)
        assert not result.populations.requires_grad

    def test_a_batched_evolution_plan_reports_the_problem_dimension(self) -> None:
        plan = plan_density_matrix_evolution(
            IDENTITY_TWO_WIRE,
            _two_wire_batch(),
            2,
            TWO_WIRE_TIMES,
            batch_size=3,
        )
        assert plan.dimension == 4
        assert plan.n_times == 4
        assert plan.trajectory_bytes == 3072
        assert plan.require_gradients is False
