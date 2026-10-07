"""Weighted-MaxCut QAOA as one solver: cost operator, ansatz, run, and refusals.

The assertions here are of two kinds, and the difference is deliberate.  Where a
quantity is a *claim* -- that the cost operator minimizes at the largest cuts,
that the QUBO form describes the same optimum, that the uniform superposition is
a stationary point, that the two most probable sampled bitstrings are the maximum
cuts -- the assertion is a candidate-set identity or a strict inequality, so it
fails when the claim is weakened and not merely when a number moves.  Where a
quantity is the *evidence* for one run, it is pinned to the value this repository
measured, so a behaviour change is visible rather than absorbed.

What is not asserted here: that QAOA at a finite layer count reaches the optimum,
that a larger layer count is better in general, that the optimizer's trajectory
decreases, or that a reported energy is a bound.  The measured counterexamples to
the second and third are the subject of their own tests rather than being assumed
away.

Nor is a gradient-free run asserted to converge in the sense a convergence flag
means.  The simplex search reaches this instance's optimum to 3.4e-05 in the
angles and still reports converged=False, because that flag is about the simplex
collapsing and not about the energy arriving, so the assertions on that route are
about the energy and the flag is read as it is rather than relaxed.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import (
    QAOA_ASSUMPTIONS,
    QAOA_LIMITATIONS,
    SPSA_ASSUMPTIONS,
    SPSA_LIMITATIONS,
    Hamiltonian,
    NelderMeadOptimizer,
    QAOAResult,
    SPSAOptimizer,
    SPSAResult,
    maxcut_hamiltonian,
    maxcut_objective,
    qaoa_circuit,
    qaoa_loss,
    run_qaoa,
)
from flagquantum.algorithms.qubo import max_cut_qubo, qubo_to_ising
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit

TRIANGLE = ((0, 1), (1, 2), (2, 0))
RING_WITH_CHORD = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))

#: A probability above which a sampled bitstring counts as one the optimized
#: circuit actually produces, as opposed to one it leaves at noise level.
_HEAVY_THRESHOLD = 0.1


def _weighted(edges):
    """Return the edges as ``(source, target, weight)`` triples."""

    return tuple(
        (edge[0], edge[1], 1.0 if len(edge) == 2 else float(edge[2])) for edge in edges
    )


def _cuts(n_qubits, edges):
    """Return ``{bits: weighted cut size}`` by enumerating every assignment."""

    weighted = _weighted(edges)
    return {
        bits: sum(
            weight
            for source, target, weight in weighted
            if bits[source] != bits[target]
        )
        for bits in itertools.product((0, 1), repeat=n_qubits)
    }


def _basis_circuit(n_qubits, bits):
    """Return the circuit preparing one computational basis state."""

    circuit = fq.Circuit(n_qubits)
    for wire, bit in enumerate(bits):
        if bit:
            circuit = circuit.x(wire)
    return circuit


def _basis_energies(n_qubits, hamiltonian):
    """Return ``{bits: expectation}`` for every basis state of ``n_qubits``."""

    return {
        bits: float(hamiltonian.expectation(_basis_circuit(n_qubits, bits)))
        for bits in itertools.product((0, 1), repeat=n_qubits)
    }


def _sampled_probabilities(n_qubits, edges, gammas, betas):
    """Return ``{bits: probability}`` for the circuit the angles describe."""

    state = qaoa_circuit(n_qubits, edges, gammas, betas).state().reshape(-1)
    probabilities = (state.abs() ** 2).real
    return {
        tuple((int(index) >> (n_qubits - 1 - wire)) & 1 for wire in range(n_qubits)): (
            float(probabilities[index])
        )
        for index in range(probabilities.numel())
    }


def _argmin(mapping):
    """Return the keys whose value is the minimum, as a set."""

    best = min(mapping.values())
    return {key for key, value in mapping.items() if value == best}


def _argmax(mapping):
    """Return the keys whose value is the maximum, as a set."""

    best = max(mapping.values())
    return {key for key, value in mapping.items() if value == best}


# --- the cost operator -------------------------------------------------------


def test_the_cost_operator_carries_one_zz_term_per_edge_in_edge_order() -> None:
    hamiltonian = maxcut_hamiltonian(3, TRIANGLE)
    assert [term.pauli for term in hamiltonian.terms] == ["ZZ", "ZZ", "ZZ"]
    assert [term.wires for term in hamiltonian.terms] == [(0, 1), (1, 2), (2, 0)]


def test_the_cost_operator_keeps_a_weight_and_spans_the_graph_wires() -> None:
    hamiltonian = maxcut_hamiltonian(4, ((0, 3, 0.25),))
    assert [float(term.coefficient) for term in hamiltonian.terms] == [0.25]
    assert hamiltonian.n_wires == 4


def test_a_pair_and_a_triple_may_be_mixed_in_one_edge_list() -> None:
    hamiltonian = maxcut_hamiltonian(3, ((0, 1, 0.5), (1, 2)))
    assert [float(term.coefficient) for term in hamiltonian.terms] == [0.5, 1.0]


def test_the_cost_operator_is_an_ordinary_hamiltonian() -> None:
    assert isinstance(maxcut_hamiltonian(3, TRIANGLE), Hamiltonian)


# --- what the operator minimizes --------------------------------------------


def test_the_operator_value_on_every_basis_state_is_the_total_less_twice_the_cut() -> (
    None
):
    hamiltonian = maxcut_hamiltonian(4, RING_WITH_CHORD)
    cuts = _cuts(4, RING_WITH_CHORD)
    total = sum(weight for _, _, weight in _weighted(RING_WITH_CHORD))
    energies = _basis_energies(4, hamiltonian)
    assert set(energies) == set(cuts)
    for bits, cut in cuts.items():
        assert energies[bits] == pytest.approx(total - 2 * cut, abs=1e-9)


def test_the_operator_minimizes_on_exactly_the_set_of_largest_cuts() -> None:
    for n_qubits, edges in ((3, TRIANGLE), (4, RING_WITH_CHORD)):
        hamiltonian = maxcut_hamiltonian(n_qubits, edges)
        assert _argmin(_basis_energies(n_qubits, hamiltonian)) == _argmax(
            _cuts(n_qubits, edges)
        )


def test_the_ground_energy_is_the_total_less_twice_the_largest_cut() -> None:
    for n_qubits, edges in ((3, TRIANGLE), (4, RING_WITH_CHORD)):
        total = sum(weight for _, _, weight in _weighted(edges))
        largest = max(_cuts(n_qubits, edges).values())
        assert float(
            maxcut_hamiltonian(n_qubits, edges).ground_energy()
        ) == pytest.approx(total - 2 * largest, abs=1e-9)


def test_weighting_the_edges_moves_the_optimum_away_from_the_unweighted_one() -> None:
    weighted = tuple(
        (source, target, 0.5 + 0.25 * index)
        for index, (source, target) in enumerate(RING_WITH_CHORD)
    )
    plain = maxcut_hamiltonian(4, RING_WITH_CHORD)
    tilted = maxcut_hamiltonian(4, weighted)
    assert float(tilted.ground_energy()) != float(plain.ground_energy())
    assert _argmin(_basis_energies(4, tilted)) != _argmin(_basis_energies(4, plain))


# --- the relation to the QUBO form ------------------------------------------


def test_the_qubo_form_is_an_affine_image_of_the_cost_operator() -> None:
    for n_qubits, edges in ((3, TRIANGLE), (4, RING_WITH_CHORD)):
        hamiltonian = maxcut_hamiltonian(n_qubits, edges)
        qubo = qubo_to_ising(max_cut_qubo(edges, n_nodes=n_qubits))
        edge_count = len(edges)
        for bits, energy in _basis_energies(n_qubits, hamiltonian).items():
            value = float(qubo.expectation(_basis_circuit(n_qubits, bits)))
            assert value == pytest.approx((energy - edge_count) / 2, abs=1e-9)


def test_the_qubo_form_and_the_cost_operator_share_their_argmin_set() -> None:
    for n_qubits, edges in ((3, TRIANGLE), (4, RING_WITH_CHORD)):
        hamiltonian = maxcut_hamiltonian(n_qubits, edges)
        qubo = qubo_to_ising(max_cut_qubo(edges, n_nodes=n_qubits))
        assert _argmin(_basis_energies(n_qubits, qubo)) == _argmin(
            _basis_energies(n_qubits, hamiltonian)
        )


# --- the stationary point ----------------------------------------------------


@pytest.mark.parametrize("layers", [1, 2, 3, 4])
def test_the_uniform_superposition_is_a_stationary_point_at_every_layer_count(
    layers: int,
) -> None:
    hamiltonian = maxcut_hamiltonian(3, TRIANGLE)
    parameters = torch.zeros(2 * layers, requires_grad=True)
    loss = qaoa_loss(3, TRIANGLE, parameters[:layers], parameters[layers:], hamiltonian)
    torch.autograd.backward(loss)
    assert float(loss.detach()) == 0.0
    assert parameters.grad is not None
    assert torch.equal(parameters.grad, torch.zeros(2 * layers))


@pytest.mark.parametrize("layers", [1, 2, 3])
@pytest.mark.parametrize("steps", [1, 3, 25])
def test_a_zero_start_returns_the_zero_start_for_every_step_count(
    layers: int, steps: int
) -> None:
    result = run_qaoa(3, TRIANGLE, torch.zeros(2 * layers), steps=steps)
    assert result.history == (0.0,) * steps
    assert float(result.energy) == 0.0
    assert torch.equal(result.parameters, torch.zeros(2 * layers))


def test_a_nonzero_start_leaves_the_saddle_that_the_zero_start_sits_on() -> None:
    # The saddle run cannot move at all: every gradient is zero, so what it
    # returns is what it was given. A displaced start reaches a lower value, which
    # is what makes the displacement the caller's job rather than a detail.
    saddle = run_qaoa(3, TRIANGLE, torch.zeros(2))
    moved = run_qaoa(3, TRIANGLE, torch.full((2,), 0.05))
    assert torch.equal(saddle.parameters, torch.zeros(2))
    assert float(saddle.energy) == 0.0
    assert float(moved.energy) < 0.0


# --- convergence, and what it does not establish ----------------------------


def test_one_layer_reaches_the_triangle_optimum_from_a_small_nonzero_start() -> None:
    ground = float(maxcut_hamiltonian(3, TRIANGLE).ground_energy())
    result = run_qaoa(3, TRIANGLE, torch.full((2,), 0.05))
    assert float(result.energy) == pytest.approx(-0.999333381652832, abs=1e-12)
    assert result.history[0] == pytest.approx(0.05950213968753815, abs=1e-12)
    assert ground == -1.0
    assert float(result.energy) > ground
    assert float(result.energy) - ground < 1e-3


def test_a_single_layer_cannot_reach_the_ring_optimum_that_three_layers_approach() -> (
    None
):
    ground = float(maxcut_hamiltonian(4, RING_WITH_CHORD).ground_energy())
    shallow = run_qaoa(4, RING_WITH_CHORD, torch.full((2,), 0.1))
    deep = run_qaoa(4, RING_WITH_CHORD, torch.full((6,), 0.1))
    assert float(shallow.energy) == pytest.approx(-1.4735270738601685, abs=1e-12)
    assert float(deep.energy) == pytest.approx(-2.9724555015563965, abs=1e-12)
    assert ground == -3.0
    # Both are above the optimum, and the deeper ansatz is strictly closer. That
    # is a statement about this instance and this start, not a general law.
    assert float(shallow.energy) > float(deep.energy) > ground
    assert float(deep.energy) - ground < float(shallow.energy) - ground


def test_the_trajectory_is_a_record_and_not_a_descent() -> None:
    history = run_qaoa(3, TRIANGLE, torch.full((2,), 0.05)).history
    assert history[0] == pytest.approx(0.05950213968753815, abs=1e-12)
    assert not all(later <= earlier for earlier, later in itertools.pairwise(history))
    assert history[-1] < history[0]


def test_the_reported_energy_is_the_objective_at_the_returned_parameters() -> None:
    for n_qubits, edges, start in (
        (3, TRIANGLE, torch.full((4,), 0.1)),
        (4, RING_WITH_CHORD, torch.full((2,), 0.1)),
    ):
        result = run_qaoa(n_qubits, edges, start)
        recomputed = qaoa_loss(
            n_qubits,
            edges,
            result.gammas,
            result.betas,
            maxcut_hamiltonian(n_qubits, edges),
        )
        assert float(result.energy) == pytest.approx(float(recomputed), abs=1e-12)


def test_two_identical_runs_return_bit_identical_results() -> None:
    first = run_qaoa(4, RING_WITH_CHORD, torch.full((6,), 0.1))
    second = run_qaoa(4, RING_WITH_CHORD, torch.full((6,), 0.1))
    assert torch.equal(first.parameters, second.parameters)
    assert first.history == second.history


# --- the sampled end of the path --------------------------------------------


def test_the_sampled_bitstrings_the_optimized_circuit_favours_are_the_largest_cuts() -> (
    None
):
    result = run_qaoa(4, RING_WITH_CHORD, torch.full((6,), 0.1))
    probabilities = _sampled_probabilities(
        4, RING_WITH_CHORD, result.gammas, result.betas
    )
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    heavy = {
        bits
        for bits, probability in probabilities.items()
        if probability > _HEAVY_THRESHOLD
    }
    assert heavy == _argmax(_cuts(4, RING_WITH_CHORD))
    assert len(heavy) == 2
    for bits in heavy:
        assert probabilities[bits] == pytest.approx(0.4942, abs=1e-3)


# --- the result carrier ------------------------------------------------------


def test_the_result_splits_the_flat_vector_into_gammas_then_betas() -> None:
    result = run_qaoa(3, TRIANGLE, torch.tensor([0.3, 0.7, -0.3, -0.4]))
    assert result.layers == 2
    assert torch.equal(result.gammas, result.parameters[:2])
    assert torch.equal(result.betas, result.parameters[2:])
    assert result.n_steps == 100


def test_the_result_carries_the_documented_assumptions_and_limitations() -> None:
    result = run_qaoa(3, TRIANGLE, torch.full((2,), 0.05))
    assert result.assumptions is QAOA_ASSUMPTIONS
    assert result.limitations is QAOA_LIMITATIONS
    saddle = [entry for entry in QAOA_LIMITATIONS if "stationary point" in entry]
    assert len(saddle) == 1
    assert "0.0" in saddle[0]
    assert "NelderMeadOptimizer" in " ".join(QAOA_LIMITATIONS)
    assert "SPSAOptimizer" in " ".join(QAOA_LIMITATIONS)


def test_the_gradient_free_entry_quotes_the_three_routes_it_measured() -> None:
    """A quoted number in the limitations is a claim about the arithmetic.

    The entry is located by what it says rather than by its index, and every
    number in it is then recomputed from the three runs it describes, so a
    sentence that drifts away from the code it summarizes turns red here instead
    of being read as decoration.
    """

    entries = [
        entry for entry in QAOA_LIMITATIONS if "gradient-free route is the" in entry
    ]
    assert len(entries) == 1
    text = entries[0]
    assert "maxcut_objective" in text
    assert "SPSAOptimizer.minimize" in text
    assert "NelderMeadOptimizer.minimize" in text

    start = _gradient_free_start()
    objective = maxcut_objective(3, TRIANGLE)
    adam = float(run_qaoa(3, TRIANGLE, start, steps=300).energy)
    spsa = _seeded_spsa().minimize(objective, start)
    simplex = NelderMeadOptimizer(maxiter=200).minimize(objective, start)
    for measured in (
        f"{adam:.7f}",
        f"{spsa.value:.7f}",
        f"{simplex.value:.7f}",
        str(spsa.evaluations),
        str(simplex.evaluations),
    ):
        assert measured in text, measured


def test_the_result_owns_its_parameters_and_the_input_keeps_its_identity() -> None:
    start = torch.full((2,), 0.05)
    result = run_qaoa(3, TRIANGLE, start)
    assert result.parameters.data_ptr() != start.data_ptr()
    assert not result.parameters.requires_grad
    assert torch.equal(start, torch.full((2,), 0.05))
    assert result.parameters.dtype == torch.float32


def test_a_float64_start_is_downcast_rather_than_refused() -> None:
    result = run_qaoa(3, TRIANGLE, torch.full((2,), 0.05, dtype=torch.float64), steps=3)
    assert result.parameters.dtype == torch.float32
    assert result.energy.dtype == torch.float32


def test_the_optimizer_factory_receives_the_owned_parameter_and_the_rate() -> None:
    seen: list[tuple[tuple[torch.Tensor, ...], float]] = []

    def factory(params, *, lr):
        parameters = tuple(params)
        seen.append((parameters, lr))
        return torch.optim.SGD(parameters, lr=lr)

    start = torch.full((4,), 0.1)
    run_qaoa(3, TRIANGLE, start, steps=5, lr=0.25, optimizer_factory=factory)
    assert len(seen) == 1
    parameters, lr = seen[0]
    assert lr == 0.25
    assert len(parameters) == 1
    assert parameters[0].data_ptr() != start.data_ptr()
    assert parameters[0].is_leaf and parameters[0].requires_grad


@pytest.mark.parametrize("optimizer_factory", [torch.optim.Adam, torch.optim.SGD])
def test_the_solver_accepts_the_optimizers_that_expose_a_parameter_group(
    optimizer_factory,
) -> None:
    result = run_qaoa(
        3,
        TRIANGLE,
        torch.full((2,), 0.05),
        steps=40,
        optimizer_factory=optimizer_factory,
    )
    assert float(result.energy) < result.history[0]


def test_a_gradient_free_optimizer_cannot_be_supplied_as_the_optimizer_factory() -> (
    None
):
    for gradient_free in (SPSAOptimizer, NelderMeadOptimizer):
        with pytest.raises(TypeError):
            run_qaoa(
                3,
                TRIANGLE,
                torch.full((2,), 0.05),
                steps=1,
                optimizer_factory=gradient_free,  # type: ignore[arg-type]
            )


# --- the gradient-free route over the same objective -------------------------
#
# The two units this route reaches are SPSA and Nelder-Mead, which the package
# already shipped and which no solver entry point could reach: an SPSA step is a
# rate-limited update from two objective values, so it is not a torch.optim
# optimizer, and a simplex is a whole run released at once, so it is not one
# either. The route is the objective, not a second optimizer parameter.


def _gradient_free_start() -> torch.Tensor:
    """The one-layer start every route below is compared from."""

    return torch.full((2,), 0.05)


def _seeded_spsa() -> SPSAOptimizer:
    """The SPSA optimizer every replay claim below is made about."""

    return SPSAOptimizer(
        maxiter=400,
        perturbation=0.05,
        generator=torch.Generator().manual_seed(11),
    )


def test_the_cost_objective_is_the_objective_the_solver_minimizes() -> None:
    objective = maxcut_objective(3, TRIANGLE)
    start = _gradient_free_start()
    result = run_qaoa(3, TRIANGLE, start, steps=300)
    assert float(objective(result.parameters.detach())) == float(result.energy)
    assert float(objective(start)) == result.history[0]


def test_the_cost_objective_splits_the_vector_as_gammas_then_betas() -> None:
    """The split convention is asserted against the interleaved one it is not.

    ``torch.stack([gammas, betas], dim=1).reshape(-1)`` holds the same four
    numbers in the other plausible order, so an implementation that read the
    halves by alternating index would match that vector instead of this one. Two
    layers are used because at one layer both orders are the same vector.
    """

    gammas = torch.tensor([0.3, 0.7])
    betas = torch.tensor([-0.5, 0.2])
    objective = maxcut_objective(3, TRIANGLE)
    stacked = torch.cat([gammas, betas])
    interleaved = torch.stack([gammas, betas], dim=1).reshape(-1)
    assert float(objective(stacked)) == pytest.approx(0.4985719323158264, abs=1e-12)
    assert float(objective(stacked)) == float(
        qaoa_loss(3, TRIANGLE, gammas, betas, maxcut_hamiltonian(3, TRIANGLE))
    )
    assert float(objective(interleaved)) != pytest.approx(
        float(objective(stacked)), abs=1e-6
    )


def test_the_cost_objective_serves_both_routes_without_a_mode_switch() -> None:
    """The same callable is a differentiable objective and a plain value.

    A gradient-free step differences two calls and never asks for a backward
    pass; the autograd route hands the same callable a leaf that requires grad.
    Both are asserted here against one another, because an objective that only
    worked in one of the two modes would make the gradient-free route a second
    implementation of the cost rather than the same one.
    """

    objective = maxcut_objective(3, TRIANGLE)
    start = _gradient_free_start()
    detached = float(objective(start))
    assert detached == pytest.approx(0.05950213968753815, abs=1e-12)
    with torch.no_grad():
        assert float(objective(start)) == detached
    graded = start.clone().requires_grad_(True)
    graded_value = objective(graded)
    assert graded_value.requires_grad
    assert float(graded_value.detach()) == detached
    torch.autograd.backward(graded_value)
    assert graded.grad is not None
    assert graded.grad.shape == start.shape


def test_the_cost_objective_reaches_the_solver_through_the_same_edge_checks() -> None:
    for edges, message in (
        (((0, 0), (1, 2)), "self-loop"),
        (((0, 1), (1, 0)), "repeated"),
        ((), "at least one edge"),
    ):
        with pytest.raises(ValueError, match=message):
            maxcut_objective(3, edges)


def test_spsa_reaches_the_triangle_optimum_through_the_cost_objective() -> None:
    start = _gradient_free_start()
    optimizer = _seeded_spsa()
    result = optimizer.minimize(maxcut_objective(3, TRIANGLE), start)
    adam = float(run_qaoa(3, TRIANGLE, start, steps=300).energy)
    assert result.value == pytest.approx(-0.9999911189079285, abs=1e-9)
    assert result.value < result.history[0]
    assert result.value - adam < 1e-3
    assert result.steps == 400
    assert result.evaluations == 3 * 400 + 1
    assert len(result.history) == result.steps


def test_the_spsa_cost_sequence_is_a_record_and_not_a_descent() -> None:
    start = _gradient_free_start()
    result = _seeded_spsa().minimize(maxcut_objective(3, TRIANGLE), start)
    rises = [
        later - earlier
        for earlier, later in itertools.pairwise(result.history)
        if later > earlier
    ]
    assert len(rises) == 140
    assert max(rises) == pytest.approx(0.18285489082336426, abs=1e-9)
    assert result.history[-1] < result.history[0]
    assert min(result.history) < result.value


def test_nelder_mead_reaches_the_triangle_optimum_through_the_same_objective() -> None:
    start = _gradient_free_start()
    adam = float(run_qaoa(3, TRIANGLE, start, steps=300).energy)
    result = NelderMeadOptimizer(maxiter=200).minimize(
        maxcut_objective(3, TRIANGLE), start
    )
    assert result.value == pytest.approx(-1.0000003576278687, abs=1e-9)
    assert (
        result.parameters - torch.tensor([0.3077400028705597, -0.3077400028705597])
    ).abs().max() < 1e-4
    assert result.evaluations == 706
    assert abs(result.value - adam) < 1e-4
    # The energy is at the optimum and the simplex has not collapsed. Reading
    # the flag as `True` here would be asserting a convergence this run did not
    # report, so the measured `False` is what is pinned.
    assert result.converged is False
    assert result.value_spread > result.atol


def test_both_gradient_free_routes_land_where_the_autograd_route_lands() -> None:
    start = _gradient_free_start()
    objective = maxcut_objective(3, TRIANGLE)
    adam = run_qaoa(3, TRIANGLE, start, steps=300).parameters
    spsa = _seeded_spsa().minimize(objective, start)
    simplex = NelderMeadOptimizer(maxiter=200).minimize(objective, start)
    assert (spsa.parameters - adam).abs().max() < 1e-3
    assert (simplex.parameters - adam).abs().max() < 1e-4
    assert float(objective(spsa.parameters)) == pytest.approx(spsa.value, abs=1e-12)
    assert float(objective(simplex.parameters)) == pytest.approx(
        simplex.value, abs=1e-12
    )


def test_a_gradient_free_run_replays_from_a_fresh_optimizer_with_the_same_seed() -> (
    None
):
    objective = maxcut_objective(3, TRIANGLE)
    start = _gradient_free_start()
    first = _seeded_spsa().minimize(objective, start)
    second = _seeded_spsa().minimize(objective, start)
    assert torch.equal(first.parameters, second.parameters)
    assert first.value == second.value
    assert first.history == second.history


def test_the_spsa_recursion_continues_rather_than_restarting_on_a_second_call() -> None:
    """A call continues the sequences; only a fresh optimizer replays them.

    The step size is indexed by the optimizer's step counter, so a second call on
    the same object starts from a later, smaller step. Pinning the counter is how
    this is asserted: a restart would report ``steps == 400`` again, and the
    trajectory a restart would produce is the one the replay test above owns.
    """

    objective = maxcut_objective(3, TRIANGLE)
    start = _gradient_free_start()
    optimizer = _seeded_spsa()
    first = optimizer.minimize(objective, start)
    second = optimizer.minimize(objective, start)
    assert optimizer.steps == 800
    assert optimizer.evaluations == 2 * 1201
    assert second.steps == 400
    assert second.evaluations == 1201
    assert first.history != second.history
    assert second.value != first.value


def test_a_spsa_run_without_a_budget_is_refused_by_naming_the_budget() -> None:
    with pytest.raises(ValidationError, match="needs a step budget"):
        SPSAOptimizer(stability=5.0).minimize(
            maxcut_objective(3, TRIANGLE), _gradient_free_start()
        )


@pytest.mark.parametrize("steps", [0, -1, True, 1.5, "10"])
def test_a_spsa_step_budget_that_is_not_a_positive_whole_number_is_refused(
    steps,
) -> None:
    optimizer = SPSAOptimizer(maxiter=10)
    with pytest.raises(ValidationError, match="steps must be a positive integer"):
        optimizer.minimize(
            maxcut_objective(3, TRIANGLE), _gradient_free_start(), steps=steps
        )


def test_an_absent_step_budget_falls_back_to_the_maxiter_it_was_built_with() -> None:
    optimizer = SPSAOptimizer(maxiter=10)
    result = optimizer.minimize(maxcut_objective(3, TRIANGLE), _gradient_free_start())
    assert result.steps == 10
    assert result.evaluations == 3 * 10 + 1


def test_the_cost_objective_refuses_an_angle_vector_a_run_could_not_describe() -> None:
    objective = maxcut_objective(3, TRIANGLE)
    refusals = (
        (torch.full((2, 2), 0.1), "one flat vector"),
        ([[0.1, 0.2], [0.3, 0.4]], "one flat vector"),
        (torch.full((3,), 0.1), "must be even"),
        (torch.full((0,), 0.1), "none were given"),
    )
    for angles, message in refusals:
        with pytest.raises(ValueError, match=message):
            objective(angles)


def test_the_spsa_result_refuses_a_report_that_contradicts_its_run() -> None:
    described = {
        "parameters": torch.zeros(2),
        "value": 1.0,
        "history": (1.0,),
        "evaluations": 4,
        "steps": 1,
    }
    result = SPSAResult(**described)
    assert result.evaluations == 3 * result.steps + 1
    assert result.assumptions is SPSA_ASSUMPTIONS
    assert result.limitations is SPSA_LIMITATIONS

    contradictions = (
        ({"evaluations": 3}, "spends two evaluations"),
        ({"history": (1.0, 2.0)}, "one cost per update"),
        ({"value": float("nan")}, "value must be finite"),
        ({"value": float("inf")}, "value must be finite"),
        ({"parameters": torch.zeros(0)}, "at least one value"),
        ({"parameters": [0.0, 0.0]}, "must be the tensor"),
        ({"history": (float("inf"),)}, "finite costs"),
        ({"steps": 0}, "steps must be a positive integer"),
        ({"steps": True}, "steps must be a positive integer"),
    )
    for override, message in contradictions:
        with pytest.raises((ValueError, TypeError), match=message):
            SPSAResult(**{**described, **override})


def test_the_solver_is_reachable_from_the_package_namespace() -> None:
    from flagquantum import algorithms as fqa

    assert fqa.run_qaoa is run_qaoa
    assert fqa.maxcut_hamiltonian is maxcut_hamiltonian
    assert "run_qaoa" in fqa.__all__


# --- refusals ---------------------------------------------------------------


@pytest.mark.parametrize("n_qubits", [0, -1, 3.0, True, "3", None])
def test_a_qubit_count_that_is_not_a_positive_integer_is_refused(n_qubits) -> None:
    with pytest.raises(ValueError, match="n_qubits must be a positive integer"):
        maxcut_hamiltonian(n_qubits, TRIANGLE)


@pytest.mark.parametrize(
    "edges",
    [
        ((0, 1, 2.0, 4.0),),
        ((0, 1.0),),
        (("a", 1),),
        ((1, 1),),
        ((0, 1, "heavy"),),
        ((0, 1, float("inf")),),
        ((0, 1, float("nan")),),
        ((0, 1, True),),
        ((0, 1), (1, 0)),
        ((0, 1), (0, 1)),
        (),
    ],
)
def test_an_edge_list_a_cost_layer_could_not_implement_is_refused(edges) -> None:
    with pytest.raises(ValueError):
        maxcut_hamiltonian(3, edges)


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        (((0, 4),), "names qubit 4, and the graph has 3 qubits"),
        (((0, 1, 2.0, 4.0),), "has 4 entries"),
        (((0, 1.0),), "must name qubit indices"),
        (((1, 1),), "joins qubit 1 to itself"),
        (((0, 1, "heavy"),), "weight must be a real number"),
        (((0, 1, float("inf")),), "weight must be finite"),
        (((0, 1), (1, 0)), "the same undirected pair"),
        ((), "needs at least one edge"),
    ],
)
def test_each_edge_refusal_names_what_a_cost_layer_would_have_mishandled(
    edges, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        maxcut_hamiltonian(3, edges)


@pytest.mark.parametrize("steps", [0, -1, 2.5, "3", True, None])
def test_a_step_count_that_is_not_a_positive_whole_number_is_refused(steps) -> None:
    with pytest.raises(ValueError, match="steps"):
        run_qaoa(3, TRIANGLE, torch.full((2,), 0.05), steps=steps)


def test_a_run_with_no_step_is_refused_because_the_start_is_not_an_optimum() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_qaoa(3, TRIANGLE, torch.full((2,), 0.05), steps=0)


@pytest.mark.parametrize(
    ("start", "message"),
    [
        (torch.zeros(0), "none were given"),
        (torch.zeros(3), "the count must be even"),
        (torch.zeros((2, 2)), "one flat vector"),
        (torch.zeros((3, 1, 2)), "one flat vector"),
    ],
)
def test_a_start_that_is_not_one_even_length_vector_is_refused(start, message) -> None:
    with pytest.raises(ValueError, match=message):
        run_qaoa(3, TRIANGLE, start, steps=1)


def test_a_nested_start_is_refused_rather_than_flattened() -> None:
    with pytest.raises(ValueError, match="nested sequence is refused"):
        run_qaoa(3, TRIANGLE, [[0.1, 0.2], [0.3, 0.4]], steps=1)


def test_a_start_vector_of_any_even_length_is_read_as_layers() -> None:
    for layers in (1, 2, 5):
        result = run_qaoa(3, TRIANGLE, torch.full((2 * layers,), 0.05), steps=1)
        assert result.layers == layers


@pytest.mark.parametrize(
    ("parameters", "energy", "message"),
    [
        (torch.zeros(3), torch.zeros(()), "the count must be even"),
        (torch.zeros(0), torch.zeros(()), "none were given"),
        (torch.zeros((2, 2)), torch.zeros(()), "one flat vector"),
        (torch.zeros(2), torch.zeros(2), "energy must be one scalar"),
    ],
)
def test_a_result_that_cannot_describe_a_run_is_refused(
    parameters, energy, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        QAOAResult(parameters, energy, ())


def test_the_solver_refuses_a_start_it_cannot_describe_before_any_evaluation() -> None:
    # The circuit is never built, so no work is spent on a run that cannot be
    # reported: the refusal is a validation error and not a mid-run failure.
    with pytest.raises(ValueError):
        run_qaoa(3, TRIANGLE, torch.zeros(7), steps=1)


def test_the_cost_operator_refuses_a_self_loop_as_a_constant_and_not_as_a_cut() -> None:
    with pytest.raises(ValueError, match="constant shift rather than a cut"):
        maxcut_hamiltonian(3, ((1, 1),))


def test_a_triple_whose_weight_is_a_bool_is_refused_as_a_typing_error() -> None:
    with pytest.raises(ValueError, match="weight must be a real number"):
        maxcut_hamiltonian(3, ((0, 1, True),))


def test_a_cost_operator_for_a_graph_whose_edges_are_not_integers_is_refused() -> None:
    with pytest.raises(ValueError, match="must name qubit indices"):
        maxcut_hamiltonian(3, ((0, 1.0),))


def test_an_edge_naming_a_qubit_outside_the_graph_is_refused_with_both_numbers() -> (
    None
):
    with pytest.raises(ValueError, match="names qubit 4, and the graph has 3 qubits"):
        maxcut_hamiltonian(3, ((0, 4),))


def test_the_solver_declares_the_saddle_and_the_layer_inexactness_it_measures() -> None:
    result = run_qaoa(3, TRIANGLE, torch.full((2,), 0.05))
    text = " ".join(result.limitations)
    assert "not exact" in text
    assert "float32" in text
    assert "no constraint" in text
    assert math.isfinite(float(result.energy))
