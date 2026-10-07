"""The four VQE entry points as one capability: one objective, four descents.

What is asserted here is the surface and the semantics the capability maturity
entry claims, and nothing about accuracy. Where a claim is about *which* run
produced a number the assertion is a candidate-set identity or a strict
inequality; where a quantity is the evidence for one run it is pinned to the
value this repository measured.

The two readings a caller can get wrong are asserted rather than described.
``converged`` reports why the loop stopped and not how close the energy came, so
a pool too small to express the ground state and a run that spent its iteration
budget are compared directly: the first reports True further from the ground
energy than the second reports False. And an ADAPT screening pass over a
reference state that is an eigenstate of every pool generator returns exactly
zero for every direction, so the same pool selects from ``|00>`` and selects
nothing from ``|++>``.

What is not asserted: that a deeper ansatz is better, that the optimizer's
trajectory decreases, that a reported energy bounds the ground energy, or that
any of the four entry points reaches an optimum at all.
"""

from __future__ import annotations

import pytest
import torch

from flagquantum.algorithms import (
    Hamiltonian,
    LayerwiseVQEResult,
    VQEResult,
    hardware_efficient_ansatz,
    hardware_efficient_parameter_count,
    pauli_term,
    run_adapt_vqe,
    run_hybrid_vqe,
    run_layerwise_vqe,
    run_vqe,
    vqe_loss,
)
from flagquantum.algorithms.core import AdaptVQEResult
from flagquantum.algorithms.optimization import OptimizationStage
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit

N_QUBITS = 2
ROTATIONS = ("ry",)
DEPTHS = (1, 2, 3)

#: The instance every assertion below runs on: a two-qubit transverse-field Ising
#: model whose ground energy is ``-sqrt(5)``, computed from the operator rather
#: than quoted. Small enough that an exact screening gradient is readable.
_GROUND = -(5.0**0.5)


def _cost() -> Hamiltonian:
    return Hamiltonian(
        (
            pauli_term(1.0, "ZZ", (0, 1)),
            pauli_term(-1.0, "X", (0,)),
            pauli_term(-1.0, "X", (1,)),
        )
    )


def _ansatz(parameters: torch.Tensor) -> Circuit:
    return hardware_efficient_ansatz(
        N_QUBITS, 1, parameters, rotations=ROTATIONS, entanglement="linear"
    )


def _layerwise(depth: int, parameters: torch.Tensor) -> Circuit:
    return hardware_efficient_ansatz(
        N_QUBITS, depth, parameters, rotations=ROTATIONS, entanglement="linear"
    )


def _layerwise_count(depth: int) -> int:
    return hardware_efficient_parameter_count(N_QUBITS, depth, ROTATIONS)


def _pool_circuit(
    reference, operators: tuple[object, ...], parameters: torch.Tensor
) -> Circuit:
    circuit = reference()
    for index, (operator, theta) in enumerate(zip(operators, parameters, strict=True)):
        circuit = circuit.gate(operator, index % N_QUBITS, theta=theta)
    return circuit


def _kets() -> Circuit:
    return Circuit(N_QUBITS, dtype=torch.complex128)


def _plus() -> Circuit:
    circuit = Circuit(N_QUBITS, dtype=torch.complex128)
    for wire in range(N_QUBITS):
        circuit = circuit.h(wire)
    return circuit


def _fixed_parameters() -> torch.Tensor:
    return torch.full(
        (hardware_efficient_parameter_count(N_QUBITS, 1, ROTATIONS),), 0.1
    )


def _stage(steps: int = 300, lr: float = 0.1) -> OptimizationStage:
    return OptimizationStage(group="quantum", method="adam", steps=steps, lr=lr)


def test_the_ground_energy_is_the_operator_s_own_value() -> None:
    """The instance's optimum is read from the operator, not from a run."""

    assert float(_cost().ground_energy()) == pytest.approx(_GROUND, abs=1e-12)


def test_the_four_entry_points_return_their_own_result_types() -> None:
    """Each entry point is reachable and reports a result of its own shape."""

    start = _fixed_parameters()
    fixed = run_vqe(_ansatz, start, _cost(), steps=5, lr=0.1)

    assert isinstance(fixed, VQEResult)
    assert fixed.n_steps == 5
    assert len(fixed.history) == 5

    adapted = run_adapt_vqe(
        lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
        ("ry",),
        _cost(),
        max_adapt_iterations=1,
        optimization_steps=5,
    )
    assert isinstance(adapted, AdaptVQEResult)
    assert adapted.selected_pool_indices == (0,)
    assert adapted.n_adapt_iterations == 1

    hybrid = run_hybrid_vqe(_ansatz, start, _cost(), stages=(_stage(steps=5),))
    assert hybrid.evaluations >= 5
    assert tuple(hybrid.parameters) == ("quantum",)

    layerwise = run_layerwise_vqe(
        _layerwise,
        _layerwise_count,
        torch.zeros(_layerwise_count(DEPTHS[0])),
        _cost(),
        depths=DEPTHS,
        stages=(_stage(steps=5),),
    )
    assert isinstance(layerwise, LayerwiseVQEResult)
    assert layerwise.depths == DEPTHS
    assert len(layerwise.stages) == len(DEPTHS)


def test_run_vqe_copies_the_caller_s_parameters_and_is_deterministic() -> None:
    """The start is an input: it survives, and two identical runs agree."""

    start = _fixed_parameters()
    before = start.clone()

    first = run_vqe(_ansatz, start, _cost(), steps=50, lr=0.1)
    second = run_vqe(_ansatz, start, _cost(), steps=50, lr=0.1)

    assert torch.equal(start, before)
    assert torch.equal(first.parameters, second.parameters)
    assert first.history == second.history
    # `history` records the objective at the start of each step and `energy` is a
    # fresh evaluation of the returned parameters, so the two are different
    # measurements: here the last Adam step moved the energy *up*, which is why
    # the reported energy is a fresh evaluation rather than `history[-1]`, and why
    # it is read as where the run stopped rather than as the best value it saw.
    assert float(first.energy) == pytest.approx(
        float(vqe_loss(_ansatz, first.parameters, _cost()).detach()), abs=1e-12
    )
    assert first.energy > min(first.history)
    assert first.energy < first.history[0]
    # Both are handed back detached: the descent's graph is an implementation
    # detail of the loop, so a run whose result still carried it would leak a
    # graph through a plain result object.
    assert first.parameters.requires_grad is False
    assert first.energy.requires_grad is False


def test_run_vqe_reports_an_energy_below_the_operator_s_ground_energy() -> None:
    """Single precision: the measured value, recorded rather than assumed away."""

    result = run_vqe(_ansatz, _fixed_parameters(), _cost(), steps=600, lr=0.1)

    # The value this repository measured. The point is not the digits but that
    # the run is *at* the optimum and the difference is the arithmetic's own.
    assert float(result.energy) == pytest.approx(-2.2360680103302002, abs=1e-9)
    assert abs(float(result.energy) - _GROUND) < 1e-6


def test_adapt_screening_is_exact_at_the_appended_angle() -> None:
    """The screened gradient is the derivative of the objective, not an estimate."""

    result = run_adapt_vqe(
        lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
        ("ry", "rz", "rx"),
        _cost(),
        max_adapt_iterations=1,
        optimization_steps=20,
        lr=0.2,
    )
    iteration = result.iterations[0]

    # From |00>, X and Z act as a stabilizer, so only the Y rotation moves: the
    # screened gradient is -sin(0) * 2 = -1 exactly, and the other two are zero.
    assert iteration.selected_pool_index == 0
    assert iteration.selected_gradient == pytest.approx(-1.0, abs=1e-12)
    assert iteration.pool_gradients[1] == pytest.approx(0.0, abs=1e-12)
    assert iteration.pool_gradients[2] == pytest.approx(0.0, abs=1e-12)
    assert iteration.energy_before == pytest.approx(1.0, abs=1e-12)
    assert iteration.energy_after < iteration.energy_before


def test_a_stationary_reference_state_screens_every_direction_to_zero() -> None:
    """The premise the unit cannot check, asserted as the same pool, two states."""

    pool = ("ry", "rz", "rx")
    moving = run_adapt_vqe(
        lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
        pool,
        _cost(),
        max_adapt_iterations=4,
        optimization_steps=300,
        lr=0.2,
    )
    stationary = run_adapt_vqe(
        lambda operators, parameters: _pool_circuit(_plus, operators, parameters),
        pool,
        _cost(),
        max_adapt_iterations=4,
        optimization_steps=300,
        lr=0.2,
    )

    assert moving.selected_pool_indices == (0,)
    assert stationary.selected_pool_indices == ()
    assert stationary.n_adapt_iterations == 0
    # Reported as an exit, and the energy is above the ground state: the flag is
    # about the pool at that reference state rather than about the optimum.
    assert stationary.converged is True
    assert float(stationary.energy) > _GROUND


def test_converged_reports_the_exit_and_not_the_distance_to_the_ground_state() -> None:
    """One boolean, three runs: the budget-limited one is the only False."""

    exhausted = run_adapt_vqe(
        lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
        ("ry",),
        _cost(),
        max_adapt_iterations=8,
        optimization_steps=300,
        lr=0.2,
        gradient_tolerance=0.0,
    )
    budget = run_adapt_vqe(
        lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
        ("ry", "rz", "rx"),
        _cost(),
        max_adapt_iterations=1,
        optimization_steps=1,
        lr=1e-9,
        gradient_tolerance=0.0,
    )

    assert exhausted.converged is True
    assert budget.converged is False
    # The run reporting the exit is the one closer to the ground energy, so the
    # flag cannot be read as an arrival test in either direction.
    assert abs(float(exhausted.energy) - _GROUND) < abs(float(budget.energy) - _GROUND)


def test_an_objective_without_a_gradient_is_refused_by_name() -> None:
    """Torch's backward error names neither the run nor the cause; this does."""

    start = _fixed_parameters()
    with pytest.raises(
        ValueError, match=r"^run_vqe: the objective carries no gradient"
    ) as fixed:
        run_vqe(lambda parameters: _kets(), start, _cost(), steps=2, lr=0.1)

    with pytest.raises(
        ValueError, match=r"^run_adapt_vqe: the objective carries no gradient"
    ) as adapted:
        run_adapt_vqe(
            lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
            ("h",),
            _cost(),
            max_adapt_iterations=2,
        )

    # The entry point is half of it; the other half is which builder produced the
    # objective, because that is the part the caller has to change. A message that
    # named only the run would be a marginally better torch error.
    assert "because the circuit that circuit_builder returns does not depend" in str(
        fixed.value
    )
    assert "because the operator at pool index 0 adds no dependence" in str(
        adapted.value
    )


def test_the_selected_operator_is_appended_at_angle_zero() -> None:
    """The new coordinate starts as identity, which is where the screen measured."""

    seen: list[tuple[tuple[object, ...], tuple[float, ...]]] = []

    def recording(operators, parameters):
        seen.append((tuple(operators), tuple(parameters.detach().tolist())))
        return _pool_circuit(_kets, operators, parameters)

    result = run_adapt_vqe(
        recording,
        ("ry", "rz", "rx"),
        _cost(),
        max_adapt_iterations=1,
        optimization_steps=3,
        lr=0.1,
        gradient_tolerance=1e-6,
    )

    assert result.selected_pool_indices == (0,)
    # The screening pass builds once for the reference energy, once per candidate,
    # then once more for the energy before the descent; the loop's first build
    # follows. Its appended coordinate is the one the screen never moved, so a run
    # that appended a non-zero angle would descend from somewhere the screen's
    # gradient does not describe.
    loop = seen[1 + 3 + 1 :]
    assert loop, seen
    assert loop[0][0] == ("ry",)
    assert loop[0][1] == (0.0,)
    assert loop[1][0] == ("ry",)


def test_a_builder_that_drops_selected_operators_is_refused_by_name() -> None:
    """The accumulated check is reachable when the screen is the caller's own."""

    with pytest.raises(
        ValueError, match=r"^run_adapt_vqe: the objective carries no gradient"
    ) as refusal:
        run_adapt_vqe(
            # A builder that applies nothing, with the caller supplying the screen:
            # the callback reports a gradient the circuit never carries, so the
            # refusal is reached by the accumulated objective rather than by
            # screening.
            lambda operators, parameters: _kets(),
            ("ry",),
            _cost(),
            screening_function=lambda selected, parameters, pool: [1.0],
            max_adapt_iterations=2,
            optimization_steps=2,
        )

    assert "because the circuit builder does not apply every selected operator" in str(
        refusal.value
    )


def test_layerwise_deepening_keeps_the_shallower_optimum_and_pads_with_zeros() -> None:
    """The deepening path continues rather than restarting, and says which depths."""

    seen: list[tuple[int, tuple[float, ...]]] = []

    def recording(depth: int, parameters: torch.Tensor) -> Circuit:
        seen.append((depth, tuple(parameters.detach().tolist())))
        return _layerwise(depth, parameters)

    result = run_layerwise_vqe(
        recording,
        _layerwise_count,
        torch.zeros(_layerwise_count(DEPTHS[0])),
        _cost(),
        depths=DEPTHS,
        stages=(_stage(steps=400),),
    )

    assert result.depths == DEPTHS
    # The builder is called at every optimization step, so the schedule is read
    # from the first call at each depth rather than from every call.
    first_call: dict[int, tuple[float, ...]] = {}
    for depth, values in seen:
        first_call.setdefault(depth, values)
    assert tuple(first_call) == DEPTHS
    assert {depth: len(values) for depth, values in first_call.items()} == {
        depth: _layerwise_count(depth) for depth in DEPTHS
    }
    # Every deeper stage starts from the previous stage's coordinates with the
    # new ones at zero, which is what "preserving optimized earlier layers" means.
    # The head is checked against the previous stage's own result rather than
    # against "not the initial zeros": a run that restarted every depth from the
    # initial point would still pad with zeros, so the padding alone does not
    # distinguish continuation from restart.
    assert first_call[2][2:] == (0.0, 0.0)
    assert first_call[3][4:] == (0.0, 0.0)
    assert first_call[2][:2] == tuple(result.stages[0].parameters["quantum"].tolist())
    assert first_call[3][:4] == tuple(result.stages[1].parameters["quantum"].tolist())
    assert first_call[2][:2] != (0.0, 0.0)
    # A strict inequality on the claim, not a pinned energy: the deepest stage is
    # at least as good as the shallowest on this instance.
    assert float(result.energy) <= float(result.stages[0].energy) + 1e-9


@pytest.mark.parametrize(
    ("depths", "message"),
    [
        ((), "depths must contain positive integers"),
        ((0,), "depths must contain positive integers"),
        ((2, 1), "depths must be strictly increasing"),
        ((1, 1), "depths must be strictly increasing"),
    ],
)
def test_layerwise_refuses_a_depth_schedule_it_cannot_deepen(
    depths: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_layerwise_vqe(
            _layerwise,
            _layerwise_count,
            torch.zeros(_layerwise_count(DEPTHS[0])),
            _cost(),
            depths=depths,
            stages=(_stage(steps=1),),
        )


def test_layerwise_refuses_a_parameter_count_that_does_not_grow() -> None:
    """A deeper ansatz that asks for no new coordinates cannot be a deepening."""

    with pytest.raises(ValueError, match="must not shrink as depth increases"):
        run_layerwise_vqe(
            # A builder that reads one coordinate, so the shrink is what this
            # test reaches rather than the ansatz's own parameter count.
            lambda depth, parameters: Circuit(N_QUBITS).gate(
                "ry", 0, theta=parameters[0]
            ),
            lambda depth: 10 - depth,
            torch.zeros(9),
            _cost(),
            depths=(1, 2),
            stages=(_stage(steps=1),),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_adapt_iterations": 0}, "max_adapt_iterations must be positive"),
        ({"optimization_steps": 0}, "optimization_steps must be positive"),
        ({"gradient_tolerance": -1.0}, "gradient_tolerance must be non-negative"),
    ],
)
def test_adapt_refuses_an_unusable_schedule(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_adapt_vqe(
            lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
            ("ry",),
            _cost(),
            **kwargs,
        )


def test_adapt_refuses_an_empty_pool_and_an_ambiguous_objective() -> None:
    with pytest.raises(ValueError, match="non-empty operator pool"):
        run_adapt_vqe(
            lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
            (),
            _cost(),
        )
    with pytest.raises(
        ValueError, match="exactly one of hamiltonian or energy_function"
    ):
        run_adapt_vqe(
            lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
            ("ry",),
        )
    with pytest.raises(
        ValueError, match="exactly one of hamiltonian or energy_function"
    ):
        run_adapt_vqe(
            lambda operators, parameters: _pool_circuit(_kets, operators, parameters),
            ("ry",),
            _cost(),
            energy_function=lambda circuit: circuit.state(),
        )
