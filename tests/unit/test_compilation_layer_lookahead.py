"""Native layer proposals preserve dependencies and bounded search behavior."""

from __future__ import annotations

import itertools
import random
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.compiler.topology import sabre
from flagquantum.compiler.topology._layer_lookahead import choose_layer_swaps
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit


def test_disjoint_pair_resolves_single_swap_tie_without_mutating_inputs() -> None:
    pairs = ((0, 2),) * 8
    successors = [[i + 1] if i < 7 else [] for i in range(8)]
    pending = [0] + [1] * 7
    layout = [0, 2, 3, 1]
    arguments = (
        {(0, 1), (2, 3)},
        [0],
        pairs,
        (True,) * 8,
        successors,
        pending,
        layout,
        CouplingMap.line(4).distance_matrix(),
    )
    before = deepcopy(arguments)
    assert choose_layer_swaps(*arguments) == ((0, 1), (2, 3))
    assert arguments == before


def test_short_complete_horizon_retains_upstream_deterministic_tie() -> None:
    assert choose_layer_swaps(
        {(0, 1), (2, 3)},
        [0],
        ((0, 3),),
        (True,),
        [[]],
        [0],
        list(range(4)),
        CouplingMap.line(4).distance_matrix(),
    ) == ((0, 1),)


def test_dependency_join_waits_for_all_predecessors() -> None:
    # The omitted predecessor is not ready. Its dependent distant gate must not
    # influence the proposal before that dependency has executed.
    assert choose_layer_swaps(
        {(0, 1), (2, 3)},
        [0],
        ((0, 3), (1,), (1, 3)),
        (True, False, True),
        [[2], [2], []],
        [0, 1, 2],
        list(range(4)),
        CouplingMap.line(4).distance_matrix(),
    ) == ((0, 1),)


def test_projection_stops_before_reading_more_than_160_operations() -> None:
    class UnreadableSuccessors(list[list[int]]):
        def __getitem__(self, index: int) -> list[int]:
            raise AssertionError("oversized frontier must not be traversed")

    assert (
        choose_layer_swaps(
            {(0, 1)},
            list(range(161)),
            ((0, 3),) * 161,
            (True,) * 161,
            UnreadableSuccessors(),
            [0] * 161,
            list(range(4)),
            CouplingMap.line(4).distance_matrix(),
        )
        is None
    )


def test_pair_enumeration_is_skipped_above_32_candidates() -> None:
    pairs = ((0, 3),) * 8
    with patch(
        "flagquantum.compiler.topology._layer_lookahead.itertools.combinations",
        side_effect=AssertionError("pair budget exceeded"),
    ):
        assert (
            choose_layer_swaps(
                {(0, 1), (2, 3)} | {(q, q + 1) for q in range(4, 36)},
                [0],
                pairs,
                (True,) * 8,
                [[i + 1] if i < 7 else [] for i in range(8)],
                [0] + [1] * 7,
                list(range(37)),
                CouplingMap.line(37).distance_matrix(),
            )
            is None
        )


@pytest.mark.parametrize("seed", range(10))
def test_extra_candidate_never_increases_delivered_cost(seed: int) -> None:
    rng = random.Random(seed)
    source = CircuitIR(
        8, tuple(Instruction("cx", tuple(rng.sample(range(8), 2))) for _ in range(32))
    )
    coupling = CouplingMap.grid(2, 4)
    with patch.object(sabre, "choose_layer_swaps", return_value=None):
        previous = sabre.plan_sabre_swaps(source, coupling)
    selected = sabre.plan_sabre_swaps(source, coupling)
    assert sabre._restored_swap_count(selected, coupling) <= sabre._restored_swap_count(
        previous, coupling
    )


def test_selection_matches_cirq_source_on_small_dependency_graphs() -> None:
    pytest.importorskip("cirq")
    from cirq.transformers.routing.route_circuit_cqc import RouteCQC

    class Mapping:
        def __init__(self, layout: list[int], distances: tuple[tuple[float, ...], ...]):
            self.layout = list(layout)
            self.distances = distances

        def apply_swap(self, first: int, second: int) -> None:
            self.layout = [
                second if value == first else first if value == second else value
                for value in self.layout
            ]

        def dist_on_device(self, first: int, second: int, *, undirected: bool) -> float:
            assert undirected
            return self.distances[self.layout[first]][self.layout[second]]

    for width in (4, 6, 8):
        topology = CouplingMap.line(width)
        distances = topology.distance_matrix()
        for seed in range(100):
            rng = random.Random(seed)
            pairs = tuple(tuple(rng.sample(range(width), 2)) for _ in range(5 * width))
            successors: list[list[int]] = [[] for _ in pairs]
            pending = [0] * len(pairs)
            previous: dict[int, int] = {}
            for index, pair in enumerate(pairs):
                for qubit in pair:
                    if qubit in previous:
                        successors[previous[qubit]].append(index)
                        pending[index] += 1
                    previous[qubit] = index
            front = [i for i, degree in enumerate(pending) if degree == 0]
            remaining = pending.copy()
            ready = list(front)
            layers = []
            while ready:
                layers.append([pairs[i] for i in ready])
                following = []
                for index in ready:
                    for successor in successors[index]:
                        remaining[successor] -= 1
                        if remaining[successor] == 0:
                            following.append(successor)
                ready = following
            layout = list(range(width))
            rng.shuffle(layout)
            candidates = {
                tuple(sorted((layout[q], neighbor)))
                for index in front
                for q in pairs[index]
                for neighbor in topology.neighbors(layout[q])
            }
            manager = Mapping(layout, distances)
            expected = RouteCQC._choose_optimal_swap(
                manager, layers, 0, 8, [(edge,) for edge in sorted(candidates)]
            )
            if expected is None:
                options = [
                    (first, second)
                    for first, second in itertools.combinations(sorted(candidates), 2)
                    if set(first).isdisjoint(second)
                ]
                expected = (
                    RouteCQC._choose_optimal_swap(manager, layers, 0, 8, options)
                    if options
                    else None
                )
            actual = choose_layer_swaps(
                candidates,
                front,
                pairs,
                (True,) * len(pairs),
                successors,
                pending,
                layout,
                distances,
            )
            assert actual == expected, (width, seed)
            assert manager.layout == layout


def _layer_trial(
    source: CircuitIR,
    coupling: CouplingMap,
    baseline: sabre.SabrePlan,
) -> sabre.SabrePlan:
    plan = sabre._plan_sabre_trial(
        source,
        coupling,
        initial_layout=baseline.initial_logical_to_physical,
        layer_lookahead=True,
    )
    assert plan is not None
    return plan


def test_layer_trial_preserves_complex_state_and_shared_parameter_gradient() -> None:
    def source(theta: torch.Tensor) -> fq.Circuit:
        return (
            fq.Circuit(4, dtype=torch.complex128)
            .h(0)
            .ry(2, theta=theta)
            .cx(0, 3)
            .rz(3, theta=2 * theta)
            .cx(2, 0)
            .rxx(1, 3, theta=-theta)
            .ry(0, theta=theta + 0.13)
            .cx(3, 1)
        )

    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    reference = source(theta).state()
    with (
        patch.object(sabre, "_select_sabre_trials", side_effect=_layer_trial),
        patch.object(
            sabre, "choose_layer_swaps", wraps=choose_layer_swaps
        ) as proposals,
    ):
        routed = route_to_topology(source(theta), CouplingMap.line(4), strategy="sabre")
    assert proposals.call_count > 0
    observed = fq.Circuit.from_ir(routed).state()
    torch.testing.assert_close(observed, reference, atol=1e-12, rtol=1e-12)
    weights = torch.arange(reference.numel(), dtype=torch.float64).reshape(
        reference.shape
    )
    expected_loss = (weights * reference.real + weights.flip(-1) * reference.imag).sum()
    observed_loss = (weights * observed.real + weights.flip(-1) * observed.imag).sum()
    (expected_gradient,) = torch.autograd.grad(expected_loss, theta, retain_graph=True)
    (actual_gradient,) = torch.autograd.grad(observed_loss, theta)
    assert abs(float(expected_gradient)) > 1e-5
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )
    assert routed.metadata["routing"]["mapping_restored"] is True


def test_layer_trial_preserves_symbols_and_classical_dependency_order() -> None:
    theta = Parameter("theta")
    source = CircuitIR(
        6,
        (
            Instruction("cx", (1, 5)),
            Instruction(
                "measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
            Instruction(
                "rz",
                (4,),
                {"theta": theta},
                metadata={"conditions": ((0, 1),), "is_dynamic": True},
            ),
            Instruction("cx", (5, 0)),
            Instruction("cx", (1, 4)),
        ),
    )
    with (
        patch.object(sabre, "_select_sabre_trials", side_effect=_layer_trial),
        patch.object(
            sabre, "choose_layer_swaps", wraps=choose_layer_swaps
        ) as proposals,
    ):
        routed = route_to_topology(source, CouplingMap.line(6), strategy="sabre")
    assert proposals.call_count > 0
    original = [op for op in routed if op.metadata.get("routing_phase") is None]
    conditional = next(op for op in original if op.name == "rz")
    assert conditional.params["theta"] is theta
    assert conditional.metadata["conditions"] == ((0, 1),)
    assert [op.name for op in original].index("measure") < original.index(conditional)
    assert sorted(op.metadata["source_instruction_index"] for op in original) == list(
        range(len(source))
    )
    assert routed.metadata["routing"]["mapping_restored"] is True
