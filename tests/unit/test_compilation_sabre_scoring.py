"""Compare native relative scoring with independent exhaustive absolute costs."""

from __future__ import annotations

import itertools
import math
import random
from unittest.mock import patch

import pytest

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.topology import sabre
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _absolute_choice(
    candidates: set[tuple[int, int]],
    wires: tuple[tuple[int, ...], ...],
    front: list[int],
    extended: list[int],
    layout: list[int],
    distances: tuple[tuple[float, ...], ...],
    *,
    tie_rng: random.Random | None = None,
) -> tuple[int, int] | None:
    costs = {}
    ordered = sorted(candidates)
    if tie_rng is not None:
        tie_rng.shuffle(ordered)
    for edge in ordered:
        moved = [
            edge[1] if x == edge[0] else edge[0] if x == edge[1] else x for x in layout
        ]
        basic = math.fsum(
            distances[moved[wires[i][0]]][moved[wires[i][1]]] for i in front
        ) / len(front)
        ahead = (
            math.fsum(
                sabre.LOOKAHEAD_DECAY**k
                * distances[moved[wires[i][0]]][moved[wires[i][1]]]
                for k, i in enumerate(extended)
            )
            * sabre.LOOKAHEAD_WEIGHT
            / len(extended)
            if extended
            else 0.0
        )
        costs[edge] = basic + ahead
    return min(costs, key=costs.__getitem__) if costs else None


@pytest.mark.parametrize("repeats", [1, 7])
@pytest.mark.parametrize("lookahead", [False, True])
def test_every_layout_and_front_pair_matches_absolute_minimum(
    lookahead: bool, repeats: int
) -> None:
    coupling = CouplingMap(4, ((0, 1), (1, 2), (1, 3)))
    distances = tuple(
        tuple(float(x) for x in row) for row in coupling.distance_matrix()
    )
    pairs = tuple(itertools.combinations(range(4), 2))
    for permutation in itertools.permutations(range(4)):
        for first, second in itertools.product(pairs, repeat=2):
            wires = (first, second, (0, 3), (1, 2))
            front = [0, 1] * repeats
            extended = [2, 3] if lookahead else []
            expected = _absolute_choice(
                set(coupling.edges),
                wires,
                front,
                extended,
                list(permutation),
                distances,
            )
            actual = sabre._relative_score_swap(
                set(coupling.edges),
                wires,
                front,
                extended,
                list(permutation),
                distances,
            )
            assert actual == expected


def test_equal_cost_candidates_choose_first_edge_and_empty_candidates_return_none() -> (
    None
):
    distances = ((0.0, 1.0, 2.0), (1.0, 0.0, 1.0), (2.0, 1.0, 0.0))
    assert sabre._relative_score_swap(
        {(1, 2), (0, 1)}, ((0, 2),), [0], [], [0, 1, 2], distances
    ) == (0, 1)
    assert (
        sabre._relative_score_swap(set(), ((0, 2),), [0], [], [0, 1, 2], distances)
        is None
    )


@pytest.mark.parametrize("seed", range(12))
def test_complete_route_matches_absolute_scoring_replacement(seed: int) -> None:
    rng = random.Random(seed)
    program = CircuitIR(
        16, tuple(Instruction("cx", tuple(rng.sample(range(16), 2))) for _ in range(32))
    )
    coupling = CouplingMap.grid(4, 4)
    actual = sabre.plan_sabre_swaps(program, coupling)
    with patch.object(sabre, "_relative_score_swap", _absolute_choice):
        reference = sabre.plan_sabre_swaps(program, coupling)
    assert actual == reference


@pytest.mark.parametrize("seed", [0, 1])
def test_relative_scoring_preserves_state_and_shared_parameter_gradient(
    seed: int,
) -> None:
    import torch

    import flagquantum as fq
    from flagquantum.compiler import route_to_topology

    def circuit(angle: torch.Tensor) -> fq.Circuit:
        result = fq.Circuit(14, dtype=torch.complex128).h(seed).ry(13, theta=angle)
        rng = random.Random(seed)
        for _ in range(20):
            result.cx(*rng.sample(range(14), 2))
        return result.rx(0, theta=angle).rz(13, theta=0.31)

    angle = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    routed_angle = angle.detach().clone().requires_grad_(True)
    expected = circuit(angle).state()
    routed = route_to_topology(
        circuit(routed_angle), CouplingMap.line(14), strategy="sabre"
    )
    actual = fq.Circuit.from_ir(routed).state()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    weights = torch.linspace(0, 1, expected.numel(), dtype=torch.float64).reshape(
        expected.shape
    )
    reference_gradient = torch.autograd.grad(
        (expected.abs().square() * weights).sum(), angle
    )[0]
    actual_gradient = torch.autograd.grad(
        (actual.abs().square() * weights).sum(), routed_angle
    )[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, atol=1e-11, rtol=0)
