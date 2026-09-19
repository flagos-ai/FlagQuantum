from __future__ import annotations

import pytest
import torch

from flagquantum.algorithms.kmedians import (
    _MAX_CENTROIDS,
    KMediansResult,
    _nearest_centroid,
    _strictly_better,
    kmedians,
)
from flagquantum.algorithms.primitives.oracle import marked_states

pytestmark = pytest.mark.unit

# Every expected value in this file was produced by a real run of the construction it
# checks. The searches are sampled, so the sample size is part of every value: 1024 is
# the module's own default and is used everywhere except in the test that measures what
# a smaller sample does.
_SHOTS = 1024
_SEED = 7


def _classical_labels(points: torch.Tensor, centroids: torch.Tensor) -> list[int]:
    """Return the nearest centroid of every point, computed here rather than searched for.

    This is the independent reference path. It forms the distance from each point to
    each centroid with its own arithmetic, exhaustively over the ``(point, centroid)``
    pairs, and takes the lexicographic minimum of distance and index, so it shares no
    code with the module under test and no value with it. The tie rule is the module's
    documented one, restated here as the reference's own rule: of the centroids at the
    smallest distance, the lowest index wins.
    """
    rows = points.to(torch.float64)
    positions = centroids.to(torch.float64)
    labels: list[int] = []
    for row in rows:
        distances = [
            float(torch.linalg.vector_norm(row - position)) for position in positions
        ]
        labels.append(min(range(len(distances)), key=lambda j: (distances[j], j)))
    return labels


def _plane(
    points: list[list[float]], centroids: list[list[float]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the two tensors of a run, in double precision."""
    return (
        torch.tensor(points, dtype=torch.float64),
        torch.tensor(centroids, dtype=torch.float64),
    )


def _chain(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a point set whose nearest centroids sit at the far end of the register.

    The centroids are ``(0, 0)`` through ``(k - 1, 0)``, and the points are at ``(7, 0)``
    and ``(4, 0)``: with eight centroids the nearest ones are index 7 and index 4, so
    each point's loop has to move several times before it settles, which is what makes
    this instance show what the sample size does.
    """
    return _plane(
        [[7.0, 0.0], [4.0, 0.0]],
        [[float(index), 0.0] for index in range(k)],
    )


def test_the_assignment_is_the_classical_labelling_of_every_point() -> None:
    """Six points against three centroids, compared point by point with the reference.

    The reference forms all 18 ``(point, centroid)`` distances itself and takes each
    point's nearest one, so an assignment that skipped a point, reordered the points, or
    returned a centroid index for something other than the nearest centroid fails here.
    Measured, the labels are ``(0, 0, 1, 1, 2, 1)`` and they held for every sampling seed
    from 0 through 3.
    """
    points, centroids = _plane(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 0.0],
            [5.2, 0.1],
            [0.2, 1.5],
            [4.9, -0.4],
        ],
        [[0.0, 0.0], [5.0, 0.0], [0.0, 2.0]],
    )
    want = _classical_labels(points, centroids)
    assert want == [0, 0, 1, 1, 2, 1]

    for seed in range(4):
        result = kmedians(points, centroids, shots=_SHOTS, seed=seed)

        assert list(result.labels) == want, seed
        assert len(result.labels) == points.shape[0], seed


def test_the_assignment_holds_at_every_centroid_count_the_register_carries() -> None:
    """Two to eight centroids, so every register width the search can carry is exercised.

    The register is one wire at two centroids and three at eight, and the number of
    register slots the centroid set does not fill grows as the count moves away from a
    power of two -- two slots at three centroids in a two-wire register. The reference is
    recomputed at each count, and every width has to agree with it.
    """
    points = torch.tensor(
        [
            [0.0, 0.0],
            [0.4, 0.2],
            [2.1, 0.0],
            [2.0, 1.9],
            [3.6, 2.0],
            [0.5, 3.4],
        ],
        dtype=torch.float64,
    )
    grid = [(x, y) for x in (0.0, 2.0, 4.0) for y in (0.0, 2.0, 4.0)]

    for k in range(2, _MAX_CENTROIDS + 1):
        centroids = torch.tensor(grid[:k], dtype=torch.float64)
        want = _classical_labels(points, centroids)

        for seed in range(4):
            result = kmedians(points, centroids, shots=_SHOTS, seed=seed)

            assert list(result.labels) == want, (k, seed)


def test_a_point_equidistant_from_two_centroids_takes_the_lower_index() -> None:
    """The tie rule: the comparison is on the pair of distance and index.

    The point is at ``(3, 3)`` and the centroids are at ``(0, 0)``, ``(2, 0)`` and
    ``(0, 2)``. Its distances to the last two are exactly equal -- both are
    ``3.1622776601683795``, the square root of ten, and the two sums of squares that
    produce it are the same two terms in the other order -- and both are shorter than the
    ``4.242640687119285`` to the first, so the assignment is a tie between indices 1 and
    2. Measured over every sampling seed from 0 through 199 at 1024 shots, the assignment
    is index 1, and two of its rounds are what establishes that: the round that leaves
    index 0 marks both tied centroids, and the round after it marks only index 1, because
    index 2 is at exactly the threshold distance with a larger index.

    This is the test a distance-only comparison fails. Measured with the index dropped
    from the pair, the same instance returns index 2 at seed 0 and both indices across
    seeds 0 through 199, because which tied centroid the sample favours is then the
    assignment and the seed is what decides it.
    """
    points, centroids = _plane([[3.0, 3.0]], [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    distances = [
        float(torch.linalg.vector_norm(points[0] - centroid)) for centroid in centroids
    ]
    assert distances[1] == distances[2] == 3.1622776601683795
    assert distances[1] < distances[0] == 4.242640687119285

    for seed in range(10):
        result = kmedians(points, centroids, shots=_SHOTS, seed=seed)

        assert result.labels == (1,), seed


def test_a_point_equidistant_from_every_centroid_takes_index_zero() -> None:
    """The other edge of the tie rule, and the case the brief's "does not crash" names.

    The point is at ``(1, 1)``, the centroids at ``(0, 3)``, ``(3, 0)`` and ``(2, -1)``,
    and all three distances are exactly ``2.23606797749979``. No centroid is strictly
    nearer than the index the loop starts from, and none is equally near with a smaller
    index, so the first round finds nothing and the assignment is index 0. The run
    completes and the medians are formed rather than the run failing on the tie.
    """
    points, centroids = _plane([[1.0, 1.0]], [[0.0, 3.0], [3.0, 0.0], [2.0, -1.0]])
    distances = [
        float(torch.linalg.vector_norm(points[0] - centroid)) for centroid in centroids
    ]
    assert distances == [2.23606797749979] * 3

    for seed in range(4):
        result = kmedians(points, centroids, shots=_SHOTS, seed=seed)

        assert result.labels == (0,), seed
        assert result.medians == ((1.0, 1.0), (3.0, 0.0), (2.0, -1.0)), seed
        assert result.searches == 1, seed


def test_the_search_never_returns_a_slot_no_centroid_occupies() -> None:
    """Three centroids in a two-wire register: the fourth slot names no centroid.

    The register holds four values and the centroid set fills three of them, so an
    assignment of 3 would be a label no centroid answers to. The measured sets below are
    the predicate's own marked states read over the whole register: against index 0 it
    marks indices 1 and 2 and not 3, and against the index carrying the smallest distance
    it marks nothing at all.
    """
    distances = [5.0, 4.0, 3.0]
    keys = [(distance, index) for index, distance in enumerate(distances)]
    assert marked_states(_strictly_better(keys, keys[0], 3), 2) == (1, 2)
    assert marked_states(_strictly_better(keys, keys[2], 3), 2) == ()

    points, centroids = _plane([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]])
    for seed in range(8):
        result = kmedians(points, centroids, shots=_SHOTS, seed=seed)

        assert all(label in range(3) for label in result.labels), seed


def test_the_run_is_bounded_at_eight_centroids() -> None:
    """Nine centroids need a four-wire index register, which the search refuses.

    Eight centroids is three wires, the widest the Grover search this unit runs carries,
    and it works. The ninth is refused by this module's own check rather than deeper in
    the search, so the message names the bound and the reason.
    """
    points = torch.zeros((2, 1), dtype=torch.float64)
    grid = [[float(index)] for index in range(9)]

    eight = kmedians(
        points, torch.tensor(grid[:8], dtype=torch.float64), shots=_SHOTS, seed=0
    )
    assert len(eight.labels) == 2

    with pytest.raises(ValueError, match="bounded at 8 centroids"):
        kmedians(points, torch.tensor(grid, dtype=torch.float64), shots=_SHOTS, seed=0)


def test_the_median_update_takes_the_lower_middle_and_keeps_an_empty_cluster() -> None:
    """One run that shows both median conventions, because neither is forced by the word.

    Points ``(0,)``, ``(1,)``, ``(2,)`` and ``(10,)`` against centroids ``(0,)`` and
    ``(50,)``: the far point is nearer to index 0 than to index 1, so all four are
    assigned to index 0 and cluster 1 is empty. The cluster of four has an even size, so
    its medians are a range -- ``1`` and ``2`` both minimise the sum of absolute
    deviations -- and this unit takes the lower one, which is the measured ``1.0``. The
    empty cluster keeps the ``50.0`` it was given. Measured at every sampling seed from 0
    through 3.
    """
    points, centroids = _plane([[0.0], [1.0], [2.0], [10.0]], [[0.0], [50.0]])

    for seed in range(4):
        result = kmedians(points, centroids, shots=_SHOTS, seed=seed)

        assert result.labels == (0, 0, 0, 0), seed
        assert result.medians == ((1.0,), (50.0,)), seed


def test_the_search_cost_is_one_round_per_point_plus_one_per_move() -> None:
    """A point costs a round even when its first round finds nothing, and a round per move.

    The first instance puts every point nearer to index 0 than to index 1, so no point
    ever moves, every round is a round that found nothing, and the cost is one round per
    point: measured at 3 for three points, at every sampling seed from 0 through 5. The
    second instance is the one behind this file's sample-size test, and none of its
    points starts on its nearest centroid, so its cost is above that floor. Its ceiling
    is the module's own bound -- one round per point per centroid, because each moving
    round strictly decreases the pair the loop compares -- and both bounds are checked
    against the run rather than assumed.
    """
    points, centroids = _plane([[0.0], [0.1], [0.2]], [[0.0], [10.0]])
    results = [
        kmedians(points, centroids, shots=_SHOTS, seed=seed) for seed in range(6)
    ]
    assert all(result.labels == (0, 0, 0) for result in results)
    assert {result.searches for result in results} == {3}

    chain_points, chain_centroids = _chain(8)
    for seed in range(4):
        moved = kmedians(chain_points, chain_centroids, shots=_SHOTS, seed=seed)

        assert moved.labels == (7, 4), seed
        assert chain_points.shape[0] < moved.searches, seed
        assert moved.searches <= chain_points.shape[0] * chain_centroids.shape[0], seed


def test_a_small_sample_leaves_the_search_short() -> None:
    """The sample is what the assignment rests on, and a small one is visibly not enough.

    The instance is the one whose nearest centroids are index 7 and index 4 of an
    eight-centroid register, so each point's loop is several rounds long. Measured over
    sampling seeds 0 through 49 at one shot, 43 of the 50 assignments differ from the
    classical labelling; at two shots, 23 differ; at four shots, 8; and at 16 shots and
    above, none. This test exists so that the module's statement that the search is
    sampled and not read out is a measured property of the code and not a caveat: an
    implementation that replaced the search with the classical answer would return the
    classical labelling at one shot too, and the floor below fails.
    """
    points, centroids = _chain(8)
    want = _classical_labels(points, centroids)
    assert want == [7, 4]

    short = sum(
        list(kmedians(points, centroids, shots=1, seed=seed).labels) != want
        for seed in range(50)
    )
    assert short == 43
    assert (
        sum(
            list(kmedians(points, centroids, shots=2, seed=seed).labels) != want
            for seed in range(50)
        )
        == 23
    )
    assert (
        sum(
            list(kmedians(points, centroids, shots=4, seed=seed).labels) != want
            for seed in range(50)
        )
        == 8
    )

    for shots in (16, 64, 1024):
        assert all(
            list(kmedians(points, centroids, shots=shots, seed=seed).labels) == want
            for seed in range(50)
        ), shots


def test_the_same_seed_replays_the_same_result_field_for_field() -> None:
    """The seed decides the sample, so the same seed rebuilds the same result."""
    points, centroids = _chain(8)

    first = kmedians(points, centroids, shots=512, seed=_SEED)
    second = kmedians(points, centroids, shots=512, seed=_SEED)

    assert first == second
    assert first.labels == second.labels
    assert first.medians == second.medians
    assert first.searches == second.searches


def test_the_callers_dtype_does_not_change_the_assignment() -> None:
    """A single-precision input is promoted before the distances are compared.

    The comparison the search makes is a comparison of floats, and the module forms the
    distance table in double precision whatever dtype it was handed, so the single
    precision spelling of an instance assigns what the double precision spelling does.
    Both runs are the same seed, and the values here are exactly representable in both
    dtypes, so the two spellings are the same points.
    """
    points, centroids = _plane(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]
    )

    double = kmedians(points, centroids, shots=_SHOTS, seed=5)
    single = kmedians(
        points.to(torch.float32), centroids.to(torch.float32), shots=_SHOTS, seed=5
    )

    assert double.labels == single.labels == (0, 0, 1)
    assert double.medians == single.medians == ((0.0, 0.0), (3.0, 0.0), (4.0, 0.0))


def test_the_round_count_is_the_loops_own() -> None:
    """Round counts read directly off the loop, including the round that finds nothing.

    The first call's nearest centroid is index 1, and its first round marks indices 1 and
    2 while its second marks nothing, so it is three rounds: two that moved and the one
    that ended the loop. The second call starts on the nearest centroid, so its first
    round is the one that finds nothing and the whole point costs one round. The third is
    a tie between indices 1 and 2, and it also settles on index 1.
    """
    assert _nearest_centroid([9.0, 0.5, 3.0], n_wires=2, shots=_SHOTS, seed=0) == (1, 3)
    assert _nearest_centroid([0.5, 9.0, 3.0], n_wires=2, shots=_SHOTS, seed=0) == (0, 1)
    assert _nearest_centroid([9.0, 1.0, 1.0], n_wires=2, shots=_SHOTS, seed=0) == (1, 3)


def test_the_points_and_centroids_are_validated() -> None:
    """Both operands are checked before a circuit is built, and each check has its case.

    The coordinate-count check is the one a point-by-point comparison cannot catch on its
    own: two operands in spaces of different dimension have no distance at all, and the
    message names the two counts rather than letting the arithmetic broadcast or fail
    somewhere further down.
    """
    one = torch.zeros((1, 1), dtype=torch.float64)

    with pytest.raises(ValueError, match="torch.Tensor"):
        kmedians([[0.0], [1.0]], one)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two-dimensional"):
        kmedians(torch.zeros(4, dtype=torch.float64), one)
    with pytest.raises(ValueError, match="floating-point"):
        kmedians(torch.zeros((2, 1), dtype=torch.int64), one)
    with pytest.raises(ValueError, match="finite"):
        kmedians(torch.tensor([[float("nan")]]), one)
    with pytest.raises(ValueError, match="at least one point"):
        kmedians(torch.zeros((0, 1), dtype=torch.float64), one)
    with pytest.raises(ValueError, match="at least two slots"):
        kmedians(torch.zeros((2, 1), dtype=torch.float64), one)
    with pytest.raises(ValueError, match="agree on the number of coordinates"):
        kmedians(
            torch.zeros((2, 2), dtype=torch.float64),
            torch.zeros((2, 1), dtype=torch.float64),
        )


def test_the_run_validates_its_shot_count() -> None:
    """A sample with no shots is refused rather than run into a fixed assignment."""
    points, centroids = _plane([[0.0], [1.0]], [[0.0], [1.0]])

    with pytest.raises(ValueError, match="at least one shot"):
        kmedians(points, centroids, shots=0)


def test_the_result_validates_its_own_fields() -> None:
    """A result that could not come from an assignment is refused at construction."""
    fields = {"medians": ((0.0,), (1.0,)), "searches": 2}

    with pytest.raises(ValueError, match="at least one point"):
        KMediansResult(labels=(), **fields)
    with pytest.raises(ValueError, match="at least one centroid position"):
        KMediansResult(labels=(0,), medians=(), searches=1)
    with pytest.raises(ValueError, match="cannot be negative"):
        KMediansResult(labels=(-1,), **fields)
    with pytest.raises(ValueError, match="costs at least one search"):
        KMediansResult(labels=(0, 1), medians=((0.0,), (1.0,)), searches=1)
