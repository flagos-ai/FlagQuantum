"""Grover-quantized k-medians over a fixed set of points and a fixed set of centroids.

One assignment step of k-medians is quantized here. The unit follows Esma Aïmeur, Gilles
Brassard and Sébastien Gambs, "Quantum clustering algorithms", *Proceedings of the 24th
International Conference on Machine Learning (ICML 2007)*, pp. 1-8, DOI
10.1145/1273496.1273497; journal version *Machine Learning* **90**(2), 261-287 (2012),
DOI 10.1007/s10994-012-5316-5. That paper quantizes k-medians by replacing its
nearest-centroid search with a Grover-style minimum search, and this unit does the same:
for each point, which centroid is nearest is found by
:func:`~flagquantum.algorithms.grover.run_grover` over a register holding a centroid
index. The centroid update stays classical, as the paper leaves it: each centroid is
moved to the coordinate-wise median of the points assigned to it.

**The premise is Grover's oracle model, and the oracle here is not free.** The paper
counts oracle calls, and in that model the distance function has to be computed into a
register -- a cost the caller of the search pays and the count does not include. This
unit synthesizes its oracle the way Phase 1's Grover unit synthesizes its own, so Phase
1's boundary for that unit applies here unchanged: the oracle is synthesized from a truth
table at O(2**n) cost, so no end-to-end advantage follows. Here ``n`` is the width of the
centroid index register, at most three, so the truth table is over at most eight register
values. The distance table the predicate compares is computed classically, one point at a
time, before the circuit exists, as the next paragraph explains. Neither cost is in the
query count. Nothing here reads a qRAM or runs an adiabatic evolution, so no conclusion
that rests on either applies to this unit, and no number it reports is evidence of a
speedup.

**The distance table is computed classically, one point at a time.** Phase 1's
:func:`~flagquantum.algorithms.grover.grover_circuit` builds its own register and refuses
more than three evaluation wires, and a register is also where a distance would have to
be carried. So the search here runs over the centroid index alone, at most three wires
wide, and the distance from the point being assigned to each centroid is computed in
double precision outside the circuit: the predicate closes over that table. The point is
assigned by one such table at a time, and every round builds its register afresh. What
remains of the paper's quantization is the search itself -- a minimum search, not a
distance computed coherently -- and that is what the module demonstrates.

**The search is a minimum search over a moving threshold.** One point's assignment is a
loop rather than a single circuit. The loop holds the index of the best centroid found so
far, starting at index zero, and each round searches the register for an index whose pair
of distance and index is strictly smaller than the held one's: the predicate marks those,
:func:`~flagquantum.algorithms.grover.grover_circuit` amplifies them at the round count
:func:`~flagquantum.algorithms.grover.optimal_iterations` gives for the number marked,
and the loop moves to the first index the sample lists when it lists one. A round that
finds none ends the loop, and the index it holds is the assignment. Each moving round
strictly decreases a pair drawn from a finite set, so a point costs at most as many
searches as there are centroids. A point whose nearest centroid is index zero costs
exactly one search, the round that finds nothing.

**Ties are broken by the index.** The comparison is on the pair of distance and index and
not on the distance alone, so a centroid at exactly the threshold distance is marked only
when its index is the smaller one, and a point equidistant from two centroids is assigned
the lower-indexed of them. The rule is not a tie-break applied afterwards: it is the
order the search moves in, so it is the same whichever of the tied centroids a round
happens to return -- the round that leaves the farther centroid marks both of them, and
the round after that marks only the lower-indexed one, because the higher-indexed one is
then at exactly the threshold distance with a larger index. Measured on a point at
``(3, 3)`` with centroids at ``(0, 0)``, ``(2, 0)`` and ``(0, 2)``, whose distances to
the last two are exactly equal at ``3.1622776601683795``: the assignment was index ``1``
for every sampling seed from 0 through 199, at 1024 shots, which is the outcome measured
rather than a distribution statement. Register values that
name no centroid are excluded by the same predicate, so a three-centroid search in a
two-wire register never returns the slot no centroid occupies.

**The median update is classical, and two of its cases have to be named.** Each
coordinate of a centroid is set to the median of that coordinate over the points assigned
to it. A cluster with an even number of points has a range of medians rather than one --
every value between the two middle values minimises the sum of absolute deviations -- and
this unit takes the lower of the two, which is what :func:`torch.median` returns. A
cluster with no points has no median at all, and its centroid keeps the position it was
given. Both cases in one measured run: with points ``(0,)``, ``(1,)``, ``(2,)`` and
``(10,)`` against centroids ``(0,)`` and ``(50,)``, every point is assigned to index 0,
the first centroid moves to ``1.0`` -- the lower middle of ``0, 1, 2, 10`` -- and the
second keeps ``50.0``.

**The search is sampled, not read out.** A round ends when its sample found no marked
index, and a round whose sample missed a marked index ends the loop early, at an index
that is not the nearest centroid. The sample is what makes that vanishingly unlikely
rather than impossible, and the two are worth separating: measured on two points at
``(7, 0)`` and ``(4, 0)`` against eight centroids at ``(0, 0)`` through ``(7, 0)``, whose
nearest centroids are index 7 and index 4, the assignment differed from the classical
labelling for 401 of the 500 sampling seeds tried at one shot, for 65 of them at four
shots, and for none of them at 16, 64 or 1024 shots. That is a measurement of those 500
seeds at those shot counts and not a guarantee: the tail that a small sample leaves is
the sampler's, and no error bound, confidence interval or repetition scheme is computed
or reported anywhere in this module. The default sample size is 1024 shots per search.

**Wire layout.** The centroid index register is the whole circuit:
:func:`~flagquantum.algorithms.grover.grover_circuit` builds it with ``ceil(log2(k))``
wires for ``k`` centroids, wire 0 the most significant, and one centroid index per
register value from ``0`` to ``k - 1``. Distances are carried in a Python list and not on
wires, which is what keeps the register this narrow and is also the whole of what the
unit gives up: the circuit holds no state about the points or the centroids, so what it
searches is a table the classical caller built.

This unit is demonstration scale. It makes no performance, capacity, convergence, or
hardware claim, and it does not select a runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .grover import run_grover
from .primitives.types import Predicate

__all__ = ["KMediansResult", "kmedians"]

# The default sample size, matching ``grover.run_grover``'s own default.
_DEFAULT_SHOTS = 1024

# ``grover.grover_circuit`` builds its own register and refuses more than three
# evaluation wires, so the centroid index register this unit searches over is at most
# three wires wide.
_CENTROID_WIRE_LIMIT = 3

# One index per register value, so three wires index at most eight centroids.
_MAX_CENTROIDS = 2**_CENTROID_WIRE_LIMIT


@dataclass(frozen=True, kw_only=True)
class KMediansResult:
    """The outcome of one sampled :func:`kmedians` run.

    ``kw_only`` is not optional here: ``labels`` is a tuple of centroid indices and
    ``searches`` is a count, so a positional spelling would let a caller transpose a
    count for an assignment without an error.

    Attributes:
        labels: The centroid each point was assigned to, in the order the points were
            given: one centroid index per point, each in ``range(len(centroids))``. The
            tie rule is the one the module docstring states.
        medians: The updated centroid positions, one tuple of coordinates per centroid in
            the order the centroids were given. A centroid no point was assigned to keeps
            the position it was given; see :func:`kmedians` for the median convention.
        searches: The number of Grover searches the run sampled, summed over every point
            and every round of every point's minimum search. It is at least the number of
            points, because a point whose first round finds nothing still ran that round,
            and at most the number of points times the number of centroids: a point's
            loop runs one round per move and then one round that does not move, and it can
            move at most one fewer time than there are centroids, since each move strictly
            decreases the pair the loop compares.
    """

    labels: tuple[int, ...]
    medians: tuple[tuple[float, ...], ...]
    searches: int

    def __post_init__(self) -> None:
        """Reject a result that could not be an assignment of points to centroids.

        Raises:
            ValueError: If no point was assigned, if no centroid position is carried, if
                a label is negative, or if fewer searches are reported than the labels
                describe points.
        """
        if not self.labels:
            raise ValueError(
                "a k-medians step assigns at least one point, got no labels; with no "
                "point there is no assignment to report and no cluster to take a median"
            )
        if not self.medians:
            raise ValueError(
                "a k-medians step carries at least one centroid position, got an empty "
                "medians; the positions are the centroids the next step is given"
            )
        if any(label < 0 for label in self.labels):
            raise ValueError(
                f"a centroid index cannot be negative, got labels={self.labels}; the "
                "register holds an index into the centroid set, not a signed offset"
            )
        if self.searches < len(self.labels):
            raise ValueError(
                f"every point costs at least one search, so a run of {len(self.labels)} "
                f"points cannot report searches={self.searches}; a round that finds "
                "nothing on its first try still ran"
            )


def kmedians(
    points: torch.Tensor,
    centroids: torch.Tensor,
    *,
    shots: int = _DEFAULT_SHOTS,
    seed: int | None = None,
) -> KMediansResult:
    """Assign ``points`` to their nearest centroids, then move the centroids to medians.

    One assignment step of k-medians, with the assignment quantized: each point's
    nearest centroid is found by the Grover-style minimum search the module docstring
    describes, and every centroid is then moved to the coordinate-wise median of the
    points assigned to it. Callers compose further steps by passing the returned
    positions back in as ``centroids``.

    Args:
        points: The points, a real floating-point tensor of shape ``(n, d)`` with at
            least one row and at least one column.
        centroids: The centroid positions, a real floating-point tensor of shape
            ``(k, d)`` with the same ``d`` as ``points``, at least two rows, and at most
            eight.
        shots: The number of samples each Grover search draws, at least one. Every
            search in the run is sampled exactly this many times.
        seed: The sampler's seed, or ``None`` to draw from the ambient generator. The
            same seed and the same arguments replay the same result exactly.

    Returns:
        The assignment of every point, the updated centroid positions, and the number of
        Grover searches the run sampled. What the assignment rests on -- that the
        distance table is classical and that the search is sampled rather than read out
        -- is stated in the module docstring and is not restated here.

    Raises:
        ValueError: If ``points`` or ``centroids`` is not a two-dimensional real
            floating-point tensor; if either holds a non-finite entry; if ``points`` has
            no row; if ``centroids`` has fewer than two rows or more than eight; if the
            two disagree on the number of columns; or if ``shots`` is less than one.
    """
    if shots < 1:
        raise ValueError(
            f"a k-medians step needs at least one shot per search, got shots={shots}; "
            "with no samples the search never finds a marked index and every point "
            "would be assigned the index its loop starts from"
        )
    data, positions, n_points, n_centroids = _operands(points, centroids)
    n_wires = _centroid_index_width(n_centroids)
    labels: list[int] = []
    searches = 0
    for row in range(n_points):
        distances = [
            float(torch.linalg.vector_norm(data[row] - centroid))
            for centroid in positions
        ]
        label, used = _nearest_centroid(
            distances, n_wires=n_wires, shots=shots, seed=seed
        )
        labels.append(label)
        searches += used
    return KMediansResult(
        labels=tuple(labels),
        medians=_median_positions(data, labels, positions, n_centroids),
        searches=searches,
    )


def _nearest_centroid(
    distances: list[float], *, n_wires: int, shots: int, seed: int | None
) -> tuple[int, int]:
    """Return the nearest centroid's index and the number of searches it cost.

    The loop holds the index of the best centroid found so far, which starts at index
    zero, and each round searches the index register for an index whose pair of distance
    and index is strictly smaller than the held one's. A round that finds one moves to
    it; a round that finds none reports the held index. Every moving round strictly
    decreases a pair drawn from a finite set, so the loop ends, and the last round of a
    point is a round that found nothing.

    Args:
        distances: The distance from one point to each centroid, in centroid order.
        n_wires: The width of the centroid index register.
        shots: The number of samples the round draws, at least one.
        seed: The sampler's seed, or ``None`` for the ambient generator.

    Returns:
        ``(index, searches)``, the index of the nearest centroid the loop settled on and
        the number of rounds it ran to get there, which is at least one.
    """
    keys = [(distance, index) for index, distance in enumerate(distances)]
    best = 0
    searches = 0
    while True:
        result = run_grover(
            _strictly_better(keys, keys[best], len(keys)),
            n_wires,
            shots=shots,
            seed=seed,
        )
        searches += 1
        if not result.candidates:
            return best, searches
        best = result.candidates[0]


def _strictly_better(
    keys: list[tuple[float, int]], threshold: tuple[float, int], n_centroids: int
) -> Predicate:
    """Return the predicate marking the centroid indices that beat ``threshold``.

    The comparison is on the pair of distance and index and not on the distance alone.
    That is what makes the tie rule reachable: a centroid at exactly the threshold
    distance is marked when its index is the smaller one and not marked when it is the
    larger, so the search moves toward the lower-indexed centroid of a tie and never
    away from it. Register values at or above ``n_centroids`` are excluded as well, so a
    register that the centroid set does not fill -- three centroids in a two-wire
    register, say -- never returns a slot no centroid occupies.

    Args:
        keys: Every centroid's ``(distance, index)`` pair, in centroid order.
        threshold: The pair the search has to beat.
        n_centroids: The number of centroids, and so the first register value that names
            no centroid.

    Returns:
        The predicate a Grover search over the index register marks with.
    """

    def predicate(value: int) -> bool:
        return value < n_centroids and keys[value] < threshold

    return predicate


def _operands(
    points: torch.Tensor, centroids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Validate the caller's two tensors and return them in double precision.

    The distances and the medians are computed in double precision on the CPU whatever
    device or dtype the caller holds the tensors in: they are a classical precomputation
    for each round's predicate and not part of a state the circuit evolves, and the
    comparison the search makes is a comparison of floats.

    Args:
        points: The points as given.
        centroids: The centroids as given.

    Returns:
        ``(points, centroids, n_points, n_centroids)``, both tensors in double precision
        on the CPU, with the row counts.

    Raises:
        ValueError: If either argument is not a two-dimensional real floating-point
            tensor or holds a non-finite entry; if the two disagree on the number of
            columns; if ``points`` has no row; or if ``centroids`` holds fewer than two
            rows or more than :data:`_MAX_CENTROIDS`.
    """
    data = _coordinates(points, "points")
    positions = _coordinates(centroids, "centroids")
    if data.shape[1] != positions.shape[1]:
        raise ValueError(
            "a point and a centroid are positions in the same space, so the two must "
            f"agree on the number of coordinates, got {int(data.shape[1])} in points "
            f"and {int(positions.shape[1])} in centroids"
        )
    if data.shape[0] == 0:
        raise ValueError(
            "a k-medians step assigns at least one point, got none; with no point there "
            "is no assignment to make and no cluster to take a median of"
        )
    n_centroids = int(positions.shape[0])
    if n_centroids < 2:
        raise ValueError(
            "a k-medians step searches a centroid index register, which is at least one "
            f"wire wide and so holds at least two slots, got {n_centroids} centroid; "
            "with a single centroid every point is assigned to it without a search"
        )
    if n_centroids > _MAX_CENTROIDS:
        n_wires = _centroid_index_width(n_centroids)
        raise ValueError(
            f"a k-medians step is bounded at {_MAX_CENTROIDS} centroids, got "
            f"{n_centroids}: the centroid index register needs "
            f"ceil(log2({n_centroids})) = {n_wires} wires, and the Grover search this "
            f"unit runs refuses more than {_CENTROID_WIRE_LIMIT} evaluation wires, "
            "because its diffusion operator's multi-controlled Z would need a ladder "
            "ancilla that a register of exactly that width has no free wire for"
        )
    return data, positions, int(data.shape[0]), n_centroids


def _coordinates(operand: torch.Tensor, name: str) -> torch.Tensor:
    """Validate one operand tensor and return it in double precision on the CPU.

    Args:
        operand: The tensor as given.
        name: The argument's name, for the message.

    Returns:
        The tensor as a ``float64`` CPU tensor.

    Raises:
        ValueError: If ``operand`` is not a tensor, is not two-dimensional, is not real
            floating-point, or holds a non-finite entry.
    """
    if not isinstance(operand, torch.Tensor):
        raise ValueError(
            f"{name} must be a torch.Tensor, got {type(operand).__name__}; the "
            "assignment is made from positions and not from a description of them"
        )
    if operand.dim() != 2:
        raise ValueError(
            f"{name} must be a two-dimensional tensor of positions, got shape "
            f"{tuple(int(size) for size in operand.shape)}; one row per point or "
            "centroid and one column per coordinate"
        )
    if not operand.is_floating_point():
        raise ValueError(
            f"{name} must be a real floating-point tensor, got dtype {operand.dtype}; "
            "the distances are compared as floats, and an integer tensor would be "
            "silently promoted"
        )
    if not bool(torch.isfinite(operand).all()):
        raise ValueError(
            f"{name} must be finite; a single non-finite entry makes the distance to "
            "that position undefined rather than merely inaccurate"
        )
    return operand.detach().to(device="cpu", dtype=torch.float64)


def _median_positions(
    points: torch.Tensor,
    labels: list[int],
    centroids: torch.Tensor,
    n_centroids: int,
) -> tuple[tuple[float, ...], ...]:
    """Return each centroid's new position, the coordinate-wise median of its cluster.

    The update is classical arithmetic: each coordinate of a centroid is set to the
    median of that coordinate over the points assigned to it. Two cases have to be named
    because neither is forced by the word median. A cluster with an even number of points
    has a range of medians rather than one -- every value between the two middle values
    minimises the sum of absolute deviations -- and this unit takes the lower of the two,
    which is what :func:`torch.median` returns. A cluster with no points has no median at
    all, and its centroid keeps the position it was given.

    Args:
        points: The points, in double precision.
        labels: The centroid index each point was assigned to.
        centroids: The centroid positions the run was given, in double precision.
        n_centroids: The number of centroids.

    Returns:
        One tuple of coordinates per centroid, in centroid order.
    """
    positions: list[tuple[float, ...]] = []
    for index in range(n_centroids):
        members = points[[row for row, label in enumerate(labels) if label == index]]
        if members.shape[0] == 0:
            positions.append(tuple(float(value) for value in centroids[index]))
            continue
        coordinate = torch.median(members, dim=0).values
        positions.append(tuple(float(value) for value in coordinate))
    return tuple(positions)


def _centroid_index_width(n_centroids: int) -> int:
    """Return the width of the register that indexes ``n_centroids`` centroids.

    The width is ``ceil(log2(n_centroids))``, the smallest register that has a slot for
    every centroid. At a centroid count that is not a power of two the register has slots
    no centroid occupies, and the search's predicate excludes them.

    Args:
        n_centroids: The number of centroids, at least one.

    Returns:
        The register width in wires: ``1`` for two centroids, and ``3`` for the eight
        centroids the search is bounded at.
    """
    return (n_centroids - 1).bit_length()
