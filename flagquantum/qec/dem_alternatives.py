"""Correlated mechanisms, stated as error ids.

A detector error model is a list of mechanisms that fire independently: each
carries its own probability and a detector's rate is the parity of the mechanisms
that touch it. That is all the parity matrices can say, because two columns of a
parity matrix are independent by construction. A correlated or decomposed fault
needs the opposite statement -- at most one of these fires -- and the matrices
cannot carry it, so the statement lives on the mechanisms' own records and this
module is the arithmetic that reads it.

An id groups mechanisms that are alternatives. The group is one fault, so a
detector its members touch flips when that one fault fires, which makes the
members' probabilities disjoint pieces of one shot: the rate over the group is
their sum, not their parity, and the left-over mass is the group firing none of
them. Sampling follows the same reading, drawing once per group and landing in
one member's interval or in the left-over mass. A mechanism with no id is a group
of one whose flip probability is its own probability, so the independent case is
this arithmetic's empty case rather than a second implementation of it: a model
with no ids folds in exactly the order it did before ids existed.

The membership is what the format cannot state, so an operation that assumes
independence refuses an id-carrying model rather than dropping the structure.
Those refusals live at their call sites, where the operation's own premise is the
thing to explain; what lives here is the reading they all use, :func:`stated_ids`,
which is empty exactly when the model is independent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for typing only
    from flagquantum.qec.dem import DemError

__all__ = (
    "alternative_groups",
    "fault_groups",
    "fold_marginals",
    "projected_ids",
    "require_groups_fit",
    "sample_groups",
    "stated_ids",
)


def stated_ids(errors: Sequence[DemError]) -> tuple[int, ...]:
    """Return the distinct error ids the mechanisms state, ascending.

    A model states an id only where it means to exclude something, so this is the
    set of groups the model itself named and not the groups its arithmetic uses.
    It is empty exactly when the model is independent, which is what the refusals
    at the call sites and the group arithmetic both read.
    """

    return tuple(
        sorted({error.error_id for error in errors if error.error_id is not None})
    )


def projected_ids(errors: Sequence[DemError]) -> tuple[int, ...] | None:
    """Return one error id per mechanism, or ``None`` if the model states none.

    The vector is parallel to the rates vector upstream: entry ``i`` is the id of
    mechanism ``i``, and mechanisms sharing an id are alternatives rather than
    independent faults. A mechanism that states no id is independent of every
    other, which is the same statement as an id of its own, so a model that mixes
    stated and unstated ids still has a vector here: every mechanism without one
    is given an id no other mechanism holds, numbered above the stated ids.
    ``None`` is returned only when no mechanism states an id, which is the state
    every construction route in this package produces and the state stim's text
    always describes.

    The values are opaque labels and the numbering is a normalization rather than
    part of the model, so two models that differ only in how their ids are
    numbered are the same model and their vectors differ. What is a fact about the
    model is the partition the vector induces.
    """

    if all(error.error_id is None for error in errors):
        return None
    following = max(stated_ids(errors), default=-1) + 1
    ids: list[int] = []
    for error in errors:
        if error.error_id is not None:
            ids.append(error.error_id)
            continue
        ids.append(following)
        following += 1
    return tuple(ids)


def fault_groups(errors: Sequence[DemError]) -> tuple[tuple[int, ...], ...]:
    """Return the mechanism index groups that behave as one fault.

    Every mechanism belongs to exactly one group: the mechanisms sharing an error
    id form one, and a mechanism with no id forms one of its own. The groups are
    ordered by their first index, so a fold over them visits a model's mechanisms
    in the order the model stores them and an id-free model folds in exactly the
    order it did before ids existed.
    """

    groups: dict[object, list[int]] = {}
    for index, error in enumerate(errors):
        key: object = index if error.error_id is None else ("id", error.error_id)
        groups.setdefault(key, []).append(index)
    return tuple(sorted(tuple(indices) for indices in groups.values()))


def alternative_groups(
    errors: Sequence[DemError],
) -> tuple[tuple[int, ...], ...]:
    """Return the index groups of mechanisms that exclude one another.

    Each group is the ascending indices of the mechanisms sharing one error id,
    and the groups are ordered by their first index. Only groups of more than one
    mechanism are returned, because a lone mechanism excludes nothing and
    grouping it would say otherwise. A model with no ids has no groups, which is
    what makes the independent arithmetic the empty case of this arithmetic
    rather than a separate path.
    """

    return tuple(indices for indices in fault_groups(errors) if len(indices) > 1)


def require_groups_fit(errors: Sequence[DemError]) -> None:
    """Fail unless the alternatives in every id group can exclude each other.

    The members of a group are alternatives, so a shot that lands in the group
    fires exactly one of them or none. Each member keeps the probability it
    states, so the members' probabilities are disjoint pieces of one shot and the
    left-over mass is the group firing nothing. That reading exists only while the
    pieces fit: a group whose probabilities sum above one would state that the
    group fires more than always, so it is refused rather than renormalized,
    because renormalizing would change every member's stated rate and there would
    be no value left to read back. A group that sums to exactly one is admitted --
    the group then fires every shot -- as is a group of one member, which excludes
    nothing.

    Raises:
        ValueError: A group's probabilities sum above one, naming the id and the
            sum.
    """

    totals: dict[int, float] = {}
    for error in errors:
        if error.error_id is None:
            continue
        totals[error.error_id] = totals.get(error.error_id, 0.0) + error.probability
    for error_id, total in totals.items():
        if total > 1.0:
            raise ValueError(
                f"the alternatives sharing error id {error_id} hold "
                f"probabilities summing to {total}, which states that the group "
                "fires more often than every shot; an exclusive group may hold "
                "at most one shot's worth of probability"
            )


def fold_marginals(
    errors: Sequence[DemError],
    count: int,
    select: Callable[[DemError], tuple[int, ...]],
) -> torch.Tensor:
    """Return the exact marginal flip probability of ``count`` targets.

    A target flips when an odd number of the independent faults touching it fire,
    so its rate is ``(1 - prod(1 - 2q)) / 2`` over those faults, where ``q`` is a
    fault's own flip probability. An exclusive group is one fault rather than
    several: at most one of its members fires, so the members that touch a target
    are disjoint events and the group's own flip probability is the sum of their
    probabilities, not their parity. That sum is at most one, because the model
    that holds it was required to fit one shot, so each factor is on ``[-1, 1]``
    and every rate stays on ``[0, 1]`` -- a group that fires every shot drives its
    targets to certainty rather than past it. A mechanism with no id is a group of
    one, whose flip probability is its own probability, so the independent case is
    this arithmetic rather than a second implementation of it.
    """

    rates = torch.zeros((count,), dtype=torch.float64)
    if count == 0:
        # Reachable only for ``num_observables == 0``; detectors are never zero.
        return rates
    complements = torch.ones((count,), dtype=torch.float64)
    for indices in fault_groups(errors):
        members = [errors[index] for index in indices]
        touched: list[int] = []
        for member in members:
            for index in select(member):
                if index not in touched:
                    touched.append(index)
        if len(members) == 1:
            factor = 1.0 - 2.0 * members[0].probability
            for index in touched:
                complements[index] *= factor
            continue
        for index in touched:
            group_rate = sum(
                member.probability for member in members if index in select(member)
            )
            complements[index] *= 1.0 - 2.0 * group_rate
    return (1.0 - complements) / 2.0


def sample_groups(
    errors: Sequence[DemError],
    *,
    generator: torch.Generator,
    detector_bits: torch.Tensor,
    observable_bits: torch.Tensor,
) -> None:
    """Flip the sampled signature of every fault group into the bit tensors.

    Every group is one fault, so it is drawn once: a group whose members hold
    probabilities summing to ``q`` fires with probability ``q``, and the shot
    lands in one member's interval or in the left-over mass that fires none of
    them. Membership follows the member's stated probability exactly, so a member
    fires in exactly the shots the model says it does; the exclusion shows up in
    the shots the group fires at all, which are a share of the group's mass rather
    than the larger share two independent draws would give.

    The bits are written in place because the caller owns the two tensors and a
    group's members share them. This path is reached only by a model that has a
    group of alternatives; the id-free path draws a uniform per mechanism and is
    kept as it was, because a model with no ids is the state every construction
    route produces and its draws are pinned against the circuit simulator.
    """

    shots = int(detector_bits.shape[0])
    if shots == 0:
        return
    for indices in fault_groups(errors):
        members = [errors[index] for index in indices]
        draws = torch.rand((shots,), generator=generator, dtype=torch.float64)
        firing = torch.zeros((shots,), dtype=torch.bool)
        reach = torch.zeros((shots,), dtype=torch.float64)
        for member in members:
            reach = reach + member.probability
            active = ~firing & (draws < reach)
            firing = firing | active
            bits = active.to(torch.int8)
            for index in member.detectors:
                detector_bits[:, index] ^= bits
            for index in member.observables:
                observable_bits[:, index] ^= bits
