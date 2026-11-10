"""Explicit closed-shell active-space selection with original orbital labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...core.algebra.pauli import count
from ..molecular import fold_frozen_core
from .problem import MolecularProblem


@dataclass(frozen=True, slots=True)
class ActiveSpaceResult:
    """A transformed problem and orbital indices relative to its input."""

    problem: MolecularProblem
    active_orbitals: tuple[int, ...]
    frozen_occupied: tuple[int, ...]
    discarded_virtual: tuple[int, ...]


def select_active_space(
    problem: MolecularProblem,
    *,
    active_orbitals: Sequence[int],
    frozen_occupied: Sequence[int] = (),
) -> ActiveSpaceResult:
    """Select arbitrary orbitals and freeze explicitly doubly occupied orbitals.

    Reference occupancy follows the problem's explicit occupied orbitals. Every
    occupied orbital must remain active or be explicitly frozen. Dropping a
    virtual orbital is a projection, not a full-space energy equivalence claim.
    """
    active = tuple(count(q, "active orbital") for q in active_orbitals)
    frozen = tuple(count(q, "frozen orbital") for q in frozen_occupied)
    selected = frozen + active
    if (
        not active
        or len(set(selected)) != len(selected)
        or any(q >= problem.n_orbitals for q in selected)
    ):
        raise ValueError(
            "active and frozen orbitals must be nonempty-active, distinct and in range"
        )
    occupied = set(problem.reference_occupied or ())
    if not set(frozen) <= occupied or not occupied <= set(selected):
        raise ValueError(
            "all reference occupied orbitals must be active or explicitly frozen"
        )
    # Reordering lets the existing authoritative frozen-core formula handle
    # arbitrary selections; it is not duplicated here.
    indices = torch.tensor(selected, dtype=torch.long)
    one = problem.one_body.index_select(0, indices).index_select(1, indices)
    two = problem.two_body
    for axis in range(4):
        two = two.index_select(axis, indices)
    effective, interaction, shift = fold_frozen_core(one, two, len(frozen))
    result = MolecularProblem(
        effective,
        interaction,
        problem.n_electrons - 2 * len(frozen),
        problem.nuclear_repulsion,
        problem.core_energy + shift,
        tuple(problem.orbital_labels[q] for q in active),
        reference_occupied=tuple(i for i, q in enumerate(active) if q in occupied),
    )
    discarded = tuple(q for q in range(problem.n_orbitals) if q not in selected)
    return ActiveSpaceResult(result, active, frozen, discarded)
