"""Feature selection as a QUBO: a subset's scores plus a penalty on its size.

Feature selection stated as a binary objective for an annealer is the route of Ferrari
Dacrema, Moroni, Nembrini, Ferro, Faggioli and Cremonesi, SIGIR 2022, DOI
10.1145/3477495.3531755, arXiv:2205.04346; the paper's own figure for the largest problem
it solved directly on the QPU is 124 features.

**This unit does not solve.** The repository has no annealer, and this module runs no
solver of any kind. What it does is build the objective of one feature-selection instance
and evaluate that objective at an assignment a caller supplies. Which subset a solver
returns, and at what cost, belongs to the solver the problem is handed to: any advantage
such a solver observes is the solver's, and this construction carries none of its own.
The position is :mod:`flagquantum.algorithms.qubo`'s -- a polynomial classical
transformation of an objective into the form a solver consumes, which yields no speedup
of its own.

**The objective.** With ``r_i`` the relevance of feature ``i`` and ``s_ij`` the
redundancy of the pair ``(i, j)``, a subset ``S`` of the features is scored

    F(S) = - sum_{i in S} r_i + sum_{i < j, both in S} s_ij
           + penalty * (|S| - n_selected) ** 2.

The first term lowers the objective as a feature's relevance rises, the second raises it
as a pair's redundancy rises, and the third prices a subset that is not of the target
size. Both scores are the caller's data: this module defines no relevance measure and no
redundancy measure, and puts no interpretation on either.

**The penalty weight is the caller's decision, and the module supplies no default.** A
weight small enough against the scores can leave a subset of another size cheapest, and
this module computes no weight at which that stops being so: how far the size term should
outweigh the scores is the caller's choice rather than this module's, and no weight is
chosen or predicted here. The weight is required to be positive and finite, because the
term it multiplies exists to price a subset's size.

**What the module hands over.** :func:`feature_selection_qubo` returns a
:class:`FeatureSelectionProblem`, whose :attr:`~FeatureSelectionProblem.qubo` is a
:class:`~flagquantum.algorithms.qubo.QuboProblem` for ``F``, with the squared size term
expanded into the coefficient maps. :meth:`FeatureSelectionProblem.energy` evaluates that
objective at an assignment through
:func:`~flagquantum.algorithms.qubo.qubo_energy`, and
:meth:`FeatureSelectionProblem.to_ising` maps it to a spin Hamiltonian through
:func:`~flagquantum.algorithms.qubo.qubo_to_ising`. The evaluation and the mapping are
those functions'; this module adds neither.

This unit is demonstration scale. It makes no performance, capacity, or hardware claim,
and it does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .core import Hamiltonian
from .qubo import QuboProblem, qubo_energy, qubo_to_ising

__all__ = [
    "FeatureSelectionProblem",
    "feature_selection_qubo",
]


@dataclass(frozen=True, kw_only=True)
class FeatureSelectionProblem:
    """One feature-selection instance, carried as the binary objective it scores.

    The objective is ``F`` of the module docstring, carried as a
    :class:`~flagquantum.algorithms.qubo.QuboProblem`: a feature's relevance and a pair's
    redundancy sit in that problem's linear and quadratic coefficients, and the size
    penalty is expanded into both of those maps and the offset. The feature count is the
    problem's variable count.

    The target size and the penalty weight are carried here rather than read back off the
    problem. A ``QuboProblem`` holds a variable count, a linear map, a quadratic map and
    an offset, and neither of these two is one of those; and both enter those maps as
    sums the caller's scores share, the weight joining the relevance in the linear map
    and the pairs' redundancies in the quadratic one.

    ``kw_only`` is not optional here, and the reason is how a call site reads rather than
    what it would catch: the fields are a problem object, a count and a weight, so a
    positional spelling would read as one object followed by two bare numbers of
    unrelated kind. A transposition of those two numbers is not what ``kw_only``
    protects against either: where a transposed spelling is refused, it is this class's
    own reading of the fields that refuses it, and a target size must be an integer where
    a weight need not be, so a weight handed in as a fraction cannot be read as a size.
    Measured, a transposed spelling constructs exactly where its weight was an integer no
    larger than the feature count and its target size was positive; the two fields then
    hold each other's values.

    Attributes:
        qubo: The binary objective. ``linear[i]`` is feature ``i``'s coefficient,
            ``quadratic[(i, j)]`` is the coefficient of the pair with the lower index
            first, and ``offset`` is the constant the size penalty contributes.
        n_selected: The target number of features, an integer in
            ``range(qubo.n_variables + 1)``.
        penalty: The weight the objective puts on the target size, positive and finite.
    """

    qubo: QuboProblem
    n_selected: int
    penalty: float

    def __post_init__(self) -> None:
        """Reject an instance whose fields do not describe one problem.

        Raises:
            ValueError: If ``qubo`` is not a :class:`QuboProblem`; if ``n_selected`` is
                not an integer in ``range(qubo.n_variables + 1)``; or if ``penalty`` is
                not positive and finite.
        """
        if not isinstance(self.qubo, QuboProblem):
            raise ValueError(
                "a feature-selection problem is carried as a QuboProblem, got "
                f"{type(self.qubo).__name__}; the objective this class evaluates and "
                "maps is that problem's"
            )
        _validated_target(self.n_selected, self.qubo.n_variables)
        _validated_penalty(self.penalty)

    def energy(self, assignment: Sequence[int]) -> float:
        """Return the objective value of ``assignment``.

        The value is :func:`~flagquantum.algorithms.qubo.qubo_energy` applied to the
        problem this instance carries, so the offset the size penalty contributes is part
        of it.

        Args:
            assignment: One binary value per feature, in feature order, where
                ``assignment[i]`` is 1 when feature ``i`` is in the subset.

        Returns:
            The objective value of that subset.

        Raises:
            ValueError: If the assignment does not carry one binary value per feature.
        """
        return qubo_energy(self.qubo, assignment)

    def to_ising(self) -> Hamiltonian:
        """Return the spin form of the same objective.

        The mapping is :func:`~flagquantum.algorithms.qubo.qubo_to_ising`, applied to the
        problem this instance carries. The spin form is what a solver of that kind
        consumes; this module neither consumes it nor runs one.

        Returns:
            A Hamiltonian that matches the objective this instance evaluates, on the
            assignment each spin configuration encodes.
        """
        return qubo_to_ising(self.qubo)


def feature_selection_qubo(
    relevance: torch.Tensor,
    *,
    n_selected: int,
    penalty: float,
    redundancy: torch.Tensor | None = None,
) -> FeatureSelectionProblem:
    """Build the binary objective of one feature-selection instance.

    The objective is ``F`` of the module docstring. Its size penalty is a square in the
    subset's cardinality, and for a binary variable ``x_i ** 2 = x_i``, so the square
    expands into terms the coefficient maps can carry:

        penalty * (sum_i x_i - n_selected) ** 2
            = penalty * (1 - 2 * n_selected) * sum_i x_i
            + 2 * penalty * sum_{i < j} x_i x_j
            + penalty * n_selected ** 2.

    Feature ``i``'s linear coefficient is therefore ``-relevance[i] + penalty * (1 - 2 *
    n_selected)``, the pair ``(i, j)``'s quadratic coefficient is ``redundancy[i][j] + 2 *
    penalty``, and the offset is ``penalty * n_selected ** 2``. The squared term's
    diagonal is where a binary variable's own square goes, which is why the penalty
    reaches the linear map at all as well as the quadratic one.

    Args:
        relevance: The relevance of each feature, a real floating-point tensor of ``m``
            entries, one per feature; a higher value is a more relevant feature. The
            values are used as given and are not normalized.
        n_selected: The target number of features, an integer in ``range(m + 1)``.
        penalty: The weight on the target size, positive and finite. There is no default;
            see the module docstring for whose choice it is.
        redundancy: The pairwise redundancy, a real floating-point tensor of shape
            ``(m, m)`` whose entry ``(i, j)`` is the redundancy of features ``i`` and
            ``j``; a higher value is a more redundant pair. The diagonal is not read --
            a pair is two features -- and the two triangles must agree exactly, because
            ``(i, j)`` and ``(j, i)`` are one pair with one score. ``None`` means no pair
            carries a redundancy, so the quadratic terms are the size penalty's alone.

    Returns:
        The instance, carrying the problem it built and the two parameters of the
        instance that the problem itself does not name.

    Raises:
        ValueError: If ``relevance`` is not a one-dimensional real floating-point tensor,
            holds no entry, or holds a non-finite entry; if ``redundancy`` is not a tensor
            of shape ``(m, m)``, holds a non-finite entry, or is not symmetric; if
            ``n_selected`` is not an integer in ``range(m + 1)``; or if ``penalty`` is not
            positive and finite.
    """
    scores = _validated_relevance(relevance)
    pairs = _validated_redundancy(redundancy, len(scores))
    target = _validated_target(n_selected, len(scores))
    weight = _validated_penalty(penalty)

    # The size penalty is expanded here rather than left in its squared form, because the
    # QUBO carries a linear map and a quadratic map and no place for a cardinality term.
    n_features = len(scores)
    linear = {
        index: -scores[index] + weight * (1 - 2 * target) for index in range(n_features)
    }
    quadratic = {
        (first, second): pairs[first][second] + 2.0 * weight
        for first in range(n_features)
        for second in range(first + 1, n_features)
    }
    qubo = QuboProblem(
        n_variables=n_features,
        linear=linear,
        quadratic=quadratic,
        offset=weight * target * target,
    )
    return FeatureSelectionProblem(qubo=qubo, n_selected=target, penalty=weight)


def _validated_relevance(relevance: torch.Tensor) -> tuple[float, ...]:
    """Validate the relevance vector and return it as Python floats.

    Args:
        relevance: The candidate relevance of each feature.

    Returns:
        One score per feature, in feature order.

    Raises:
        ValueError: If ``relevance`` is not a one-dimensional real floating-point tensor,
            holds no entry, or holds a non-finite entry.
    """
    if not isinstance(relevance, torch.Tensor):
        raise ValueError(
            f"the relevance must be a torch.Tensor, got {type(relevance).__name__}; a "
            "feature is scored by a number, and a description of the scores is not one"
        )
    if relevance.dim() != 1:
        raise ValueError(
            "the relevance must be a one-dimensional tensor, one entry per feature, got "
            f"shape {tuple(int(size) for size in relevance.shape)}"
        )
    if not relevance.is_floating_point():
        raise ValueError(
            "the relevance must be a real floating-point tensor, got dtype "
            f"{relevance.dtype}; a score is a real number, and an integer tensor would "
            "be silently promoted"
        )
    scores = tuple(
        float(value)
        for value in relevance.detach().to(device="cpu", dtype=torch.float64)
    )
    if not scores:
        raise ValueError(
            "a feature-selection instance scores at least one feature, got none; with no "
            "feature there is no subset to choose"
        )
    if not all(math.isfinite(value) for value in scores):
        raise ValueError(
            "every relevance must be finite, and one is not; a non-finite score makes "
            "the objective's value at a subset holding that feature undefined"
        )
    return scores


def _validated_redundancy(
    redundancy: torch.Tensor | None, n_features: int
) -> tuple[tuple[float, ...], ...]:
    """Validate the pairwise redundancy and return it as a square of Python floats.

    ``None`` is a redundancy of zero for every pair, which leaves the objective's
    quadratic terms to the size penalty alone.

    Args:
        redundancy: The candidate pairwise redundancy, or ``None``.
        n_features: The number of features the matrix must span.

    Returns:
        ``n_features`` rows of ``n_features`` values each.

    Raises:
        ValueError: If ``redundancy`` is not a tensor of shape
            ``(n_features, n_features)``, holds a non-finite entry, or is not symmetric.
    """
    if redundancy is None:
        return tuple(tuple(0.0 for _ in range(n_features)) for _ in range(n_features))
    if not isinstance(redundancy, torch.Tensor):
        raise ValueError(
            f"the redundancy must be a torch.Tensor or None, got "
            f"{type(redundancy).__name__}; a pair is scored by a number, and a "
            "description of the scores is not one"
        )
    shape = tuple(int(size) for size in redundancy.shape)
    if redundancy.dim() != 2 or shape != (n_features, n_features):
        raise ValueError(
            "the redundancy must be a two-dimensional tensor with one row and one "
            f"column per feature, so of shape ({n_features}, {n_features}), got shape "
            f"{shape}"
        )
    if not redundancy.is_floating_point():
        raise ValueError(
            "the redundancy must be a real floating-point tensor, got dtype "
            f"{redundancy.dtype}; a score is a real number, and an integer tensor would "
            "be silently promoted"
        )
    matrix = tuple(
        tuple(float(value) for value in row)
        for row in redundancy.detach().to(device="cpu", dtype=torch.float64)
    )
    for row in matrix:
        if not all(math.isfinite(value) for value in row):
            raise ValueError(
                "every redundancy must be finite, and one entry is not; a non-finite "
                "redundancy makes the objective's value at a subset holding that pair "
                "undefined"
            )
    for first in range(n_features):
        for second in range(first + 1, n_features):
            if matrix[first][second] != matrix[second][first]:
                raise ValueError(
                    "a pair of features has one redundancy, so the entry above the "
                    "diagonal and the one below it must be the same number, got "
                    f"{matrix[first][second]} at ({first}, {second}) against "
                    f"{matrix[second][first]} at ({second}, {first}); symmetrize the "
                    "matrix if the two triangles differ"
                )
    return matrix


def _validated_target(n_selected: object, n_features: int) -> int:
    """Validate the target number of features.

    Args:
        n_selected: The candidate target size.
        n_features: The number of features it is a size of.

    Returns:
        The target size.

    Raises:
        ValueError: If it is not an integer in ``range(n_features + 1)``.
    """
    if isinstance(n_selected, bool) or not isinstance(n_selected, int):
        raise ValueError(
            f"the target number of features must be an integer, got {n_selected!r}; a "
            "size of a subset is a count, and a float or a flag is not one"
        )
    if not 0 <= n_selected <= n_features:
        raise ValueError(
            f"the target number of features must be a size in range({n_features + 1}), "
            f"got {n_selected}; a subset of {n_features} features cannot have that size, "
            "and no subset would then be at the size the penalty prices"
        )
    return n_selected


def _validated_penalty(penalty: object) -> float:
    """Validate the weight on the target size.

    Args:
        penalty: The candidate weight.

    Returns:
        The weight as a Python float.

    Raises:
        ValueError: If it is not a positive and finite real number.
    """
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
        raise ValueError(
            f"the penalty weight must be a real number, got {penalty!r}; a weight is a "
            "number, and a flag or a tensor is not one"
        )
    weight = float(penalty)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            f"the penalty weight must be positive and finite, got {weight}; the term it "
            "weights is what prices a subset that is not of the target size, and a "
            "non-positive weight is not a price"
        )
    return weight
