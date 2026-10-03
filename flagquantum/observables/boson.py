"""Bosonic operator algebra and its dense value on a truncated Fock space.

An operator here is stored normal-ordered: every creation operator precedes every
annihilation operator and each group is listed in ascending degree.  The canonical
commutation relation ``[a[p], ad[q]] = delta(p, q)`` is applied at construction, so
``a[p] ad[p]`` is rewritten as ``ad[p] a[p] + 1`` before any caller sees it.  Two
consequences are load-bearing, exactly as they are for the fermionic algebra.  Equality is
structural, so a Hermiticity test is exact rather than a tolerance.  And the stored
monomials are the Poincare-Birkhoff-Witt basis of the oscillator algebra, so two
mathematically equal operators this module can build hold equal terms.

Two differences from the fermionic algebra decide the shape of this module.  Two factors
of the same kind and degree do not vanish, they are powers, so ``ad[k] ad[k]`` is the
repeated degree ``(k, k)`` and a monomial has no bounded length.  And the algebra
therefore has no finite matrix representation of its own, because every creation operator
raises the occupation without limit; a numerical value needs the truncation dimension of
each degree, which :meth:`BosonOperator.to_matrix` takes as a parameter rather than
assuming.  The truncation is not a detail.  The commutation relations are applied here to
untruncated monomials, so ``[a, ad]`` is the identity of the algebra, while the commutator
of two ladder matrices truncated to the same level count differs from the identity at the
top level of every degree.  Both are true of different operators, and only an explicit
dimension lets a caller say which one the numbers describe.

``BosonOperator()`` with no terms is the zero operator and `identity` is the unit; both
are ordinary values of the algebra rather than separate constructors in the reference
arithmetic, because the zero is the empty sum and the unit is the empty monomial.

The dense form places the lowest degree on the most significant index, which is the order
the package gives a wire inside an amplitude, so a Kronecker product over ascending
degrees is the whole of the construction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Complex

import torch

from . import _wire, _wires

_DEFAULT_TRUNCATION_DIMENSION = 2
"""The Fock truncation a degree gets when a caller declares no space for it."""

DEFAULT_DENSE_MATRIX_BYTES = 16 * 1024 * 1024
"""Byte ceiling above which :meth:`BosonOperator.to_matrix` refuses to allocate.

The space is the product of the truncation dimensions and the matrix is its square, so
the entry count grows with the square of that product.  Two levels per degree admits ten
degrees and refuses the eleventh, which keeps the dense form a reference and a test
oracle rather than a production path.
"""

# One factor of a monomial reads as (degree, is_creation), which is the pair the fermionic
# rewrite is built on as well; the two rewrites differ in their rule, not in their input.
_Factor = tuple[int, bool]
_Monomial = tuple[_Factor, ...]
_MonomialKey = tuple[tuple[int, ...], tuple[int, ...]]


def _complex_scalar(value: object) -> complex | None:
    if isinstance(value, bool) or not isinstance(value, Complex):
        return None
    result = complex(value)
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError("bosonic operator coefficients must be finite")
    return result


def _degree(value: object) -> int:
    """Read one degree index by the same protocol the package reads a wire label with."""

    return _wire("bosonic", value, noun="degree")


def _degrees(value: Iterable[int] | int, *, owner: str) -> tuple[int, ...]:
    """Read one group of degrees and require them to be in canonical order.

    A repeated degree is a power rather than an error, which is the one respect in which
    this reader differs from the fermionic one: ``ad[k] ad[k]`` is ``(ad[k])**2`` and does
    not vanish, so the group is a multiset.  Ascending order is still required, because
    that is what makes the stored monomial the one form of its value.
    """

    degrees = _wires(value, owner=owner, noun="degree", unique=False)
    if degrees != tuple(sorted(degrees)):
        raise ValueError(
            f"{owner} degrees must be listed in ascending order; a canonical monomial "
            "orders creation operators before annihilation operators and each group by "
            "degree, and a repeated degree is the power of that operator"
        )
    return degrees


def _first_defect(monomial: _Monomial) -> int | None:
    """Return the position of the first adjacent pair that is out of canonical order.

    The canonical order is a total order on a factor: a creation operator precedes an
    annihilation operator, and within one kind the lower degree precedes the higher.  A
    pair is a defect when it is adjacent and the left factor stands after the right one
    in that order, which is either an annihilation before a creation or two factors of
    one kind in descending degree.
    """

    for position in range(len(monomial) - 1):
        left, right = monomial[position], monomial[position + 1]
        if left[1] != right[1]:
            if not left[1]:
                return position
            continue
        if left[0] > right[0]:
            return position
    return None


def _canonical_terms(monomial: _Monomial) -> dict[_MonomialKey, complex]:
    """Rewrite one product of factors as a sum of normal-ordered monomials.

    The rewrite resolves the leftmost defect until none is left.  A defect whose two
    factors are of one kind, or are an annihilation and a creation of two different
    degrees, is a free exchange: the factors commute and the pair is stored reversed at
    no cost.  An adjacent ``a[p] ad[p]`` is where the canonical commutation relation
    ``a[p] ad[p] = ad[p] a[p] + 1`` applies, and that step splits in two: the
    contraction, a monomial with the pair left out, and the exchanged pair.

    Termination is by the pair ``(defects, length)`` read lexicographically, which never
    rises.  Exchanging an adjacent out-of-order pair lowers the defect count by exactly
    one, as one step of bubble sort does.  Leaving the pair out lowers the length and adds
    no defect after it, because every factor that preceded the pair still precedes every
    factor that followed it.
    """

    pending: list[tuple[_Monomial, complex]] = [(monomial, 1.0 + 0.0j)]
    resolved: dict[_MonomialKey, complex] = {}
    while pending:
        current, coefficient = pending.pop()
        position = _first_defect(current)
        if position is None:
            creations = tuple(degree for degree, is_creation in current if is_creation)
            annihilations = tuple(
                degree for degree, is_creation in current if not is_creation
            )
            key = (creations, annihilations)
            resolved[key] = resolved.get(key, 0.0j) + coefficient
            continue
        left, right = current[position], current[position + 1]
        head, tail = current[:position], current[position + 2 :]
        if not left[1] and right[1] and left[0] == right[0]:
            pending.append((head + tail, coefficient))
        pending.append((head + (right, left) + tail, coefficient))
    return resolved


def _factor_sequence(term: BosonTerm) -> _Monomial:
    """Read a canonical term back as the flat left-to-right factor sequence."""

    creations = tuple((degree, True) for degree in term.creations)
    annihilations = tuple((degree, False) for degree in term.annihilations)
    return creations + annihilations


@dataclass(frozen=True, slots=True)
class BosonTerm:
    """One normal-ordered bosonic monomial with a complex coefficient.

    The operator is ``coefficient * ad[creations[0]] ... a[annihilations[-1]]``: creation
    operators first and each group in ascending degree, where a degree that occurs twice
    is that power of the operator.  Both groups are validated to be in that order, which
    is what makes the representation canonical -- two equal monomials hold equal
    ``creations`` and equal ``annihilations``, with no rewrite step to run first.
    """

    coefficient: complex
    creations: tuple[int, ...] = ()
    annihilations: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        coefficient = _complex_scalar(self.coefficient)
        if coefficient is None:
            raise TypeError("a bosonic term requires a numeric coefficient")
        object.__setattr__(self, "coefficient", coefficient)
        for attribute, owner in (
            ("creations", "bosonic creation"),
            ("annihilations", "bosonic annihilation"),
        ):
            object.__setattr__(
                self,
                attribute,
                _degrees(getattr(self, attribute), owner=owner),
            )

    @property
    def degrees(self) -> tuple[int, ...]:
        """Return every distinct degree this monomial acts on, in ascending order."""

        return tuple(sorted(set(self.creations) | set(self.annihilations)))

    def dagger(self) -> BosonTerm:
        """Return the adjoint, which exchanges the two groups.

        The factors commute, so reversing the product costs no sign and restoring a group
        to ascending degree is free.  The adjoint of the basis monomial
        ``(ad)**p (a)**q`` is therefore ``(ad)**q (a)**p``, which is already
        normal-ordered, so the exchange is the whole of the operation.
        """

        return BosonTerm(
            self.coefficient.conjugate(), self.annihilations, self.creations
        )


@dataclass(frozen=True, slots=True)
class BosonOperator:
    """An immutable complex linear combination of normal-ordered monomials.

    Terms that share a monomial are combined and a term whose coefficient is exactly zero
    is dropped, so the stored sum is canonical and comparison is exact.  No threshold is
    applied: a small coefficient is kept rather than quietly rounded away.

    Examples:
        >>> from flagquantum.observables.boson import annihilate, create, identity
        >>> commutation = annihilate(0) * create(0) - create(0) * annihilate(0)
        >>> commutation == identity()
        True
        >>> create(0).to_matrix({0: 3}).diagonal(-1).tolist()
        [(1+0j), (1.4142135623730951+0j)]
    """

    terms: tuple[BosonTerm, ...] = ()

    def __post_init__(self) -> None:
        combined: dict[_MonomialKey, complex] = {}
        for term in self.terms:
            if not isinstance(term, BosonTerm):
                raise TypeError(
                    "a bosonic operator holds BosonTerm values, got "
                    f"{type(term).__name__}"
                )
            key = (term.creations, term.annihilations)
            combined[key] = combined.get(key, 0.0j) + term.coefficient
        object.__setattr__(
            self,
            "terms",
            tuple(
                BosonTerm(coefficient, creations, annihilations)
                for (creations, annihilations), coefficient in sorted(combined.items())
                if coefficient != 0.0j
            ),
        )

    def __add__(self, other: object) -> BosonOperator:
        if not isinstance(other, BosonOperator):
            return NotImplemented
        return BosonOperator((*self.terms, *other.terms))

    def __sub__(self, other: object) -> BosonOperator:
        if not isinstance(other, BosonOperator):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> BosonOperator:
        return BosonOperator(
            tuple(
                BosonTerm(-term.coefficient, term.creations, term.annihilations)
                for term in self.terms
            )
        )

    def __mul__(self, other: object) -> BosonOperator:
        """Multiply by a scalar or compose with another operator.

        Both cases are the same product in the algebra: a scalar is its degree-zero
        element, and composition concatenates the factors of the two monomials and
        reapplies the commutation relations.
        """

        if isinstance(other, BosonOperator):
            return BosonOperator(
                tuple(
                    composed
                    for left in self.terms
                    for right in other.terms
                    for composed in _compose(left, right)
                )
            )
        value = _complex_scalar(other)
        if value is None:
            return NotImplemented
        return BosonOperator(
            tuple(
                BosonTerm(value * term.coefficient, term.creations, term.annihilations)
                for term in self.terms
            )
        )

    def __rmul__(self, other: object) -> BosonOperator:
        return self * other

    @property
    def degrees(self) -> tuple[int, ...]:
        """Return every distinct degree this operator acts on, in ascending order."""

        return tuple(
            sorted({degree for term in self.terms for degree in term.degrees})
        )

    def dagger(self) -> BosonOperator:
        """Return the adjoint, which exchanges the two groups of every monomial."""

        return BosonOperator(tuple(term.dagger() for term in self.terms))

    def is_hermitian(self) -> bool:
        """Report whether this operator equals its own adjoint.

        The canonical form makes this an exact structural test rather than a tolerance.
        ``position`` and ``momentum`` are Hermitian by this test on the algebra, before
        any truncation is chosen.
        """

        return self == self.dagger()

    def commutator(self, other: BosonOperator) -> BosonOperator:
        """Return ``self other - other self``."""

        if not isinstance(other, BosonOperator):
            raise TypeError("a commutator requires two bosonic operators")
        return self * other - other * self

    def anticommutator(self, other: BosonOperator) -> BosonOperator:
        """Return ``self other + other self``."""

        if not isinstance(other, BosonOperator):
            raise TypeError("an anticommutator requires two bosonic operators")
        return self * other + other * self

    def to_matrix(self, dimensions: Mapping[int, int] | None = None) -> torch.Tensor:
        """Return the dense matrix of this operator on a truncated Fock space.

        ``dimensions`` declares the space: it maps a degree to the number of levels kept
        for it, every degree this operator acts on must appear in it, and a degree it does
        not act on becomes an identity factor, so the returned matrix is always the whole
        space the caller declared.  ``None`` reads as the degrees this operator acts on,
        each keeping :data:`_DEFAULT_TRUNCATION_DIMENSION` levels.

        The lowest degree is placed on the most significant index, which is the order the
        package gives a wire inside an amplitude.  The commutation relations are applied
        before this evaluation rather than after it, so the value returned here is the
        truncation of the operator on the untruncated space.  The commutator of the
        truncation is a different matrix and differs at the top level of every degree: on
        four levels ``[a, ad]`` evaluates to the identity, while the commutator of the two
        four-level ladder matrices has ``1 - 4 = -3`` in its last diagonal entry.  Both
        statements are true of different operators, and the way to relate them is to build
        one level more and truncate the result, which is what the reference implementation
        does to state the relation on a bounded space.

        This is a reference and a test oracle rather than a production path: the space is
        the product of the truncation dimensions and the matrix is its square, so ten
        degrees at the default truncation already reach the byte ceiling.

        Raises:
            ValueError: If an acted-on degree has no declared dimension, if a truncation
                keeps no level, or if the materialized matrix would exceed
                :data:`DEFAULT_DENSE_MATRIX_BYTES`.
        """

        ladders, side = _truncation(dimensions, self.degrees)
        entries = side * side
        if entries * torch.complex128.itemsize > DEFAULT_DENSE_MATRIX_BYTES:
            raise ValueError(
                f"a bosonic operator on {side} levels is {side} x {side} with "
                f"{entries} entries ({entries * torch.complex128.itemsize} bytes), above "
                f"the {DEFAULT_DENSE_MATRIX_BYTES}-byte ceiling; truncate the space"
            )
        matrix = torch.zeros((side, side), dtype=torch.complex128)
        for term in self.terms:
            matrix = matrix + _term_matrix(term, ladders)
        return matrix


def _compose(left: BosonTerm, right: BosonTerm) -> tuple[BosonTerm, ...]:
    """Return the canonical expansion of the product of two canonical terms."""

    scale = left.coefficient * right.coefficient
    return tuple(
        BosonTerm(scale * coefficient, creations, annihilations)
        for (creations, annihilations), coefficient in _canonical_terms(
            _factor_sequence(left) + _factor_sequence(right)
        ).items()
    )


def _truncation(
    dimensions: Mapping[int, int] | None, acted_on: tuple[int, ...]
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], int]:
    """Read the space to evaluate on as the ladder pair of every degree, and its size."""

    levels: dict[int, int] = {}
    if dimensions is None:
        levels = dict.fromkeys(acted_on, _DEFAULT_TRUNCATION_DIMENSION)
    else:
        if not isinstance(dimensions, Mapping):
            raise TypeError(
                "a bosonic truncation must map a degree to a level count, got "
                f"{type(dimensions).__name__}"
            )
        for degree, dimension in dimensions.items():
            levels[_degree(degree)] = _levels(degree, dimension)
        missing = sorted(set(acted_on) - set(levels))
        if missing:
            raise ValueError(
                f"this operator acts on degree(s) {missing} and the truncation declares "
                "no dimension for them; a dimension is not defaulted here, because a "
                "silent truncation is the failure this parameter exists to name"
            )
    side = 1
    for dimension in levels.values():
        side *= dimension
    return (
        {degree: _ladder_matrices(dimension) for degree, dimension in levels.items()},
        side,
    )


def _levels(degree: object, dimension: object) -> int:
    """Read one level count, which keeps at least the vacuum level."""

    levels = _wire(f"bosonic degree {degree}", dimension, noun="truncation")
    if levels < 1:
        raise ValueError(
            f"bosonic degree {degree} truncation must keep at least one level, got "
            f"{levels}"
        )
    return levels


def _ladder_matrices(dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(annihilate, create)`` on ``dimension`` levels of one degree."""

    ladder = torch.sqrt(torch.arange(1, dimension, dtype=torch.float64)).to(
        torch.complex128
    )
    # create[n, n - 1] = sqrt(n) and annihilate[n - 1, n] = sqrt(n) are the standard
    # oscillator action, with the top level annihilating to nothing because the space
    # keeps no level above it.
    return torch.diag(ladder, 1), torch.diag(ladder, -1)


def _term_matrix(
    term: BosonTerm, ladders: dict[int, tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """Return the dense matrix of one monomial, coefficient included."""

    matrix = torch.ones((1, 1), dtype=torch.complex128)
    for degree, (annihilate, create) in sorted(ladders.items()):
        matrix = torch.kron(
            matrix,
            torch.linalg.matrix_power(create, term.creations.count(degree))
            @ torch.linalg.matrix_power(annihilate, term.annihilations.count(degree)),
        )
    return term.coefficient * matrix


def identity(degree: int | None = None) -> BosonOperator:
    """Return the unit of the algebra, which is the empty monomial.

    ``degree`` may be supplied for readable notation but does not change the value, the
    way `flagquantum.observables.I` treats a wire label.
    """

    if degree is not None:
        _degree(degree)
    return BosonOperator((BosonTerm(1.0),))


def create(degree: int) -> BosonOperator:
    """Return the creation operator ``ad[degree]``."""

    mode = _degree(degree)
    return BosonOperator((BosonTerm(1.0, (mode,), ()),))


def annihilate(degree: int) -> BosonOperator:
    """Return the annihilation operator ``a[degree]``."""

    mode = _degree(degree)
    return BosonOperator((BosonTerm(1.0, (), (mode,)),))


def number(degree: int) -> BosonOperator:
    """Return the level-number operator ``ad[degree] a[degree]``."""

    mode = _degree(degree)
    return BosonOperator((BosonTerm(1.0, (mode,), (mode,)),))


def position(degree: int) -> BosonOperator:
    """Return the position operator ``(a + ad) / 2`` of one degree.

    Examples:
        >>> from flagquantum.observables.boson import annihilate, create, position
        >>> position(0) == 0.5 * (create(0) + annihilate(0))
        True
        >>> position(0).is_hermitian()
        True
    """

    return 0.5 * (create(degree) + annihilate(degree))


def momentum(degree: int) -> BosonOperator:
    """Return the momentum operator ``1j (ad - a) / 2`` of one degree."""

    return 0.5j * (create(degree) - annihilate(degree))


__all__ = (
    "BosonOperator",
    "BosonTerm",
    "DEFAULT_DENSE_MATRIX_BYTES",
    "annihilate",
    "create",
    "identity",
    "momentum",
    "number",
    "position",
)
