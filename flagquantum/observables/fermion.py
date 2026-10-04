"""Fermionic operator algebra and its Pauli images as observable values.

An operator here is stored normal-ordered with respect to the vacuum, so the canonical
anticommutation relations are applied at construction and every operator this module
returns is a canonical sum: ``c[p] c†[q]`` is rewritten as ``delta(p, q) - c†[q] c[p]``
before any caller sees it.  Two consequences are load-bearing.  Equality is structural,
so a symbolic Hermiticity test is exact rather than a tolerance.  And a fermion-to-qubit
transform's image of a Hermitian operator is an `Observable`, which the existing execution
path already measures, so no runtime needs a fermionic special case.

Two transforms ship, `jordan_wigner` and `parity_encoding`, which are the same operator
written in two qubit bases.  They share one assembly path and differ only in the string a
ladder operator's image sits on, which is what makes a transform a value the boundary
takes rather than a copy of the boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Complex

from ..errors import CapabilityError
from . import Observable, _PauliTerm, _qubit, _qubits

# sigma_minus = (X + iY) / 2 annihilates and sigma_plus = (X - iY) / 2 creates in the
# occupation basis where |1> is occupied.  A factor on mode k is the Pauli string
# Z[0] ... Z[k-1] sigma[k]: the Z string runs over the modes strictly below k, so with mode
# k on wire k and wire 0 the most significant amplitude bit -- the package's own wire order
# -- the mode index is also the wire index and the string is the modes that precede k.
# Placing the string over the modes above k instead is the same transformation composed
# with a reversal of the mode order, so this module fixes one convention and states it
# rather than leaving the choice to each reader.
_ANNIHILATION_IMAGE = ((0.5 + 0.0j, "x"), (0.5j, "y"))
_CREATION_IMAGE = ((0.5 + 0.0j, "x"), (-0.5j, "y"))

# The Pauli group is projective: the product of two different axes is the third one times
# a fourth root of unity.  Same-axis products are the identity and carry no phase.
_PAULI_PRODUCT = {
    ("x", "y"): ("z", 1j),
    ("y", "x"): ("z", -1j),
    ("y", "z"): ("x", 1j),
    ("z", "y"): ("x", -1j),
    ("z", "x"): ("y", 1j),
    ("x", "z"): ("y", -1j),
}

# A structurally Hermitian fermionic operator has a real-coefficient Pauli image, and the
# images here use dyadic factors, so the imaginary part of every collected coefficient was
# exactly zero over 4000 randomized Hermitian trials.  This bound is therefore a
# fail-closed net rather than a tuned threshold: no input this module's own algebra can
# build needs it, and a coefficient whose imaginary part exceeded it would be reported
# instead of having that part silently dropped.
_IMAGINARY_TOLERANCE = 1e-9

PauliFactors = tuple[tuple[int, str], ...]
_PauliSum = dict[PauliFactors, complex]
_Monomial = tuple[tuple[int, bool], ...]


def _complex_scalar(value: object) -> complex | None:
    if isinstance(value, bool) or not isinstance(value, Complex):
        return None
    result = complex(value)
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError("fermionic operator coefficients must be finite")
    return result


def _degree(value: object) -> int:
    """Read one mode index by the same protocol the package reads a wire label with."""

    return _qubit("fermionic", value, noun="degree")


def _degrees(value: Iterable[int] | int, *, owner: str) -> tuple[int, ...]:
    """Read one group of mode indices and require them to be in canonical order."""

    degrees = _qubits(value, owner=owner, noun="degree")
    if degrees != tuple(sorted(degrees)):
        raise ValueError(
            f"{owner} degrees must be listed in ascending order; a canonical monomial "
            "orders creation operators before annihilation operators and each group by "
            "degree, and the anticommutation sign depends on that order"
        )
    return degrees


def _order_key(factor: tuple[int, bool]) -> tuple[int, int]:
    """Rank one factor: creation before annihilation, ascending degree inside a group."""

    return (0, factor[0]) if factor[1] else (1, factor[0])


def _first_defect(monomial: _Monomial) -> tuple[str, int] | None:
    """Return the first adjacent pair that is out of order or that Pauli exclusion kills.

    ``c[k] c[k] = 0`` and ``c†[k] c†[k] = 0`` hold for two factors that end up next to each
    other, and normal ordering brings them together: ``c†[k] c[k] c†[k]`` contracts to
    ``c†[k]`` rather than to zero, because the contraction pairs the middle ``c[k]`` with
    the following ``c†[k]`` first.  Testing the whole word for a repeated degree instead
    would wrongly reject that product, so only adjacency decides.
    """

    for position in range(len(monomial) - 1):
        left, right = monomial[position], monomial[position + 1]
        if left == right:
            return ("vanish", position)
        if _order_key(left) > _order_key(right):
            return ("swap", position)
    return None


def _canonical_terms(
    monomial: _Monomial,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], complex]:
    """Rewrite one product of factors as a sum of normal-ordered monomials.

    The rewrite resolves the leftmost defect until none is left.  A defect is either two
    adjacent equal factors, which the Pauli exclusion principle annihilates, or an adjacent
    out-of-order pair, which is replaced by the same pair reversed at the cost of a sign.
    Reversing the pair is the sole rule for two different kinds with different degrees and
    for two equal kinds; when the pair is an annihilation followed by a creation of the
    same degree it is instead the contraction ``c[p] c†[p] = delta - c†[p] c[p]``, whose
    delta branch removes the pair and whose other branch is that same reversal.

    Termination is by a measure that never rises: removing a pair shortens the word, and
    reversing an inverted adjacent pair lowers the number of pairs in the wrong order by
    exactly one, as one step of bubble sort does.
    """

    pending: list[tuple[_Monomial, complex]] = [(monomial, 1.0 + 0.0j)]
    resolved: dict[tuple[tuple[int, ...], tuple[int, ...]], complex] = {}
    while pending:
        current, coefficient = pending.pop()
        defect = _first_defect(current)
        if defect is None:
            creations = tuple(degree for degree, is_creation in current if is_creation)
            annihilations = tuple(
                degree for degree, is_creation in current if not is_creation
            )
            key = (creations, annihilations)
            resolved[key] = resolved.get(key, 0.0j) + coefficient
            continue
        kind, position = defect
        if kind == "vanish":
            continue
        left, right = current[position], current[position + 1]
        head, tail = current[:position], current[position + 2 :]
        if left[0] == right[0]:
            # `_first_defect` only reverses a creation-before-annihilation pair, so equal
            # degrees mean the contraction; its delta branch is the only length drop.
            pending.append((head + tail, coefficient))
        pending.append((head + (right, left) + tail, -coefficient))
    return resolved


def _factor_sequence(term: "FermionTerm") -> _Monomial:
    """Read a canonical term back as the flat left-to-right factor sequence."""

    creations = tuple((degree, True) for degree in term.creations)
    annihilations = tuple((degree, False) for degree in term.annihilations)
    return creations + annihilations


def _merge_pauli(
    left: PauliFactors, right: PauliFactors
) -> tuple[PauliFactors, complex]:
    """Multiply two Pauli products, returning the result and the accumulated phase."""

    merged: dict[int, str] = dict(left)
    phase = 1.0 + 0.0j
    for wire, axis in right:
        existing = merged.get(wire)
        if existing is None:
            merged[wire] = axis
        elif existing == axis:
            del merged[wire]
        else:
            replacement, factor = _PAULI_PRODUCT[(existing, axis)]
            merged[wire] = replacement
            phase *= factor
    return tuple(sorted(merged.items())), phase


def _multiply_pauli(left: _PauliSum, right: _PauliSum) -> _PauliSum:
    """Multiply two Pauli sums and drop the terms that cancel exactly."""

    product: _PauliSum = {}
    for left_factors, left_coefficient in left.items():
        for right_factors, right_coefficient in right.items():
            factors, phase = _merge_pauli(left_factors, right_factors)
            coefficient = left_coefficient * right_coefficient * phase
            if coefficient != 0.0j:
                product[factors] = product.get(factors, 0.0j) + coefficient
    return {factors: value for factors, value in product.items() if value != 0.0j}


def _jordan_wigner_factor(degree: int, is_creation: bool) -> _PauliSum:
    """Return the Pauli sum one factor's Jordan-Wigner image expands to."""

    string = tuple((mode, "z") for mode in range(degree))
    image = _CREATION_IMAGE if is_creation else _ANNIHILATION_IMAGE
    return {(*string, (degree, axis)): coefficient for coefficient, axis in image}


def _parity_factor(degree: int, is_creation: bool, modes: int) -> _PauliSum:
    """Return the Pauli sum one factor's parity-encoding image expands to.

    The two branches are the same ``sigma_minus`` and ``sigma_plus`` expansion the
    Jordan-Wigner image uses; what differs is the string they sit on.  On the X branch a
    single Z sits on wire ``degree - 1`` and an X runs over every mode above the factor; on
    the Y branch there is no Z at all.  The X string therefore runs over the modes strictly
    above the factor where Jordan-Wigner's Z string runs over the modes strictly below it,
    which is what makes the two encodings mirror images on the mode list.
    """

    above = tuple((mode, "x") for mode in range(degree + 1, modes))
    lower = () if degree == 0 else ((degree - 1, "z"),)
    image = _CREATION_IMAGE if is_creation else _ANNIHILATION_IMAGE
    return {
        (*lower, (degree, "x"), *above): image[0][0],
        ((degree, "y"), *above): image[1][0],
    }


@dataclass(frozen=True, slots=True)
class FermionTerm:
    """One normal-ordered fermionic monomial with a complex coefficient.

    The operator is ``coefficient * c†[creations[0]] ... c[annihilations[-1]]``: creation
    operators first and each group in ascending degree.  Both groups are validated to be
    in that order, which is what makes the representation canonical -- two equal operators
    hold equal `creations` and equal `annihilations`, with no rewrite step to run first.
    """

    coefficient: complex
    creations: tuple[int, ...] = ()
    annihilations: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        coefficient = _complex_scalar(self.coefficient)
        if coefficient is None:
            raise TypeError("a fermionic term requires a numeric coefficient")
        object.__setattr__(self, "coefficient", coefficient)
        for attribute, owner in (
            ("creations", "fermionic creation"),
            ("annihilations", "fermionic annihilation"),
        ):
            object.__setattr__(
                self,
                attribute,
                _degrees(getattr(self, attribute), owner=owner),
            )

    @property
    def degrees(self) -> tuple[int, ...]:
        """Return every distinct mode this monomial acts on, in ascending order."""

        return tuple(sorted(set(self.creations) | set(self.annihilations)))

    def dagger(self) -> "FermionTerm":
        """Return the adjoint, which reverses the order of the product.

        Reversing a product costs a sign: the creation degrees come back in descending
        order and the annihilation degrees likewise, and restoring each group to ascending
        order is a reversal of that group, with sign ``(-1)^(k choose 2)``.  Without that
        sign the adjoint of ``c†[0] c†[1]`` would be reported as ``c[0] c[1]`` rather than
        ``-c[0] c[1]``, and a Hermiticity test built on it would accept operators that are
        in fact anti-Hermitian.
        """

        def reversal_sign(count: int) -> int:
            return -1 if (count * (count - 1) // 2) % 2 else 1

        sign = reversal_sign(len(self.creations)) * reversal_sign(
            len(self.annihilations)
        )
        return FermionTerm(
            sign * self.coefficient.conjugate(), self.annihilations, self.creations
        )


@dataclass(frozen=True, slots=True)
class FermionOperator:
    """An immutable complex linear combination of normal-ordered monomials.

    Terms that share a monomial are combined and a term whose coefficient is exactly zero
    is dropped, so the stored sum is canonical and comparison is exact.  No threshold is
    applied: a small coefficient is kept rather than quietly rounded away.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.observables import annihilate, create, jordan_wigner
        >>> hopping = create(0) * annihilate(1) + create(1) * annihilate(0)
        >>> observable = jordan_wigner(hopping, n_modes=2)
        >>> sorted(term.factors for term in observable.terms)
        [((0, 'x'), (1, 'x')), ((0, 'y'), (1, 'y'))]
        >>> fq.expectation(observable).observable is observable
        True
    """

    terms: tuple[FermionTerm, ...] = ()

    def __post_init__(self) -> None:
        combined: dict[tuple[tuple[int, ...], tuple[int, ...]], complex] = {}
        for term in self.terms:
            if not isinstance(term, FermionTerm):
                raise TypeError(
                    "a fermionic operator holds FermionTerm values, got "
                    f"{type(term).__name__}"
                )
            key = (term.creations, term.annihilations)
            combined[key] = combined.get(key, 0.0j) + term.coefficient
        object.__setattr__(
            self,
            "terms",
            tuple(
                FermionTerm(coefficient, creations, annihilations)
                for (creations, annihilations), coefficient in sorted(combined.items())
                if coefficient != 0.0j
            ),
        )

    def __add__(self, other: object) -> "FermionOperator":
        if not isinstance(other, FermionOperator):
            return NotImplemented
        return FermionOperator((*self.terms, *other.terms))

    def __sub__(self, other: object) -> "FermionOperator":
        if not isinstance(other, FermionOperator):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> "FermionOperator":
        return FermionOperator(
            tuple(
                FermionTerm(-term.coefficient, term.creations, term.annihilations)
                for term in self.terms
            )
        )

    def __mul__(self, other: object) -> "FermionOperator":
        """Multiply by a scalar or compose with another operator.

        Both cases are the same product in the algebra: a scalar is its degree-zero
        element, and composition concatenates the factors of the two monomials and
        reapplies the anticommutation relations.
        """

        if isinstance(other, FermionOperator):
            return FermionOperator(
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
        return FermionOperator(
            tuple(
                FermionTerm(
                    value * term.coefficient, term.creations, term.annihilations
                )
                for term in self.terms
            )
        )

    def __rmul__(self, other: object) -> "FermionOperator":
        return self * other

    def dagger(self) -> "FermionOperator":
        """Return the adjoint, which reverses the order of every monomial."""

        return FermionOperator(tuple(term.dagger() for term in self.terms))

    def is_hermitian(self) -> bool:
        """Report whether this operator equals its own adjoint.

        The canonical form makes this an exact structural test rather than a tolerance,
        which is what lets `jordan_wigner` refuse a non-Hermitian operator on a fact
        instead of on a numerical impression.
        """

        return self == self.dagger()

    def commutator(self, other: "FermionOperator") -> "FermionOperator":
        """Return ``self other - other self``."""

        if not isinstance(other, FermionOperator):
            raise TypeError("a commutator requires two fermionic operators")
        return self * other - other * self

    def anticommutator(self, other: "FermionOperator") -> "FermionOperator":
        """Return ``self other + other self``, which is zero for two odd monomials."""

        if not isinstance(other, FermionOperator):
            raise TypeError("an anticommutator requires two fermionic operators")
        return self * other + other * self


def _compose(left: FermionTerm, right: FermionTerm) -> tuple[FermionTerm, ...]:
    """Return the canonical expansion of the product of two canonical terms."""

    scale = left.coefficient * right.coefficient
    return tuple(
        FermionTerm(scale * coefficient, creations, annihilations)
        for (creations, annihilations), coefficient in _canonical_terms(
            _factor_sequence(left) + _factor_sequence(right)
        ).items()
    )


def create(degree: int) -> FermionOperator:
    """Return the creation operator ``c†[degree]``."""

    mode = _degree(degree)
    return FermionOperator((FermionTerm(1.0, (mode,), ()),))


def annihilate(degree: int) -> FermionOperator:
    """Return the annihilation operator ``c[degree]``."""

    mode = _degree(degree)
    return FermionOperator((FermionTerm(1.0, (), (mode,)),))


def number(degree: int) -> FermionOperator:
    """Return the occupation-number operator ``c†[degree] c[degree]``."""

    mode = _degree(degree)
    return FermionOperator((FermionTerm(1.0, (mode,), (mode,)),))


def _image_of(
    operator: FermionOperator,
    *,
    modes: int,
    transform: str,
    factor: Callable[[int, bool], _PauliSum],
) -> Observable:
    """Map a Hermitian operator onto a Pauli `Observable` through a factor image.

    Everything here is the same for every fermion-to-qubit transform: the mode count and
    Hermiticity are checked, each monomial's factor images are multiplied in the order the
    monomial states them, terms that share a Pauli string are collected, and a coefficient
    with an imaginary part outside the fail-closed bound is reported.  Only ``factor``
    differs between transforms, which is what lets a second transform be a second
    implementation behind the one boundary callers already read.

    Only a Hermitian operator has a Pauli image with real coefficients, which is what an
    `Observable` stores.  A non-Hermitian operator is refused rather than quietly given a
    complex-coefficient representation this package does not have.
    """

    if modes < 1:
        raise ValueError(f"{transform} requires at least one mode")
    if not operator.is_hermitian():
        raise CapabilityError(
            f"{transform} maps a Hermitian fermionic operator onto a "
            "real-coefficient Pauli observable; this operator is not Hermitian, and a "
            "complex-coefficient Pauli sum is not a representation FlagQuantum has"
        )
    image: _PauliSum = {}
    for term in operator.terms:
        for degree in term.degrees:
            if degree >= modes:
                raise ValueError(
                    f"fermionic degree {degree} is outside the {modes} mode(s) requested"
                )
        monomial: _PauliSum = {(): 1.0 + 0.0j}
        for degree, is_creation in _factor_sequence(term):
            monomial = _multiply_pauli(monomial, factor(degree, is_creation))
        for factors, coefficient in monomial.items():
            image[factors] = image.get(factors, 0.0j) + term.coefficient * coefficient
    magnitude = max((abs(coefficient) for coefficient in image.values()), default=0.0)
    tolerance = _IMAGINARY_TOLERANCE * max(1.0, magnitude)
    terms: list[_PauliTerm] = []
    for factors, coefficient in sorted(image.items()):
        if abs(coefficient.imag) > tolerance:
            raise CapabilityError(
                f"the {transform} image of this operator carries an imaginary "
                f"coefficient {coefficient!r}; a Hermitian fermionic operator has a "
                "real-coefficient Pauli image"
            )
        if coefficient.real == 0.0:
            continue
        terms.append(_PauliTerm(float(coefficient.real), factors))
    if not terms:
        # The zero fermionic operator is a valid input and its image is the zero
        # observable, which an Observable states as one identity term of weight zero.
        return Observable((_PauliTerm(0.0, ()),))
    return Observable(tuple(terms))


def jordan_wigner(operator: FermionOperator, *, n_modes: int) -> Observable:
    """Map a Hermitian fermionic operator onto a Pauli `Observable`.

    Mode ``k`` is placed on wire ``k`` and ``c[k]`` maps to the Pauli string
    ``Z[0] ... Z[k-1] sigma_minus[k]``, so a product of factors maps to the product of
    their images.  The result is exact for a sum of monomials and carries no fermionic
    special case into the runtime: it is the same `Observable` that `fq.expectation`
    already lowers.

    Only a Hermitian operator has a Pauli image with real coefficients, which is what an
    `Observable` stores.  A non-Hermitian operator is refused rather than quietly given a
    complex-coefficient representation this package does not have.

    Examples:
        >>> from flagquantum.observables import annihilate, create, jordan_wigner
        >>> hopping = create(0) * annihilate(1) + create(1) * annihilate(0)
        >>> observable = jordan_wigner(hopping, n_modes=2)
        >>> [(term.coefficient, term.factors) for term in observable.terms]
        [(0.5, ((0, 'x'), (1, 'x'))), (0.5, ((0, 'y'), (1, 'y')))]
    """

    return _image_of(
        operator,
        modes=_qubit("fermionic operator", n_modes, noun="mode"),
        transform="jordan_wigner",
        factor=_jordan_wigner_factor,
    )


def parity_encoding(operator: FermionOperator, *, n_modes: int) -> Observable:
    """Map a Hermitian fermionic operator onto a Pauli `Observable` in the parity basis.

    Wire ``k`` holds the parity of modes ``0`` through ``k`` instead of the occupation of
    mode ``k``, so ``c[k]`` maps to
    ``(Z[k-1] X[k] ... X[n-1] + i Y[k] X[k+1] ... X[n-1]) / 2``: the X string runs over the
    modes strictly above the factor, where `jordan_wigner`'s Z string runs over the modes
    strictly below it.  The two transforms are the same operator written in two bases, so a
    consumer written against `Observable` reads either one without being told which it
    holds; the images differ as matrices by the basis change between them.

    This is a second encoding, not a smaller one.  The widest factor image still spans one
    wire per mode and is reached at the opposite end of the mode list, so the reduction in
    worst-case Pauli weight that motivates the Bravyi-Kitaev and ternary-tree encodings
    does not follow from it.

    Only a Hermitian operator has a Pauli image with real coefficients, which is what an
    `Observable` stores.  A non-Hermitian operator is refused rather than quietly given a
    complex-coefficient representation this package does not have.

    Examples:
        >>> from flagquantum.observables import annihilate, create, parity_encoding
        >>> hopping = create(0) * annihilate(1) + create(1) * annihilate(0)
        >>> observable = parity_encoding(hopping, n_modes=2)
        >>> [(term.coefficient, term.factors) for term in observable.terms]
        [(0.5, ((0, 'x'),)), (-0.5, ((0, 'x'), (1, 'z')))]
    """

    modes = _qubit("fermionic operator", n_modes, noun="mode")
    return _image_of(
        operator,
        modes=modes,
        transform="parity_encoding",
        factor=lambda degree, is_creation: _parity_factor(degree, is_creation, modes),
    )


__all__ = (
    "FermionOperator",
    "FermionTerm",
    "annihilate",
    "create",
    "jordan_wigner",
    "number",
    "parity_encoding",
)
