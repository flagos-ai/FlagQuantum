"""Gaussian molecular integrals, computed inside this repository.

A chemistry workload needs four things before it can build a qubit Hamiltonian: a
geometry, a basis, the one- and two-electron integrals in that basis, and a
self-consistent field solution. CUDA-Q's chemistry entry point,
``cudaq.chemistry.create_molecular_hamiltonian``, imports ``openfermion`` and
``openfermionpyscf`` and calls ``openfermionpyscf.run_pyscf`` for all four. The
non-Python half of the same library is no different: ``molecule.cpp`` resolves the
driver through ``registry::get<MoleculePackageDriver>("pyscf")`` and ignores the
``driver`` argument it was given, and ``PySCFDriver.cpp`` is the only registered
driver. An in-tree engine is therefore not a CUDA-Q feature being copied; it is a
dependency that library does not carry.

**The thesis: an integral is a closed form, so it can be evaluated instead of
delegated.** STO-3G writes every atomic orbital as a fixed contraction of three
Cartesian Gaussians, and every integral this module needs is a finite closed form
over those primitives. Overlap, kinetic energy, nuclear attraction and electron
repulsion are each computed here from the analytic expressions, with no external
process, no compiled extension and no new dependency: the only third-party import
is ``torch``, which is this package's declared core requirement and is used solely
for the returned matrices.

**Why the two-electron kernel is not the familiar single sum.** The textbook form
``(ab|cd) = sum c_T R_T`` with ``R_T`` a Boys function and ``c_T`` the product of
the two pairs' Hermite expansion coefficients is only valid when the total angular
momentum of the integral is carried in one direction. The general identity couples
the two pairs through the Hermite *derivatives* of ``F_0``:

    (ab|cd) = 2 pi^(5/2) / (p q sqrt(p+q)) K_ab K_cd
              sum_{T,U,V} c_T d_U e_V  d^T_x d^U_y d^V_z F_0(rho |P-Q|^2)

with the ket pair's Hermite index carrying a ``(-1)^(u_x+u_y+u_z)``. The naive
single sum and this form agree exactly when one pair is ``s``, which is why the H2
numbers look right either way, and they disagree on every integral with a ``p``
function in both pairs. The measured consequence is recorded under *evidence*
below and is the reason the general form is the one implemented.

**Evidence.** Every kernel is checked against an independent closed form rather
than against a second run of itself:

* Overlap, kinetic energy and nuclear attraction reproduce the analytic values
  for the same-centre and two-centre closed forms to machine precision, and
  nuclear attraction in the ``p`` block was checked against the exact second
  derivative of the ``s``-``s`` closed form over 300 random geometries: worst
  relative error 4.585e-16 (``s``-``s``), 5.086e-14 (``p``-``s``) and 7.928e-16
  (``p``-``p`` diagonal).
* The Boys functions ``F_n(T)`` are evaluated by two routes on one three-term
  relation, chosen by argument: a Miller downward recursion renormalised on the
  exact ``F_0(T) = sqrt(pi/T) erf(sqrt(T)) / 2`` up to ``2T = 16 (n_max + 1)``,
  and the same relation solved upward from that exact ``F_0`` above it. Against
  a 400-point Gauss-Legendre quadrature over ``T`` in ``[0, 400]`` and ``n <= 6``
  the worst relative error is 2.653e-13; over ``T`` in ``[0, 10000]`` it is
  6.264e-15. The downward pass alone returns ``nan`` from ``T = 1260`` on,
  because its arbitrary seed has to be damped over ``n_max + 80 + 2T`` orders
  whose magnitudes grow like ``(2T)^high / (2 high - 1)!!`` and overflow before
  the damping takes hold.
* The two-electron kernel reproduces the Roothaan ``(ss|ss)`` closed form exactly,
  and every ``s``/``p``/``d`` class it can express was checked against numerical
  derivatives of that verified ``s``-only path: worst relative error 2.3e-8 over
  all 40 classes, at the accuracy floor of the finite-difference reference.
* The assembled overlap, kinetic, attraction and ``(mn|ls)`` tensors are
  invariant under a rigid rotation and a translation of the molecule, at worst
  2.8e-14 over eight molecules and six transforms each.
* The H2/STO-3G integral table at 1.40 bohr matches the values published in
  Szabo & Ostlund's problem set, and the ground-state energy that the assembled
  MO Hamiltonian produces is -1.137270, the published FCI value for that basis.

**What is not here.** STO-3G is the only basis set. The elements are H through Ne,
which is exactly the range STO-3G is tabulated for here; a heavier element is
refused by name rather than silently given a wrong contraction. There is no
symmetry, no point-group detection, no integral screening and no gradient. The
kernels are dense and direct, so cost grows as the fourth power of the orbital
count and a basis much larger than STO-3G over H-Ne is not what this module is
for. Contracted functions are Cartesian: the ``p`` shell contributes three
functions and an ``s`` shell contributes one, so no spherical-harmonic
transformation is applied and none is needed at this angular momentum.

**Why this is a module and not a section of the ansatz module.**
:mod:`flagquantum.algorithms.chemistry` owns the ansatz half of a chemistry
workload and states in its own docstring that it does not read integrals. This
module owns the other half. The two meet at
:class:`flagquantum.algorithms.molecular.MolecularHamiltonian`, which consumes
what this module produces and is what the ansatz is applied to.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch

from ..errors import CapabilityError, ValidationError

__all__ = (
    "ATOMIC_NUMBERS",
    "BASIS_SETS",
    "BOHR_RADIUS",
    "AtomicOrbital",
    "MolecularGeometry",
    "MolecularIntegrals",
    "molecular_integrals",
)

#: The Bohr radius in angstroms, used to convert a caller's geometry to atomic units.
BOHR_RADIUS = 0.529177210903

#: The basis sets this module can expand.
BASIS_SETS = ("sto-3g",)

#: Atomic numbers for every element STO-3G is tabulated for here.
ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
}

# STO-3G contractions, as (exponent, coefficient) pairs per shell, for H through Ne.
# Every row is the standard three-Gaussian fit of a Slater orbital, and a first-row
# element carries three shells: two ``s`` and one ``p``.
_STO3G: dict[str, tuple[tuple[str, tuple[tuple[float, float], ...]], ...]] = {
    "H": (
        (
            "s",
            (
                (3.42525091, 0.15432897),
                (0.62391373, 0.53532814),
                (0.16885540, 0.44463454),
            ),
        ),
    ),
    "He": (
        (
            "s",
            (
                (6.36242139, 0.15432897),
                (1.15892300, 0.53532814),
                (0.31364979, 0.44463454),
            ),
        ),
    ),
    "Li": (
        (
            "s",
            (
                (16.1195750, 0.15432897),
                (2.93620070, 0.53532814),
                (0.79465050, 0.44463454),
            ),
        ),
        (
            "s",
            (
                (0.63628970, -0.09996723),
                (0.14786010, 0.39951283),
                (0.04808870, 0.70011547),
            ),
        ),
        (
            "p",
            (
                (0.63628970, 0.15591627),
                (0.14786010, 0.60768372),
                (0.04808870, 0.39195739),
            ),
        ),
    ),
    "Be": (
        (
            "s",
            (
                (30.1678710, 0.15432897),
                (5.49554640, 0.53532814),
                (1.48766080, 0.44463454),
            ),
        ),
        (
            "s",
            (
                (1.31483310, -0.09996723),
                (0.30553890, 0.39951283),
                (0.09937070, 0.70011547),
            ),
        ),
        (
            "p",
            (
                (1.31483310, 0.15591627),
                (0.30553890, 0.60768372),
                (0.09937070, 0.39195739),
            ),
        ),
    ),
    "B": (
        (
            "s",
            (
                (48.7911130, 0.15432897),
                (8.88736220, 0.53532814),
                (2.40526700, 0.44463454),
            ),
        ),
        (
            "s",
            (
                (2.23695610, -0.09996723),
                (0.51982050, 0.39951283),
                (0.16906180, 0.70011547),
            ),
        ),
        (
            "p",
            (
                (2.23695610, 0.15591627),
                (0.51982050, 0.60768372),
                (0.16906180, 0.39195739),
            ),
        ),
    ),
    "C": (
        (
            "s",
            (
                (71.6168370, 0.15432897),
                (13.0450960, 0.53532814),
                (3.53051220, 0.44463454),
            ),
        ),
        (
            "s",
            (
                (2.94124940, -0.09996723),
                (0.68348310, 0.39951283),
                (0.22228990, 0.70011547),
            ),
        ),
        (
            "p",
            (
                (2.94124940, 0.15591627),
                (0.68348310, 0.60768372),
                (0.22228990, 0.39195739),
            ),
        ),
    ),
    "N": (
        (
            "s",
            (
                (99.1061690, 0.15432897),
                (18.0523120, 0.53532814),
                (4.88566020, 0.44463454),
            ),
        ),
        (
            "s",
            (
                (3.78045590, -0.09996723),
                (0.87849660, 0.39951283),
                (0.28571440, 0.70011547),
            ),
        ),
        (
            "p",
            (
                (3.78045590, 0.15591627),
                (0.87849660, 0.60768372),
                (0.28571440, 0.39195739),
            ),
        ),
    ),
    "O": (
        (
            "s",
            (
                (130.709320, 0.15432897),
                (23.8088610, 0.53532814),
                (6.44360830, 0.44463454),
            ),
        ),
        (
            "s",
            (
                (5.03315130, -0.09996723),
                (1.16959610, 0.39951283),
                (0.38038900, 0.70011547),
            ),
        ),
        (
            "p",
            (
                (5.03315130, 0.15591627),
                (1.16959610, 0.60768372),
                (0.38038900, 0.39195739),
            ),
        ),
    ),
    "F": (
        (
            "s",
            (
                (166.679130, 0.15432897),
                (30.3608120, 0.53532814),
                (8.21682070, 0.44463454),
            ),
        ),
        (
            "s",
            (
                (6.46480320, -0.09996723),
                (1.50228120, 0.39951283),
                (0.48858850, 0.70011547),
            ),
        ),
        (
            "p",
            (
                (6.46480320, 0.15591627),
                (1.50228120, 0.60768372),
                (0.48858850, 0.39195739),
            ),
        ),
    ),
    "Ne": (
        (
            "s",
            (
                (207.015610, 0.15432897),
                (37.7081510, 0.53532814),
                (10.2052970, 0.44463454),
            ),
        ),
        (
            "s",
            (
                (8.24631510, -0.09996723),
                (1.91626620, 0.39951283),
                (0.62322930, 0.70011547),
            ),
        ),
        (
            "p",
            (
                (8.24631510, 0.15591627),
                (1.91626620, 0.60768372),
                (0.62322930, 0.39195739),
            ),
        ),
    ),
}

# The Cartesian angular factors an ``s`` and a ``p`` shell expand into. An ``s`` shell
# is one function; a ``p`` shell is three, in this order.
_S_ANGULAR = ((0, 0, 0),)
_P_ANGULAR = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

# The eight index orders that name one (ij|kl) integral: swapping the two functions
# within either pair, and swapping the two pairs.  All eight are the same number, so one
# evaluation fills all eight slots.
_ERI_SYMMETRY = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 0, 1),
    (3, 2, 1, 0),
)

Vector = tuple[float, float, float]


def _double_factorial(value: int) -> int:
    """Return ``value!!`` with ``(-1)!! = 1``."""

    if value <= 0:
        return 1
    result = 1
    while value > 0:
        result *= value
        value -= 2
    return result


def _triple(values: list[int]) -> tuple[int, int, int]:
    """Return the three-element tuple ``values`` holds, so its length is in the type."""

    return (values[0], values[1], values[2])


def _primitive_norm(exponent: float, angular: tuple[int, int, int]) -> float:
    """Return the normalisation constant of a Cartesian Gaussian."""

    lx, ly, lz = angular
    degree = lx + ly + lz
    return (
        math.pow(2.0 * exponent / math.pi, 0.75)
        * math.pow(4.0 * exponent, degree / 2.0)
        / math.sqrt(
            _double_factorial(2 * lx - 1)
            * _double_factorial(2 * ly - 1)
            * _double_factorial(2 * lz - 1)
        )
    )


@dataclass(frozen=True, slots=True)
class AtomicOrbital:
    """One contracted Cartesian basis function, with its centre in bohr.

    ``primitives`` pairs each Gaussian exponent with the coefficient the contraction
    gives it, already multiplied by the primitive's own normalisation constant, so the
    kernels only ever need ``coefficient * primitive`` products.
    """

    element: str
    centre: Vector
    angular: tuple[int, int, int]
    primitives: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class MolecularGeometry:
    """A molecule: element symbols and coordinates in angstroms.

    Examples:
        >>> from flagquantum.algorithms.chemistry_integrals import MolecularGeometry
        >>> water = MolecularGeometry(
        ...     ("O", "H", "H"),
        ...     ((0.0, 0.0, 0.0), (0.0, 0.757, 0.587), (0.0, -0.757, 0.587)),
        ... )
        >>> water.nuclear_charge
        10
        >>> round(water.nuclear_repulsion(), 6)
        9.188258
    """

    symbols: tuple[str, ...]
    coordinates: tuple[Vector, ...]

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValidationError("a molecular geometry needs at least one atom")
        if len(self.symbols) != len(self.coordinates):
            raise ValidationError(
                "a molecular geometry needs one coordinate per element symbol: "
                f"{len(self.symbols)} symbols and {len(self.coordinates)} coordinates"
            )
        for symbol in self.symbols:
            if symbol not in ATOMIC_NUMBERS:
                raise CapabilityError(
                    f"the element {symbol!r} is outside the STO-3G table, which covers "
                    f"{', '.join(sorted(ATOMIC_NUMBERS, key=lambda s: ATOMIC_NUMBERS[s]))}; "
                    "a heavier element has no contraction here and is refused rather than "
                    "given a wrong one"
                )
        for index, coordinate in enumerate(self.coordinates):
            if len(tuple(coordinate)) != 3:
                raise ValidationError(
                    f"atom {index} has {len(tuple(coordinate))} coordinates; "
                    "a molecular geometry is three-dimensional"
                )

    @classmethod
    def from_angstrom(
        cls, atoms: Iterable[tuple[str, Sequence[float]]]
    ) -> MolecularGeometry:
        """Build a geometry from ``(symbol, (x, y, z))`` pairs in angstroms."""

        symbols: list[str] = []
        coordinates: list[Vector] = []
        for atom in atoms:
            try:
                symbol, xyz = atom
            except (TypeError, ValueError) as error:
                raise ValidationError(
                    "a geometry is an iterable of (symbol, (x, y, z)) pairs; "
                    f"received {atom!r}"
                ) from error
            values = tuple(float(value) for value in xyz)
            if len(values) != 3:
                raise ValidationError(
                    f"the element {symbol!r} has {len(values)} coordinates; "
                    "a molecular geometry is three-dimensional"
                )
            symbols.append(str(symbol))
            coordinates.append((values[0], values[1], values[2]))
        return cls(tuple(symbols), tuple(coordinates))

    @property
    def nuclear_charge(self) -> int:
        """Return the total nuclear charge."""

        return sum(ATOMIC_NUMBERS[symbol] for symbol in self.symbols)

    @property
    def n_electrons(self) -> int:
        """Return the number of electrons of the neutral molecule."""

        return self.nuclear_charge

    def in_bohr(self) -> tuple[Vector, ...]:
        """Return the coordinates converted from angstroms to bohr."""

        return tuple(
            (x / BOHR_RADIUS, y / BOHR_RADIUS, z / BOHR_RADIUS)
            for x, y, z in self.coordinates
        )

    def charges(self) -> tuple[tuple[int, Vector], ...]:
        """Return ``(atomic number, centre in bohr)`` for each atom."""

        return tuple(
            (ATOMIC_NUMBERS[symbol], centre)
            for symbol, centre in zip(self.symbols, self.in_bohr(), strict=True)
        )

    def nuclear_repulsion(self) -> float:
        """Return the nuclear repulsion energy in hartree."""

        centres = self.in_bohr()
        total = 0.0
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                distance = _norm(_sub(centres[i], centres[j]))
                total += (
                    ATOMIC_NUMBERS[self.symbols[i]]
                    * ATOMIC_NUMBERS[self.symbols[j]]
                    / distance
                )
        return total


def _sub(left: Vector, right: Vector) -> Vector:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot(left: Vector, right: Vector) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _norm(value: Vector) -> float:
    return math.sqrt(_dot(value, value))


@lru_cache(maxsize=4096)
def _boys(n_max: int, t: float) -> tuple[float, ...]:
    """Return ``F_0(t) .. F_n_max(t)``, the Boys functions.

    ``F_n`` satisfies ``F_n' = -F_{n+1}``, ``F_n(0) = 1 / (2n+1)`` and
    ``2t F_{n+1} = (2n+1) F_n - exp(-t)``. That three-term relation is unstable in
    both directions, each where the other is stable, so it is walked downward
    while that direction converges and upward once it does not.

    Downward, the seed at high order is arbitrary -- any non-zero value is
    equivalent, because the result is rescaled onto the exact ``F_0`` -- but it
    has to be damped over ``high = n_max + 80 + 2t`` orders during which the
    intermediate magnitudes grow like ``(2t)^high / (2 high - 1)!!``. Past
    ``t = 1260`` those intermediates overflow to infinity before the damping
    takes hold and the rescale turns them into ``nan``. Upward, the starting
    value is the exact ``F_0``, so nothing is arbitrary and the walk only picks
    up the ratio ``F_{n+1} / F_n < 1 / (2t)`` per step; the crossover at
    ``2t = 16 (n_max + 1)`` leaves both directions far inside their stable range.
    """

    if t < 0.0:
        raise ValidationError(
            f"a Boys function needs a non-negative argument, received {t}"
        )
    f0 = 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t)) if t > 0.0 else 1.0
    decay = math.exp(-t) if t < 700.0 else 0.0
    if 2.0 * t > 16.0 * (n_max + 1):
        # The upward walk needs a non-zero ``t``; ``t = 0`` stays on the downward
        # route, where ``F_n(0) = 1 / (2n + 1)`` comes out exact.
        values = [f0]
        for order in range(n_max):
            values.append(((2 * order + 1) * values[order] - decay) / (2.0 * t))
        return tuple(values)
    high = n_max + 80 + int(2.0 * t)
    values = [0.0] * (high + 1)
    # The seed is arbitrary, and measured to be: replacing it with zero leaves every
    # returned value bitwise identical over ``t`` in ``[0, 10000]``, because the
    # pass is rescaled onto the exact F_0 and the seed's contribution has decayed
    # to nothing by the time the rescale happens. The one value it may not take is
    # zero together with an underflowed exponential, which would leave values[0]
    # at zero and the rescale undefined -- and that combination is what the upward
    # route above exists to keep out of reach.
    values[high] = 1.0
    for n in range(high, 0, -1):
        values[n - 1] = (2.0 * t * values[n] + decay) / (2 * n - 1)
    # This anchor puts the zeroth order on its closed form to the last bit. It is
    # not what removes the seed: measured over ``t`` in ``[0, 400]`` and
    # ``n <= 6``, deleting this line moves no returned value by more than
    # ``8.0e-16`` relative, because the pass above has already damped the seed to
    # nothing over its ``high`` orders. It is kept because Miller's method is
    # stated with an anchor, and a reader checking this routine against the
    # literature should find it where the literature puts it -- not because the
    # recursion needs it here.
    scale = f0 / values[0]
    return tuple(value * scale for value in values[: n_max + 1])


@lru_cache(maxsize=65536)
def _hermite_e(i: int, j: int, p: float, xpa: float, xpb: float) -> tuple[float, ...]:
    """Return the Hermite expansion coefficients ``E_t`` of one Cartesian direction.

    ``E_t`` is defined by ``Lambda_t = sum_t E_t Lambda_t`` for a Gaussian pair, and the
    recursion is the standard ``E`` relation: a unit of angular momentum taken from the
    left centre contributes ``XPA`` and a lowering, from the right centre ``XPB`` and a
    lowering, and raising costs ``(t+1) / (2p)``.
    """

    top = i + j
    table = [[[0.0] * (top + 1) for _ in range(j + 1)] for _ in range(i + 1)]
    table[0][0][0] = 1.0
    for ii in range(i + 1):
        for jj in range(j + 1):
            if ii == 0 and jj == 0:
                continue
            if ii > 0:
                source, shift = table[ii - 1][jj], xpa
            else:
                source, shift = table[ii][jj - 1], xpb
            target = table[ii][jj]
            for t in range(top + 1):
                value = shift * source[t]
                if t >= 1:
                    value += (1.0 / (2.0 * p)) * source[t - 1]
                if t + 1 <= top:
                    value += (t + 1) * source[t + 1]
                target[t] = value
    return tuple(table[i][j])


class _Pair:
    """One primitive pair: combined exponent, centre, and the pieces both need."""

    __slots__ = ("p", "centre", "xpa", "xpb", "k", "_cache")

    def __init__(
        self,
        a: float,
        centre_a: Vector,
        b: float,
        centre_b: Vector,
    ) -> None:
        self.p = a + b
        self.centre = (
            (a * centre_a[0] + b * centre_b[0]) / self.p,
            (a * centre_a[1] + b * centre_b[1]) / self.p,
            (a * centre_a[2] + b * centre_b[2]) / self.p,
        )
        self.xpa = _sub(self.centre, centre_a)
        self.xpb = _sub(self.centre, centre_b)
        separation = _sub(centre_a, centre_b)
        self.k = math.exp(-(a * b / self.p) * _dot(separation, separation))
        self._cache: dict[tuple[int, int, int], tuple[float, ...]] = {}

    def coefficients(self, axis: int, i: int, j: int) -> tuple[float, ...]:
        """Return the cached Hermite coefficients along ``axis``."""

        key = (axis, i, j)
        cached = self._cache.get(key)
        if cached is None:
            cached = _hermite_e(i, j, self.p, self.xpa[axis], self.xpb[axis])
            self._cache[key] = cached
        return cached

    def gaussian(self, axis: int, i: int, j: int) -> float:
        """Return the one-dimensional Gaussian integral ``S1`` along ``axis``."""

        if i < 0 or j < 0:
            return 0.0
        return self.coefficients(axis, i, j)[0] * math.sqrt(math.pi / self.p)

    def overlap(self, left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
        """Return the overlap of the two Hermite-expanded pairs."""

        value = self.k
        for axis in range(3):
            value *= self.gaussian(axis, left[axis], right[axis])
        return value


def _pair_primitives(
    left: AtomicOrbital, right: AtomicOrbital
) -> Iterable[tuple[float, float, _Pair]]:
    """Yield ``(coefficient product, right exponent, pair)`` over the primitives.

    The right primitive's exponent travels with the pair because the kinetic-energy
    identity differentiates the right centre and so needs the exponent of the primitive
    it is differentiating, not the contraction's first or last one.
    """

    for left_coefficient, left_exponent in left.primitives:
        for right_coefficient, right_exponent in right.primitives:
            yield (
                left_coefficient * right_coefficient,
                right_exponent,
                _Pair(left_exponent, left.centre, right_exponent, right.centre),
            )


def _overlap(left: AtomicOrbital, right: AtomicOrbital) -> float:
    """Return ``<a|b>``."""

    return sum(
        weight * pair.overlap(left.angular, right.angular)
        for weight, _exponent, pair in _pair_primitives(left, right)
    )


def _kinetic(left: AtomicOrbital, right: AtomicOrbital) -> float:
    """Return ``<a| -1/2 grad^2 |b>``.

    The Laplacian is taken on the right centre through the exact primitive identity

        <a| d^2/dx^2 |b> = j(j-1) S(i,j-2) - 2b(2j+1) S(i,j) + 4b^2 S(i,j+2)

    for the right function's Cartesian index ``j`` along that axis. The same identity on
    the left centre returns the same number, because translation invariance makes
    ``d/dA`` and ``-d/dB`` act identically on the overlap; the tests check that.
    """

    total = 0.0
    for weight, right_exponent, pair in _pair_primitives(left, right):
        accumulated = 0.0
        for axis in range(3):
            index = right.angular[axis]
            lowered = list(right.angular)
            lowered[axis] -= 2
            raised = list(right.angular)
            raised[axis] += 2
            accumulated -= (
                index * (index - 1) * pair.overlap(left.angular, _triple(lowered))
                - 2.0
                * right_exponent
                * (2 * index + 1)
                * pair.overlap(left.angular, right.angular)
                + 4.0 * right_exponent**2 * pair.overlap(left.angular, _triple(raised))
            )
        total += weight * 0.5 * accumulated
    return total


def _differentiate(
    state: dict[tuple[int, int, int, int], float], axis: int, p: float
) -> dict[tuple[int, int, int, int], float]:
    """Return the derivative of a Hermite-potential state along one axis.

    A state is a sparse polynomial in the displacement ``u = P - C`` times a Boys value:
    ``u_x^a u_y^b u_z^c F_k``. Differentiating one such monomial uses
    ``dF_k/du_x = -2 p u_x F_{k+1}``, which is ``F_k' = -F_{k+1}`` written in ``u``.
    """

    out: dict[tuple[int, int, int, int], float] = {}
    for (a, b, c, k), value in state.items():
        powers = [a, b, c]
        if powers[axis] > 0:
            reduced = list(powers)
            reduced[axis] -= 1
            key = (reduced[0], reduced[1], reduced[2], k)
            out[key] = out.get(key, 0.0) + value * powers[axis]
        raised = list(powers)
        raised[axis] += 1
        key = (raised[0], raised[1], raised[2], k + 1)
        out[key] = out.get(key, 0.0) - 2.0 * p * value
    return out


@lru_cache(maxsize=4096)
def _potential_grid(
    order: tuple[int, int, int], p: float
) -> dict[tuple[int, int, int], dict[tuple[int, int, int, int], float]]:
    """Return ``Q[t][u][v]`` as polynomial states, for the Hermite Coulomb expansion.

    ``Q_tuv = d^t_x d^u_y d^v_z F_0(p |u|^2)``, which is what
    ``integral Lambda_tuv(r,P) / |r-C| dr`` reduces to once the angular part is
    integrated out. Building the whole grid once per ``(order, p)`` is what makes the
    three nested sums over Hermite indices cheap.
    """

    nx, ny, nz = order
    grid: dict[tuple[int, int, int], dict[tuple[int, int, int, int], float]] = {
        (0, 0, 0): {(0, 0, 0, 0): 1.0}
    }
    for t in range(nx + 1):
        for u in range(ny + 1):
            for v in range(nz + 1):
                current = grid[(t, u, v)]
                if t < nx:
                    grid[(t + 1, u, v)] = _differentiate(current, 0, p)
                if u < ny:
                    grid[(t, u + 1, v)] = _differentiate(current, 1, p)
                if v < nz:
                    grid[(t, u, v + 1)] = _differentiate(current, 2, p)
    return grid


def _evaluate(
    state: dict[tuple[int, int, int, int], float], u: Vector, boys: tuple[float, ...]
) -> float:
    """Evaluate a Hermite-potential state at a displacement and Boys vector."""

    total = 0.0
    for (a, b, c, k), coefficient in state.items():
        if coefficient == 0.0 or boys[k] == 0.0:
            continue
        total += coefficient * u[0] ** a * u[1] ** b * u[2] ** c * boys[k]
    return total


def _nuclear(
    left: AtomicOrbital, right: AtomicOrbital, charges: tuple[tuple[int, Vector], ...]
) -> float:
    """Return ``<a| sum_C -Z_C / |r - C| |b>``.

    The Coulomb kernel is not a Gaussian, so it has no finite Hermite expansion of the
    overlap kind. The identity used here instead is

        integral Lambda_tuv(r,P) / |r - C| dr = (2 pi / p) Q_tuv,

    with ``Q_tuv`` the Hermite derivatives of ``F_0`` evaluated at ``p |P - C|^2``. That
    is what makes the ``p`` block exact; the ``(-1)^n R_n`` single sum used elsewhere is
    only valid when the whole angular momentum sits on one direction.
    """

    order = tuple(left.angular[axis] + right.angular[axis] for axis in range(3))
    highest = sum(order)
    total = 0.0
    for weight, _exponent, pair in _pair_primitives(left, right):
        ex = pair.coefficients(0, left.angular[0], right.angular[0])
        ey = pair.coefficients(1, left.angular[1], right.angular[1])
        ez = pair.coefficients(2, left.angular[2], right.angular[2])
        grid = _potential_grid(order, pair.p)
        for atomic_number, centre in charges:
            u = _sub(pair.centre, centre)
            boys = _boys(highest, pair.p * _dot(u, u))
            accumulated = 0.0
            for t in range(order[0] + 1):
                if ex[t] == 0.0:
                    continue
                for s in range(order[1] + 1):
                    if ey[s] == 0.0:
                        continue
                    for w in range(order[2] + 1):
                        if ez[w] == 0.0:
                            continue
                        accumulated += (
                            ex[t] * ey[s] * ez[w] * _evaluate(grid[(t, s, w)], u, boys)
                        )
            total += (
                weight
                * (-2.0 * math.pi / pair.p)
                * pair.k
                * atomic_number
                * accumulated
            )
    return total


def _electron_repulsion(
    left: AtomicOrbital,
    right: AtomicOrbital,
    ket_left: AtomicOrbital,
    ket_right: AtomicOrbital,
) -> float:
    """Return the chemist-ordered ``(ab|cd)``.

    Both pairs are expanded in Hermite Gaussians, which is a finite expansion, and the
    Coulomb coupling of two Hermite Gaussians is a Hermite derivative of ``F_0``:

        (ab|cd) = 2 pi^(5/2) / (p q sqrt(p+q)) K_ab K_cd
                  sum_{T,U,V} c_T d_U e_V d^T_x d^U_y d^V_z F_0(rho |P-Q|^2)

    The sign belongs to the ket pair's Hermite index, ``(-1)^(u_x+u_y+u_z)``, because the
    ``Lambda_t`` basis is defined with an alternating sign on the two centres. Applying
    the sign to the *combined* index instead is what the naive single-sum form does, and
    it is wrong for every integral whose angular momentum is split between the pairs.
    """

    total = 0.0
    for left_weight, _left_exponent, left_pair in _pair_primitives(left, right):
        for right_weight, _right_exponent, right_pair in _pair_primitives(
            ket_left, ket_right
        ):
            order = tuple(
                left.angular[axis]
                + right.angular[axis]
                + ket_left.angular[axis]
                + ket_right.angular[axis]
                for axis in range(3)
            )
            rho = left_pair.p * right_pair.p / (left_pair.p + right_pair.p)
            displacement = _sub(left_pair.centre, right_pair.centre)
            boys = _boys(sum(order), rho * _dot(displacement, displacement))
            grid = _potential_grid(order, rho)
            accumulated = 0.0
            bra_x = left_pair.coefficients(0, left.angular[0], right.angular[0])
            bra_y = left_pair.coefficients(1, left.angular[1], right.angular[1])
            bra_z = left_pair.coefficients(2, left.angular[2], right.angular[2])
            ket_x = right_pair.coefficients(
                0, ket_left.angular[0], ket_right.angular[0]
            )
            ket_y = right_pair.coefficients(
                1, ket_left.angular[1], ket_right.angular[1]
            )
            ket_z = right_pair.coefficients(
                2, ket_left.angular[2], ket_right.angular[2]
            )
            for tx in range(len(bra_x)):
                if bra_x[tx] == 0.0:
                    continue
                for ux in range(len(ket_x)):
                    if ket_x[ux] == 0.0:
                        continue
                    for ty in range(len(bra_y)):
                        if bra_y[ty] == 0.0:
                            continue
                        for uy in range(len(ket_y)):
                            if ket_y[uy] == 0.0:
                                continue
                            for tz in range(len(bra_z)):
                                if bra_z[tz] == 0.0:
                                    continue
                                for uz in range(len(ket_z)):
                                    if ket_z[uz] == 0.0:
                                        continue
                                    sign = -1.0 if (ux + uy + uz) % 2 else 1.0
                                    accumulated += (
                                        bra_x[tx]
                                        * ket_x[ux]
                                        * bra_y[ty]
                                        * ket_y[uy]
                                        * bra_z[tz]
                                        * ket_z[uz]
                                        * sign
                                        * _evaluate(
                                            grid[(tx + ux, ty + uy, tz + uz)],
                                            displacement,
                                            boys,
                                        )
                                    )
            prefactor = (
                2.0
                * math.pi**2.5
                / (left_pair.p * right_pair.p * math.sqrt(left_pair.p + right_pair.p))
            )
            total += (
                left_weight
                * right_weight
                * prefactor
                * left_pair.k
                * right_pair.k
                * accumulated
            )
    return total


@dataclass(frozen=True, slots=True)
class MolecularIntegrals:
    """The one- and two-electron integrals of a molecule in an atomic-orbital basis.

    ``overlap``, ``kinetic`` and ``nuclear_attraction`` are ``(n, n)`` and
    ``core_hamiltonian`` is their sum. ``two_electron`` is ``(n, n, n, n)`` in the
    chemist convention, ``two_electron[i, j, k, l] = (ij|kl)``.

    Examples:
        >>> from flagquantum.algorithms.chemistry_integrals import molecular_integrals
        >>> integrals = molecular_integrals([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))])
        >>> integrals.n_orbitals
        2
        >>> round(float(integrals.overlap[0, 1]), 6)
        0.658957
    """

    geometry: MolecularGeometry
    basis: str
    atomic_orbitals: tuple[AtomicOrbital, ...]
    overlap: torch.Tensor
    kinetic: torch.Tensor
    nuclear_attraction: torch.Tensor
    two_electron: torch.Tensor
    nuclear_repulsion: float

    @property
    def n_orbitals(self) -> int:
        """Return the number of contracted basis functions."""

        return len(self.atomic_orbitals)

    @property
    def core_hamiltonian(self) -> torch.Tensor:
        """Return the one-electron Hamiltonian ``h = T + V``."""

        return self.kinetic + self.nuclear_attraction


def _atomic_orbitals(
    geometry: MolecularGeometry, basis: str
) -> tuple[AtomicOrbital, ...]:
    """Return the contracted Cartesian functions of the geometry, in shell order."""

    if basis not in BASIS_SETS:
        raise CapabilityError(
            f"the basis set {basis!r} is not available; this module expands "
            f"{', '.join(BASIS_SETS)}"
        )
    orbitals: list[AtomicOrbital] = []
    for symbol, centre in zip(geometry.symbols, geometry.in_bohr(), strict=True):
        for shell, contraction in _STO3G[symbol]:
            angles = _S_ANGULAR if shell == "s" else _P_ANGULAR
            for angular in angles:
                primitives = tuple(
                    (coefficient * _primitive_norm(exponent, angular), exponent)
                    for exponent, coefficient in contraction
                )
                orbitals.append(AtomicOrbital(symbol, centre, angular, primitives))
    return tuple(orbitals)


def molecular_integrals(
    geometry: Iterable[tuple[str, Sequence[float]]] | MolecularGeometry,
    *,
    basis: str = "sto-3g",
    dtype: torch.dtype = torch.float64,
) -> MolecularIntegrals:
    """Return the one- and two-electron integrals of ``geometry`` in ``basis``.

    The geometry is an iterable of ``(element, (x, y, z))`` pairs in angstroms, or a
    :class:`MolecularGeometry`. Two-electron integrals are placed with all eight
    permutation symmetries of ``(ij|kl)`` filled from one evaluation.

    Examples:
        >>> from flagquantum.algorithms.chemistry_integrals import molecular_integrals
        >>> integrals = molecular_integrals(
        ...     [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.757, 0.587)), ("H", (0.0, -0.757, 0.587))]
        ... )
        >>> integrals.n_orbitals
        7
        >>> round(integrals.nuclear_repulsion, 6)
        9.188258
    """

    if isinstance(geometry, MolecularGeometry):
        molecule = geometry
    else:
        molecule = MolecularGeometry.from_angstrom(geometry)
    orbitals = _atomic_orbitals(molecule, basis)
    n = len(orbitals)
    overlap = torch.zeros((n, n), dtype=dtype)
    kinetic = torch.zeros((n, n), dtype=dtype)
    attraction = torch.zeros((n, n), dtype=dtype)
    charges = molecule.charges()
    for i in range(n):
        for j in range(i, n):
            s = _overlap(orbitals[i], orbitals[j])
            t = _kinetic(orbitals[i], orbitals[j])
            v = _nuclear(orbitals[i], orbitals[j], charges)
            overlap[i, j] = overlap[j, i] = s
            kinetic[i, j] = kinetic[j, i] = t
            attraction[i, j] = attraction[j, i] = v
    two_electron = torch.zeros((n, n, n, n), dtype=dtype)
    for i in range(n):
        for j in range(i + 1):
            for k in range(i + 1):
                # The pair (k, l) is walked only up to the pair (i, j) in this order, so
                # exactly one of the four index orderings that name this (ij|kl) class is
                # evaluated and the other three read their value from the table.
                for ell in range(k + 1):
                    if k == i and ell > j:
                        continue
                    value = _electron_repulsion(
                        orbitals[i], orbitals[j], orbitals[k], orbitals[ell]
                    )
                    base = (i, j, k, ell)
                    for order in _ERI_SYMMETRY:
                        two_electron[
                            base[order[0]],
                            base[order[1]],
                            base[order[2]],
                            base[order[3]],
                        ] = value
    return MolecularIntegrals(
        geometry=molecule,
        basis=basis,
        atomic_orbitals=orbitals,
        overlap=overlap,
        kinetic=kinetic,
        nuclear_attraction=attraction,
        two_electron=two_electron,
        nuclear_repulsion=molecule.nuclear_repulsion(),
    )
