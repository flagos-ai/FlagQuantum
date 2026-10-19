"""Unit tests for the native STO-3G Gaussian-integral engine.

``flagquantum.algorithms.chemistry_integrals`` is the half of the molecular
driver that turns a geometry into one- and two-electron integrals. It is the
replacement for the OpenFermion/PySCF middle layer CUDA-Q delegates to, so it
has to be checked against mathematics rather than against another library's
output.

**What this file proves**

1. The contracted engine reproduces an independently written closed-form
   contraction of ``s`` primitives: the overlap, kinetic, nuclear-attraction
   and all four two-electron classes of H2 at 1.4 bohr, to a measured relative
   error of ``2.9e-16``.
2. The ``p`` and ``d`` classes are the derivative content they claim to be:
   every one agrees with the exact Cartesian raising operator applied to the
   verified ``s``-only closed forms, to a measured worst relative error of
   ``3.1e-10`` over the ``s``, ``p`` and ``d`` classes.
3. The Boys function agrees with a 400-point Gauss-Legendre quadrature of its
   own definition over ``T`` in ``[0, 10000]``, to a measured worst relative
   error of ``6.3e-15``, which is computed in this file and shares no code with
   either recursion. The sweep crosses the ``2T = 16 (n_max + 1)`` point where
   the engine changes route, and it reaches ``T = 1260``, where the downward
   route alone overflows to ``nan``.
4. The two-electron table carries all eight index orders of ``(ij|kl)`` with a
   spread of exactly ``0.0``, so no permutation can read a half-filled slot.
5. The integrals are invariant under a rigid translation of the whole molecule
   (measured worst ``1.4e-14``) and, for an ``s``-only molecule, under a rigid
   rotation (bitwise ``0.0`` for the overlap and the core Hamiltonian). For a
   molecule with ``p`` functions the tensor is not elementwise rotation
   invariant, and the test says so: the shell-block scalars are invariant
   (measured worst ``7.1e-14``) while the elementwise difference is measured
   above ``0.1``, which is the control that keeps the first claim honest.
6. The structural invariants hold: the matrices are symmetric, the core
   Hamiltonian is the kinetic plus the attraction, every overlap eigenvalue is
   positive, and the basis-function layout is the one STO-3G defines.
7. Every refusal condition refuses, with the message that names the condition.

**What this file does not prove**

1. Nothing here covers an atom heavier than neon, a basis set other than
   STO-3G, or a function of angular momentum above ``d``: the engine does not
   claim any of them.
2. The ``s``-only closed forms used as the reference are themselves verified
   only against the engine in one direction — the H2 table and the coincident
   four-``s`` value anchor them, but this file does not derive them from the
   Gaussian product theorem step by step.
3. Accuracy in the *contracted* basis is checked on H2 alone, because H2 is the
   only molecule in the standard set whose basis is entirely ``s`` functions.
4. The cost of the dense ``n^4`` accumulation is not measured here, and no test
   in this file would fail if it became ten times slower.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from flagquantum.algorithms.chemistry_integrals import (
    _ERI_SYMMETRY,
    _STO3G,
    ATOMIC_NUMBERS,
    BASIS_SETS,
    BOHR_RADIUS,
    AtomicOrbital,
    MolecularGeometry,
    MolecularIntegrals,
    _boys,
    _electron_repulsion,
    _kinetic,
    _nuclear,
    _overlap,
    _primitive_norm,
    molecular_integrals,
)
from flagquantum.errors import CapabilityError, ValidationError

pytestmark = pytest.mark.unit


# --- geometries, in angstroms, shared with the molecular driver's tests ---


def _h2(separation: float = 0.7414) -> list[tuple[str, tuple[float, float, float]]]:
    return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, separation))]


def _lih(separation: float = 1.595) -> list[tuple[str, tuple[float, float, float]]]:
    return [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, separation))]


def _hf(separation: float = 0.917) -> list[tuple[str, tuple[float, float, float]]]:
    return [("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, separation))]


def _h2o(
    separation: float = 0.9893, angle: float = 104.52
) -> list[tuple[str, tuple[float, float, float]]]:
    half = math.radians(angle) / 2.0
    x = separation * math.sin(half)
    z = separation * math.cos(half)
    return [("O", (0.0, 0.0, 0.0)), ("H", (x, 0.0, z)), ("H", (-x, 0.0, z))]


def _nh3(
    separation: float = 1.0124, angle: float = 106.67
) -> list[tuple[str, tuple[float, float, float]]]:
    cosine = math.cos(math.radians(angle))
    # cos(angle) = cos^2(t) - sin^2(t) / 2, so cos^2(t) = (1 + 2 cos(angle)) / 3.
    axial = math.sqrt((1.0 + 2.0 * cosine) / 3.0)
    radial = separation * math.sqrt(1.0 - axial * axial)
    hydrogens = [
        (
            "H",
            (
                radial * math.cos(2.0 * math.pi * step / 3.0),
                radial * math.sin(2.0 * math.pi * step / 3.0),
                -separation * axial,
            ),
        )
        for step in range(3)
    ]
    return [("N", (0.0, 0.0, 0.0)), *hydrogens]


def _ch4(separation: float = 1.0915) -> list[tuple[str, tuple[float, float, float]]]:
    scale = separation / math.sqrt(3.0)
    return [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (scale, scale, scale)),
        ("H", (-scale, -scale, scale)),
        ("H", (-scale, scale, -scale)),
        ("H", (scale, -scale, -scale)),
    ]


def _n2(separation: float = 1.0977) -> list[tuple[str, tuple[float, float, float]]]:
    return [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, separation))]


# --- independent closed forms for contracted ``s`` primitives ---
#
# These are the textbook Gaussian product-theorem results written out by hand, in
# terms of the *unnormalised* primitive ``exp(-a |r - A|^2)``. The engine works
# in normalised primitives, so the primitive normalisation is applied here too.
# Nothing in this block calls into the module under test.

# The published STO-3G contraction of hydrogen: exponent, coefficient.
_STO3G_HYDROGEN = (
    (3.42525091, 0.15432897),
    (0.62391373, 0.53532814),
    (0.1688554, 0.44463454),
)

_SPACING = 1e-3


def _primitive_normalisation(exponent: float) -> float:
    return math.pow(2.0 * exponent / math.pi, 0.75)


def _square(value: tuple[float, float, float]) -> float:
    return value[0] * value[0] + value[1] * value[1] + value[2] * value[2]


def _difference(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _boys_zero(argument: float) -> float:
    if argument == 0.0:
        return 1.0
    return 0.5 * math.sqrt(math.pi / argument) * math.erf(math.sqrt(argument))


def _s_overlap(
    a: float,
    centre_a: tuple[float, float, float],
    b: float,
    centre_b: tuple[float, float, float],
) -> float:
    exponent = a + b
    reduced = a * b / exponent
    return (math.pi / exponent) ** 1.5 * math.exp(
        -reduced * _square(_difference(centre_a, centre_b))
    )


def _s_kinetic(
    a: float,
    centre_a: tuple[float, float, float],
    b: float,
    centre_b: tuple[float, float, float],
) -> float:
    exponent = a + b
    reduced = a * b / exponent
    separation = _square(_difference(centre_a, centre_b))
    return (
        reduced
        * (3.0 - 2.0 * reduced * separation)
        * (math.pi / exponent) ** 1.5
        * math.exp(-reduced * separation)
    )


def _s_attraction(
    a: float,
    centre_a: tuple[float, float, float],
    b: float,
    centre_b: tuple[float, float, float],
    charge_centre: tuple[float, float, float],
) -> float:
    exponent = a + b
    reduced = a * b / exponent
    midpoint = tuple(
        (a * centre_a[axis] + b * centre_b[axis]) / exponent for axis in range(3)
    )
    argument = exponent * _square(_difference(midpoint, charge_centre))
    return (
        -(2.0 * math.pi / exponent)
        * math.exp(-reduced * _square(_difference(centre_a, centre_b)))
        * _boys_zero(argument)
    )


def _s_repulsion(
    a: float,
    centre_a: tuple[float, float, float],
    b: float,
    centre_b: tuple[float, float, float],
    c: float,
    centre_c: tuple[float, float, float],
    d: float,
    centre_d: tuple[float, float, float],
) -> float:
    left = a + b
    right = c + d
    reduced_left = a * b / left
    reduced_right = c * d / right
    combined = left * right / (left + right)
    midpoint_left = tuple(
        (a * centre_a[axis] + b * centre_b[axis]) / left for axis in range(3)
    )
    midpoint_right = tuple(
        (c * centre_c[axis] + d * centre_d[axis]) / right for axis in range(3)
    )
    argument = combined * _square(_difference(midpoint_left, midpoint_right))
    return (
        (2.0 * math.pi**2.5)
        / (left * right * math.sqrt(left + right))
        * math.exp(
            -reduced_left * _square(_difference(centre_a, centre_b))
            - reduced_right * _square(_difference(centre_c, centre_d))
        )
        * _boys_zero(argument)
    )


def _contract(two_centre):
    return sum(
        left_coefficient
        * _primitive_normalisation(left_exponent)
        * right_coefficient
        * _primitive_normalisation(right_exponent)
        * two_centre(left_exponent, right_exponent)
        for left_exponent, left_coefficient in _STO3G_HYDROGEN
        for right_exponent, right_coefficient in _STO3G_HYDROGEN
    )


def _contract_four(four_centre, centres):
    return sum(
        left_coefficient
        * _primitive_normalisation(left_exponent)
        * inner_left_coefficient
        * _primitive_normalisation(inner_left_exponent)
        * inner_right_coefficient
        * _primitive_normalisation(inner_right_exponent)
        * right_coefficient
        * _primitive_normalisation(right_exponent)
        * four_centre(
            left_exponent,
            centres[0],
            inner_left_exponent,
            centres[1],
            inner_right_exponent,
            centres[2],
            right_exponent,
            centres[3],
        )
        for left_exponent, left_coefficient in _STO3G_HYDROGEN
        for inner_left_exponent, inner_left_coefficient in _STO3G_HYDROGEN
        for inner_right_exponent, inner_right_coefficient in _STO3G_HYDROGEN
        for right_exponent, right_coefficient in _STO3G_HYDROGEN
    )


# --- the raising operator that turns the ``s`` closed forms into ``p`` and ``d`` ---
#
# ``d/dA_x g_l = 2a g_{l+e_x} - l_x g_{l-e_x}`` is the exact Cartesian identity,
# so ``g_{l+e_x} = (d/dA_x g_l + l_x g_{l-e_x}) / (2a)``. That inverts the
# derivative relation into a construction: raise one unit at a time, taking a
# numerical derivative of the state below and adding its lowering partner. The
# states below are built in the order the loop walks the axes, so the lowering
# partner always exists by the time it is needed.


def _central_difference(state, axis: int):
    def derivative(centre):
        def shifted(step):
            moved = list(centre)
            moved[axis] += step * _SPACING
            return tuple(moved)

        return (
            -state(shifted(2))
            + 8.0 * state(shifted(1))
            - 8.0 * state(shifted(-1))
            + state(shifted(-2))
        ) / (12.0 * _SPACING)

    return derivative


def _raised_state(closed_form, angular: tuple[int, int, int], exponent: float):
    states = {(0, 0, 0): closed_form}
    current = (0, 0, 0)
    for axis in range(3):
        for _ in range(angular[axis]):
            lowered = list(current)
            lowered[axis] -= 1
            raised = list(current)
            raised[axis] += 1
            derivative = _central_difference(states[current], axis)
            if current[axis] > 0:
                partner = states[tuple(lowered)]
                index = current[axis]

                def step(centre, d=derivative, p=partner, k=index, a=exponent):
                    return (d(centre) + k * p(centre)) / (2.0 * a)

            else:

                def step(centre, d=derivative, a=exponent):
                    return d(centre) / (2.0 * a)

            current = tuple(raised)
            states[current] = step
    return states[angular]


def _primitive(exponent: float, centre, angular) -> AtomicOrbital:
    """One bare Cartesian Gaussian, with no contraction and no normalisation."""

    return AtomicOrbital("H", centre, angular, ((1.0, exponent),))


# --- section 1: the declared surface ---


def test_the_module_declares_exactly_the_seven_documented_names() -> None:
    from flagquantum.algorithms import chemistry_integrals

    assert chemistry_integrals.__all__ == (
        "ATOMIC_NUMBERS",
        "BASIS_SETS",
        "BOHR_RADIUS",
        "AtomicOrbital",
        "MolecularGeometry",
        "MolecularIntegrals",
        "molecular_integrals",
    )


def test_only_sto3g_is_offered() -> None:
    assert BASIS_SETS == ("sto-3g",)
    assert BOHR_RADIUS == 0.529177210903


def test_the_element_table_covers_hydrogen_through_neon() -> None:
    assert ATOMIC_NUMBERS == {
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


def test_the_contraction_table_holds_the_published_sto3g_hydrogen_exponents() -> None:
    """The hydrogen contraction is published data, so it is checked, not assumed."""

    contraction = _STO3G["H"]
    assert [shell for shell, _ in contraction] == ["s"]
    primitives = contraction[0][1]
    assert len(primitives) == 3
    for measured, published in zip(primitives, _STO3G_HYDROGEN, strict=True):
        assert measured[0] == pytest.approx(published[0], rel=1e-12)
        assert measured[1] == pytest.approx(published[1], rel=1e-12)


@pytest.mark.parametrize(
    ("element", "expected"),
    [
        ("H", 1),
        ("He", 1),
        ("Li", 5),
        ("Be", 5),
        ("B", 5),
        ("C", 5),
        ("N", 5),
        ("O", 5),
        ("F", 5),
        ("Ne", 5),
    ],
)
def test_every_listed_element_expands_to_its_sto3g_shells(
    element: str, expected: int
) -> None:
    integrals = molecular_integrals([(element, (0.0, 0.0, 0.0))])
    assert integrals.n_orbitals == expected
    angular = [orbital.angular for orbital in integrals.atomic_orbitals]
    if element in ("H", "He"):
        assert angular == [(0, 0, 0)]
    else:
        assert angular == [
            (0, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ]


def test_the_shell_order_follows_the_geometry() -> None:
    integrals = molecular_integrals(_h2o())
    assert [orbital.element for orbital in integrals.atomic_orbitals] == [
        "O",
        "O",
        "O",
        "O",
        "O",
        "H",
        "H",
    ]
    assert [orbital.angular for orbital in integrals.atomic_orbitals] == [
        (0, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, 0),
        (0, 0, 0),
    ]


# --- section 2: the eight index orders of the two-electron table ---


@pytest.mark.parametrize("factory", [_h2o, _nh3, _ch4])
def test_the_two_electron_table_fills_all_eight_index_orders(factory) -> None:
    """One evaluated class must land in all eight slots it names, bitwise.

    The eight orders are the two swaps inside each pair and the swap of the two
    pairs. If any slot were left at zero, or filled from a second evaluation,
    the spread over the eight would be non-zero; it is exactly ``0.0``.

    That spread cannot see the failure mode the pair walk's skip rule is one
    step away from, because a class that is never evaluated is *uniformly*
    zero: all eight slots agree at ``0.0``. Positivity is the assertion that
    separates the two, and it is a property of the integral rather than of the
    bookkeeping — ``(ij|ij)`` is ``|phi_i phi_j|^2`` against the Coulomb kernel,
    so it is strictly positive for every ordered pair of these functions, and
    the walk has to have evaluated each one.
    """

    integrals = molecular_integrals(factory())
    n = integrals.n_orbitals
    torch.manual_seed(20250608)
    quadruples = torch.randint(0, n, (80, 4)).tolist()
    worst = 0.0
    for i, j, k, l in quadruples:  # noqa: E741
        base = (i, j, k, l)
        values = [
            float(
                integrals.two_electron[
                    base[order[0]], base[order[1]], base[order[2]], base[order[3]]
                ]
            )
            for order in _ERI_SYMMETRY
        ]
        worst = max(worst, max(values) - min(values))
    assert worst == 0.0
    for i in range(n):
        for j in range(n):
            assert float(integrals.two_electron[i, j, i, j]) > 0.0


def test_the_eight_orders_are_the_eight_distinct_permutations_they_claim() -> None:
    """The table must name eight *different* slots, not one slot eight times."""

    assert len(set(_ERI_SYMMETRY)) == 8
    for order in _ERI_SYMMETRY:
        assert sorted(order) == [0, 1, 2, 3]
    # The two swaps inside the pairs and the swap of the pairs generate exactly
    # these eight, so eight is the whole orbit of (0, 1, 2, 3).
    generators = ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1))
    orbit = {(0, 1, 2, 3)}
    frontier = [(0, 1, 2, 3)]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = tuple(current[generator[position]] for position in range(4))
            if candidate not in orbit:
                orbit.add(candidate)
                frontier.append(candidate)
    assert set(_ERI_SYMMETRY) == orbit


# --- section 3: the Boys function against an independent quadrature ---


def _gauss_legendre(nodes: int) -> tuple[list[float], list[float]]:
    """Return the nodes and weights of ``nodes``-point Gauss-Legendre on [0, 1]."""

    abscissas: list[float] = []
    weights: list[float] = []
    for index in range(1, nodes + 1):
        root = math.cos(math.pi * (index - 0.25) / (nodes + 0.5))
        derivative = 0.0
        for _ in range(100):
            previous, current = 1.0, root
            for order in range(2, nodes + 1):
                previous, current = (
                    current,
                    ((2 * order - 1) * root * current - (order - 1) * previous) / order,
                )
            derivative = nodes * (root * current - previous) / (root * root - 1.0)
            step = -current / derivative
            root += step
            if abs(step) < 1e-16:
                break
        abscissas.append(0.5 * (root + 1.0))
        weights.append(1.0 / ((1.0 - root * root) * derivative * derivative))
    return abscissas, weights


def _boys_by_quadrature(order: int, argument: float) -> float:
    """Return ``F_n(t) = integral_0^1 x^(2n) exp(-t x^2) dx`` by quadrature."""

    if argument == 0.0:
        return 1.0 / (2 * order + 1)
    abscissas, weights = _BOYS_QUADRATURE
    return sum(
        weight * abscissa ** (2 * order) * math.exp(-argument * abscissa * abscissa)
        for abscissa, weight in zip(abscissas, weights, strict=True)
    )


_BOYS_QUADRATURE = _gauss_legendre(400)


def test_the_boys_function_matches_a_four_hundred_point_quadrature() -> None:
    """Both routes of the recursion are checked against its own definition.

    ``F_n(t) = integral_0^1 x^(2n) exp(-t x^2) dx`` is integrated with a
    400-point Gauss-Legendre rule, which knows nothing about the recursion, the
    seed, the renormalisation, or the argument at which the engine stops walking
    downward. The sweep runs well past both the ``2t = 16 (n_max + 1) = 56``
    crossover and ``t = 1260``, where the downward route on its own overflows to
    ``nan``; every value is required to be finite before it is compared, so a
    regression to the single-route engine fails here rather than passing as a
    ``nan``-skipping comparison. The bound is the one this test measured.
    """

    worst = 0.0
    worst_at = (0, 0.0)
    arguments = (
        0.0,
        1e-6,
        1e-4,
        1e-2,
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        55.0,
        56.0,
        60.0,
        100.0,
        200.0,
        400.0,
        700.0,
        1000.0,
        1260.0,
        2000.0,
        5000.0,
        10000.0,
    )
    for argument in arguments:
        measured = _boys(6, argument)
        assert len(measured) == 7
        assert all(
            math.isfinite(value) for value in measured
        ), f"the Boys functions are not finite at t={argument}: {measured}"
        for order in range(7):
            reference = _boys_by_quadrature(order, argument)
            if reference == 0.0:
                continue
            relative = abs(measured[order] - reference) / abs(reference)
            if relative > worst:
                worst = relative
                worst_at = (order, argument)
    assert worst < 1e-13, f"worst relative error {worst:.3e} at {worst_at}"


def test_the_two_boys_routes_agree_where_the_engine_switches_between_them() -> None:
    """A route taken by argument must not put a step in a smooth function.

    The engine walks the three-term relation downward below ``2t = 16 (n_max + 1)``
    and upward above it. Evaluating either side of that crossover separately, by
    calling the private recursion under a temporarily patched threshold, would
    make this test depend on the threshold; instead the two routes are compared
    where they overlap. The upward route reimplemented here from the exact
    ``F_0`` is the independent side: it shares no code with the engine and is
    used on both sides of the switch, so a step at the switch shows up as a
    disagreement on the side the engine no longer uses.
    """

    def upward(n_max: int, t: float) -> tuple[float, ...]:
        values = [_boys_zero(t)]
        decay = math.exp(-t) if t < 700.0 else 0.0
        for order in range(n_max):
            values.append(((2 * order + 1) * values[order] - decay) / (2.0 * t))
        return tuple(values)

    worst = 0.0
    for n_max, crossover in ((0, 8.0), (1, 16.0), (3, 32.0), (6, 56.0), (10, 88.0)):
        for argument in (
            crossover / 2.0,
            crossover - 1e-9,
            crossover + 1e-9,
            4.0 * crossover,
        ):
            measured = _boys(n_max, argument)
            reference = upward(n_max, argument)
            for got, want in zip(measured, reference, strict=True):
                assert want != 0.0
                worst = max(worst, abs(got - want) / abs(want))
    assert worst < 1e-9, f"worst relative disagreement between the routes {worst:.3e}"


def test_the_boys_recursion_starts_from_the_exact_zeroth_order() -> None:
    """``F_0`` is closed form, so the returned tuple is anchored besides by quadrature."""

    for argument in (0.0, 0.25, 1.0, 4.0, 25.0, 100.0):
        assert _boys(0, argument)[0] == pytest.approx(
            _boys_zero(argument), rel=1e-14, abs=1e-300
        )


def test_the_boys_function_refuses_a_negative_argument() -> None:
    with pytest.raises(ValidationError, match="non-negative argument"):
        _boys(3, -0.5)


# --- section 4: the contracted engine against a closed-form contraction ---


def _h2_at_one_point_four_bohr() -> MolecularIntegrals:
    """H2 at the classic 1.4 bohr separation, reached by converting on construction.

    ``MolecularGeometry`` reads its coordinates as angstroms, so the bohr
    separation is divided by ``BOHR_RADIUS`` before it is handed over.
    """

    return molecular_integrals(
        MolecularGeometry(("H", "H"), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.4 * BOHR_RADIUS)))
    )


def test_the_contracted_engine_reproduces_a_closed_form_contraction() -> None:
    """Every H2/STO-3G integral is compared to a hand-written contraction.

    The reference applies the Gaussian product theorem to the published
    exponent/coefficient pairs directly. It shares no code with the engine: no
    Hermite expansion, no Boys recursion, no symmetry filling. The bound is the
    one this test measured.
    """

    integrals = _h2_at_one_point_four_bohr()
    origin = (0.0, 0.0, 0.0)
    other = (0.0, 0.0, 1.4)
    reference = {
        "overlap diagonal": _contract(lambda a, b: _s_overlap(a, origin, b, origin)),
        "overlap mixed": _contract(lambda a, b: _s_overlap(a, origin, b, other)),
        "kinetic diagonal": _contract(lambda a, b: _s_kinetic(a, origin, b, origin)),
        "kinetic mixed": _contract(lambda a, b: _s_kinetic(a, origin, b, other)),
        "attraction diagonal": _contract(
            lambda a, b: (
                _s_attraction(a, origin, b, origin, origin)
                + _s_attraction(a, origin, b, origin, other)
            )
        ),
        "attraction mixed": _contract(
            lambda a, b: (
                _s_attraction(a, origin, b, other, origin)
                + _s_attraction(a, origin, b, other, other)
            )
        ),
        "(11|11)": _contract_four(_s_repulsion, (origin, origin, origin, origin)),
        "(11|22)": _contract_four(_s_repulsion, (origin, origin, other, other)),
        "(12|21)": _contract_four(_s_repulsion, (origin, other, other, origin)),
    }
    measured = {
        "overlap diagonal": float(integrals.overlap[0, 0]),
        "overlap mixed": float(integrals.overlap[0, 1]),
        "kinetic diagonal": float(integrals.kinetic[0, 0]),
        "kinetic mixed": float(integrals.kinetic[0, 1]),
        "attraction diagonal": float(integrals.nuclear_attraction[0, 0]),
        "attraction mixed": float(integrals.nuclear_attraction[0, 1]),
        "(11|11)": float(integrals.two_electron[0, 0, 0, 0]),
        "(11|22)": float(integrals.two_electron[0, 0, 1, 1]),
        "(12|21)": float(integrals.two_electron[0, 1, 1, 0]),
    }
    worst = 0.0
    for label, wanted in reference.items():
        relative = abs(measured[label] - wanted) / abs(wanted)
        worst = max(worst, relative)
        assert relative < 1e-12, f"{label}: {measured[label]} vs {wanted}"
    assert worst < 3e-16, f"worst relative error {worst:.3e}"


def test_the_h2_integrals_carry_the_tabulated_four_decimals() -> None:
    """The four decimals the H2/STO-3G integrals are tabulated with.

    ``(12|21)`` is deliberately not asserted: the value quoted in the literature
    for a "minimal H2" changes with the basis, and STO-3G gives ``0.2970``, not
    the ``0.1780`` that belongs to a different contraction. The closed-form test
    above is the evidence for it.
    """

    integrals = _h2_at_one_point_four_bohr()
    assert round(float(integrals.overlap[0, 1]), 4) == 0.6593
    assert round(float(integrals.core_hamiltonian[0, 0]), 4) == -1.1204
    assert round(float(integrals.core_hamiltonian[0, 1]), 4) == -0.9584
    assert round(float(integrals.two_electron[0, 0, 0, 0]), 4) == 0.7746
    assert round(float(integrals.two_electron[0, 0, 1, 1]), 4) == 0.5697
    assert round(integrals.nuclear_repulsion, 4) == 0.7143


def test_the_self_normalised_hydrogen_orbital_is_normalised() -> None:
    """A contraction built through the engine's own normalisation has unit overlap.

    ``S_11 = 1`` is a real check: the primitive normalisation, the contraction
    coefficients and the overlap kernel all have to be consistent for it to come
    out, and it is the one place where the engine's normalisation convention is
    observable on its own. The published STO-3G coefficients are rounded to
    eight decimals, which is why this is not exactly one.
    """

    integrals = _h2_at_one_point_four_bohr()
    assert float(integrals.overlap[0, 0]) == pytest.approx(1.0, abs=1e-8)
    assert float(integrals.overlap[1, 1]) == float(integrals.overlap[0, 0])


def test_the_coincident_four_s_repulsion_is_the_closed_form_value() -> None:
    """Four coincident unit Gaussian ``s`` functions have ``(ss|ss) = pi^2.5 / 4``."""

    coincident = _primitive(1.0, (0.0, 0.0, 0.0), (0, 0, 0))
    measured = _electron_repulsion(coincident, coincident, coincident, coincident)
    assert measured == pytest.approx(math.pi**2.5 / 4.0, rel=1e-14)


# --- section 5: the p and d classes against the exact raising operator ---


@pytest.mark.parametrize(
    "angular",
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (0, 0, 2),
    ],
)
def test_every_p_and_d_class_follows_from_the_s_only_closed_forms(
    angular: tuple[int, int, int],
) -> None:
    """The ``p`` and ``d`` kernels are the derivative content they claim to be.

    A Cartesian Gaussian of higher angular momentum is a fixed multiple of a
    derivative of the ``s`` function at its centre, so the exact raising
    operator inverts the derivative identity into a construction that reaches
    ``p`` and ``d`` from the ``s``-only closed forms the tests above already
    pin.

    Which centre the raising acts at is part of the contract and differs per
    kernel. Overlap, the kinetic energy and the nuclear attraction are assembled
    from the *left* function's angular momentum here, and the electron repulsion
    from the bra pair's. The kinetic energy is the exception the sequence of
    assertions below exists to catch: its identity is the one-dimensional
    Laplacian on the *right* pair's second centre, so raising the left function
    would agree with its own reference for the wrong reason. The repulsion is
    checked on both sides, because its ``(-1)^(u_x+u_y+u_z)`` belongs to the ket
    pair and a sign applied to the combined index instead is invisible to a
    single-side check.
    """

    exponent = 3.425
    partner_exponent = 0.623
    first = (0.11, -0.27, 0.43)
    second = (1.31, 0.62, -0.18)
    third = (-0.44, 1.05, 0.77)
    fourth = (0.9, -0.5, 1.4)
    exponent_third = 1.117
    exponent_fourth = 2.204
    raised_left = _primitive(exponent, first, angular)
    raised_right = _primitive(partner_exponent, second, angular)
    raised_ket = _primitive(exponent_third, third, angular)
    bare_left = _primitive(exponent, first, (0, 0, 0))
    partner = _primitive(partner_exponent, second, (0, 0, 0))
    third_orbital = _primitive(exponent_third, third, (0, 0, 0))
    fourth_orbital = _primitive(exponent_fourth, fourth, (0, 0, 0))

    # Each entry is (measured, closed form, the centre the raise acts at, the
    # exponent of the function being raised at that centre).
    cases = {
        "overlap": (
            _overlap(raised_left, partner),
            lambda centre: _s_overlap(exponent, centre, partner_exponent, second),
            first,
            exponent,
        ),
        "kinetic": (
            _kinetic(bare_left, raised_right),
            lambda centre: _s_kinetic(exponent, first, partner_exponent, centre),
            second,
            partner_exponent,
        ),
        "attraction": (
            _nuclear(raised_left, partner, ((1, third),)),
            lambda centre: _s_attraction(
                exponent, centre, partner_exponent, second, third
            ),
            first,
            exponent,
        ),
        "repulsion, bra side": (
            _electron_repulsion(raised_left, partner, third_orbital, fourth_orbital),
            lambda centre: _s_repulsion(
                exponent,
                centre,
                partner_exponent,
                second,
                exponent_third,
                third,
                exponent_fourth,
                fourth,
            ),
            first,
            exponent,
        ),
        "repulsion, ket side": (
            _electron_repulsion(bare_left, partner, raised_ket, fourth_orbital),
            lambda centre: _s_repulsion(
                exponent,
                first,
                partner_exponent,
                second,
                exponent_third,
                centre,
                exponent_fourth,
                fourth,
            ),
            third,
            exponent_third,
        ),
    }
    worst = 0.0
    for label, (measured, closed_form, centre, raised_exponent) in cases.items():
        reference = _raised_state(closed_form, angular, raised_exponent)(centre)
        assert reference != 0.0
        relative = abs(measured - reference) / abs(reference)
        worst = max(worst, relative)
        assert relative < 1e-8, f"{label} at {angular}: {measured!r} vs {reference!r}"
    assert worst < 1e-9


def test_the_raising_route_only_degrades_with_the_angular_momentum_it_stacks() -> None:
    """A control on the derivative route: it is exact at ``s`` and worsens with ``l``.

    Without this, a wrong reference that happened to be off by a constant would
    pass the test above. The ``s``-only case has no derivative in it at all, so
    it must reproduce the closed form to machine precision, and a function of
    total degree four must be visibly worse than one of degree two. If the
    reference were the same assembly as the engine, the first comparison would
    still pass but the second could not be worse.
    """

    exponent = 3.425
    second_exponent = 0.623
    first = (0.11, -0.27, 0.43)
    second = (1.31, 0.62, -0.18)

    def closed_form(centre):
        return _s_overlap(exponent, centre, second_exponent, second)

    exact = _s_overlap(exponent, first, second_exponent, second)
    bare = _overlap(
        _primitive(exponent, first, (0, 0, 0)),
        _primitive(second_exponent, second, (0, 0, 0)),
    )
    assert abs(bare - exact) / abs(exact) < 1e-15
    reference = _raised_state(closed_form, (2, 2, 0), exponent)(first)
    measured = _overlap(
        _primitive(exponent, first, (2, 2, 0)),
        _primitive(second_exponent, second, (0, 0, 0)),
    )
    assert abs(measured - reference) / abs(reference) > 1e-9


def test_the_primitive_normalisation_is_the_closed_form_for_every_degree() -> None:
    """The primitive constant is pinned at degrees the basis table never reaches.

    ``_primitive_norm`` is

        (2a/pi)^(3/4) (4a)^(l/2) / sqrt((2lx-1)!! (2ly-1)!! (2lz-1)!!),

    and the shipped STO-3G table only carries ``s`` and ``p`` shells, where
    ``l <= 1`` and every double factorial in the denominator is one. Both the
    ``(4a)^(l/2)`` angular factor at ``l >= 2`` and the denominator are
    therefore invisible to every integral this module computes for a real
    molecule, and no test that goes through ``molecular_integrals`` can see
    them. This one calls the constant directly, against an independent
    evaluation of the closed form, and it also records the reachability as a
    measurement rather than an assumption: the table's shells are read from the
    table itself.
    """

    # (2l-1)!! for the l this test reaches: 0 -> 1, 1 -> 1, 2 -> 3.
    double_factorials = {0: 1, 1: 1, 2: 3}
    classes = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (0, 0, 2),
    ]
    for exponent in (0.1688554, 1.0, 3.42525091):
        for angular in classes:
            degree = sum(angular)
            denominator = 1
            for axis in angular:
                denominator *= double_factorials[axis]
            expected = (
                math.pow(2.0 * exponent / math.pi, 0.75)
                * math.pow(4.0 * exponent, degree / 2.0)
                / math.sqrt(denominator)
            )
            assert _primitive_norm(exponent, angular) == pytest.approx(
                expected, rel=1e-15
            )
    shells = {shell for contraction in _STO3G.values() for shell, _ in contraction}
    assert shells == {"s", "p"}, (
        "the double-factorial denominator is exercised only through this test "
        f"while the table carries {sorted(shells)}"
    )


# --- section 6: translation invariance ---


@pytest.mark.parametrize("factory", [_h2, _lih, _h2o, _nh3, _ch4, _hf, _n2])
def test_a_rigid_translation_leaves_every_integral_alone(factory) -> None:
    """Shifting the whole molecule must not move a single matrix element.

    Translation invariance is exact for any angular momentum because every
    basis function is defined relative to its own centre. The bound is the one
    measured here; the largest deviation comes from the ``1 / |r - C|``
    attraction, where the cancellation in the Boys argument is worst.
    """

    shift = (0.37, -1.21, 0.55)
    original = molecular_integrals(factory())
    moved = molecular_integrals(
        [
            (symbol, tuple(centre[axis] + shift[axis] for axis in range(3)))
            for symbol, centre in factory()
        ]
    )
    assert float((original.overlap - moved.overlap).abs().max()) < 1e-13
    assert (
        float((original.core_hamiltonian - moved.core_hamiltonian).abs().max()) < 1e-12
    )
    assert float((original.two_electron - moved.two_electron).abs().max()) < 1e-12
    assert original.nuclear_repulsion == pytest.approx(
        moved.nuclear_repulsion, rel=1e-13
    )


# --- section 7: rotation invariance, and where it stops being elementwise ---


def _proper_rotation(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if float(torch.linalg.det(rotation)) < 0.0:
        rotation[:, 0] = -rotation[:, 0]
    return rotation


def _rotated(atoms, rotation: torch.Tensor, shift: torch.Tensor):
    return [
        (
            symbol,
            tuple(
                (rotation @ torch.tensor(centre, dtype=torch.float64) + shift).tolist()
            ),
        )
        for symbol, centre in atoms
    ]


def test_a_rigid_rotation_leaves_an_s_only_molecule_alone() -> None:
    """With only ``s`` functions there is no axis to rotate away from.

    H2 is the one molecule in the standard set whose STO-3G basis is entirely
    ``s`` functions, so this is the case where elementwise rotation invariance
    is a true statement rather than an accident of the indices.
    """

    rotation = _proper_rotation(3)
    zeros = torch.zeros(3, dtype=torch.float64)
    original = molecular_integrals(_h2())
    turned = molecular_integrals(_rotated(_h2(), rotation, zeros))
    assert float((original.overlap - turned.overlap).abs().max()) == 0.0
    assert (
        float((original.core_hamiltonian - turned.core_hamiltonian).abs().max()) == 0.0
    )
    assert float((original.two_electron - turned.two_electron).abs().max()) < 1e-12


@pytest.mark.parametrize("factory", [_h2, _lih, _h2o, _nh3, _ch4, _hf, _n2])
def test_a_rigid_rotation_leaves_the_shell_block_scalars_alone(factory) -> None:
    """The right rotation statement when ``p`` functions are present.

    A rotation acts inside each shell of three ``p`` functions, so it conjugates
    the matrices by a block-orthogonal transformation. Elements move — the
    control below measures that — but the eigenvalue lists and the Frobenius
    norm of the two-electron tensor are invariant, and those are what is
    asserted. The bound is the one measured here.
    """

    rotation = _proper_rotation(11)
    shift = torch.tensor([0.37, -1.21, 0.55], dtype=torch.float64)
    original = molecular_integrals(factory())
    turned = molecular_integrals(_rotated(factory(), rotation, shift))
    for left, right in (
        (original.overlap, turned.overlap),
        (original.core_hamiltonian, turned.core_hamiltonian),
    ):
        symmetric_left = 0.5 * (left + left.T)
        symmetric_right = 0.5 * (right + right.T)
        difference = float(
            (
                torch.linalg.eigvalsh(symmetric_left)
                - torch.linalg.eigvalsh(symmetric_right)
            )
            .abs()
            .max()
        )
        assert difference < 1e-11
    assert (
        abs(
            float((original.two_electron**2).sum())
            - float((turned.two_electron**2).sum())
        )
        < 1e-11
    )


def test_a_rigid_rotation_moves_the_elements_of_a_p_bearing_molecule() -> None:
    """The control for the section above: the tensor is *not* elementwise invariant.

    Stating rotation invariance without this control would be false for any
    molecule with a ``p`` function, because the basis functions stay aligned to
    the laboratory axes. The measured difference is large, so it separates the
    two claims.
    """

    rotation = _proper_rotation(3)
    zeros = torch.zeros(3, dtype=torch.float64)
    original = molecular_integrals(_h2o())
    turned = molecular_integrals(_rotated(_h2o(), rotation, zeros))
    assert float((original.overlap - turned.overlap).abs().max()) > 0.1
    assert (
        float((original.core_hamiltonian - turned.core_hamiltonian).abs().max()) > 0.1
    )
    # The nuclear repulsion is a scalar and does not move.
    assert original.nuclear_repulsion == pytest.approx(
        turned.nuclear_repulsion, rel=1e-13
    )


# --- section 8: structural invariants of the returned integrals ---


@pytest.mark.parametrize("factory", [_h2, _lih, _h2o, _nh3, _ch4, _hf, _n2])
def test_the_returned_matrices_have_the_structure_they_document(factory) -> None:
    integrals = molecular_integrals(factory())
    n = integrals.n_orbitals
    assert integrals.overlap.shape == (n, n)
    assert integrals.kinetic.shape == (n, n)
    assert integrals.nuclear_attraction.shape == (n, n)
    assert integrals.core_hamiltonian.shape == (n, n)
    assert integrals.two_electron.shape == (n, n, n, n)
    assert n == len(integrals.atomic_orbitals)
    for matrix in (integrals.overlap, integrals.kinetic, integrals.nuclear_attraction):
        assert float((matrix - matrix.T).abs().max()) < 1e-14
    assert torch.equal(
        integrals.core_hamiltonian, integrals.kinetic + integrals.nuclear_attraction
    )


@pytest.mark.parametrize("factory", [_h2, _lih, _h2o, _nh3, _ch4, _hf, _n2])
def test_the_overlap_matrix_is_positive_definite(factory) -> None:
    """A Gram matrix of linearly independent functions has positive eigenvalues.

    This is an independent statement about the overlap: it would fail if two
    contracted functions were accidentally identical, or if a normalisation
    were dropped, neither of which any single matrix element would reveal.
    """

    integrals = molecular_integrals(factory())
    eigenvalues = torch.linalg.eigvalsh(integrals.overlap)
    assert float(eigenvalues.min()) > 0.0
    # Every contracted function arrives normalised, so every diagonal element of
    # the overlap is one and its trace is the number of functions. The published
    # STO-3G coefficients are rounded, which is why this is not exact. An upper
    # bound on the eigenvalues would be false: a non-orthogonal basis of
    # normalised functions routinely has an eigenvalue above one.
    diagonal = integrals.overlap.diagonal()
    assert float((diagonal - 1.0).abs().max()) < 1e-7
    assert abs(float(diagonal.sum()) - integrals.n_orbitals) < 1e-6


def test_the_integrals_are_returned_in_the_requested_precision() -> None:
    single = molecular_integrals(_h2(), dtype=torch.float32)
    assert single.overlap.dtype == torch.float32
    assert single.two_electron.dtype == torch.float32
    assert single.core_hamiltonian.dtype == torch.float32
    double = molecular_integrals(_h2())
    assert double.overlap.dtype == torch.float64
    # The two must still agree to single precision, which is what makes the
    # dtype argument a storage choice rather than a second algorithm.
    assert float((single.overlap - double.overlap.float()).abs().max()) < 1e-6


def test_a_geometry_object_is_accepted_as_well_as_a_pair_list() -> None:
    pairs = molecular_integrals(_h2())
    geometry = MolecularGeometry.from_angstrom(_h2())
    assert isinstance(pairs, MolecularIntegrals)
    assert pairs.geometry == geometry
    assert torch.equal(molecular_integrals(geometry).overlap, pairs.overlap)
    assert pairs.basis == "sto-3g"


# --- section 9: geometry construction and derived quantities ---


def test_a_geometry_converts_between_angstroms_and_bohr() -> None:
    geometry = MolecularGeometry.from_angstrom(_h2())
    assert geometry.symbols == ("H", "H")
    assert geometry.coordinates[1] == (0.0, 0.0, 0.7414)
    assert geometry.in_bohr()[1][2] == pytest.approx(0.7414 / BOHR_RADIUS, rel=1e-15)
    assert geometry.charges() == (
        (1, (0.0, 0.0, 0.0)),
        (1, (0.0, 0.0, 0.7414 / BOHR_RADIUS)),
    )


def test_a_geometry_reports_its_nuclear_charge_and_repulsion() -> None:
    water = MolecularGeometry.from_angstrom(_h2o())
    assert water.nuclear_charge == 10
    assert water.n_electrons == 10
    assert water.nuclear_repulsion() == pytest.approx(8.896614129637591, rel=1e-12)
    # The parameter-free identity: the repulsion is the sum over pairs of Z_i Z_j / r_ij.
    centres = water.in_bohr()
    charges = [ATOMIC_NUMBERS[symbol] for symbol in water.symbols]
    total = 0.0
    for i, j in itertools.combinations(range(3), 2):
        distance = math.dist(centres[i], centres[j])
        total += charges[i] * charges[j] / distance
    assert water.nuclear_repulsion() == pytest.approx(total, rel=1e-15)


def test_the_raw_constructor_and_the_angstrom_factory_agree() -> None:
    """``MolecularGeometry`` stores angstroms, and ``in_bohr`` is the only conversion.

    The direction matters and is observable: a geometry built from the *bohr*
    separation of H2 has a nuclear repulsion smaller by exactly ``BOHR_RADIUS``,
    because the same numbers read as angstroms describe a molecule further
    apart. The test pins that direction rather than assuming it.
    """

    raw = MolecularGeometry(("H", "H"), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.7414)))
    converted = MolecularGeometry.from_angstrom(_h2())
    assert raw == converted
    bohr_separation = MolecularGeometry(
        ("H", "H"), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.7414 / BOHR_RADIUS))
    )
    assert bohr_separation.nuclear_repulsion() != pytest.approx(
        converted.nuclear_repulsion(), rel=1e-3
    )
    assert bohr_separation.nuclear_repulsion() == pytest.approx(
        converted.nuclear_repulsion() * BOHR_RADIUS, rel=1e-15
    )


# --- section 10: every refusal condition refuses ---


def test_an_empty_geometry_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least one atom"):
        MolecularGeometry((), ())


def test_a_symbol_without_its_coordinate_is_refused() -> None:
    with pytest.raises(ValidationError, match="one coordinate per element symbol"):
        MolecularGeometry(("H", "H"), ((0.0, 0.0, 0.0),))


@pytest.mark.parametrize("coordinates", [((0.0, 0.0),), ((0.0, 0.0, 0.0, 0.0),)])
def test_a_coordinate_that_is_not_three_dimensional_is_refused(coordinates) -> None:
    with pytest.raises(ValidationError, match="three-dimensional"):
        MolecularGeometry(("H",), coordinates)


def test_an_element_outside_the_table_is_refused_with_the_range_named() -> None:
    with pytest.raises(CapabilityError, match="outside the STO-3G table, which covers"):
        MolecularGeometry(("Ar",), ((0.0, 0.0, 0.0),))
    with pytest.raises(CapabilityError, match="refused rather than given a wrong one"):
        molecular_integrals([("Xx", (0.0, 0.0, 0.0))])


@pytest.mark.parametrize("atom", [("H",), ("H", (0.0, 0.0, 0.0), 7)])
def test_a_malformed_atom_pair_is_refused(atom) -> None:
    with pytest.raises(ValidationError, match=r"iterable of \(symbol,"):
        MolecularGeometry.from_angstrom([atom])


def test_a_coordinate_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(ValueError, match="could not convert string to float"):
        MolecularGeometry.from_angstrom([("H", ("a", 0.0, 0.0))])


def test_a_basis_other_than_sto3g_is_refused_with_the_available_set_named() -> None:
    with pytest.raises(
        CapabilityError, match=r"'6-31g' is not available; this module expands sto-3g"
    ):
        molecular_integrals(_h2(), basis="6-31g")


def test_the_refusal_names_the_whole_available_basis_list() -> None:
    """The message must be generated from ``BASIS_SETS``, not restated by hand."""

    with pytest.raises(CapabilityError) as raised:
        molecular_integrals(_h2(), basis="cc-pvdz")
    for basis in BASIS_SETS:
        assert basis in str(raised.value)


# --- section 11: the module's own examples ---


def test_the_module_examples_still_run() -> None:
    """No repository-wide doctest gate exists, so the examples are run here."""

    import doctest

    from flagquantum.algorithms import chemistry_integrals

    results = doctest.testmod(chemistry_integrals, verbose=False)
    assert results.failed == 0
    assert results.attempted >= 2
