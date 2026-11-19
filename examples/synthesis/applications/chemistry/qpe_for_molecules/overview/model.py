"""Regenerate the LiH sector and encode it with the source product order."""

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from examples.synthesis.applications.chemistry.qpe_for_molecules.physics import (
    TERMS,
    Term,
)
from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.observables.fermion import jordan_wigner

GEOMETRIES = {
    "H2": (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))),
    "O2": (("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.16))),
    "LiH": (("H", (0.0, 0.0, 0.0)), ("Li", (0.0, 0.0, 1.596))),
    "H2O": (
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.586, 0.757)),
        ("H", (0.0, -0.586, 0.757)),
    ),
    "BeH2": (
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.334)),
        ("H", (0.0, 0.0, -1.334)),
    ),
}
ACTIVE_MODES = 10
WIDTH = 6
HF_OCCUPATION = 0b11
# Even/odd mode spin parities and the two transverse-orbital parities.
SYMMETRIES = (0b0101010101, 0b1010101010, 0b0000110000, 0b0011000000)
RETAINED = (0, 1, 2, 3, 4, 6)
# Columns of the binary conversion into retained-occupation coordinates.
SOURCE_COLUMNS = (5, 19, 35, 1, 10, 2)
# A Z on bit 3 and CZs on pairs (0, 5), (3, 5), (4, 5).
SIGN_MASKS = (8, 33, 40, 48)
CUTOFF = 0.03
NUMERICAL_ZERO = 1e-10


@dataclass(frozen=True)
class Model:
    terms: tuple[Term, ...]
    coefficient_magnitudes: tuple[float, ...]
    normalization: float
    offset: float
    rhf: float
    full_ci: float
    active_ci: float
    reference_delta: float


def sector_words() -> tuple[int, ...]:
    """Map each source basis index to its ten-mode occupation word."""
    retained_words = {}
    for occupation in range(1 << ACTIVE_MODES):
        flips = occupation ^ HF_OCCUPATION
        if any((flips & mask).bit_count() % 2 for mask in SYMMETRIES):
            continue
        word = sum(((flips >> mode) & 1) << bit for bit, mode in enumerate(RETAINED))
        retained_words[word] = occupation
    if len(retained_words) != 1 << WIDTH:
        raise ValueError("The retained bits do not label the entire symmetry sector")
    columns = [0]
    for column in SOURCE_COLUMNS:
        columns.extend([word ^ column for word in columns])
    return tuple(retained_words[word] for word in columns)


def masks(support: tuple[tuple[int, str], ...]) -> tuple[int, int]:
    return (
        sum(1 << bit for bit, axis in support if axis.upper() in "XY"),
        sum(1 << bit for bit, axis in support if axis.upper() in "ZY"),
    )


def pauli_support(x: int, z: int) -> tuple[tuple[int, str], ...]:
    return tuple(
        (bit, "IXZY"[((x >> bit) & 1) + 2 * ((z >> bit) & 1)])
        for bit in range(WIDTH)
        if ((x | z) >> bit) & 1
    )


def gauge(word: int) -> int:
    return sum((word & mask) == mask for mask in SIGN_MASKS) % 2


def taper(terms: tuple[Term, ...]) -> dict[tuple[tuple[int, str], ...], float]:
    words = sector_words()
    encoded = {word: index for index, word in enumerate(words)}
    result: dict[tuple[tuple[int, str], ...], float] = defaultdict(float)
    for support, coefficient in terms:
        x, z = masks(support)
        if any((x & mask).bit_count() % 2 for mask in SYMMETRIES):
            raise ValueError("The electronic Hamiltonian must preserve each parity")
        target_flip = encoded[words[0] ^ x]
        sign = ((words[0] & z).bit_count() + gauge(target_flip)) % 2
        target_z = sum(
            (
                (
                    ((words[0] ^ words[1 << bit]) & z).bit_count()
                    + gauge(1 << bit)
                    + gauge((1 << bit) ^ target_flip)
                    + gauge(target_flip)
                )
                % 2
            )
            << bit
            for bit in range(WIDTH)
        )
        relative_phase = (-1) ** sign * 1j ** (
            (x & z).bit_count() - (target_flip & target_z).bit_count()
        )
        if abs(relative_phase.imag) > NUMERICAL_ZERO:
            raise ValueError("The LiH example assumes a real electronic Hamiltonian")
        result[pauli_support(target_flip, target_z)] += (
            coefficient * relative_phase.real
        )
    return {
        support: value
        for support, value in result.items()
        if abs(value) > NUMERICAL_ZERO
    }


def build_model() -> Model:
    geometry = GEOMETRIES["LiH"]
    full = create_molecular_hamiltonian(geometry)
    active = create_molecular_hamiltonian(
        geometry, n_active_electrons=2, n_active_orbitals=5
    )
    if full.fci_energy is None or active.fci_energy is None:
        raise ValueError("Exact full and active CI references are required")
    problem = from_native_molecule(active)
    observable = jordan_wigner(fermion_hamiltonian(problem), n_modes=ACTIVE_MODES)
    fresh = taper(
        tuple(
            (term.factors, float(term.coefficient))
            for term in observable.terms
            if abs(float(term.coefficient)) > NUMERICAL_ZERO
        )
    )
    fresh[()] -= problem.energy_offset
    trimmed = {
        support: value for support, value in fresh.items() if abs(value) > CUTOFF
    }
    order = tuple(support for support, _ in TERMS)
    if set(order) != trimmed.keys():
        raise ValueError("The freshly tapered LiH support must match the source order")
    normalization = 2 * math.fsum(abs(value) for value in trimmed.values())
    terms = tuple((support, trimmed[support] / normalization) for support in order)
    delta = max(
        abs(value - saved) for (_, value), (_, saved) in zip(terms, TERMS, strict=True)
    )
    if delta >= 1e-7:
        raise ValueError(
            "The regenerated normalized coefficients must match the LiH source"
        )
    return Model(
        terms,
        tuple(sorted(map(abs, fresh.values()), reverse=True)),
        normalization,
        problem.energy_offset,
        full.hf_energy,
        full.fci_energy,
        active.fci_energy,
        delta,
    )


def gamma_two(terms: tuple[Term, ...]) -> float:
    """Sum the absolute coefficients of all ordered double commutators."""
    factors = tuple((*masks(support), abs(value)) for support, value in terms)
    return math.fsum(
        4 * a * b * c
        for (x, z, a), (y, t, b), (v, w, c) in product(factors, repeat=3)
        if ((y & w).bit_count() + (t & v).bit_count()) % 2
        and ((x & (t ^ w)).bit_count() + (z & (y ^ v)).bit_count()) % 2
    )
