"""Spinful chain and ring Hamiltonians in the native fermionic algebra."""

from __future__ import annotations

from numbers import Real

from ...core.algebra.pauli import count, scalar
from ...observables.fermion import (
    FermionOperator,
    FermionTerm,
    annihilate,
    create,
    number,
)


def _real_coefficient(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite static real number, not a tensor")
    return scalar(value).real


def fermi_hubbard_chain(
    n_sites: int, *, t: float = 1.0, u: float = 0.0, mu: float = 0.0
) -> FermionOperator:
    """Build -t hopping + u double occupancy - mu population on an open chain.

    Site i owns modes 2*i (up) and 2*i+1 (down). Interaction is unshifted;
    each neighboring bond is counted once. No dense arrays are allocated.

    Args:
        n_sites: Positive site count. Encodings require explicit width 2*n_sites.
        t: Real hopping amplitude, with the minus sign applied by this function.
        u: Real onsite interaction, without a half-filling energy shift.
        mu: Real chemical potential, subtracted from each occupied mode.

    Raises:
        TypeError: An input has the wrong type, including bool, complex or tensor.
        ValueError: Site count is nonpositive or a coefficient is nonfinite.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.fermion.encoding import encode_fermion
        >>> from flagquantum.algorithms.fermion.observables import to_observable
        >>> h = fermi_hubbard_chain(1, u=4.0, mu=1.0)
        >>> encoded = encode_fermion(h, n_modes=2, encoding="jordan_wigner")
        >>> circuit = fq.Circuit(2).x(0).x(1)
        >>> result = fq.run(circuit, outputs=[fq.expectation(to_observable(encoded.operator))])
        >>> float(result.expectation().reshape(-1)[0])
        2.0
    """
    n_sites = count(n_sites, "n_sites", minimum=1)
    t = _real_coefficient(t, "t")
    u = _real_coefficient(u, "u")
    mu = _real_coefficient(mu, "mu")
    terms: list[FermionTerm] = []
    for site in range(n_sites):
        up, down = 2 * site, 2 * site + 1
        terms.extend((u * number(up) * number(down)).terms)
        for mode in (up, down):
            terms.extend((-mu * number(mode)).terms)
            if site + 1 < n_sites:
                hop = create(mode) * annihilate(mode + 2)
                terms.extend((-t * (hop + hop.dagger())).terms)
    return FermionOperator(tuple(terms))


def fermi_hubbard_ring(
    n_sites: int, *, t: float = 1.0, u: float = 0.0, mu: float = 0.0
) -> FermionOperator:
    """Build a periodic spinful Hubbard Hamiltonian with each bond counted once.

    Reuses the open-chain terms and adds hopping between sites n_sites-1 and 0.
    Modes 2*i and 2*i+1 carry up and down spin. The interaction is unshifted.
    Assembly is symbolic and allocates no dense arrays.

    Args:
        n_sites: At least three sites, defining a cycle without self/duplicate bonds.
        t: Finite static real hopping amplitude; hopping has coefficient -t.
        u: Finite static real onsite interaction.
        mu: Finite static real chemical potential, subtracted from population.

    Raises:
        TypeError: An input is bool, complex, tensor or otherwise the wrong type.
        ValueError: Fewer than three sites or a nonfinite coefficient is supplied.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.fermion.encoding import encode_fermion
        >>> from flagquantum.algorithms.fermion.observables import to_observable
        >>> operator = fermi_hubbard_ring(3, u=4.0, mu=1.0)
        >>> encoded = encode_fermion(operator, n_modes=6, encoding="jordan_wigner")
        >>> circuit = fq.Circuit(6).x(0).x(1)
        >>> result = fq.run(circuit, outputs=[fq.expectation(to_observable(encoded.operator))])
        >>> float(result.expectation().reshape(-1)[0])
        2.0
    """
    n_sites = count(n_sites, "n_sites", minimum=3)
    operator = fermi_hubbard_chain(n_sites, t=t, u=u, mu=mu)
    for spin in (0, 1):
        hop = create(2 * (n_sites - 1) + spin) * annihilate(spin)
        operator = operator - t * (hop + hop.dagger())
    return operator
