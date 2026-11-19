"""Spinful Hubbard and spinless t-V models in native fermionic algebra."""

from __future__ import annotations

from collections.abc import Sequence
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


def fermi_hubbard_graph(
    n_sites: int,
    edges: Sequence[tuple[int, int]],
    *,
    t: float = 1.0,
    u: float = 0.0,
    mu: float = 0.0,
) -> FermionOperator:
    """Build a spinful Hubbard model on an explicit undirected simple graph.

    Reuses chain onsite terms. Each supplied edge contributes minus-t hopping
    for both spins, with interleaved modes 2*i and 2*i+1. No graph or dense
    numerical representation is constructed. Empty graphs retain onsite terms.

    Args:
        n_sites: Positive site count; encode with explicit width 2*n_sites.
        edges: Finite sequence of endpoint pairs, each undirected bond once.
        t: Finite static real hopping amplitude, with coefficient -t.
        u: Finite static real unshifted onsite interaction.
        mu: Finite static real chemical potential, subtracted from population.

    Raises:
        TypeError: An input has the wrong type or an endpoint is not an integer.
        ValueError: A size/coefficient is invalid, a pair is malformed, or an
            edge is self-connected, repeated or outside the site register.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.fermion.encoding import encode_fermion
        >>> from flagquantum.algorithms.fermion.observables import to_observable
        >>> h = fermi_hubbard_graph(3, ((0, 1), (0, 2)), u=4, mu=1)
        >>> encoded = encode_fermion(h, n_modes=6)
        >>> result = fq.run(fq.Circuit(6).x(0).x(1),
        ...                 outputs=[fq.expectation(to_observable(encoded.operator))])
        >>> float(result.expectation().reshape(()))
        2.0
    """
    n_sites = count(n_sites, "n_sites", minimum=1)
    t = _real_coefficient(t, "t")
    onsite = fermi_hubbard_chain(n_sites, t=0.0, u=u, mu=mu)
    if isinstance(edges, (str, bytes)) or not isinstance(edges, Sequence):
        raise TypeError("edges must be a finite sequence of endpoint pairs")
    terms = list(onsite.terms)
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        if isinstance(edge, (str, bytes)) or not isinstance(edge, Sequence):
            raise TypeError("each edge must be an endpoint pair")
        if len(edge) != 2:
            raise ValueError("each edge must contain exactly two endpoints")
        left, right = (count(endpoint, "edge endpoint") for endpoint in edge)
        if left >= n_sites or right >= n_sites:
            raise ValueError("edge endpoint is outside n_sites")
        if left == right:
            raise ValueError("self edges are unsupported")
        bond = (min(left, right), max(left, right))
        if bond in seen:
            raise ValueError("each undirected edge must occur exactly once")
        seen.add(bond)
        for spin in (0, 1):
            hop = create(2 * left + spin) * annihilate(2 * right + spin)
            terms.extend((-t * (hop + hop.dagger())).terms)
    return FermionOperator(tuple(terms))


def spinless_fermion_chain(
    n_sites: int, *, t: float = 1.0, v: float = 0.0, mu: float = 0.0
) -> FermionOperator:
    """Build an open spinless t-V chain with one fermionic mode per site.

    Each neighboring bond contributes -t*(a†_i*a_j + h.c.) + v*n_i*n_j.
    Chemical potential contributes -mu*n_i. Densities are unshifted.

    Args:
        n_sites: Positive site count; encode with explicit width n_sites.
        t: Finite static real hopping amplitude.
        v: Finite static real nearest-neighbor density interaction.
        mu: Finite static real chemical potential.

    Raises:
        TypeError: A size has no integer index, is bool, or a coefficient is not static real.
        ValueError: Site count is nonpositive or a coefficient is nonfinite.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.fermion.encoding import encode_fermion
        >>> from flagquantum.algorithms.fermion.observables import to_observable
        >>> encoded = encode_fermion(spinless_fermion_chain(2, v=3), n_modes=2)
        >>> result = fq.run(fq.Circuit(2).x(0).x(1),
        ...                 outputs=[fq.expectation(to_observable(encoded.operator))])
        >>> float(result.expectation().reshape(()))
        3.0
    """
    n_sites = count(n_sites, "n_sites", minimum=1)
    t = _real_coefficient(t, "t")
    v = _real_coefficient(v, "v")
    mu = _real_coefficient(mu, "mu")
    terms: list[FermionTerm] = []
    for site in range(n_sites):
        terms.extend((-mu * number(site)).terms)
        if site + 1 < n_sites:
            hop = create(site) * annihilate(site + 1)
            terms.extend((-t * (hop + hop.dagger())).terms)
            terms.extend((v * number(site) * number(site + 1)).terms)
    return FermionOperator(tuple(terms))


def spinless_fermion_ring(
    n_sites: int, *, t: float = 1.0, v: float = 0.0, mu: float = 0.0
) -> FermionOperator:
    """Build a spinless t-V cycle with each undirected bond counted once.

    Reuses the open chain and adds the closing hopping and density interaction.
    Densities are unshifted; one site is one mode. No dense arrays are allocated.

    Args:
        n_sites: At least three sites, avoiding self/duplicate cycle bonds.
        t: Finite static real hopping amplitude, with coefficient -t.
        v: Finite static real nearest-neighbor density interaction.
        mu: Finite static real chemical potential, subtracted from population.

    Raises:
        TypeError: A size has no integer index, is bool, or a coefficient is not static real.
        ValueError: Fewer than three sites or a nonfinite coefficient is supplied.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.fermion.encoding import encode_fermion
        >>> from flagquantum.algorithms.fermion.observables import to_observable
        >>> encoded = encode_fermion(spinless_fermion_ring(3, v=2), n_modes=3)
        >>> result = fq.run(fq.Circuit(3).x(0).x(1).x(2),
        ...                 outputs=[fq.expectation(to_observable(encoded.operator))])
        >>> float(result.expectation().reshape(()))
        6.0
    """
    n_sites = count(n_sites, "n_sites", minimum=3)
    operator = spinless_fermion_chain(n_sites, t=t, v=v, mu=mu)
    hop = create(n_sites - 1) * annihilate(0)
    return operator - t * (hop + hop.dagger()) + v * number(n_sites - 1) * number(0)
