"""Quadratic unconstrained binary optimization expressed as an Ising Hamiltonian.

The mapping substitutes ``x_i = (1 + s_i) / 2`` to turn a binary objective into a spin
objective, which the existing variational workflows in :mod:`flagquantum.algorithms.core`
already consume.

The mapping is a polynomial classical transformation. It carries no advantage of its own:
any advantage a caller observes belongs to the solver that consumes the Hamiltonian, not to
this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch

from .core import Hamiltonian, pauli_term

__all__ = [
    "QuboProblem",
    "ising_to_qubo",
    "max_cut_qubo",
    "qubo_energy",
    "qubo_to_ising",
]

_IDENTITY_WIRE = 0


@dataclass(frozen=True)
class QuboProblem:
    """A binary objective ``offset + sum_i c_i x_i + sum_{i<j} q_ij x_i x_j``.

    Args:
        n_variables: The number of binary variables.
        linear: Coefficient of each single variable, keyed by its index. Variables with a
            zero coefficient may be omitted.
        quadratic: Coefficient of each pair, keyed by an ordered pair with the lower index
            first. Pairs with a zero coefficient may be omitted.
        offset: A constant added to every objective value. It is zero for a problem built
            from a QUBO objective and carries the constant recovered from a Hamiltonian.
    """

    n_variables: int
    linear: Mapping[int, float]
    quadratic: Mapping[tuple[int, int], float]
    offset: float = 0.0


def _constant_wire(n_variables: int) -> int:
    """Return the wire that carries the constant term.

    A constant term has no operator, so it contributes nothing to the Hamiltonian's wire
    count on any wire. It is placed on the register's last wire so that the term also
    records the register width: for a problem with no coefficients the constant term is
    all a Hamiltonian carries, and the declared wire is then the only record of how many
    variables the problem had.

    Args:
        n_variables: The number of binary variables.

    Returns:
        The wire index for the constant term.
    """

    return max(n_variables - 1, _IDENTITY_WIRE)


def qubo_energy(problem: QuboProblem, assignment: Sequence[int]) -> float:
    """Return the objective value of ``assignment``.

    Args:
        problem: The problem to evaluate.
        assignment: One binary value per variable, in index order.

    Returns:
        The objective value, including the problem's offset.

    Raises:
        ValueError: If the assignment length does not match ``n_variables``, or a value is
            not 0 or 1.
    """

    if len(assignment) != problem.n_variables:
        raise ValueError(
            f"assignment has {len(assignment)} values, expected {problem.n_variables}"
        )
    for value in assignment:
        if value not in (0, 1):
            raise ValueError(f"assignment values must be 0 or 1, got {value}")

    total = problem.offset
    total += sum(
        coefficient * assignment[index] for index, coefficient in problem.linear.items()
    )
    total += sum(
        coefficient * assignment[first] * assignment[second]
        for (first, second), coefficient in problem.quadratic.items()
    )
    return float(total)


def qubo_to_ising(problem: QuboProblem) -> Hamiltonian:
    """Return the Ising Hamiltonian whose objective matches the QUBO objective.

    Substituting ``x_i = (1 + s_i) / 2`` makes each variable a Pauli Z on its own wire.
    Each declared linear coefficient contributes ``c_i / 2`` to its single-wire term, plus
    a quarter of every pair weight that touches the variable; each declared pair
    contributes ``q_ij / 4`` to a two-wire term. A constant is always emitted as one
    identity term, so the result is a valid Hamiltonian for every problem, including one
    that declares no coefficients at all.

    Args:
        problem: The problem to convert.

    Returns:
        A Hamiltonian whose terms are single-wire Z products, two-wire ZZ products, and
        one identity term carrying the constant offset.
    """

    pair_totals: dict[int, float] = {}
    for (first, second), coefficient in problem.quadratic.items():
        pair_totals[first] = pair_totals.get(first, 0.0) + coefficient
        pair_totals[second] = pair_totals.get(second, 0.0) + coefficient

    constant = (
        sum(problem.linear.values()) / 2.0 + sum(problem.quadratic.values()) / 4.0
    )
    weights = {
        index: coefficient / 2.0 + pair_totals.get(index, 0.0) / 4.0
        for index, coefficient in problem.linear.items()
    }

    terms = [
        pauli_term(weight, "Z", (index,)) for index, weight in weights.items() if weight
    ]
    terms.extend(
        pauli_term(coefficient / 4.0, "ZZ", pair)
        for pair, coefficient in problem.quadratic.items()
    )
    terms.append(pauli_term(constant, "I", (_constant_wire(problem.n_variables),)))
    return Hamiltonian(terms)


def ising_to_qubo(hamiltonian: Hamiltonian) -> QuboProblem:
    """Return the QUBO problem whose objective matches a Hamiltonian in the Z basis.

    Args:
        hamiltonian: The Hamiltonian to convert. It may carry single-wire ``Z`` terms,
            two-wire ``ZZ`` terms, and one or more identity terms.

    Returns:
        A problem whose objective reproduces the Hamiltonian's value on every assignment.
        The constant that the Hamiltonian carries is returned as ``offset``, so a caller
        comparing the two forms sees identical values.

    Raises:
        ValueError: If a term's Pauli string is not one of ``I``, ``Z``, or ``ZZ``. This
            workflow is defined for Z-basis objectives only.
    """

    constant = 0.0
    linear_weights: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    declared_wires = 0
    for term in hamiltonian.terms:
        pauli = term.pauli
        if term.wires:
            declared_wires = max(declared_wires, 1 + max(term.wires))
        raw = term.coefficient
        if isinstance(raw, complex) or (
            isinstance(raw, torch.Tensor) and raw.is_complex()
        ):
            raise ValueError(
                "Ising to QUBO conversion requires real coefficients, got a complex one."
            )
        coefficient = float(raw)
        if pauli == "I":
            constant += coefficient
        elif pauli == "Z":
            wire = term.wires[0]
            linear_weights[wire] = linear_weights.get(wire, 0.0) + coefficient
        elif pauli == "ZZ":
            first, second = sorted(term.wires)
            quadratic[first, second] = quadratic.get((first, second), 0.0) + coefficient
        else:
            raise ValueError(
                f"Ising to QUBO conversion supports I, Z, and ZZ terms only, got {pauli!r}."
            )

    pair_totals: dict[int, float] = {}
    for (first, second), coefficient in quadratic.items():
        pair_totals[first] = pair_totals.get(first, 0.0) + coefficient
        pair_totals[second] = pair_totals.get(second, 0.0) + coefficient

    linear = {
        index: 2.0 * weight - 2.0 * pair_totals.get(index, 0.0)
        for index, weight in linear_weights.items()
    }
    recovered = {pair: 4.0 * coefficient for pair, coefficient in quadratic.items()}
    offset = constant - sum(linear_weights.values()) + sum(quadratic.values())
    n_variables = max(hamiltonian.n_wires, declared_wires)
    return QuboProblem(
        n_variables=n_variables,
        linear=linear,
        quadratic=recovered,
        offset=offset,
    )


def max_cut_qubo(edges: Iterable[tuple[int, int]], *, n_nodes: int) -> QuboProblem:
    """Return the MaxCut objective for a graph, whose optimum is minus the cut size.

    Each edge ``(i, j)`` contributes ``2 x_i x_j - x_i - x_j``, so minimizing the objective
    over all assignments maximizes the number of edges whose endpoints differ.

    Args:
        edges: The graph's undirected edges as node-index pairs.
        n_nodes: The number of graph nodes, and the number of binary variables.

    Returns:
        The objective, whose minimum is the negated maximum cut size.
    """

    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    for first, second in edges:
        pair = (first, second) if first < second else (second, first)
        linear[first] = linear.get(first, 0.0) - 1.0
        linear[second] = linear.get(second, 0.0) - 1.0
        quadratic[pair] = quadratic.get(pair, 0.0) + 2.0
    return QuboProblem(n_variables=n_nodes, linear=linear, quadratic=quadratic)
