"""Grover search over the states a classical predicate marks.

Amplitude amplification for an unstructured search, following Lov K. Grover, "A fast
quantum mechanical algorithm for database search", *Proc. 28th Annual ACM Symposium on
Theory of Computing (STOC '96)*, pp. 212-219, 1996, DOI 10.1145/237814.237866,
arXiv:quant-ph/9605043. The paper shows that a search over ``N`` items finds a marked one
in ``O(sqrt(N))`` oracle queries where a classical search needs ``N/2`` for even odds, and
that this is within a small constant of optimal.

**The premise is the oracle, and the oracle here is not free.** The ``sqrt(N)`` count is a
statement about *queries*, and the query model assumes an oracle that can be evaluated in
one step. This unit synthesizes its oracle from the predicate's truth table through
:func:`~flagquantum.algorithms.primitives.oracle.append_phase_oracle`, which enumerates all
``2**n`` inputs classically, so the oracle costs as much as the classical search it stands
in for and **no end-to-end advantage follows at this scale**. Nothing here reads a qRAM,
block-encodes a matrix, or amplitude-encodes a vector. What the unit demonstrates is the
circuit and the query count, not a speedup.

**Wire layout.** The evaluation register is wires ``0 .. n_wires - 1``, wire 0 most
significant, and the circuit carries exactly ``n_wires`` wires: the register and nothing
else. The initial uniform superposition is one Hadamard per wire, and each round is the
phase oracle on the register followed by the diffusion operator.

**The three-wire bound belongs to the register.** The diffusion operator's multi-controlled
Z is a multi-controlled X with ``n_wires - 1`` controls, and a multi-controlled X above two
controls is built as an ancilla ladder needing ``len(controls) - 2`` wires in ``|0>``. A
circuit pinned at exactly ``n_wires`` wires has no free wire for those, so
:func:`grover_circuit` refuses ``n_wires > 3`` instead of allocating one: a wire added to
this circuit would be part of the register the samples are read over, so the "ancilla"
would be sampled along with the answer. The append form,
:func:`~flagquantum.algorithms.primitives.oracle.append_phase_oracle`, takes a register and
the ancillas the caller lays out, which is how a wider search is built.

Sample keys are big-endian bit strings, one character per wire with wire 0 the most
significant, as :meth:`flagquantum.circuit.Circuit.counts` returns them.

This unit is demonstration scale. It makes no performance, capacity, or hardware claim, and
it does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..circuit import Circuit
from .primitives.oracle import (
    append_multi_controlled_x,
    append_phase_oracle,
    marked_states,
)
from .primitives.types import Predicate

__all__ = ["GroverResult", "grover_circuit", "optimal_iterations", "run_grover"]

# The register is the whole circuit here, so the diffusion operator's multi-controlled Z
# has no free wire to fold its ladder onto once it passes three wires.
_GROVER_WIRE_LIMIT = 3


def optimal_iterations(n_wires: int, n_marked: int) -> int:
    """Return the number of amplification rounds for an ``n_wires`` search.

    Each round rotates the state by the angle ``theta = asin(sqrt(n_marked / 2**n_wires))``
    toward the marked subspace, so the round count that leaves the marked states with the
    largest total amplitude is the number of quarter-turns of ``theta`` that fit in a half
    turn: ``floor(pi / (4 * theta))``. At ``n_marked == 0`` there is nothing to amplify and
    the count is zero.

    Args:
        n_wires: The width of the evaluation register, at least one.
        n_marked: How many of the register's ``2**n_wires`` states the predicate marks.

    Returns:
        The number of Grover rounds to apply.

    Raises:
        ValueError: If ``n_wires`` is less than one, if ``n_marked`` is negative, or if
            ``n_marked`` exceeds ``2**n_wires``; a register cannot mark more states than it
            has.
    """
    _validate_register_width(n_wires)
    if n_marked < 0:
        raise ValueError(
            f"a marked-state count cannot be negative, got n_marked={n_marked}; the "
            "predicate's truth table is a set of states"
        )
    if n_marked > 2**n_wires:
        raise ValueError(
            f"a register of {n_wires} wires holds 2**{n_wires} = {2**n_wires} states, so "
            f"n_marked={n_marked} of them cannot all be marked"
        )
    if n_marked == 0:
        return 0
    theta = math.asin(math.sqrt(n_marked / 2**n_wires))
    # The knife edge: at exactly half the register marked the quotient is 0.9999999999999999
    # in floating point, so a bare floor returns 0 where the mathematics gives 1. The
    # success probability is 1/2 either way at that resonance, but the count would then
    # disagree with this docstring.
    return math.floor(math.pi / (4 * theta) + 1e-12)


@dataclass(frozen=True)
class GroverResult:
    """The outcome of one sampled :func:`run_grover` run.

    Attributes:
        candidates: The marked states the sample actually landed on, most frequent first,
            with ties broken by ascending state value.
        counts: The full sample, keyed by the big-endian bit string of each observed state,
            one character per wire with wire 0 the most significant.
        iterations: The number of amplification rounds the sampled circuit was built with.
    """

    candidates: tuple[int, ...]
    counts: Mapping[str, int]
    iterations: int

    @property
    def success_probability(self) -> float:
        """The share of the sample that landed on a marked state.

        The count is summed over the marked states, which is ``candidates``: the states the
        oracle marks are exactly the ones the search amplifies, so a marked state the
        sample never reached contributes nothing. An empty predicate, or a sample that
        missed every marked state, reports ``0.0``.
        """
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        marked = set(self.candidates)
        return (
            sum(count for key, count in self.counts.items() if int(key, 2) in marked)
            / total
        )


def grover_circuit(
    predicate: Predicate, n_wires: int, *, iterations: int | None = None
) -> Circuit:
    """Build the Grover search circuit for ``predicate`` over ``n_wires`` wires.

    The circuit is the evaluation register alone: a Hadamard on every wire, then
    ``iterations`` rounds of the phase oracle and the diffusion operator. Measuring it in
    the computational basis gives the marked states with amplified probability.

    The oracle is the truth-table form from
    :func:`~flagquantum.algorithms.primitives.oracle.append_phase_oracle`, which enumerates
    the predicate's ``2**n_wires`` inputs classically, so building the circuit is
    exponential in the register width. The register is the whole circuit, which is why
    ``n_wires`` is capped at three: see the module docstring.

    Args:
        predicate: The property the search marks.
        n_wires: The width of the evaluation register, at most three.
        iterations: The number of amplification rounds; ``None`` applies the count from
            :func:`optimal_iterations` for the predicate's own marked count, and ``0``
            leaves the register in the uniform superposition.

    Returns:
        The search circuit, its wires numbered ``0`` to ``n_wires - 1``.

    Raises:
        ValueError: If ``n_wires`` is less than one, if it is more than three and the
            register therefore has no wire for the diffusion operator's ladder ancillas, or
            if ``iterations`` is negative.
    """
    _validate_search_width(n_wires)
    if iterations is not None and iterations < 0:
        raise ValueError(
            f"the round count of a Grover search cannot be negative, got "
            f"iterations={iterations}; pass None for the optimal count or 0 for the bare "
            "uniform superposition"
        )
    if iterations is None:
        rounds = optimal_iterations(n_wires, len(marked_states(predicate, n_wires)))
    else:
        rounds = iterations
    wires = list(range(n_wires))
    circuit = Circuit(n_wires)
    for wire in wires:
        circuit.gate("h", wire)
    for _ in range(rounds):
        append_phase_oracle(circuit, predicate, wires)
        _append_diffusion(circuit, wires)
    return circuit


def run_grover(
    predicate: Predicate, n_wires: int, *, shots: int = 1024, seed: int | None = None
) -> GroverResult:
    """Run the search for ``predicate`` and report the marked states the sample found.

    The circuit is built by :func:`grover_circuit` at the optimal round count for the
    predicate's marked count, sampled ``shots`` times, and the marked states are ranked by
    how often the sample landed on them. ``seed`` seeds the sampler's generator, so the
    same seed and the same arguments replay the same counts exactly.

    Args:
        predicate: The property the search marks.
        n_wires: The width of the evaluation register, at most three.
        shots: The number of samples to draw, at least one.
        seed: The sampler's seed, or ``None`` to draw from the ambient generator.

    Returns:
        The ranked candidates, the full counts, and the round count the circuit used.

    Raises:
        ValueError: If ``n_wires`` is less than one or more than three, or if ``shots`` is
            less than one.
    """
    _validate_search_width(n_wires)
    if shots < 1:
        raise ValueError(
            f"a Grover run needs at least one shot, got shots={shots}; with no samples "
            "there is nothing to rank"
        )
    marked = marked_states(predicate, n_wires)
    rounds = optimal_iterations(n_wires, len(marked))
    circuit = grover_circuit(predicate, n_wires, iterations=rounds)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    observed = circuit.counts(shots, generator=generator)[0]
    # ``counts`` is typed ``dict[str | int, int]`` because its ``format`` can be "int";
    # this call takes the default "bin", so every key is already the bit string that this
    # mapping is typed as, and the coercion below is a no-op at run time.
    counts = {str(key): count for key, count in observed.items()}
    keys = {state: format(state, f"0{n_wires}b") for state in marked}
    candidates = tuple(
        sorted(
            (state for state in marked if keys[state] in counts),
            key=lambda state: (-counts[keys[state]], state),
        )
    )
    return GroverResult(candidates=candidates, counts=counts, iterations=rounds)


def _append_diffusion(circuit: Circuit, wires: list[int]) -> None:
    """Append one diffusion round -- the reflection about the uniform superposition.

    The reflection about ``|s>`` is ``H X Z_all X H``: the two runs of Hadamards map the
    uniform superposition onto the all-zeros pattern and back, and the multi-controlled Z in
    between flips the sign of that pattern alone, so the composition inverts the amplitude
    about the mean. The gates are emitted in that order on the register wires, the only
    wires the circuit has.

    Args:
        circuit: The circuit to extend.
        wires: The evaluation register's wires.
    """
    for wire in wires:
        circuit.gate("h", wire)
    for wire in wires:
        circuit.gate("x", wire)
    _append_multi_controlled_z(circuit, wires)
    for wire in wires:
        circuit.gate("x", wire)
    for wire in wires:
        circuit.gate("h", wire)


def _append_multi_controlled_z(circuit: Circuit, wires: list[int]) -> None:
    """Append a Z on the all-ones pattern of ``wires``.

    One wire is a plain ``z``. Above that the gate is a Hadamard on the last wire, a
    multi-controlled X across the rest, and the Hadamard again: the Hadamard pair turns the
    last wire's ``|1>`` into a sign, so the composed gate multiplies exactly the all-ones
    pattern by ``-1``. Three wires is the widest this circuit can carry, because the
    multi-controlled X over two controls is the last one the circuit layer supports
    natively.

    Args:
        circuit: The circuit to extend.
        wires: The wires the pattern is read over, at least one.
    """
    if len(wires) == 1:
        circuit.gate("z", wires[0])
        return
    target = wires[-1]
    circuit.gate("h", target)
    append_multi_controlled_x(circuit, wires[:-1], target)
    circuit.gate("h", target)


def _validate_register_width(n_wires: int) -> None:
    """Check the width of a search's evaluation register.

    Args:
        n_wires: The register width as given.

    Raises:
        ValueError: If ``n_wires`` is less than one.
    """
    if n_wires < 1:
        raise ValueError(
            f"a Grover search needs at least one evaluation wire, got n_wires={n_wires}; "
            "a register with no wires has no input to enumerate and nothing to mark"
        )


def _validate_search_width(n_wires: int) -> None:
    """Check that a register width can carry a whole Grover circuit.

    Args:
        n_wires: The register width as given.

    Raises:
        ValueError: If ``n_wires`` is less than one, or more than
            ``_GROVER_WIRE_LIMIT``.
    """
    _validate_register_width(n_wires)
    if n_wires > _GROVER_WIRE_LIMIT:
        raise ValueError(
            f"a Grover search is bounded at {_GROVER_WIRE_LIMIT} evaluation wires, got "
            f"n_wires={n_wires}: the diffusion operator's multi-controlled Z is a "
            f"multi-controlled X with {n_wires - 1} controls, which needs "
            f"len(controls) - 2 = {n_wires - 3} ancillas in |0>, and this circuit is the "
            f"register, so it has no free wire for them. Use append_phase_oracle to append "
            "the oracle to a circuit whose register and ancillas the caller lays out."
        )
