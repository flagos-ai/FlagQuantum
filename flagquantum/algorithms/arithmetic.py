"""Reversible integer addition, built so the compiler can already cost it.

An in-place ripple-carry adder: two ``n_bits``-wide registers go in, the sum
appears on the second, and the first is returned unchanged. The construction is
Cuccaro's majority/unmajority pair -- three operations per bit forward, the
inverse gadget plus two ``cx`` per bit back -- and the whole ripple is carried
by **one** working ancilla rather than one carry wire per bit. ``8 n + 1``
operations carry an ``n``-bit addition: ``2 n`` Toffolis and ``6 n + 1`` ``cx``,
for ``2 n + 2`` wires in total.

**Why this construction and not another.** The naive ripple puts a carry wire
under every bit, which costs ``n`` ancillas that a caller has to allocate, keep
clean, and give back. Cuccaro's pair folds the running carry into the ancilla
itself: ``MAJ(a, b, c)`` maps ``(a, b, c)`` to ``(a ^ c, b ^ c, maj(a, b, c))``,
so the third wire leaves holding the carry *out* of the position it just
processed and is immediately reusable as the carry *into* the next one. The
reverse sweep then walks that carry back down, and by the time ``z`` holds
``c_i`` the two data wires are their original selves, which is exactly when the
sum bit ``a_i ^ b_i ^ c_i`` can be written into ``b_i``. One ancilla, restored to
``|0>`` at the end. Both gadgets are three operations with no angle of their
own, and neither introduces a phase the caller did not write.

**Wire order.** Wire ``0`` is the most significant bit of the first addend and
wire ``n_bits`` is the most significant bit of the second, so the same index
convention holds for both registers and for the state vector: wire ``0`` is the
leftmost bit of a basis-state label. This is the ordering
:mod:`~flagquantum.algorithms.data_encoding` documents and it is applied here
rather than made configurable, because a second index convention would be a
second source of truth for what a basis state is. :func:`adder_wires` returns the
whole map, including the two wires below.

**The ancilla contract.** The working carry wire and the carry-out wire must
enter the circuit holding ``|0>``. The working wire is returned to ``|0>``; the
carry-out wire is left holding the carry out of the most significant position,
which is the ``n_bits + 1``-th bit of the exact integer sum and is otherwise
discarded. The adder is a permutation of the ``2 n + 2`` wires over that domain
and it computes the sum exactly: ``a + b`` is written into the second register
modulo ``2 ** n_bits``, and no approximation, phase, or rounding enters.

**What is not here.** The addends are quantum registers, so there is no
classical addend and no ``add_constant`` entry point, no modular or
controlled variant, and no overflow flag beyond the carry-out wire. The
Gidney-Ekera adder is a different construction that uncomputes its ancillas by
measuring them and feeding the outcome forward, so reaching it needs mid-circuit
measurement and a classical feedforward path rather than a smaller circuit here.

**Where the T-cost comes from.** This module builds the construction and does
not cost it. The compiler already owns the Toffoli rule: ``ccx`` decomposes to
the fifteen-gate ``h``/``t``/``tdg``/``cx`` identity in
:mod:`flagquantum.compiler.basis_translation`, so
:func:`~flagquantum.compiler.basis_conversion.convert_basis` lowers an adder into
a Clifford+T basis and
:func:`~flagquantum.algorithms.logical_resources.estimate_logical_resources`
prices it. A second seven-T expansion here would be a second source of truth for
what a Toffoli costs, which is the one number such a report must not disagree
with itself about.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..circuit import Circuit

__all__ = ["AdderWires", "adder_circuit", "adder_wires"]

_MINIMUM_BITS = 1

# ``MAJ(a, b, c)`` and its inverse, written over local wire labels ``(a, b, c)``.
# The two are exact inverses of each other and each is three operations: ``MAJ``
# maps ``(a, b, c)`` to ``(a ^ c, b ^ c, maj(a, b, c))``, and ``UMA`` maps that
# back. The local labels are resolved to circuit wires per position.
_MAJ = (("cx", (2, 0)), ("cx", (2, 1)), ("ccx", (0, 1, 2)))
_UMA = (("ccx", (0, 1, 2)), ("cx", (2, 1)), ("cx", (2, 0)))


@dataclass(frozen=True)
class AdderWires:
    """Which wire of :func:`adder_circuit` carries which register.

    A program is positional, so the meaning of its wires is not recoverable from
    the program itself. This record is that meaning: the two addend registers
    most significant bit first, the single working carry wire, and the carry-out
    wire.

    Attributes:
        a: The first addend's wires, most significant bit first. Returned
            unchanged by the circuit.
        b: The second addend's wires, most significant bit first. Holds the sum
            once the circuit has run.
        carry: The working carry wire. Must enter as ``|0>`` and is returned to
            ``|0>``.
        carry_out: The carry out of the most significant position. Must enter as
            ``|0>`` and is left holding that carry.
    """

    a: tuple[int, ...]
    b: tuple[int, ...]
    carry: int
    carry_out: int

    @property
    def n_bits(self) -> int:
        """The width of each addend, in bits."""
        return len(self.a)

    @property
    def n_wires(self) -> int:
        """The number of wires :func:`adder_circuit` allocates."""
        return 2 * self.n_bits + 2


def adder_wires(n_bits: int) -> AdderWires:
    """Return the register map for an ``n_bits``-wide adder.

    Args:
        n_bits: The width of each addend, in bits.

    Returns:
        The map of :func:`adder_circuit`'s wires onto its registers.

    Raises:
        TypeError: If ``n_bits`` is not an integer.
        ValueError: If ``n_bits`` is smaller than one.

    Examples:
        >>> from flagquantum.algorithms.arithmetic import adder_wires
        >>> wires = adder_wires(3)
        >>> print(wires.a, wires.b, wires.carry, wires.carry_out, wires.n_wires)
        (0, 1, 2) (3, 4, 5) 6 7 8
    """

    count = _require_bit_count(n_bits)
    return AdderWires(
        a=tuple(range(count)),
        b=tuple(range(count, 2 * count)),
        carry=2 * count,
        carry_out=2 * count + 1,
    )


def adder_circuit(n_bits: int) -> Circuit:
    """Return the in-place ripple-carry adder over ``n_bits`` bits.

    The returned circuit is a :class:`~flagquantum.circuit.Circuit` and nothing
    else: it is not executed, it selects no runtime, and it carries no simulator.
    :func:`adder_wires` says which wire is which, and the module docstring states
    the ancilla contract the caller has to meet.

    Args:
        n_bits: The width of each addend, in bits.

    Returns:
        A circuit of ``2 n_bits + 2`` wires applying ``2 n_bits`` Toffolis and
        ``6 n_bits + 1`` ``cx``.

    Raises:
        TypeError: If ``n_bits`` is not an integer.
        ValueError: If ``n_bits`` is smaller than one.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.arithmetic import adder_circuit
        >>> circuit = adder_circuit(3)
        >>> print(circuit.n_wires)
        8
        >>> from collections import Counter
        >>> print(sorted(Counter(i.name for i in circuit.to_ir().instructions).items()))
        [('ccx', 6), ('cx', 19)]
    """

    count = _require_bit_count(n_bits)
    wires = adder_wires(count)
    circuit = Circuit(wires.n_wires)

    # Position ``i`` carries bit weight ``2 ** i``, and wire ``0`` is the most
    # significant bit, so the least significant position is the highest wire of
    # each register. The sweep below runs least significant first; the sweep back
    # is therefore the register order taken at face value.
    for a_wire, b_wire in zip(reversed(wires.a), reversed(wires.b), strict=True):
        _append(circuit, _MAJ, a_wire, b_wire, wires.carry)

    # The carry out of the most significant position leaves the ripple here, and
    # the ancilla takes the carries back down over the next sweep.
    circuit.gate("cx", (wires.carry, wires.carry_out))

    for a_wire, b_wire in zip(wires.a, wires.b, strict=True):
        _append(circuit, _UMA, a_wire, b_wire, wires.carry)
        # ``UMA`` restores both data wires and leaves the carry into this
        # position on the ancilla, which is the ``c_i`` the sum bit is missing.
        circuit.gate("cx", (a_wire, b_wire))
        circuit.gate("cx", (wires.carry, b_wire))

    return circuit


def _append(
    circuit: Circuit,
    template: tuple[tuple[str, tuple[int, ...]], ...],
    a: int,
    b: int,
    carry: int,
) -> None:
    """Append one gadget, resolving its local wire labels to circuit wires."""

    local = (a, b, carry)
    for opcode, labels in template:
        circuit.gate(opcode, tuple(local[label] for label in labels))


def _require_bit_count(n_bits: int) -> int:
    """Return ``n_bits`` after refusing anything that is not a positive integer."""

    if isinstance(n_bits, bool) or not isinstance(n_bits, int):
        raise TypeError(
            f"n_bits must be an integer, got {type(n_bits).__name__}: {n_bits!r}"
        )
    if n_bits < _MINIMUM_BITS:
        raise ValueError(f"n_bits must be at least 1, got {n_bits}")
    return n_bits
