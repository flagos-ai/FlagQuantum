"""Reversible integer addition, built so the compiler can already cost it.

An in-place ripple-carry adder: two ``n_bits``-wide registers go in, the sum
appears on the second, and the first is returned unchanged. The construction is
Cuccaro's majority/unmajority pair -- three operations per bit forward, the
inverse gadget plus two ``cx`` per bit back -- and the whole ripple is carried
by **one** working ancilla rather than one carry qubit per bit. ``8 n + 1``
operations carry an ``n``-bit addition: ``2 n`` Toffolis and ``6 n + 1`` ``cx``,
for ``2 n + 2`` qubits in total.

**Why this construction and not another.** The naive ripple puts a carry qubit
under every bit, which costs ``n`` ancillas that a caller has to allocate, keep
clean, and give back. Cuccaro's pair folds the running carry into the ancilla
itself: ``MAJ(a, b, c)`` maps ``(a, b, c)`` to ``(a ^ c, b ^ c, maj(a, b, c))``,
so the third qubit leaves holding the carry *out* of the position it just
processed and is immediately reusable as the carry *into* the next one. The
reverse sweep then walks that carry back down, and by the time ``z`` holds
``c_i`` the two data qubits are their original selves, which is exactly when the
sum bit ``a_i ^ b_i ^ c_i`` can be written into ``b_i``. One ancilla, restored to
``|0>`` at the end. Both gadgets are three operations with no angle of their
own, and neither introduces a phase the caller did not write.

**Qubit order.** Qubit ``0`` is the most significant bit of the first addend and
qubit ``n_bits`` is the most significant bit of the second, so the same index
convention holds for both registers and for the state vector: qubit ``0`` is the
leftmost bit of a basis-state label. This is the ordering
:mod:`~flagquantum.algorithms.data_encoding` documents and it is applied here
rather than made configurable, because a second index convention would be a
second source of truth for what a basis state is. :func:`adder_qubits` returns the
whole map, including the two qubits below.

**The ancilla contract.** The working carry qubit and the carry-out qubit must
enter the circuit holding ``|0>``. The working qubit is returned to ``|0>``; the
carry-out qubit is left holding the carry out of the most significant position,
which is the ``n_bits + 1``-th bit of the exact integer sum and is otherwise
discarded. The adder is a permutation of the ``2 n + 2`` qubits over that domain
and it computes the sum exactly: ``a + b`` is written into the second register
modulo ``2 ** n_bits``, and no approximation, phase, or rounding enters.

**What is not here.** The addends are quantum registers, so there is no
classical addend and no ``add_constant`` entry point, no modular or
controlled variant, and no overflow flag beyond the carry-out qubit. The
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
from ..compiler._arithmetic_kernels import ripple_add

__all__ = ["AdderQubits", "adder_circuit", "adder_qubits"]

_MINIMUM_BITS = 1


@dataclass(frozen=True)
class AdderQubits:
    """Which qubit of :func:`adder_circuit` carries which register.

    A program is positional, so the meaning of its qubits is not recoverable from
    the program itself. This record is that meaning: the two addend registers
    most significant bit first, the single working carry qubit, and the carry-out
    qubit.

    Attributes:
        a: The first addend's qubits, most significant bit first. Returned
            unchanged by the circuit.
        b: The second addend's qubits, most significant bit first. Holds the sum
            once the circuit has run.
        carry: The working carry qubit. Must enter as ``|0>`` and is returned to
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
    def n_qubits(self) -> int:
        """The number of qubits :func:`adder_circuit` allocates."""
        return 2 * self.n_bits + 2


def adder_qubits(n_bits: int) -> AdderQubits:
    """Return the register map for an ``n_bits``-wide adder.

    Args:
        n_bits: The width of each addend, in bits.

    Returns:
        The map of :func:`adder_circuit`'s qubits onto its registers.

    Raises:
        TypeError: If ``n_bits`` is not an integer.
        ValueError: If ``n_bits`` is smaller than one.

    Examples:
        >>> from flagquantum.algorithms.arithmetic import adder_qubits
        >>> registers = adder_qubits(3)
        >>> print(registers.a, registers.b, registers.carry, registers.carry_out)
        (0, 1, 2) (3, 4, 5) 6 7
        >>> print(registers.n_qubits)
        8
    """

    count = _require_bit_count(n_bits)
    return AdderQubits(
        a=tuple(range(count)),
        b=tuple(range(count, 2 * count)),
        carry=2 * count,
        carry_out=2 * count + 1,
    )


def adder_circuit(n_bits: int) -> Circuit:
    """Return the in-place ripple-carry adder over ``n_bits`` bits.

    The returned circuit is a :class:`~flagquantum.circuit.Circuit` and nothing
    else: it is not executed, it selects no runtime, and it carries no simulator.
    :func:`adder_qubits` says which qubit is which, and the module docstring states
    the ancilla contract the caller has to meet.

    Args:
        n_bits: The width of each addend, in bits.

    Returns:
        A circuit of ``2 n_bits + 2`` qubits applying ``2 n_bits`` Toffolis and
        ``6 n_bits + 1`` ``cx``.

    Raises:
        TypeError: If ``n_bits`` is not an integer.
        ValueError: If ``n_bits`` is smaller than one.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.algorithms.arithmetic import adder_circuit
        >>> circuit = adder_circuit(3)
        >>> print(circuit.n_qubits)
        8
        >>> from collections import Counter
        >>> print(sorted(Counter(i.name for i in circuit.to_ir().instructions).items()))
        [('ccx', 6), ('cx', 19)]
    """

    count = _require_bit_count(n_bits)
    registers = adder_qubits(count)
    circuit = Circuit(registers.n_qubits)

    for instruction in ripple_add(
        registers.a, registers.b, registers.carry, registers.carry_out
    ):
        circuit.gate(instruction.name, instruction.wires)
    return circuit


def _require_bit_count(n_bits: int) -> int:
    """Return ``n_bits`` after refusing anything that is not a positive integer."""

    if isinstance(n_bits, bool) or not isinstance(n_bits, int):
        raise TypeError(
            f"n_bits must be an integer, got {type(n_bits).__name__}: {n_bits!r}"
        )
    if n_bits < _MINIMUM_BITS:
        raise ValueError(f"n_bits must be at least 1, got {n_bits}")
    return n_bits
