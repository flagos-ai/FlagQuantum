"""Reversible classical logic: multi-controlled X and a bit-string comparator.

Both units here are reversible classical logic written as gates. Multi-controlled X flips
one target wire exactly on the operand pattern that sets every control. The comparator XORs
one target wire with the truth value of ``lhs > rhs`` for two bit strings of equal width,
read most significant bit first, and leaves every other wire it was given where it found it.
They are the building blocks the oracle units are composed from.

**No advantage premise.** These are reversible classical circuits of ``O(n)`` Toffoli-style
cost. Each computes exactly what the matching classical predicate computes, over the same
input, with no asymptotic saving of any kind. They carry no performance, capacity, or
hardware claim, and their cost is paid by whatever algorithm calls them.

The comparator follows D. S. Oliveira and R. V. Ramos, "Quantum bit string comparator:
circuits and applications", *Quantum Computers and Computing* **7**(1), 17-26 (2007). That
record is **semi-verified**: its venue is not indexed by Crossref, DBLP, or INSPIRE, so the
volume and page numbers are reported by citing works rather than index-confirmed. A fully
verified adjacent record by the same group is D. S. Oliveira, P. B. M. de Sousa and R. V.
Ramos, 2006 International Telecommunications Symposium, DOI 10.1109/ITS.2006.4433341.
Multi-controlled X is standard reversible logic, and no paper is cited for it here.

**The ancilla precondition.** There is no native gate for three or more controls, so
:func:`append_multi_controlled_x` builds its ladder on ``len(controls) - 2`` caller-supplied
ancillas, and each of those must be in ``|0>`` on entry. Measured with a dirty ancilla, the
target comes out wrong on a large fraction of inputs -- 8 of 16 operand patterns at three
controls and 32 of 96 at four -- and nothing is raised; the ancilla itself is never
corrupted, so the failure is invisible from the ancilla. A circuit builder has no way to
learn where a wire starts, so meeting that precondition is the caller's to do.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...circuit import Circuit

__all__ = ["append_comparator", "append_multi_controlled_x"]


def append_multi_controlled_x(
    circuit: Circuit,
    controls: Sequence[int],
    target: int,
    *,
    ancillas: Sequence[int] | None = None,
) -> None:
    """Append a multi-controlled X to ``circuit`` in place.

    The gate flips ``target`` exactly when every wire in ``controls`` is set, and moves no
    other wire. One control is a ``cx`` and two are a ``ccx``, which the circuit layer
    already carries. Three or more controls have no native gate, so the gate is built as an
    ancilla ladder: the controls are folded onto the ancillas from the first to the last,
    the last control is folded onto the target, and then the whole ladder is uncomputed in
    reverse. Every gate in the ladder is its own inverse, so the ancillas come back to the
    value they entered with.

    Args:
        circuit: The circuit to extend.
        controls: The control wires, all of them distinct and none of them ``target``.
        target: The wire to flip on saturation.
        ancillas: The wires the ladder computes into, ``len(controls) - 2`` of them for
            three or more controls and none for fewer. **Each ancilla must be in ``|0>``
            on entry, and each is restored to ``|0>`` on exit.** A wire that enters in
            ``|1>``, or in any state other than ``|0>``, makes the target silently wrong on
            a large fraction of operand patterns and raises nothing; the ancilla's own
            value is still left unchanged by the ladder, so the fault cannot be seen from
            the ancilla either. These must be wired in the order the ladder computes them:
            ``ancillas[i]`` carries the conjunction of ``controls[:i + 2]``.

    Raises:
        ValueError: If ``controls`` is empty, repeats a wire, or holds ``target``; if three
            or more controls are given without ``ancillas``; if the ancilla count is not
            ``len(controls) - 2``; or if an ancilla repeats a wire or is also a control or
            the target.
    """
    ordered = list(controls)
    _validate_controls(ordered, target)
    needed = max(len(ordered) - 2, 0)
    spare: list[int] = []
    if ancillas is None:
        if needed:
            raise ValueError(
                f"a multi-controlled X with {len(ordered)} controls needs {needed} "
                f"ancillas, each starting in |0>, and none were given; pass "
                f"ancillas=[...] with {needed} free wire(s)"
            )
    else:
        spare = list(ancillas)
        if len(spare) != needed:
            raise ValueError(
                f"a multi-controlled X with {len(ordered)} controls needs {needed} "
                f"ancillas, got {len(spare)}"
            )
        _validate_ancillas(spare, ordered, target)
    _append_ladder(circuit, ordered, target, spare)


def append_comparator(
    circuit: Circuit,
    *,
    lhs: Sequence[int],
    rhs: Sequence[int],
    target: int,
    equality: Sequence[int],
    scratch: int,
) -> None:
    """Append a reversible ``lhs > rhs`` comparator to ``circuit`` in place.

    The comparison reads both operands most significant bit first, so ``lhs[0]`` is the
    highest bit of the left operand. The circuit XORs ``target`` with the truth value of
    ``lhs > rhs``, and that is the only wire it changes: this is the clean form, in which
    the prefix-equality ladder and the scratch wire are uncomputed, so the primitive
    composes into a larger oracle instead of leaving ``n`` wires holding intermediate
    flags.

    The construction first builds a ladder of prefix-equality flags, with ``equality[i]``
    holding "the first ``i`` bits of the two operands are equal"; ``equality[0]`` is driven
    to ``|1>`` so every flag has a definite control above it, and ``equality[n]`` ends
    holding ``lhs == rhs``. Each flag then conditions a three-control X whose other two
    controls are the two operand bits with the right-hand bit flipped, so the gate fires
    exactly at the most significant position where the operands differ and the left operand
    is set -- which is ``lhs > rhs``. The middle pass swaps the scratch wire in as that
    ladder's ancilla and takes it back out, and the equality ladder is then uncomputed in
    descending ``i``.

    Args:
        circuit: The circuit to extend.
        lhs: The left operand's wires, most significant first.
        rhs: The right operand's wires, most significant first, as many as ``lhs``.
        target: The wire XORed with ``lhs > rhs``.
        equality: The ``len(lhs) + 1`` flag wires, in ladder order. All of them must start
            in ``|0>``; they are returned to ``|0>``, so the caller may reuse them.
        scratch: One wire used as the ladder's ancilla. It must start in ``|0>`` and is
            returned to ``|0>``.

    Raises:
        ValueError: If the operands differ in width, are empty, or if ``equality`` is not
            ``len(lhs) + 1`` wires; or if a wire repeats across the operands, the target,
            the equality flags, and the scratch wire.
    """
    left = list(lhs)
    right = list(rhs)
    flags = list(equality)
    _validate_comparator(left, right, target, flags, scratch)

    circuit.gate("x", flags[0])
    for index in range(len(left)):
        _append_equality_step(
            circuit, left[index], right[index], flags[index], flags[index + 1], scratch
        )
    for index in range(len(left)):
        circuit.gate("x", right[index])
        append_multi_controlled_x(
            circuit,
            [flags[index], left[index], right[index]],
            target,
            ancillas=[scratch],
        )
        circuit.gate("x", right[index])
    for index in reversed(range(len(left))):
        _append_equality_step(
            circuit, left[index], right[index], flags[index], flags[index + 1], scratch
        )
    circuit.gate("x", flags[0])


def _append_equality_step(
    circuit: Circuit,
    left: int,
    right: int,
    previous: int,
    following: int,
    scratch: int,
) -> None:
    """Fold one operand bit into the prefix-equality ladder, or undo one fold.

    The body is the five-step unit the comparator runs in both directions. It borrows
    ``scratch`` to hold "these two bits are equal", combines that with the flag above into
    the next flag, and gives ``scratch`` back set to ``|0>``. It reads only ``previous``,
    ``left``, ``right``, and ``scratch``, and writes only ``following`` and ``scratch``;
    every gate in it is its own inverse, so running it again undoes the fold. That is why
    the uncompute pass can be the same call in descending ``index``: the flag above is
    still carrying its forward value at the moment the flag below is cleared.

    Args:
        circuit: The circuit to extend.
        left: The left operand's bit for this position.
        right: The right operand's bit for this position.
        previous: The flag holding "the bits above this one are equal".
        following: The flag to XOR with ``previous`` and the two bits' equality.
        scratch: The wire borrowed for the equality of the two bits.
    """
    circuit.gate("cx", (left, scratch))
    circuit.gate("cx", (right, scratch))
    circuit.gate("x", scratch)
    circuit.gate("ccx", (previous, scratch, following))
    circuit.gate("x", scratch)
    circuit.gate("cx", (right, scratch))
    circuit.gate("cx", (left, scratch))


def _append_ladder(
    circuit: Circuit, controls: Sequence[int], target: int, ancillas: Sequence[int]
) -> None:
    """Emit the multi-controlled X ladder for ``controls``.

    The first control pair seeds ``ancillas[0]``; each inner step folds the next control
    onto the next ancilla, so ``ancillas[i]`` ends holding the conjunction of
    ``controls[:i + 2]``. The last control is folded onto the target, after which the two
    runs are replayed backwards to clear every ancilla. Each inner gate is emitted twice,
    once on the way up and once on the way down, which is what returns the ancillas to
    ``|0>``; a ladder that skipped the descending run would compute the right target and
    leave every ancilla dirty.

    Args:
        circuit: The circuit to extend.
        controls: The control wires, at least one, none of them ``target``.
        target: The wire to flip on saturation.
        ancillas: The ladder's ancillas, ``max(len(controls) - 2, 0)`` of them.
    """
    count = len(controls)
    if count == 1:
        circuit.gate("cx", (controls[0], target))
        return
    if count == 2:
        circuit.gate("ccx", (controls[0], controls[1], target))
        return
    circuit.gate("ccx", (controls[0], controls[1], ancillas[0]))
    for index in range(2, count - 1):
        circuit.gate("ccx", (ancillas[index - 2], controls[index], ancillas[index - 1]))
    circuit.gate("ccx", (ancillas[count - 3], controls[count - 1], target))
    for index in reversed(range(2, count - 1)):
        circuit.gate("ccx", (ancillas[index - 2], controls[index], ancillas[index - 1]))
    circuit.gate("ccx", (controls[0], controls[1], ancillas[0]))


def _validate_controls(controls: Sequence[int], target: int) -> None:
    """Check the control and target wires of a multi-controlled X.

    Args:
        controls: The control wires as given.
        target: The target wire as given.

    Raises:
        ValueError: If ``controls`` is empty, repeats a wire, or contains ``target``.
    """
    if not controls:
        raise ValueError(
            "a multi-controlled X needs at least one control wire, got none; a gate with "
            "no controls would flip the target unconditionally"
        )
    if len(set(controls)) != len(controls):
        raise ValueError(
            f"control wires must be distinct, got {list(controls)}; a repeated control "
            "would drive the same wire twice"
        )
    if target in controls:
        raise ValueError(
            f"the target wire {target} must not also be a control; a wire that controls "
            "its own flip would clear itself"
        )


def _validate_ancillas(
    ancillas: Sequence[int], controls: Sequence[int], target: int
) -> None:
    """Check the ancilla wires of a multi-controlled X.

    Args:
        ancillas: The ancilla wires as given, already known to be the right count.
        controls: The control wires.
        target: The target wire.

    Raises:
        ValueError: If an ancilla repeats a wire, a control, or the target.
    """
    if len(set(ancillas)) != len(ancillas):
        raise ValueError(
            f"ancilla wires must be distinct, got {list(ancillas)}; a repeated ancilla "
            "would be folded onto itself and cannot hold the ladder's conjunction"
        )
    occupied = set(controls)
    occupied.add(target)
    for wire in ancillas:
        if wire in occupied:
            raise ValueError(
                f"ancilla wire {wire} is also a control or the target; the ladder needs "
                "each ancilla to itself and in |0> on entry"
            )


def _validate_comparator(
    left: Sequence[int],
    right: Sequence[int],
    target: int,
    flags: Sequence[int],
    scratch: int,
) -> None:
    """Check the wires of a comparator.

    Args:
        left: The left operand's wires.
        right: The right operand's wires.
        target: The wire XORed with the comparison's truth value.
        flags: The prefix-equality flag wires.
        scratch: The wire the inner ladder borrows.

    Raises:
        ValueError: If the operands differ in width or are empty, if the flag register is
            not one wider than an operand, or if a wire repeats.
    """
    if len(left) != len(right):
        raise ValueError(
            f"the operands must be as wide as each other, got {len(left)} left wires "
            f"and {len(right)} right wires"
        )
    if not left:
        raise ValueError(
            "the comparator needs at least one operand bit, got two empty operands"
        )
    if len(flags) != len(left) + 1:
        raise ValueError(
            f"the equality register is len(lhs) + 1 = {len(left) + 1} wires, one flag "
            f"per bit plus the constant input, got {len(flags)}"
        )
    occupied = list(left) + list(right) + list(flags) + [target, scratch]
    if len(set(occupied)) != len(occupied):
        raise ValueError(
            "the operands, the target, the equality flags, and the scratch wire must be "
            f"distinct wires, got {occupied}"
        )
