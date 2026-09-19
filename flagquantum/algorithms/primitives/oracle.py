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

**Truth-table synthesis.** :func:`marked_states` and the oracle units above the building
blocks -- :func:`phase_oracle` and :func:`append_phase_oracle`, :func:`bit_oracle` and
:func:`append_bit_oracle` -- turn a classical predicate into a quantum oracle by
enumerating the predicate's truth table. Each marked input is mapped onto the all-ones
pattern with X gates, acted on by the multi-controlled X above, and mapped back, so the
phase oracle multiplies exactly the marked amplitudes by ``-1``, and the bit oracle XORs
the predicate's value onto its target wire and then restores every wire it allocated. The
loop enumerates all ``2**n`` inputs, so synthesis costs ``O(2**n)`` classically and **no
advantage follows from it at any scale**: the oracle a query-model algorithm is handed
here is no cheaper than the classical search it is meant to replace. Truth-table synthesis
is standard textbook material, and no paper is cited for it.

The standalone :func:`phase_oracle` is capped at three wires. Its multi-controlled Z is a
multi-controlled X with ``n - 1`` controls, and a multi-controlled X above two controls
needs ``len(controls) - 2`` caller-supplied ancillas -- ``n - 3`` of them from ``n = 4`` up
-- while a circuit pinned at exactly ``n_wires`` wires has none to spare. The append form
takes the caller's register and its ancillas. :func:`bit_oracle` has no such cap, because
it allocates the ladder ancillas it needs and restores them.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...circuit import Circuit
from .types import Predicate

__all__ = [
    "append_bit_oracle",
    "append_comparator",
    "append_multi_controlled_x",
    "append_phase_oracle",
    "bit_oracle",
    "marked_states",
    "phase_oracle",
]

# A standalone phase oracle has one wire per register bit and none to spare, so the
# multi-controlled X inside its multi-controlled Z can fold at most two controls onto it.
_PHASE_ORACLE_WIRE_LIMIT = 3


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
            the target. A non-empty ``ancillas`` alongside one or two controls is refused,
            because that form consumes no ancilla, while an empty one is accepted.
    """
    ordered = list(controls)
    _validate_controls(ordered, target)
    needed = max(len(ordered) - 2, 0)
    spare: list[int] = []
    if ancillas is None:
        if needed:
            # One ladder wire per control above the two a native gate takes, and this
            # branch is only reached when there is at least one, so only a single-wire
            # ladder takes the singular noun.
            ancilla_word = "ancilla" if needed == 1 else "ancillas"
            free_word = "wire" if needed == 1 else "wires"
            raise ValueError(
                f"a multi-controlled X with {len(ordered)} controls needs {needed} "
                f"{ancilla_word}, each starting in |0>, and none were given; pass "
                f"ancillas=[...] with {needed} free {free_word}"
            )
    else:
        spare = list(ancillas)
        if len(spare) != needed:
            # A count of zero is what one or two controls need, and it takes the plural
            # noun; only the single-wire ladder takes the singular. This branch is reached
            # from one control up, so the control noun takes the singular there.
            control_word = "control" if len(ordered) == 1 else "controls"
            ancilla_word = "ancilla" if needed == 1 else "ancillas"
            raise ValueError(
                f"a multi-controlled X with {len(ordered)} {control_word} needs {needed} "
                f"{ancilla_word}, got {len(spare)}"
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

    **All ``len(lhs) + 1`` equality flags and the scratch wire must be in ``|0>`` on
    entry**: a dirty ``equality[0]`` makes the target silently wrong, and the ladder
    restores the flag, so nothing is raised.

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


def marked_states(predicate: Predicate, n_wires: int) -> tuple[int, ...]:
    """List the register values ``predicate`` marks, in ascending order.

    The set is the predicate's truth table, read by calling ``predicate`` once per input of
    an ``n_wires``-wide register, all ``2**n_wires`` of them. That enumeration is the whole
    cost of the synthesis in the oracle builders below: a classical pass of ``O(2**n)``
    predicate calls, which is why an oracle built this way carries no advantage of its own.

    Args:
        predicate: The property to evaluate, one register value at a time.
        n_wires: The width of the register the predicate is read over.

    Returns:
        Every ``value`` in ``range(2**n_wires)`` for which ``predicate(value)`` is truthy,
        in ascending order.

    Raises:
        ValueError: If ``n_wires`` is less than one; a register with no wires has no input
            to enumerate.
    """
    _validate_register_width(n_wires)
    return tuple(value for value in range(2**n_wires) if predicate(value))


def phase_oracle(predicate: Predicate, n_wires: int) -> Circuit:
    """Build a standalone phase oracle over ``n_wires`` wires for ``predicate``.

    The circuit carries exactly ``n_wires`` wires, one per register bit, and multiplies the
    amplitude of every marked basis state by ``-1`` while leaving the other basis states
    alone. That fixed width is what an algorithm wrapping the register needs, and it is
    also why this form stops at three wires: the ancilla the multi-controlled Z needs above
    that has no free wire to sit on. The append form takes a register the caller supplies.

    Args:
        predicate: The property the oracle marks.
        n_wires: The width of the register, at most three.

    Returns:
        The standalone phase oracle, its wires numbered ``0`` to ``n_wires - 1``.

    Raises:
        ValueError: If ``n_wires`` is less than one, or more than three; the message names
            the ancillas the wider construction needs and the form that takes them.
    """
    _validate_register_width(n_wires)
    if n_wires > _PHASE_ORACLE_WIRE_LIMIT:
        # This branch is only reached above the cap, so at least one ancilla is needed;
        # four wires is the only width that takes the singular noun.
        ancilla_word = "ancilla" if n_wires == 4 else "ancillas"
        raise ValueError(
            f"a standalone phase oracle is capped at {_PHASE_ORACLE_WIRE_LIMIT} wires, got "
            f"{n_wires}: its multi-controlled Z is a multi-controlled X with {n_wires - 1} "
            f"controls, and a multi-controlled X above two controls needs "
            f"len(controls) - 2 = {n_wires - 3} {ancilla_word} in |0>, which a circuit of "
            f"exactly {n_wires} wires has no free wire for. Use append_phase_oracle to "
            f"append the oracle to a circuit whose register and ancillas the caller lays "
            f"out, or bit_oracle, which allocates its own ladder ancillas."
        )
    circuit = Circuit(n_wires)
    append_phase_oracle(circuit, predicate, list(range(n_wires)))
    return circuit


def append_phase_oracle(
    circuit: Circuit,
    predicate: Predicate,
    wires: Sequence[int],
    *,
    ancillas: Sequence[int] = (),
) -> None:
    """Append a phase oracle for ``predicate`` to ``circuit`` in place.

    For each marked register value, the wires whose bit is zero are flipped, so that
    value's basis state is mapped onto the all-ones pattern; a Z on that pattern is applied;
    and the flips are undone. The Z on the all-ones pattern is the only gate that touches a
    phase, so the net effect is that exactly the marked basis states pick up a factor of
    ``-1``, and every other basis state is returned to where it started.

    Args:
        circuit: The circuit to extend.
        predicate: The property the oracle marks.
        wires: The register's wires, most significant first.
        ancillas: The wires the multi-controlled Z's multi-controlled X folds onto, one per
            control above two of them: ``max(len(wires) - 3, 0)`` wires, none at all at
            three wires or fewer. **Each must be in ``|0>`` on entry**, and each is
            restored to ``|0>`` on exit. They must lie outside the register.

    Raises:
        ValueError: If ``wires`` is empty or repeats a wire; or if the ancilla count is not
            ``max(len(wires) - 3, 0)``, or an ancilla repeats a register wire.
    """
    ordered = list(wires)
    _validate_register(ordered, "phase oracle")
    spare = list(ancillas)
    _validate_ladder_ancillas(ordered, spare, 3, "phase oracle")
    for value in range(2 ** len(ordered)):
        if not predicate(value):
            continue
        zeros = _zero_wires(ordered, value)
        for wire in zeros:
            circuit.gate("x", wire)
        _append_multi_controlled_z(circuit, ordered, spare)
        for wire in zeros:
            circuit.gate("x", wire)


def bit_oracle(predicate: Predicate, n_wires: int) -> Circuit:
    """Build a standalone bit oracle over ``n_wires`` wires for ``predicate``.

    The circuit XORs the predicate's value onto one output wire, so on a register holding
    ``value`` the output wire carries ``predicate(value)``. Because the value is XORed and
    not assigned, and because every wire the oracle allocates is restored, applying the
    oracle twice is the identity.

    The circuit carries ``n_wires + 1 + max(0, n_wires - 2)`` wires: the evaluation
    register, then the output wire, then the ladder ancillas the multi-controlled X needs
    above two controls. They are allocated, started in ``|0>``, and restored by the builder,
    which is why this form has no wire cap.

    Args:
        predicate: The property the oracle marks.
        n_wires: The width of the evaluation register.

    Returns:
        The standalone bit oracle, its evaluation wires numbered ``0`` to ``n_wires - 1``,
        its output wire ``n_wires``, and its ladder ancillas above that.

    Raises:
        ValueError: If ``n_wires`` is less than one, since a register with no wires has no
            input to enumerate.
    """
    _validate_register_width(n_wires)
    spare = max(0, n_wires - 2)
    circuit = Circuit(n_wires + 1 + spare)
    append_bit_oracle(
        circuit,
        predicate,
        list(range(n_wires)),
        target=n_wires,
        ancillas=list(range(n_wires + 1, n_wires + 1 + spare)),
    )
    return circuit


def append_bit_oracle(
    circuit: Circuit,
    predicate: Predicate,
    wires: Sequence[int],
    *,
    target: int,
    ancillas: Sequence[int] = (),
) -> None:
    """Append a bit oracle for ``predicate`` to ``circuit`` in place.

    The construction mirrors :func:`append_phase_oracle`, with the multi-controlled X
    writing to ``target`` instead of a phase being applied to the all-ones pattern: each
    marked value's zero bits are flipped, the multi-controlled X XORs ``target`` with the
    all-ones pattern's occupancy, and the flips are undone. The output wire therefore ends
    holding ``predicate(value)`` XORed with whatever it entered with, and every other wire,
    the ladder ancillas included, comes back to the value it entered with.

    Args:
        circuit: The circuit to extend.
        predicate: The property the oracle marks.
        wires: The evaluation register's wires, most significant first.
        target: The wire XORed with the predicate's value. It must not be a register wire.
        ancillas: The wires the ladder folds onto, ``max(len(wires) - 2, 0)`` of them, none
            at all at two wires or fewer. **Each must be in ``|0>`` on entry** and each is
            restored to ``|0>`` on exit. They must be wired in the order the ladder computes
            them: ``ancillas[i]`` carries the conjunction of ``wires[:i + 2]``.

    Raises:
        ValueError: If ``wires`` is empty or repeats a wire; if the ancilla count is not
            ``max(len(wires) - 2, 0)``; or if the target or an ancilla repeats a wire from
            the register, the target, or the ancillas.
    """
    ordered = list(wires)
    _validate_register(ordered, "bit oracle")
    spare = list(ancillas)
    _validate_ladder_ancillas(ordered, spare, 2, "bit oracle")
    occupied = ordered + [target] + spare
    if len(set(occupied)) != len(occupied):
        raise ValueError(
            "the evaluation register, the target, and the ladder ancillas must be "
            f"distinct wires, got wires={ordered}, target={target}, ancillas={spare}; a "
            "wire that is read and written by the same oracle cannot carry both"
        )
    for value in range(2 ** len(ordered)):
        if not predicate(value):
            continue
        zeros = _zero_wires(ordered, value)
        for wire in zeros:
            circuit.gate("x", wire)
        append_multi_controlled_x(circuit, ordered, target, ancillas=spare)
        for wire in zeros:
            circuit.gate("x", wire)


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
        left_word = "wire" if len(left) == 1 else "wires"
        right_word = "wire" if len(right) == 1 else "wires"
        raise ValueError(
            f"the operands must be as wide as each other, got {len(left)} left "
            f"{left_word} and {len(right)} right {right_word}"
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


def _validate_register_width(n_wires: int) -> None:
    """Check the width of a truth table's evaluation register.

    Args:
        n_wires: The register width as given.

    Raises:
        ValueError: If ``n_wires`` is less than one.
    """
    if n_wires < 1:
        raise ValueError(
            f"a register is at least one wire wide, got n_wires={n_wires}; with no wires "
            "there is no input to enumerate and nothing to mark"
        )


def _validate_register(wires: Sequence[int], caller: str) -> None:
    """Check the evaluation register of an oracle builder.

    Args:
        wires: The register wires as given.
        caller: The name of the builder being validated, for the message.

    Raises:
        ValueError: If ``wires`` is empty or repeats a wire.
    """
    if not wires:
        raise ValueError(
            f"a {caller} needs at least one register wire, got none; with no wires there "
            "is no input to enumerate and nothing to mark"
        )
    if len(set(wires)) != len(wires):
        raise ValueError(
            f"the register wires of a {caller} must be distinct, got {list(wires)}; a "
            "repeated wire would be conditioned on and flipped by the same oracle"
        )


def _validate_ladder_ancillas(
    register: Sequence[int],
    ancillas: Sequence[int],
    free_at: int,
    caller: str,
) -> None:
    """Check the ladder ancillas of an oracle against the register that drives them.

    Args:
        register: The evaluation wires the predicate is read over.
        ancillas: The ladder ancillas as given.
        free_at: The register width at or below which the oracle's widest controlled gate
            consumes no ancilla -- three for a phase oracle's multi-controlled Z, two for a
            bit oracle's multi-controlled X.
        caller: The name of the builder being validated, for the message.

    Raises:
        ValueError: If the ancilla count is not ``max(len(register) - free_at, 0)``, or an
            ancilla is also a register wire or repeats another ancilla.
    """
    needed = max(len(register) - free_at, 0)
    if len(ancillas) != needed:
        raise ValueError(
            f"a {caller} folds its ladder onto max(0, n_wires - {free_at}) wires -- "
            f"{needed} of them for this register -- and was given {len(ancillas)}; the "
            "ladder wires hold successive prefix conjunctions of the controls above the "
            "two a native gate takes, and each must be in |0> on entry"
        )
    occupied = list(register) + list(ancillas)
    if len(set(occupied)) != len(occupied):
        raise ValueError(
            f"the register and the ladder ancillas of a {caller} must be distinct wires, "
            f"got wires={list(register)}, ancillas={list(ancillas)}; an ancilla folded "
            "onto a register wire would corrupt the input it is reading"
        )


def _zero_wires(wires: Sequence[int], value: int) -> list[int]:
    """Return the wires whose bit is zero in ``value``, read most significant first.

    Args:
        wires: The register wires, most significant first.
        value: The register value to read.

    Returns:
        The wires carrying a zero bit in ``value``, in register order.
    """
    width = len(wires)
    return [
        wire
        for index, wire in enumerate(wires)
        if not (value >> (width - 1 - index)) & 1
    ]


def _append_multi_controlled_z(
    circuit: Circuit, wires: Sequence[int], ancillas: Sequence[int]
) -> None:
    """Append a Z on the all-ones pattern of ``wires``.

    One wire is a plain ``z``. Above that the gate is a Hadamard on the last wire, a
    multi-controlled X across the rest, and the Hadamard again: the Hadamard pair turns the
    target wire's ``|1>`` into a sign, so the composed gate multiplies exactly the all-ones
    pattern by ``-1`` and leaves every other pattern alone. Above three wires the
    multi-controlled X folds onto the ancillas it is given, each of which must be in
    ``|0>``.

    Args:
        circuit: The circuit to extend.
        wires: The wires the pattern is read over, at least one.
        ancillas: The ladder ancillas, ``max(len(wires) - 3, 0)`` of them.
    """
    if len(wires) == 1:
        circuit.gate("z", wires[0])
        return
    target = wires[-1]
    circuit.gate("h", target)
    append_multi_controlled_x(circuit, wires[:-1], target, ancillas=list(ancillas))
    circuit.gate("h", target)
