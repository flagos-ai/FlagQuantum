"""Remove a diagonal gate whose only consumers are measurements.

A gate whose matrix is diagonal in the computational basis multiplies each basis
state by a phase and moves amplitude nowhere: it is a monomial in ``Z``. Every
computational-basis measurement downstream of it therefore returns the outcome it
would have returned without it, so the gate can be dropped as soon as nothing but
such a measurement reads its wires:

    z(0) measure(0)              # `z` is unobservable here
    z(0) h(0) measure(0)         # `h` is not, and `z` is not adjacent to one
    cz(0, 1) measure(0) measure(1)   # diagonal, both wires read out

The rule is one sentence long: **remove a diagonal gate when the next instruction
on every one of its wires is an unconditional measurement.** ``measure`` and
``reset`` are not declared opcodes, so a ``measure`` is one of the two
instructions the optimizer can meet that the operator schema cannot describe; the
predicate below identifies it by name and refuses to reason about a conditional
one, because a condition means this pass is not looking at the whole program.

**Where diagonality comes from.** Not from a table this module writes. A
single-qubit ``U3(theta, phi, lam)`` is
``[[cos(t/2), -exp(1j*lam) sin(t/2)], [exp(1j*phi) sin(t/2),
exp(1j*(phi+lam)) cos(t/2)]]``, so both off-diagonal entries carry the same
factor ``sin(theta / 2)`` and the matrix is diagonal exactly when that factor
vanishes -- whatever ``phi`` and ``lam`` are. The polar angle alone decides it,
and `one_qubit_synthesis` is already the module that tabulates that angle for
every declared single-qubit unitary, so `is_diagonal_one_qubit` asks its tables
rather than restating them. That makes the single-qubit half of this pass exact
*per instruction* instead of per opcode, which is strictly more reach than a name
list would have: ``u3(0, phi, lam)``, ``rx(0)`` and ``ry(0)`` are diagonal and
are removed, and ``remove_identity_gates`` cannot take them because none of the
three is the identity on the nose.

Two-wire diagonality has no such declaration to read. `one_qubit_synthesis`
covers one wire, `two_qubit_synthesis` is a Weyl-chamber synthesis whose
coordinates do not distinguish a diagonal operator from a locally equivalent
one, and `basis_translation`'s equivalence table is keyed on reaching a target
basis rather than on a matrix shape. The four declared two-wire opcodes whose
matrix is diagonal for *every* angle are therefore named here, each with its
closed form on the index convention ``2 * bit(wires[0]) + bit(wires[1])``:

    cz           diag(1, 1, 1, -1)
    cphase(t)    diag(1, 1, 1, exp(1j*t))
    crz(t)       diag(1, 1, exp(-1j*t/2), exp(1j*t/2))
    rzz(t)       diag(exp(-1j*t/2), exp(1j*t/2), exp(1j*t/2), exp(-1j*t/2))

The remaining seven two-wire opcodes -- ``cx``, ``cy``, ``crx``, ``cry``,
``swap``, ``rxx`` and ``ryy`` -- carry a nonzero off-diagonal entry at every
angle and are diagonal for none. Because the classification is per *opcode* here
rather than per value, this is the one place in the pass where a list is the
honest answer; `tests/unit/test_compilation_diagonal_before_measure.py` measures
the whole two-wire set out of the runtime gate matrices and asserts that this
list equals it, so the day Core declares a fifth diagonal two-wire opcode the
test fails rather than the pass silently missing it. The single-qubit half is
measured the same way, against all eighteen declared single-qubit unitaries at
several angles.

Four consequences are worth stating, because each is a case this pass deliberately
does not take.

**Only an unconditional measurement licenses a removal.** A ``measure`` carrying
``conditions`` or ``condition_clauses``, and a gate carrying either or the
dynamic flag, are left alone. The single-qubit rule would still hold for a
conditional diagonal gate -- its phase is unobservable whichever branch runs --
but a condition is exactly the event this pass cannot see the whole program
across, and a false removal is a program change while a declined one is only a
missed optimization.

**A wire whose next instruction is not a measurement blocks the removal.** That
covers the three shapes worth naming: a non-diagonal gate between the candidate
and the measurement, no measurement at all on one of a two-wire candidate's
wires, and the end of the program. It is the same requirement Qiskit 1.2.4's
`RemoveDiagonalGatesBeforeMeasure` implements, whose two-wire branch demands that
*every* quantum successor of the gate be a ``Measure``.

**A removed gate does not block the gate behind it.** A run of diagonal gates
before one measurement is therefore removed in full rather than one per call.
That is not a stronger rule than the one above; it is the same rule read on the
program that remains, and it is what makes ``s(0) t(0) measure(0)`` collapse
completely instead of in two rounds.

**A candidate carrying a caller-supplied matrix is left alone**, since the
tabulated angles describe the opcode and not an arbitrary matrix a caller
attached to its name. `is_diagonal_one_qubit` states that refusal; nothing here
reaches into a matrix to decide.
"""

from __future__ import annotations

from dataclasses import replace

from ..core.ir import CircuitIR, Instruction
from .one_qubit_synthesis import is_diagonal_one_qubit

#: `measure` is not a declared opcode, so there is no alias table to consult and
#: this literal is the opcode itself rather than a spelling of it.
_MEASURE = "measure"

#: The metadata keys `runtime.dynamic` uses to record a classical condition, read
#: the way that package's own classifiers read them.
_CONDITION_KEYS = ("conditions", "condition_clauses")

#: The dynamic flag the IR requires of an opcode the operator schema does not
#: declare, and which `runtime.dynamic` puts on every instruction it rewrites.
_DYNAMIC_FLAG = "is_dynamic"

# The declared two-wire unitary opcodes whose matrix is diagonal in the
# computational basis for every angle, with the closed forms stated in the module
# docstring. A unit test measures this set out of the runtime gate matrices over
# the whole declared two-wire unitary set, so a new Core opcode fails that test
# instead of being silently declined here.
_DIAGONAL_TWO_WIRE = frozenset({"cz", "cphase", "crz", "rzz"})


def _is_conditional(instruction: Instruction) -> bool:
    """Whether ``instruction`` carries a classical condition of either spelling."""

    return any(key in instruction.metadata for key in _CONDITION_KEYS)


def _is_unconditional_measure(instruction: Instruction | None) -> bool:
    """Whether ``instruction`` is a measurement this pass may reason across."""

    return (
        instruction is not None
        and instruction.name == _MEASURE
        and not _is_conditional(instruction)
    )


def _is_diagonal(instruction: Instruction) -> bool:
    """Whether ``instruction`` is a diagonal gate this pass is willing to drop.

    A caller-supplied matrix is refused for both arities: the tables consulted
    below describe an opcode's operator, not an operator a caller attached to the
    opcode's name.
    """

    if instruction.matrix is not None or _is_conditional(instruction):
        return False
    if instruction.metadata.get(_DYNAMIC_FLAG):
        return False
    if len(instruction.wires) == 1:
        return is_diagonal_one_qubit(instruction)
    return len(instruction.wires) == 2 and instruction.name in _DIAGONAL_TWO_WIRE


def remove_diagonal_gates_before_measure(ir: CircuitIR) -> CircuitIR:
    """Remove every diagonal gate read out by measurements alone.

    Returns ``ir`` itself when nothing is removable, so a caller can tell
    "nothing to do" from "something changed" without diffing and the pass stays
    cheap on a program whose shape it has nothing to say about.
    """

    instructions = list(ir)
    successor: dict[int, Instruction] = {}
    removed: set[int] = set()
    for index in range(len(instructions) - 1, -1, -1):
        instruction = instructions[index]
        if _is_diagonal(instruction) and all(
            _is_unconditional_measure(successor.get(wire))
            for wire in instruction.wires
        ):
            # A removed gate keeps its slot in `successor` unclaimed, so the gate
            # behind it sees the measurement too and the whole run collapses in
            # one call.
            removed.add(index)
            continue
        for wire in instruction.wires:
            successor[wire] = instruction
    if not removed:
        return ir
    return replace(
        ir,
        instructions=tuple(
            instruction
            for index, instruction in enumerate(instructions)
            if index not in removed
        ),
    )


__all__ = ["remove_diagonal_gates_before_measure"]
