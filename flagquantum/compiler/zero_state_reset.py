"""Remove a reset on a wire no surviving instruction has touched.

`DynamicCircuit.reset` lowers to a dynamic ``reset`` instruction, and a reset on a
wire that is still in ``|0>`` is the identity: it computes its own outcome -- which
can only be zero -- and leaves the wire where it already was. The optimization
pipeline cancelled the identities it could see in the opcode tables (``i``, a zero
angle, a self-inverse pair, a pair of rotations that sum to zero) and could not see
this one, because what makes it the identity is not in a table at all:

    reset(0)                 # kept, although it is the identity
    reset(0) h(0) reset(0)   # the first is the identity, the last is not

The rule is one sentence long: a ``reset`` is removed when no instruction that
survives before it in program order touches its wire.

That is a rule of proof rather than a rule of record. It reads no matrix, no
parameter, no angle, and no operator schema. It needs exactly one fact -- that a
``CircuitIR`` register starts in ``|0...0>`` -- and that fact is checkable as a
property of the IR rather than assumed: ``CircuitIR`` has no field that could carry
another initial state, and `tests/unit/test_compilation_zero_state_reset.py`
asserts the field list that makes that true, so the day such a field is added the
test fails rather than the rule quietly becoming wrong.

Three consequences are worth stating, because each is a case this pass deliberately
does not take.

**A removed reset does not count as touching its wire.** A leading run of resets is
therefore removed in full rather than one per call. That is not a stronger rule than
the one above; it is the same rule read on the program that remains, and it is what
makes `reset(1) reset(1)` collapse completely.

**Anything else that touched the wire blocks the removal.** That includes a
``measure``, an instruction this layer cannot identify, and a gate that provably
leaves ``|0>`` fixed, such as ``z``. Only the first two are correctness: after a
measurement the wire holds the outcome, and an unidentified instruction is not known
to preserve anything. The third is deferred reach -- taking it would mean reading
gate matrices, which is `one_qubit_synthesis` and `native_gate_legalization`'s job
and not this pass's. `benchmarks/compiler_zero_state_reset.py` measures that row
instead of asserting it away.

**A ``reset`` carrying parameters, a caller-supplied matrix, or more than one wire is
left alone**, since the rule above is a statement about one wire's own state.

There is no declaration to read here. A pass whose rule is an opcode table's own
entry holds no gate name at all; this pass has no such entry, so naming ``reset``
once *is* the rule, and a test asserts that the module holds exactly that one string
literal and imports neither the operator schema nor a matrix source.
"""

from __future__ import annotations

from dataclasses import replace

from ..core.ir import CircuitIR, Instruction

# `reset` is not a declared opcode, so there is no alias table to consult and this
# literal is the opcode itself rather than a spelling of it.
_ZERO_STATE_RESET = "reset"


def _is_zero_state_reset(instruction: Instruction) -> bool:
    """Whether ``instruction`` is a reset this pass is willing to reason about."""

    return (
        instruction.name == _ZERO_STATE_RESET
        and instruction.matrix is None
        and not instruction.params
        and len(instruction.wires) == 1
    )


def remove_zero_state_resets(ir: CircuitIR) -> CircuitIR:
    """Remove every reset whose wire no surviving instruction has touched.

    Returns ``ir`` itself when nothing is removable, so a caller can tell "nothing to
    do" from "something changed" without diffing and the pass stays cheap on a
    program whose shape it has nothing to say about.
    """

    touched: set[int] = set()
    surviving: list[Instruction] = []
    removed = 0
    for instruction in ir:
        if _is_zero_state_reset(instruction) and not touched.intersection(
            instruction.wires
        ):
            removed += 1
            continue
        surviving.append(instruction)
        touched.update(instruction.wires)
    if not removed:
        return ir
    return replace(ir, instructions=tuple(surviving))


__all__ = ["remove_zero_state_resets"]
