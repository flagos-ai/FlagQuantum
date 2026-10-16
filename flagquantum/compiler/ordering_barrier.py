"""The one reading of a barrier that the compiler passes share.

``barrier`` is not a Core opcode and it is not an operation on state: it carries
no unitary and exists only to fix the order of the operations around it. The
optimizer, the commutation rules, and the schedule legalizer already read it that
way, but each of them recognizes it on its own. Routing needs the same reading in
its estimate and in each of its three strategy loops, so it lives here once
instead of being spelled out at every site.

The reading is deliberately two conditions rather than one. An instruction that
names the barrier while carrying a matrix is a gate wearing the name, and a gate
has to be routed like any other unitary.
"""

from __future__ import annotations

from ..core.ir import Instruction

BARRIER_OPCODE = "barrier"


def is_ordering_barrier(instruction: Instruction) -> bool:
    """Return whether ``instruction`` orders operations instead of acting on state.

    Such an instruction needs no coupling, cannot be decomposed onto one, and is
    never the reason a SWAP is worth inserting. It is still part of the program:
    every routing pass that skips it has to emit it, remapped onto the physical
    qubits the layout in force puts it on.
    """

    return instruction.name == BARRIER_OPCODE and instruction.matrix is None


__all__ = ("BARRIER_OPCODE", "is_ordering_barrier")
