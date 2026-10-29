"""Pure reversible arithmetic instructions shared by compiler and algorithms.

Registers are MSB first. The caller supplies disjoint physical wires and zero
carry/carry-out. The Cuccaro sweep restores carry, preserves a, and leaves the
low sum in b and its high bit in carry_out. No execution objects enter here.
"""

from ..core.ir import Instruction


def ripple_add(
    a: tuple[int, ...], b: tuple[int, ...], carry: int, carry_out: int
) -> tuple[Instruction, ...]:
    """Emit the existing exact 8*n+1 gate adder on explicit disjoint wire maps."""
    if not a or len(a) != len(b):
        raise ValueError("ripple_add requires nonempty equal-width registers")
    wires = (*a, *b, carry, carry_out)
    if any(isinstance(w, bool) or not isinstance(w, int) for w in wires):
        raise TypeError("ripple_add wires must be Python integers")
    if min(wires) < 0 or len(set(wires)) != len(wires):
        raise ValueError("ripple_add wires must be nonnegative and disjoint")
    instructions: list[Instruction] = []
    for left, right in zip(reversed(a), reversed(b), strict=True):
        instructions.extend(
            (
                Instruction("cx", (carry, left)),
                Instruction("cx", (carry, right)),
                Instruction("ccx", (left, right, carry)),
            )
        )
    instructions.append(Instruction("cx", (carry, carry_out)))
    for left, right in zip(a, b, strict=True):
        instructions.extend(
            (
                Instruction("ccx", (left, right, carry)),
                Instruction("cx", (carry, right)),
                Instruction("cx", (carry, left)),
                Instruction("cx", (left, right)),
                Instruction("cx", (carry, right)),
            )
        )
    return tuple(instructions)
